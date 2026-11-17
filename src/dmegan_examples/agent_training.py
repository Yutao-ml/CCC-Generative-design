from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(
        "PyTorch is required. Install torch in the environment used to run this script."
    ) from exc

from .agent_feedback import EventTriggeredPhysicsFeedback


MODEL_NAME = Path(__file__).stem


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sorted_raw_paths(data_dir: Path) -> List[Path]:
    paths = list(data_dir.glob("RAW_*.png"))
    if not paths:
        raise FileNotFoundError(f"No RAW_*.png files found in {data_dir}")

    def raw_index(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"Unexpected RAW image name: {path.name}") from exc

    return sorted(paths, key=raw_index)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_signature(paths: Sequence[Path]) -> Dict[str, object]:
    entries = [{"name": p.name, "sha256": file_sha256(p)} for p in paths]
    joined = "".join(item["sha256"] for item in entries).encode("utf-8")
    return {
        "count": len(entries),
        "combined_sha256": hashlib.sha256(joined).hexdigest(),
        "files": entries,
    }


def pil_resample_filter() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def load_carbon_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    carbon = np.clip(arr[..., 0], 0.0, 255.0).astype(np.uint8)
    gray = Image.fromarray(carbon, mode="L")
    if gray.size != (image_size, image_size):
        gray = gray.resize((image_size, image_size), pil_resample_filter())
    out = np.asarray(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(out).unsqueeze(0)


class FairEDSAugment:
    """Shared EDS data augmentation used consistently across model runs."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[1])
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k=k, dims=[1, 2])

        mean = x.mean(dim=(1, 2), keepdim=True)
        contrast = torch.empty((), dtype=x.dtype, device=x.device).uniform_(0.85, 1.15)
        brightness = torch.empty((), dtype=x.dtype, device=x.device).uniform_(-0.06, 0.06)
        gamma = torch.empty((), dtype=x.dtype, device=x.device).uniform_(0.85, 1.15)
        x = (x - mean) * contrast + mean + brightness
        x = x.clamp(0.0, 1.0).pow(gamma)

        if torch.rand(()) < 0.35:
            x = (x + torch.randn_like(x) * 0.01).clamp(0.0, 1.0)
        return x.clamp(0.0, 1.0)


class CarbonEDSDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_size: int = 512,
        augment: bool = False,
        augment_repeats: int = 16,
    ) -> None:
        self.paths = sorted_raw_paths(data_dir)
        self.images = [load_carbon_tensor(path, image_size) for path in self.paths]
        self.augment = augment
        self.augment_repeats = max(1, augment_repeats)
        self.transform = FairEDSAugment()

    def __len__(self) -> int:
        if self.augment:
            return len(self.images) * self.augment_repeats
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.images[index % len(self.images)].clone()
        if self.augment:
            image = self.transform(image)
        return image


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0**2))


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": output_dir,
        "generated": output_dir / "generated",
        "logs": output_dir / "logs",
        "checkpoints": output_dir / "checkpoints",
        "previews": output_dir / "previews",
        "intermediate_generated": output_dir / "intermediate_generated",
        "candidate_previews": output_dir / "candidate_previews",
        "candidate_metrics": output_dir / "candidate_metrics",
        "best_candidates": output_dir / "best_candidates",
        "analysis_plots": output_dir / "analysis_plots",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_csv_row(path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=json_default)


def save_generated_images(samples: torch.Tensor, output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = samples.detach().cpu().clamp(0.0, 1.0)
    saved = []
    for idx, sample in enumerate(samples, start=1):
        arr = (sample.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        image = Image.fromarray(arr, mode="L")
        path = output_dir / f"generated_{idx:03d}.png"
        image.save(path)
        saved.append(str(path))
    return saved


def save_preview_grid(samples: torch.Tensor, path: Path, grid_cols: int = 4) -> None:
    samples = samples.detach().cpu().clamp(0.0, 1.0)
    n, _, h, w = samples.shape
    cols = min(grid_cols, n)
    rows = int(math.ceil(n / cols))
    grid = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx, sample in enumerate(samples):
        r = idx // cols
        c = idx % cols
        arr = (sample.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="L").save(path)


def tensors_to_arrays(tensors: Sequence[torch.Tensor]) -> np.ndarray:
    return np.stack([tensor.squeeze(0).cpu().numpy().astype(np.float32) for tensor in tensors])


def otsu_threshold(img: np.ndarray) -> float:
    hist, _ = np.histogram(img, bins=256, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.5
    prob = hist.astype(np.float64) / total
    omega = np.cumsum(prob)
    values = np.linspace(0.0, 1.0, 256)
    mu = np.cumsum(prob * values)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    sigma_b = np.zeros_like(denom)
    valid = denom > 1e-12
    sigma_b[valid] = ((mu_t * omega[valid] - mu[valid]) ** 2) / denom[valid]
    return float(values[int(np.argmax(sigma_b))])


def connected_component_stats(mask: np.ndarray) -> Tuple[int, float, float]:
    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    areas: List[int] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            areas.append(area)
    if not areas:
        return 0, 0.0, 0.0
    return count, float(np.mean(areas)), float(np.max(areas))


def microstructure_stats_single(img: np.ndarray) -> Dict[str, float]:
    img = np.asarray(img, dtype=np.float64)
    threshold = max(otsu_threshold(img), 0.03)
    mask = img > threshold
    area_fraction = float(mask.mean())
    interface_density = 0.0
    if mask.size:
        interface_density = 0.5 * (
            np.mean(mask[:, 1:] != mask[:, :-1]) + np.mean(mask[1:, :] != mask[:-1, :])
        )
    count, mean_area, max_area = connected_component_stats(mask)

    corr_length = 0.0
    mask_centered = mask.astype(np.float64) - area_fraction
    mask_var = float(mask_centered.var())
    if mask_var > 1e-12:
        max_lag = min(128, img.shape[1] - 1)
        for lag in range(1, max_lag + 1):
            corr = float(np.mean(mask_centered[:, :-lag] * mask_centered[:, lag:]) / mask_var)
            if corr < math.exp(-1.0):
                corr_length = float(lag)
                break
        if corr_length == 0.0:
            corr_length = float(max_lag)

    hist, _ = np.histogram(img, bins=64, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64) / max(1, hist.sum())
    entropy = float(-(prob[prob > 0] * np.log(prob[prob > 0])).sum())

    gy, gx = np.gradient(img)
    grad = np.sqrt(gx * gx + gy * gy)
    centered = img - float(img.mean())
    var = float(centered.var())
    autocorr_x = 0.0
    autocorr_y = 0.0
    if var > 1e-12:
        autocorr_x = float(np.mean(centered[:, :-1] * centered[:, 1:]) / var)
        autocorr_y = float(np.mean(centered[:-1, :] * centered[1:, :]) / var)

    return {
        "mean_intensity": float(img.mean()),
        "std_intensity": float(img.std()),
        "p95_intensity": float(np.quantile(img, 0.95)),
        "area_fraction": area_fraction,
        "interface_density": float(interface_density),
        "component_count": float(count),
        "mean_component_area": float(mean_area),
        "correlation_length": corr_length,
        "hist_entropy": entropy,
        "gradient_mean": float(grad.mean()),
        "edge_density": float((grad > 0.05).mean()),
        "autocorr_x": autocorr_x,
        "autocorr_y": autocorr_y,
        "component_area_max": float(max_area),
    }


def microstructure_stats_batch(images: np.ndarray) -> Dict[str, float]:
    stats = [microstructure_stats_single(img.squeeze()) for img in images]
    keys = stats[0].keys()
    return {key: float(np.mean([s[key] for s in stats])) for key in keys}


def feature_matrix(images: torch.Tensor) -> np.ndarray:
    x = images.detach().cpu().float().clamp(0.0, 1.0)
    small = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
    small_np = small.flatten(1).numpy()
    arr = x.numpy()
    feats = []
    for i in range(arr.shape[0]):
        img = arr[i, 0]
        hist, _ = np.histogram(img, bins=32, range=(0.0, 1.0), density=True)
        quantiles = np.quantile(img, [0.05, 0.25, 0.5, 0.75, 0.95])
        micro = microstructure_stats_single(img)
        base = np.array(
            [
                img.mean(),
                img.std(),
                img.min(),
                img.max(),
                *quantiles,
                micro["area_fraction"],
                micro["interface_density"],
                micro["component_count"] / img.size,
                micro["mean_component_area"] / img.size,
                micro["correlation_length"] / max(img.shape),
                micro["hist_entropy"],
                micro["gradient_mean"],
                micro["edge_density"],
                micro["autocorr_x"],
                micro["autocorr_y"],
                micro["component_area_max"] / img.size,
            ],
            dtype=np.float64,
        )
        feats.append(np.concatenate([base, hist.astype(np.float64), small_np[i].astype(np.float64)]))
    return np.stack(feats, axis=0)


def sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) * 0.5)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
    cov_y = np.cov(y, rowvar=False) + np.eye(y.shape[1]) * 1e-6
    diff = mu_x - mu_y
    sqrt_x = sqrtm_psd(cov_x)
    trace_covmean = float(np.trace(sqrtm_psd(sqrt_x @ cov_y @ sqrt_x)))
    fid = float(diff.dot(diff) + np.trace(cov_x) + np.trace(cov_y) - 2.0 * trace_covmean)
    return max(fid, 0.0)


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    z = np.concatenate([x, y], axis=0)
    dists = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=-1)
    positive = dists[dists > 0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    gamma = 1.0 / max(2.0 * sigma2, 1e-12)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def ssim_pair(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def psnr_pair(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def nearest_ssim(real: torch.Tensor, generated: torch.Tensor) -> Dict[str, float]:
    real_np = real.detach().cpu().numpy()[:, 0]
    gen_np = generated.detach().cpu().numpy()[:, 0]
    scores = []
    psnr_scores = []
    for g in gen_np:
        pairs = [(ssim_pair(r, g), psnr_pair(r, g)) for r in real_np]
        best_ssim, best_psnr = max(pairs, key=lambda item: item[0])
        scores.append(best_ssim)
        psnr_scores.append(best_psnr)
    return {
        "ssim_nearest": float(np.mean(scores)),
        "ssim_nearest_std": float(np.std(scores)),
        "ssim_nearest_min": float(np.min(scores)),
        "psnr_nearest": float(np.mean(psnr_scores)),
        "psnr_nearest_std": float(np.std(psnr_scores)),
        "psnr_nearest_min": float(np.min(psnr_scores)),
    }


def nearest_binary_mask_ssim(real: torch.Tensor, generated: torch.Tensor) -> Dict[str, float]:
    real_np = real.detach().cpu().numpy()[:, 0]
    gen_np = generated.detach().cpu().numpy()[:, 0]
    real_masks = [(r > max(otsu_threshold(r), 0.03)).astype(np.float64) for r in real_np]
    gen_masks = [(g > max(otsu_threshold(g), 0.03)).astype(np.float64) for g in gen_np]
    scores = []
    for g in gen_masks:
        scores.append(max(ssim_pair(r, g) for r in real_masks))
    return {
        "binary_mask_ssim_nearest": float(np.mean(scores)),
        "binary_mask_ssim_nearest_std": float(np.std(scores)),
    }


def nearest_multiscale_ssim(real: torch.Tensor, generated: torch.Tensor) -> Dict[str, float]:
    scores = []
    real_t = real.detach().float().clamp(0.0, 1.0)
    gen_t = generated.detach().float().clamp(0.0, 1.0)
    for size in (512, 256, 128):
        if real_t.shape[-1] != size:
            real_scaled = F.interpolate(real_t, size=(size, size), mode="bilinear", align_corners=False)
            gen_scaled = F.interpolate(gen_t, size=(size, size), mode="bilinear", align_corners=False)
        else:
            real_scaled = real_t
            gen_scaled = gen_t
        real_np = real_scaled.cpu().numpy()[:, 0]
        gen_np = gen_scaled.cpu().numpy()[:, 0]
        scale_scores = []
        for g in gen_np:
            scale_scores.append(max(ssim_pair(r, g) for r in real_np))
        scores.append(float(np.mean(scale_scores)))
    return {"ms_ssim_nearest": float(np.mean(scores))}


def arrays_to_eval_tensor(images: np.ndarray) -> torch.Tensor:
    arr = np.asarray(images, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    if arr.ndim != 4:
        raise ValueError(f"Expected image array with 3 or 4 dims, got {arr.shape}")
    return torch.from_numpy(arr).float().clamp(0.0, 1.0)


def evaluate_generated(real: torch.Tensor, generated: torch.Tensor) -> Dict[str, object]:
    real = real.float().clamp(0.0, 1.0)
    generated = generated.float().clamp(0.0, 1.0)
    real_features = feature_matrix(real)
    gen_features = feature_matrix(generated)
    real_micro = microstructure_stats_batch(real.numpy()[:, 0])
    gen_micro = microstructure_stats_batch(generated.numpy()[:, 0])
    micro_abs_diff = {key: float(abs(gen_micro[key] - real_micro[key])) for key in real_micro.keys()}
    out: Dict[str, object] = {
        "fid_feature_no_pretrained": frechet_distance(real_features, gen_features),
        "mmd_rbf": mmd_rbf(real_features, gen_features),
        "microstructure_real_mean": real_micro,
        "microstructure_generated_mean": gen_micro,
        "microstructure_abs_diff": micro_abs_diff,
        "metric_note": (
            "FID uses fixed grayscale, histogram, quantile, and microstructure features; "
            "no external pretrained weights or extra training data are used."
        ),
    }
    out.update(nearest_ssim(real, generated))
    out.update(nearest_binary_mask_ssim(real, generated))
    out.update(nearest_multiscale_ssim(real, generated))
    return out


def compute_metrics(real: np.ndarray, generated: np.ndarray) -> Dict[str, object]:
    return evaluate_generated(arrays_to_eval_tensor(real), arrays_to_eval_tensor(generated))


class TextureResidualBlock(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.25) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        groups = max(1, min(8, channels // 8))
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.residual_scale * self.net(x))


class NoiseInjection(nn.Module):
    def __init__(self, channels: int, init_scale: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class UpsampleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = True,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
        norm: str = "instance",
        residual_scale: float = 0.25,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if upsample_mode == "nearest":
            upsample: nn.Module = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        layers: List[nn.Module] = [
            upsample,
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
        ]
        if norm == "group":
            layers.append(nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if noise_scale > 0.0:
            layers.append(NoiseInjection(out_channels, init_scale=noise_scale))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.main = nn.Sequential(*layers)
        self.residual = TextureResidualBlock(out_channels, residual_scale=residual_scale) if residual else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.main(x))


class ParticleRefinementHead(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, noise_scale: float = 0.035) -> None:
        super().__init__()
        groups = max(1, min(8, hidden_channels // 4))
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            NoiseInjection(hidden_channels, init_scale=noise_scale),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 32,
        image_size: int = 512,
        feature_dropout: float = 0.04,
    ) -> None:
        super().__init__()
        if image_size != 512:
            raise ValueError("This WGAN-GP generator is configured for 512x512 output.")
        self.latent_dim = latent_dim
        self.start_size = 16
        c = base_channels
        half_channels = max(c // 2, 8)
        quarter_channels = max(c // 4, 8)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, c * 8 * self.start_size * self.start_size, bias=False),
            nn.BatchNorm1d(c * 8 * self.start_size * self.start_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            UpsampleConvBlock(c * 8, c * 4, residual=True, norm="group", residual_scale=0.08, noise_scale=0.0),
            UpsampleConvBlock(c * 4, c * 2, residual=True, norm="group", residual_scale=0.10, noise_scale=0.0),
            UpsampleConvBlock(c * 2, c, residual=True, norm="group", residual_scale=0.12, noise_scale=0.0),
            UpsampleConvBlock(c, half_channels, residual=True, norm="group", residual_scale=0.14, noise_scale=0.006),
            UpsampleConvBlock(half_channels, quarter_channels, residual=True, norm="group", residual_scale=0.15, noise_scale=0.010),
        )
        self.coarse_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(quarter_channels, 1, kernel_size=3, stride=1, padding=0, bias=False),
        )
        self.detail_head = ParticleRefinementHead(
            quarter_channels,
            hidden_channels=max(quarter_channels, 12),
            noise_scale=0.012,
        )
        self.detail_scale = nn.Parameter(torch.tensor(0.12))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(z.size(0), -1, self.start_size, self.start_size)
        features = self.net(x)
        coarse = self.coarse_head(features)
        detail = self.detail_head(features)
        return torch.tanh(coarse + self.detail_scale.clamp(0.0, 0.45) * detail)


def spectral_conv2d(*args, spectral: bool = True, **kwargs) -> nn.Module:
    conv = nn.Conv2d(*args, **kwargs)
    return nn.utils.spectral_norm(conv) if spectral else conv


class CriticBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, spectral: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                spectral=spectral,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def critic_edge_map(x: torch.Tensor) -> torch.Tensor:
    dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
    return (0.5 * (dx.abs() + dy.abs())).clamp(0.0, 1.0)


def critic_laplacian_map(x: torch.Tensor) -> torch.Tensor:
    kernel = x.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3)
    padded = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel).abs().clamp(0.0, 1.0)


class Discriminator(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        image_size: int = 512,
        score_scale: float = 8.0,
        patch_weight: float = 0.35,
    ) -> None:
        super().__init__()
        if image_size != 512:
            raise ValueError("This WGAN-GP critic is configured for 512x512 input.")
        c = base_channels
        self.score_scale = max(float(score_scale), 1.0)
        self.global_features = nn.Sequential(
            CriticBlock(1, c),
            CriticBlock(c, c * 2),
            CriticBlock(c * 2, c * 4),
            CriticBlock(c * 4, c * 8),
            CriticBlock(c * 8, c * 16),
            CriticBlock(c * 16, c * 16),
            CriticBlock(c * 16, c * 16),
        )
        self.global_head = nn.utils.spectral_norm(nn.Conv2d(c * 16, 1, kernel_size=4, stride=1, padding=0, bias=True))
        self.patch_features = nn.Sequential(
            CriticBlock(3, c, stride=2),
            CriticBlock(c, c * 2, stride=2),
            CriticBlock(c * 2, c * 4, stride=2),
            CriticBlock(c * 4, c * 4, stride=1),
        )
        self.patch_head = nn.utils.spectral_norm(nn.Conv2d(c * 4, 1, kernel_size=3, stride=1, padding=1, bias=True))
        self.patch_weight = float(patch_weight)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        global_features = self.global_features(x)
        global_score = self.global_head(global_features).flatten(1).mean(dim=1)
        patch_input = torch.cat([x, critic_edge_map(x), critic_laplacian_map(x)], dim=1)
        patch_features = self.patch_features(patch_input)
        patch_score = self.patch_head(patch_features).flatten(1).mean(dim=1)
        score = (global_score + self.patch_weight * patch_score) / self.score_scale
        if return_features:
            return score, (global_features, patch_features)
        return score


def init_dcgan_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


def sample_latent(batch_size: int, latent_dim: int, device: torch.device, truncation: float = 1.0) -> torch.Tensor:
    z = torch.randn(batch_size, latent_dim, device=device)
    return z * truncation


def regularize_latent_for_training(z: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.latent_jitter_std > 0.0:
        z = z + torch.randn_like(z) * float(args.latent_jitter_std)
    if args.latent_dropout > 0.0:
        keep_prob = 1.0 - float(args.latent_dropout)
        keep_prob = max(0.05, min(1.0, keep_prob))
        mask = torch.empty_like(z).bernoulli_(keep_prob) / keep_prob
        z = z * mask
    return z


def image_to_gan_range(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def gan_to_image_range(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def set_requires_grad(model: nn.Module, enabled: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(enabled)


def one_sided_real_labels(batch_size: int, device: torch.device, low: float, high: float) -> torch.Tensor:
    low = float(max(0.0, min(1.0, low)))
    high = float(max(low, min(1.0, high)))
    if math.isclose(low, high):
        return torch.full((batch_size,), high, device=device)
    return torch.empty(batch_size, device=device).uniform_(low, high)


def instance_noise_std_for_epoch(epoch: int, initial_std: float, decay_epochs: int) -> float:
    if initial_std <= 0.0:
        return 0.0
    if decay_epochs <= 0:
        return float(initial_std)
    progress = min(max((epoch - 1) / float(decay_epochs), 0.0), 1.0)
    return float(initial_std * (1.0 - progress))


def add_instance_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0.0:
        return x
    return (x + torch.randn_like(x) * std).clamp(-1.0, 1.0)


def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)
    interpolates = (alpha * real_samples + (1.0 - alpha) * fake_samples).requires_grad_(True)
    critic_scores = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.flatten(1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def topology_constraint_weight(epoch: int, start_epoch: int, warmup_epochs: int) -> float:
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    progress = min(1.0, max(0.0, (epoch - start_epoch + 1) / float(warmup_epochs)))
    return float(progress * progress * (3.0 - 2.0 * progress))


def differentiable_topology_metrics(images_01: torch.Tensor, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    x = images_01.float().clamp(0.0, 1.0)
    threshold = x.new_tensor(float(args.topology_threshold))
    temperature = max(float(args.topology_temperature), 1e-4)
    soft_mask = torch.sigmoid((x - threshold) / temperature)

    density = soft_mask.mean()

    block_size = max(int(args.topology_uniformity_block), 2)
    local_blocks = F.avg_pool2d(soft_mask, kernel_size=block_size, stride=block_size, ceil_mode=True)
    block_mean = local_blocks.mean()
    block_std = local_blocks.std(unbiased=False)
    uniformity = 1.0 / (1.0 + block_std / block_mean.clamp_min(1e-6))

    kernel = max(int(args.topology_aggregation_kernel), 3)
    if kernel % 2 == 0:
        kernel += 1
    local_density = F.avg_pool2d(soft_mask, kernel_size=kernel, stride=1, padding=kernel // 2)
    aggregation = (soft_mask * local_density).mean() / density.clamp_min(1e-6)

    dx = soft_mask[:, :, :, 1:] - soft_mask[:, :, :, :-1]
    dy = soft_mask[:, :, 1:, :] - soft_mask[:, :, :-1, :]
    interface_density = 0.5 * (dx.abs().mean() + dy.abs().mean())

    centered = soft_mask - soft_mask.mean(dim=(2, 3), keepdim=True)
    var = (centered * centered).mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    autocorr_x = (centered[:, :, :, :-1] * centered[:, :, :, 1:]).mean() / var.mean()
    autocorr_y = (centered[:, :, :-1, :] * centered[:, :, 1:, :]).mean() / var.mean()

    low_kernel = max(int(args.low_frequency_kernel), 4)
    if low_kernel > min(x.shape[-2:]):
        low_kernel = min(x.shape[-2:])
    low_frequency_map = F.avg_pool2d(x, kernel_size=low_kernel, stride=low_kernel)
    tv_loss = torch_gradient_mean(x)

    return {
        "density": density,
        "uniformity": uniformity,
        "aggregation": aggregation,
        "interface_density": interface_density,
        "autocorr_x": autocorr_x,
        "autocorr_y": autocorr_y,
        "low_frequency_map": low_frequency_map.mean(dim=0, keepdim=True),
        "tv_loss": tv_loss,
    }


def relative_smooth_l1_loss(value: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    relative_error = (value - target).abs() / target.detach().abs().clamp_min(1e-6)
    if threshold > 0.0:
        relative_error = F.relu(relative_error - float(threshold))
    return F.smooth_l1_loss(relative_error, torch.zeros_like(relative_error), beta=0.05)


def differentiable_topology_proxy_loss(
    fake_01: torch.Tensor,
    target_metrics: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    metrics = differentiable_topology_metrics(fake_01, args)
    thresholds = {
        "density": float(args.topology_density_threshold),
        "uniformity": float(args.topology_uniformity_threshold),
        "aggregation": float(args.topology_aggregation_threshold),
    }
    weights = {
        "density": float(args.topology_density_weight),
        "uniformity": float(args.topology_uniformity_weight),
        "aggregation": float(args.topology_aggregation_weight),
    }
    losses = []
    relative_errors: Dict[str, torch.Tensor] = {}
    for key in ("density", "uniformity", "aggregation"):
        target = target_metrics[key].to(device=fake_01.device, dtype=fake_01.dtype).detach()
        relative_errors[key] = (metrics[key] - target).abs() / target.abs().clamp_min(1e-6)
        losses.append(weights[key] * relative_smooth_l1_loss(metrics[key], target, thresholds[key]))
    interface_target = target_metrics["interface_density"].to(device=fake_01.device, dtype=fake_01.dtype).detach()
    autocorr_x_target = target_metrics["autocorr_x"].to(device=fake_01.device, dtype=fake_01.dtype).detach()
    autocorr_y_target = target_metrics["autocorr_y"].to(device=fake_01.device, dtype=fake_01.dtype).detach()
    low_frequency_target = target_metrics["low_frequency_map"].to(device=fake_01.device, dtype=fake_01.dtype).detach()
    losses.append(
        float(args.interface_loss_weight)
        * relative_smooth_l1_loss(metrics["interface_density"], interface_target, float(args.interface_density_threshold))
    )
    losses.append(
        float(args.autocorr_loss_weight)
        * (
            relative_smooth_l1_loss(metrics["autocorr_x"], autocorr_x_target, float(args.autocorr_threshold))
            + relative_smooth_l1_loss(metrics["autocorr_y"], autocorr_y_target, float(args.autocorr_threshold))
        )
    )
    losses.append(float(args.low_frequency_loss_weight) * F.l1_loss(metrics["low_frequency_map"], low_frequency_target))
    losses.append(float(args.tv_loss_weight) * metrics["tv_loss"])
    relative_errors["interface_density"] = (
        (metrics["interface_density"] - interface_target).abs() / interface_target.abs().clamp_min(1e-6)
    )
    relative_errors["autocorr_x"] = (metrics["autocorr_x"] - autocorr_x_target).abs() / autocorr_x_target.abs().clamp_min(1e-6)
    relative_errors["autocorr_y"] = (metrics["autocorr_y"] - autocorr_y_target).abs() / autocorr_y_target.abs().clamp_min(1e-6)
    return sum(losses), metrics, relative_errors


def training_balance_score(stats: Dict[str, float], args: argparse.Namespace) -> float:
    w_dist = float(stats.get("w_dist_raw", 0.0))
    real_raw = float(stats.get("critic_real_score_raw", 0.0))
    fake_raw = float(stats.get("critic_fake_score_raw", 0.0))
    g_loss = min(abs(float(stats.get("g_adv_loss_raw", stats.get("g_loss", 0.0)))), 20.0)
    return (
        max(0.0, args.min_healthy_w_dist - w_dist)
        + max(0.0, w_dist - args.max_healthy_w_dist)
        + max(0.0, fake_raw - real_raw)
        + args.balance_g_loss_weight * g_loss
    )


def range_distance(value: float, low: float, high: float) -> float:
    if value < low:
        return float(low - value)
    if value > high:
        return float(value - high)
    return 0.0


def gan_curve_health_penalty(stats: Dict[str, float], args: argparse.Namespace) -> float:
    w_dist = float(stats.get("w_dist_raw", 0.0))
    real_raw = float(stats.get("critic_real_score_raw", 0.0))
    fake_raw = float(stats.get("critic_fake_score_raw", 0.0))
    g_adv = float(stats.get("g_adv_loss_raw", stats.get("g_adv_loss", 0.0)))
    gap = real_raw - fake_raw
    w_penalty = max(0.0, args.min_healthy_w_dist - w_dist) + max(0.0, w_dist - args.max_healthy_w_dist)
    order_penalty = max(0.0, fake_raw - real_raw)
    gap_penalty = max(0.0, args.min_healthy_w_dist - gap) + max(0.0, gap - args.max_healthy_w_dist)
    g_penalty = max(0.0, abs(g_adv) - args.max_abs_g_adv_loss) / max(args.max_abs_g_adv_loss, 1e-6)
    stability_penalty = (
        max(0.0, float(stats.get("w_dist_tail_std", 0.0)) - float(args.max_w_dist_tail_std))
        + max(0.0, float(stats.get("g_adv_tail_std", 0.0)) - float(args.max_g_adv_tail_std))
        + max(0.0, float(stats.get("d_loss_tail_std", 0.0)) - float(args.max_d_loss_tail_std))
        + max(0.0, float(stats.get("score_gap_tail_std", 0.0)) - float(args.max_score_gap_tail_std))
    )
    return float(w_penalty + order_penalty + gap_penalty + g_penalty + stability_penalty)


def training_controls_for_epoch(
    epoch: int,
    previous_stats: Dict[str, float],
    args: argparse.Namespace,
) -> Tuple[int, int, float, str, float]:
    base_g = max(1, int(args.g_updates_per_step))

    if epoch <= max(int(args.d_warmup_epochs), 1):
        return 1, max(1, int(args.warmup_g_updates_per_step)), 0.20, "critic_warmup", 0.0

    if epoch < int(args.topology_constraint_start_epoch):
        return 1, base_g, 0.35, "free_adversarial", 0.0

    if epoch >= int(args.late_stabilize_epoch):
        return 1, max(base_g, int(args.late_g_updates_per_step)), 0.30, "late_physical_focus", 1.0

    if epoch < int(args.topology_constraint_start_epoch) + int(args.topology_constraint_warmup_epochs):
        return 1, base_g, 0.36, "topology_warmup", 0.62
    return 1, base_g, 0.34, "late_stable", 0.82


def tensor_diversity(images: torch.Tensor) -> float:
    x = images.detach().cpu().float().clamp(0.0, 1.0)
    if x.size(0) <= 1:
        return 0.0
    return float(x.std(dim=0, unbiased=False).mean().item())


def tensor_pairwise_l1(images: torch.Tensor) -> float:
    x = images.detach().cpu().float().clamp(0.0, 1.0)
    if x.size(0) <= 1:
        return 0.0
    flat = x.flatten(1)
    distances = torch.pdist(flat, p=1) / float(flat.size(1))
    return float(distances.mean().item())


def tensor_shift_texture(images: torch.Tensor) -> float:
    x = images.detach().cpu().float().clamp(0.0, 1.0)
    if x.size(0) == 0:
        return 0.0
    return float(shift_texture_response(x).mean().item())


def batch_diversity_loss(images_01: torch.Tensor, target_std: float) -> torch.Tensor:
    if images_01.size(0) <= 1 or target_std <= 0.0:
        return images_01.new_zeros(())
    target = images_01.new_tensor(float(target_std)).clamp_min(1e-6)
    batch_std = images_01.float().std(dim=0, unbiased=False).mean()
    return F.relu(target - batch_std) / target


def batch_pairwise_diversity_loss(images_01: torch.Tensor, target_l1: float) -> torch.Tensor:
    if images_01.size(0) <= 1 or target_l1 <= 0.0:
        return images_01.new_zeros(())
    flat = images_01.float().flatten(1)
    distances = torch.pdist(flat, p=1) / float(flat.size(1))
    target = images_01.new_tensor(float(target_l1)).clamp_min(1e-6)
    return (F.relu(target - distances) / target).mean()


def shift_texture_response(images_01: torch.Tensor, shifts: Sequence[int] = (16, 32, 64)) -> torch.Tensor:
    values = []
    x = images_01.float().clamp(0.0, 1.0)
    for shift in shifts:
        if shift <= 0 or shift >= min(x.shape[-2:]):
            continue
        dx = (x - torch.roll(x, shifts=shift, dims=-1)).abs().mean(dim=(1, 2, 3))
        dy = (x - torch.roll(x, shifts=shift, dims=-2)).abs().mean(dim=(1, 2, 3))
        values.append(0.5 * (dx + dy))
    if not values:
        return x.new_zeros((x.size(0),))
    return torch.stack(values, dim=0).mean(dim=0)


def shifted_autocorr_response(images_01: torch.Tensor, shifts: Sequence[int] = (4, 8, 16, 32)) -> torch.Tensor:
    x = images_01.float().clamp(0.0, 1.0)
    centered = x - x.mean(dim=(2, 3), keepdim=True)
    var = centered.square().mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    values = []
    for shift in shifts:
        if shift <= 0 or shift >= min(x.shape[-2:]):
            continue
        corr_x = (centered * torch.roll(centered, shifts=shift, dims=-1)).mean(dim=(1, 2, 3)) / var.mean(dim=(1, 2, 3))
        corr_y = (centered * torch.roll(centered, shifts=shift, dims=-2)).mean(dim=(1, 2, 3)) / var.mean(dim=(1, 2, 3))
        corr_d1 = (centered * torch.roll(centered, shifts=(shift, shift), dims=(-2, -1))).mean(dim=(1, 2, 3)) / var.mean(dim=(1, 2, 3))
        corr_d2 = (centered * torch.roll(centered, shifts=(shift, -shift), dims=(-2, -1))).mean(dim=(1, 2, 3)) / var.mean(dim=(1, 2, 3))
        values.append(torch.stack([corr_x.abs(), corr_y.abs(), corr_d1.abs(), corr_d2.abs()], dim=0).mean(dim=0))
    if not values:
        return x.new_zeros((x.size(0),))
    return torch.stack(values, dim=0).mean(dim=0)


def anti_periodic_texture_loss(fake_01: torch.Tensor, real_01: torch.Tensor) -> torch.Tensor:
    real_response = shift_texture_response(real_01.detach()).mean().detach()
    fake_response = shift_texture_response(fake_01).mean()
    response_loss = F.smooth_l1_loss(fake_response, real_response, beta=0.02)
    real_corr = shifted_autocorr_response(real_01.detach()).mean().detach()
    fake_corr = shifted_autocorr_response(fake_01).mean()
    periodic_excess = F.relu(fake_corr - real_corr * 1.12)
    return response_loss + periodic_excess


def generator_score_stability_loss(fake_score: torch.Tensor, target_abs: float = 0.65) -> torch.Tensor:
    target = fake_score.new_tensor(float(target_abs)).abs().clamp_min(1e-6)
    return F.relu(fake_score.abs() - target).square() / target



def gradient_energy(images_01: torch.Tensor) -> torch.Tensor:
    x = images_01.float().clamp(0.0, 1.0)
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return 0.5 * (dx + dy)


def gradient_energy_match_loss(fake_01: torch.Tensor, real_01: torch.Tensor) -> torch.Tensor:
    real_grad = gradient_energy(real_01.detach()).detach()
    fake_grad = gradient_energy(fake_01)
    return F.smooth_l1_loss(fake_grad, real_grad, beta=0.01)


def critic_feature_matching_loss(discriminator: nn.Module, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    _, fake_features = discriminator(fake, return_features=True)
    with torch.no_grad():
        _, real_features = discriminator(real.detach(), return_features=True)
    losses = []
    for fake_feat, real_feat in zip(fake_features, real_features):
        real_feat = real_feat.detach()
        fake_mean = fake_feat.mean(dim=(0, 2, 3))
        real_mean = real_feat.mean(dim=(0, 2, 3))
        fake_std = fake_feat.std(dim=(0, 2, 3), unbiased=False)
        real_std = real_feat.std(dim=(0, 2, 3), unbiased=False)
        losses.append(F.smooth_l1_loss(fake_mean, real_mean, beta=0.05))
        losses.append(F.smooth_l1_loss(fake_std, real_std, beta=0.05))
    return sum(losses) / max(len(losses), 1)


def nearest_reference_batch(fake_01: torch.Tensor, real_01: torch.Tensor, size: int = 64) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = real_01.detach().float().clamp(0.0, 1.0)
    if real.size(0) == 0:
        return real
    match_size = int(size)
    if match_size > 0 and fake.shape[-1] != match_size:
        fake_small = F.interpolate(fake, size=(match_size, match_size), mode="area")
        real_small = F.interpolate(real, size=(match_size, match_size), mode="area")
    else:
        fake_small = fake
        real_small = real
    with torch.no_grad():
        fake_flat = fake_small.detach().flatten(1)
        real_flat = real_small.detach().flatten(1)
        distances = torch.cdist(fake_flat, real_flat, p=1) / max(fake_flat.size(1), 1)
        nearest = distances.argmin(dim=1)
    return real.index_select(0, nearest)


def differentiable_ssim_loss(fake_01: torch.Tensor, real_01: torch.Tensor, size: int = 128) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = nearest_reference_batch(fake, real_01, size=max(32, min(96, int(size) if size > 0 else 64)))
    full_fake_ref = fake
    full_real_ref = real.detach().float().clamp(0.0, 1.0)
    if size > 0 and fake.shape[-1] != size:
        fake = F.interpolate(fake, size=(size, size), mode="bilinear", align_corners=False)
        real = F.interpolate(real, size=(size, size), mode="bilinear", align_corners=False)
    window = 7
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(fake, window, stride=1, padding=window // 2)
    mu_y = F.avg_pool2d(real, window, stride=1, padding=window // 2)
    sigma_x = F.avg_pool2d(fake * fake, window, stride=1, padding=window // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(real * real, window, stride=1, padding=window // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(fake * real, window, stride=1, padding=window // 2) - mu_x * mu_y
    ssim = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-6)
    ssim_loss = (1.0 - ssim.clamp(-1.0, 1.0)).mean()

    anchor_size = max(32, min(128, int(size) if size > 0 else 96))
    if fake.shape[-1] != anchor_size:
        fake_anchor = F.interpolate(fake, size=(anchor_size, anchor_size), mode="area")
        real_anchor = F.interpolate(real, size=(anchor_size, anchor_size), mode="area")
    else:
        fake_anchor = fake
        real_anchor = real
    lowfreq_loss = F.smooth_l1_loss(fake_anchor, real_anchor.detach(), beta=0.05)
    fake_mask = torch.sigmoid((fake_anchor - 0.35) / 0.08)
    real_mask = torch.sigmoid((real_anchor.detach() - 0.35) / 0.08)
    mask_block = 8 if anchor_size >= 96 else 4
    fake_mask_low = F.avg_pool2d(fake_mask, kernel_size=mask_block, stride=mask_block)
    real_mask_low = F.avg_pool2d(real_mask, kernel_size=mask_block, stride=mask_block)
    mask_loss = F.smooth_l1_loss(fake_mask_low, real_mask_low.detach(), beta=0.05)

    patch_losses: List[torch.Tensor] = []
    patch_size = min(128, fake_01.shape[-1], fake_01.shape[-2])
    if patch_size >= 64:
        full_fake = full_fake_ref
        full_real = full_real_ref
        h, w = full_fake.shape[-2:]
        for _ in range(2):
            y = int(torch.randint(0, max(h - patch_size + 1, 1), (1,), device=full_fake.device).item())
            x = int(torch.randint(0, max(w - patch_size + 1, 1), (1,), device=full_fake.device).item())
            f_patch = full_fake[..., y : y + patch_size, x : x + patch_size]
            r_patch = full_real[..., y : y + patch_size, x : x + patch_size]
            mu_f = F.avg_pool2d(f_patch, window, stride=1, padding=window // 2)
            mu_r = F.avg_pool2d(r_patch, window, stride=1, padding=window // 2)
            sigma_f = F.avg_pool2d(f_patch * f_patch, window, stride=1, padding=window // 2) - mu_f * mu_f
            sigma_r = F.avg_pool2d(r_patch * r_patch, window, stride=1, padding=window // 2) - mu_r * mu_r
            sigma_fr = F.avg_pool2d(f_patch * r_patch, window, stride=1, padding=window // 2) - mu_f * mu_r
            patch_ssim = ((2.0 * mu_f * mu_r + c1) * (2.0 * sigma_fr + c2)) / (
                (mu_f.square() + mu_r.square() + c1) * (sigma_f + sigma_r + c2)
            ).clamp_min(1e-6)
            patch_losses.append((1.0 - patch_ssim.clamp(-1.0, 1.0)).mean())
    patch_loss = sum(patch_losses) / len(patch_losses) if patch_losses else fake.new_zeros(())
    return ssim_loss + 0.35 * patch_loss + 0.08 * lowfreq_loss + 0.12 * mask_loss


def carbon_morphology_anchor_loss(
    fake_01: torch.Tensor,
    real_01: torch.Tensor,
    threshold: float = 0.35,
    temperature: float = 0.08,
) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = nearest_reference_batch(fake, real_01, size=96).detach().float().clamp(0.0, 1.0)
    temp = max(float(temperature), 1e-3)
    losses: List[torch.Tensor] = []
    for size in (32, 64, 128):
        if size > min(fake.shape[-2:]):
            continue
        fake_low = F.interpolate(fake, size=(size, size), mode="area")
        real_low = F.interpolate(real, size=(size, size), mode="area")
        losses.append(F.smooth_l1_loss(fake_low, real_low.detach(), beta=0.045))

        fake_mask = torch.sigmoid((fake_low - float(threshold)) / temp)
        real_mask = torch.sigmoid((real_low.detach() - float(threshold)) / temp)
        intersection = (fake_mask * real_mask).sum(dim=(1, 2, 3))
        union = fake_mask.sum(dim=(1, 2, 3)) + real_mask.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-5) / union.clamp_min(1e-5)
        losses.append(0.30 * dice_loss.mean())

        fake_edge = critic_edge_map(fake_low)
        real_edge = critic_edge_map(real_low)
        losses.append(0.35 * F.smooth_l1_loss(fake_edge, real_edge.detach(), beta=0.025))

        if size >= 64:
            block = max(4, size // 8)
            fake_blocks = F.avg_pool2d(fake_mask, kernel_size=block, stride=block)
            real_blocks = F.avg_pool2d(real_mask, kernel_size=block, stride=block)
            losses.append(0.25 * F.smooth_l1_loss(fake_blocks, real_blocks.detach(), beta=0.040))

    return sum(losses) / max(len(losses), 1)


def local_structure_stats_loss(fake_01: torch.Tensor, real_01: torch.Tensor, scales: Sequence[int] = (16, 32, 64)) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = nearest_reference_batch(fake, real_01, size=64)
    losses = []
    for scale in scales:
        if scale <= 1 or scale > min(fake.shape[-2:]):
            continue
        fake_mean = F.avg_pool2d(fake, kernel_size=scale, stride=scale)
        real_mean = F.avg_pool2d(real, kernel_size=scale, stride=scale)
        fake_sq = F.avg_pool2d(fake * fake, kernel_size=scale, stride=scale)
        real_sq = F.avg_pool2d(real * real, kernel_size=scale, stride=scale)
        fake_std = (fake_sq - fake_mean.square()).clamp_min(1e-8).sqrt()
        real_std = (real_sq - real_mean.square()).clamp_min(1e-8).sqrt()
        losses.append(F.smooth_l1_loss(fake_mean, real_mean, beta=0.03))
        losses.append(F.smooth_l1_loss(fake_std, real_std, beta=0.03))
    if not losses:
        return fake.new_zeros(())
    return sum(losses) / len(losses)


def patch_texture_stats_loss(
    fake_01: torch.Tensor,
    real_01: torch.Tensor,
    patch_size: int = 96,
    num_patches: int = 4,
    threshold: float = 0.35,
) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = real_01.detach().float().clamp(0.0, 1.0)
    h, w = fake.shape[-2:]
    size = max(16, min(int(patch_size), h, w))
    count = max(1, int(num_patches))
    losses = []
    for _ in range(count):
        y = int(torch.randint(0, max(h - size + 1, 1), (1,), device=fake.device).item())
        x = int(torch.randint(0, max(w - size + 1, 1), (1,), device=fake.device).item())
        perm = torch.randperm(real.size(0), device=real.device)
        f = fake[..., y : y + size, x : x + size]
        r = real.index_select(0, perm)[..., y : y + size, x : x + size]
        f_mean = f.mean(dim=(2, 3))
        r_mean = r.mean(dim=(2, 3))
        f_std = f.std(dim=(2, 3), unbiased=False)
        r_std = r.std(dim=(2, 3), unbiased=False)
        f_grad = torch_gradient_mean(f)
        r_grad = torch_gradient_mean(r).detach()
        f_density = torch.sigmoid((f - threshold) / 0.08).mean(dim=(2, 3))
        r_density = torch.sigmoid((r - threshold) / 0.08).mean(dim=(2, 3))
        f_interface = 0.5 * (
            (f[..., 1:] - f[..., :-1]).abs().mean(dim=(2, 3))
            + (f[..., 1:, :] - f[..., :-1, :]).abs().mean(dim=(2, 3))
        )
        r_interface = 0.5 * (
            (r[..., 1:] - r[..., :-1]).abs().mean(dim=(2, 3))
            + (r[..., 1:, :] - r[..., :-1, :]).abs().mean(dim=(2, 3))
        )
        losses.append(F.smooth_l1_loss(f_mean, r_mean.detach(), beta=0.04))
        losses.append(F.smooth_l1_loss(f_std, r_std.detach(), beta=0.04))
        losses.append(F.smooth_l1_loss(f_density, r_density.detach(), beta=0.04))
        losses.append(F.smooth_l1_loss(f_interface, r_interface.detach(), beta=0.02))
        losses.append(F.smooth_l1_loss(f_grad, r_grad, beta=0.01))
    return sum(losses) / len(losses)


def binary_edge_distribution_loss(
    fake_01: torch.Tensor,
    real_01: torch.Tensor,
    threshold: float = 0.35,
    temperature: float = 0.08,
) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = nearest_reference_batch(fake, real_01, size=64)
    temp = max(float(temperature), 1e-3)
    fake_mask = torch.sigmoid((fake - float(threshold)) / temp)
    real_mask = torch.sigmoid((real.detach() - float(threshold)) / temp)

    losses: List[torch.Tensor] = []
    for scale in (8, 16, 32):
        if scale <= min(fake.shape[-2:]):
            fake_density = F.avg_pool2d(fake_mask, kernel_size=scale, stride=scale)
            real_density = F.avg_pool2d(real_mask, kernel_size=scale, stride=scale)
            losses.append(F.smooth_l1_loss(fake_density, real_density.detach(), beta=0.035))

    fake_dx = (fake_mask[..., 1:] - fake_mask[..., :-1]).abs()
    real_dx = (real_mask[..., 1:] - real_mask[..., :-1]).abs()
    fake_dy = (fake_mask[..., 1:, :] - fake_mask[..., :-1, :]).abs()
    real_dy = (real_mask[..., 1:, :] - real_mask[..., :-1, :]).abs()
    losses.append(F.smooth_l1_loss(fake_dx.mean(), real_dx.detach().mean(), beta=0.01))
    losses.append(F.smooth_l1_loss(fake_dy.mean(), real_dy.detach().mean(), beta=0.01))

    fake_grad = critic_edge_map(fake)
    real_grad = critic_edge_map(real.detach())
    for scale in (8, 16):
        fake_edge = F.avg_pool2d(fake_grad, kernel_size=scale, stride=scale)
        real_edge = F.avg_pool2d(real_grad, kernel_size=scale, stride=scale)
        losses.append(F.smooth_l1_loss(fake_edge, real_edge.detach(), beta=0.02))

    return sum(losses) / max(len(losses), 1)


def carbon_phase_transition_envelope_loss(
    fake_01: torch.Tensor,
    real_01: torch.Tensor,
    threshold: float = 0.35,
) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = nearest_reference_batch(fake, real_01, size=64).detach().float().clamp(0.0, 1.0)
    losses: List[torch.Tensor] = []

    fake_dark = torch.sigmoid((0.06 - fake) / 0.025)
    real_dark = torch.sigmoid((0.06 - real) / 0.025)
    fake_bright = torch.sigmoid((fake - 0.90) / 0.040)
    real_bright = torch.sigmoid((real - 0.90) / 0.040)
    fake_saturated = torch.sigmoid((fake - 0.965) / 0.018)
    real_saturated = torch.sigmoid((real - 0.965) / 0.018)
    fake_transition = torch.sigmoid((fake - 0.08) / 0.045) * torch.sigmoid((0.92 - fake) / 0.045)
    real_transition = torch.sigmoid((real - 0.08) / 0.045) * torch.sigmoid((0.92 - real) / 0.045)

    for scale in (32, 64, 128):
        if scale > min(fake.shape[-2:]):
            continue
        fake_dark_blocks = F.avg_pool2d(fake_dark, kernel_size=scale, stride=scale)
        real_dark_blocks = F.avg_pool2d(real_dark, kernel_size=scale, stride=scale)
        fake_bright_blocks = F.avg_pool2d(fake_bright, kernel_size=scale, stride=scale)
        real_bright_blocks = F.avg_pool2d(real_bright, kernel_size=scale, stride=scale)
        fake_sat_blocks = F.avg_pool2d(fake_saturated, kernel_size=scale, stride=scale)
        real_sat_blocks = F.avg_pool2d(real_saturated, kernel_size=scale, stride=scale)
        fake_trans_blocks = F.avg_pool2d(fake_transition, kernel_size=scale, stride=scale)
        real_trans_blocks = F.avg_pool2d(real_transition, kernel_size=scale, stride=scale)
        fake_mean_blocks = F.avg_pool2d(fake, kernel_size=scale, stride=scale)
        real_mean_blocks = F.avg_pool2d(real, kernel_size=scale, stride=scale)

        for fake_blocks, real_blocks, beta in (
            (fake_dark_blocks, real_dark_blocks, 0.040),
            (fake_bright_blocks, real_bright_blocks, 0.030),
            (fake_trans_blocks, real_trans_blocks, 0.035),
        ):
            losses.append(F.smooth_l1_loss(fake_blocks.mean(), real_blocks.mean().detach(), beta=beta))
            losses.append(F.smooth_l1_loss(fake_blocks.std(unbiased=False), real_blocks.std(unbiased=False).detach(), beta=beta))

        real_bright_hi = torch.quantile(real_bright_blocks.detach().flatten(), 0.97)
        real_sat_hi = torch.quantile(real_sat_blocks.detach().flatten(), 0.98)
        real_dark_hi = torch.quantile(real_dark_blocks.detach().flatten(), 0.995)
        real_mean_low = torch.quantile(real_mean_blocks.detach().flatten(), 0.01)
        real_mean_high = torch.quantile(real_mean_blocks.detach().flatten(), 0.99)
        real_trans_low = torch.quantile(real_trans_blocks.detach().flatten(), 0.05)

        losses.append(F.relu(fake_bright_blocks - (real_bright_hi + 0.055)).square().mean())
        losses.append(1.5 * F.relu(fake_sat_blocks - (real_sat_hi + 0.020)).square().mean())
        losses.append(0.35 * F.relu(fake_dark_blocks - (real_dark_hi + 0.040)).square().mean())
        losses.append(0.45 * F.relu((real_mean_low - 0.025) - fake_mean_blocks).square().mean())
        losses.append(0.65 * F.relu(fake_mean_blocks - (real_mean_high + 0.025)).square().mean())
        non_dark_or_white = ((fake_dark_blocks < 0.78) & (fake_bright_blocks < 0.62)).float()
        transition_floor = F.relu((real_trans_low - 0.020) - fake_trans_blocks).square() * non_dark_or_white
        losses.append(0.75 * transition_floor.mean())

    fake_edge = critic_edge_map(fake)
    real_edge = critic_edge_map(real)
    fake_edge_weight = fake_edge / fake_edge.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    real_edge_weight = real_edge / real_edge.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    fake_edge_transition = (fake_edge_weight * fake_transition).mean()
    real_edge_transition = (real_edge_weight * real_transition).mean().detach()
    fake_edge_saturation = (fake_edge_weight * (fake_saturated + fake_dark)).mean()
    real_edge_saturation = (real_edge_weight * (real_saturated + real_dark)).mean().detach()
    losses.append(F.smooth_l1_loss(fake_edge_transition, real_edge_transition, beta=0.030))
    losses.append(0.35 * F.smooth_l1_loss(fake_edge_saturation, real_edge_saturation, beta=0.040))

    return sum(losses) / max(len(losses), 1)


def tensor_phase_envelope_stats(images: torch.Tensor, block_size: int = 64) -> Dict[str, float]:
    x = images.detach().float().clamp(0.0, 1.0)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.numel() == 0:
        return {
            "block_bright64_q95": 0.0,
            "block_dark64_q95": 0.0,
            "block_transition64_q05": 0.0,
            "bright_pixel_fraction": 0.0,
            "dark_pixel_fraction": 0.0,
        }
    block = max(4, min(int(block_size), int(x.shape[-1]), int(x.shape[-2])))
    bright = (x >= 0.97).float()
    dark = (x <= 0.03).float()
    transition = ((x > 0.08) & (x < 0.92)).float()
    bright_blocks = F.avg_pool2d(bright, kernel_size=block, stride=block).flatten(1)
    dark_blocks = F.avg_pool2d(dark, kernel_size=block, stride=block).flatten(1)
    transition_blocks = F.avg_pool2d(transition, kernel_size=block, stride=block).flatten(1)
    bright_block_max = bright_blocks.max(dim=1).values
    dark_block_max = dark_blocks.max(dim=1).values
    transition_block_min = transition_blocks.min(dim=1).values
    return {
        "block_bright64_q95": float(torch.quantile(bright_block_max, 0.95).cpu()),
        "block_dark64_q95": float(torch.quantile(dark_block_max, 0.95).cpu()),
        "block_transition64_q05": float(torch.quantile(transition_block_min, 0.05).cpu()),
        "bright_pixel_fraction": float(bright.mean().cpu()),
        "dark_pixel_fraction": float(dark.mean().cpu()),
    }


def mode_seeking_diversity_loss(images_01: torch.Tensor, latents: torch.Tensor, target_ratio: float) -> torch.Tensor:
    if images_01.size(0) <= 1 or target_ratio <= 0.0:
        return images_01.new_zeros(())
    rolled_images = torch.roll(images_01, shifts=1, dims=0)
    rolled_latents = torch.roll(latents, shifts=1, dims=0)
    image_delta = (images_01 - rolled_images).abs().flatten(1).mean(dim=1)
    latent_delta = (latents - rolled_latents).abs().mean(dim=1).clamp_min(1e-6)
    ratio = image_delta / latent_delta
    target = images_01.new_tensor(float(target_ratio)).clamp_min(1e-6)
    return (F.relu(target - ratio) / target).mean()


def torch_gradient_mean(images_01: torch.Tensor) -> torch.Tensor:
    dx = images_01[:, :, :, 1:] - images_01[:, :, :, :-1]
    dy = images_01[:, :, 1:, :] - images_01[:, :, :-1, :]
    return 0.5 * (dx.abs().mean() + dy.abs().mean())


def differentiable_micro_loss(fake_01: torch.Tensor, real_01: torch.Tensor) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = real_01.float().clamp(0.0, 1.0)
    fake_std = fake.std(unbiased=False)
    real_std = real.std(unbiased=False).detach().clamp_min(1e-6)
    fake_grad = torch_gradient_mean(fake)
    real_grad = torch_gradient_mean(real).detach().clamp_min(1e-6)
    mean_loss = (fake.mean() - real.mean().detach()).abs() / real.mean().detach().abs().clamp_min(1e-6)
    std_loss = (fake_std - real_std).abs() / real_std
    grad_loss = (fake_grad - real_grad).abs() / real_grad
    return mean_loss + std_loss + grad_loss


def intensity_distribution_loss(fake_01: torch.Tensor, real_01: torch.Tensor) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0)
    real = real_01.detach().float().clamp(0.0, 1.0)
    fake_mean = fake.mean()
    real_mean = real.mean().detach().clamp_min(1e-6)
    fake_std = fake.std(unbiased=False)
    real_std = real.std(unbiased=False).detach().clamp_min(1e-6)
    losses: List[torch.Tensor] = [
        (fake_mean - real_mean).abs() / real_mean,
        (fake_std - real_std).abs() / real_std,
    ]
    for level in (0.55, 0.70, 0.82):
        fake_tail = F.relu(fake - level).mean()
        real_tail = F.relu(real - level).mean().detach().clamp_min(1e-5)
        losses.append(0.65 * (fake_tail - real_tail).abs() / real_tail)
    return torch.stack(losses).mean()


def quantile_distribution_loss(fake_01: torch.Tensor, real_01: torch.Tensor) -> torch.Tensor:
    fake = fake_01.float().clamp(0.0, 1.0).flatten()
    real = real_01.detach().float().clamp(0.0, 1.0).flatten()
    levels = fake.new_tensor([0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98])
    fake_q = torch.quantile(fake, levels)
    real_q = torch.quantile(real, levels).detach()
    return F.smooth_l1_loss(fake_q, real_q, beta=0.025)


def raw_output_distribution_loss(fake_01: torch.Tensor, real_01: torch.Tensor) -> torch.Tensor:
    return intensity_distribution_loss(fake_01, real_01) + 2.0 * quantile_distribution_loss(fake_01, real_01)


def build_reference_quantiles(real_eval: torch.Tensor) -> torch.Tensor:
    return real_eval.detach().cpu().float().clamp(0.0, 1.0).flatten().sort().values


def quantile_match_images(images_01: torch.Tensor, reference_sorted: torch.Tensor) -> torch.Tensor:
    if reference_sorted.numel() == 0:
        return images_01
    x = images_01.detach().cpu().float().clamp(0.0, 1.0)
    ref = reference_sorted.detach().cpu().float().clamp(0.0, 1.0)
    n, c, h, w = x.shape
    flat_count = c * h * w
    if ref.numel() == flat_count:
        ref_values = ref
    else:
        idx = torch.linspace(0, ref.numel() - 1, flat_count).round().long()
        ref_values = ref[idx]
    out = torch.empty_like(x)
    for i in range(n):
        flat = x[i].flatten()
        order = torch.argsort(flat)
        matched = torch.empty_like(flat)
        matched[order] = ref_values.to(dtype=flat.dtype)
        out[i] = matched.view_as(x[i])
    return out


def mean_micro_relative_diff(metrics: Dict[str, object], clip: float = 2.0) -> float:
    real = metrics.get("microstructure_real_mean", {})
    generated = metrics.get("microstructure_generated_mean", {})
    if not isinstance(real, dict) or not isinstance(generated, dict):
        return 0.0
    values = []
    for key, real_value in real.items():
        if key in generated:
            denom = abs(float(real_value)) + 1e-6
            values.append(min(abs(float(generated[key]) - float(real_value)) / denom, clip))
    return float(np.mean(values)) if values else 0.0


GAN_TARGET_RANGES: Dict[str, Tuple[float, float]] = {
    "mean_intensity": (0.225, 0.246),
    "std_intensity": (0.260, 0.300),
    "p95_intensity": (0.82, 0.91),
    "area_fraction": (0.258, 0.286),
    "interface_density": (0.085, 0.105),
    "gradient_mean": (0.105, 0.125),
    "edge_density": (0.57, 0.64),
    "autocorr_x": (0.89, 0.93),
    "autocorr_y": (0.89, 0.93),
    "component_count": (1200.0, 1450.0),
    "mean_component_area": (48.0, 65.0),
    "component_area_max": (5000.0, 9500.0),
    "correlation_length": (3.0, 3.5),
    "hist_entropy": (3.0, 3.4),
}


def range_penalty(value: float, low: float, high: float) -> float:
    width = max(high - low, 1e-6)
    if value < low:
        return float((low - value) / width)
    if value > high:
        return float((value - high) / width)
    return 0.0


def target_compliance_details(metrics: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    fid = float(metrics["fid_feature_no_pretrained"])
    mmd = float(metrics["mmd_rbf"])
    ssim = float(metrics["ssim_nearest"])
    ms_ssim = float(metrics.get("ms_ssim_nearest", ssim))
    binary_mask_ssim = float(metrics.get("binary_mask_ssim_nearest", ssim))
    generated = metrics.get("microstructure_generated_mean", {})
    micro_penalties: Dict[str, float] = {}
    if isinstance(generated, dict):
        for key, (low, high) in GAN_TARGET_RANGES.items():
            if key in generated:
                micro_penalties[key] = range_penalty(float(generated[key]), low, high)
    hard_penalties = {
        "fid": max(0.0, fid - args.target_fid_max) / max(args.target_fid_max, 1e-6),
        "mmd": max(0.0, mmd - args.target_mmd_max) / max(args.target_mmd_max, 1e-6),
        "ssim": max(0.0, args.target_ssim_min - ssim) / max(args.target_ssim_min, 1e-6),
        "ms_ssim": max(0.0, args.target_ms_ssim_min - ms_ssim) / max(args.target_ms_ssim_min, 1e-6),
        "binary_mask_ssim": max(0.0, args.target_binary_mask_ssim_min - binary_mask_ssim)
        / max(args.target_binary_mask_ssim_min, 1e-6),
    }
    return {
        "target_pass": (
            hard_penalties["fid"] == 0.0
            and hard_penalties["mmd"] == 0.0
            and hard_penalties["ssim"] == 0.0
        ),
        "hard_penalties": hard_penalties,
        "micro_penalties": micro_penalties,
        "micro_penalty_mean": float(np.mean(list(micro_penalties.values()))) if micro_penalties else 0.0,
        "micro_penalty_max": float(max(micro_penalties.values())) if micro_penalties else 0.0,
        "target_ranges": GAN_TARGET_RANGES,
    }


def candidate_selection_score(
    metrics: Dict[str, object],
    generated: torch.Tensor,
    real_eval: torch.Tensor,
    train_stats: Dict[str, float],
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, float]]:
    fid = float(metrics["fid_feature_no_pretrained"])
    mmd = float(metrics["mmd_rbf"])
    ssim = float(metrics["ssim_nearest"])
    ms_ssim = float(metrics.get("ms_ssim_nearest", ssim))
    binary_mask_ssim = float(metrics.get("binary_mask_ssim_nearest", ssim))
    micro_relative_diff = mean_micro_relative_diff(metrics, clip=args.micro_relative_clip)
    gen_diversity = tensor_diversity(generated)
    real_diversity = tensor_diversity(real_eval)
    gen_pairwise_l1 = tensor_pairwise_l1(generated)
    real_pairwise_l1 = tensor_pairwise_l1(real_eval)
    diversity_gap = abs(gen_diversity - real_diversity) / (real_diversity + 1e-8)
    pairwise_l1_gap = abs(gen_pairwise_l1 - real_pairwise_l1) / (real_pairwise_l1 + 1e-8)
    diversity_ratio = gen_diversity / (real_diversity + 1e-8)
    pairwise_l1_ratio = gen_pairwise_l1 / (real_pairwise_l1 + 1e-8)
    gen_shift_texture = tensor_shift_texture(generated)
    real_shift_texture = tensor_shift_texture(real_eval)
    shift_texture_ratio = gen_shift_texture / (real_shift_texture + 1e-8)
    periodicity_penalty = max(0.0, args.min_shift_texture_ratio - shift_texture_ratio)
    gen_phase_stats = tensor_phase_envelope_stats(generated)
    real_phase_stats = tensor_phase_envelope_stats(real_eval)
    phase_bright_excess = max(
        0.0,
        gen_phase_stats["block_bright64_q95"]
        - (real_phase_stats["block_bright64_q95"] + args.phase_bright_block_margin),
    )
    phase_dark_excess = max(
        0.0,
        gen_phase_stats["block_dark64_q95"]
        - (real_phase_stats["block_dark64_q95"] + args.phase_dark_block_margin),
    )
    phase_transition_deficit = max(
        0.0,
        (real_phase_stats["block_transition64_q05"] - args.phase_transition_block_margin)
        - gen_phase_stats["block_transition64_q05"],
    )
    phase_envelope_penalty = (
        phase_bright_excess
        + 0.40 * phase_dark_excess
        + 0.75 * phase_transition_deficit
    )
    phase_envelope_pass = phase_envelope_penalty <= args.max_phase_envelope_penalty
    mode_collapse_penalty = (
        max(0.0, args.min_generated_diversity_ratio - diversity_ratio)
        + max(0.0, args.min_pairwise_l1_ratio - pairwise_l1_ratio)
    )
    compliance = target_compliance_details(metrics, args)
    hard_penalties = compliance["hard_penalties"]
    micro_penalty_mean = float(compliance["micro_penalty_mean"])
    micro_penalty_max = float(compliance["micro_penalty_max"])
    real_raw = float(train_stats.get("critic_real_score_raw", 0.0))
    fake_raw = float(train_stats.get("critic_fake_score_raw", 0.0))
    w_dist_raw = float(train_stats.get("w_dist_raw", real_raw - fake_raw))
    d_loss_raw = float(train_stats.get("d_loss_raw", fake_raw - real_raw))
    g_adv_raw = float(train_stats.get("g_adv_loss_raw", train_stats.get("g_adv_loss", 0.0)))
    topology_train_errors = {
        "density": float(train_stats.get("topology_density_error", 0.0)),
        "uniformity": float(train_stats.get("topology_uniformity_error", 0.0)),
        "aggregation": float(train_stats.get("topology_aggregation_error", 0.0)),
    }
    topology_tail_error_mean = float(train_stats.get("topology_tail_error_mean", 0.0))
    topology_tail_error_max = float(train_stats.get("topology_tail_error_max", 0.0))
    topology_tail_error_std = float(train_stats.get("topology_tail_error_std", 0.0))
    w_dist_tail_std = float(train_stats.get("w_dist_tail_std", 0.0))
    g_adv_tail_std = float(train_stats.get("g_adv_tail_std", 0.0))
    d_loss_tail_std = float(train_stats.get("d_loss_tail_std", 0.0))
    score_gap_tail_std = float(train_stats.get("score_gap_tail_std", 0.0))
    topology_train_penalty = sum(
        max(0.0, value - args.target_topology_error_max) for value in topology_train_errors.values()
    )
    topology_train_penalty += max(0.0, topology_tail_error_max - args.target_topology_error_max)
    topology_train_penalty += max(0.0, topology_tail_error_std - args.target_topology_error_std_max)
    topology_train_pass = (
        all(value <= args.target_topology_error_max for value in topology_train_errors.values())
        and topology_tail_error_mean <= args.target_topology_error_max
        and topology_tail_error_std <= args.target_topology_error_std_max
    )
    d_confidence_penalty = (
        max(0.0, args.min_healthy_w_dist - w_dist_raw)
        + max(0.0, w_dist_raw - args.max_healthy_w_dist)
        + max(0.0, fake_raw - real_raw)
    )
    curve_health_penalty = gan_curve_health_penalty(train_stats, args)
    generated_micro = metrics.get("microstructure_generated_mean", {})
    micro_hard_penalties: Dict[str, float] = {}
    if isinstance(generated_micro, dict):
        hard_keys = (
            "component_count",
            "mean_component_area",
            "interface_density",
            "autocorr_x",
            "autocorr_y",
            "component_area_max",
        )
        for key in hard_keys:
            if key in generated_micro and key in GAN_TARGET_RANGES:
                low, high = GAN_TARGET_RANGES[key]
                micro_hard_penalties[key] = range_penalty(float(generated_micro[key]), low, high)
    if isinstance(generated_micro, dict):
        hard_micro_pass = (
            micro_hard_penalties.get("component_count", float("inf")) <= args.best_hard_micro_penalty_max
            and micro_hard_penalties.get("mean_component_area", float("inf")) <= args.best_hard_micro_penalty_max
            and micro_hard_penalties.get("interface_density", float("inf")) <= args.best_hard_micro_penalty_max
            and float(generated_micro.get("autocorr_x", 0.0)) >= args.best_min_autocorr
            and float(generated_micro.get("autocorr_y", 0.0)) >= args.best_min_autocorr
            and micro_hard_penalties.get("component_area_max", float("inf"))
            <= args.best_component_area_max_penalty_max
            and float(generated_micro.get("component_area_max", float("inf"))) <= args.best_max_component_area_max
        )
    else:
        hard_micro_pass = False
    raw_wgan_pass = (
        w_dist_raw >= args.min_healthy_w_dist
        and w_dist_raw <= args.max_healthy_w_dist
        and real_raw > fake_raw
        and abs(g_adv_raw) <= args.max_abs_g_adv_loss
        and w_dist_tail_std <= args.max_w_dist_tail_std
        and g_adv_tail_std <= args.max_g_adv_tail_std
        and d_loss_tail_std <= args.max_d_loss_tail_std
        and score_gap_tail_std <= args.max_score_gap_tail_std
    )
    balanced_candidate_pass = (
        diversity_ratio >= args.min_generated_diversity_ratio
        and pairwise_l1_ratio >= args.min_pairwise_l1_ratio
        and curve_health_penalty <= args.max_curve_health_penalty
        and ssim >= args.target_ssim_min
        and ms_ssim >= args.target_ms_ssim_min
        and binary_mask_ssim >= args.best_balanced_binary_min
        and micro_relative_diff <= args.best_balanced_micro_relative_max
        and fid <= args.best_balanced_fid_max
        and mmd <= args.best_balanced_mmd_max
        and raw_wgan_pass
        and hard_micro_pass
        and topology_train_pass
        and periodicity_penalty == 0.0
        and phase_envelope_pass
    )
    closure_pass = (
        bool(compliance["target_pass"])
        and balanced_candidate_pass
        and binary_mask_ssim >= args.target_binary_mask_ssim_min
    )
    best_model_eligible = balanced_candidate_pass
    structural_surplus_bonus = 0.0
    if best_model_eligible:
        structural_surplus_bonus = args.score_structural_bonus_weight * (
            max(0.0, ssim - args.target_ssim_min)
            + max(0.0, ms_ssim - args.target_ms_ssim_min)
            + 0.5 * max(0.0, binary_mask_ssim - args.target_binary_mask_ssim_min)
            + 0.25 * max(0.0, binary_mask_ssim - args.best_balanced_binary_min)
        )
    fallback_score = (
        args.score_hard_target_weight
        * (
            float(hard_penalties["fid"])
            + float(hard_penalties["mmd"])
            + float(hard_penalties["ssim"])
            + 0.5 * float(hard_penalties["ms_ssim"])
            + 0.5 * float(hard_penalties["binary_mask_ssim"])
        )
        + args.score_fid_weight * fid
        + args.score_mmd_weight * mmd
        + args.score_ssim_weight * max(0.0, args.target_ssim_min - ssim)
        + args.score_structural_similarity_weight
        * (
            max(0.0, args.target_ms_ssim_min - ms_ssim)
            + max(0.0, args.target_binary_mask_ssim_min - binary_mask_ssim)
        )
        + args.score_micro_weight * (micro_relative_diff + micro_penalty_mean + 0.5 * micro_penalty_max)
        + args.score_diversity_weight * (diversity_gap + pairwise_l1_gap)
        + args.score_mode_collapse_weight * mode_collapse_penalty
        + args.score_d_balance_weight * d_confidence_penalty
        + args.score_curve_health_weight * curve_health_penalty
        + args.score_topology_train_weight * topology_train_penalty
        + args.score_periodicity_weight * periodicity_penalty
        + args.score_phase_envelope_weight * phase_envelope_penalty
        + args.score_hard_target_weight * (0.0 if raw_wgan_pass else 1.0)
        + args.score_hard_target_weight * (0.0 if hard_micro_pass else 1.0)
        + args.score_hard_target_weight * (0.0 if topology_train_pass else 1.0)
        - structural_surplus_bonus
    )
    eligible_distribution_score = (
        args.eligible_score_fid_weight * fid
        + args.eligible_score_mmd_weight * mmd
        + args.eligible_score_ssim_penalty_weight * max(0.0, args.eligible_score_ssim_soft_min - ssim)
        + args.eligible_score_ms_ssim_penalty_weight
        * max(0.0, args.eligible_score_ms_ssim_soft_min - ms_ssim)
        + args.eligible_score_binary_penalty_weight
        * max(0.0, args.eligible_score_binary_soft_min - binary_mask_ssim)
        + args.eligible_score_micro_weight * micro_relative_diff
        + args.eligible_score_curve_weight * curve_health_penalty
        + args.score_phase_envelope_weight * phase_envelope_penalty
    )
    score = eligible_distribution_score if best_model_eligible else fallback_score
    details = {
        "fid": fid,
        "mmd": mmd,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
        "binary_mask_ssim": binary_mask_ssim,
        "micro_relative_diff_mean": micro_relative_diff,
        "real_diversity": real_diversity,
        "generated_diversity": gen_diversity,
        "diversity_ratio": diversity_ratio,
        "diversity_gap": diversity_gap,
        "real_pairwise_l1": real_pairwise_l1,
        "generated_pairwise_l1": gen_pairwise_l1,
        "pairwise_l1_ratio": pairwise_l1_ratio,
        "pairwise_l1_gap": pairwise_l1_gap,
        "real_shift_texture": real_shift_texture,
        "generated_shift_texture": gen_shift_texture,
        "shift_texture_ratio": shift_texture_ratio,
        "periodicity_penalty": periodicity_penalty,
        "phase_envelope_penalty": phase_envelope_penalty,
        "phase_bright_excess": phase_bright_excess,
        "phase_dark_excess": phase_dark_excess,
        "phase_transition_deficit": phase_transition_deficit,
        "phase_envelope_pass": phase_envelope_pass,
        "generated_phase_envelope_stats": gen_phase_stats,
        "real_phase_envelope_stats": real_phase_stats,
        "mode_collapse_penalty": mode_collapse_penalty,
        "target_compliance": compliance,
        "critic_real_score_raw": real_raw,
        "critic_fake_score_raw": fake_raw,
        "w_dist_raw": w_dist_raw,
        "d_loss_raw": d_loss_raw,
        "g_adv_loss_raw": g_adv_raw,
        "d_confidence_penalty": d_confidence_penalty,
        "curve_health_penalty": curve_health_penalty,
        "w_dist_tail_std": w_dist_tail_std,
        "g_adv_tail_std": g_adv_tail_std,
        "d_loss_tail_std": d_loss_tail_std,
        "score_gap_tail_std": score_gap_tail_std,
        "topology_train_errors": topology_train_errors,
        "topology_tail_error_mean": topology_tail_error_mean,
        "topology_tail_error_max": topology_tail_error_max,
        "topology_tail_error_std": topology_tail_error_std,
        "topology_train_penalty": topology_train_penalty,
        "topology_train_pass": topology_train_pass,
        "micro_hard_penalties": micro_hard_penalties,
        "raw_wgan_pass": raw_wgan_pass,
        "hard_micro_pass": hard_micro_pass,
        "balanced_candidate_pass": balanced_candidate_pass,
        "fallback_selection_score": fallback_score,
        "eligible_distribution_score": eligible_distribution_score,
        "structural_surplus_bonus": structural_surplus_bonus,
        "best_model_eligible": best_model_eligible,
        "closure_pass": closure_pass,
    }
    return float(score), details


def candidate_rank_key(record: Dict[str, object]) -> Tuple[int, float]:
    details = record.get("selection_details", {})
    best_model_eligible = False
    if isinstance(details, dict):
        best_model_eligible = bool(details.get("best_model_eligible", False))
    return (0 if best_model_eligible else 1, float(record["selection_score"]))


def is_top_candidate(record: Dict[str, object], top_records: List[Dict[str, object]], keep_top_k: int) -> bool:
    if len(top_records) < keep_top_k:
        return True
    worst_record = max(top_records, key=candidate_rank_key)
    return candidate_rank_key(record) < candidate_rank_key(worst_record)


def update_top_candidates(top_records: List[Dict[str, object]], record: Dict[str, object], keep_top_k: int) -> List[Dict[str, object]]:
    merged = top_records + [record]
    merged.sort(key=candidate_rank_key)
    return merged[:keep_top_k]


PAPER_COMPARISON_COLUMNS = [
    "comparison_criterion",
    "reference_or_target",
    "description",
    "measured_statistics",
    "AR",
    "VAE",
    "Diffusion",
    "GAN",
    "WGAN",
    "selected_result_or_note",
    "note_1",
    "note_2",
]


def csv_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        if not math.isfinite(value_float):
            return ""
        return f"{value_float:.6g}"
    return str(value)


def nested_get(mapping: object, keys: Sequence[str], default: object = "") -> object:
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def load_paper_template_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    return [{column: str(row.get(column, "")) for column in PAPER_COMPARISON_COLUMNS} for row in rows]


def write_dict_rows_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_text(row.get(key, "")) for key in fieldnames})


def paper_wgan_value_for_dimension(record: Dict[str, object], args: argparse.Namespace, dimension: str) -> object:
    metrics = record.get("metrics", {})
    details = record.get("selection_details", {})
    train_stats = record.get("train_stats", {})
    generated_micro = nested_get(metrics, ["microstructure_generated_mean"], {})
    real_micro = nested_get(metrics, ["microstructure_real_mean"], {})
    epoch = record.get("epoch", "")

    fixed_values = {
        "Measured image count": record.get("num_real_images", ""),
        "Image dimensions": f"{args.image_size}x{args.image_size}",
        "Augmentation presentations per image": args.augment_repeats,
        "Generated evaluation count": record.get("num_generated_images", args.candidate_samples),
        "Training epochs / stopping epoch": epoch,
        "FID-like": nested_get(metrics, ["fid_feature_no_pretrained"]),
        "MMD-RBF": nested_get(metrics, ["mmd_rbf"]),
        "SSIM-nearest": nested_get(metrics, ["ssim_nearest"]),
        "PSNR-nearest": nested_get(metrics, ["psnr_nearest"]),
        "aggregate_morphology_error": nested_get(details, ["micro_relative_diff_mean"]),
        "Inter-image variation ratio": nested_get(details, ["diversity_ratio"]),
        "GAN pairwise L1 ratio": nested_get(details, ["pairwise_l1_ratio"]),
        "GAN d_real": f"{csv_text(nested_get(details, ['critic_real_score_raw']))} (critic)",
        "GAN d_fake": f"{csv_text(nested_get(details, ['critic_fake_score_raw']))} (critic)",
        "Composite selection score": record.get("selection_score", ""),
        "Parameter count (million)": "",
        "Training time (h)": "",
        "Generation time per image (s)": "",
        "Peak allocated GPU memory (GiB)": "",
        "Generation peak allocated GPU memory (MiB)": nested_get(train_stats, ["peak_vram_mb"]),
    }
    if dimension in fixed_values:
        return fixed_values[dimension]
    if isinstance(generated_micro, dict) and dimension in generated_micro:
        return generated_micro[dimension]
    if isinstance(real_micro, dict) and dimension == "measured_statistics":
        return ""
    return ""


def build_candidate_paper_rows(
    record: Dict[str, object],
    args: argparse.Namespace,
    template_rows: Sequence[Dict[str, str]],
    model_best_epoch: int,
    model_best_score: float,
) -> List[Dict[str, object]]:
    if template_rows:
        rows = [dict(row) for row in template_rows]
    else:
        rows = [
            {"comparison_criterion": key, "reference_or_target": "", "description": "", "measured_statistics": "", "AR": "", "VAE": "", "Diffusion": "", "GAN": "", "WGAN": "", "selected_result_or_note": "", "note_1": "", "note_2": ""}
            for key in (
                "Measured image count",
                "Image dimensions",
                "Augmentation presentations per image",
                "Generated evaluation count",
                "Training epochs / stopping epoch",
                "FID-like",
                "MMD-RBF",
                "SSIM-nearest",
                "PSNR-nearest",
                "aggregate_morphology_error",
                "Inter-image variation ratio",
                "GAN pairwise L1 ratio",
                "GAN d_real",
                "GAN d_fake",
                *GAN_TARGET_RANGES.keys(),
                "Composite selection score",
            )
        ]

    metrics = record.get("metrics", {})
    real_micro = nested_get(metrics, ["microstructure_real_mean"], {})
    for row in rows:
        dimension = row.get("comparison_criterion", "")
        if isinstance(real_micro, dict) and dimension in real_micro:
            row["measured_statistics"] = csv_text(real_micro[dimension])
        wgan_value = paper_wgan_value_for_dimension(record, args, dimension)
        if wgan_value != "":
            row["WGAN"] = csv_text(wgan_value)
        row.setdefault("note_1", "")
        row.setdefault("note_2", "")

    candidate_epoch = int(record.get("epoch", 0) or 0)
    selected_note = "Selected checkpoint epoch" if candidate_epoch == model_best_epoch else ""
    extra_rows = [
        {
            "comparison_criterion": "Candidate epoch",
            "reference_or_target": "DME-GAN candidate",
            "description": "Epoch associated with the retained candidate record",
            "measured_statistics": "",
            "AR": "",
            "VAE": "",
            "Diffusion": "",
            "GAN": "",
            "WGAN": candidate_epoch,
            "selected_result_or_note": selected_note,
            "note_1": f"selection_score={csv_text(record.get('selection_score', ''))}",
            "note_2": "",
        },
        {
            "comparison_criterion": "Checkpoint selected by the recorded criterion",
            "reference_or_target": "DME-GAN checkpoint selection",
            "description": "Ranking according to candidate_selection_score and retained candidate records",
            "measured_statistics": "",
            "AR": "",
            "VAE": "",
            "Diffusion": "",
            "GAN": "",
            "WGAN": model_best_epoch if model_best_epoch else "",
            "selected_result_or_note": "Model selection",
            "note_1": f"best_selection_score={csv_text(model_best_score)}" if math.isfinite(model_best_score) else "",
            "note_2": "",
        },
        {
            "comparison_criterion": "DME-GAN eligibility conditions satisfied",
            "reference_or_target": "TRUE indicates eligibility",
            "description": "Includes distribution, resemblance, training-state, morphology and variation conditions",
            "measured_statistics": "",
            "AR": "",
            "VAE": "",
            "Diffusion": "",
            "GAN": "",
            "WGAN": nested_get(record, ["selection_details", "best_model_eligible"]),
            "selected_result_or_note": "",
            "note_1": f"closure_pass={csv_text(nested_get(record, ['selection_details', 'closure_pass']))}",
            "note_2": "",
        },
    ]
    return [{column: row.get(column, "") for column in PAPER_COMPARISON_COLUMNS} for row in [*rows, *extra_rows]]


def flatten_candidate_record(record: Dict[str, object], model_best_epoch: int, model_best_score: float) -> Dict[str, object]:
    metrics = record.get("metrics", {})
    details = record.get("selection_details", {})
    train_stats = record.get("train_stats", {})
    generated_micro = nested_get(metrics, ["microstructure_generated_mean"], {})
    real_micro = nested_get(metrics, ["microstructure_real_mean"], {})
    abs_diff = nested_get(metrics, ["microstructure_abs_diff"], {})
    row: Dict[str, object] = {
        "epoch": record.get("epoch", ""),
        "is_model_selected_best": int(record.get("epoch", 0) == model_best_epoch),
        "model_selected_best_epoch": model_best_epoch if model_best_epoch else "",
        "model_selected_best_score": model_best_score if math.isfinite(model_best_score) else "",
        "selection_score": record.get("selection_score", ""),
        "best_model_eligible": nested_get(details, ["best_model_eligible"]),
        "closure_pass": nested_get(details, ["closure_pass"]),
        "target_pass": nested_get(details, ["target_compliance", "target_pass"]),
        "raw_wgan_pass": nested_get(details, ["raw_wgan_pass"]),
        "hard_micro_pass": nested_get(details, ["hard_micro_pass"]),
        "topology_train_pass": nested_get(details, ["topology_train_pass"]),
        "fid_feature_no_pretrained": nested_get(metrics, ["fid_feature_no_pretrained"]),
        "mmd_rbf": nested_get(metrics, ["mmd_rbf"]),
        "ssim_nearest": nested_get(metrics, ["ssim_nearest"]),
        "psnr_nearest": nested_get(metrics, ["psnr_nearest"]),
        "ms_ssim_nearest": nested_get(metrics, ["ms_ssim_nearest"]),
        "binary_mask_ssim_nearest": nested_get(metrics, ["binary_mask_ssim_nearest"]),
        "micro_relative_diff_mean": nested_get(details, ["micro_relative_diff_mean"]),
        "diversity_ratio": nested_get(details, ["diversity_ratio"]),
        "pairwise_l1_ratio": nested_get(details, ["pairwise_l1_ratio"]),
        "shift_texture_ratio": nested_get(details, ["shift_texture_ratio"]),
        "critic_real_score_raw": nested_get(details, ["critic_real_score_raw"]),
        "critic_fake_score_raw": nested_get(details, ["critic_fake_score_raw"]),
        "w_dist_raw": nested_get(details, ["w_dist_raw"]),
        "g_adv_loss_raw": nested_get(details, ["g_adv_loss_raw"]),
        "curve_health_penalty": nested_get(details, ["curve_health_penalty"]),
        "topology_tail_error_mean": nested_get(details, ["topology_tail_error_mean"]),
        "topology_tail_error_max": nested_get(details, ["topology_tail_error_max"]),
        "topology_tail_error_std": nested_get(details, ["topology_tail_error_std"]),
        "training_phase": nested_get(train_stats, ["training_phase"]),
    }
    if isinstance(generated_micro, dict):
        for key, value in generated_micro.items():
            row[f"gen_{key}"] = value
    if isinstance(real_micro, dict):
        for key, value in real_micro.items():
            row[f"real_{key}"] = value
    if isinstance(abs_diff, dict):
        for key, value in abs_diff.items():
            row[f"absdiff_{key}"] = value
    return row


def write_candidate_tables(
    all_candidate_records: Sequence[Dict[str, object]],
    top_records: Sequence[Dict[str, object]],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    model_best_epoch: int,
    model_best_score: float,
) -> None:
    template_rows = load_paper_template_rows(args.paper_template_csv)
    records_with_candidate_dirs = [record for record in all_candidate_records if record.get("top_candidate_dir")]
    for record in records_with_candidate_dirs:
        top_dir_raw = record.get("top_candidate_dir")
        if not top_dir_raw:
            continue
        top_dir = Path(str(top_dir_raw))
        paper_rows = build_candidate_paper_rows(record, args, template_rows, model_best_epoch, model_best_score)
        write_dict_rows_csv(top_dir / "candidate_paper_comparison_row.csv", paper_rows, PAPER_COMPARISON_COLUMNS)

    summary_rows = [flatten_candidate_record(record, model_best_epoch, model_best_score) for record in all_candidate_records]
    if summary_rows:
        base_fields = [
            "epoch",
            "is_model_selected_best",
            "model_selected_best_epoch",
            "model_selected_best_score",
            "selection_score",
            "best_model_eligible",
            "closure_pass",
            "target_pass",
            "raw_wgan_pass",
            "hard_micro_pass",
            "topology_train_pass",
            "fid_feature_no_pretrained",
            "mmd_rbf",
            "ssim_nearest",
            "psnr_nearest",
            "ms_ssim_nearest",
            "binary_mask_ssim_nearest",
            "micro_relative_diff_mean",
            "diversity_ratio",
            "pairwise_l1_ratio",
            "shift_texture_ratio",
            "critic_real_score_raw",
            "critic_fake_score_raw",
            "w_dist_raw",
            "g_adv_loss_raw",
            "curve_health_penalty",
            "topology_tail_error_mean",
            "topology_tail_error_max",
            "topology_tail_error_std",
            "training_phase",
        ]
        dynamic_fields = sorted({key for row in summary_rows for key in row.keys() if key not in base_fields})
        write_dict_rows_csv(dirs["logs"] / "candidate_metrics_summary_table.csv", summary_rows, [*base_fields, *dynamic_fields])
        top_epoch_set = {int(record.get("epoch", -1)) for record in top_records}
        top_summary_rows = [row for row in summary_rows if int(row.get("epoch", -2) or -2) in top_epoch_set]
        write_dict_rows_csv(dirs["logs"] / "top_candidate_metrics_summary_table.csv", top_summary_rows, [*base_fields, *dynamic_fields])
    write_dict_rows_csv(
        dirs["logs"] / "model_selected_best_epoch.csv",
        [
            {
                "model": MODEL_NAME,
                "model_selected_best_epoch": model_best_epoch if model_best_epoch else "",
                "model_selected_best_score": model_best_score if math.isfinite(model_best_score) else "",
                "selection_rule": "eligible top candidate first, otherwise lowest candidate_selection_score among kept top_records",
            }
        ],
        ["model", "model_selected_best_epoch", "model_selected_best_score", "selection_rule"],
    )
    write_three_column_candidate_tables(
        all_candidate_records=all_candidate_records,
        top_records=top_records,
        dirs=dirs,
        args=args,
        model_best_epoch=model_best_epoch,
        model_best_score=model_best_score,
    )


PAPER_THREE_COLUMN_COLUMNS = ["comparison_criterion", "reference_or_target", "description", "Measured", "Baseline GAN", "DME-GAN", "note"]
PAPER_THREE_COLUMN_STACKED_COLUMNS = ["epoch", *PAPER_THREE_COLUMN_COLUMNS]
PER_IMAGE_SIMILARITY_KEYS = [
    "nearest_ssim",
    "nearest_psnr",
    "nearest_binary_mask_ssim",
    "nearest_ms_ssim",
]
PER_IMAGE_STAT_KEYS = [*GAN_TARGET_RANGES.keys(), *PER_IMAGE_SIMILARITY_KEYS]
PER_IMAGE_LONG_COLUMNS = ["population", "image_index", "image_file", *PER_IMAGE_STAT_KEYS]
PER_IMAGE_PAPER_COLUMNS = ["comparison_criterion", "reference_or_target", "description", "image_index", "Measured", "Baseline GAN", "DME-GAN", "note"]


def load_json_dict(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def load_paper_template_rows_native(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        normalized.append(
            {
                "comparison_criterion": str(row.get("comparison_criterion", "")),
                "reference_or_target": str(row.get("reference_or_target", "")),
                "description": str(row.get("description", "")),
                "Measured": str(row.get("measured_statistics", row.get("Measured", ""))),
                "Baseline GAN": str(row.get("GAN", row.get("Baseline GAN", ""))),
                "DME-GAN": str(row.get("WGAN", row.get("DME-GAN", ""))),
                "note": str(row.get("selected_result_or_note", row.get("note", ""))),
            }
        )
    return normalized


def default_paper_template_rows() -> List[Dict[str, str]]:
    dimensions = [
        "Measured image count",
        "Image dimensions",
        "Augmentation presentations per image",
        "Generated evaluation count",
        "Training epochs / stopping epoch",
        "Parameter count (million)",
        "Training time (h)",
        "Generation time per image (s)",
        "Peak allocated GPU memory (GiB)",
        "Generation peak allocated GPU memory (MiB)",
        "FID-like",
        "MMD-RBF",
        "SSIM-nearest",
        "MS-SSIM-nearest",
        "binary_mask_ssim_nearest",
        "PSNR-nearest",
        "aggregate_morphology_error",
        "Inter-image variation ratio",
        "GAN pairwise L1 ratio",
        "GAN d_real",
        "GAN d_fake",
        *GAN_TARGET_RANGES.keys(),
        "Composite selection score",
        "Candidate epoch",
        "Checkpoint selected by the recorded criterion",
    ]
    return [{"comparison_criterion": item, "reference_or_target": "", "description": "", "Measured": "", "Baseline GAN": "", "DME-GAN": "", "note": ""} for item in dimensions]


def ensure_extra_paper_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out = [dict(row) for row in rows] if rows else default_paper_template_rows()
    existing = {row.get("comparison_criterion", "") for row in out}
    for dimension in ("MS-SSIM-nearest", "binary_mask_ssim_nearest", "Candidate epoch", "Checkpoint selected by the recorded criterion"):
        if dimension not in existing:
            out.append({"comparison_criterion": dimension, "reference_or_target": "", "description": "", "Measured": "", "Baseline GAN": "", "DME-GAN": "", "note": ""})
    return out


def standard_gan_summary(args: argparse.Namespace) -> Dict[str, object]:
    return load_json_dict(args.standard_gan_summary_json)


def model_summary_config(summary: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    config = summary.get("config", {})
    return config if isinstance(config, dict) else vars(args)


def model_metrics_value(
    summary_or_record: Dict[str, object],
    args: argparse.Namespace,
    dimension: str,
    model_best_epoch: int = 0,
    model_best_score: float = float("nan"),
) -> object:
    metrics = summary_or_record.get("metrics", {})
    details = summary_or_record.get("final_closure_checks", summary_or_record.get("selection_details", {}))
    train_stats = summary_or_record.get("final_training_stats", summary_or_record.get("train_stats", {}))
    config = model_summary_config(summary_or_record, args)
    generated_micro = nested_get(metrics, ["microstructure_generated_mean"], {})

    fixed_values = {
        "Measured image count": nested_get(summary_or_record, ["data_signature", "count"], summary_or_record.get("num_real_images", "")),
        "Image dimensions": f"{config.get('image_size', args.image_size)}x{config.get('image_size', args.image_size)}",
        "Augmentation presentations per image": config.get("augment_repeats", args.augment_repeats),
        "Generated evaluation count": summary_or_record.get("num_generated_images", config.get("num_generate", args.num_generate)),
        "Training epochs / stopping epoch": summary_or_record.get("best_candidate_epoch", summary_or_record.get("epoch", "")),
        "Parameter count (million)": float(summary_or_record.get("parameter_count", 0.0)) / 1e6 if summary_or_record.get("parameter_count") else "",
        "Training time (h)": float(summary_or_record.get("training_time_seconds", 0.0)) / 3600.0 if summary_or_record.get("training_time_seconds") else "",
        "Generation time per image (s)": summary_or_record.get("single_image_generation_time_seconds", ""),
        "Peak allocated GPU memory (GiB)": float(summary_or_record.get("peak_vram_mb", 0.0)) / 1024.0 if summary_or_record.get("peak_vram_mb") else "",
        "Generation peak allocated GPU memory (MiB)": summary_or_record.get("generation_peak_vram_mb", summary_or_record.get("peak_vram_mb", nested_get(train_stats, ["peak_vram_mb"]))),
        "FID-like": nested_get(metrics, ["fid_feature_no_pretrained"]),
        "MMD-RBF": nested_get(metrics, ["mmd_rbf"]),
        "SSIM-nearest": nested_get(metrics, ["ssim_nearest"]),
        "MS-SSIM-nearest": nested_get(metrics, ["ms_ssim_nearest"]),
        "binary_mask_ssim_nearest": nested_get(metrics, ["binary_mask_ssim_nearest"]),
        "PSNR-nearest": nested_get(metrics, ["psnr_nearest"]),
        "aggregate_morphology_error": nested_get(details, ["micro_relative_diff_mean"]),
        "Inter-image variation ratio": nested_get(details, ["diversity_ratio"]),
        "GAN pairwise L1 ratio": nested_get(details, ["pairwise_l1_ratio"]),
        "GAN d_real": nested_get(details, ["critic_real_score_raw"], nested_get(details, ["d_real"], nested_get(train_stats, ["critic_real_score_raw"]))),
        "GAN d_fake": nested_get(details, ["critic_fake_score_raw"], nested_get(details, ["d_fake"], nested_get(train_stats, ["critic_fake_score_raw"]))),
        "Composite selection score": summary_or_record.get("final_selection_score", summary_or_record.get("selection_score", "")),
        "Candidate epoch": summary_or_record.get("epoch", ""),
        "Checkpoint selected by the recorded criterion": model_best_epoch if model_best_epoch else summary_or_record.get("best_candidate_epoch", ""),
    }
    if dimension in fixed_values:
        value = fixed_values[dimension]
        if dimension == "Checkpoint selected by the recorded criterion" and model_best_epoch and math.isfinite(model_best_score):
            return f"{model_best_epoch} (score={csv_text(model_best_score)})"
        return value
    if isinstance(generated_micro, dict) and dimension in generated_micro:
        return generated_micro[dimension]
    return ""


def original_paper_value(metrics: Dict[str, object], args: argparse.Namespace, dimension: str, num_real: int) -> object:
    real_micro = nested_get(metrics, ["microstructure_real_mean"], {})
    fixed_values = {
        "Measured image count": num_real,
        "Image dimensions": f"{args.image_size}x{args.image_size}",
        "Augmentation presentations per image": "Unaugmented",
        "Generated evaluation count": "-",
        "aggregate_morphology_error": 0,
    }
    if dimension in fixed_values:
        return fixed_values[dimension]
    if isinstance(real_micro, dict) and dimension in real_micro:
        return real_micro[dimension]
    return ""


def build_three_column_paper_rows(
    record_or_summary: Dict[str, object],
    args: argparse.Namespace,
    standard_summary: Dict[str, object],
    template_rows: Sequence[Dict[str, str]],
    model_best_epoch: int = 0,
    model_best_score: float = float("nan"),
) -> List[Dict[str, object]]:
    rows = ensure_extra_paper_rows(template_rows)
    metrics = record_or_summary.get("metrics", {})
    num_real = int(record_or_summary.get("num_real_images", nested_get(record_or_summary, ["data_signature", "count"], 0)) or 0)
    out: List[Dict[str, object]] = []
    for row in rows:
        dimension = row.get("comparison_criterion", "")
        out.append(
            {
                "comparison_criterion": dimension,
                "reference_or_target": row.get("reference_or_target", ""),
                "description": row.get("description", ""),
                "Measured": original_paper_value(metrics, args, dimension, num_real),
                "Baseline GAN": model_metrics_value(standard_summary, args, dimension),
                "DME-GAN": model_metrics_value(record_or_summary, args, dimension, model_best_epoch, model_best_score),
                "note": row.get("note", ""),
            }
        )
    return out


def write_three_column_candidate_tables(
    all_candidate_records: Sequence[Dict[str, object]],
    top_records: Sequence[Dict[str, object]],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    model_best_epoch: int,
    model_best_score: float,
) -> None:
    template_rows = load_paper_template_rows_native(args.paper_template_csv)
    gan_summary = standard_gan_summary(args)
    stacked_rows: List[Dict[str, object]] = []
    for record in all_candidate_records:
        rows = build_three_column_paper_rows(record, args, gan_summary, template_rows, model_best_epoch, model_best_score)
        epoch = int(record.get("epoch", 0) or 0)
        for row in rows:
            stacked_rows.append({"epoch": epoch, **row})
        top_dir_raw = record.get("top_candidate_dir")
        if top_dir_raw:
            write_dict_rows_csv(Path(str(top_dir_raw)) / "candidate_paper_three_column_comparison.csv", rows, PAPER_THREE_COLUMN_COLUMNS)

    if stacked_rows:
        write_dict_rows_csv(dirs["logs"] / "candidate_paper_three_column_records.csv", stacked_rows, PAPER_THREE_COLUMN_STACKED_COLUMNS)

    selected = next(
        (
            item
            for item in top_records
            if isinstance(item.get("selection_details"), dict)
            and bool(item["selection_details"].get("best_model_eligible", False))  # type: ignore[index, union-attr]
        ),
        None,
    )
    if selected is None and top_records:
        selected = top_records[0]
    if selected is not None:
        rows = build_three_column_paper_rows(selected, args, gan_summary, template_rows, model_best_epoch, model_best_score)
        write_dict_rows_csv(dirs["logs"] / "paper_three_column_model_selected_best.csv", rows, PAPER_THREE_COLUMN_COLUMNS)


def generated_image_sort_key(path: Path) -> Tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 0, path.name)


def image_file_to_array(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    carbon = np.clip(arr[..., 0], 0.0, 255.0).astype(np.uint8)
    gray = Image.fromarray(carbon, mode="L")
    if gray.size != (image_size, image_size):
        gray = gray.resize((image_size, image_size), pil_resample_filter())
    return (np.asarray(gray, dtype=np.float32) / 255.0).astype(np.float32)


def load_generated_arrays_from_dir(directory: Path, image_size: int, limit: int) -> Tuple[np.ndarray, List[str]]:
    if not directory.exists():
        return np.empty((0, image_size, image_size), dtype=np.float32), []
    paths = sorted(directory.glob("*.png"), key=generated_image_sort_key)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        return np.empty((0, image_size, image_size), dtype=np.float32), []
    arrays = np.stack([image_file_to_array(path, image_size) for path in paths], axis=0)
    return arrays.astype(np.float32), [str(path) for path in paths]


def resize_stack_for_similarity(arrays: np.ndarray, size: int, batch_size: int = 32) -> np.ndarray:
    if arrays.size == 0:
        return np.empty((0, size, size), dtype=np.float32)
    if arrays.shape[-1] == size and arrays.shape[-2] == size:
        return arrays.astype(np.float32)
    out = []
    for start in range(0, arrays.shape[0], batch_size):
        tensor = arrays_to_eval_tensor(arrays[start : start + batch_size])
        resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        out.append(resized.numpy()[:, 0].astype(np.float32))
    return np.concatenate(out, axis=0)


def per_image_nearest_similarity(real_arrays: np.ndarray, query_arrays: np.ndarray, exclude_same_index: bool = False) -> List[Dict[str, float]]:
    if real_arrays.size == 0 or query_arrays.size == 0:
        return []
    real_masks = [(img > max(otsu_threshold(img), 0.03)).astype(np.float64) for img in real_arrays]
    query_masks = [(img > max(otsu_threshold(img), 0.03)).astype(np.float64) for img in query_arrays]
    scaled_real = {size: resize_stack_for_similarity(real_arrays, size) for size in (512, 256, 128)}
    scaled_query = {size: resize_stack_for_similarity(query_arrays, size) for size in (512, 256, 128)}
    rows = []
    for query_index, img in enumerate(query_arrays):
        candidates = []
        binary_candidates = []
        psnr_candidates = []
        for real_index, real in enumerate(real_arrays):
            if exclude_same_index and query_index == real_index and len(query_arrays) == len(real_arrays):
                continue
            candidates.append(ssim_pair(real, img))
            psnr_candidates.append(psnr_pair(real, img))
            binary_candidates.append(ssim_pair(real_masks[real_index], query_masks[query_index]))
        ms_scores = []
        for size in (512, 256, 128):
            scale_candidates = []
            for real_index, real_scaled in enumerate(scaled_real[size]):
                if exclude_same_index and query_index == real_index and len(query_arrays) == len(real_arrays):
                    continue
                scale_candidates.append(ssim_pair(real_scaled, scaled_query[size][query_index]))
            if scale_candidates:
                ms_scores.append(max(scale_candidates))
        rows.append(
            {
                "nearest_ssim": float(max(candidates)) if candidates else "",
                "nearest_psnr": float(max(psnr_candidates)) if psnr_candidates else "",
                "nearest_binary_mask_ssim": float(max(binary_candidates)) if binary_candidates else "",
                "nearest_ms_ssim": float(np.mean(ms_scores)) if ms_scores else "",
            }
        )
    return rows


def per_image_metric_rows(
    source: str,
    arrays: np.ndarray,
    image_paths: Sequence[str],
    real_arrays: np.ndarray,
    exclude_same_index: bool = False,
) -> List[Dict[str, object]]:
    if arrays.size == 0:
        return []
    similarity_rows = per_image_nearest_similarity(real_arrays, arrays, exclude_same_index=exclude_same_index)
    rows: List[Dict[str, object]] = []
    for index, img in enumerate(arrays):
        row: Dict[str, object] = {
            "population": source,
            "image_index": index + 1,
            "image_file": image_paths[index] if index < len(image_paths) else "",
        }
        row.update(microstructure_stats_single(img))
        if index < len(similarity_rows):
            row.update(similarity_rows[index])
        rows.append(row)
    return rows


def target_text_for_metric(metric: str) -> str:
    if metric in GAN_TARGET_RANGES:
        low, high = GAN_TARGET_RANGES[metric]
        return f"{low}-{high}"
    if metric == "nearest_ssim":
        return f">={csv_text(0.0)}; Higher values are favorable for this criterion"
    if metric == "nearest_ms_ssim":
        return "Higher values are favorable for this criterion"
    if metric == "nearest_binary_mask_ssim":
        return "Higher values are favorable for this criterion"
    if metric == "nearest_psnr":
        return "Higher values are favorable for this criterion"
    return ""


def write_final_evaluation_tables(
    summary: Dict[str, object],
    dirs: Dict[str, Path],
    args: argparse.Namespace,
    real_arrays: np.ndarray,
    real_paths: Sequence[Path],
    wgan_arrays: np.ndarray,
    wgan_image_paths: Sequence[str],
) -> List[str]:
    saved: List[str] = []
    template_rows = load_paper_template_rows_native(args.paper_template_csv)
    gan_summary = standard_gan_summary(args)
    paper_rows = build_three_column_paper_rows(
        summary,
        args,
        gan_summary,
        template_rows,
        int(summary.get("best_candidate_epoch", 0) or 0),
        float(summary.get("best_selection_score", float("nan")) or float("nan")),
    )
    final_paper_path = dirs["logs"] / "paper_three_column_final500_comparison.csv"
    write_dict_rows_csv(final_paper_path, paper_rows, PAPER_THREE_COLUMN_COLUMNS)
    saved.append(str(final_paper_path))

    gan_arrays, gan_paths = load_generated_arrays_from_dir(args.standard_gan_generated_dir, args.image_size, args.num_generate)
    real_rows = per_image_metric_rows("Measured", real_arrays, [str(path) for path in real_paths], real_arrays, exclude_same_index=True)
    gan_rows = per_image_metric_rows("Baseline GAN", gan_arrays, gan_paths, real_arrays)
    wgan_rows = per_image_metric_rows("DME-GAN", wgan_arrays, wgan_image_paths, real_arrays)
    long_rows = [*real_rows, *gan_rows, *wgan_rows]
    long_path = dirs["logs"] / "per_image_metrics_long_for_violin.csv"
    write_dict_rows_csv(long_path, long_rows, PER_IMAGE_LONG_COLUMNS)
    saved.append(str(long_path))

    by_source = {
        "Measured": real_rows,
        "Baseline GAN": gan_rows,
        "DME-GAN": wgan_rows,
    }
    max_count = max((len(rows) for rows in by_source.values()), default=0)
    paper_image_rows: List[Dict[str, object]] = []
    for metric in PER_IMAGE_STAT_KEYS:
        for image_index in range(1, max_count + 1):
            row = {
                "comparison_criterion": metric,
                "reference_or_target": target_text_for_metric(metric),
                "description": "Per-image statistics; generated populations contain 500 fields; unfilled measured entries have no corresponding observation",
                "image_index": image_index,
                "Measured": "",
                "Baseline GAN": "",
                "DME-GAN": "",
                "note": "Source values for violin plots",
            }
            for source, rows in by_source.items():
                if image_index <= len(rows):
                    row[source] = rows[image_index - 1].get(metric, "")
            paper_image_rows.append(row)
    paper_image_path = dirs["logs"] / "per_image_metrics_paper_three_columns.csv"
    write_dict_rows_csv(paper_image_path, paper_image_rows, PER_IMAGE_PAPER_COLUMNS)
    saved.append(str(paper_image_path))
    return saved


def parse_epoch_set(text: str) -> set[int]:
    if not text.strip():
        return set()
    return {int(item.strip()) for item in text.split(",") if item.strip()}


@torch.no_grad()
def generate_gan(
    generator: Generator,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
    sample_batch_size: int,
    truncation: float,
) -> torch.Tensor:
    generator.eval()
    outputs = []
    for start in range(0, num_samples, sample_batch_size):
        batch_count = min(sample_batch_size, num_samples - start)
        z = sample_latent(batch_count, latent_dim, device, truncation=truncation)
        outputs.append(gan_to_image_range(generator(z)).detach().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def generate_gan_from_noise(
    generator: Generator,
    noise: torch.Tensor,
    device: torch.device,
    sample_batch_size: int,
) -> torch.Tensor:
    generator.eval()
    outputs = []
    noise = noise.to(device)
    for start in range(0, noise.size(0), sample_batch_size):
        z = noise[start : start + sample_batch_size]
        outputs.append(gan_to_image_range(generator(z)).detach().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def evaluate_candidate_epoch(
    generator: Generator,
    epoch: int,
    args: argparse.Namespace,
    dirs: Dict[str, Path],
    device: torch.device,
    candidate_noise: torch.Tensor,
    real_arrays: np.ndarray,
    real_eval_tensor: torch.Tensor,
    reference_sorted: torch.Tensor | None,
    train_stats: Dict[str, float],
) -> Tuple[Dict[str, object], torch.Tensor]:
    generator.eval()
    generated_batches = []
    for start in range(0, candidate_noise.size(0), args.sample_batch_size):
        noise = candidate_noise[start : start + args.sample_batch_size]
        generated_batches.append(gan_to_image_range(generator(noise)).detach().cpu())
    samples = torch.cat(generated_batches, dim=0)
    samples = samples[: int(args.candidate_samples)].float().clamp(0.0, 1.0)
    gen_arrays = samples.squeeze(1).numpy().astype(np.float32)
    metrics = compute_metrics(real_arrays, gen_arrays)
    score, score_details = candidate_selection_score(metrics, samples, real_eval_tensor, train_stats, args)
    record: Dict[str, object] = {
        "epoch": epoch,
        "selection_score": score,
        "selection_details": score_details,
        "train_stats": dict(train_stats),
        "metrics": metrics,
        "num_real_images": int(real_arrays.shape[0]),
        "num_generated_images": int(samples.size(0)),
    }
    save_preview_grid(samples[: max(1, int(args.preview_grid_samples))], dirs["candidate_previews"] / f"candidate_epoch_{epoch:04d}.png")
    save_json(dirs["candidate_metrics"] / f"candidate_epoch_{epoch:04d}.json", record)
    return record, samples


def plot_supplemental_training_results(history: Dict[str, object], save_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{MODEL_NAME}] matplotlib not available; supplemental training plots were skipped.")
        return []

    save_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "success": "#18A999",
        "warning": "#F18F01",
        "danger": "#C73E1D",
        "info": "#6C5B7B",
        "text": "#2D3142",
    }

    def arr(key: str) -> np.ndarray:
        return np.asarray(history.get(key, []), dtype=np.float64)

    def sampled(values: np.ndarray, max_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        if values.size == 0:
            return np.asarray([]), np.asarray([])
        step = max(1, values.size // max_points)
        idx = np.arange(0, values.size, step)
        return idx, values[::step]

    def sampled_aligned(values: np.ndarray, total_progress: int, max_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        if values.size == 0:
            return np.asarray([]), np.asarray([])
        step = max(1, values.size // max_points)
        pick = np.arange(0, values.size, step)
        if values.size == 1:
            idx = np.zeros_like(pick, dtype=np.float64)
        else:
            idx = np.linspace(0.0, float(max(total_progress - 1, 0)), values.size)[pick]
        return idx, values[pick]

    def moving_average(values: np.ndarray, max_windows: int = 350) -> np.ndarray:
        if values.size < 3:
            return values
        window = max(1, values.size // max_windows)
        if window <= 1:
            return values
        return np.convolve(values, np.ones(window) / window, mode="valid")

    saved: List[str] = []
    g_losses = arr("G_losses")
    g_adv_losses = arr("G_adv_losses")
    g_plot_losses = g_adv_losses if g_adv_losses.size else g_losses
    d_losses = arr("D_losses")
    d_total_losses = arr("D_total_losses")
    w_dists = arr("Wasserstein_distances")
    real_scores = arr("D_real_scores")
    fake_scores = arr("D_fake_scores")
    gradient_penalties = arr("Gradient_penalties")
    critic_drift_losses = arr("Critic_drift_losses")
    topology_losses = arr("Topology_losses")
    weights = arr("Constraint_weights")
    density = arr("Topology_density")
    uniformity = arr("Topology_uniformity")
    aggregation = arr("Topology_aggregation")
    density_errors = arr("Topology_density_errors")
    uniformity_errors = arr("Topology_uniformity_errors")
    aggregation_errors = arr("Topology_aggregation_errors")
    candidate_epochs = arr("Candidate_epochs")
    aligned_loss_progress = int(max(g_plot_losses.size, d_losses.size, g_losses.size, 1))

    fig, ax = plt.subplots(figsize=(12, 8))
    if g_plot_losses.size:
        idx, vals = sampled_aligned(g_plot_losses, aligned_loss_progress)
        ax.plot(idx, vals, label="Generator Adv Loss", alpha=0.9, linewidth=1.8, color=colors["primary"])
    if g_losses.size and g_adv_losses.size:
        idx, vals = sampled_aligned(g_losses, aligned_loss_progress)
        ax.plot(idx, vals, label="Generator Total Loss", alpha=0.35, linewidth=1.2, linestyle="--", color=colors["info"])
    if d_losses.size:
        idx, vals = sampled_aligned(d_losses, aligned_loss_progress)
        ax.plot(idx, vals, label="Critic Loss", alpha=0.85, linewidth=1.8, color=colors["secondary"])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.set_title("Generator and Critic Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = save_dir / "loss_curves.png"
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()
    saved.append(str(loss_path))

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("WGAN-GP Training Curves Analysis", fontsize=20, color=colors["text"], fontweight="bold")

    ax1 = plt.subplot(3, 3, 1)
    if g_plot_losses.size:
        idx, vals = sampled_aligned(g_plot_losses, aligned_loss_progress)
        ax1.plot(idx, vals, label="Generator Adv Loss", alpha=0.8, linewidth=1.5, color=colors["primary"])
    if d_losses.size:
        idx, vals = sampled_aligned(d_losses, aligned_loss_progress)
        ax1.plot(idx, vals, label="Critic Loss", alpha=0.8, linewidth=1.5, color=colors["secondary"])
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    ax1.set_title("Generator and Critic Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    if w_dists.size:
        idx, vals = sampled(w_dists)
        ax2.plot(idx, vals, alpha=0.8, linewidth=1.5, color=colors["success"])
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Wasserstein Distance")
    ax2.set_title("Wasserstein Distance")
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    if real_scores.size:
        idx, vals = sampled(real_scores)
        ax3.plot(idx, vals, label="Raw Real Score", alpha=0.8, linewidth=1.5, color=colors["info"])
    if fake_scores.size:
        idx, vals = sampled(fake_scores)
        ax3.plot(idx, vals, label="Raw Fake Score", alpha=0.8, linewidth=1.5, color=colors["warning"])
    ax3.set_xlabel("Iterations")
    ax3.set_ylabel("Raw Critic Score")
    ax3.set_title("Raw Critic Scores")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    metric_specs = [
        (density, float(history.get("target_density", 0.0)), "Density", colors["primary"], 4),
        (uniformity, float(history.get("target_uniformity", 0.0)), "Uniformity", colors["info"], 5),
        (aggregation, float(history.get("target_aggregation", 0.0)), "Aggregation", colors["success"], 6),
    ]
    for values, target, title, color, plot_idx in metric_specs:
        ax = plt.subplot(3, 3, plot_idx)
        if values.size:
            idx, vals = sampled(values)
            ax.plot(idx, vals, alpha=0.8, linewidth=1.5, color=color)
            ax.axhline(y=target, color="r", linestyle="--", linewidth=2, label=f"Target: {target:.3f}")
            ax.legend()
        ax.set_xlabel("Generator Updates")
        ax.set_ylabel(title)
        ax.set_title(f"Topology Proxy {title}")
        ax.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 7)
    if topology_losses.size:
        idx, vals = sampled(topology_losses)
        ax7.plot(idx, vals, alpha=0.8, linewidth=1.5, color=colors["danger"])
    ax7.set_xlabel("Generator Updates")
    ax7.set_ylabel("Loss")
    ax7.set_title("Topology Constraint Loss")
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    if weights.size:
        ax8.plot(np.arange(weights.size), weights, alpha=0.8, linewidth=2, color=colors["warning"])
    ax8.set_xlabel("Epoch")
    ax8.set_ylabel("Weight")
    ax8.set_title("Constraint Warmup Weight")
    ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")
    summary_text = (
        "Training Summary\n\n"
        f"Total generator updates: {g_losses.size:,}\n"
        f"Total critic updates: {d_losses.size:,}\n"
        f"Best candidate epoch: {history.get('best_epoch', 'N/A')}\n"
        f"Constraint start epoch: {history.get('constraint_start_epoch', 'N/A')}\n\n"
        f"Target density: {float(history.get('target_density', 0.0)):.4f}\n"
        f"Target uniformity: {float(history.get('target_uniformity', 0.0)):.4f}\n"
        f"Target aggregation: {float(history.get('target_aggregation', 0.0)):.4f}\n\n"
        f"Mean W distance: {float(np.mean(w_dists)) if w_dists.size else 0.0:.4f}\n"
        f"Mean topology loss: {float(np.mean(topology_losses)) if topology_losses.size else 0.0:.4f}"
    )
    ax9.text(0.05, 0.95, summary_text, fontsize=11, verticalalignment="top")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    main_path = save_dir / "training_curves_main.png"
    plt.savefig(main_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(main_path))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Training Metrics Distribution Analysis", fontsize=18, color=colors["text"], fontweight="bold")
    hist_specs = [
        (g_plot_losses, d_losses, "Loss Distribution", "Loss Value", 1),
        (w_dists, None, "Wasserstein Distance Distribution", "Wasserstein Distance", 2),
        (topology_losses, None, "Topology Loss Distribution", "Topology Loss", 3),
    ]
    for first, second, title, xlabel, plot_idx in hist_specs:
        ax = plt.subplot(2, 3, plot_idx)
        if first.size:
            sample = first[np.random.choice(first.size, min(5000, first.size), replace=False)]
            ax.hist(sample, bins=50, alpha=0.7, label=title.split()[0], color=colors["primary"])
        if second is not None and second.size:
            sample = second[np.random.choice(second.size, min(5000, second.size), replace=False)]
            ax.hist(sample, bins=50, alpha=0.7, label="Critic", color=colors["secondary"])
            ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    ax4 = plt.subplot(2, 3, 4)
    if density.size and uniformity.size:
        n = min(density.size, uniformity.size)
        den_s = moving_average(density[:n])
        uni_s = moving_average(uniformity[:n])
        n_s = min(den_s.size, uni_s.size)
        scatter = ax4.scatter(den_s[:n_s], uni_s[:n_s], c=np.arange(n_s), cmap="viridis", alpha=0.7, s=42)
        ax4.scatter([history.get("target_density", 0.0)], [history.get("target_uniformity", 0.0)], color="red", s=160, marker="*", label="Target")
        plt.colorbar(scatter, ax=ax4, label="Training Progress")
        ax4.legend()
    ax4.set_xlabel("Density")
    ax4.set_ylabel("Uniformity")
    ax4.set_title("Density vs Uniformity")
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    if density.size and aggregation.size:
        n = min(density.size, aggregation.size)
        den_s = moving_average(density[:n])
        agg_s = moving_average(aggregation[:n])
        n_s = min(den_s.size, agg_s.size)
        scatter = ax5.scatter(den_s[:n_s], agg_s[:n_s], c=np.arange(n_s), cmap="plasma", alpha=0.7, s=42)
        ax5.scatter([history.get("target_density", 0.0)], [history.get("target_aggregation", 0.0)], color="red", s=160, marker="*", label="Target")
        plt.colorbar(scatter, ax=ax5, label="Training Progress")
        ax5.legend()
    ax5.set_xlabel("Density")
    ax5.set_ylabel("Aggregation")
    ax5.set_title("Density vs Aggregation")
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    ax6.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment="top")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    dist_path = save_dir / "training_distributions.png"
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    saved.append(str(dist_path))

    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle("Raw WGAN-GP and Topology Diagnostics", fontsize=18, color=colors["text"], fontweight="bold")

    ax = axes[0, 0]
    if g_plot_losses.size:
        idx, vals = sampled_aligned(g_plot_losses, aligned_loss_progress)
        ax.plot(idx, vals, label="Raw Generator Adv Loss", color=colors["primary"], linewidth=1.5)
    if d_losses.size:
        idx, vals = sampled_aligned(d_losses, aligned_loss_progress)
        ax.plot(idx, vals, label="Raw Critic Loss", color=colors["secondary"], linewidth=1.5)
    ax.set_title("Raw G/D Loss")
    ax.set_xlabel("Updates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if real_scores.size:
        idx, vals = sampled(real_scores)
        ax.plot(idx, vals, label="critic_real_score_raw", color=colors["info"], linewidth=1.5)
    if fake_scores.size:
        idx, vals = sampled(fake_scores)
        ax.plot(idx, vals, label="critic_fake_score_raw", color=colors["warning"], linewidth=1.5)
    ax.set_title("Raw Critic Scores")
    ax.set_xlabel("Critic Updates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if w_dists.size:
        idx, vals = sampled(w_dists)
        ax.plot(idx, vals, label="w_dist_raw", color=colors["success"], linewidth=1.5)
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Raw Wasserstein Distance")
    ax.set_xlabel("Critic Updates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if gradient_penalties.size:
        idx, vals = sampled(gradient_penalties)
        ax.plot(idx, vals, label="gradient_penalty", color=colors["danger"], linewidth=1.5)
    if d_total_losses.size:
        idx, vals = sampled(d_total_losses)
        ax.plot(idx, vals, label="d_total_loss", color=colors["secondary"], linewidth=1.5)
    if critic_drift_losses.size:
        idx, vals = sampled(critic_drift_losses)
        ax.plot(idx, vals, label="critic_drift_loss", color=colors["warning"], linewidth=1.2)
    ax.set_title("Gradient Penalty and D Total Loss")
    ax.set_xlabel("Critic Updates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    for values, target, label, color in (
        (density, float(history.get("target_density", 0.0)), "density", colors["primary"]),
        (uniformity, float(history.get("target_uniformity", 0.0)), "uniformity", colors["info"]),
        (aggregation, float(history.get("target_aggregation", 0.0)), "aggregation", colors["success"]),
    ):
        if values.size:
            idx, vals = sampled(values)
            ax.plot(idx, vals, label=label, color=color, linewidth=1.3)
            ax.axhline(target, color=color, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Topology Metrics and Targets")
    ax.set_xlabel("Generator Updates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for values, label, color in (
        (density_errors, "density_error", colors["primary"]),
        (uniformity_errors, "uniformity_error", colors["info"]),
        (aggregation_errors, "aggregation_error", colors["success"]),
    ):
        if values.size:
            idx, vals = sampled(values)
            ax.plot(idx, vals, label=label, color=color, linewidth=1.3)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="5% target")
    ax.set_title("Topology Relative Errors")
    ax.set_xlabel("Generator Updates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    for key, label, color in (
        ("Candidate_component_count", "component_count", colors["primary"]),
        ("Candidate_mean_component_area", "mean_component_area", colors["secondary"]),
        ("Candidate_interface_density", "interface_density", colors["danger"]),
        ("Candidate_autocorr_x", "autocorr_x", colors["info"]),
        ("Candidate_autocorr_y", "autocorr_y", colors["success"]),
    ):
        values = arr(key)
        if candidate_epochs.size and values.size == candidate_epochs.size:
            ax.plot(candidate_epochs, values, marker="o", markersize=3, label=label, color=color, linewidth=1.2)
    ax.set_title("Candidate Structure Metrics")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    for key, label, color in (
        ("Candidate_fid", "FID", colors["danger"]),
        ("Candidate_mmd", "MMD", colors["secondary"]),
        ("Candidate_ssim", "SSIM", colors["primary"]),
        ("Candidate_ms_ssim", "MS-SSIM", colors["info"]),
        ("Candidate_binary_mask_ssim", "Binary Mask SSIM", colors["success"]),
    ):
        values = arr(key)
        if candidate_epochs.size and values.size == candidate_epochs.size:
            ax.plot(candidate_epochs, values, marker="o", markersize=3, label=label, linewidth=1.2)
    ax.set_title("Candidate Distribution and Similarity Metrics")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    diagnostics_path = save_dir / "training_diagnostics_raw_wgan.png"
    plt.savefig(diagnostics_path, dpi=180, bbox_inches="tight")
    plt.close()
    saved.append(str(diagnostics_path))

    print(f"[{MODEL_NAME}] supplemental training plots saved to {save_dir}")
    return saved


def save_training_history_tables(history: Dict[str, object], logs_dir: Path) -> Dict[str, str]:
    json_path = logs_dir / "training_history_full.json"
    csv_path = logs_dir / "training_history_full.csv"
    save_json(json_path, history)

    numeric_series: Dict[str, List[float]] = {}
    for key, value in history.items():
        if not isinstance(value, list):
            continue
        cleaned: List[float] = []
        convertible = True
        for item in value:
            try:
                cleaned.append(float(item))
            except (TypeError, ValueError):
                convertible = False
                break
        if convertible:
            numeric_series[key] = cleaned

    max_len = max((len(values) for values in numeric_series.values()), default=0)
    if max_len:
        fields = ["iteration", *sorted(numeric_series.keys())]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for idx in range(max_len):
                row: Dict[str, object] = {"iteration": idx}
                for key in fields[1:]:
                    values = numeric_series[key]
                    row[key] = values[idx] if idx < len(values) else ""
                writer.writerow(row)
    return {"json": str(json_path), "csv": str(csv_path)}


def apply_quick_overrides(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.epochs = min(args.epochs, 2)
    args.augment_repeats = min(args.augment_repeats, 2)
    args.batch_size = min(args.batch_size, 2)
    args.base_channels = min(args.base_channels, 16)
    args.sample_batch_size = min(args.sample_batch_size, 4)
    args.num_generate = min(args.num_generate, 4)
    args.preview_interval = 1
    args.preview_grid_samples = min(getattr(args, "preview_grid_samples", 20), 4)
    args.intermediate_save_samples = min(getattr(args, "intermediate_save_samples", 20), 4)
    args.early_stop_patience = 0
    args.candidate_eval_interval = 1
    args.candidate_start_epoch = 1
    args.candidate_samples = min(args.candidate_samples, 4)
    args.physics_feedback_min_epoch = 1
    args.physics_plateau_window = 3
    args.physics_ensemble_members = min(args.physics_ensemble_members, 3)
    args.physics_rff_features = min(args.physics_rff_features, 16)


def parse_args(argv=None) -> argparse.Namespace:
    here = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Train a WGAN-GP with unified data augmentation and evaluation.")
    parser.add_argument("--data-dir", type=Path, default=here / "data" / "characterization")
    parser.add_argument("--output-dir", type=Path, default=here / f"outputs_{Path(__file__).stem}")
    parser.add_argument("--paper-template-csv", type=Path, default=here / "results" / "optional_comparison_template.csv")
    parser.add_argument("--image-size", type=int, default=512)
    parser.set_defaults(paper_template_csv=here / "results" / "optional_comparison_template.csv")
    parser.add_argument(
        "--standard-gan-summary-json",
        type=Path,
        default=here / "results" / "optional_baseline_summary.json",
    )
    parser.add_argument(
        "--standard-gan-generated-dir",
        type=Path,
        default=here / "results" / "optional_baseline_images",
    )
    parser.add_argument("--num-generate", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr-g", type=float, default=1.25e-5)
    parser.add_argument("--lr-d", type=float, default=8.0e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--critic-weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--augment-repeats", type=int, default=24)
    parser.add_argument("--base-channels", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--feature-dropout", type=float, default=0.02)
    parser.add_argument("--latent-jitter-std", type=float, default=0.01)
    parser.add_argument("--latent-dropout", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--truncation", type=float, default=1.0)
    parser.add_argument("--critic-score-scale", type=float, default=115.0)
    parser.add_argument("--critic-patch-weight", type=float, default=0.48)
    parser.add_argument("--critic-grad-clip", type=float, default=1.0)
    parser.add_argument("--generator-grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-batch-size", type=int, default=20)
    parser.add_argument("--preview-interval", type=int, default=2)
    parser.add_argument("--d-update-interval", type=int, default=1)
    parser.add_argument("--g-updates-per-step", type=int, default=1)
    parser.add_argument("--lambda-gp", type=float, default=8.0)
    parser.add_argument("--gradient-penalty-interval", type=int, default=4)
    parser.add_argument("--lazy-gp-multiplier", type=float, default=1.0)
    parser.add_argument("--early-critic-epochs", type=int, default=0)
    parser.add_argument("--mid-critic-epochs", type=int, default=100)
    parser.add_argument("--critic-recovery-patience", type=int, default=2)
    parser.add_argument("--critic-overfit-patience", type=int, default=2)
    parser.add_argument("--min-healthy-w-dist", type=float, default=0.06)
    parser.add_argument("--max-healthy-w-dist", type=float, default=0.95)
    parser.add_argument("--max-abs-g-adv-loss", type=float, default=2.50)
    parser.add_argument("--recovery-lr-g-scale", type=float, default=0.90)
    parser.add_argument("--recovery-lr-d-scale", type=float, default=1.05)
    parser.add_argument("--catchup-lr-g-scale", type=float, default=1.02)
    parser.add_argument("--catchup-lr-d-scale", type=float, default=0.75)
    parser.add_argument("--critic-drift-weight", type=float, default=1.0e-2)
    parser.add_argument("--real-label-smooth-min", type=float, default=0.72)
    parser.add_argument("--real-label-smooth-max", type=float, default=0.88)
    parser.add_argument("--fake-label-smooth-max", type=float, default=0.0)
    parser.add_argument("--instance-noise-std", type=float, default=0.025)
    parser.add_argument("--instance-noise-decay-epochs", type=int, default=170)
    parser.add_argument("--d-warmup-epochs", type=int, default=4)
    parser.add_argument("--warmup-d-update-interval", type=int, default=1)
    parser.add_argument("--warmup-g-updates-per-step", type=int, default=1)
    parser.add_argument("--warmup-diversity-loss-scale", type=float, default=0.45)
    parser.add_argument("--strong-d-real-threshold", type=float, default=0.80)
    parser.add_argument("--strong-d-fake-threshold", type=float, default=0.12)
    parser.add_argument("--strong-d-update-interval", type=int, default=8)
    parser.add_argument("--strong-g-updates-per-step", type=int, default=4)
    parser.add_argument("--strong-diversity-loss-scale", type=float, default=2.20)
    parser.add_argument("--weak-d-real-threshold", type=float, default=0.52)
    parser.add_argument("--weak-d-fake-threshold", type=float, default=0.48)
    parser.add_argument("--weak-d-update-interval", type=int, default=1)
    parser.add_argument("--weak-g-updates-per-step", type=int, default=1)
    parser.add_argument("--weak-diversity-loss-scale", type=float, default=0.65)
    parser.add_argument("--late-stabilize-epoch", type=int, default=230)
    parser.add_argument("--late-d-update-interval", type=int, default=2)
    parser.add_argument("--late-g-updates-per-step", type=int, default=1)
    parser.add_argument("--diversity-loss-weight", type=float, default=0.45)
    parser.add_argument("--diversity-target-ratio", type=float, default=0.95)
    parser.add_argument("--pairwise-diversity-loss-weight", type=float, default=0.35)
    parser.add_argument("--pairwise-diversity-target-ratio", type=float, default=0.90)
    parser.add_argument("--mode-seeking-loss-weight", type=float, default=0.20)
    parser.add_argument("--mode-seeking-target-ratio", type=float, default=0.22)
    parser.add_argument("--periodicity-loss-weight", type=float, default=0.16)
    parser.add_argument("--gradient-match-loss-weight", type=float, default=0.48)
    parser.add_argument("--feature-matching-loss-weight", type=float, default=0.16)
    parser.add_argument("--distribution-loss-weight", type=float, default=0.18)
    parser.add_argument("--structure-ssim-loss-weight", type=float, default=0.50)
    parser.add_argument("--local-stats-loss-weight", type=float, default=0.14)
    parser.add_argument("--patch-texture-loss-weight", type=float, default=0.46)
    parser.add_argument("--binary-edge-loss-weight", type=float, default=0.42)
    parser.add_argument("--morphology-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--morphology-anchor-start-epoch", type=int, default=42)
    parser.add_argument("--morphology-anchor-warmup-epochs", type=int, default=78)
    parser.add_argument("--cptec-loss-weight", type=float, default=0.06)
    parser.add_argument("--cptec-start-epoch", type=int, default=155)
    parser.add_argument("--cptec-warmup-epochs", type=int, default=80)
    parser.add_argument("--patch-texture-size", type=int, default=96)
    parser.add_argument("--patch-texture-count", type=int, default=4)
    parser.add_argument("--g-score-stability-weight", type=float, default=0.005)
    parser.add_argument("--g-score-target-abs", type=float, default=1.50)
    parser.add_argument("--structure-loss-start-epoch", type=int, default=12)
    parser.add_argument("--structure-loss-warmup-epochs", type=int, default=58)
    parser.add_argument("--structure-loss-size", type=int, default=128)
    parser.add_argument("--micro-loss-weight", type=float, default=0.0)
    parser.add_argument("--topology-loss-weight", type=float, default=0.28)
    parser.add_argument("--topology-constraint-start-epoch", type=int, default=105)
    parser.add_argument("--topology-constraint-warmup-epochs", type=int, default=70)
    parser.add_argument("--topology-threshold", type=float, default=0.35)
    parser.add_argument("--topology-temperature", type=float, default=0.08)
    parser.add_argument("--topology-uniformity-block", type=int, default=32)
    parser.add_argument("--topology-aggregation-kernel", type=int, default=17)
    parser.add_argument("--topology-density-weight", type=float, default=1.90)
    parser.add_argument("--topology-uniformity-weight", type=float, default=0.90)
    parser.add_argument("--topology-aggregation-weight", type=float, default=1.80)
    parser.add_argument("--topology-density-threshold", type=float, default=0.024)
    parser.add_argument("--topology-uniformity-threshold", type=float, default=0.035)
    parser.add_argument("--topology-aggregation-threshold", type=float, default=0.024)
    parser.add_argument("--interface-loss-weight", type=float, default=0.22)
    parser.add_argument("--interface-density-threshold", type=float, default=0.025)
    parser.add_argument("--autocorr-loss-weight", type=float, default=0.14)
    parser.add_argument("--autocorr-threshold", type=float, default=0.035)
    parser.add_argument("--low-frequency-loss-weight", type=float, default=0.010)
    parser.add_argument("--low-frequency-kernel", type=int, default=64)
    parser.add_argument("--tv-loss-weight", type=float, default=5e-6)
    parser.add_argument("--aux-loss-clip", type=float, default=1.60)
    parser.add_argument("--topology-loss-clip", type=float, default=1.60)
    parser.add_argument("--candidate-eval-interval", type=int, default=10)
    parser.add_argument("--candidate-start-epoch", type=int, default=80)
    parser.add_argument("--candidate-end-epoch", type=int, default=300)
    parser.add_argument("--candidate-samples", type=int, default=500)
    parser.add_argument("--candidate-extra-epochs", type=parse_epoch_set, default=parse_epoch_set("80,90,100,110,120,130,140,160,180,200,210,220,230,240,250,260,270,280,285,290,295,300"))
    parser.add_argument("--keep-top-k", type=int, default=16)
    parser.add_argument("--min-generated-diversity-ratio", type=float, default=0.72)
    parser.add_argument("--min-pairwise-l1-ratio", type=float, default=0.70)
    parser.add_argument("--max-d-real-score", type=float, default=0.82)
    parser.add_argument("--min-d-fake-score", type=float, default=0.12)
    parser.add_argument("--target-fid-max", type=float, default=50.0)
    parser.add_argument("--target-mmd-max", type=float, default=0.22)
    parser.add_argument("--target-ssim-min", type=float, default=0.125)
    parser.add_argument("--target-ms-ssim-min", type=float, default=0.120)
    parser.add_argument("--target-binary-mask-ssim-min", type=float, default=0.058)
    parser.add_argument("--best-hard-micro-penalty-max", type=float, default=0.20)
    parser.add_argument("--best-balanced-binary-min", type=float, default=0.058)
    parser.add_argument("--best-balanced-micro-relative-max", type=float, default=0.125)
    parser.add_argument("--best-balanced-fid-max", type=float, default=55.0)
    parser.add_argument("--best-balanced-mmd-max", type=float, default=0.22)
    parser.add_argument("--best-min-autocorr", type=float, default=0.89)
    parser.add_argument("--best-component-area-max-penalty-max", type=float, default=0.60)
    parser.add_argument("--best-max-component-area-max", type=float, default=12500.0)
    parser.add_argument("--micro-relative-clip", type=float, default=2.0)
    parser.add_argument("--score-hard-target-weight", type=float, default=1600.0)
    parser.add_argument("--score-fid-weight", type=float, default=1.10)
    parser.add_argument("--score-mmd-weight", type=float, default=150.0)
    parser.add_argument("--score-ssim-weight", type=float, default=7800.0)
    parser.add_argument("--score-structural-similarity-weight", type=float, default=6200.0)
    parser.add_argument("--score-structural-bonus-weight", type=float, default=62000.0)
    parser.add_argument("--eligible-score-fid-weight", type=float, default=1.0)
    parser.add_argument("--eligible-score-mmd-weight", type=float, default=220.0)
    parser.add_argument("--eligible-score-ssim-soft-min", type=float, default=0.135)
    parser.add_argument("--eligible-score-ms-ssim-soft-min", type=float, default=0.130)
    parser.add_argument("--eligible-score-binary-soft-min", type=float, default=0.060)
    parser.add_argument("--eligible-score-ssim-penalty-weight", type=float, default=7000.0)
    parser.add_argument("--eligible-score-ms-ssim-penalty-weight", type=float, default=5600.0)
    parser.add_argument("--eligible-score-binary-penalty-weight", type=float, default=4200.0)
    parser.add_argument("--eligible-score-micro-weight", type=float, default=80.0)
    parser.add_argument("--eligible-score-curve-weight", type=float, default=60.0)
    parser.add_argument("--score-micro-weight", type=float, default=380.0)
    parser.add_argument("--score-diversity-weight", type=float, default=75.0)
    parser.add_argument("--score-mode-collapse-weight", type=float, default=160.0)
    parser.add_argument("--score-d-balance-weight", type=float, default=120.0)
    parser.add_argument("--score-curve-health-weight", type=float, default=120.0)
    parser.add_argument("--score-topology-train-weight", type=float, default=420.0)
    parser.add_argument("--score-periodicity-weight", type=float, default=160.0)
    parser.add_argument("--score-phase-envelope-weight", type=float, default=520.0)
    parser.add_argument("--phase-bright-block-margin", type=float, default=0.30)
    parser.add_argument("--phase-dark-block-margin", type=float, default=0.06)
    parser.add_argument("--phase-transition-block-margin", type=float, default=0.010)
    parser.add_argument("--max-phase-envelope-penalty", type=float, default=0.22)
    parser.add_argument("--target-topology-error-max", type=float, default=0.05)
    parser.add_argument("--target-topology-error-std-max", type=float, default=0.030)
    parser.add_argument("--topology-stability-window-updates", type=int, default=256)
    parser.add_argument("--topology-stability-window-epochs", type=int, default=28)
    parser.add_argument("--max-w-dist-tail-std", type=float, default=0.45)
    parser.add_argument("--max-g-adv-tail-std", type=float, default=0.85)
    parser.add_argument("--max-d-loss-tail-std", type=float, default=0.85)
    parser.add_argument("--max-score-gap-tail-std", type=float, default=0.45)
    parser.add_argument("--min-shift-texture-ratio", type=float, default=0.82)
    parser.add_argument("--ideal-d-real-low", type=float, default=0.55)
    parser.add_argument("--ideal-d-real-high", type=float, default=0.82)
    parser.add_argument("--ideal-d-fake-low", type=float, default=0.12)
    parser.add_argument("--ideal-d-fake-high", type=float, default=0.42)
    parser.add_argument("--ideal-d-gap-low", type=float, default=0.12)
    parser.add_argument("--ideal-d-gap-high", type=float, default=0.68)
    parser.add_argument("--ideal-g-adv-high", type=float, default=3.0)
    parser.add_argument("--max-curve-health-penalty", type=float, default=0.35)
    parser.add_argument("--target-stop-min-epoch", type=int, default=100)
    parser.add_argument("--no-stop-on-target-pass", dest="stop_on_target_pass", action="store_false")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-epochs", type=int, default=190)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--collapse-stop-min-epoch", type=int, default=190)
    parser.add_argument("--collapse-stop-patience", type=int, default=0)
    parser.add_argument("--collapse-w-dist-threshold", type=float, default=2.2)
    parser.add_argument("--collapse-curve-penalty-threshold", type=float, default=1.8)
    parser.add_argument("--preview-grid-samples", type=int, default=20)
    parser.add_argument("--intermediate-save-samples", type=int, default=20)
    parser.add_argument("--balance-target-d-real", type=float, default=0.75)
    parser.add_argument("--balance-target-d-fake", type=float, default=0.25)
    parser.add_argument("--balance-g-loss-weight", type=float, default=0.02)
    # Event-triggered COMSOL-surrogate feedback.  The original fixed epoch
    # schedules remain as realism priors; physics gain is controlled online.
    parser.add_argument("--no-physics-feedback", dest="physics_feedback", action="store_false")
    parser.set_defaults(physics_feedback=True)
    parser.add_argument(
        "--physics-feedback-database",
        type=Path,
        default=here / "data" / "agent" / "feedback_soft21.csv",
    )
    parser.add_argument("--physics-feedback-min-epoch", type=int, default=80)
    parser.add_argument("--physics-feedback-group-size", type=int, default=4,
                        help="Monte-Carlo volume quadrature samples; not a physical layer count.")
    parser.add_argument("--physics-feedback-initial-gain", type=float, default=0.08)
    parser.add_argument("--physics-feedback-max-gain", type=float, default=0.55)
    parser.add_argument("--physics-feedback-decay", type=float, default=0.985)
    parser.add_argument("--physics-plateau-window", type=int, default=12)
    parser.add_argument("--physics-plateau-min-delta", type=float, default=0.012)
    parser.add_argument("--physics-risk-target", type=float, default=0.20)
    parser.add_argument("--physics-controller-kp", type=float, default=1.20)
    parser.add_argument("--physics-controller-ki", type=float, default=0.08)
    parser.add_argument("--physics-integral-clip", type=float, default=2.0)
    parser.add_argument("--physics-uncertainty-trigger", type=float, default=0.30)
    parser.add_argument("--physics-uncertainty-attenuation", type=float, default=0.50)
    parser.add_argument("--physics-oracle-cooldown-epochs", type=int, default=15)
    parser.add_argument("--physics-realism-guard", type=float, default=0.60)
    parser.add_argument("--physics-wdist-guard", type=float, default=1.25)
    parser.add_argument("--physics-guard-attenuation", type=float, default=0.65)
    parser.add_argument("--physics-rollback-risk-fraction", type=float, default=0.15)
    parser.add_argument("--physics-target-jmax", type=float, default=135.0)
    parser.add_argument("--physics-target-cum-delta-t", type=float, default=0.85)
    parser.add_argument("--physics-ucb-beta", type=float, default=0.75)
    parser.add_argument("--physics-cvar-alpha", type=float, default=0.90)
    parser.add_argument("--physics-ensemble-members", type=int, default=12)
    parser.add_argument("--physics-rff-features", type=int, default=96)
    parser.add_argument("--physics-rff-gamma", type=float, default=0.35)
    parser.add_argument("--physics-ridge", type=float, default=2.0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Short smoke run for syntax/runtime debugging.")
    args = parser.parse_args(argv)
    args.candidate_samples = max(int(args.candidate_samples), int(args.num_generate))
    apply_quick_overrides(args)
    return args


def train(args: argparse.Namespace) -> Dict[str, object]:
    seed_everything(args.seed)
    device = choose_device(args.device)
    dirs = ensure_dirs(args.output_dir)
    physics_feedback = EventTriggeredPhysicsFeedback.from_args(args, device)
    paths = sorted_raw_paths(args.data_dir)
    signature = dataset_signature(paths)

    train_dataset = CarbonEDSDataset(
        args.data_dir,
        image_size=args.image_size,
        augment=True,
        augment_repeats=args.augment_repeats,
    )
    real_dataset = CarbonEDSDataset(args.data_dir, image_size=args.image_size, augment=False)
    loader = make_loader(train_dataset, args.batch_size, shuffle=True, device=device)
    real_arrays = tensors_to_arrays(real_dataset.images)
    real_eval_tensor = arrays_to_eval_tensor(real_arrays)
    real_diversity_target = tensor_diversity(real_eval_tensor) * max(args.diversity_target_ratio, 0.0)
    real_pairwise_target = tensor_pairwise_l1(real_eval_tensor) * max(args.pairwise_diversity_target_ratio, 0.0)
    reference_sorted = build_reference_quantiles(real_eval_tensor)
    with torch.no_grad():
        topology_targets = {
            key: value.detach()
            for key, value in differentiable_topology_metrics(real_eval_tensor.to(device), args).items()
        }
    topology_target_summary = {
        key: float(value.detach().cpu())
        for key, value in topology_targets.items()
        if value.numel() == 1
    }

    generator = Generator(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        image_size=args.image_size,
        feature_dropout=args.feature_dropout,
    ).to(device)
    discriminator = Discriminator(
        base_channels=args.base_channels,
        image_size=args.image_size,
        score_scale=args.critic_score_scale,
        patch_weight=args.critic_patch_weight,
    ).to(device)
    generator.apply(init_dcgan_weights)
    discriminator.apply(init_dcgan_weights)

    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=args.lr_g,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(args.beta1, args.beta2),
        weight_decay=args.critic_weight_decay,
    )

    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)
    total_params = gen_params + disc_params
    print(
        f"[{MODEL_NAME}] device={device} generator_params={gen_params:,} "
        f"discriminator_params={disc_params:,} images={len(real_dataset)}"
    )
    print(
        f"[{MODEL_NAME}] WGAN-GP lambda_gp={args.lambda_gp:.2f} "
        f"gp_interval={max(args.gradient_penalty_interval, 1)} "
        f"topology_targets={topology_target_summary} "
        f"constraint_start={args.topology_constraint_start_epoch}"
    )

    log_path = dirs["logs"] / "training_log.csv"
    fieldnames = [
        "epoch",
        "d_loss",
        "g_loss",
        "g_adv_loss",
        "g_diversity_loss",
        "g_pairwise_diversity_loss",
        "g_mode_seeking_loss",
        "g_periodicity_loss",
        "g_gradient_match_loss",
        "g_feature_match_loss",
        "g_distribution_loss",
        "g_structure_ssim_loss",
        "g_local_stats_loss",
        "g_patch_texture_loss",
        "g_binary_edge_loss",
        "g_morphology_anchor_loss",
        "g_cptec_loss",
        "g_score_stability_loss",
        "g_micro_loss",
        "g_adv_loss_raw",
        "d_loss_real",
        "d_loss_fake",
        "critic_real_score_raw",
        "critic_fake_score_raw",
        "w_dist_raw",
        "d_loss_raw",
        "d_total_loss",
        "gradient_penalty",
        "critic_drift_loss",
        "compact_real_score",
        "compact_fake_score",
        "balance_score",
        "best_balance_score",
        "best_balance_epoch",
        "critic_updates",
        "generator_updates",
        "n_critic",
        "n_gen",
        "training_phase",
        "critic_recovery_count",
        "critic_overfit_count",
        "excessive_g_count",
        "diversity_loss_scale",
        "topology_loss_scale",
        "instance_noise_std",
        "epoch_seconds",
        "lr_g",
        "lr_d",
        "peak_vram_mb",
        "candidate_score",
        "best_selection_score",
        "best_candidate_epoch",
        "target_pass",
        "closure_pass",
        "curve_health_penalty",
        "collapse_alert_count",
        "d_score_gap",
        "w_dist_tail_mean",
        "w_dist_tail_std",
        "g_adv_tail_std",
        "d_loss_tail_std",
        "score_gap_tail_mean",
        "score_gap_tail_std",
        "generated_diversity_ratio",
        "generated_pairwise_l1_ratio",
        "g_topology_loss",
        "topology_constraint_weight",
        "topology_density",
        "topology_uniformity",
        "topology_aggregation",
        "topology_density_error",
        "topology_uniformity_error",
        "topology_aggregation_error",
        "topology_interface_density_error",
        "topology_autocorr_x_error",
        "topology_autocorr_y_error",
        "topology_tail_error_mean",
        "topology_tail_error_max",
        "topology_tail_error_std",
        "morphology_anchor_weight",
        "cptec_constraint_weight",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    reset_peak_memory(device)
    train_start = time.perf_counter()
    preview_count = max(1, min(int(args.preview_grid_samples), int(args.num_generate)))
    preview_noise = sample_latent(preview_count, args.latent_dim, device, truncation=args.truncation)
    candidate_pool_count = int(args.candidate_samples)
    candidate_noise = sample_latent(candidate_pool_count, args.latent_dim, device, truncation=args.truncation)
    top_records: List[Dict[str, object]] = []
    all_candidate_records: List[Dict[str, object]] = []
    best_candidate_score = float("inf")
    best_candidate_epoch = 0
    last_stats = {
        "d_loss": 0.0,
        "g_loss": 0.0,
        "g_adv_loss": 0.0,
        "g_diversity_loss": 0.0,
        "g_pairwise_diversity_loss": 0.0,
        "g_mode_seeking_loss": 0.0,
        "g_periodicity_loss": 0.0,
        "g_gradient_match_loss": 0.0,
        "g_feature_match_loss": 0.0,
        "g_distribution_loss": 0.0,
        "g_structure_ssim_loss": 0.0,
        "g_local_stats_loss": 0.0,
        "g_patch_texture_loss": 0.0,
        "g_binary_edge_loss": 0.0,
        "g_morphology_anchor_loss": 0.0,
        "g_cptec_loss": 0.0,
        "g_score_stability_loss": 0.0,
        "g_micro_loss": 0.0,
        "g_adv_loss_raw": 0.0,
        "g_topology_loss": 0.0,
        "topology_constraint_weight": 0.0,
        "topology_density": 0.0,
        "topology_uniformity": 0.0,
        "topology_aggregation": 0.0,
        "topology_density_error": 0.0,
        "topology_uniformity_error": 0.0,
        "topology_aggregation_error": 0.0,
        "topology_interface_density_error": 0.0,
        "topology_autocorr_x_error": 0.0,
        "topology_autocorr_y_error": 0.0,
        "topology_tail_error_mean": 0.0,
        "topology_tail_error_max": 0.0,
        "topology_tail_error_std": 0.0,
        "morphology_anchor_weight": 0.0,
        "cptec_constraint_weight": 0.0,
        "d_loss_real": 0.0,
        "d_loss_fake": 0.0,
        "gradient_penalty": 0.0,
        "critic_drift_loss": 0.0,
        "w_dist_raw": 0.0,
        "d_loss_raw": 0.0,
        "d_total_loss": 0.0,
        "critic_real_score_raw": 0.0,
        "critic_fake_score_raw": 0.0,
        "compact_real_score": 0.0,
        "compact_fake_score": 0.0,
        "balance_score": float("inf"),
        "critic_updates": 0.0,
        "generator_updates": 0.0,
        "n_critic": 3.0,
        "n_gen": 1.0,
        "training_phase": "initial",
        "diversity_loss_scale": 1.0,
        "topology_loss_scale": 1.0,
        "instance_noise_std": 0.0,
        "critic_recovery_count": 0.0,
        "critic_overfit_count": 0.0,
        "excessive_g_count": 0.0,
    }
    best_balance_score = float("inf")
    best_balance_epoch = 0
    best_balance_checkpoint_path = dirs["checkpoints"] / "best_balance.pt"
    physics_safe_checkpoint_path = dirs["checkpoints"] / "physics_safe.pt"
    global_step = 0
    collapse_alert_count = 0
    training_history: Dict[str, object] = {
        "G_losses": [],
        "G_adv_losses": [],
        "G_periodicity_losses": [],
        "G_gradient_match_losses": [],
        "G_feature_match_losses": [],
        "G_distribution_losses": [],
        "G_structure_ssim_losses": [],
        "G_local_stats_losses": [],
        "G_patch_texture_losses": [],
        "G_binary_edge_losses": [],
        "G_morphology_anchor_losses": [],
        "G_cptec_losses": [],
        "G_score_stability_losses": [],
        "D_losses": [],
        "D_total_losses": [],
        "Wasserstein_distances": [],
        "D_real_scores": [],
        "D_fake_scores": [],
        "Gradient_penalties": [],
        "Critic_drift_losses": [],
        "Topology_losses": [],
        "Constraint_weights": [],
        "Morphology_anchor_weights": [],
        "CPTEC_weights": [],
        "Topology_density": [],
        "Topology_uniformity": [],
        "Topology_aggregation": [],
        "Topology_density_errors": [],
        "Topology_uniformity_errors": [],
        "Topology_aggregation_errors": [],
        "Topology_interface_density_errors": [],
        "Topology_autocorr_x_errors": [],
        "Topology_autocorr_y_errors": [],
        "Candidate_epochs": [],
        "Candidate_fid": [],
        "Candidate_mmd": [],
        "Candidate_ssim": [],
        "Candidate_ms_ssim": [],
        "Candidate_binary_mask_ssim": [],
        "Candidate_component_count": [],
        "Candidate_mean_component_area": [],
        "Candidate_interface_density": [],
        "Candidate_autocorr_x": [],
        "Candidate_autocorr_y": [],
        "target_density": topology_target_summary["density"],
        "target_uniformity": topology_target_summary["uniformity"],
        "target_aggregation": topology_target_summary["aggregation"],
        "constraint_start_epoch": args.topology_constraint_start_epoch,
    }
    epoch_topology_error_history: List[Tuple[float, float, float]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        if physics_feedback is not None:
            physics_feedback.begin_epoch(epoch, last_stats)
        accum = {
            "d_loss": 0.0,
            "g_loss": 0.0,
            "g_adv_loss": 0.0,
            "g_diversity_loss": 0.0,
            "g_pairwise_diversity_loss": 0.0,
            "g_mode_seeking_loss": 0.0,
            "g_periodicity_loss": 0.0,
            "g_gradient_match_loss": 0.0,
            "g_feature_match_loss": 0.0,
            "g_distribution_loss": 0.0,
            "g_structure_ssim_loss": 0.0,
            "g_local_stats_loss": 0.0,
            "g_patch_texture_loss": 0.0,
            "g_binary_edge_loss": 0.0,
            "g_morphology_anchor_loss": 0.0,
            "g_cptec_loss": 0.0,
            "g_score_stability_loss": 0.0,
            "g_micro_loss": 0.0,
            "g_adv_loss_raw": 0.0,
            "g_topology_loss": 0.0,
            "topology_density": 0.0,
            "topology_uniformity": 0.0,
            "topology_aggregation": 0.0,
            "topology_density_error": 0.0,
            "topology_uniformity_error": 0.0,
            "topology_aggregation_error": 0.0,
            "topology_interface_density_error": 0.0,
            "topology_autocorr_x_error": 0.0,
            "topology_autocorr_y_error": 0.0,
            "d_loss_real": 0.0,
            "d_loss_fake": 0.0,
            "gradient_penalty": 0.0,
            "critic_drift_loss": 0.0,
            "w_dist_raw": 0.0,
            "d_loss_raw": 0.0,
            "d_total_loss": 0.0,
            "critic_real_score_raw": 0.0,
            "critic_fake_score_raw": 0.0,
            "compact_real_score": 0.0,
            "compact_fake_score": 0.0,
        }
        critic_updates = 0
        generator_updates = 0
        epoch_noise_std = instance_noise_std_for_epoch(
            epoch,
            args.instance_noise_std,
            args.instance_noise_decay_epochs,
        )
        n_critic, n_gen, diversity_loss_scale, training_phase, topology_phase_scale = training_controls_for_epoch(
            epoch,
            last_stats,
            args,
        )
        if training_phase == "critic_soft_guard":
            for group in optimizer_g.param_groups:
                group["lr"] = args.lr_g * args.recovery_lr_g_scale
            for group in optimizer_d.param_groups:
                group["lr"] = args.lr_d * args.recovery_lr_d_scale
        elif training_phase == "generator_soft_relief":
            for group in optimizer_g.param_groups:
                group["lr"] = args.lr_g * args.catchup_lr_g_scale
            for group in optimizer_d.param_groups:
                group["lr"] = args.lr_d * args.catchup_lr_d_scale
        else:
            for group in optimizer_g.param_groups:
                group["lr"] = args.lr_g
            for group in optimizer_d.param_groups:
                group["lr"] = args.lr_d
        generator.train()
        discriminator.train()
        current_topology_weight = topology_phase_scale * topology_constraint_weight(
            epoch,
            args.topology_constraint_start_epoch,
            args.topology_constraint_warmup_epochs,
        )
        current_structure_weight = topology_constraint_weight(
            epoch,
            args.structure_loss_start_epoch,
            args.structure_loss_warmup_epochs,
        )
        current_morphology_weight = topology_constraint_weight(
            epoch,
            args.morphology_anchor_start_epoch,
            args.morphology_anchor_warmup_epochs,
        )
        current_cptec_weight = topology_constraint_weight(
            epoch,
            args.cptec_start_epoch,
            args.cptec_warmup_epochs,
        )

        for batch_index, batch in enumerate(loader, start=1):
            real_01 = batch.to(device, non_blocking=True)
            real = image_to_gan_range(real_01)
            batch_size = real.size(0)
            if training_phase == "late_physical_focus":
                d_interval = args.late_d_update_interval
            elif training_phase == "generator_soft_relief":
                d_interval = args.strong_d_update_interval
            elif training_phase == "critic_soft_guard":
                d_interval = args.weak_d_update_interval
            elif epoch <= args.d_warmup_epochs:
                d_interval = args.warmup_d_update_interval
            else:
                d_interval = args.d_update_interval
            should_update_critic = batch_index % max(int(d_interval), 1) == 0 or critic_updates == 0

            for _critic_iter in range(max(n_critic, 1) if should_update_critic else 0):
                set_requires_grad(discriminator, True)
                discriminator.train()
                z = regularize_latent_for_training(sample_latent(batch_size, args.latent_dim, device), args)
                fake = generator(z).detach()
                d_real = discriminator(add_instance_noise(real, epoch_noise_std))
                d_fake = discriminator(add_instance_noise(fake, epoch_noise_std))
                real_score = d_real.mean()
                fake_score = d_fake.mean()
                w_dist_raw_tensor = real_score - fake_score
                gp_interval = max(args.gradient_penalty_interval, 1)
                should_apply_gp = global_step % gp_interval == 0
                if should_apply_gp:
                    gradient_penalty = compute_gradient_penalty(discriminator, real, fake)
                else:
                    gradient_penalty = real.new_zeros(())
                d_loss_real = -real_score
                d_loss_fake = fake_score
                d_loss_raw = fake_score - real_score
                critic_drift_loss = (real_score.square() + fake_score.square()) * 0.5
                d_total_loss = (
                    d_loss_raw
                    + float(args.lambda_gp * args.lazy_gp_multiplier) * gradient_penalty
                    + float(args.critic_drift_weight) * critic_drift_loss
                )

                optimizer_d.zero_grad(set_to_none=True)
                if not torch.isfinite(d_total_loss):
                    optimizer_d.zero_grad(set_to_none=True)
                    continue
                d_total_loss.backward()
                if args.critic_grad_clip > 0.0:
                    grad_norm_d = nn.utils.clip_grad_norm_(discriminator.parameters(), args.critic_grad_clip)
                    if not torch.isfinite(grad_norm_d):
                        optimizer_d.zero_grad(set_to_none=True)
                        continue
                optimizer_d.step()

                d_loss_raw_value = float(d_loss_raw.detach().cpu())
                d_total_loss_value = float(d_total_loss.detach().cpu())
                gp_value = float(gradient_penalty.detach().cpu())
                drift_value = float(critic_drift_loss.detach().cpu())
                w_value = float(w_dist_raw_tensor.detach().cpu())
                critic_real_value = float(real_score.detach().cpu())
                critic_fake_value = float(fake_score.detach().cpu())
                compact_real_value = float(torch.sigmoid(d_real).mean().detach().cpu())
                compact_fake_value = float(torch.sigmoid(d_fake).mean().detach().cpu())
                accum["d_loss"] += d_total_loss_value
                accum["d_loss_real"] += float(d_loss_real.detach().cpu())
                accum["d_loss_fake"] += float(d_loss_fake.detach().cpu())
                accum["gradient_penalty"] += gp_value
                accum["critic_drift_loss"] += drift_value
                accum["w_dist_raw"] += w_value
                accum["d_loss_raw"] += d_loss_raw_value
                accum["d_total_loss"] += d_total_loss_value
                accum["critic_real_score_raw"] += critic_real_value
                accum["critic_fake_score_raw"] += critic_fake_value
                accum["compact_real_score"] += compact_real_value
                accum["compact_fake_score"] += compact_fake_value
                history_lists = training_history  # local alias for clearer append calls
                history_lists["D_losses"].append(d_loss_raw_value)  # type: ignore[index, union-attr]
                history_lists["D_total_losses"].append(d_total_loss_value)  # type: ignore[index, union-attr]
                history_lists["Wasserstein_distances"].append(w_value)  # type: ignore[index, union-attr]
                history_lists["Gradient_penalties"].append(gp_value)  # type: ignore[index, union-attr]
                history_lists["Critic_drift_losses"].append(drift_value)  # type: ignore[index, union-attr]
                history_lists["D_real_scores"].append(critic_real_value)  # type: ignore[index, union-attr]
                history_lists["D_fake_scores"].append(critic_fake_value)  # type: ignore[index, union-attr]
                critic_updates += 1
                global_step += 1

            set_requires_grad(discriminator, False)
            discriminator.train()
            for _gen_iter in range(max(n_gen, 1)):
                z = regularize_latent_for_training(sample_latent(batch_size, args.latent_dim, device), args)
                fake = generator(z)
                fake_01 = gan_to_image_range(fake)
                fake_score_for_g = discriminator(fake).mean()
                g_adv_loss = -fake_score_for_g
                g_score_stability_loss = generator_score_stability_loss(
                    fake_score_for_g,
                    target_abs=float(args.g_score_target_abs),
                )
                g_diversity_loss = batch_diversity_loss(fake_01, real_diversity_target)
                g_pairwise_diversity_loss = batch_pairwise_diversity_loss(fake_01, real_pairwise_target)
                g_mode_seeking_loss = mode_seeking_diversity_loss(
                    fake_01,
                    z,
                    args.mode_seeking_target_ratio,
                )
                g_periodicity_loss = anti_periodic_texture_loss(fake_01, real_01)
                g_gradient_match_loss = gradient_energy_match_loss(fake_01, real_01)
                g_feature_match_loss = critic_feature_matching_loss(discriminator, fake, real)
                g_distribution_loss = raw_output_distribution_loss(fake_01, real_01)
                g_structure_ssim_loss = differentiable_ssim_loss(
                    fake_01,
                    real_01,
                    size=int(args.structure_loss_size),
                )
                g_local_stats_loss = local_structure_stats_loss(fake_01, real_01)
                g_patch_texture_loss = patch_texture_stats_loss(
                    fake_01,
                    real_01,
                    patch_size=int(args.patch_texture_size),
                    num_patches=int(args.patch_texture_count),
                    threshold=float(args.topology_threshold),
                )
                g_binary_edge_loss = binary_edge_distribution_loss(
                    fake_01,
                    real_01,
                    threshold=float(args.topology_threshold),
                    temperature=float(args.topology_temperature),
                )
                g_morphology_anchor_loss = carbon_morphology_anchor_loss(
                    fake_01,
                    real_01,
                    threshold=float(args.topology_threshold),
                    temperature=float(args.topology_temperature),
                )
                g_cptec_loss = carbon_phase_transition_envelope_loss(
                    fake_01,
                    real_01,
                    threshold=float(args.topology_threshold),
                )
                g_topology_loss, topology_metrics, topology_errors = differentiable_topology_proxy_loss(
                    fake_01,
                    topology_targets,
                    args,
                )
                if physics_feedback is not None:
                    g_physics_feedback_loss, _physics_diagnostics = physics_feedback.loss(fake_01)
                else:
                    g_physics_feedback_loss = fake_01.new_zeros(())
                g_micro_loss = g_topology_loss
                aux_clip = max(float(args.aux_loss_clip), 0.0)
                topology_clip = max(float(args.topology_loss_clip), 0.0)
                g_diversity_term = torch.clamp(g_diversity_loss, max=aux_clip) if aux_clip > 0.0 else g_diversity_loss
                g_pairwise_term = (
                    torch.clamp(g_pairwise_diversity_loss, max=aux_clip) if aux_clip > 0.0 else g_pairwise_diversity_loss
                )
                g_mode_term = torch.clamp(g_mode_seeking_loss, max=aux_clip) if aux_clip > 0.0 else g_mode_seeking_loss
                g_periodicity_term = torch.clamp(g_periodicity_loss, max=aux_clip) if aux_clip > 0.0 else g_periodicity_loss
                g_gradient_term = torch.clamp(g_gradient_match_loss, max=aux_clip) if aux_clip > 0.0 else g_gradient_match_loss
                g_feature_term = torch.clamp(g_feature_match_loss, max=aux_clip) if aux_clip > 0.0 else g_feature_match_loss
                g_distribution_term = (
                    torch.clamp(g_distribution_loss, max=aux_clip) if aux_clip > 0.0 else g_distribution_loss
                )
                g_structure_term = torch.clamp(g_structure_ssim_loss, max=aux_clip) if aux_clip > 0.0 else g_structure_ssim_loss
                g_local_stats_term = torch.clamp(g_local_stats_loss, max=aux_clip) if aux_clip > 0.0 else g_local_stats_loss
                g_patch_texture_term = (
                    torch.clamp(g_patch_texture_loss, max=aux_clip) if aux_clip > 0.0 else g_patch_texture_loss
                )
                g_binary_edge_term = torch.clamp(g_binary_edge_loss, max=aux_clip) if aux_clip > 0.0 else g_binary_edge_loss
                g_morphology_anchor_term = (
                    torch.clamp(g_morphology_anchor_loss, max=aux_clip)
                    if aux_clip > 0.0
                    else g_morphology_anchor_loss
                )
                g_cptec_term = torch.clamp(g_cptec_loss, max=aux_clip) if aux_clip > 0.0 else g_cptec_loss
                g_score_stability_term = (
                    torch.clamp(g_score_stability_loss, max=aux_clip) if aux_clip > 0.0 else g_score_stability_loss
                )
                g_topology_term = torch.clamp(g_topology_loss, max=topology_clip) if topology_clip > 0.0 else g_topology_loss
                g_loss = (
                    g_adv_loss
                    + args.g_score_stability_weight * g_score_stability_term
                    + diversity_loss_scale * args.diversity_loss_weight * g_diversity_term
                    + diversity_loss_scale * args.pairwise_diversity_loss_weight * g_pairwise_term
                    + diversity_loss_scale * args.mode_seeking_loss_weight * g_mode_term
                    + args.periodicity_loss_weight * g_periodicity_term
                    + args.gradient_match_loss_weight * g_gradient_term
                    + current_structure_weight * args.feature_matching_loss_weight * g_feature_term
                    + current_structure_weight * args.distribution_loss_weight * g_distribution_term
                    + current_structure_weight * args.structure_ssim_loss_weight * g_structure_term
                    + current_structure_weight * args.local_stats_loss_weight * g_local_stats_term
                    + current_structure_weight * args.patch_texture_loss_weight * g_patch_texture_term
                    + current_structure_weight * args.binary_edge_loss_weight * g_binary_edge_term
                    + current_morphology_weight * args.morphology_anchor_loss_weight * g_morphology_anchor_term
                    + current_cptec_weight * args.cptec_loss_weight * g_cptec_term
                    + current_topology_weight * args.topology_loss_weight * g_topology_term
                    + args.micro_loss_weight * g_topology_term
                    + g_physics_feedback_loss
                )

                optimizer_g.zero_grad(set_to_none=True)
                if not torch.isfinite(g_loss):
                    optimizer_g.zero_grad(set_to_none=True)
                    continue
                g_loss.backward()
                if args.generator_grad_clip > 0.0:
                    grad_norm_g = nn.utils.clip_grad_norm_(generator.parameters(), args.generator_grad_clip)
                    if not torch.isfinite(grad_norm_g):
                        optimizer_g.zero_grad(set_to_none=True)
                        continue
                optimizer_g.step()

                g_loss_value = float(g_loss.detach().cpu())
                g_adv_loss_value = float(g_adv_loss.detach().cpu())
                g_topology_loss_value = float(g_topology_loss.detach().cpu())
                topology_density_value = float(topology_metrics["density"].detach().cpu())
                topology_uniformity_value = float(topology_metrics["uniformity"].detach().cpu())
                topology_aggregation_value = float(topology_metrics["aggregation"].detach().cpu())
                topology_density_error = float(topology_errors["density"].detach().cpu())
                topology_uniformity_error = float(topology_errors["uniformity"].detach().cpu())
                topology_aggregation_error = float(topology_errors["aggregation"].detach().cpu())
                topology_interface_error = float(topology_errors["interface_density"].detach().cpu())
                topology_autocorr_x_error = float(topology_errors["autocorr_x"].detach().cpu())
                topology_autocorr_y_error = float(topology_errors["autocorr_y"].detach().cpu())
                accum["g_loss"] += g_loss_value
                accum["g_adv_loss"] += g_adv_loss_value
                accum["g_adv_loss_raw"] += g_adv_loss_value
                accum["g_diversity_loss"] += float(g_diversity_loss.detach().cpu())
                accum["g_pairwise_diversity_loss"] += float(g_pairwise_diversity_loss.detach().cpu())
                accum["g_mode_seeking_loss"] += float(g_mode_seeking_loss.detach().cpu())
                accum["g_periodicity_loss"] += float(g_periodicity_loss.detach().cpu())
                accum["g_gradient_match_loss"] += float(g_gradient_match_loss.detach().cpu())
                accum["g_feature_match_loss"] += float(g_feature_match_loss.detach().cpu())
                accum["g_distribution_loss"] += float(g_distribution_loss.detach().cpu())
                accum["g_structure_ssim_loss"] += float(g_structure_ssim_loss.detach().cpu())
                accum["g_local_stats_loss"] += float(g_local_stats_loss.detach().cpu())
                accum["g_patch_texture_loss"] += float(g_patch_texture_loss.detach().cpu())
                accum["g_binary_edge_loss"] += float(g_binary_edge_loss.detach().cpu())
                accum["g_morphology_anchor_loss"] += float(g_morphology_anchor_loss.detach().cpu())
                accum["g_cptec_loss"] += float(g_cptec_loss.detach().cpu())
                accum["g_score_stability_loss"] += float(g_score_stability_loss.detach().cpu())
                accum["g_micro_loss"] += g_topology_loss_value
                accum["g_topology_loss"] += g_topology_loss_value
                accum["topology_density"] += topology_density_value
                accum["topology_uniformity"] += topology_uniformity_value
                accum["topology_aggregation"] += topology_aggregation_value
                accum["topology_density_error"] += topology_density_error
                accum["topology_uniformity_error"] += topology_uniformity_error
                accum["topology_aggregation_error"] += topology_aggregation_error
                accum["topology_interface_density_error"] += topology_interface_error
                accum["topology_autocorr_x_error"] += topology_autocorr_x_error
                accum["topology_autocorr_y_error"] += topology_autocorr_y_error
                training_history["G_losses"].append(g_loss_value)  # type: ignore[index, union-attr]
                training_history["G_adv_losses"].append(g_adv_loss_value)  # type: ignore[index, union-attr]
                training_history["G_periodicity_losses"].append(float(g_periodicity_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history["G_gradient_match_losses"].append(float(g_gradient_match_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history["G_feature_match_losses"].append(float(g_feature_match_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history["G_distribution_losses"].append(float(g_distribution_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history["G_structure_ssim_losses"].append(float(g_structure_ssim_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history["G_local_stats_losses"].append(float(g_local_stats_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history["G_patch_texture_losses"].append(float(g_patch_texture_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history.setdefault("G_binary_edge_losses", []).append(float(g_binary_edge_loss.detach().cpu()))  # type: ignore[union-attr]
                training_history.setdefault("G_morphology_anchor_losses", []).append(float(g_morphology_anchor_loss.detach().cpu()))  # type: ignore[union-attr]
                training_history.setdefault("G_cptec_losses", []).append(float(g_cptec_loss.detach().cpu()))  # type: ignore[union-attr]
                training_history["G_score_stability_losses"].append(float(g_score_stability_loss.detach().cpu()))  # type: ignore[index, union-attr]
                training_history["Topology_losses"].append(g_topology_loss_value)  # type: ignore[index, union-attr]
                training_history["Topology_density"].append(topology_density_value)  # type: ignore[index, union-attr]
                training_history["Topology_uniformity"].append(topology_uniformity_value)  # type: ignore[index, union-attr]
                training_history["Topology_aggregation"].append(topology_aggregation_value)  # type: ignore[index, union-attr]
                training_history["Topology_density_errors"].append(topology_density_error)  # type: ignore[index, union-attr]
                training_history["Topology_uniformity_errors"].append(topology_uniformity_error)  # type: ignore[index, union-attr]
                training_history["Topology_aggregation_errors"].append(topology_aggregation_error)  # type: ignore[index, union-attr]
                training_history["Topology_interface_density_errors"].append(topology_interface_error)  # type: ignore[index, union-attr]
                training_history["Topology_autocorr_x_errors"].append(topology_autocorr_x_error)  # type: ignore[index, union-attr]
                training_history["Topology_autocorr_y_errors"].append(topology_autocorr_y_error)  # type: ignore[index, union-attr]
                generator_updates += 1
            set_requires_grad(discriminator, True)
            discriminator.train()

        d_update_count = max(critic_updates, 1)
        g_update_count = max(generator_updates, 1)
        previous_recovery_count = float(last_stats.get("critic_recovery_count", 0.0))
        previous_overfit_count = float(last_stats.get("critic_overfit_count", 0.0))
        previous_excessive_g_count = float(last_stats.get("excessive_g_count", 0.0))
        last_stats = {
            "d_loss": accum["d_loss"] / d_update_count,
            "g_loss": accum["g_loss"] / g_update_count,
            "g_adv_loss": accum["g_adv_loss"] / g_update_count,
            "g_diversity_loss": accum["g_diversity_loss"] / g_update_count,
            "g_pairwise_diversity_loss": accum["g_pairwise_diversity_loss"] / g_update_count,
            "g_mode_seeking_loss": accum["g_mode_seeking_loss"] / g_update_count,
            "g_periodicity_loss": accum["g_periodicity_loss"] / g_update_count,
            "g_gradient_match_loss": accum["g_gradient_match_loss"] / g_update_count,
            "g_feature_match_loss": accum["g_feature_match_loss"] / g_update_count,
            "g_distribution_loss": accum["g_distribution_loss"] / g_update_count,
            "g_structure_ssim_loss": accum["g_structure_ssim_loss"] / g_update_count,
            "g_local_stats_loss": accum["g_local_stats_loss"] / g_update_count,
            "g_patch_texture_loss": accum["g_patch_texture_loss"] / g_update_count,
            "g_binary_edge_loss": accum["g_binary_edge_loss"] / g_update_count,
            "g_morphology_anchor_loss": accum["g_morphology_anchor_loss"] / g_update_count,
            "g_cptec_loss": accum["g_cptec_loss"] / g_update_count,
            "g_score_stability_loss": accum["g_score_stability_loss"] / g_update_count,
            "g_micro_loss": accum["g_micro_loss"] / g_update_count,
            "g_adv_loss_raw": accum["g_adv_loss_raw"] / g_update_count,
            "g_topology_loss": accum["g_topology_loss"] / g_update_count,
            "topology_constraint_weight": current_topology_weight,
            "topology_density": accum["topology_density"] / g_update_count,
            "topology_uniformity": accum["topology_uniformity"] / g_update_count,
            "topology_aggregation": accum["topology_aggregation"] / g_update_count,
            "topology_density_error": accum["topology_density_error"] / g_update_count,
            "topology_uniformity_error": accum["topology_uniformity_error"] / g_update_count,
            "topology_aggregation_error": accum["topology_aggregation_error"] / g_update_count,
            "topology_interface_density_error": accum["topology_interface_density_error"] / g_update_count,
            "topology_autocorr_x_error": accum["topology_autocorr_x_error"] / g_update_count,
            "topology_autocorr_y_error": accum["topology_autocorr_y_error"] / g_update_count,
            "d_loss_real": accum["d_loss_real"] / d_update_count,
            "d_loss_fake": accum["d_loss_fake"] / d_update_count,
            "gradient_penalty": accum["gradient_penalty"] / d_update_count,
            "critic_drift_loss": accum["critic_drift_loss"] / d_update_count,
            "w_dist_raw": accum["w_dist_raw"] / d_update_count,
            "d_loss_raw": accum["d_loss_raw"] / d_update_count,
            "d_total_loss": accum["d_total_loss"] / d_update_count,
            "critic_real_score_raw": accum["critic_real_score_raw"] / d_update_count,
            "critic_fake_score_raw": accum["critic_fake_score_raw"] / d_update_count,
            "compact_real_score": accum["compact_real_score"] / d_update_count,
            "compact_fake_score": accum["compact_fake_score"] / d_update_count,
            "balance_score": 0.0,
            "critic_updates": float(critic_updates),
            "generator_updates": float(generator_updates),
            "n_critic": float(n_critic),
            "n_gen": float(n_gen),
            "training_phase": training_phase,
            "diversity_loss_scale": float(diversity_loss_scale),
            "topology_loss_scale": float(topology_phase_scale),
            "instance_noise_std": epoch_noise_std,
            "morphology_anchor_weight": current_morphology_weight,
            "cptec_constraint_weight": current_cptec_weight,
        }
        for topo_key in ("density", "uniformity", "aggregation"):
            target_value = float(topology_target_summary.get(topo_key, 0.0))
            metric_value = float(last_stats[f"topology_{topo_key}"])
            denom = max(abs(target_value), 1e-6)
            last_stats[f"topology_{topo_key}_error"] = abs(metric_value - target_value) / denom
        last_stats["balance_score"] = training_balance_score(last_stats, args)
        last_stats["curve_health_penalty"] = gan_curve_health_penalty(last_stats, args)
        last_stats["d_score_gap"] = last_stats["critic_real_score_raw"] - last_stats["critic_fake_score_raw"]
        feedback_event: Dict[str, object] = {}
        if physics_feedback is not None:
            feedback_event = physics_feedback.end_epoch(epoch, last_stats, dirs["logs"])
            if bool(feedback_event.get("save_safe_checkpoint", False)):
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "optimizer_g": optimizer_g.state_dict(),
                        "epoch": epoch,
                        "physics_feedback": physics_feedback.state_dict_for_checkpoint(),
                    },
                    physics_safe_checkpoint_path,
                )
            elif bool(feedback_event.get("rollback_requested", False)) and physics_safe_checkpoint_path.exists():
                try:
                    safe = torch.load(physics_safe_checkpoint_path, map_location=device, weights_only=False)
                except TypeError:
                    safe = torch.load(physics_safe_checkpoint_path, map_location=device)
                generator.load_state_dict(safe["generator"])
                optimizer_g.load_state_dict(safe["optimizer_g"])
                print(
                    f"[{MODEL_NAME}] physics/realism rollback at epoch {epoch}; "
                    f"restored safe generator from epoch {safe['epoch']}."
                )
        epoch_topology_error_history.append(
            (
                float(last_stats["topology_density_error"]),
                float(last_stats["topology_uniformity_error"]),
                float(last_stats["topology_aggregation_error"]),
            )
        )
        stability_window = max(1, int(args.topology_stability_window_updates))
        epoch_stability_window = max(1, int(getattr(args, "topology_stability_window_epochs", 24)))
        topology_epoch_tail = np.asarray(epoch_topology_error_history[-epoch_stability_window:], dtype=np.float64)
        if topology_epoch_tail.size:
            last_stats["topology_tail_error_mean"] = float(np.mean(topology_epoch_tail))
            last_stats["topology_tail_error_max"] = float(np.max(topology_epoch_tail))
            last_stats["topology_tail_error_std"] = float(np.std(topology_epoch_tail))
        else:
            last_stats["topology_tail_error_mean"] = 0.0
            last_stats["topology_tail_error_max"] = 0.0
            last_stats["topology_tail_error_std"] = 0.0
        w_tail = np.asarray(training_history.get("Wasserstein_distances", []), dtype=np.float64)[-stability_window:]
        g_tail = np.asarray(training_history.get("G_adv_losses", []), dtype=np.float64)[-stability_window:]
        d_tail = np.asarray(training_history.get("D_losses", []), dtype=np.float64)[-stability_window:]
        real_tail = np.asarray(training_history.get("D_real_scores", []), dtype=np.float64)[-stability_window:]
        fake_tail = np.asarray(training_history.get("D_fake_scores", []), dtype=np.float64)[-stability_window:]
        gap_tail = real_tail[: min(real_tail.size, fake_tail.size)] - fake_tail[: min(real_tail.size, fake_tail.size)]
        last_stats["w_dist_tail_mean"] = float(np.mean(w_tail)) if w_tail.size else 0.0
        last_stats["w_dist_tail_std"] = float(np.std(w_tail)) if w_tail.size else 0.0
        last_stats["g_adv_tail_std"] = float(np.std(g_tail)) if g_tail.size else 0.0
        last_stats["d_loss_tail_std"] = float(np.std(d_tail)) if d_tail.size else 0.0
        last_stats["score_gap_tail_mean"] = float(np.mean(gap_tail)) if gap_tail.size else 0.0
        last_stats["score_gap_tail_std"] = float(np.std(gap_tail)) if gap_tail.size else 0.0
        critic_overfit = (
            last_stats["w_dist_raw"] > args.max_healthy_w_dist
            or last_stats["d_score_gap"] > args.max_healthy_w_dist
            or last_stats["d_loss_raw"] < -args.max_healthy_w_dist
        )
        if critic_overfit:
            last_stats["critic_overfit_count"] = previous_overfit_count + 1.0
        else:
            last_stats["critic_overfit_count"] = 0.0
        if (
            not critic_overfit
            and (last_stats["w_dist_raw"] < 0.0 or last_stats["critic_fake_score_raw"] >= last_stats["critic_real_score_raw"])
        ):
            last_stats["critic_recovery_count"] = previous_recovery_count + 1.0
        else:
            last_stats["critic_recovery_count"] = 0.0
        if last_stats["g_adv_loss_raw"] < -abs(args.max_abs_g_adv_loss):
            last_stats["excessive_g_count"] = previous_excessive_g_count + 1.0
        else:
            last_stats["excessive_g_count"] = 0.0
        training_history["Constraint_weights"].append(last_stats["topology_constraint_weight"])  # type: ignore[index, union-attr]
        training_history.setdefault("Morphology_anchor_weights", []).append(last_stats["morphology_anchor_weight"])  # type: ignore[union-attr]
        training_history.setdefault("CPTEC_weights", []).append(last_stats["cptec_constraint_weight"])  # type: ignore[union-attr]

        improved = last_stats["balance_score"] < best_balance_score - args.early_stop_min_delta
        if improved:
            best_balance_score = last_stats["balance_score"]
            best_balance_epoch = epoch
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "args": vars(args),
                    "stats": last_stats,
                    "epoch": epoch,
                    "balance_score": best_balance_score,
                },
                best_balance_checkpoint_path,
            )

        candidate_score: object = ""
        candidate_diversity_ratio: object = ""
        candidate_pairwise_l1_ratio: object = ""
        target_pass: object = ""
        closure_pass: object = ""
        curve_health_penalty: object = last_stats["curve_health_penalty"]
        should_evaluate_candidate = (
            epoch >= args.candidate_start_epoch
            and epoch <= args.candidate_end_epoch
            and (
                epoch % max(args.candidate_eval_interval, 1) == 0
                or epoch in args.candidate_extra_epochs
                or epoch == args.epochs
            )
        )
        if should_evaluate_candidate:
            record, candidate_samples = evaluate_candidate_epoch(
                generator=generator,
                epoch=epoch,
                args=args,
                dirs=dirs,
                device=device,
                candidate_noise=candidate_noise,
                real_arrays=real_arrays,
                real_eval_tensor=real_eval_tensor,
                reference_sorted=reference_sorted,
                train_stats=last_stats,
            )
            all_candidate_records.append(record)
            candidate_score = float(record["selection_score"])
            metrics_for_history = record.get("metrics", {})
            if isinstance(metrics_for_history, dict):
                micro_for_history = metrics_for_history.get("microstructure_generated_mean", {})
                training_history["Candidate_epochs"].append(epoch)  # type: ignore[index, union-attr]
                training_history["Candidate_fid"].append(float(metrics_for_history.get("fid_feature_no_pretrained", 0.0)))  # type: ignore[index, union-attr]
                training_history["Candidate_mmd"].append(float(metrics_for_history.get("mmd_rbf", 0.0)))  # type: ignore[index, union-attr]
                training_history["Candidate_ssim"].append(float(metrics_for_history.get("ssim_nearest", 0.0)))  # type: ignore[index, union-attr]
                training_history["Candidate_ms_ssim"].append(float(metrics_for_history.get("ms_ssim_nearest", 0.0)))  # type: ignore[index, union-attr]
                training_history["Candidate_binary_mask_ssim"].append(float(metrics_for_history.get("binary_mask_ssim_nearest", 0.0)))  # type: ignore[index, union-attr]
                if isinstance(micro_for_history, dict):
                    training_history["Candidate_component_count"].append(float(micro_for_history.get("component_count", 0.0)))  # type: ignore[index, union-attr]
                    training_history["Candidate_mean_component_area"].append(float(micro_for_history.get("mean_component_area", 0.0)))  # type: ignore[index, union-attr]
                    training_history["Candidate_interface_density"].append(float(micro_for_history.get("interface_density", 0.0)))  # type: ignore[index, union-attr]
                    training_history["Candidate_autocorr_x"].append(float(micro_for_history.get("autocorr_x", 0.0)))  # type: ignore[index, union-attr]
                    training_history["Candidate_autocorr_y"].append(float(micro_for_history.get("autocorr_y", 0.0)))  # type: ignore[index, union-attr]
            details = record.get("selection_details", {})
            if isinstance(details, dict):
                candidate_diversity_ratio = details.get("diversity_ratio", "")
                candidate_pairwise_l1_ratio = details.get("pairwise_l1_ratio", "")
                compliance = details.get("target_compliance", {})
                target_pass = compliance.get("target_pass", "") if isinstance(compliance, dict) else ""
                closure_pass = details.get("closure_pass", "")
                curve_health_penalty = details.get("curve_health_penalty", curve_health_penalty)

            if is_top_candidate(record, top_records, args.keep_top_k):
                top_dir = dirs["best_candidates"] / f"epoch_{epoch:04d}_score_{float(candidate_score):.6f}"
                image_dir = top_dir / "generated"
                checkpoint_path = top_dir / "checkpoint.pt"
                saved_candidate_images = save_generated_images(candidate_samples[: args.num_generate], image_dir)
                torch.save(
                    {
                        "generator": generator.state_dict(),
                        "discriminator": discriminator.state_dict(),
                        "args": vars(args),
                        "stats": last_stats,
                        "candidate_record": record,
                    },
                    checkpoint_path,
                )
                record["top_candidate_dir"] = str(top_dir)
                record["top_candidate_checkpoint"] = str(checkpoint_path)
                record["top_candidate_generated_images"] = saved_candidate_images
                save_json(top_dir / "candidate_summary.json", record)
                top_records = update_top_candidates(top_records, record, args.keep_top_k)

            top_records.sort(key=candidate_rank_key)
            eligible_record = next(
                (
                    item
                    for item in top_records
                    if isinstance(item.get("selection_details"), dict)
                    and bool(item["selection_details"].get("best_model_eligible", False))  # type: ignore[index, union-attr]
                ),
                None,
            )
            if eligible_record:
                best_candidate_score = float(eligible_record["selection_score"])
                best_candidate_epoch = int(eligible_record["epoch"])
            elif top_records:
                best_candidate_score = float(top_records[0]["selection_score"])
                best_candidate_epoch = int(top_records[0]["epoch"])
            else:
                best_candidate_score = float("inf")
                best_candidate_epoch = 0
            save_json(
                dirs["logs"] / "best_candidates.json",
                {
                    "best_epoch": best_candidate_epoch,
                    "best_selection_score": best_candidate_score,
                    "top_candidates": top_records,
                    "all_candidate_count": len(all_candidate_records),
                },
            )
            write_candidate_tables(
                all_candidate_records=all_candidate_records,
                top_records=top_records,
                dirs=dirs,
                args=args,
                model_best_epoch=best_candidate_epoch,
                model_best_score=best_candidate_score,
            )

        severe_instability = (
            epoch >= args.collapse_stop_min_epoch
            and best_candidate_epoch > 0
            and (
                abs(float(last_stats["w_dist_raw"])) >= args.collapse_w_dist_threshold
                or float(last_stats["curve_health_penalty"]) >= args.collapse_curve_penalty_threshold
                or float(last_stats["critic_fake_score_raw"] - last_stats["critic_real_score_raw"])
                >= args.collapse_w_dist_threshold
            )
        )
        if severe_instability:
            collapse_alert_count += 1
        else:
            collapse_alert_count = max(0, collapse_alert_count - 1)

        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "d_loss": last_stats["d_loss"],
            "g_loss": last_stats["g_loss"],
            "g_adv_loss": last_stats["g_adv_loss"],
            "g_diversity_loss": last_stats["g_diversity_loss"],
            "g_pairwise_diversity_loss": last_stats["g_pairwise_diversity_loss"],
            "g_mode_seeking_loss": last_stats["g_mode_seeking_loss"],
            "g_periodicity_loss": last_stats["g_periodicity_loss"],
            "g_gradient_match_loss": last_stats["g_gradient_match_loss"],
            "g_feature_match_loss": last_stats["g_feature_match_loss"],
            "g_distribution_loss": last_stats["g_distribution_loss"],
            "g_structure_ssim_loss": last_stats["g_structure_ssim_loss"],
            "g_local_stats_loss": last_stats["g_local_stats_loss"],
            "g_patch_texture_loss": last_stats["g_patch_texture_loss"],
            "g_binary_edge_loss": last_stats["g_binary_edge_loss"],
            "g_morphology_anchor_loss": last_stats["g_morphology_anchor_loss"],
            "g_cptec_loss": last_stats["g_cptec_loss"],
            "g_score_stability_loss": last_stats["g_score_stability_loss"],
            "g_micro_loss": last_stats["g_micro_loss"],
            "g_adv_loss_raw": last_stats["g_adv_loss_raw"],
            "g_topology_loss": last_stats["g_topology_loss"],
            "topology_constraint_weight": last_stats["topology_constraint_weight"],
            "topology_density": last_stats["topology_density"],
            "topology_uniformity": last_stats["topology_uniformity"],
            "topology_aggregation": last_stats["topology_aggregation"],
            "topology_density_error": last_stats["topology_density_error"],
            "topology_uniformity_error": last_stats["topology_uniformity_error"],
            "topology_aggregation_error": last_stats["topology_aggregation_error"],
            "topology_interface_density_error": last_stats["topology_interface_density_error"],
            "topology_autocorr_x_error": last_stats["topology_autocorr_x_error"],
            "topology_autocorr_y_error": last_stats["topology_autocorr_y_error"],
            "topology_tail_error_mean": last_stats["topology_tail_error_mean"],
            "topology_tail_error_max": last_stats["topology_tail_error_max"],
            "topology_tail_error_std": last_stats["topology_tail_error_std"],
            "morphology_anchor_weight": last_stats["morphology_anchor_weight"],
            "cptec_constraint_weight": last_stats["cptec_constraint_weight"],
            "d_loss_real": last_stats["d_loss_real"],
            "d_loss_fake": last_stats["d_loss_fake"],
            "gradient_penalty": last_stats["gradient_penalty"],
            "w_dist_raw": last_stats["w_dist_raw"],
            "d_loss_raw": last_stats["d_loss_raw"],
            "d_total_loss": last_stats["d_total_loss"],
            "critic_real_score_raw": last_stats["critic_real_score_raw"],
            "critic_fake_score_raw": last_stats["critic_fake_score_raw"],
            "critic_drift_loss": last_stats["critic_drift_loss"],
            "compact_real_score": last_stats["compact_real_score"],
            "compact_fake_score": last_stats["compact_fake_score"],
            "balance_score": last_stats["balance_score"],
            "best_balance_score": best_balance_score if math.isfinite(best_balance_score) else "",
            "best_balance_epoch": best_balance_epoch if best_balance_epoch else "",
            "critic_updates": last_stats["critic_updates"],
            "generator_updates": last_stats["generator_updates"],
            "n_critic": last_stats["n_critic"],
            "n_gen": last_stats["n_gen"],
            "training_phase": last_stats["training_phase"],
            "critic_recovery_count": last_stats["critic_recovery_count"],
            "critic_overfit_count": last_stats["critic_overfit_count"],
            "excessive_g_count": last_stats["excessive_g_count"],
            "diversity_loss_scale": last_stats["diversity_loss_scale"],
            "topology_loss_scale": last_stats["topology_loss_scale"],
            "instance_noise_std": last_stats["instance_noise_std"],
            "epoch_seconds": epoch_seconds,
            "lr_g": optimizer_g.param_groups[0]["lr"],
            "lr_d": optimizer_d.param_groups[0]["lr"],
            "peak_vram_mb": peak_memory_mb(device),
            "candidate_score": candidate_score,
            "best_selection_score": best_candidate_score if math.isfinite(best_candidate_score) else "",
            "best_candidate_epoch": best_candidate_epoch if best_candidate_epoch else "",
            "target_pass": target_pass,
            "closure_pass": closure_pass,
            "curve_health_penalty": curve_health_penalty,
            "collapse_alert_count": collapse_alert_count,
            "d_score_gap": last_stats["d_score_gap"],
            "w_dist_tail_mean": last_stats["w_dist_tail_mean"],
            "w_dist_tail_std": last_stats["w_dist_tail_std"],
            "g_adv_tail_std": last_stats["g_adv_tail_std"],
            "d_loss_tail_std": last_stats["d_loss_tail_std"],
            "score_gap_tail_mean": last_stats["score_gap_tail_mean"],
            "score_gap_tail_std": last_stats["score_gap_tail_std"],
            "generated_diversity_ratio": candidate_diversity_ratio,
            "generated_pairwise_l1_ratio": candidate_pairwise_l1_ratio,
        }
        write_csv_row(log_path, row, fieldnames)

        if epoch % args.preview_interval == 0 or epoch == args.epochs:
            with torch.no_grad():
                preview = gan_to_image_range(generator(preview_noise))
            preview = preview.detach().cpu().float().clamp(0.0, 1.0)
            save_preview_grid(preview, dirs["previews"] / f"preview_epoch_{epoch:04d}.png")
            save_generated_images(preview[: max(1, int(args.intermediate_save_samples))], dirs["intermediate_generated"] / f"epoch_{epoch:04d}")
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "args": vars(args),
                    "stats": last_stats,
                    "physics_feedback": physics_feedback.state_dict_for_checkpoint() if physics_feedback is not None else None,
                },
                dirs["checkpoints"] / "latest.pt",
            )

        if epoch % args.log_interval == 0:
            print(
                f"[{MODEL_NAME}] epoch {epoch:04d}/{args.epochs} "
                f"phase={last_stats['training_phase']} "
                f"d_raw={last_stats['d_loss_raw']:.4f} d_total={last_stats['d_total_loss']:.4f} "
                f"g={last_stats['g_loss']:.4f} g_adv={last_stats['g_adv_loss_raw']:.4f} "
                f"real_raw={last_stats['critic_real_score_raw']:.4f} "
                f"fake_raw={last_stats['critic_fake_score_raw']:.4f} "
                f"w={last_stats['w_dist_raw']:.4f} gp={last_stats['gradient_penalty']:.4f} "
                f"drift={last_stats['critic_drift_loss']:.4f} "
                f"g_div={last_stats['g_diversity_loss']:.4f} "
                f"g_l1={last_stats['g_pairwise_diversity_loss']:.4f} "
                f"g_ms={last_stats['g_mode_seeking_loss']:.4f} "
                f"g_fm={last_stats['g_feature_match_loss']:.4f} "
                f"g_dist={last_stats['g_distribution_loss']:.4f} "
                f"g_ssim={last_stats['g_structure_ssim_loss']:.4f} "
                f"g_loc={last_stats['g_local_stats_loss']:.4f} "
                f"g_patch={last_stats['g_patch_texture_loss']:.4f} "
                f"g_morph={last_stats['g_morphology_anchor_loss']:.4f}@{last_stats['morphology_anchor_weight']:.2f} "
                f"g_cptec={last_stats['g_cptec_loss']:.4f}@{last_stats['cptec_constraint_weight']:.2f} "
                f"g_score_stab={last_stats['g_score_stability_loss']:.4f} "
                f"g_topo={last_stats['g_topology_loss']:.4f}@{last_stats['topology_constraint_weight']:.2f} "
                f"topo_re={last_stats['topology_density_error']:.3f}/"
                f"{last_stats['topology_uniformity_error']:.3f}/"
                f"{last_stats['topology_aggregation_error']:.3f} "
                f"balance={last_stats['balance_score']:.4f} "
                f"best_candidate={best_candidate_epoch if best_candidate_epoch else '-'} "
                f"critic_up={int(last_stats['critic_updates'])} gen_up={int(last_stats['generator_updates'])} "
                f"ctrl=C{int(last_stats['n_critic'])}:G{int(last_stats['n_gen'])} "
                f"overfit={int(last_stats['critic_overfit_count'])} "
                f"curve={last_stats['curve_health_penalty']:.4f} "
                f"tail_std={last_stats['w_dist_tail_std']:.3f}/"
                f"{last_stats['g_adv_tail_std']:.3f}/"
                f"{last_stats['d_loss_tail_std']:.3f} "
                f"noise={last_stats['instance_noise_std']:.4f}"
                + (
                    f" physics={physics_feedback.last_event}@{physics_feedback.gain:.3f}"
                    if physics_feedback is not None else ""
                )
            )

        if (
            args.stop_on_target_pass
            and closure_pass is True
            and epoch >= args.target_stop_min_epoch
        ):
            print(
                f"[{MODEL_NAME}] target pass at epoch {epoch}; "
                f"FID/MMD/SSIM, diversity, and GAN curve checks reached the configured closure targets."
            )
            break

        if (
            args.collapse_stop_patience > 0
            and collapse_alert_count >= args.collapse_stop_patience
            and best_candidate_epoch > 0
        ):
            print(
                f"[{MODEL_NAME}] instability stop at epoch {epoch}; "
                f"late WGAN oscillation persisted for {collapse_alert_count} epochs, "
                f"so final export will use best candidate epoch {best_candidate_epoch}."
            )
            break

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and best_candidate_epoch > 0
            and epoch - best_candidate_epoch >= args.early_stop_patience
        ):
            print(
                f"[{MODEL_NAME}] early stop at epoch {epoch}; "
                f"best eligible candidate epoch {best_candidate_epoch} stayed unchanged for {args.early_stop_patience} epochs."
            )
            break

    training_time = time.perf_counter() - train_start
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "args": vars(args),
            "stats": last_stats,
            "physics_feedback": physics_feedback.state_dict_for_checkpoint() if physics_feedback is not None else None,
        },
        dirs["checkpoints"] / "final.pt",
    )

    if all_candidate_records:
        save_json(
            dirs["logs"] / "all_candidate_records.json",
            {"records": all_candidate_records},
        )
        write_candidate_tables(
            all_candidate_records=all_candidate_records,
            top_records=top_records,
            dirs=dirs,
            args=args,
            model_best_epoch=best_candidate_epoch,
            model_best_score=best_candidate_score,
        )

    selected_record = next(
        (
            item
            for item in top_records
            if isinstance(item.get("selection_details"), dict)
            and bool(item["selection_details"].get("best_model_eligible", False))  # type: ignore[index, union-attr]
        ),
        None,
    )
    selected_record_is_eligible = selected_record is not None
    if selected_record is None and top_records:
        selected_record = top_records[0]
    selected_model_source = "final_epoch"
    if selected_record and selected_record.get("top_candidate_checkpoint"):
        selected_checkpoint = Path(str(selected_record["top_candidate_checkpoint"]))
        try:
            checkpoint = torch.load(selected_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(selected_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        selected_model_source = str(selected_checkpoint)
        if not selected_record_is_eligible:
            print(
                f"[{MODEL_NAME}] no candidate satisfied all hard best gates; "
                f"using best available candidate checkpoint instead of final epoch: {selected_checkpoint}"
            )
    elif not top_records:
        print(
            f"[{MODEL_NAME}] no candidate checkpoints were evaluated; final epoch weights will be used for final image export."
        )

    gen_start = time.perf_counter()
    final_pool_count = int(args.num_generate)
    if selected_record and candidate_noise.size(0) >= final_pool_count:
        samples = generate_gan_from_noise(
            generator=generator,
            noise=candidate_noise[:final_pool_count],
            device=device,
            sample_batch_size=args.sample_batch_size,
        )
        final_generation_noise = "selected_candidate_raw_noise"
    else:
        samples = generate_gan(
            generator=generator,
            num_samples=final_pool_count,
            latent_dim=args.latent_dim,
            device=device,
            sample_batch_size=args.sample_batch_size,
            truncation=args.truncation,
        )
        final_generation_noise = "fresh_raw_noise"
    samples = samples[: int(args.num_generate)].float().clamp(0.0, 1.0)
    generation_time = time.perf_counter() - gen_start
    saved_images = save_generated_images(samples, dirs["generated"])

    gen_arrays = samples.squeeze(1).numpy().astype(np.float32)
    metrics = compute_metrics(real_arrays, gen_arrays)
    final_target_compliance = target_compliance_details(metrics, args)
    selected_train_stats = last_stats
    if selected_record and isinstance(selected_record.get("train_stats"), dict):
        selected_train_stats = selected_record["train_stats"]  # type: ignore[assignment]
    final_selection_score, final_selection_details = candidate_selection_score(
        metrics,
        samples,
        real_eval_tensor,
        selected_train_stats,
        args,
    )
    training_history["best_epoch"] = best_candidate_epoch if best_candidate_epoch else best_balance_epoch
    training_history_paths = save_training_history_tables(training_history, dirs["logs"])
    supplemental_plot_paths = plot_supplemental_training_results(training_history, dirs["analysis_plots"])

    summary: Dict[str, object] = {
        "model": MODEL_NAME,
        "config": vars(args),
        "data_signature": signature,
        "parameter_count": total_params,
        "generator_parameter_count": gen_params,
        "discriminator_parameter_count": disc_params,
        "training_time_seconds": training_time,
        "single_image_generation_time_seconds": generation_time / max(args.num_generate, 1),
        "total_generation_time_seconds": generation_time,
        "peak_vram_mb": peak_memory_mb(device),
        "final_training_stats": last_stats,
        "best_balance_epoch": best_balance_epoch,
        "best_balance_score": best_balance_score if math.isfinite(best_balance_score) else None,
        "best_candidate_epoch": best_candidate_epoch,
        "best_selection_score": best_candidate_score if math.isfinite(best_candidate_score) else None,
        "selected_model_source": selected_model_source,
        "selected_candidate_is_eligible": selected_record_is_eligible,
        "final_generation_noise": final_generation_noise,
        "postprocessing_disabled": True,
        "disabled_output_rewrite_steps": [
            "sample-level intensity adjustment",
            "quality-based resampling",
            "post-evaluation distribution matching",
        ],
        "material_science_constraint": {
            "name": "CPTEC",
            "full_name": "Carbon Phase Transition Envelope Constraint",
            "description": (
                "Differentiable training-stage constraint for carbon EDS phase realism. "
                "It matches local dark pore, carbon-rich, and transition-gray envelopes "
                "and penalizes unrealistic pure-white blocks, hard black/white jumps, "
                "and missing phase-transition bands without rewriting generated samples."
            ),
            "loss_weight": args.cptec_loss_weight,
            "start_epoch": args.cptec_start_epoch,
            "warmup_epochs": args.cptec_warmup_epochs,
            "morphology_anchor": {
                "name": "MS-CMSA",
                "full_name": "Multi-scale Carbon Morphology Skeleton Anchor",
                "description": (
                    "A differentiable coarse-to-local anchor that aligns low-frequency pore/carbon skeletons, "
                    "soft carbon masks, and material interfaces to nearest raw EDS references during training."
                ),
                "loss_weight": args.morphology_anchor_loss_weight,
                "start_epoch": args.morphology_anchor_start_epoch,
                "warmup_epochs": args.morphology_anchor_warmup_epochs,
            },
            "selection_score_weight": args.score_phase_envelope_weight,
            "postprocess": False,
        },
        "selected_candidate": selected_record,
        "top_candidates": top_records,
        "all_candidate_count": len(all_candidate_records),
        "real_diversity_target": real_diversity_target,
        "real_pairwise_diversity_target": real_pairwise_target,
        "topology_proxy_targets": topology_target_summary,
        "training_history_files": training_history_paths,
        "supplemental_training_plots": supplemental_plot_paths,
        "target_compliance": final_target_compliance,
        "final_selection_score": final_selection_score,
        "final_closure_checks": final_selection_details,
        "generated_images": saved_images,
        "metrics": metrics,
    }
    final_table_paths = write_final_evaluation_tables(
        summary=summary,
        dirs=dirs,
        args=args,
        real_arrays=real_arrays,
        real_paths=paths,
        wgan_arrays=gen_arrays,
        wgan_image_paths=saved_images,
    )
    summary["final_evaluation_tables"] = final_table_paths
    save_json(dirs["logs"] / "summary_metrics.json", summary)
    print(f"[{MODEL_NAME}] summary saved to {dirs['logs'] / 'summary_metrics.json'}")
    return summary


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
