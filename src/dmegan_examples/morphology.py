"""Per-image descriptors on normalized two-dimensional carbon-intensity fields."""
from typing import Dict,List,Tuple
import math
import numpy as np
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
