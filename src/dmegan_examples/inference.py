"""Inference with the selected generator and its reference-bank inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from .analysis import prepare_reference_fields
from .generator import Generator


def load_generator(root, device="cpu"):
    root = Path(root)
    config = json.loads((root / "configs/inference.json").read_text("utf-8"))
    metadata = json.loads((root / "data/metadata.json").read_text("utf-8"))
    path = root / "data/checkpoint/selected_generator.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint checksum does not match its metadata")
    generator = Generator(**config["generator"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    generator.load_state_dict(state, strict=True)
    generator.to(device).eval()
    references = torch.from_numpy(prepare_reference_fields(root / "data/characterization")).to(device)
    generator.set_reference_skeleton_bank(references, resolution=config["reference_resolution"])
    return generator, config


def sample(root, out, count=20, device="cpu", seed=1729):
    if not 1 <= count <= 100:
        raise ValueError("This example supports 1 to 100 fields per run")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty output directory")
    generator, config = load_generator(root, device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    fields = []
    with torch.no_grad():
        for _ in range(count):
            z = torch.randn(1, config["generator"]["latent_dim"], device=device) * config["latent_scale"]
            fields.append(((generator(z) + 1) * 0.5).clamp(0, 1).cpu().numpy())
    arrays = np.concatenate(fields).astype(np.float32)
    if not np.isfinite(arrays).all() or arrays.shape != (count, 1, 512, 512):
        raise ValueError("Invalid generated field")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "fields_float32.npz", fields=arrays)
    torch.save(state, out / "rng_state.pt")
    for index, field in enumerate(arrays[:, 0], 1):
        Image.fromarray(np.rint(field * 255).astype(np.uint8)).save(out / f"new_{index:02d}.png")
    record = dict(count=count, device=str(device), seed=seed, selected_epoch=config["selected_epoch"],
                  tensor_order="N,C,H,W", intensity_range=[0, 1], reference_bank_initialized=True,
                  torch_version=str(torch.__version__), scope="New inference examples; separate from retained study data")
    (out / "sampling.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    return record
