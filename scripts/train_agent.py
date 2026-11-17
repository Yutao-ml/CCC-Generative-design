"""Start a new DME-GAN run with frozen-surrogate, event-triggered feedback."""
from pathlib import Path
import argparse
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples import agent_training


def make_args(config_path, output, device="auto", quick=False):
    args = agent_training.parse_args([])
    config = json.loads(Path(config_path).read_text("utf-8"))
    for name, value in config["parameters"].items():
        if not hasattr(args, name):
            raise ValueError(f"Unknown training parameter: {name}")
        if isinstance(getattr(args, name), Path):
            value = (ROOT / value).resolve()
        setattr(args, name, value)
    args.data_dir = ROOT / "data/characterization"
    args.physics_feedback_database = ROOT / "data/agent/feedback_soft21.csv"
    args.output_dir = Path(output).resolve()
    if args.output_dir.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Write new training outputs outside data/")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    args.device = device
    args.quick = quick
    agent_training.apply_quick_overrides(args)
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent_training.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quick", action="store_true", help="Reduced-capacity software check; not a scientific comparison")
    options = parser.parse_args()
    torch.set_num_threads(4)
    args = make_args(options.config, options.out, options.device, options.quick)
    result = agent_training.train(args)
    print(json.dumps(dict(output=str(args.output_dir), quick=options.quick,
                         scope="New agent-assisted training run", selected_epoch=result.get("best_candidate_epoch")), indent=2))


if __name__ == "__main__":
    main()
