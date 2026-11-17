"""Run the released generation, descriptor and candidate-screening examples."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples.analysis import load_carbon_field, response_endpoints, classify_responses, read_csv, check_structure_response_table
from dmegan_examples.morphology import microstructure_stats_single


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    descriptors = []
    index = read_csv(ROOT / "data/image_index.csv")
    for number, record in enumerate(index, 1):
        path = ROOT / record["file"]
        descriptors.append(dict(group=record["group"], image=path.name, **microstructure_stats_single(load_carbon_field(path))))
        if number % 100 == 0:
            print(f"Calculated image descriptors: {number}/{len(index)}", flush=True)
    means = []
    for group in ["characterization", "ar", "vae", "diffusion", "gan", "dmegan"]:
        selected = [r for r in descriptors if r["group"] == group]
        keys = [k for k in selected[0] if k not in {"group", "image"}]
        means.append(dict(group=group, images=len(selected), **{k: float(np.mean([r[k] for r in selected])) for k in keys}))
    write_csv(args.out / "image_descriptors.csv", descriptors)
    write_csv(args.out / "descriptor_group_means.csv", means)
    endpoints = response_endpoints(ROOT / "data")
    screening = classify_responses(endpoints)
    write_csv(args.out / "response_screening.csv", screening)
    report = dict(saved_image_groups={row["group"]: row["images"] for row in means}, response_records=len(screening),
                  candidate_records=sum(r["role"] == "candidate" for r in screening),
                  structure_response_check=check_structure_response_table(ROOT / "data", endpoints),
                  scope="Demonstration on the released example subsets")
    if not args.skip_generation:
        import torch
        torch.set_num_threads(4)
        from dmegan_examples.inference import sample
        report["sampling"] = sample(ROOT, args.out / "new_fields", args.count, args.device, args.seed)
    (args.out / "example_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
