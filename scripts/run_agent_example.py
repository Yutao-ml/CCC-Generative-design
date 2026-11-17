"""Verify paired section encoding, fit a response ensemble and check feedback gradients."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples import agent_feedback as feedback
from dmegan_examples import agent_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    out = options.out.resolve()
    if out.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Write new outputs outside data/")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    x, y = feedback._read_numeric_csv(ROOT / "data/agent/feedback_soft21.csv")
    encoded, ids, example = [], [], None
    for path in sorted((ROOT / "data/agent").glob("section_fields_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            fields = torch.from_numpy(data["fields"]).float().reshape(-1, 1, 100, 100)
            with torch.no_grad():
                encoded.append(feedback.differentiable_volume_fingerprint(fields).numpy())
            ids.extend(data["run_ids"].tolist())
            if example is None:
                example = fields[:12].clone().requires_grad_(True)
    difference = float(np.max(np.abs(np.concatenate(encoded) - x)))
    if difference > 1e-7 or len(ids) != 300:
        raise ValueError("Feedback table does not match paired section encoding")
    model = feedback.FrozenRFFEnsemble(x, y, members=12, features=96, gamma=0.35, ridge=2.0, seed=1729).eval()
    args = agent_training.parse_args([])
    agent = feedback.EventTriggeredPhysicsFeedback(model, args, torch.device("cpu"))
    agent.gain = args.physics_feedback_initial_gain
    loss, diagnostics = agent.loss(example)
    gradient = torch.autograd.grad(loss, example)[0]
    if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise ValueError("Feedback gradient must be finite and nonzero")
    with torch.no_grad():
        mean, sd, _ = model(torch.from_numpy(x).float())
    with (out / "ensemble_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["run_id", "Jmax_mean_A_cm2", "Jmax_sd_A_cm2", "thermal_integral_mean_K_s", "thermal_integral_sd_K_s"])
        for i, name in enumerate(ids):
            writer.writerow([name, float(mean[i,0]), float(sd[i,0]), float(mean[i,1]), float(sd[i,1])])
    torch.save(model.state_dict(), out / "new_fit_surrogate.pt")
    report = dict(status="PASS", paired_records=len(x), features=x.shape[1], ensemble_members=12, random_features=96,
        encoding_max_abs_difference=difference, frozen_surrogate_parameters=True, input_gradient_finite=True,
        input_gradient_abs_sum=float(gradient.abs().sum()), current_density_unit="A/cm^2", thermal_integral_unit="K*s",
        scope="Encoding consistency and feedback software check on retained paired inputs")
    (out / "agent_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
