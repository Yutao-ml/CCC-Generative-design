"""Recompute event decisions from state observations in a released training log."""
from pathlib import Path
import argparse
import csv
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples.agent_feedback import EventTriggeredPhysicsFeedback, FrozenRFFEnsemble, _read_numeric_csv
from train_agent import make_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="physics_feedback_log.csv from train_agent.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/agent_training.json")
    parser.add_argument("--quick", action="store_true", help="Match a log produced using --quick")
    options = parser.parse_args()
    args = make_args(options.config, options.out, "cpu", options.quick)
    x, y = _read_numeric_csv(ROOT / "data/agent/feedback_soft21.csv")
    model = FrozenRFFEnsemble(x, y, members=args.physics_ensemble_members, features=args.physics_rff_features,
        gamma=args.physics_rff_gamma, ridge=args.physics_ridge, seed=args.seed + 917)
    controller = EventTriggeredPhysicsFeedback(model, args, torch.device("cpu"))
    options.out.mkdir(parents=True, exist_ok=True)
    with options.log.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mismatches = []
    for row in rows:
        epoch = int(row["epoch"])
        controller.begin_epoch(epoch, {})
        controller.current_epoch_values = [tuple(float(row[k]) for k in
            ["surrogate_risk_ucb", "predicted_Jmax_Acm2", "predicted_cum_deltaT_Ks", "epistemic_uncertainty"])]
        state = controller.end_epoch(epoch, dict(curve_health_penalty=float(row["curve_health_penalty"]),
            w_dist_raw=float(row["abs_wasserstein_distance"])), options.out)
        if state["event"] != row["event"] or abs(state["feedback_gain"] - float(row["feedback_gain"])) > 1e-10:
            mismatches.append(epoch)
    report = dict(records=len(rows), mismatch_epochs=mismatches, status="PASS" if not mismatches else "FAIL",
        scope="Policy replay from supplied state observations; no finite-element solves")
    (options.out / "replay_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
