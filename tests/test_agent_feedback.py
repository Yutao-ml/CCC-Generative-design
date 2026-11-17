from pathlib import Path
import csv
import json
import sys
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dmegan_examples import agent_feedback as af
from dmegan_examples import agent_training as at

torch.set_num_threads(4)


def model_and_args():
    x, y = af._read_numeric_csv(ROOT / "data/agent/feedback_soft21.csv")
    return af.FrozenRFFEnsemble(x, y, members=12, features=96, seed=1729), at.parse_args([])


def observation(controller, epoch, risk=0.4, uncertainty=0.1, safe=True, folder=None):
    controller.begin_epoch(epoch, {})
    controller.current_epoch_values = [(risk, 135.0, 0.85, uncertainty)]
    return controller.end_epoch(epoch, dict(curve_health_penalty=0.0 if safe else 1.0,
        w_dist_raw=0.0 if safe else 2.0), folder)


def test_paired_feedback_rows_preserve_the_existing_physical_targets():
    with (ROOT / "data/responses/structure_response.csv").open(encoding="utf-8") as f:
        responses = {r["run_id"]:r for r in csv.DictReader(f)}
    with (ROOT / "data/agent/feedback_soft21.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 300
    for row in rows:
        original = responses[row["run_id"]]
        assert row["feature_schema"] == "soft21-grid100-v1"
        for key in ["Jmax_20ms_A_cm2", "cum_deltaT_Ks"]:
            assert float(row[key]) == float(original[key])


def test_paired_fields_reproduce_every_smooth_feature():
    x, _ = af._read_numeric_csv(ROOT / "data/agent/feedback_soft21.csv")
    values = []
    for path in sorted((ROOT / "data/agent").glob("section_fields_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            fields = torch.from_numpy(data["fields"]).float().reshape(-1, 1, 100, 100)
            with torch.no_grad():
                values.append(af.differentiable_volume_fingerprint(fields).numpy())
    assert np.allclose(np.concatenate(values), x, rtol=0, atol=1e-7)


def test_screening_features_cannot_be_used_as_smooth_training_inputs():
    with pytest.raises(ValueError, match="soft21-grid100-v1"):
        af._read_numeric_csv(ROOT / "data/responses/structure_response.csv")


def test_ensemble_is_frozen_and_preserves_finite_input_gradients():
    model, _ = model_and_args()
    x, _ = af._read_numeric_csv(ROOT / "data/agent/feedback_soft21.csv")
    features = torch.tensor(x[:3], dtype=torch.float32, requires_grad=True)
    mean, sd, members = model(features)
    gradient = torch.autograd.grad(mean.sum(), features)[0]
    assert len(list(model.parameters())) == 0
    assert members.shape == (12, 3, 2)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert (sd >= 0).all()


def test_tail_loss_uses_channel_units_and_largest_candidate_penalties():
    class SpecifiedResponse(nn.Module):
        def forward(self, x):
            mean = torch.stack([135.0 * x[:,0], 0.85 * x[:,0] * 2], dim=1)
            sd = torch.tensor([13.5, 0.085], dtype=x.dtype).expand_as(mean)
            return mean, sd, mean[None]
    args = at.parse_args([])
    agent = af.EventTriggeredPhysicsFeedback(SpecifiedResponse(), args, torch.device("cpu"))
    agent.gain = 1.0
    images = torch.linspace(0.0, 1.0, 12 * 100 * 100).reshape(12,1,100,100)
    x = af.differentiable_volume_fingerprint(images)
    mean, sd, _ = agent.surrogate(x)
    penalty = F.softplus((mean + 0.75 * sd) / torch.tensor([135.0,0.85]) - 1).mean(dim=1)
    loss, _ = agent.loss(images)
    assert torch.allclose(loss, penalty.max())


def test_section_statistics_are_permutation_invariant():
    with np.load(ROOT / "data/agent/section_fields_01.npz", allow_pickle=False) as data:
        x = torch.from_numpy(data["fields"][:1]).float().reshape(4,1,100,100)
    assert torch.allclose(af.differentiable_volume_fingerprint(x),
        af.differentiable_volume_fingerprint(x[[3,1,0,2]]), atol=1e-7)


def test_high_uncertainty_requests_a_query_and_respects_cooldown(tmp_path):
    model, args = model_and_args()
    controller = af.EventTriggeredPhysicsFeedback(model, args, torch.device("cpu"))
    controller.gain = 0.2
    first = observation(controller, 80, uncertainty=0.5, folder=tmp_path)
    assert first["event"] == "oracle_query" and first["feedback_gain"] < 0.2
    second = observation(controller, 81, uncertainty=0.5, folder=tmp_path)
    assert second["event"] == "uncertainty_hold"
    assert (tmp_path / "physics_oracle_query_queue.csv").is_file()


def test_plateau_gain_is_applied_after_the_control_step(tmp_path):
    model, args = model_and_args()
    controller = af.EventTriggeredPhysicsFeedback(model, args, torch.device("cpu"))
    for epoch in range(80, 91):
        state = observation(controller, epoch, folder=tmp_path)
    assert controller.gain == 0
    controller.begin_epoch(91, {})
    assert controller.gain == 0
    controller.current_epoch_values = [(0.4,135.0,0.85,0.1)]
    state = controller.end_epoch(91, dict(curve_health_penalty=0.0,w_dist_raw=0.0), tmp_path)
    assert state["event"] == "feedback_boost" and state["feedback_gain"] > 0
    controller.begin_epoch(92, {})
    assert controller.gain == state["feedback_gain"]


def test_joint_deterioration_requests_checkpoint_protection(tmp_path):
    model, args = model_and_args()
    controller = af.EventTriggeredPhysicsFeedback(model, args, torch.device("cpu"))
    first = observation(controller, 80, risk=0.2, folder=tmp_path)
    assert first["save_safe_checkpoint"]
    controller.gain = 0.2
    second = observation(controller, 81, risk=0.5, safe=False, folder=tmp_path)
    assert second["rollback_requested"] and second["event"] == "rollback"


def test_training_and_inference_share_the_published_generator_architecture():
    from dmegan_examples.inference import load_generator
    generator, config = load_generator(ROOT)
    training_generator = at.Generator(**config["generator"])
    training_generator.load_state_dict(generator.state_dict(), strict=True)
    assert sum(p.numel() for p in training_generator.parameters()) == 11_881_726
