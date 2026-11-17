"""Event-triggered, differentiable COMSOL-surrogate feedback for DME-GAN.

The module is intentionally independent from the GAN implementation.  It learns
an ensemble surrogate from real lightweight COMSOL endpoints, exposes a
differentiable upper-confidence-bound risk to the generator, and changes the
feedback gain only when an event controller detects stagnation, excessive
epistemic uncertainty, or coupled physics/realism deterioration.

COMSOL itself remains a non-differentiable high-fidelity oracle.  Gradients flow
through the frozen ensemble surrogate, never through COMSOL.
"""
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_COLUMNS = (
    "mean_gray_0_1_mean", "mean_gray_0_1_section_sd", "mean_gray_0_1_section_range",
    "std_gray_0_1_mean", "std_gray_0_1_section_sd", "std_gray_0_1_section_range",
    "rho_carbon_fraction_mean", "rho_carbon_fraction_section_sd", "rho_carbon_fraction_section_range",
    "U_uniformity_mean", "U_uniformity_section_sd", "U_uniformity_section_range",
    "A_agglomeration_mean", "A_agglomeration_section_sd", "A_agglomeration_section_range",
    "I_interface_density_mean", "I_interface_density_section_sd", "I_interface_density_section_range",
    "R_autocorrelation_length_mean", "R_autocorrelation_length_section_sd", "R_autocorrelation_length_section_range",
)
TARGET_COLUMNS = ("Jmax_20ms_A_cm2", "cum_deltaT_Ks")


def _read_numeric_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError("Physics feedback database is empty: %s" % path)
    if any(row.get("feature_schema") != "soft21-grid100-v1" for row in rows):
        raise ValueError("Feedback requires the released soft21-grid100-v1 encoding")
    x = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=np.float64)
    y = np.asarray([[float(row[name]) for name in TARGET_COLUMNS] for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Physics feedback database contains non-finite values")
    return x, y


def _robust_scale(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centre = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25.0, 75.0], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return centre, scale


class FrozenRFFEnsemble(nn.Module):
    """Bootstrap random-Fourier-feature ridge ensemble.

    The nonlinear ensemble is cheap, differentiable with respect to descriptors,
    and its member spread is used only as an epistemic-risk indicator.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        members: int = 12,
        features: int = 96,
        gamma: float = 0.35,
        ridge: float = 2.0,
        seed: int = 1729,
    ) -> None:
        super().__init__()
        rng = np.random.RandomState(seed)
        x_centre, x_scale = _robust_scale(x)
        y_centre, y_scale = _robust_scale(y)
        xs = (x - x_centre) / x_scale
        ys = (y - y_centre) / y_scale

        w_all, b_all, beta_all = [], [], []
        n, dim = xs.shape
        for _ in range(max(int(members), 3)):
            index = rng.randint(0, n, size=n)
            w = rng.normal(0.0, math.sqrt(2.0 * gamma), size=(dim, features))
            b = rng.uniform(0.0, 2.0 * math.pi, size=(features,))
            phi = math.sqrt(2.0 / features) * np.cos(xs[index].dot(w) + b)
            design = np.concatenate([np.ones((n, 1)), phi], axis=1)
            reg = np.eye(design.shape[1], dtype=np.float64) * ridge
            reg[0, 0] = 0.0
            beta = np.linalg.solve(design.T.dot(design) + reg, design.T.dot(ys[index]))
            w_all.append(w.astype(np.float32))
            b_all.append(b.astype(np.float32))
            beta_all.append(beta.astype(np.float32))

        self.register_buffer("x_centre", torch.tensor(x_centre, dtype=torch.float32))
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.register_buffer("y_centre", torch.tensor(y_centre, dtype=torch.float32))
        self.register_buffer("y_scale", torch.tensor(y_scale, dtype=torch.float32))
        self.register_buffer("rff_w", torch.tensor(np.stack(w_all), dtype=torch.float32))
        self.register_buffer("rff_b", torch.tensor(np.stack(b_all), dtype=torch.float32))
        self.register_buffer("ridge_beta", torch.tensor(np.stack(beta_all), dtype=torch.float32))
        self.rff_scale = math.sqrt(2.0 / features)

    def forward(self, descriptors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs = (descriptors - self.x_centre) / self.x_scale
        phase = torch.einsum("bd,mdf->mbf", xs, self.rff_w) + self.rff_b[:, None, :]
        phi = torch.cos(phase) * self.rff_scale
        ones = torch.ones(phi.shape[0], phi.shape[1], 1, device=phi.device, dtype=phi.dtype)
        design = torch.cat([ones, phi], dim=-1)
        pred_scaled = torch.einsum("mbf,mfo->mbo", design, self.ridge_beta)
        pred = pred_scaled * self.y_scale + self.y_centre
        mean = pred.mean(dim=0)
        std = pred.std(dim=0, unbiased=False)
        return mean, std, pred


def _per_image_soft_descriptors(images: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    x = F.interpolate(images, size=(100, 100), mode="area").clamp(0.0, 1.0)
    p = torch.sigmoid((x - threshold) / max(temperature, 1e-4))
    mean_gray = x.mean(dim=(1, 2, 3))
    std_gray = x.flatten(1).std(dim=1, unbiased=False)
    rho = p.mean(dim=(1, 2, 3))

    block = max(4, min(int(x.shape[-1] // 16), 32))
    local = F.avg_pool2d(p, kernel_size=block, stride=block)
    local_mean = local.mean(dim=(1, 2, 3))
    local_std = local.flatten(1).std(dim=1, unbiased=False)
    uniformity = (1.0 - local_std / local_mean.clamp_min(1e-4)).clamp(0.0, 1.0)

    kernel = max(3, min(int(x.shape[-1] // 32) * 2 + 1, 17))
    neighbourhood = F.avg_pool2d(p, kernel_size=kernel, stride=1, padding=kernel // 2)
    aggregation = ((neighbourhood - rho[:, None, None, None]).relu().mean(dim=(1, 2, 3)) * 4.0).clamp(0.0, 1.0)

    dx = (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    dy = (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    interface = ((dx + dy) * 0.5).clamp_min(1e-6)

    centred = p - p.mean(dim=(2, 3), keepdim=True)
    variance = centred.square().mean(dim=(1, 2, 3)).clamp_min(1e-6)
    corr_x = (centred[:, :, :, :-1] * centred[:, :, :, 1:]).mean(dim=(1, 2, 3)) / variance
    corr_y = (centred[:, :, :-1, :] * centred[:, :, 1:, :]).mean(dim=(1, 2, 3)) / variance
    corr = ((corr_x + corr_y) * 0.5).clamp(-0.95, 0.995)
    correlation_length = (1.0 / (1.0 - corr).clamp_min(0.02) / float(x.shape[-1])).clamp(0.0, 1.0)
    return torch.stack([mean_gray, std_gray, rho, uniformity, aggregation, interface, correlation_length], dim=1)


def differentiable_volume_fingerprint(
    images: torch.Tensor,
    group_size: int = 4,
    threshold: float = 0.35,
    temperature: float = 0.08,
) -> torch.Tensor:
    """Map generated 2-D sections to one differentiable volumetric fingerprint.

    ``group_size`` is a Monte-Carlo quadrature batch, not a claim that the finite
    element model contains only that many physical layers.
    """
    per_image = _per_image_soft_descriptors(images, threshold, temperature)
    group_size = max(2, int(group_size))
    usable = (per_image.shape[0] // group_size) * group_size
    if usable == 0:
        per_image = per_image.repeat(group_size, 1)[:group_size]
        usable = group_size
    grouped = per_image[:usable].reshape(-1, group_size, per_image.shape[1])
    mean = grouped.mean(dim=1)
    std = grouped.std(dim=1, unbiased=False)
    span = grouped.max(dim=1).values - grouped.min(dim=1).values
    # Database order is seven descriptors, each with mean/std/range.
    return torch.stack(
        [value for descriptor in range(7) for value in (mean[:, descriptor], std[:, descriptor], span[:, descriptor])],
        dim=1,
    )


class EventTriggeredPhysicsFeedback:
    def __init__(self, surrogate: FrozenRFFEnsemble, args, device: torch.device) -> None:
        self.surrogate = surrogate.to(device).eval()
        for parameter in self.surrogate.parameters():
            parameter.requires_grad_(False)
        self.args = args
        self.device = device
        self.gain = 0.0
        self.integral_error = 0.0
        self.risk_history: List[float] = []
        self.uncertainty_history: List[float] = []
        self.current_epoch_values: List[Tuple[float, float, float, float]] = []
        self.last_event = "observe"
        self.last_reason = "minimum epoch not reached"
        self.safe_risk = float("inf")
        self.safe_epoch = 0
        self.last_oracle_epoch = -10**9
        self.rollback_requested = False

    @classmethod
    def from_args(cls, args, device: torch.device) -> Optional["EventTriggeredPhysicsFeedback"]:
        if not bool(getattr(args, "physics_feedback", False)):
            return None
        database = Path(args.physics_feedback_database)
        x, y = _read_numeric_csv(database)
        surrogate = FrozenRFFEnsemble(
            x, y,
            members=args.physics_ensemble_members,
            features=args.physics_rff_features,
            gamma=args.physics_rff_gamma,
            ridge=args.physics_ridge,
            seed=args.seed + 917,
        )
        return cls(surrogate, args, device)

    def begin_epoch(self, epoch: int, last_stats: Dict[str, float]) -> None:
        self.current_epoch_values = []
        self.rollback_requested = False
        if epoch < int(self.args.physics_feedback_min_epoch):
            self.gain = 0.0

    def loss(self, images: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fingerprint = differentiable_volume_fingerprint(
            images,
            group_size=self.args.physics_feedback_group_size,
            threshold=self.args.topology_threshold,
            temperature=self.args.topology_temperature,
        )
        mean, std, _ = self.surrogate(fingerprint)
        j_scale = max(float(self.args.physics_target_jmax), 1e-6)
        t_scale = max(float(self.args.physics_target_cum_delta_t), 1e-6)
        normalized_mean = torch.stack([mean[:, 0] / j_scale, mean[:, 1] / t_scale], dim=1)
        normalized_std = torch.stack([std[:, 0] / j_scale, std[:, 1] / t_scale], dim=1)
        ucb = normalized_mean + float(self.args.physics_ucb_beta) * normalized_std
        # Tail-risk feedback: optimize the worst candidate volumes in the
        # generated batch instead of only improving its mean response.
        per_volume_risk = F.softplus(ucb - 1.0).mean(dim=1)
        alpha = float(getattr(self.args, "physics_cvar_alpha", 0.90))
        alpha = min(max(alpha, 0.0), 0.99)
        tail_count = max(1, int(math.ceil((1.0 - alpha) * per_volume_risk.numel())))
        raw_risk = torch.topk(per_volume_risk, k=tail_count, largest=True).values.mean()
        diagnostics = {
            "risk": raw_risk.detach(),
            "jmax": mean[:, 0].mean().detach(),
            "cum_delta_t": mean[:, 1].mean().detach(),
            "uncertainty": normalized_std.mean().detach(),
        }
        self.current_epoch_values.append(tuple(float(diagnostics[k].cpu()) for k in ("risk", "jmax", "cum_delta_t", "uncertainty")))
        return raw_risk * float(self.gain), diagnostics

    def end_epoch(self, epoch: int, last_stats: Dict[str, float], log_dir: Path) -> Dict[str, object]:
        if self.current_epoch_values:
            values = np.asarray(self.current_epoch_values, dtype=np.float64)
            risk, jmax, cum_dt, uncertainty = values.mean(axis=0)
        else:
            risk = jmax = cum_dt = uncertainty = float("nan")
        if math.isfinite(risk):
            self.risk_history.append(float(risk))
            self.uncertainty_history.append(float(uncertainty))

        curve_penalty = float(last_stats.get("curve_health_penalty", float("inf")))
        wdist = abs(float(last_stats.get("w_dist_raw", float("inf"))))
        realism_safe = curve_penalty <= float(self.args.physics_realism_guard) and wdist <= float(self.args.physics_wdist_guard)
        min_epoch = int(self.args.physics_feedback_min_epoch)
        window = max(3, int(self.args.physics_plateau_window))
        plateau = False
        relative_improvement = float("nan")
        if len(self.risk_history) >= window:
            tail = self.risk_history[-window:]
            reference = max(abs(tail[0]), 1e-6)
            relative_improvement = (tail[0] - min(tail[1:])) / reference
            plateau = relative_improvement < float(self.args.physics_plateau_min_delta)

        event, reason = "observe", "risk still improving or controller not eligible"
        if epoch < min_epoch:
            self.gain = 0.0
            event, reason = "warmup", "physics feedback locked before minimum epoch"
        elif (
            math.isfinite(uncertainty)
            and uncertainty > float(self.args.physics_uncertainty_trigger)
            and epoch - self.last_oracle_epoch >= int(self.args.physics_oracle_cooldown_epochs)
        ):
            self.gain *= float(self.args.physics_uncertainty_attenuation)
            self.last_oracle_epoch = int(epoch)
            event, reason = "oracle_query", "surrogate epistemic uncertainty exceeded threshold"
        elif math.isfinite(uncertainty) and uncertainty > float(self.args.physics_uncertainty_trigger):
            self.gain *= float(self.args.physics_feedback_decay)
            event, reason = "uncertainty_hold", "oracle query is inside cooldown; feedback is not amplified"
        elif plateau and realism_safe:
            target_error = max(risk - float(self.args.physics_risk_target), 0.0)
            self.integral_error = float(np.clip(
                self.integral_error + target_error,
                -float(self.args.physics_integral_clip),
                float(self.args.physics_integral_clip),
            ))
            exponent = float(self.args.physics_controller_kp) * target_error + float(self.args.physics_controller_ki) * self.integral_error
            candidate_gain = max(float(self.args.physics_feedback_initial_gain), self.gain)
            self.gain = float(np.clip(candidate_gain * math.exp(exponent), 0.0, float(self.args.physics_feedback_max_gain)))
            event, reason = "feedback_boost", "physics improvement plateaued while realism guard passed"
        elif not realism_safe and self.gain > 0.0:
            self.gain *= float(self.args.physics_guard_attenuation)
            event, reason = "feedback_relief", "adversarial realism guard requested lower physics gain"
        else:
            self.gain *= float(self.args.physics_feedback_decay)

        if realism_safe and math.isfinite(risk) and risk < self.safe_risk:
            self.safe_risk, self.safe_epoch = float(risk), int(epoch)
            save_safe = True
        else:
            save_safe = False

        if (
            epoch >= min_epoch
            and self.safe_epoch > 0
            and math.isfinite(risk)
            and risk > self.safe_risk * (1.0 + float(self.args.physics_rollback_risk_fraction))
            and not realism_safe
        ):
            self.rollback_requested = True
            self.gain *= float(self.args.physics_guard_attenuation)
            event, reason = "rollback", "physics risk and realism deteriorated together"

        self.last_event, self.last_reason = event, reason
        row = {
            "epoch": epoch, "event": event, "reason": reason, "feedback_gain": self.gain,
            "surrogate_risk_ucb": risk, "predicted_Jmax_Acm2": jmax,
            "predicted_cum_deltaT_Ks": cum_dt, "epistemic_uncertainty": uncertainty,
            "relative_improvement_window": relative_improvement,
            "curve_health_penalty": curve_penalty, "abs_wasserstein_distance": wdist,
            "realism_guard_pass": realism_safe, "safe_epoch": self.safe_epoch,
            "safe_risk": self.safe_risk if math.isfinite(self.safe_risk) else "",
        }
        self._append_csv(Path(log_dir) / "physics_feedback_log.csv", row)
        if event == "oracle_query":
            self._append_csv(Path(log_dir) / "physics_oracle_query_queue.csv", row)
        with (Path(log_dir) / "physics_feedback_state.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, indent=2)
        return {**row, "save_safe_checkpoint": save_safe, "rollback_requested": self.rollback_requested}

    @staticmethod
    def _append_csv(path: Path, row: Dict[str, object]) -> None:
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def state_dict_for_checkpoint(self) -> Dict[str, object]:
        return {
            "gain": self.gain, "integral_error": self.integral_error,
            "risk_history": self.risk_history, "uncertainty_history": self.uncertainty_history,
            "safe_risk": self.safe_risk, "safe_epoch": self.safe_epoch,
            "last_oracle_epoch": self.last_oracle_epoch,
        }
