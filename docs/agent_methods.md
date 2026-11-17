# Agent implementation and paired inputs

## Scope and manuscript correspondence

The agent code implements the ensemble-surrogate and feedback mechanisms in Methods 5.6: a bootstrap ridge ensemble with random Fourier features, a differentiable two-channel response penalty, an upper-tail minibatch objective, state-dependent gain control and generator checkpoint protection. `agent_training.py` provides the DME-GAN structural training loop with the feedback term. `agent_feedback.py` contains the response model, smooth section encoding and event controller.

`train_agent.py` starts a new run. The selected epoch-270 inference checkpoint and the existing 500-image release remain unchanged. The manuscript's Fig. 5c reports retrospective controller assessment. `replay_agent_controller.py` recomputes decisions from the supplied state log of a run; its outputs are identified by that input log.

## Paired response-model inputs

Each of the 300 retained screening records is paired with its four original 100x100 interpolation fields. `run_id` links the section arrays, the smooth descriptor table and the existing response records. The two targets are maximum current density at 20 ms (A/cm^2) and the 0-50 ms integral of spatial temperature range (K*s). Their numerical values are preserved.

The feedback encoding uses a common 100x100 area-resampled grid for retained fields and generated inputs. A sigmoid with threshold 0.35 and temperature 0.08 defines smooth occupancy. Seven section quantities describe intensity mean, intensity standard deviation, occupancy, uniformity, aggregation, interfaces and spatial correlation. Means, population standard deviations and ranges across four representative sections give 21 inputs, ordered descriptor by descriptor. `feature_schema=soft21-grid100-v1` is checked when fitting the feedback model.

The smooth uniformity proxy uses block-occupancy dispersion. The aggregation proxy measures positive neighborhood occupancy above global occupancy. Interface density uses directional differences in smooth occupancy. The correlation proxy derives a normalized length from nearest-neighbor occupancy correlations. Their exact operators and regularizers are in `_per_image_soft_descriptors`. The retained screening-feature table has its own definitions, described in the data dictionary. The paired smooth table is regenerated from the released sections with the same feedback encoder used on generated fields.

## Frozen ensemble and response guidance

Twelve bootstrap ridge models use 96 random Fourier features each, RBF gamma 0.35 and ridge coefficient 2.0. The fitted intercept is unpenalized. Inputs and responses are standardized by medians and interquartile ranges. Ensemble means and population standard deviations are restored to response units before guidance is calculated.

For each channel, the predicted mean plus 0.75 times ensemble standard deviation is divided by its response scale (135 A/cm^2 or 0.85 K*s). Softplus of the normalized value minus one gives a smooth exceedance penalty, averaged over the two channels. The largest ceil(0.1*N) candidate penalties form the minibatch tail objective. Ensemble disagreement is a search diagnostic. Gradients flow through the fitted response model's inputs, while its parameters and coefficients remain fixed.

The training loop adds the weighted tail objective to the DME-GAN structural objective. Fitted response-model buffers do not receive optimizer updates. Current feedback gain remains fixed during an epoch; the state at epoch end selects the next gain.

## Ordered controller and protection

The controller starts with a warm-up interval. Excessive ensemble disagreement attenuates guidance and writes an evaluation request, subject to a cooldown. When progress stalls, the proportional-integral rule can increase guidance if the morphology guard passes and disagreement is limited. Morphology deterioration reduces guidance. Joint risk and morphology deterioration can request restoration of a safe generator checkpoint. The safe checkpoint includes generator and generator-optimizer states.

Parameters and thresholds are recorded in `configs/agent_training.json`. Requests are saved in `physics_oracle_query_queue.csv`; the training entry point uses the supplied fixed response dataset. New response evaluations require separately selected candidate fields and the external simulation workflow. The default entry point does not launch finite-element jobs or replace the response labels.

## Commands and validation

```text
python scripts/run_agent_example.py --out results/agent_check
python scripts/train_agent.py --out results/agent_training --device cuda
python scripts/replay_agent_controller.py --log results/agent_training/logs/physics_feedback_log.csv --out results/controller_replay
```

`--quick` reduces model capacity, data presentations and run length for software testing. It can also be passed to the replay command to match such a log. New outputs are stored outside `data/`. The example encoder check reconstructs all 300 descriptor vectors from the NPZ arrays, checks finite nonzero input gradients and fits a response ensemble. Software validation records report these checks; predictive accuracy or new material-performance improvements are not inferred from them.
