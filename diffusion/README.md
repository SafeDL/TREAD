# Naturalistic Action Diffusion Prior

`diffusion/` now contains a clean highD-based natural driving prior for car-following events. Its only job is to learn the natural distribution of future lead-car actions from short ego-lead interaction histories.

The model answers:

```text
Given recent ego-lead interaction history, can we generate natural, smooth,
physically feasible lead-car future actions close to highD statistics?
```

It does not condition on target severity or try to control generated severity. Tail calibration, guided generation, RSS checks, and closed-loop adversarial testing belong in later modules outside this natural-prior training path.

## Current Scope

- Event type: `following`
- Input: ego/lead history states, relative-history stream, and current/history-only context features
- Output: lead-car future action sequence, defaulting to jerk `[jx(t+1), ..., jx(t+H)]`
- Training objective: DDPM noise MSE plus optional `x0` reconstruction L1 and smoothness auxiliary loss
- Sampling: standard unconditional DDPM sampling conditioned only on scene history/context

## Key Files

```text
diffusion/
  scripts/
    configs/natural_following.yaml
    01_build_natural_dataset.py
    02_train_natural_diffusion.py
    03_evaluate_natural_prior.py
    04_sample_natural_rollouts.py
    diagnose_validation_variance.py
  src/
    data.py
    features.py
    normalization.py
    types.py
    model.py
    train.py
    kinematics.py
    utils.py
```

`src/analysis_risk.py` is retained only for offline post-generation analysis. It is not used by natural-prior dataset construction, training, or sampling.

## Environment

The base conda environment may not have PyTorch. The existing project environment can be used with:

```bash
conda activate jzm
```

or explicitly:

```bash
/home/hp/anaconda3/envs/jzm/bin/python
```

## Build Dataset

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/01_build_natural_dataset.py
```

The default config writes to:

```text
data/diffusion_natural/following/
```

Main dataset arrays:

```text
context_states
context_features
relative_history
actions
split_index
recording_id
event_id
anchor_frame
ego_length
adv_length
lane_width
```

## Train

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/02_train_natural_diffusion.py
```

To force a dataset rebuild before training:

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/02_train_natural_diffusion.py --rebuild-dataset
```

Checkpoints:

```text
checkpoints/best.pt
checkpoints/best_noise_mse.pt
checkpoints/last.pt
```

## Evaluate Natural Prior

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/03_evaluate_natural_prior.py
```

Outputs:

```text
naturalness_summary.json
naturalness_metrics.csv
diversity_summary.json
natural_prior_plots/ax_distribution_real_vs_generated.png
natural_prior_plots/jerk_distribution_real_vs_generated.png
natural_prior_plots/speed_distribution_real_vs_generated.png
natural_prior_plots/example_rollouts.png
```

The evaluator reports:

- validation denoising/reconstruction/smoothness metrics
- acceleration and jerk distribution statistics
- Wasserstein, KS, and histogram L1 distances
- action clipping, speed, jerk, acceleration, and trajectory discontinuity rates
- lead speed, final speed, displacement, and gap statistics after trajectory integration
- multi-sample diversity for repeated contexts

When the action representation is jerk, evaluation first integrates generated jerk to acceleration before calling `integrate_following_actions()`.

## Sample Rollouts

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/04_sample_natural_rollouts.py
```

This writes decoded actions, acceleration, and integrated lead trajectories to:

```text
natural_rollouts.npz
natural_rollouts_summary.json
```

## Validation Stability Diagnostic

Use this to estimate how much validation variation comes from diffusion timestep/noise sampling:

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/diagnose_validation_variance.py
```

It writes:

```text
validation_variance_summary.json
validation_variance.csv
```

The core tracked signals are:

- `loss`
- `noise_mse`
- `x0_l1`
- `smooth`
