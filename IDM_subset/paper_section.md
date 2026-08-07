# IDM Baseline Evaluation

We estimate the safety-critical probability of an IDM-controlled ego vehicle under
highD long-tail driving conditions. A test case consists of a scenario condition
`c` sampled from the highD tail scenario-condition distribution and a diffusion
latent variable `z ~ N(0, I)`. A frozen conditional diffusion model decodes
`(c, z)` into the adversary action sequence, while the IDM ego vehicle responds
closed-loop using `tools/idm_ego.yaml`.

```text
c ~ highD tail scenario-condition distribution
z ~ N(0, I)
adversary actions = deterministic DDIM(c, z)
ego actions = IDM(state)
score = S_EVT(Y_sim)
failure = score >= S_EVT(x_c(e)), x_c(following) = 4.7773, x_c(cut-in) = 4.6859
```

The reported quantity is the tail-conditional probability
`P(failure | highD tail scenario condition)`, rather than a collision probability
over the complete naturalistic highD distribution. When the exposure reliability
conditions pass, the output also maps this probability to a global highD exposure
intensity through the independent tail-peak rate.

For each event \(e\), \(x_c(e)\) is the event-specific EVT-calibrated
collision-critical level recorded in its paired exposure summary. It is not a
shared constant of 5.0.

## Current Configuration

| Event | Closed-loop setup | Subset simulation | Independent Monte Carlo |
| --- | --- | --- | --- |
| Following | 125 steps, 1 lane, kinematic bicycle | `N=3000`, `p0=0.20`, `proposal_std=0.12`, context refresh `0.70`, 6 MH retries, 8 maximum levels, adaptive stop disabled | 200,000 samples |
| Cut-in | 100 steps, 2 lanes, point-mass target vehicle | `N=1000`, `p0=0.10`, `proposal_std=0.10`, context refresh `0.50`, 4 MH retries, 8 maximum levels, adaptive stop enabled | 20,000 samples |

The diffusion checkpoints are the frozen train/validation/test-selected models in
`results/diffusion_natural/*/checkpoints/`; no fallback to older full-data
checkpoints is used.

## Estimation and Uncertainty

Subset simulation advances through conditional levels by retaining the highest-risk
`p0` fraction and generating conditional samples with Metropolis--Hastings proposals
in the joint `(c, z)` space. Its per-run estimate is

```text
P_hat = p0^level_idx * mean(score >= failure_threshold at the final level).
```

The main uncertainty evidence for the current default configuration is five
independent SS seeds (`101, 202, 303, 404, 505`), not the internal binomial-style
standard error from a single MCMC-correlated run. The five-seed mean is compared
with a separately generated high-budget IID MC estimate. Diagnostics include the
number of closed-loop evaluations, context/state diversity, largest context/state
share, and MH acceptance rate.

## Current Saved Results

| Event | Five-seed SS mean (sample SD) | Independent MC |
| --- | --- | --- |
| Following | `0.00249333` (`0.00053073`) | `0.00241000` from 200,000 samples (SE `0.00010964`) |
| Cut-in | `0.00672500` (`0.00193488`) | `0.00695000` from 20,000 samples (SE `0.00058744`) |

The cut-in five-seed 95% t interval `[0.004323, 0.009127]` overlaps the MC interval
`[0.005799, 0.008101]`. A single current cut-in SS output (`0.009`) is retained for
diagnostics and playback but is not substituted for the five-seed estimate.

The artifacts are under `IDM_subset/results/`. Final-level playback reuses the
saved subset samples and replays safety-critical final-level cases without
resampling the scenario distribution.
