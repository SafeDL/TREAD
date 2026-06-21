# SAIRL Long-Tail Evaluation Summary

Runtime: `conda activate tread`; CUDA enabled on `NVIDIA GeForce RTX 4090 D`.

Fairness invariant: diffusion checkpoint, highD tail context distribution, EVT threshold, scoring, seed, subset `N`, and `p0` match `IDM_subset`; only the tested ADS differs.

| Event | Estimator | ADS | Probability | 95% CI | Reliability | Closed-loop evals |
|---|---:|---:|---:|---:|---:|---:|
| car-following | Monte Carlo | SAIRL | 0.16621 | [0.16390265, 0.16851735] | - | 100000 |
| car-following | Subset | SAIRL | 0.17946667 | [0.17729438, 0.18163895] | pass | 10585 |
| car-following | Monte Carlo | IDM | 0.00255 | [0.0022374124, 0.0028625876] | - | 100000 |
| car-following | Subset | IDM | 0.0024906667 | [0.0023581096, 0.0026232237] | pass | 29303 |
| cut-in | Monte Carlo | SAIRL | 0.0371 | [0.033395466, 0.040804534] | - | 10000 |
| cut-in | Subset | SAIRL | 0.0353 | [0.032337928, 0.038262072] | pass | 3388 |
| cut-in | Monte Carlo | IDM | 0.0065 | [0.0049249415, 0.0080750585] | - | 10000 |
| cut-in | Subset | IDM | 0.0068 | [0.0052396627, 0.0083603373] | pass | 3267 |

| Event | SAIRL subset / IDM subset | SAIRL MC / IDM MC | SAIRL vs highD global intensity | MC-subset relative diff |
|---|---:|---:|---:|---:|
| car-following | 72.056 | 65.180 | 9.135 | 7.976% |
| cut-in | 5.191 | 5.708 | 0.950 | 4.852% |

Validation:
- car-following: same EVT threshold = `True`; same subset N = `True`; same p0 = `True`; subset reliability = `pass`.
- cut-in: same EVT threshold = `True`; same subset N = `True`; same p0 = `True`; subset reliability = `pass`.

Conclusion: SAIRL is comparable under the same test pipeline, but this checkpoint/adapter does not reduce risk versus IDM in the current long-tail evaluation.
