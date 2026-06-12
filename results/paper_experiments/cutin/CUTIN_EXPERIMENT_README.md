# Cut-in Paper Experiments

This directory contains post-processed cut-in paper artifacts built from existing results only.
No cut-in diffusion training, EVT fitting, or subset simulation rerun was performed.

## Inputs

- `event_scores`: `results/highd_events/cutin_event_scores.csv`
- `event_cache_summary`: `results/highd_events/cutin_event_cache_summary.json`
- `subset_summary`: `results/subset_simulation_cutin/latent_subset_summary.json`
- `subset_level_stats`: `results/subset_simulation_cutin/latent_subset_level_stats.csv`
- `subset_samples`: `results/subset_simulation_cutin/latent_subset_samples.npz`
- `subset_score_histograms`: `results/subset_simulation_cutin/figures/subset_score_histograms.png`
- `monte_carlo_summary`: `results/monte_carlo_cutin/latent_monte_carlo_summary.json`
- `naturalness_summary`: `results/diffusion_natural/cutin/naturalness_summary.json`
- `natural_ax_plot`: `results/diffusion_natural/cutin/natural_prior_plots/ax_distribution_real_vs_generated.png`
- `natural_lateral_accel_plot`: `results/diffusion_natural/cutin/natural_prior_plots/target_lateral_accel_distribution_real_vs_generated.png`
- `natural_trajectory_plot`: `results/diffusion_natural/cutin/natural_prior_plots/trajectory_reconstruction_errors.png`
- `natural_lateral_offset_plot`: `results/diffusion_natural/cutin/natural_prior_plots/lateral_offset_distribution_real_vs_generated.png`
- `evt_model`: `results/highd_cutin_tail/evt/cutin_peak_evt_model.json`
- `evt_summary`: `results/highd_cutin_tail/evt/cutin_peak_evt_summary.json`
- `exposure_summary`: `results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json`

## Generated Artifacts

- `results/paper_experiments/cutin/tables/exp1_cutin_event_exposure_stats.csv`
- `results/paper_experiments/cutin/figures/exp1_cutin_y_cutin_hist_ccdf.png`
- `results/paper_experiments/cutin/tables/exp2_cutin_evt_params.csv`
- `results/paper_experiments/cutin/figures/exp2_cutin_evt_survival_curve.png`
- `results/paper_experiments/cutin/figures/exp2_cutin_evt_return_level_curve.png`
- `results/paper_experiments/cutin/tables/exp3_cutin_naturalness_summary.csv`
- `results/paper_experiments/cutin/tables/exp4_cutin_subset_main_results.csv`
- `results/paper_experiments/cutin/figures/exp4_cutin_level_score_shift.png`
- `results/paper_experiments/cutin/tables/exp5_cutin_ads_vs_highd_intensity.csv`
- `results/paper_experiments/cutin/figures/exp5_cutin_ads_vs_highd_intensity.png`
- `results/paper_experiments/cutin/figures/exp5_cutin_ads_vs_highd_return_period.png`
- `results/paper_experiments/cutin/tables/exp6_cutin_sampling_ablation.csv`
- `results/paper_experiments/cutin/figures/exp6_cutin_sampling_efficiency.png`
- `results/paper_experiments/cutin/tables/exp7_cutin_risk_target_ablation.csv`
- `results/paper_experiments/cutin/tables/exp8_cutin_context_distribution_ablation.csv`
- `results/paper_experiments/cutin/tables/exp9_cutin_reliability_diagnostics.csv`
- `results/paper_experiments/cutin/figures/exp9_cutin_reliability_diagnostics.png`

## Reused Existing Artifacts

- reused existing artifact: `exp3_naturalness: results/diffusion_natural/cutin/natural_prior_plots/ax_distribution_real_vs_generated.png`
- reused existing artifact: `exp3_naturalness: results/diffusion_natural/cutin/natural_prior_plots/target_lateral_accel_distribution_real_vs_generated.png`
- reused existing artifact: `exp3_naturalness: results/diffusion_natural/cutin/natural_prior_plots/trajectory_reconstruction_errors.png`
- reused existing artifact: `exp3_naturalness: results/diffusion_natural/cutin/natural_prior_plots/lateral_offset_distribution_real_vs_generated.png`
- reused existing artifact: `exp4_subset_main_results: results/subset_simulation_cutin/figures/subset_score_histograms.png`

## Skipped Artifacts

- None

## Interpretation Notes

- Main exposure denominator is `all_vehicle_miles`.
- ADS intensity is `conditional exceedance probability x highD tail peak exposure rate`.
- The probabilities are conditional on the highD cutin tail scenario-condition distribution, not unconditional road crash rates.
