# Following Paper Experiments

This directory contains post-processed car-following paper artifacts built from existing results only.
No following diffusion training, EVT fitting, or subset simulation rerun was performed.

## Inputs

- `event_scores`: `results/highd_events/following_event_scores.csv`
- `event_cache_summary`: `results/highd_events/following_event_cache_summary.json`
- `subset_summary`: `results/subset_simulation_following/latent_subset_summary.json`
- `subset_level_stats`: `results/subset_simulation_following/latent_subset_level_stats.csv`
- `subset_samples`: `results/subset_simulation_following/latent_subset_samples.npz`
- `subset_score_histograms`: `results/subset_simulation_following/figures/subset_score_histograms.png`
- `monte_carlo_summary`: `results/monte_carlo_following/latent_monte_carlo_summary.json`
- `naturalness_summary`: `results/diffusion_natural/following/naturalness_summary.json`
- `natural_ax_plot`: `results/diffusion_natural/following/natural_prior_plots/ax_distribution_real_vs_generated.png`
- `natural_jerk_plot`: `results/diffusion_natural/following/natural_prior_plots/jerk_distribution_real_vs_generated.png`
- `natural_interaction_plot`: `results/diffusion_natural/following/natural_prior_plots/phase_space_gap_delta_v.png`
- `evt_model`: `results/highd_following_tail/evt/longitudinal_peak_evt_model.json`
- `evt_summary`: `results/highd_following_tail/evt/longitudinal_peak_evt_summary.json`
- `evt_return_level_distance`: `results/highd_following_tail/exposure/figures/peak_evt_return_level_distance.png`
- `exposure_summary`: `results/highd_following_tail/exposure/highd_exposure_summary.json`

## Generated Artifacts

- `results/paper_experiments/following/tables/exp1_following_event_exposure_stats.csv`
- `results/paper_experiments/following/figures/exp1_following_y_long_hist_ccdf.png`
- `results/paper_experiments/following/tables/exp2_following_evt_params.csv`
- `results/paper_experiments/following/figures/exp2_following_evt_survival_curve.png`
- `results/paper_experiments/following/tables/exp3_following_naturalness_summary.csv`
- `results/paper_experiments/following/tables/exp4_following_subset_main_results.csv`
- `results/paper_experiments/following/figures/exp4_following_level_score_shift.png`
- `results/paper_experiments/following/tables/exp5_following_ads_vs_highd_intensity.csv`
- `results/paper_experiments/following/figures/exp5_following_ads_vs_highd_intensity.png`
- `results/paper_experiments/following/figures/exp5_following_ads_vs_highd_return_period.png`
- `results/paper_experiments/following/tables/exp6_following_sampling_ablation.csv`
- `results/paper_experiments/following/figures/exp6_following_sampling_efficiency.png`
- `results/paper_experiments/following/tables/exp7_following_risk_target_ablation.csv`
- `results/paper_experiments/following/tables/exp8_following_context_distribution_ablation.csv`
- `results/paper_experiments/following/tables/exp9_following_reliability_diagnostics.csv`
- `results/paper_experiments/following/figures/exp9_following_reliability_diagnostics.png`

## Reused Existing Artifacts

- reused existing artifact: `exp2_evt_params_and_curves: results/highd_following_tail/exposure/figures/peak_evt_return_level_distance.png`
- reused existing artifact: `exp3_naturalness: results/diffusion_natural/following/natural_prior_plots/ax_distribution_real_vs_generated.png`
- reused existing artifact: `exp3_naturalness: results/diffusion_natural/following/natural_prior_plots/jerk_distribution_real_vs_generated.png`
- reused existing artifact: `exp3_naturalness: results/diffusion_natural/following/natural_prior_plots/phase_space_gap_delta_v.png`
- reused existing artifact: `exp4_subset_main_results: results/subset_simulation_following/figures/subset_score_histograms.png`

## Skipped Artifacts

- None

## Interpretation Notes

- Main exposure denominator is `following_ego_miles`.
- All-vehicle exposure values are reported only as background.
- ADS intensity is `conditional exceedance probability x highD tail peak exposure rate`.
- The probabilities are conditional on the highD following tail scenario-condition distribution, not unconditional road crash rates.
