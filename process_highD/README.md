# process_highD：highD 事件与长尾空间

`process_highD/` 从 highD 原始 CSV 生成 following/cut-in 事件，缓存风险与
context，拟合 peak-level EVT，并输出 subset simulation 使用的长尾 context 空间。

## 运行顺序

following 主流程：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python diffusion/scripts/train_following_diffusion.py
conda run -n tread python process_highD/scripts/estimate_following_exposure.py
conda run -n tread python process_highD/scripts/select_following_tail_contexts.py
```

`build_natural_dataset.py` 不带参数时默认构建 following diffusion 数据集。训练入口按
事件类型拆分；cut-in 分支需要先生成 cut-in score/context cache，再显式使用 cut-in
配置构建数据集：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
conda run -n tread python diffusion/scripts/train_cutin_diffusion.py
conda run -n tread python process_highD/scripts/estimate_cutin_exposure.py
conda run -n tread python process_highD/scripts/select_cutin_tail_contexts.py
```

可选回放：

```bash
conda run -n tread python process_highD/scripts/play_following_tail_events.py
conda run -n tread python process_highD/scripts/play_cutin_tail_events.py
```

两个回放脚本分别配置 following/cut-in 的 diffusion generated scenario 路径、输出路径和
`SCENARIO_SELECTION`。共享 GIF 渲染与 highD 事件反查逻辑在
`process_highD/src/event_playback.py`。

## 主要文件

```text
process_highD/scripts/configs/highd_default.yaml
process_highD/scripts/extract_highd_events.py
process_highD/scripts/build_natural_dataset.py
process_highD/scripts/estimate_following_exposure.py
process_highD/scripts/estimate_cutin_exposure.py
process_highD/scripts/select_following_tail_contexts.py
process_highD/scripts/select_cutin_tail_contexts.py
process_highD/scripts/play_following_tail_events.py
process_highD/scripts/play_cutin_tail_events.py
process_highD/src/context_selection.py
process_highD/src/event_playback.py
process_highD/src/event_extraction.py
process_highD/src/loader.py
process_highD/src/preprocess.py
```

## 主要输出

```text
results/highd_events/events.csv
results/highd_events/following_event_scores.csv
results/highd_events/following_event_contexts.npz
results/highd_events/cutin_event_scores.csv
results/highd_events/cutin_event_contexts.npz

results/highd_following_tail/evt/longitudinal_peak_evt_model.json
results/highd_following_tail/evt/longitudinal_peak_evt_summary.json
results/highd_following_tail/exposure/highd_exposure_summary.json
results/highd_following_tail/exposure/highd_independent_tail_peaks.csv
results/highd_following_tail/contexts/tail_contexts.npz
results/highd_following_tail/contexts/tail_context_summary.json

results/highd_cutin_tail/evt/cutin_peak_evt_model.json
results/highd_cutin_tail/evt/cutin_peak_evt_summary.json
results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json
results/highd_cutin_tail/exposure/highd_independent_tail_peaks.csv
results/highd_cutin_tail/contexts/scenario_condition_distribution.npz
results/highd_cutin_tail/contexts/scenario_condition_distribution_summary.json
results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz
results/highd_cutin_tail/generated/figures/
```

cut-in 输出只有在 cut-in cache 和 EVT 分支运行后才会出现。

## 风险口径

- `extract_highd_events.py` 只做自然事件抽取和质量过滤，不用风险分数筛选候选事件。
- `estimate_following_exposure.py` 在 decluster 后的 following `Y_long` peaks 上拟合 POT/GPD，并计算 following exposure、tail peak rate 和 highD 人类驾驶基线。
- `estimate_cutin_exposure.py` 在 decluster 后的 cut-in `Y_cutin` peaks 上拟合 POT/GPD，并使用 highD 全车辆里程/时长计算 cut-in tail peak rate 和人类驾驶基线。

following 的统一风险尺度是 `S_EVT(Y_long)`；cut-in 的统一风险尺度是
`S_EVT(Y_cutin)`。

## Tail Contexts

following tail context 入口默认在全部 declustered tail peaks 上构造平滑长尾测试空间：

```text
context_source = independent_tail_peaks
empirical_context_limit = null
context_generation_method = tail_feature_kde_knn
include_empirical_contexts = true
num_synthetic_contexts = 5000
```

following 输出包含两类 context：

```text
highd_independent_tail_peak    empirical highD independent tail peak context
highd_tail_feature_kde_knn     tail feature 微弱扰动后的重构 context
```

cut-in 使用专用流程：

```text
EVT declustered independent tail peaks
-> Gaussian copula over diffusion scenario_conditions
-> 20000 sampled cut-in conditions
-> pretrained cut-in diffusion prior
-> 20000 generated cut-in scenarios
```

低维 tail feature 对齐 diffusion 的 `scenario_conditions`。following 使用：

```text
ego_vx_0, log_initial_gap, initial_delta_v, lead_ax_0,
lead_speed_change, lead_min_ax, lead_braking_duration
```

cut-in 使用：

```text
ego_vx_0, log_initial_gap, initial_lateral_offset, initial_delta_vx,
target_vy_0, target_ay_0, target_lateral_displacement
```

`contexts/` 只保存这些条件变量的联合概率分布；20000 个采样条件和扩散生成轨迹保存在
`generated/`。生成轨迹积分需要的 `initial_states` 从采样条件最近邻的 highD tail
事件重构，不作为扩散模型条件输入。

cut-in 结果同时输出：

```text
scenario_condition_distribution.npz
scenario_condition_distribution_summary.json
diffusion_generated_scenarios.npz
semantic_histograms_tail_vs_generated.png
scenario_start_speed_tail_vs_generated.png
scenario_end_speed_tail_vs_generated.png
distribution_similarity_summary.json
```
