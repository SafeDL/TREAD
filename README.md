# TREAD

TREAD 是一个基于 highD 自然驾驶数据的长尾安全测试实验工程。当前主线覆盖
following 和 cut-in 两类事件，把 highD 事件抽取、自然动作 diffusion prior、
peak-level EVT 标定和 latent-space subset simulation 串成可复现实验链路。

```text
process_highD/   highD 事件抽取、风险缓存、EVT 拟合、exposure 和 tail context 构造
diffusion/       训练 highD following/cut-in 场景的自然动作扩散先验
subset/          在 context + diffusion latent 空间中执行闭环 subset simulation
utils/           共享 IO、风险评分、EVT、context、exposure 和 RSS 基础函数
```

`ad-rss-lib/` 不是当前主流程依赖；代码中使用的是本仓库 `utils/rss.py` 和
`utils/risk.py` 里的本地实现。

## 运行环境

默认从仓库根目录运行：

```bash
conda activate tread
```

主流程脚本基本不暴露 CLI 参数，实验默认值写在脚本常量或配置文件中：

```text
process_highD/scripts/configs/highd_default.yaml
diffusion/scripts/configs/natural_following.yaml
diffusion/scripts/configs/natural_cutin.yaml
subset/scripts/configs/latent_subset_following.yaml
subset/scripts/configs/latent_subset_cutin.yaml
```

highD 原始 CSV 默认读取：

```text
highD_dataset/Matlab/data/
```

即包含 `XX_tracks.csv`、`XX_tracksMeta.csv`、`XX_recordingMeta.csv` 的目录。

## 风险口径

following 使用纵向风险变量 `Y_long`。它由 `TTC`、`THW`、`gap`、`DRAC` 的
softmax pooling 项组成，并叠加 collision、near-collision 和 hard-brake 项：

```text
longitudinal_proxy =
  w_ttc * softmax_pool(1 / TTC)
+ w_thw * softmax_pool(1 / THW)
+ w_gap * softmax_pool(1 / gap)
+ w_drac * softmax_pool(DRAC)

Y_long =
  longitudinal_proxy
+ collision_bonus * collision
+ near_collision_weight * near_collision
+ hard_brake_weight * hard_brake
```

cut-in 使用统一的 `Y_cutin`。语义上先要求目标车横向进入 ego 车道并保持在目标车道，
且 `require_front_at_cutin: true` 时，切入时刻目标车必须位于 ego 前方；不满足该
cut-in 门控的轨迹默认记为 `Y_cutin = 0`。通过门控后，风险由切入时 THW/LTG、
post-cutin 窗口内 TTC、DRAC、安全距离缺口，以及 collision、near-collision、
hard-brake 项组成。EVT 在 decluster 后的 following `Y_long` peaks 和 cut-in
`Y_cutin` peaks 上分别拟合 POT/GPD。闭环仿真的输出分数为：

```text
risk_score = S_EVT(Y_sim) = -log P_EVT(Y > Y_sim)
```

这个分数表示相对 highD 自然 peak 尾部分布的极端程度，不是 ADS 碰撞概率。

当前 subset 默认失效目标为：

```text
Y_sim > x_c,  x_c = 5.0
failure_threshold = S_EVT(x_c)
```

该目标由 `subset/scripts/configs/latent_subset_following.yaml` 或
`subset/scripts/configs/latent_subset_cutin.yaml` 中的 `evt` 配置决定。当前 cut-in
默认使用 `collision_critical_level: 10.0`。

## 数据与默认过滤

`process_highD/scripts/extract_highd_events.py` 抽取 following 和 cut-in 事件，并
同步写出风险缓存和 context 缓存。默认配置中：

- 采样频率为 `25 Hz`。
- 事件窗口长度为 `128` 帧。
- following 需要同一 preceding vehicle 持续 `128` 帧。
- cut-in 原始事件筛选要求目标车完成相邻车道切入、cross 后多数时间在 ego 前方且与
  ego 同车道，并默认要求目标车是 ego 在目标车道的最近前车；后续风险评分还会再次
  用 `is_cutin` 和 `is_front_cutin` 门控长尾风险。
- 默认要求 ego 和 lead 都是 passenger car。
- diffusion 数据集按 recording id 划分 `train / val / test = 0.70 / 0.15 / 0.15`，
  随机种子为 `42`，避免同一 recording 的窗口跨 split 泄漏。

`diffusion/` 只学习自然动作分布，不使用安全分数作为训练目标。following 默认动作
表示为 lead vehicle jerk；cut-in 默认动作表示为 target vehicle `[ax, ay]`。两条
链路都使用 anchor-frame `scenario_conditions` 作为唯一 diffusion 条件，该向量包含
初始关系和 125 步参考窗口的压缩动作/轨迹摘要；不再输入 rolling `context_states`、
`context_features` 或 `relative_history`。训练目标为 DDPM noise prediction，推理和
subset 中使用 DDIM deterministic sampling：

```text
same scenario condition + same latent z -> same action trajectory
```

## Tail Contexts

following 和 cut-in 的 tail context 构建入口按场景拆开：

```text
following: context_source = independent_tail_peaks
cut_in:    context_source = independent_tail_peaks
context_generation_method = tail_feature_kde_knn
include_empirical_contexts = true
empirical_context_limit = null
num_synthetic_contexts = 500
```

含义是：

1. following 从 `highd_independent_tail_peaks.csv` 中取 decluster 后的
   highD independent tail peaks。
2. cut-in 从 `highd_independent_tail_peaks.csv` 中取 decluster 后的
   highD independent tail peaks。
3. `empirical_context_limit = null` 表示保留全部 matched empirical tail
   contexts；若设为正整数，则先无放回抽取对应数量的 empirical contexts。
4. 默认额外生成 `500` 个 synthetic contexts。生成方式是在低维 tail feature
   空间中做 KDE 式扰动，并用最近邻 empirical context 重构 `scenario_conditions`
   和 `initial_states`。
5. 输出中用 `source_type`、`synthetic_context`、`context_model_method`、
   `base_event_id` 和 `context_feature_distance` 区分 empirical 与 synthetic
   contexts。

因此 subset 默认估计的是：

```text
P_context,z(Y_sim > x_c | context sampled from highD tail-feature distribution)
```

而不是严格的：

```text
P_context,z(Y_sim > x_c | context in finite empirical highD tail peaks)
```

这样做的目的，是缓解纯 empirical 少量失效样本导致的 Markov chain 坍缩；解释结果时
应把 context 分布理解为 highD tail feature 分布的平滑近似。若需要最干净的经验分布
解释，可把 `context_generation_method` 改为 `empirical`，并保留
`empirical_context_limit = null`。

cut-in 使用同一套 tail-feature 微扰机制，但 feature 空间按 cut-in scenario condition
单独定义。
subset 中的 cut-in 重建默认从 tail context 的 `initial_states` 开始，ego 车按初始速度
作为 IDM 目标速度正常响应，target 车由 diffusion prior 一次性生成 125 步 `[ax, ay]`。
following 同样从 `initial_states` 开始，由 diffusion prior 一次性生成 125 步 lead jerk。

## 推荐运行顺序

1. 抽取 highD 事件、following 风险缓存和 exposure per recording：

```bash
python process_highD/scripts/extract_highd_events.py
```

2. 构建 diffusion 数据集、训练自然先验、评估自然性：

```bash
python process_highD/scripts/build_natural_dataset.py
python diffusion/scripts/train_following_diffusion.py
python diffusion/scripts/evaluate_following_prior.py
```

3. 拟合 highD following peak EVT、估计 exposure、构造 tail contexts：

```bash
python process_highD/scripts/fit_following_peak_evt.py
python process_highD/scripts/estimate_following_exposure.py
python process_highD/scripts/select_following_tail_contexts.py
```

4. 执行 latent-space subset simulation：

```bash
python subset/scripts/run_latent_subset_following.py
```

cut-in 分支：

```bash
python process_highD/scripts/extract_highd_events.py
python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
python diffusion/scripts/train_cutin_diffusion.py
python process_highD/scripts/estimate_cutin_exposure.py
python process_highD/scripts/select_cutin_tail_contexts.py
python subset/scripts/run_latent_subset_cutin.py
```

可选回放：

```bash
python process_highD/scripts/play_following_tail_events.py
python process_highD/scripts/play_cutin_tail_events.py
python diffusion/scripts/sample_natural_rollouts.py
python subset/scripts/play_final_level_following.py
python subset/scripts/play_final_level_cutin.py
```

`play_following_tail_events.py` 和 `play_cutin_tail_events.py` 不暴露 CLI 配置，
分别回放 following/cut-in tail contexts 对应的 highD 完整事件并输出 GIF。
`TAIL_CONTEXT_SELECTION = "all"` 表示全部场景；设为整数表示按
`RANDOM_SEED` 随机采样这么多个场景；设为 tuple/list 表示精确指定 context index。
驾驶事件按各自 `start_frame/end_frame` 播放，不假定固定长度。共享回放逻辑位于
`process_highD/src/event_playback.py`。

## 主要输出

```text
results/highd_events/events.csv
results/highd_events/following_event_scores.csv
results/highd_events/following_event_contexts.npz
results/highd_events/cutin_event_scores.csv
results/highd_events/cutin_event_contexts.npz
results/highd_events/following_event_cache_summary.json
results/highd_events/exposure_per_recording.csv

results/diffusion_natural/following/dataset.npz
results/diffusion_natural/following/dataset_normalized.npz
results/diffusion_natural/following/feature_schema.json
results/diffusion_natural/following/normalization_stats.json
results/diffusion_natural/following/train_val_test_split.json
results/diffusion_natural/following/checkpoints/best_noise_mse.pt
results/diffusion_natural/following/training_summary.json
results/diffusion_natural/following/tensorboard/
results/diffusion_natural/following/naturalness_summary.json

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
results/highd_cutin_tail/contexts/tail_contexts.npz
results/highd_cutin_tail/contexts/tail_context_summary.json

results/subset_simulation/latent_subset_summary.json
results/subset_simulation/latent_subset_level_stats.csv
results/subset_simulation/latent_subset_top_cases.json
results/subset_simulation/figures/
```

`latent_subset_summary.json` 中最重要的字段是：

- `probability`: subset 估计的条件失效概率。
- `probability_target`: 该概率对应的 context 分布解释。
- `probability_estimate_kind`: 标准估计、低可靠性标准估计或 guarded 诊断估计。
- `reliability`: final level 的 unique context/state、最大占比和 acceptance rate 诊断。
- `mileage_return_period`: 在 strictness 条件满足时，把条件概率乘以 highD tail peak
  exposure rate 后得到的里程或时间回报周期。

## 子集模拟可靠性

`subset` 使用标准 subset simulation 概率估计，并同时输出可靠性诊断。如果 final
level 的 unique context/state 太少、最大 context/state 占比过高，或 MH acceptance
rate 过低，`strict_probability_interpretation` 会变为 `false`。

## 版本控制约定

大型模型、数组、逐样本结果和可再生成文件应由 `.gitignore` 忽略。README 中列出的
结果路径是运行产物说明，不表示这些文件都应该提交。
