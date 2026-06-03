# subset：latent-space 子集模拟

`subset/` 在 diffusion latent 空间中估计被测车辆在 highD 长尾场景下的闭环高风险概率：

```text
context ~ Uniform(selected highD tail-feature contexts)
z ~ N(0, I)
actions = DDIM(context, z)
score = S_EVT(Y_sim)
```

同一 context 和同一 latent 会通过 deterministic sampler 生成同一条动作轨迹，便于在
latent 空间做 subset simulation。

## 评分口径

following 闭环轨迹先计算 `Y_long_sim`，再用 highD following peak EVT 模型映射为
`S_EVT(Y_long_sim)`。cut-in 使用 `Y_cutin_sim` 和 cut-in peak EVT 模型。该分数只表示
相对 highD tail EVT 参考分布的极端程度，不是 ADS 真实碰撞概率。

默认失效事件：

```text
Y_sim > x_c, x_c = 5.0
failure threshold = S_EVT(x_c)
```

`x_c` 是工程临界等级，用于横向比较同一 scoring 口径下的 ADS 与 highD 人类驾驶表现。

## 失效率估计

`subset/` 使用标准 subset simulation 估计失效率：当第 `level_idx` 层达到失效阈值或
达到最大层数时，

```text
P_hat = p0^level_idx * mean(score >= failure_threshold)
```

脚本会额外输出 final level 的 unique context/state、最大占比和 MH acceptance rate，
用于判断估计是否可靠；这些诊断只影响 `strict_probability_interpretation` 和
`probability_estimate_kind`，不改变估计公式。

## 长尾测试空间

`process_highD/scripts/select_tail_contexts.py` 默认：

```text
context_source = independent_tail_peaks
empirical_context_limit = null
context_generation_method = tail_feature_kde_knn
include_empirical_contexts = true
num_synthetic_contexts = 1000
```

这会先保留全部 declustered highD independent tail peaks，再生成 1000 个
tail-feature KDE/KNN 微扰 context。默认 following 输出为：

```text
7500 empirical highd_independent_tail_peak
1000 perturbed highd_tail_feature_kde_knn
```

因此默认估计的是平滑 highD tail-feature 分布上的条件失效概率，而不是只在有限
empirical peaks 上的离散平均。`tail_contexts.npz` 会保存微扰来源：

```text
synthetic_context, context_model_method, base_context_index,
base_event_id, context_feature_distance
```

## 推荐运行顺序

`run_latent_subset_simulation.py` 不带 `--config` 时默认运行 following；cut-in 必须显式传入
`subset/scripts/configs/latent_subset_cutin.yaml`。

following：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python diffusion/scripts/train_following_diffusion.py
conda run -n tread python process_highD/scripts/fit_following_peak_evt.py
conda run -n tread python process_highD/scripts/estimate_following_exposure.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py
conda run -n tread python subset/scripts/run_latent_subset_simulation.py
```

cut-in：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
conda run -n tread python diffusion/scripts/train_cutin_diffusion.py
conda run -n tread python process_highD/scripts/fit_cutin_peak_evt.py
conda run -n tread python process_highD/scripts/estimate_cutin_exposure.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py --scenario cut_in
conda run -n tread python subset/scripts/run_latent_subset_simulation.py \
  --config subset/scripts/configs/latent_subset_cutin.yaml
```

## 主要文件

```text
subset/scripts/configs/latent_subset_following.yaml
subset/scripts/configs/latent_subset_cutin.yaml
subset/scripts/run_latent_subset_simulation.py
subset/scripts/play_final_level_scenarios.py
subset/src/subset_simulation.py
subset/src/closed_loop_runner.py
subset/src/latent_evaluator.py
subset/src/frozen_diffusion_sampler.py
```

## 输出

```text
results/subset_simulation/latent_subset_summary.json
results/subset_simulation/latent_subset_level_stats.csv
results/subset_simulation/latent_subset_top_cases.json
results/subset_simulation/figures/subset_score_histograms.png
```

`latent_subset_summary.json` 记录概率估计、可靠性诊断、里程回报周期和 highD 人类驾驶
基线对比。大型结果文件属于可再生成产物。
