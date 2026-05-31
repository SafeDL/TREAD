# subset：latent-space 子集模拟

`subset/` 在 diffusion latent 空间中估计被测车辆在 highD 长尾跟驰场景下的闭环高风险概率：

```text
context ~ Uniform(all selected highD tail contexts)
z ~ N(0, I)
actions = DDIM(context, z)
score = S_EVT(Y_long_sim)
```

同一 context 和同一 latent 会通过 deterministic sampler 生成同一条动作轨迹，便于在 latent 空间做 subset simulation。

## 评分口径

闭环轨迹先计算统一纵向风险变量 `Y_long_sim`，再用 highD peak EVT 模型映射为 `S_EVT(Y_long_sim)`。该分数只表示相对 highD tail EVT 参考分布的极端程度，不是 ADS 真实碰撞概率。

当前默认失效事件：

```text
Y_long_sim > x_c, x_c = 5.0
failure threshold = S_EVT(x_c)
```

`x_c` 是工程临界等级，用于横向比较同一 scoring 口径下的 ADS 与 highD 人类驾驶表现。

## 估计模式

默认使用 `estimator_mode: standard`，保持标准 subset simulation 概率解释。脚本会额外输出 final level 的 unique context/state、最大占比和 MH acceptance rate，用于判断估计是否可靠。

`estimator_mode: guarded` 只用于链坍缩诊断或演示，不作为严格概率估计。

## 全量长尾测试空间

`process_highD/scripts/select_tail_contexts.py` 默认：

```python
"context_source": "independent_tail_peaks"
"num_contexts": 0
```

这会把全部 decluster 后的 highD independent tail peaks 写入：

```text
results/highd_following_tail/contexts/tail_contexts.npz
```

`subset/scripts/configs/latent_subset_simulation.yaml` 默认读取该文件，并在这些 context 上均匀采样。

## 推荐运行顺序

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python diffusion/scripts/train_natural_diffusion.py
conda run -n tread python process_highD/scripts/fit_longitudinal_peak_evt.py
conda run -n tread python process_highD/scripts/estimate_highd_exposure.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py
conda run -n tread python subset/scripts/run_latent_subset_simulation.py
```

## 主要文件

```text
subset/scripts/configs/latent_subset_simulation.yaml
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
results/subset_simulation/latent_subset_samples.npz
results/subset_simulation/latent_subset_top_cases.json
results/subset_simulation/figures/subset_score_histograms.png
```

`latent_subset_summary.json` 记录概率估计、可靠性诊断、里程回报周期和 highD 人类驾驶基线对比。大型结果文件属于可再生成产物。
