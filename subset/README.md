# subset：latent-space 子集模拟

`subset/` 负责在 diffusion latent 空间中估计闭环高风险事件概率。它不直接使用
KING 输出样本，而是把随机变量定义为 diffusion latent：

```text
z ~ N(0, I)
actions = DDIM(context, z)
score = S_EVT(Y_long_sim)
```

因此，同一 context 和同一 latent 会通过 DDIM deterministic sampler 映射为同一条
自然驾驶动作轨迹，便于在 latent 空间中执行 subset simulation。

## subset 安全评分

`subset/` 用 `utils/risk.py` 评估生成的 200 帧闭环事件轨迹本身。先计算原始
纵向风险变量 `y_long`，再用 highD EVT 模型映射为
`risk_score = S_EVT(y_long)`。该分数用于 subset simulation 的中间阈值和
失效概率估计，不是 adversarial before/after 优化目标。

`y_long` 与 highD EVT 拟合、adversaray 闭环验证一致，由
collision、near collision、hard brake 和统一纵向 proxy 组成：

```text
longitudinal_proxy =
  w_ttc * softmax_pool(1/TTC)
+ w_thw * softmax_pool(1/THW)
+ w_gap * softmax_pool(1/gap)
+ w_drac * softmax_pool(DRAC)

y_long =
  collision_bonus * collision
+ near_collision_weight * near_collision
+ longitudinal_proxy
+ hard_brake_weight * hard_brake
```

subset 的最终目标等级不是 pilot 分位数，而是 EVT return level，默认：

```text
F = {Y_long_sim > z100}
score threshold = S_EVT(z100)
```

subset 闭环风险不计算 RSS margin，也不把 RSS improper response 当作
adversarial objective。

## 防止链坍缩

子集模拟每层会从超过阈值的 elite 样本启动下一层 Markov chains。为了避免高分
elite 被拒绝 proposal 反复复制成同一个场景，当前实现会：

- 对 elite 按 `(context, latent)` 去重，并优先覆盖不同 context；
- 对每个输出样本进行有限次数的 MH proposal 重试；
- MH 重试失败后尝试从先验独立刷新 context/latent；
- 如果下一层的 unique context 或 unique state 低于配置下限，则提前停止，
  不写出坍缩层。

相关参数位于 `subset_simulation` 配置块：
`mh_retries_per_sample`、`refresh_attempts_per_sample`、
`min_next_unique_contexts`、`min_next_unique_states` 和
`stop_on_collapse`。

脚本还会在日志和 `latent_subset_summary.json` 中输出估计可靠性诊断。默认检查
最后使用层的 unique context/state 数量、最大单个 context/state 占比和 MH
接受率；这些阈值只用于判断当前估计是否可报告，不会改变采样过程。demo 规模
下如果诊断为 `fail`，表示代码已跑通但概率值不应作为稳定估计解读。

## 主要文件

```text
subset/
├── scripts/
│   ├── configs/latent_subset_simulation.yaml
│   ├── pilot_subset_threshold.py
│   └── run_latent_subset_simulation.py
└── src/
    ├── latent_evaluator.py
    ├── subset_simulation.py
    ├── closed_loop_runner.py
    └── frozen_diffusion_sampler.py
```

`subset/` 不保留历史兼容 wrapper。context、归一化、diffusion adapter 和 IO
逻辑均直接从根目录 `utils/` 引入。

## 推荐运行顺序

所有命令默认从仓库根目录运行：

```bash
conda run -n tread python process_highD/scripts/fit_longitudinal_evt.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py
conda run -n tread python subset/scripts/run_latent_subset_simulation.py
```

`pilot_subset_threshold.py` 现在只是可选诊断脚本；主流程从
`paths.evt_model_path` 读取 `z100` 并使用 `S_EVT(z100)` 作为最终失效阈值。

运行前应已经完成：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python diffusion/scripts/train_natural_diffusion.py
```

## 输出文件

默认输出目录：

```text
results/subset_simulation/
```

主要产物：

```text
results/highd_tail_contexts/following/tail_contexts.npz
results/highd_tail_contexts/following/tail_scores.npz
results/highd_evt/following/longitudinal_evt_model.json
results/subset_simulation/
latent_subset_summary.json
latent_subset_level_stats.csv
latent_subset_samples.npz
figures/subset_score_histograms.png
figures/subset_threshold_progression.png
figures/subset_final_context_usage.png
```

`latent_subset_summary.json` 会记录失效概率估计、标准误、置信区间、相对误差、
每层阈值和最终诊断图路径。大型 NPZ 和逐样本评分表属于可再生成产物，默认由
`.gitignore` 忽略。
