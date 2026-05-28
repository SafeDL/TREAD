# subset：latent-space 子集模拟

`subset/` 负责在 diffusion latent 空间中估计闭环高风险事件概率。它直接把
随机变量定义为 diffusion latent：

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

`y_long` 与 highD EVT 拟合使用同一公式，由 collision、near collision、
hard brake 和统一纵向 proxy 组成：

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

subset 的最终目标等级不是 pilot 分位数，而是 EVT return level。当前配置默认：

```text
F = {Y_long_sim > z50}
score threshold = S_EVT(z50)
```

subset 闭环风险不计算 RSS margin，也不把 RSS improper response 当作
adversarial objective。

## 概率估计模式

子集模拟每层会从超过阈值的 elite 样本启动下一层 Markov chains。默认
`estimator_mode: standard`，用于保持标准 subset simulation 的概率解释：

- elite 直接来自当前层经验 score tail，不按 context/state 多样性重排；
- 不启用 fresh above-threshold refresh；
- 不因链坍缩提前停止并报告概率；
- unique context/state 和 acceptance rate 只作为可靠性诊断输出。

`estimator_mode: guarded` 保留 diverse elite selection、fresh refresh 和
stop-on-collapse，用于工程排查或 demo 稳定性；这种输出会在
`latent_subset_summary.json` 中标记为 diagnostic estimate，不给严格概率估计
解释。

`context_refresh_prob` 表示 joint `(o, Z)` 空间中的独立先验 proposal 组件：
重新从 configured contexts 和标准正态 latent prior 采样候选状态。

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

`pilot_subset_threshold.py` 现在只是可选 pilot score diagnostic 脚本；它输出
pilot 分数分布分位数，不能定义 subset final failure event。主流程从
`paths.evt_model_path` 读取 EVT 模型，并按
`configs/latent_subset_simulation.yaml` 中的 `evt.return_period` 计算
`z_m` 和 `S_EVT(z_m)` 作为最终失效阈值。
当前默认 tail contexts 从全部有效 highD following events 的 tail 中抽样得到；
tail 的划分使用 EVT 拟合的 POT 阈值 `u`，即 `y_long > u`。为便于测试，
`select_tail_contexts.py` 现在默认按固定随机种子抽取 `500` 个 tail contexts；
如果将 `num_contexts` 配成 `0`，则使用全部 tail contexts。
`select_tail_contexts.py` 默认从 `extract_highd_events.py` 生成的
`following_event_contexts.npz` 读取 context 和 `y_long`，再用 EVT model 标定
`risk_score`；其输出 metadata 中的 return level 由脚本内
`evt_return_period` 配置指定，应和 subset 实验配置保持一致。缓存或 EVT
model 缺失会直接报错。
`latent_subset_summary.json` 会写出 `probability_target`、
`probability_estimate_kind` 和 `strict_probability_interpretation`，用于区分
标准 subset 概率估计、guarded diagnostic 估计和低可靠性估计。

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
每层阈值、估计模式、严格概率解释标记和最终诊断图路径。大型 NPZ 和逐样本评分表
属于可再生成产物，默认由 `.gitignore` 忽略。
