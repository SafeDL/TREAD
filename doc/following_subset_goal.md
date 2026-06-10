# Car-following 子集模拟 Goal 设计文档

## Goal

`/goal` 设计并完善 TREAD 工程中 car-following 长尾测试空间上的
latent-space subset simulation，使其能够在已获取的 highD following 长尾
scenario-condition 联合分布与扩散模型随机正态潜在空间上，估计 IDM 控制的
highway-env 闭环 ego 在 car-following 场景中的安全关键失效概率。

代码运行系统的 conda 环境：

```bash
conda activate tread
```

最终状态应满足：

- 当前阶段的主目标不是单纯跑通 subset simulation，而是得到可严格解释的可靠估计：
  `latent_subset_summary.json.reliability.status = pass`，且
  `strict_probability_interpretation = true`；
- 在 Monte Carlo 分辨率足够时，subset 与 Monte Carlo 必须估计同一个目标概率且统计相容：
  `latent_mc_subset_comparison.json.status = pass`。`mc_resolution_insufficient` 只能说明
  当前 MC 样本数不足以验证接近性，不能作为“MC/subset 已接近”的最终证据；

- 测试空间明确定义为：

```text
c ~ highD following tail scenario-condition Gaussian-copula distribution
z ~ N(0, I)
lead_actions = DDIM(c, z)
ego_response = highway-env IDM closed-loop rollout
score = S_EVT(Y_long_sim)
```

- following horizon 固定为 125 帧，采样频率为 25 Hz，扩散 latent 形状为
  `[125, 1]`，动作语义保持为 lead vehicle longitudinal `jx`；
- following 扩散初始噪声空间为 125 维标准正态，即 `z.shape = [125, 1]`；
  加上 7 维 scenario conditions 后，联合测试输入空间可按 132 个标量维度理解；
- `subset/scripts/run_subset_following.py` 能读取 `process_highD/` 输出的
  `tail_contexts.npz` 与 `scenario_condition_distribution.npz`，而不是重新拟合
  tail distribution；
- subset 估计结果能输出标准概率、层级阈值、final-level 诊断、top cases、样本缓存和
  playback 入口；
- `subset/scripts/` 下必须提供同分布 Monte Carlo 基线入口。Monte Carlo 与 subset 使用完全相同的
  scenario-condition 联合分布、扩散 latent 先验、DDIM sampler、IDM ego 和 following EVT
  评分口径，只是不做自适应分层和 MH 条件采样；
- `subset/scripts/compare_estimators.py` 必须能通过 following config 自动验证
  following Monte Carlo 与 subset 估计是否统计相容；
- subset 运行结束时必须在日志中打印实际纳入闭环仿真计算的场景数量，包括 closed-loop
  evaluator 调用次数和唯一 scenario context 数；
- `latent_subset_summary.json` 中的概率解释必须对应：

```text
P_context,z(Y_long_sim > x_c | o sampled from highD tail scenario-condition distribution)
```

- 若可靠性诊断不通过，结果必须被标记为 low-reliability estimate，只能作为诊断结果；
  主目标要求继续调参或增大样本数，直到 final-level reliability pass。

---

## 概率目标与失效定义

### 1. 联合测试空间

following subset simulation 的随机变量由两部分组成：

```math
\omega = (\mathbf{c}, \mathbf{z})
```

其中：

- `scenario_conditions` $\mathbf{c}$ 来自
  `results/highd_following_tail/contexts/scenario_condition_distribution.npz`
  中保存的 Gaussian copula 模型；
- `z` 为扩散模型确定性 DDIM 采样的初始噪声，服从标准正态分布；
- `initial_states` 不直接参与 denoiser 条件输入，只由 sampled condition 和最近邻
  empirical highD tail context 重构，供 lead 轨迹积分和闭环仿真使用。

following 条件特征顺序必须与 `diffusion/src/features.py` 中
`FOLLOWING_SCENARIO_CONDITION_KEYS` 完全一致：

```text
ego_vx_0
initial_gap
initial_delta_v
lead_ax_0
lead_speed_change
lead_min_ax
lead_braking_duration
```

### 2. 闭环失效事件

对每个样本：

```math
\mathbf{a} = \mathcal{D}(\mathbf{c}, \mathbf{z}; \theta)
```

其中 $\mathcal{D}$ 为冻结的 following diffusion prior 与 deterministic DDIM sampler。
lead vehicle 执行生成的 `jx` 计划；ego vehicle 由 `tools/idm_ego.yaml` 中的 IDM 参数在
highway-env 中闭环响应。闭环轨迹通过 `tools/risk.py`/`tools/highd_longitudinal.py`
的 longitudinal 风险口径得到 `Y_long_sim`，再由 highD following peak EVT 模型映射为：

```math
S_{\mathrm{EVT}}(Y_{\mathrm{long,sim}})
= -\log P_{\mathrm{EVT}}(Y_{\mathrm{long}} > Y_{\mathrm{long,sim}})
```

默认失效事件为：

```text
Y_long_sim > x_c
x_c = evt.collision_critical_level = 5.0
failure_threshold = S_EVT(x_c)
```

如果使用 return-period 目标，必须在配置和结果摘要中明确写出对应的 `return_period`
和 `return_level_target`。

---

## 由以下测试或数据证据验证

### 1. 重新运行 following 主流程

完整验证顺序：

```bash
python process_highD/scripts/extract_highd_events.py
python process_highD/scripts/build_natural_dataset.py --config diffusion/scripts/configs/natural_following.yaml
python diffusion/scripts/train_following_diffusion.py
python diffusion/scripts/evaluate_following_prior.py
python process_highD/scripts/estimate_following_exposure.py
python process_highD/scripts/select_following_tail_contexts.py
python subset/scripts/run_monte_carlo_following.py
python subset/scripts/run_subset_following.py
python subset/scripts/compare_estimators.py --config subset/scripts/configs/latent_subset_following.yaml
python subset/scripts/play_final_level_following.py
```

如果只验证 subset 设计和接口，可在已有 diffusion checkpoint、EVT 模型和 tail context
均存在时运行：

```bash
python subset/scripts/run_subset_following.py
python subset/scripts/run_monte_carlo_following.py
python subset/scripts/compare_estimators.py --config subset/scripts/configs/latent_subset_following.yaml
python subset/scripts/play_final_level_following.py --no-gif
```

### 2. 检查 subset 输入文件

检查：

```text
results/diffusion_natural/following/feature_schema.json
results/diffusion_natural/following/normalization_stats.json
results/diffusion_natural/following/checkpoints/best_noise_mse_all_train.pt
results/highd_following_tail/contexts/tail_contexts.npz
results/highd_following_tail/contexts/scenario_condition_distribution.npz
results/highd_following_tail/evt/longitudinal_peak_evt_model.json
results/highd_following_tail/exposure/highd_exposure_summary.json
tools/idm_ego.yaml
```

验证：

- `feature_schema.json` 中 `event_type` 必须为 `following`，`model_input_keys`
  只能包含 `scenario_conditions`；
- diffusion checkpoint 优先使用 `best_noise_mse_all_train.pt`；如果 fallback 到
  `best_noise_mse_train_val_test.pt`，必须在结果说明中标注；
- `tail_contexts.npz` 必须包含 empirical independent tail peak rows，source type 为
  `highd_independent_tail_peak`；
- `scenario_condition_distribution.npz` 必须包含 `copula_correlation`、
  `copula_variable_mask` 和 `copula_marginal_values`；
- `copula_marginal_values` 的维度必须为 7，并且与 following condition keys 对齐；
- `initial_states` 的形状必须为 `[2, 6]`，表示 ego 与 lead 的
  `[x, y, vx, vy, ax, ay]`。

### 3. 检查 subset 输出

检查：

```text
results/subset_simulation_following/latent_subset_summary.json
results/subset_simulation_following/latent_subset_level_stats.csv
results/subset_simulation_following/latent_subset_top_cases.json
results/subset_simulation_following/latent_subset_samples.npz
results/subset_simulation_following/latent_mc_subset_comparison.json
results/subset_simulation_following/latent_mc_subset_comparison.csv
results/subset_simulation_following/figures/subset_score_histograms.png
results/subset_simulation_following/figures/final_level_playbacks/
results/monte_carlo_following/latent_monte_carlo_summary.json
results/monte_carlo_following/latent_monte_carlo_stats.csv
results/monte_carlo_following/latent_monte_carlo_top_cases.json
results/monte_carlo_following/latent_monte_carlo_samples.npz
```

验证：

- `probability` 有限且非负；
- `probability_target` 明确包含 highD tail scenario-condition distribution；
- `failure_event` 必须对应 `Y_long_sim > x_c` 或明确的 return-level 目标；
- `score_space` 为 `evt` 时，`failure_threshold` 必须等于 following EVT 模型对目标
  `Y_long` 的 score；
- `thresholds`、`acceptance_rates`、`level_stats` 必须存在；
- `input_space` 必须存在，且 following 下 `diffusion_noise_shape = [125, 1]`、
  `diffusion_noise_dimension = 125`、`scenario_condition_dimension = 7`、
  `joint_condition_noise_dimension = 132`；
- `simulation_counts` 必须存在，至少包含 `closed_loop_evaluations`、
  `stored_level_samples`、`unique_context_indices_all_levels`、
  `unique_context_indices_final_level` 和 `available_context_population`；
- `latent_subset_samples.npz` 必须保存每层 `context_indices`、`latents`、`scores`、
  `actions`、`y_long`、`min_gap`、`min_ttc`、`physical_feasible` 等关键字段；
- `actions` 的最后一维必须为 1，对应 lead vehicle longitudinal `jx`，不能被解释为
  ego 动作。

### 4. 检查 Monte Carlo 基线和 subset 对比

检查：

```text
results/monte_carlo_following/latent_monte_carlo_summary.json
results/monte_carlo_following/latent_monte_carlo_stats.csv
results/monte_carlo_following/latent_monte_carlo_top_cases.json
results/monte_carlo_following/latent_monte_carlo_samples.npz
results/subset_simulation_following/latent_mc_subset_comparison.json
```

验证：

- Monte Carlo 的 sampling space 必须与 subset 完全一致：

```text
c ~ highD following tail scenario-condition Gaussian-copula distribution
z ~ N(0, I)
actions = DDIM(c, z)
score = S_EVT(Y_long_sim)
```

- `latent_monte_carlo_summary.json` 中 `estimator` 必须为 `independent_monte_carlo`；
- `event_type` 必须为 `following`；
- `probability_target` 必须说明样本来自 highD tail scenario-condition distribution；
- `probability` 必须等于 `mean(score >= failure_threshold)`；
- `input_space` 必须与 subset 输出一致；
- `simulation_counts` 必须存在，至少包含 `closed_loop_evaluations`、
  `stored_samples`、`unique_context_indices` 和 `available_context_population`；
- `compare_estimators.py --config subset/scripts/configs/latent_subset_following.yaml`
  必须输出 `latent_mc_subset_comparison.json`，并记录输入一致性、阈值一致性、
  概率差值、combined standard error 和 CI overlap；
- 若 MC failure count 小于 10 或 relative SE 大于 0.5，比较状态应为
  `mc_resolution_insufficient`，不能把 MC 用作强一致性证据，也不能作为
  “MC/subset 已接近”的最终验收结论。
- `goal_closeness_requirement_satisfied` 只有在 comparison `status = pass` 时才能为 true；
  `diagnostic_workflow_completed` 可以在 `pass` 或 `mc_resolution_insufficient` 时为 true，
  但后者只代表诊断流程完成，不代表概率接近性已经验证。

Monte Carlo 和 subset 估计的是同一个目标概率。记：

```text
p_mc = latent_monte_carlo_summary.json.probability
se_mc = latent_monte_carlo_summary.json.probability_standard_error
p_ss = latent_subset_summary.json.probability
se_ss = latent_subset_summary.json.probability_standard_error
```

当 MC failure count 不少于 10，且 `se_mc / max(p_mc, 1e-12) <= 0.5` 时，必须满足以下至少一项：

```math
|p_{\mathrm{ss}} - p_{\mathrm{mc}}|
\le 2\sqrt{\mathrm{se}_{\mathrm{ss}}^2 + \mathrm{se}_{\mathrm{mc}}^2}
```

或两者 95% confidence interval 有重叠。若不满足，应优先排查：

- MC 与 subset 是否使用同一 `tail_context_path`、`condition_distribution_path`、
  `diffusion_checkpoint`、`evt_model_path` 和 `idm_ego_config_path`；
- 两者的 `failure_threshold`、`evt_return_level_target`、`score_space` 和
  `failure_event` 是否完全一致；
- subset final-level reliability 是否失败，尤其是 unique state/context 坍缩或
  MH acceptance rate 过低；
- MC 是否因为样本数不足而低估稀有失效概率。

### 5. 检查 adaptive subset 停止策略

默认 following 配置来自 `subset/scripts/configs/latent_subset_following.yaml`：

```text
subset_simulation.max_levels = 8
subset_simulation.adaptive_stop_enabled = true
subset_simulation.adaptive_stop_min_failure_count = 50
subset_simulation.adaptive_stop_min_levels = 2
```

验证：

- `max_levels = 8` 是最大允许层数，不代表必须跑满 8 层；
- 若当前目标在较浅层已经有足够 failure count，应记录
  `stop_reason = adaptive_failure_count_reached`；
- 若目标更稀有，failure count 不足，则继续向更深层推进，最多到 8 层；
- 若达到 8 层仍未满足 failure count，应记录 `stop_reason = max_levels_reached`，
  并结合 reliability 诊断解释估计是否可严格解释。
- 若自适应停止后的 final level 仍不通过 reliability，不能把结果作为最终概率估计接受。
  应优先增加 `num_samples`，再扫描 `proposal_std`、`context_refresh_prob` 和
  `mh_retries_per_sample`，目标是同时满足 reliability pass 和 MC/subset comparison pass。

### 6. 检查可靠性诊断

默认 following 可靠性阈值来自配置：

```text
reliability_min_unique_contexts = 50
reliability_min_unique_state_fraction = 0.50
reliability_max_largest_context_share = 0.30
reliability_max_largest_state_share = 0.10
reliability_min_acceptance_rate = 0.10
```

验证：

- final level 的 `unique_contexts` 不应低于阈值；
- final level 的 `unique_states` 不应低于阈值；
- `largest_context_share` 和 `largest_state_share` 不应超过阈值；
- 最后一个有效 MH transition level 的 acceptance rate 不应低于阈值；
- 若任何可靠性条件失败，`strict_probability_interpretation` 必须为 `false`，
  `probability_estimate_kind` 必须为 `low_reliability_standard_estimate`。

### 7. 检查 final-level playback

从 `latent_subset_samples.npz` 和 playback manifest 检查：

- `subset/scripts/play_final_level_following.py` 必须读取同一个 following config；
- 默认从 final level 选取所有超过 failure threshold 的 unique context cases；
- `--include-duplicate-contexts` 可允许同一 context 的多个高分样本；
- `--no-gif` 可只输出 PNG，用于快速验证；
- 输出的 `final_level_playback_manifest.json` 必须记录 `samples`、`tail_contexts`、
  `level`、`num_cases` 和每个 case 的风险指标。

---

## 同时必须保持以下不能破坏的约束

1. 不允许把 `Y_long`、EVT score、collision label、failure label、tail level 或 ADS
   闭环响应变量作为 diffusion denoiser 的条件输入。

2. 不允许把 following lead 动作语义改成 ego 动作。subset 中 diffusion 输出仍然是
   lead vehicle longitudinal `jx`，ego 只由 IDM 闭环控制产生。

3. 不允许在 subset 阶段重新训练 diffusion 模型、重新拟合 EVT 模型或重新拟合
   scenario-condition copula。subset 只消费已有产物。

4. 不允许把 `tail_contexts.npz` 当成有限离散测试集做均匀平均。默认口径是
   Gaussian-copula 平滑后的 highD tail scenario-condition 分布。

5. 不允许用强 rejection 或手工改写 sampled conditions 来制造高失效率。无效样本可以记录诊断，
   但概率解释必须对应实际采样分布。

6. 不允许破坏 cut-in subset simulation、following/cut-in diffusion prior evaluation、
   process_highD EVT exposure 和 tail context 生成接口。

7. 不允许新增根目录 `utils/` 兼容包；跨模块公共逻辑放在 `tools/`。

8. 不允许修改主要输出文件名和字段，除非同步更新所有读取这些字段的代码与文档。

9. 不允许在默认配置中打开 ego lane change。following 初始目标是评估 IDM 纵向闭环响应。

10. 不允许在最终报告中把 `S_EVT(Y_long_sim)` 解释为真实世界碰撞概率；它只是相对 highD
    following tail EVT 分布的极端程度。

---

## 优先修改范围

优先范围：

```text
subset/scripts/configs/latent_subset_following.yaml
subset/scripts/run_subset_following.py
subset/scripts/play_final_level_following.py
subset/src/context_distribution.py
subset/src/frozen_diffusion_sampler.py
subset/src/latent_evaluator.py
subset/src/closed_loop_runner.py
subset/src/latent_subset_runner.py
subset/src/subset_simulation.py
subset/src/final_level_playback.py
tools/risk.py
tools/evt.py
tools/idm_ego.yaml
```

只有在证明输入产物本身不一致时，才修改：

```text
process_highD/src/following_tail_generation.py
process_highD/scripts/select_following_tail_contexts.py
diffusion/src/features.py
diffusion/src/data.py
```

---

## 接受标准

一次 following subset 设计迭代可以被接受，必须同时满足：

- `python subset/scripts/run_subset_following.py` 能从当前配置完成运行；
- `python subset/scripts/run_monte_carlo_following.py` 能从同一配置完成运行；
- `python subset/scripts/compare_estimators.py --config subset/scripts/configs/latent_subset_following.yaml`
  能完成 MC/subset 对比；
- `latent_subset_summary.json`、`latent_subset_level_stats.csv`、
  `latent_subset_top_cases.json`、`latent_subset_samples.npz` 均生成；
- `latent_monte_carlo_summary.json`、`latent_monte_carlo_stats.csv`、
  `latent_monte_carlo_top_cases.json`、`latent_monte_carlo_samples.npz` 均生成；
- `latent_mc_subset_comparison.json` 生成，并明确 `pass`、`mc_resolution_insufficient`、
  `fail` 或 `incompatible_inputs`；
- `play_final_level_following.py --no-gif` 能生成 playback manifest；
- 输出概率口径与 highD tail scenario-condition distribution 一致；
- `input_space` 正确记录 7 维 conditions、125 维 diffusion noise 和 132 维联合空间；
- final-level reliability 必须为 pass，且 `strict_probability_interpretation = true`；
- `latent_mc_subset_comparison.json.status` 在主验收中必须为 `pass`。若为
  `mc_resolution_insufficient`，只能接受为“接口与诊断流程通过”，必须增加
  `monte_carlo.num_samples` 或补充更高分辨率 MC 后再判断接近性；
- 若 comparison 为 `fail` 或 `incompatible_inputs`，必须完成上述一致性排查并重新运行后才能接受；
- top cases 能解释为 following 闭环纵向风险，而不是物理不可行或 schema 错误；
- 没有破坏 cut-in 入口和现有 process_highD/diffusion 输出接口。

---

## 最终报告需要包含

最终给用户汇报时至少包含：

- 使用的 diffusion checkpoint 路径；
- 使用的 tail context 和 condition distribution 路径；
- EVT failure target：`x_c` 或 return level；
- Monte Carlo 参数：`num_samples`、`seed`；
- Monte Carlo 结果：`probability`、standard error、failure count；
- subset 参数：`num_samples`、`p0`、`max_levels`、adaptive stop 参数、
  `proposal_std`、`context_refresh_prob`、`mh_retries_per_sample`；
- `probability`、standard error、`final_failure_fraction`、`num_levels`、`stop_reason`；
- `simulation_counts` 中的 closed-loop evaluator 调用次数和唯一 context 数；
- Monte Carlo 与 subset 的差值、combined standard error、95% CI 是否重叠，以及
  `latent_mc_subset_comparison.json.status`；
- reliability pass/fail 及原因；
- top cases 中最重要的 longitudinal 风险指标；
- 如果 mileage return period 可用，说明 exposure denominator 是 following ego miles。
