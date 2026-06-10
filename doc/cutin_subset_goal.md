# Cut-in 子集模拟 Goal 设计文档

## Goal

`/goal` 设计并完善 TREAD 工程中 cut-in 长尾测试空间上的 latent-space subset
simulation，使其能够在已获取的 highD cut-in 长尾 scenario-condition 联合分布与扩散模型随机正态潜在空间上，估计 IDM 控制的 highway-env 闭环 ego 在 cut-in 场景中的安全关键失效概率。

代码运行系统的conda环境：

```bash
conda activate tread
```

最终状态应满足：

- 测试空间明确定义为：

```text
c ~ highD cut-in tail scenario-condition Gaussian-copula distribution
z ~ N(0, I)
target_actions = DDIM(c, z)
ego_response = highway-env IDM closed-loop rollout
score = S_EVT(Y_cutin_sim)
```

- cut-in horizon 固定为 100 帧，采样频率为 25 Hz，扩散 latent 形状为 `[100, 2]`，动作语义保持为 target vehicle `[ax, ay]`；
- `subset/scripts/run_latent_subset_cutin.py` 能读取 `process_highD/` 输出的 `tail_contexts.npz` 与 `scenario_condition_distribution.npz`，而不是重新拟合 tail distribution；
- subset 估计结果能输出标准概率、层级阈值、final-level 诊断、top cases、样本缓存和回放入口；
- `subset/scripts/` 下必须提供同分布 Monte Carlo 基线入口。Monte Carlo 与 subset 使用完全相同的
  scenario-condition 联合分布、扩散 latent 先验、DDIM sampler、IDM ego 和 cut-in EVT
  评分口径，只是不做自适应分层和 MH 条件采样；
- `latent_subset_summary.json` 中的概率解释必须对应：

```text
P_context,z(Y_cutin_sim > x_c | o sampled from highD tail scenario-condition distribution)
```

- 若可靠性诊断不通过，结果必须被标记为 low-reliability estimate，不能被解释为严格概率估计。

---

## 概率目标与失效定义

### 1. 联合测试空间

cut-in subset simulation 的随机变量由两部分组成：

```math
\omega = (\mathbf{c}, \mathbf{z})
```

其中：

- `scenario_conditions` $\mathbf{c}$ 来自 `results/highd_cutin_tail/contexts/scenario_condition_distribution.npz` 中保存的 Gaussian copula 模型；
- `z` 为扩散模型确定性 DDIM 采样的初始噪声，服从标准正态分布；
- `initial_states` 不直接参与 denoiser 条件输入，只由 sampled condition 和最近邻 empirical highD tail context 重构，供 target 轨迹积分和闭环仿真使用。

cut-in 条件特征顺序必须与 `diffusion/src/features.py` 中
`CUTIN_SCENARIO_CONDITION_KEYS` 完全一致：

```text
ego_vx_0
initial_gap
initial_lateral_offset
initial_delta_vx
target_ax_0
target_vy_0
target_ay_0
final_lateral_offset
time_to_cross
target_speed_change
```

### 2. 闭环失效事件

对每个样本：

```math
\mathbf{a} = \mathcal{D}(\mathbf{c}, \mathbf{z}; \theta)
```

其中 $\mathcal{D}$ 为冻结的 cut-in diffusion prior 与 deterministic DDIM sampler。
target 车辆执行生成的 `[ax, ay]` 计划；ego 车辆由 `tools/idm_ego.yaml` 中的 IDM 参数在
highway-env 中闭环响应。闭环轨迹通过 `tools/highd_cutin.py` 的 cut-in 风险口径得到
`Y_cutin_sim`，再由 highD cut-in peak EVT 模型映射为：

```math
S_{\mathrm{EVT}}(Y_{\mathrm{cutin,sim}})
= -\log P_{\mathrm{EVT}}(Y_{\mathrm{cutin}} > Y_{\mathrm{cutin,sim}})
```

默认失效事件为：

```text
Y_cutin_sim > x_c
x_c = evt.collision_critical_level = 5.0
failure_threshold = S_EVT(x_c)
```

如果使用 return-period 目标，必须在配置和结果摘要中明确写出对应的 `return_period`
和 `return_level_target`。

---

## 由以下测试或数据证据验证

### 1. 重新运行 cut-in 主流程

完整验证顺序：

```bash
python process_highD/scripts/extract_highd_events.py
python process_highD/scripts/build_natural_dataset.py --config diffusion/scripts/configs/natural_cutin.yaml
python diffusion/scripts/train_cutin_diffusion.py
python diffusion/scripts/evaluate_cutin_prior.py
python process_highD/scripts/estimate_cutin_exposure.py
python process_highD/scripts/select_cutin_tail_contexts.py
python subset/scripts/run_latent_monte_carlo_cutin.py
python subset/scripts/run_latent_subset_cutin.py
python subset/scripts/compare_latent_cutin_estimators.py
```

如果只验证 subset 设计和接口，可在已有 diffusion checkpoint、EVT 模型和 tail context
均存在时运行：

```bash
python subset/scripts/run_latent_monte_carlo_cutin.py
python subset/scripts/run_latent_subset_cutin.py
python subset/scripts/compare_latent_cutin_estimators.py
```

### 2. 检查 subset 输入文件

检查：

```text
results/diffusion_natural/cutin/feature_schema.json
results/diffusion_natural/cutin/normalization_stats.json
results/diffusion_natural/cutin/checkpoints/best_noise_mse_all_train.pt
results/highd_cutin_tail/contexts/tail_contexts.npz
results/highd_cutin_tail/contexts/scenario_condition_distribution.npz
results/highd_cutin_tail/evt/cutin_peak_evt_model.json
results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json
tools/idm_ego.yaml
```

验证：

- `feature_schema.json` 中 `event_type` 必须为 `cut_in`，`model_input_keys`
  只能包含 `scenario_conditions`；
- diffusion checkpoint 优先使用 `best_noise_mse_all_train.pt`；如果 fallback 到
  `best_noise_mse_train_val_test.pt`，必须在结果说明中标注；
- `tail_contexts.npz` 必须包含 empirical independent tail peak rows，source type 为
  `highd_evt_independent_tail_peak`；
- `scenario_condition_distribution.npz` 必须包含 `copula_correlation`、
  `copula_variable_mask` 和 `copula_marginal_values`；
- `copula_marginal_values` 的维度必须为 10，并且与 cut-in condition keys 对齐；
- `initial_states` 的形状必须为 `[2, 6]`，表示 ego 与 target 的
  `[x, y, vx, vy, ax, ay]`。

### 3. 检查 subset 输出

检查：

```text
results/subset_simulation_cutin/latent_subset_summary.json
results/subset_simulation_cutin/latent_subset_level_stats.csv
results/subset_simulation_cutin/latent_subset_top_cases.json
results/subset_simulation_cutin/latent_subset_samples.npz
results/subset_simulation_cutin/latent_mc_subset_comparison.json
results/subset_simulation_cutin/latent_mc_subset_comparison.csv
results/subset_simulation_cutin/figures/subset_score_histograms.png
```

验证：

- `probability` 有限且非负；
- `probability_target` 明确包含 highD tail scenario-condition distribution；
- `failure_event` 必须对应 `Y_cutin_sim > x_c` 或明确的 return-level 目标；
- `score_space` 为 `evt` 时，`failure_threshold` 必须等于 cut-in EVT 模型对目标
  `Y_cutin` 的 score；
- `thresholds`、`acceptance_rates`、`level_stats` 必须存在；
- `latent_subset_samples.npz` 必须保存每层 `context_indices`、`latents`、`scores`、
  `actions`、`y_cutin`、`is_cutin`、`is_front_cutin`、`min_gap`、`min_ttc`、
  `physical_feasible` 等关键字段；
- `actions` 的最后一维必须为 2，对应 target `[ax, ay]`，不能被解释为 ego 动作。
- `latent_mc_subset_comparison.json` 必须在 Monte Carlo 和 subset 都完成后生成，
  并记录两者输入一致性、阈值一致性、概率差值、combined standard error 和 CI overlap。

### 4. 检查 Monte Carlo 基线输出

检查：

```text
results/monte_carlo_cutin/latent_monte_carlo_summary.json
results/monte_carlo_cutin/latent_monte_carlo_stats.csv
results/monte_carlo_cutin/latent_monte_carlo_top_cases.json
results/monte_carlo_cutin/latent_monte_carlo_samples.npz
results/monte_carlo_cutin/figures/monte_carlo_score_histogram.png
```

验证：

- Monte Carlo 的 sampling space 必须与 subset 完全一致：

```text
c ~ highD cut-in tail scenario-condition Gaussian-copula distribution
z ~ N(0, I)
actions = DDIM(c, z)
score = S_EVT(Y_cutin_sim)
```

- `latent_monte_carlo_summary.json` 中 `estimator` 必须为 `independent_monte_carlo`；
- `probability_target` 必须说明样本来自 highD tail scenario-condition distribution；
- `probability` 必须等于 `mean(score >= failure_threshold)`；
- `probability_standard_error` 必须使用独立 Bernoulli 近似：

```math
\widehat{\mathrm{SE}} =
\sqrt{\hat{p}(1-\hat{p}) / N}
```

- `latent_monte_carlo_samples.npz` 必须保存 `context_indices`、`latents`、`scores`、
  `failure_mask`、`actions`、`y_cutin`、`is_cutin`、`is_front_cutin`、
  `physical_feasible` 等字段；
- MC 结果只作为直接采样基线。若 safety-critical failure 很少或为 0，不应据此否定
  subset simulation 的必要性，而应报告 MC 的分辨率限制。

### 5. 检查 Monte Carlo 与 subset 概率一致性

Monte Carlo 和 subset 估计的是同一个目标概率，因此在 MC 样本分辨率足够时，两者结果必须相近。
记：

```text
p_mc = latent_monte_carlo_summary.json.probability
se_mc = latent_monte_carlo_summary.json.probability_standard_error
p_ss = latent_subset_summary.json.probability
se_ss = latent_subset_summary.json.probability_standard_error
```

若 MC failure count 不少于 10，且 `se_mc / max(p_mc, 1e-12) <= 0.5`，则必须满足以下至少一项：

```math
|p_{\mathrm{ss}} - p_{\mathrm{mc}}|
\le 2\sqrt{\mathrm{se}_{\mathrm{ss}}^2 + \mathrm{se}_{\mathrm{mc}}^2}
```

或两者 95% confidence interval 有重叠。若不满足，需要先排查：

- MC 与 subset 是否使用同一 `tail_context_path`、`condition_distribution_path`、
  `diffusion_checkpoint`、`evt_model_path` 和 `idm_ego_config_path`；
- 两者的 `failure_threshold`、`evt_return_level_target`、`score_space` 是否完全一致；
- cut-in risk scoring 中 `require_cutin_for_failure`、`risk_start_index`、cut-in semantic
  gate 是否一致；
- subset final-level reliability 是否失败，尤其是 unique state/context 坍缩或 MH acceptance rate 过低；
- MC 是否因为样本数不足而低估稀有失效概率。

若 MC failure count 小于 10，或 MC relative standard error 大于 0.5，则不能把 MC 用作强一致性证据。
此时必须：

- 增加 `monte_carlo.num_samples` 后重新运行；或
- 明确报告 MC resolution insufficient，并只把 MC 作为直接采样下界/诊断基线。

当 MC 与 subset 使用相同 `num_samples` 和 `seed` 时，subset 第 0 层的独立样本分布应与 MC 基线一致；
若两者第 0 层 score 分布明显不同，应优先检查采样、latent shape、DDIM checkpoint 和闭环评分入口。

自动化比较入口：

```bash
python subset/scripts/compare_latent_cutin_estimators.py
```

该脚本读取 `latent_monte_carlo_summary.json` 与 `latent_subset_summary.json`，并输出：

```text
results/subset_simulation_cutin/latent_mc_subset_comparison.json
results/subset_simulation_cutin/latent_mc_subset_comparison.csv
```

其中 `status` 只能接受以下含义：

- `pass`: MC 分辨率足够，且满足 combined-SE 或 CI-overlap 相容性判据；
- `mc_resolution_insufficient`: MC failure count 或 relative SE 不足，不能强验证相近性，
  但已明确标记 MC 分辨率限制；
- `fail`: MC 分辨率足够但与 subset 不相容，不能接受；
- `incompatible_inputs`: 两者输入、阈值、score space 或 failure event 不一致，不能接受。

### 6. 检查可靠性诊断

默认 cut-in 可靠性阈值来自 `subset/scripts/configs/latent_subset_cutin.yaml`：

```text
reliability_min_unique_contexts = 80
reliability_min_unique_state_fraction = 0.50
reliability_max_largest_context_share = 0.20
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

### 7. 检查闭环 cut-in 语义

从 `latent_subset_samples.npz` 和 `latent_subset_top_cases.json` 中检查：

- 高分样本应主要来自语义有效 cut-in，即 `is_cutin >= 0.5`；
- 若 `require_cutin_for_failure: true`，非语义 cut-in 不应被计入高风险失效；
- `is_front_cutin`、`cutin_time_headway`、`cutin_lateral_time_gap`、
  `safety_distance_deficit`、`max_post_cutin_drac` 应能解释高分样本的来源；
- target 轨迹应满足物理可行性，`physical_feasible` 不应因大量 action clipping
  或 lateral jerk clipping 而系统性失败；
- top cases 必须可通过 `subset/scripts/play_final_level_cutin.py` 回放检查。

---

## 同时必须保持以下不能破坏的约束

1. 不允许把 `Y_cutin`、EVT score、collision label、failure label、tail level
   或 ADS 闭环响应变量作为 diffusion denoiser 的条件输入。

2. 不允许把 cut-in target 动作语义改成 ego 动作。subset 中 diffusion 输出仍然是 target
   vehicle `[ax, ay]`，ego 只由 IDM 闭环控制产生。

3. 不允许在 subset 阶段重新训练 diffusion 模型、重新拟合 EVT 模型或重新拟合
   scenario-condition copula。subset 只消费已有产物。

4. 不允许把 `tail_contexts.npz` 当成有限离散测试集做均匀平均。默认口径是
   Gaussian-copula 平滑后的 highD tail scenario-condition 分布。

5. 不允许用强 rejection 或手工改写 sampled conditions 来制造高失效率。无效样本可以记录诊断，
   但概率解释必须对应实际采样分布。

6. 不允许破坏 following subset simulation、following/cut-in diffusion prior evaluation、
   process_highD EVT exposure 和 tail context 生成接口。

7. 不允许新增根目录 `utils/` 兼容包；跨模块公共逻辑放在 `tools/`。

8. 不允许修改主要输出文件名和字段，除非同步更新所有读取这些字段的代码与文档。

9. 不允许在默认配置中打开 ego lane change。cut-in 初始目标是评估 IDM 纵向闭环响应。

10. 不允许在最终报告中把 `S_EVT(Y_cutin_sim)` 解释为真实世界碰撞概率；它只是相对 highD
    cut-in tail EVT 分布的极端程度。

11. 不允许让 Monte Carlo 基线使用不同于 subset 的采样空间、diffusion checkpoint、
    IDM 参数、risk scoring 或 EVT threshold。MC 与 subset 的差异只能是估计器本身。

---

## 优先修改范围

优先范围：

```text
subset/scripts/configs/latent_subset_cutin.yaml
subset/scripts/run_latent_monte_carlo_cutin.py
subset/scripts/run_latent_subset_cutin.py
subset/scripts/play_final_level_cutin.py
subset/src/context_distribution.py
subset/src/frozen_diffusion_sampler.py
subset/src/latent_evaluator.py
subset/src/closed_loop_runner.py
subset/src/latent_subset_runner.py
subset/src/subset_simulation.py
subset/src/final_level_playback.py
tools/highd_cutin.py
tools/risk.py
tools/evt.py
tools/idm_ego.yaml
```

只有在证明输入产物本身不一致时，才修改：

```text
process_highD/src/cutin_tail_generation.py
process_highD/scripts/select_cutin_tail_contexts.py
diffusion/src/features.py
diffusion/src/data.py
```

不优先修改：

```text
diffusion/src/model.py
diffusion/src/train.py
process_highD/src/event_extraction.py
following 相关配置与代码
```

---

## 迭代决策顺序

### 1. 先确认输入分布，再调 subset 参数

先检查：

- `scenario_condition_distribution.npz` 的 copula 维度、变量 mask 和 empirical marginals；
- sampled condition 的 gap、time-to-cross、lateral offset、target speed change 是否落在合理 highD tail 支撑内；
- `TailContextDistribution.__getitem__()` 重构的 `initial_states` 与 `scenario_conditions`
  是否一致；
- `risk_start_index` 是否由 `time_to_cross / dt` 合理映射。

若输入分布已经偏离 empirical tail，应修复 `process_highD` 的 tail condition 建模，而不是调 MH proposal。

### 2. 再确认 diffusion 解码和闭环积分

检查：

- 同一 `(context_index, z)` 重复评估是否得到完全一致的 actions 与 score；
- `actions` 是否为 `[100, 2]`，并在闭环 runner 中按 `[ax, ay]` 使用；
- target lateral motion 是否能进入 ego lane 附近；
- clipping、jerk limit、speed limit 是否导致大量生成动作被改写；
- ego IDM 参数是否来自 `tools/idm_ego.yaml`。

若 deterministic mapping 不稳定，先修复 sampler 或 runner，不要继续解释概率结果。

### 3. 再调 subset simulation 超参数

如果 final level 可靠性失败，优先按以下顺序调参：

1. 增加 `num_samples`；
2. 调整 `proposal_std`，目标是提高 latent random-walk 的有效接受率；
3. 调整 `context_refresh_prob`，目标是在 tail condition 多模态之间保持混合；
4. 增加 `mh_retries_per_sample`；
5. 必要时降低 `max_levels` 或调整失效目标，用于诊断而不是最终报告。

每次只改一个主要超参数，并记录修改前后的：

```text
monte_carlo_probability
monte_carlo_standard_error
probability
final_failure_fraction
num_levels
acceptance_rates
unique_contexts
unique_states
largest_context_share
largest_state_share
top case y_cutin / is_cutin / physical_feasible
```

### 4. 如果高分样本不是语义 cut-in

优先检查：

- `cutin_risk.require_cutin_for_failure` 是否为 true；
- `risk_start_index` 是否对齐 lane crossing；
- `lateral_overlap_threshold`、`cutin_lateral_offset`、`min_lateral_approach_speed`
  是否与 `tools/highd_cutin.py` 的评分逻辑一致；
- generated target trajectory 是否实际完成进入 ego lane。

不要通过直接抬高非 cut-in 样本风险来修复。

### 5. 如果概率估计极低或没有达到失效层

需要区分：

1. IDM ego 对当前 cut-in tail 分布本身很安全；
2. diffusion prior 生成的 target 轨迹过于温和；
3. failure threshold 过高或 EVT score 映射不一致；
4. subset proposal 混合不足，没有探索到高风险 latent 区域。

必须通过 level score histogram、top cases、generated action/trajectory 和 EVT target
metadata 证明是哪一类问题。

---

## 接受标准

一次 cut-in subset 设计迭代可以被接受，必须同时满足：

- `python subset/scripts/run_latent_subset_cutin.py` 能从当前配置完成运行；
- `python subset/scripts/run_latent_monte_carlo_cutin.py` 能从同一配置完成运行；
- Monte Carlo 和 subset 使用相同的 `tail_context_path`、`condition_distribution_path`、
  `diffusion_checkpoint`、`evt_model_path` 和 `idm_ego_config_path`；
- `latent_subset_summary.json`、`latent_subset_level_stats.csv`、
  `latent_subset_top_cases.json`、`latent_subset_samples.npz`、
  `latent_mc_subset_comparison.json` 均生成；
- `latent_monte_carlo_summary.json`、`latent_monte_carlo_stats.csv`、
  `latent_monte_carlo_top_cases.json`、`latent_monte_carlo_samples.npz` 均生成；
- 输出概率口径与 highD tail scenario-condition distribution 一致；
- 在 MC 分辨率足够时，Monte Carlo 与 subset 的概率估计必须统计相容；若不相容，
  必须完成上述一致性排查后才能接受结果；
- 在 MC 分辨率不足时，必须明确标记 MC 不能验证 subset 概率接近性，并说明是否已增加
  `monte_carlo.num_samples`；
- final-level reliability 为 pass，或文档明确说明为何只能作为 low-reliability diagnostic；
- top cases 能解释为 cut-in 闭环风险，而不是物理不可行或 schema 错误；
- 没有破坏 following 入口和现有 process_highD/diffusion 输出接口。

---

## 最终报告需要包含

最终给用户汇报时至少包含：

- 使用的 diffusion checkpoint 路径；
- 使用的 tail context 和 condition distribution 路径；
- EVT failure target：`x_c` 或 return level；
- Monte Carlo 参数：`num_samples`、`seed`；
- Monte Carlo 结果：`probability`、standard error、failure count；
- subset 参数：`num_samples`、`p0`、`max_levels`、`proposal_std`、
  `context_refresh_prob`、`mh_retries_per_sample`；
- `probability`、`final_failure_fraction`、`num_levels`；
- Monte Carlo 与 subset 的差值、combined standard error、95% CI 是否重叠，以及是否满足相近性判据；
- reliability pass/fail 及原因；
- top cases 中最重要的 cut-in 风险指标；
- 如果 mileage return period 可用，说明 exposure denominator 是 cut-in all-vehicle miles。
