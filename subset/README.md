# subset：latent-space 子集模拟

`subset/` 在 diffusion latent 空间中估计被测车辆在 highD 长尾场景下的闭环高风险概率：

```text
scenario condition ~ highD tail scenario-condition joint distribution
z ~ N(0, I)
actions = DDIM(scenario condition, z)
score = S_EVT(Y_sim)
```

同一 scenario condition 和同一 latent 会通过 deterministic sampler 生成同一条动作轨迹，
便于在 latent 空间做 subset simulation。following 默认轨迹长度为 125 步，cut-in 为
100 步；subset 不再做 rolling reconditioning。

`subset/` 默认使用经过 held-out 评估的 train/val/test diffusion checkpoint：

```text
following: results/diffusion_natural/following/checkpoints/best_noise_mse_train_val_test.pt
cut-in:    results/diffusion_natural/cutin/checkpoints/best_noise_mse_train_val_test.pt
```

不再维护第二套全量训练权重；如果配置指向不存在的 checkpoint，加载器会直接报错。

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

## 全局 highD 暴露映射

`latent_subset_summary.json` 中的顶层 `probability` 不是全局 highD 失效率，而是：

```text
P_ADS(failure | sampled from highD tail scenario-condition distribution)
```

要映射回完整 highD 数据集，脚本读取 `paths.exposure_summary_path` 中的自然驾驶暴露统计，
用 highD 独立长尾峰值率作为筛选测试空间在全局数据中的发生强度。
风险暴露对比固定使用完整 highD 数据集的 `all_vehicle_km` 分母：

```text
ADS global safety-critical intensity
  = subset tail-conditional failure probability
    × highD independent tail peak exposure rate per all_vehicle_km
```

然后在同一 EVT 安全关键阈值 `x_c` 下，与 highD 人类驾驶的自然暴露强度和回报周期对比。
该结果会同时写入：

```text
latent_subset_summary.json: global_risk_exposure_comparison
global_risk_exposure_comparison.json
global_risk_exposure_comparison.csv
```

## 长尾测试空间

`process_highD/` 输出 highD 长尾 scenario conditions 的联合分布；`subset/` 在该联合分布与
扩散 latent 的联合空间上采样测试。`tail_contexts.npz` 不是有限离散测试集，而是保存 empirical
independent tail peaks、必要的 EVT metadata，以及用于最近邻重建 initial states 的 base contexts。
`scenario_condition_distribution.npz` 保存已经建模好的 Gaussian-copula 分布参数；`subset` 只读取
该分布，不在子集模拟阶段重新拟合或从有限 context 行中均匀抽样。

`process_highD` 中基于该分布随机采样、调用扩散模型积分生成轨迹、再与 highD 长尾事件比较，
目的是验证条件扩散模型在给定 scenario condition 下能复现相似测试场景。`subset` 使用相同的
scenario-condition 分布，但估计的是闭环安全关键概率。

following 长尾 context 默认在 declustered independent tail peaks 上拟合 Gaussian copula：

```text
following: context_source = independent_tail_peaks
empirical_context_limit = null
context_generation_method = gaussian_copula
include_empirical_contexts = true
num_synthetic_contexts = 5000
```

following 会先保留全部 declustered highD independent tail peaks，并输出可复用的
`tail_contexts.npz`、`scenario_condition_distribution.npz` 和 generated scenarios。cut-in 入口
`process_highD/scripts/select_cutin_tail_contexts.py` 同时生成
`scenario_condition_distribution.npz`、subset 可读取的 `tail_contexts.npz`，以及 5000 个 diffusion
generated scenarios。cut-in 的 `tail_contexts.npz` 同时保存 empirical independent tail peaks 和
Gaussian-copula sampled scenario conditions；subset provider 读取
`scenario_condition_distribution.npz` 中的已建模分布，再按最近邻 empirical base context 重构
`initial_states` 与 `risk_start_index`。

```text
highd_independent_tail_peak        empirical highD following independent tail peak
highd_evt_independent_tail_peak    empirical highD cut-in independent tail peak
highd_tail_distribution_sample     subset following distribution sample
highd_evt_tail_distribution_sample subset cut-in distribution sample
```

因此默认估计的是平滑 highD tail scenario-condition 分布上的条件失效概率，而不是有限
empirical peaks 或预生成 synthetic rows 上的离散平均。`tail_contexts.npz` 会保存 base 来源：

```text
synthetic_context, context_model_method, base_context_index,
base_event_id, context_feature_distance
```

## 推荐运行顺序

following 和 cut-in 使用不同入口脚本。following subset 一次性生成 125 步 lead jerk；
cut-in diffusion 生成默认输出 100 步 `[ax, ay]` 轨迹。

following：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python diffusion/scripts/train_following_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_following_prior.py
conda run -n tread python process_highD/scripts/estimate_following_exposure.py
conda run -n tread python process_highD/scripts/select_following_tail_contexts.py
conda run -n tread python subset/scripts/run_monte_carlo_following.py
conda run -n tread python subset/scripts/run_subset_following.py
conda run -n tread python subset/scripts/play_final_level_following.py --no-gif
```

当前 following subset 默认使用 `num_samples=3000, p0=0.2, max_levels=8`，并关闭
adaptive stop。following 扩散噪声空间为 `[125, 1] = 125` 维；加上 7 维
scenario conditions，联合输入空间为 132 维。运行入口会在结束日志中打印实际闭环仿真
evaluator 调用次数和唯一 scenario context 数。

当前 following 安全阈值采用 300 km all-vehicle return level。审计结果保存在
`results/subset_simulation_following/`：100000 次 Monte Carlo 的估计为
`0.00255`，3000 样本 subset simulation 用 29303 次闭环评估得到 `0.00249`，
相对差约 2.3%，直接闭环评估数加速约 3.4 倍。

cut-in：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
conda run -n tread python diffusion/scripts/train_cutin_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_cutin_prior.py
conda run -n tread python process_highD/scripts/estimate_cutin_exposure.py
conda run -n tread python process_highD/scripts/select_cutin_tail_contexts.py
conda run -n tread python subset/scripts/run_monte_carlo_cutin.py
conda run -n tread python subset/scripts/run_subset_cutin.py
conda run -n tread python subset/scripts/play_final_level_cutin.py --no-gif
```

当前 cut-in 默认 MC 使用 10000 个独立样本；subset 默认使用
`num_samples=1000, p0=0.1, max_levels=8`，并开启 adaptive stop。`max_levels=8`
是最大允许层数；当前 `x_c=5` 目标下通常在 2 层后因失效样本数足够而停止，以保证
final-level context 多样性和 reliability pass。两个运行入口都会在结束日志中打印实际闭环
仿真 evaluator 调用次数和唯一 scenario context 数。cut-in 扩散噪声空间为
`[100, 2] = 200` 维；加上 10 维 scenario conditions，联合输入空间为 210 维。

## 主要文件

```text
subset/scripts/configs/latent_subset_following.yaml
subset/scripts/configs/latent_subset_cutin.yaml
subset/scripts/run_subset_following.py
subset/scripts/run_subset_cutin.py
subset/scripts/run_monte_carlo_following.py
subset/scripts/run_monte_carlo_cutin.py
subset/scripts/play_final_level_following.py
subset/scripts/play_final_level_cutin.py
subset/src/latent_subset_runner.py
subset/src/final_level_playback.py
subset/src/context_distribution.py
subset/src/subset_simulation.py
subset/src/closed_loop_runner.py
subset/src/latent_evaluator.py
subset/src/frozen_diffusion_sampler.py
```

`subset/` 不维护自己的风险公式、EVT 解析、context NPZ 读取或 IDM 参数副本；这些共享逻辑
统一从 `tools/` 读取。闭环回放和 final-level playback 使用 `tools/idm_ego.yaml` 中的
IDM ego 配置，确保 process_highD 与 subset 的 ego 响应参数一致。

这些脚本是当前公开运行入口，虽然 following/cut-in 入口结构相似，但分别绑定不同配置、事件类型、
默认样本数和输出目录，因此不删除。`latent_subset_samples.npz`、`latent_monte_carlo_samples.npz`
和 `final_level_playbacks/` 是复现和可视化产物，可由对应脚本重建，不作为源码维护对象。

## Final-Level Playback

`play_final_level_following.py` 和 `play_final_level_cutin.py` 读取
`latent_subset_samples.npz` 的最后一层样本，并用同目录的 `latent_subset_summary.json`
中的 `failure_threshold` 筛选安全关键场景。默认行为是：

```text
final level only
score >= failure_threshold
unique (scenario_conditions, initial_states, latent/actions) test scenario only
randomly select 10 matching cases by default
```

即先在最终层筛选超过安全阈值的样本，再按完整测试输入去重：
`scenario_conditions`、`initial_states`、diffusion `latents`、解码后的 `actions` 和
`action_mask` 共同定义一个测试场景。`context_index` 只是 sampled condition 最近邻匹配到的
empirical tail context 来源标识，不作为最终层可视化的唯一去重键；同一 context 下不同
latent/action plan 的危险样本仍然可以进入候选池。随后脚本用 `--random-seed` 控制的
无放回随机抽样选择 `--num-cases K` 个案例。默认 `--num-cases 10 --random-seed 42`。
`--no-gif` 只输出 overview PNG。

复现方式不同于 `process_highD/scripts/play_*_tail_events.py`：

- process_highD playback 读取 `diffusion_generated_scenarios.npz`，按脚本设置随机/指定选择
  generated scenarios；adversary 轨迹已经由 diffusion 生成并保存，ego 在回放时用 IDM 闭环滚动。
- subset final-level playback 读取 subset simulation 最终层保存的
  `scenario_conditions`、`initial_states`、`context_index`、latent 解码后的
  `actions` 和 `action_mask`，不再依赖重新从分布文件构造 context，再调用
  `ClosedLoopFollowingRunner` 或 `ClosedLoopCutInRunner` 执行同一段预采样 adversary action plan。
  因此它复现的是 subset 最终层发现的危险闭环样本，而不是重新抽样 diffusion 泛化场景。

## 输出

```text
results/subset_simulation_following/latent_subset_summary.json
results/subset_simulation_following/global_risk_exposure_comparison.json
results/subset_simulation_following/global_risk_exposure_comparison.csv
results/subset_simulation_following/latent_subset_level_stats.csv
results/subset_simulation_following/latent_subset_top_cases.json
results/subset_simulation_following/latent_subset_samples.npz
results/subset_simulation_following/final_level_playbacks/
results/monte_carlo_following/latent_monte_carlo_summary.json
results/monte_carlo_following/latent_monte_carlo_stats.csv
results/monte_carlo_following/latent_monte_carlo_top_cases.json
results/monte_carlo_following/latent_monte_carlo_samples.npz
results/subset_simulation_cutin/latent_subset_summary.json
results/subset_simulation_cutin/global_risk_exposure_comparison.json
results/subset_simulation_cutin/global_risk_exposure_comparison.csv
results/subset_simulation_cutin/latent_subset_level_stats.csv
results/subset_simulation_cutin/latent_subset_top_cases.json
results/subset_simulation_cutin/latent_subset_samples.npz
results/subset_simulation_cutin/final_level_playbacks/
results/monte_carlo_cutin/latent_monte_carlo_summary.json
results/monte_carlo_cutin/latent_monte_carlo_stats.csv
results/monte_carlo_cutin/latent_monte_carlo_top_cases.json
results/monte_carlo_cutin/latent_monte_carlo_samples.npz
```

`latent_subset_summary.json` 和 `latent_monte_carlo_summary.json` 都记录 `event_type`、
`input_paths`、`input_space` 和 `simulation_counts`，用于确认两种估计器使用同一个
scenario-condition 联合分布、diffusion latent 空间和闭环评分口径。大型结果文件属于可再生成产物。
默认配置使用精简样本保存：`latent_subset_samples.npz` 保留复现 final-level playback 所需的
`scenario_conditions`、`initial_states`、`latents`、`actions`、`action_mask` 和核心评分字段；
Monte Carlo 样本默认不保存 actions。若需要额外 cut-in 诊断指标或 MC actions，可在配置中开启
`sample_storage.include_diagnostics` 或 `sample_storage.include_monte_carlo_actions`。
论文图由 `results/build_*_paper_experiments.py` 从已有结果统一生成，输出到
`results/paper_experiments/{following,cutin}/`。following 当前生成 GPD 诊断、300 km
return-level inverse calibration、tail diffusion generalization panel 和 subset level score
histogram；其中 generalization panel 的子图 f 使用 `process_highD` 同一
`lead_braking_duration` condition 分布，subset histogram 的图例放在左侧以避开高风险阈值区域。
