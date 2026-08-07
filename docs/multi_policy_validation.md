# Codex Goal：在既有 highD 管线内补充论文修订实验

## 0. 决策、目标与完成定义

### 0.1 修订决策

本轮修订只在现有 highD 高速公路数据、following 与 cut-in 两类事件、现有冻结的
EVT/Copula/diffusion 资产和 `highway-env` 闭环环境内开展。需完成四组实验：

1. 多 ADS 长尾安全测试；
2. 子集模拟（SS）的参数敏感性；
3. 关键模块消融；
4. 与 Monte Carlo（MC）和重要性采样（IS）的稀有风险估计对比。

这是一项高收益、受时间约束的修订方案。它能实质回应审稿意见中的“多 ADS”“模块剥离”“子集模拟参数敏感性”和“重要性采样等同类方法横向对比”。它**不能**宣称已经完成城市多场景泛化，也不对复合风险指标优于单一 TTC 作新增实证。论文回复和讨论中必须诚实保留这两项限制。

### 0.2 总目标

在不改变论文当前概率对象的前提下，产出可复现、可审计、可直接用于论文表格和图片的证据：

\[
p_{a,e}=\Pr_{o\sim D_e,\;z\sim\mathcal N(0,I)}
\left[S_e\bigl(Y^{\mathrm{ADS}=a}_e(o,z)\bigr)\geq \gamma_e^\star\right],
\quad e\in\{\mathrm{following},\mathrm{cut\_in}\}.
\]

其中，`D_e` 是已保存的 highD 尾部场景条件 Gaussian Copula 分布，`z` 是冻结条件扩散模型的初始噪声，`Y_e` 和 `\gamma_e^\star` 分别沿用现有事件风险变量和人类驾驶校准 EVT 阈值。该量是**长尾条件测试分布下的安全关键条件概率**，不是全路网碰撞概率。

### 0.3 完成定义

只有同时满足以下条件，目标才可标为完成：

- 四组实验均有机器可读的原始结果、运行 manifest、汇总表、绘图数据和论文级图表；
- 多 ADS 表中的比较使用同一输入资产、相同事件定义、相同主 SS 配置和相同种子集合；
- SS 参数敏感性和估计器对比都有独立重复运行，不以单次 MCMC 的二项近似标准误作为唯一不确定性证据；
- IS 在可计算密度的上游空间实施，并保存 proposal、权重和 ESS，不能把扩散解码后的动作轨迹当成具有已知密度；
- 每一个最终结论都能从输出 JSON/CSV 追溯到所用 checkpoint、输入文件哈希、完整配置和 seed；
- 所有验收条件通过，或清楚记录失败设置及其原因；不得静默丢弃不利结果。

## 1. 研究边界与不可变约束

### 1.1 本轮包含的对象

| 项目 | 固定对象 |
| --- | --- |
| 数据与事件 | `highD`；`following`、`cut_in` |
| 条件分布 | `results/highd_following_tail/contexts/scenario_condition_distribution.npz` 与 `results/highd_cutin_tail/contexts/scenario_condition_distribution.npz` |
| 尾部 context | 对应的 `tail_contexts.npz`；默认仍经 `TailContextDistribution` 按既有 Copula 定义采样 |
| 对手车生成 | 已训练且冻结的 `results/diffusion_natural/*/checkpoints/best_noise_mse_train_val_test.pt`；50-step deterministic DDIM |
| 风险标尺 | 既有 `Y_long` / `Y_cutin`、POT-GPD 模型和 exposure；源实现为 `tools/risk.py`、`tools/highd_cutin.py` |
| 主估计器 | 既有 joint context-index + diffusion-latent subset simulation |
| 被测 ADS | `IDM`、`A2C`、`PPO`、`SAIRL` 的仓库内 checkpoint |

### 1.2 明确不做的事情

本目标严禁将下列工作混入实现或论文主结论：

- 不下载、不接入、不训练新的城市数据集，不新增城市路口、环岛、行人或非机动车 ODD；
- 不重新训练或微调 diffusion，不重拟合/重调生产用 EVT/GPD，不改变 highD 事件筛选；
- 不修改 `Y_long`、`Y_cutin` 的风险分量、权重、TTC/THW/DRAC 定义或人类暴露量换算；
- 不实施“复合风险指标 vs 单一 TTC”优劣实验，也不在文字中声称该优劣已被本轮新增实验验证；
- 不新增或训练新的 ADS，不把随机策略、对抗优化器或外部生成器伪装成 ADS/IS 对照；
- 不覆盖现有 `IDM_subset/results/`、`A2C_subset/results/`、`PPO_subset/results/`、`SAIRL_subset/results/` 的论文基线结果；
- 不为本轮实验复制四套完整 subset 管线或建立与当前接口重复的通用框架。

### 1.3 论文表述边界

新增实验后，论文可以说“在 highD 两类高速公路事件及四种既有控制策略上验证了方法的策略无关可用性、主要设计贡献和估计效率”。不得说“适用于城市多 ODD”或“证明复合指标全面优于 TTC”。讨论部分保留：城市 ODD 泛化与风险指标对照是后续工作。

## 2. 现有工程事实与统一实验协议

### 2.1 必须复用的现有入口

四个 ADS 均已有 following/cut-in 的闭环评估分支：

| ADS | 配置根目录 | 子集模拟入口 |
| --- | --- | --- |
| IDM | `IDM_subset/scripts/configs/` | `IDM_subset/scripts/run_subset_{following,cutin}.py` |
| A2C | `A2C_subset/scripts/configs/` | `A2C_subset/scripts/run_subset_{following,cutin}.py` |
| PPO | `PPO_subset/scripts/configs/` | `PPO_subset/scripts/run_subset_{following,cutin}.py` |
| SAIRL | `SAIRL_subset/scripts/configs/` | `SAIRL_subset/scripts/run_subset_{following,cutin}.py` |

它们最终均调用各自的 `run_subset_from_config` 和 `run_monte_carlo_from_config`，输入格式及摘要字段已基本一致。新代码应复用这些闭环 runner、`FrozenDiffusionSampler`、context sampler、风险评分与结果字段；不得重写车辆动力学或风险函数。

现有部分结果的 `num_samples` 和 MC 预算不同（例如 IDM following 为 `N=3000`、MC 为 200000，而 PPO following 默认 `N=1000`、MC 为 10000）。因此，**现有结果只能作为 smoke test 和参考，不能直接拼入正式多 ADS 主表**。

### 2.2 冻结清单

在首次正式运行前生成 `results/multi_policy_validation/frozen_inputs.json`。它至少记录：

- 两个场景的 context NPZ、Copula NPZ、diffusion checkpoint、normalization stats、EVT model、exposure summary 的绝对路径和 SHA-256；
- `tools/risk.py`、`tools/highd_cutin.py`、各 ADS policy adapter、各 baseline YAML 的 git revision 或文件 SHA-256；
- 每个 policy checkpoint 的路径和 SHA-256；
- 固定 episode horizon、车道数、动力学限制、DDIM step 数、`evt_return_level_target`、`failure_threshold`；
- 使用的 Python、PyTorch、highway-env/stable-baselines3 版本、设备和 worker 配置。

正式结果的每个 summary 必须引用该冻结清单的 ID。若任何上述输入改变，必须创建新的 revision ID 并从受影响的实验开始重跑。

### 2.3 公平性协议

所有主比较均满足：

1. 同一事件内，四个 ADS 使用同一 `D_e`、相同 diffusion checkpoint/50 DDIM steps、相同风险函数和同一 `\gamma_e^\star`；
2. policy 只做推理；A2C/PPO 按配置 deterministic 推理，SAIRL/IDM 的确定性设置也记录；
3. following 固定 125 steps、1 lane，cut-in 固定 100 steps、2 lanes，沿用各自原始动作能力，不为提高某一 policy 表现而开关额外能力；
4. 主 SS 参数在同一事件的全部 ADS 之间相同；不同 ADS 允许因真实接受率、层数不同而有不同实际闭环调用次数；
5. 一个 `seed` 对应相同的 context-index 初始序列和 diffusion latent 初始序列。若不同 ADS 的环境随机数也可完全对齐，则一并对齐；不能对齐时必须在 manifest 标注；
6. 结果按独立 seed 汇总，不能只挑选与 MC 接近的一个 seed。

### 2.4 统一主配置与种子

正式主表使用以下固定设置；它们来自当前 IDM 论文设置，而不是在某个 ADS 上搜索得到。

| 事件 | 冻结正式 SS | 主 MC/参考预算 | 主 seeds |
| --- | --- | --- | --- |
| following | `N=3000, p0=0.20, proposal_std=0.12, context_refresh_prob=0.70, mh_retries=6, max_levels=8, adaptive_stop=false` | 200000 次独立 MC | `[101, 202, 303, 404, 505]` |
| cut-in | `N=1000, p0=0.10, proposal_std=0.10, context_refresh_prob=0.50, mh_retries=4, max_levels=8, adaptive_stop=true` | 20000 次独立 MC | `[101, 202, 303, 404, 505]` |

表中的参数是已完成正式多策略验证时的冻结执行配置；当前 cut-in YAML 的唯一默认配置也使用 `N=1000, p0=0.10, adaptive_stop=true`。正式主表统计量由上述冻结配置产生，不得与冻结 OAT 结果混合。

MC 的角色是估计器正确性参考，并非多 ADS 主结论所必需的第二次全重复。资源不足时，MC 至少在 IDM 两事件和估计器对比指定的 ADS 上完整执行；四 ADS 主表仍必须完成五个 SS seeds。任何未完成的 MC 单元在表中标为 `not run`，不可用旧的不同预算结果替代。

## 3. 输出目录、实现位置与通用验收接口

### 3.1 输出目录

所有新结果写入版本化目录，默认：

```text
results/multi_policy_validation/
  frozen_inputs.json
  experiment_manifest.json
  ads/{policy}/{event}/seed_{seed}/...
  sensitivity/{event}/{parameter}/{value}/seed_{seed}/...
  ablation/{variant}/{event}/seed_{seed}/...
  estimators/{event}/{policy}/{method}/seed_{seed}/...
  tables/*.csv
  figures/*.png
  paper_assets/README.md
```

每个运行目录至少保留原始 `latent_subset_summary.json` 或对应估计器 summary、有效配置 JSON/YAML、日志、运行时间和 exit status。不要把大规模可重建的动作数组、GIF 或 checkpoint 加入 Git；保留 manifest、summary、CSV 和生成图。

### 3.2 最小实现策略

- 新增一个轻量的 revision experiment driver 与 summary/plot builder；它应动态调用已有 runner，而不是复制四个 `latent_subset_runner.py`。
- 可在 `tools/` 放置跨 ADS 共用的实验协议、manifest、统计汇总和 IS 代码；入口脚本可放在现有 `results/` 脚本模式附近。禁止创建仅转发旧函数的兼容 wrapper。
- 每次运行仅把覆盖后的配置写入 revision 输出目录，再传入 runner；不得改写八个正式 baseline YAML，也不得将结果写回各 ADS 的默认 `results/`。
- 任何新配置字段必须写入有效配置和 summary。对不支持 CLI 覆盖的 IDM 入口，driver 应直接调用现有 Python API，而非人工编辑默认 YAML。

### 3.3 每个 summary 的最低字段

除现有字段外，revision summary 必须包含：

```text
revision_id, experiment_group, variant, policy, event_type, seed,
effective_config, frozen_input_id, started_at, finished_at, wall_seconds,
closed_loop_evaluations, probability, ci95/replicate statistics,
failure_threshold, evt_return_level_target, reliability,
context_sampling_mode, status, failure_reason
```

IS 还必须包含 `pilot_budget`、`evaluation_budget`、总预算、proposal 参数、`log_weight` 摘要、ESS、最大归一化权重、有效失效样本数和权重数值检查结果。

## 4. 工作包 A：多 ADS 测试

### 4.1 实验矩阵

对 `IDM`、`A2C`、`PPO`、`SAIRL` × `following`、`cut-in`，使用第 2.4 节的主 SS 配置和五个独立 seeds。输出八个 policy-event 单元，每个单元五次完整 SS 运行。

IDM 是规则控制基线；A2C/PPO/SAIRL 是现有学习策略。论文应准确称它们为“不同控制策略/策略实现”，而非工业级 ADS 的全面代表。

### 4.2 统计与报告

每个 policy-event 单元报告：

- 五次 `\hat p` 的均值、中位数、标准差，以及以 seed 为独立重复单元的 95% t 区间；
- 每次运行的现有 SS 解析式 SE/RSE，但它只作诊断；主不确定性使用跨 seed 统计；
- 实际 closed-loop evaluations、层数、最终层 failure fraction、MH acceptance、unique context/state 数和 reliability 状态；
- exposure 换算得到的风险强度与重现里程；当 reliability 非 `pass` 时，该换算必须标注为不可作为正式里程结论；
- 同一事件内的 policy 间相对风险比及由 seed 配对 bootstrap 得到的区间；若置信区间跨 1，只能表述为“未观察到显著差异”。

主表不得以原始 TTC、碰撞率或一个 policy 的 reward 取代 `p_{a,e}`。可将碰撞率、near-collision 率、最小 gap/TTC 放在补充表作为案例描述，但不改变失效定义。

### 4.3 工作包 A 的验收

- [ ] 八个 policy-event 单元均完成五个 seed，且每个有效运行均包含完整输入审计信息；
- [ ] 同一事件的有效配置除 `policy` 和 policy checkpoint 外一致；自动 fairness check 必须验证这一点；
- [ ] 所有 `reliability.status != pass` 的单元在表和图中可见，不允许剔除；
- [ ] 输出 `tables/multi_ads_summary.csv`、`tables/multi_ads_seed_level.csv`、`figures/multi_ads_probability.png` 和 `figures/multi_ads_return_mileage.png`；
- [ ] 至少抽样回放每个事件中风险最高的两个 policy-case，确认无重放分数漂移或场景语义错误。

## 5. 工作包 B：子集模拟参数敏感性

### 5.1 目的与固定对象

本工作包只使用 `IDM`，因为要验证的是估计器而不是不同 policy 的风险高低。运行 following 与 cut-in 两个稀有事件，固定第 2.2 节的所有输入、风险阈值和 policy，只逐项改变 SS 超参数。

### 5.1a 当前实现与简洁约定

本工作包已在 `IDM_subset/` 中落地，且不改动既有 `src/`、`scripts/`、基础 YAML
和默认结果目录：

```text
IDM_subset/experiments/highd_ss_sensitivity/
  sensitivity_spec.py       # 网格、seed、输出命名的唯一来源
  run_experiments.py        # 唯一运行入口，直接调用已有 runner
  summarize_results.py      # 唯一跨 seed 汇总与绘图入口
  following/README.md       # 跟驰事件说明
  cutin/README.md           # cut-in 事件说明

IDM_subset/results/ss_sensitivity/
  README.md                 # 人工阅读索引
  experiment_manifest.json  # 冻结输入哈希和完整 OAT 设计
  run_plan.csv              # 64 个 seed 级单元的状态
  summary_status.json       # 当前结果是否可用于论文
  references/ runs/ tables/ figures/
```

实现遵守以下简洁性边界：

- 网格、seed 和路径命名只在 `sensitivity_spec.py` 定义；不为 following/cut-in
  复制 driver 或 YAML；
- 每个 seed 仅写覆盖后的 `effective_config.json`，不改写基线 YAML；
- `run_plan.csv` 与 manifest 是初始化即有的计划/审计元数据；`tables/` 仅在出现
  执行记录后写入，`figures/` 仅在至少一个有效 SS 估计后写入，避免把空计划当结果；
- 当前实际状态必须读取 `summary_status.json`。当前代码已冻结 64 个 SS 单元，但尚
  未产生 MC 或有效 SS 数值，因此不能据此写出任何稳定性结论。

### 5.2 受控的一因素网格

使用 one-factor-at-a-time 设计。每个 setting 运行 seeds `[101, 202, 303]`；默认 setting 另运行到五个 seeds。`mh_retries` 固定为主配置，避免将过多维度混入有限预算。

| 参数 | following | cut-in | 基准 |
| --- | --- | --- | --- |
| `num_samples` | 1000, 3000, 5000 | 500, 1000, 2000 | 3000 / 1000 |
| `p0` | 0.10, 0.20, 0.30 | 0.05, 0.10, 0.20 | 0.20 / 0.10 |
| `proposal_std` | 0.06, 0.12, 0.24 | 0.05, 0.10, 0.20 | 0.12 / 0.10 |
| `context_refresh_prob` | 0.30, 0.50, 0.70, 0.90 | 0.25, 0.50, 0.75, 0.90 | 0.70 / 0.50 |

无需做全因子组合。若某一 setting 连续两个 seed 出现运行异常、context/state 坍缩或无 elites，可停止该 setting 的剩余 seed，但必须保存错误日志并在热图中标为失败。

### 5.3 参考值、评价指标与判读规则

- following 以本轮 IDM 200000-sample MC 为参考；cut-in 以本轮至少 20000-sample MC 为参考。报告 Wilson CI，不把单点 MC 当作真值。
- 对每个 setting 报告跨 seed 的 `\hat p`、相对偏差（相对 MC 点估计）、跨 seed CoV、总闭环调用次数、每个有效估计的 RSE、层数、接受率和 reliability。
- 绘制：`probability_vs_parameter`、`closed_loop_evaluations_vs_parameter`、`acceptance_and_diversity_vs_parameter` 三组图；横轴为参数值，两个事件分面。
- “稳健”只可定义为：主设置的跨 seed 95% 区间与 MC Wilson 95% 区间重合，且全部五次运行通过 reliability；对其他 setting 只报告其稳定区间或失败边界，不能要求所有极端设置均成功。

### 5.4 工作包 B 的验收

- [ ] 两个事件的四个参数均有完整的 setting-level CSV；
- [ ] 默认设置完成五个 seed，其他未提前失败的设置完成三个 seed；
- [ ] 每个失败设置都有明确 `failure_reason`，没有空单元；
- [ ] 两个事件均产生参数敏感性图和一张可写入论文的结论表；
- [ ] 论文文字仅就观察到的“稳定参数区间”作结论，且承认 MCMC 相关性由跨 seed 重复补充评估。

## 6. 工作包 C：关键模块消融

### 6.1 原则

消融的目标不是让所有变体估计同一个严格概率；移除 Copula 或 diffusion 后，测试分布本身会变化。因此必须分别报告“统计/执行贡献”和“闭环风险结果”，不得只依据 `\hat p` 的升降宣称某模块更好。

全部消融先在 IDM 的两个事件上执行，使用主 SS 参数和 seeds `[101, 202, 303]`。若资源允许，再以 PPO cut-in 复核一个代表学习策略；它是加分项而非完成门槛。

### 6.2 必做变体

| 变体 | 保持不变 | 替换内容 | 目标与最低指标 |
| --- | --- | --- | --- |
| `full_tread` | 全部资产 | 无替换 | 共同参照 |
| `no_copula_empirical_context` | EVT、冻结 diffusion、ADS、风险阈值、闭环环境 | 用 `tail_contexts.npz` 中对应 independent-tail-peak empirical rows 的均匀有放回抽样代替 `TailContextDistribution` | context 重复率、唯一 context 数、条件变量覆盖/相关结构、闭环可执行率、`\hat p`；概率目标须改名为 empirical-tail-context 条件概率 |
| `no_diffusion_action_replay` | Copula context、EVT、ADS、风险阈值、闭环环境 | 对每个 sampled context 的 `base_event_id` 回放其对齐的原始 highD 对手车动作计划；不得用另一条任意动作替换失败样本 | 动作/轨迹可用率、无效或长度不匹配率、场景语义率、动作多样性、`\hat p`；概率目标注明为 Copula-context + empirical-action-replay |

`no_copula` 与 `no_diffusion` 是本轮必须完成的两个主要分布构建消融。`full_tread` 结果可复用工作包 A/B 中同 seed 的 IDM 运行，但必须由 manifest 明确指向同一冻结输入和配置。

### 6.3 原始动作回放实现约束

`no_diffusion_action_replay` 必须从已有 highD 缓存重建，而非重新训练生成器：

- following 从 `results/highd_events/following_event_segments.npz` 和 tail context 的 `base_event_id` 提取与 anchor 对齐的 lead 状态，再用 diffusion 数据构造时相同的动作定义得到 125-step jerk/action plan；
- cut-in 从 `results/highd_events/cutin_event_contexts.npz` 的本地状态及 `base_event_id` 提取与 anchor/cross 对齐的 target `[ax, ay]` plan，得到 100-step 计划；
- 使用与主线一致的动作单位、采样频率、物理 clipping 和 semantic gate；不允许为了填满样本数对缺失计划随机替换为不匹配事件；
- 若 `base_event_id` 对应计划缺失、窗口不足或不满足 cut-in 语义，记录该样本为 invalid。必须报告 invalid rate；不得静默重采样后假装有效率为 100%。

### 6.4 可选但低成本的 EVT 校准诊断

不把它扩展成新的风险指标实验。可在已有 highD event score 上增加 `POT-GPD` 与同一目标超越率的经验分位数 bootstrap 对照，报告阈值均值、区间与可用样本数。若目标超越率低到经验分位数没有足够样本，结果应明确写为“经验法不可稳定外推”，不能伪造一个阈值或把该诊断当作完整概率消融。

### 6.5 工作包 C 的验收

- [ ] 两个必须变体在 following、cut-in 各完成三个 seed，或有可审计的不可执行原因；
- [ ] 输出 `tables/ablation_summary.csv`，列出变体的**概率对象**、样本可用率、物理/语义有效率、重复/多样性诊断、`\hat p` 及其跨 seed 区间；
- [ ] 输出一张 distribution/validity 对比图和一张闭环风险结果图；
- [ ] 论文表格脚注明确：当采样分布变体改变时，概率值不构成严格同一 estimand 的优劣比较；
- [ ] 不新增单一 TTC 对照，也不改动复合风险公式。

## 7. 工作包 D：MC、IS 与 SS 的横向对比

### 7.1 比较问题和实验单元

目标是在**同一** `P_e=D_e\otimes\mathcal N(0,I)` 和同一 failure threshold 下比较三类概率估计器：

1. 独立 MC；
2. 两阶段、defensive-mixture cross-entropy importance sampling（CE-IS）；
3. 本文 joint-space subset simulation（SS）。

优先选择 IDM-following 和 IDM-cut-in，因为两者已有低概率基线，且能显示稀有事件加速。若完成时间允许，再对 PPO cut-in 加一组；不以增加更多 ADS 数量替代对基础比较的重复和审计。

### 7.2 IS 的正确采样空间

当前 context provider 是一个可索引的确定性 `TailContextDistribution`：基分布在 index
`I\in\{0,\ldots,M-1\}` 上均匀，`M=population_size`；同一 index 总是映射到同一 Copula 条件 context。扩散 latent 为 `z\sim\mathcal N(0,I)`。故 IS 必须在 `(I,z)` 上实现：

\[
p(I,z)=M^{-1}\phi(z),\qquad
\hat p_{\rm IS}=\frac1n\sum_{j=1}^n
\mathbf 1\{R(I_j,z_j)\geq\gamma^\star\}\frac{p(I_j,z_j)}{q(I_j,z_j)}.
\]

禁止在 DDIM 解码后的动作、轨迹或风险分数空间假设概率密度。这样既避免不可求的 implicit trajectory likelihood，也与当前 SS/MC 的 joint-space 概率对象完全一致。

### 7.3 CE-IS 设计

实现一个最小但严格的两阶段 defensive-mixture CE-IS：

1. **Pilot 阶段**：从 `p(I,z)` 独立抽样，following 用 5000 次、cut-in 用 2000 次闭环评估。用风险最高的固定比例样本拟合 proposal；若直接 failure 数不足，使用固定分位数 elite，而不是根据最终结果反复调参。
2. **Proposal**：
   \[
   q(I,z)=q_I(I)q_z(z),
   \]
   其中 `q_I` 为基均匀分布与 pilot elite index categorical distribution 的 defensive mixture，`q_z` 为标准高斯与对 elite 拟合的 diagonal Gaussian 的 defensive mixture。推荐初始 defensive 权重 `epsilon=0.10`，所有参数和随机种子写入 summary。
3. **独立 evaluation 阶段**：pilot 完成并锁定 proposal 后，使用新 seed 从 `q` 抽样，计算精确 `log p - log q`。pilot 结果不得直接混入最终 IS 估计，以避免自适应偏差。
4. **数值稳定性**：在 log-space 计算混合密度与权重；保存未归一化 log-weight 摘要。若有 NaN/Inf、proposal 支持缺失或 ESS 非正，运行必须失败并给出原因，不能回退成无权重的“危险样本比例”。

所有方法在比较中按**总闭环调用次数**计预算；CE-IS 的 pilot 也计入。初始预算曲线使用 following `{5000, 10000, 20000, 40000}`、cut-in `{2000, 5000, 10000, 20000}`；若 MC 参考预算更大，单独标注为 `reference`，不计入同预算曲线。

### 7.4 衡量指标和重复

每个 `event × method × budget` 独立运行 seeds `[101, 202, 303, 404, 505]`。输出：

- `\hat p`、跨 seed 均值/标准差/95% 区间；
- 相对误差或绝对误差相对于高预算 MC reference；reference 自身须报告 Wilson 区间；
- 闭环调用数、wall time、每个失效样本的成本；
- IS 的 ESS、ESS/N、最大归一化权重、权重 CoV、有效 failure 权重数；
- SS 的层数、接受率、唯一 context/state、reliability；
- 95% 区间对 MC reference 的覆盖情况。

论文主图为“误差/RSE 与闭环调用次数”的对数横轴曲线，并配一张方法诊断表。若 CE-IS 出现权重退化，这本身是结果，但只能如实描述为该 proposal 在当前高维 joint space 下不稳定；不得由此宣称所有 IS 方法都无效。

### 7.5 工作包 D 的验收

- [ ] MC、CE-IS、SS 均明确写出同一个 probability target、同一 event、同一 policy、同一 threshold；
- [ ] IS 具有独立 pilot/evaluation、可重算的 `log p`、`log q` 和权重审计；
- [ ] 两个 IDM 事件的每个预算点均完成五个 seeds，或记录可复现的失败原因；
- [ ] 输出 `tables/estimator_comparison.csv`、`tables/is_diagnostics.csv`、`figures/estimator_error_vs_budget.png` 和 `figures/is_weight_diagnostics.png`；
- [ ] 所有预算和加速倍数以实际 closed-loop evaluations 为分母，绝不只以 nominal `N` 计数。

## 8. 执行顺序、停止规则与资源控制

按以下顺序执行，确保可尽早发现阻塞而不浪费大量 GPU/CPU 时间：

1. **预检与冻结**：检查四个 checkpoint、highD 资产、当前 MC/SS smoke test、输出隔离和 replay 一致性；生成 frozen manifest。
2. **统一 driver**：能在任意一个 policy-event 上以覆盖配置写入 revision 输出，且不修改原有结果。
3. **工作包 A**：先跑 IDM 两事件五 seeds，后跑 A2C/PPO/SAIRL；每个 policy-event 首次运行先做小预算 smoke test。
4. **工作包 B**：IDM 两事件 OAT 网格；主设置不稳定时先修复/解释 SS 诊断，不进入大规模 IS。
5. **工作包 C**：先完成 empirical-context，再实现 action replay；无法与 source event 对齐的 action replay 必须及时停止并如实报告，不能用未验证替代物凑消融。
6. **工作包 D**：先在 following 做小预算 MC/CE-IS/SS smoke test，验证权重守恒与 summary，再展开预算曲线和 cut-in。
7. **汇总与论文资产**：只从 manifest 中状态为 `completed` 的原始 summary 生成表图；生成前做自动 completeness/fairness audit。

资源限制下，允许先完成 A、B、C 的 IDM 两事件和 D 的两个小预算 smoke test；但目标不得标为完成，直到第 0.3 节的四个工作包验收全部满足。禁止为赶进度减少 seeds 后仍把结果称为“重复实验”。

## 9. 论文交付物与审稿回复映射

| 审稿意见片段 | 本轮交付物 | 可支持的限定结论 |
| --- | --- | --- |
| 多类自动驾驶系统对照 | 多 ADS 主表、policy 风险图、统一 manifest | 方法可在现有四种高速公路控制策略上统一评估 |
| 各模块剥离对比 | Copula/动作 diffusion 回放消融表与有效性图 | 两个分布构建模块对覆盖、可执行性和风险测试输入有可观察贡献 |
| 子集模拟参数敏感性 | OAT 网格、稳定区间图、跨 seed 统计 | 当前两类事件下 SS 在报告的稳定参数区域内结果一致 |
| 重要性采样等同类方法 | MC–CE-IS–SS 预算/误差曲线及 IS 诊断 | 在同一条件概率对象下比较估计效率和权重退化情况 |
| 城市多场景 | 无新增实验 | 明确列入未覆盖限制和后续工作，不能声称已解决 |
| 复合风险 vs TTC | 无新增实验 | 保持现有风险定义；不作“优于 TTC”的新结论 |

建议新增论文内容：一个“多 ADS 与方法泛化”小节、一张“模块消融与 SS 参数稳健性”表/图组、一张“MC/IS/SS 效率比较”图。讨论部分增加一段明确的范围限定。

## 10. 最终验收清单

### 工程与可复现性

- [ ] `results/multi_policy_validation/experiment_manifest.json` 可枚举所有计划单元、状态、配置、输入哈希和输出路径；
- [ ] 从空的 revision 输出目录运行时，任意单元不会写入或覆盖原有 ADS `results/`；
- [ ] 新代码使用 `conda activate tread` 环境，至少通过 import/smoke test；
- [ ] 同一 revision 的 full-method 同 event 跨 ADS fairness check 通过；
- [ ] summary/汇总器能检测缺失单元、配置漂移、无效 reliability 和 IS 权重异常。

### 科学性

- [ ] 多 ADS、参数敏感性、模块消融、MC/IS/SS 四个工作包均通过各自验收；
- [ ] 所有概率对象、阈值和条件分布在表注中明确；
- [ ] 跨 seed 不确定性进入主表或图；
- [ ] 消融分布变化和 IS 失败均被如实呈现；
- [ ] 没有把 conditional tail probability 误写成全自然驾驶碰撞概率；
- [ ] 没有新增城市泛化或 TTC 优势的无证据表述。

### 论文资产

- [ ] `paper_assets/README.md` 将每张图/表映射到原始 CSV 和对应审稿意见；
- [ ] 图表达到当前论文样式要求（300 dpi、统一字体/标注、中文/英文图注一致）；
- [ ] 输出一份 `tables/revision_claims.md`：逐条写明可以写入论文的结论、不能写的结论及其来源；
- [ ] 最终运行 `git status --short`，确认未意外提交 checkpoint、大型可再生成 NPZ、GIF、缓存或修改用户现有结果。
