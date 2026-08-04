# IDM_subset：IDM 长尾安全概率评估

`IDM_subset/` 是本仓库中 IDM ego policy 的 highD 长尾安全评估实现。它在固定的
highD 尾部 scenario-condition 分布、冻结的条件扩散模型和冻结的 EVT 风险阈值下，估计
IDM 对扩散生成对手车轨迹的闭环响应风险。

```text
scenario condition ~ highD tail scenario-condition distribution
z ~ N(0, I)
adversary actions = deterministic DDIM(scenario condition, z)
ego actions = IDM(tools/idm_ego.yaml)
score = S_EVT(Y_sim)
failure = score >= S_EVT(x_c), where x_c = 5.0
```

因此报告的概率是

```text
P(failure | sampled from the highD tail scenario-condition distribution)
```

它不是完整 highD 自然驾驶分布上的直接碰撞概率。只有当 exposure 的可靠性条件满足时，
`global_risk_exposure_comparison.*` 才将此尾部条件概率换算为全局 highD 暴露强度。

## 事件与当前默认配置

当前可执行配置的唯一来源是下列 YAML；请不要以历史结果目录中的
`effective_config.json` 反推当前默认值。

| 事件 | 基础 YAML | 仿真设置 | 当前 SS 默认配置 | 配对独立 MC |
| --- | --- | --- | --- | --- |
| following | `scripts/configs/latent_subset_following.yaml` | 125 steps、1 lane、kinematic bicycle | `N=3000`，`p0=0.20`，`proposal_std=0.12`，`context_refresh_prob=0.70`，`mh_retries=6`，`max_levels=8`，`adaptive_stop=false` | 200,000 samples |
| cut-in | `scripts/configs/latent_subset_cutin.yaml` | 100 steps、2 lanes、point mass | `N=2000`，`p0=0.05`，`proposal_std=0.10`，`context_refresh_prob=0.50`，`mh_retries=4`，`max_levels=8`，`adaptive_stop=false` | 20,000 samples |

两个事件均使用 50 个 DDIM evaluation steps，并固定 IDM ego 的参数文件
`tools/idm_ego.yaml`。配置引用缺失的 checkpoint 或输入文件会直接报错，不回退到旧权重。

## 常规单次运行

从仓库根目录运行：

```bash
conda activate tread

# following
python IDM_subset/scripts/run_monte_carlo_following.py
python IDM_subset/scripts/run_subset_following.py
python IDM_subset/scripts/play_final_level_following.py --no-gif

# cut-in
python IDM_subset/scripts/run_monte_carlo_cutin.py
python IDM_subset/scripts/run_subset_cutin.py
python IDM_subset/scripts/play_final_level_cutin.py --no-gif
```

常规 SS 和 MC 入口均读取对应 YAML；MC 入口还可覆盖 `--num_samples`、`--seed` 和
`--output_dir`。单次 SS 输出只适合运行诊断、回放和示例展示；它不是默认配置的跨随机种子
不确定性评估。

## 当前默认配置的 5 种子重复

当前默认配置的正式重复由同一个统一入口执行，而不是由事件专用校准脚本执行：

```bash
# 五个预注册 seed：101、202、303、404、505
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow default-repeats --event following

python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow default-repeats --event cutin
```

此工作流只运行 SS，按顺序执行 5 个 seed，避免同一 GPU 上的独立 MCMC 链相互竞争；
`--workers` 不改变这一点。它不接受 `--setting`，也不执行 MC。配对 MC 必须先由上述
事件专用 MC 入口写入：

```text
results/monte_carlo_following/latent_monte_carlo_summary.json
results/monte_carlo_cutin/latent_monte_carlo_summary.json
```

重复结果分别写入 `results/following_default_repeats/` 与
`results/cutin_default_repeats/`。运行器会校验每个已有 seed 的操作性 SS 配置；若当前 YAML
已改变，不会静默混用旧结果，而会拒绝复用该目录。

## 冻结的 OAT 参数敏感性实验

`experiments/highd_ss_sensitivity/` 还维护一套独立的、冻结的 one-factor-at-a-time
(OAT) 敏感性设计，结果固定在 `results/ss_sensitivity/`。该设计有 64 个 seed 级 SS
单元：每个事件的默认设置运行 5 个 seed，其他单因素设置各运行 3 个 seed。

```bash
# 执行或续跑冻结的 OAT 设计；可选地先生成其计划内 MC 参考
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow frozen-oat --event all --run-reference-mc
```

该 OAT 目录必须与当前默认重复目录分开解释。特别是它冻结的 cut-in 基准快照为
`N=1000, p0=0.10, adaptive_stop=true`，不是当前 cut-in YAML 的默认配置；不能把两者的
结果合并平均或相互覆盖。

详细的运行约束、可选参数和结果读取顺序见
[`experiments/highd_ss_sensitivity/README.md`](experiments/highd_ss_sensitivity/README.md)。

## 当前已保存结果的解释

| 对象 | 估计 | 说明 |
| --- | ---: | --- |
| following 单次 SS | 0.00249067 | `following_current/` 的单次运行；内部 SS 标准误为 0.00006763。 |
| following 5-seed SS | 0.00249333 | 5 次均值，跨 seed 标准差为 0.00053073。 |
| following MC | 0.00241000 | 200,000 次独立 MC；标准误为 0.00010964。 |
| cut-in 单次 SS | 0.00900000 | `cutin_current/` 的单次运行，仅作诊断/回放示例。 |
| cut-in 5-seed SS | 0.00672500 | 当前默认配置的 5 次均值，跨 seed 标准差为 0.00193488。 |
| cut-in MC | 0.00695000 | 20,000 次独立 MC；标准误为 0.00058744。 |

默认配置的主要 SS 不确定性证据是跨 seed 均值、标准差和 t 区间，而不是某一个 seed
summary 中的二项式近似标准误。MC 的一次大样本估计则为 IID Bernoulli 估计，标准误和
置信区间可直接由样本总数及失效数计算；额外 MC seed 可用于质量复核，但不是计算该一次
估计置信区间的前提。

## 主要文件与结果目录

```text
scripts/configs/latent_subset_following.yaml      # 当前 following 配置
scripts/configs/latent_subset_cutin.yaml          # 当前 cut-in 配置
scripts/run_subset_{following,cutin}.py           # 常规 SS 入口
scripts/run_monte_carlo_{following,cutin}.py      # 事件专用独立 MC 入口
scripts/play_final_level_{following,cutin}.py     # 最终层回放入口
src/latent_subset_runner.py                        # SS / MC 共享运行器
src/subset_simulation.py                            # 子集模拟与 MCMC 实现
experiments/highd_ss_sensitivity/                  # 冻结 OAT + 当前默认重复的统一调度器

results/following_current/                         # following 单次常规 SS
results/cutin_current/                             # cut-in 单次常规 SS
results/following_default_repeats/                 # following 当前默认配置的 5 个 seed
results/cutin_default_repeats/                     # cut-in 当前默认配置的 5 个 seed
results/monte_carlo_following/                     # following 200k MC
results/monte_carlo_cutin/                         # cut-in 20k MC
results/ss_sensitivity/                            # 冻结的 OAT 参数敏感性结果
```

`*_samples.npz`、回放和日志是可重建的大体积产物；配置、summary、CSV、manifest 与状态文件
保留用于结果追溯。各结果目录的用途见 [`results/README.md`](results/README.md)。
