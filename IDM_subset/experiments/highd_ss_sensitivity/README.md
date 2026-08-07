# highD IDM 子集模拟：敏感性与默认重复实验

本目录只有一个调度入口：`run_experiments.py`。它始终复用
`IDM_subset.src.latent_subset_runner`，不会修改基础 YAML，也不会为 following 和 cut-in
复制两套 SS 实现。入口支持两个彼此隔离的工作流：

| 工作流 | 目的 | 输出根目录 | 是否执行 MC |
| --- | --- | --- | --- |
| `frozen-oat`（默认） | 复现冻结的单因素（OAT）参数敏感性设计 | `IDM_subset/results/ss_sensitivity/` | 仅在显式传入 `--run-reference-mc` 时执行 |
| `default-repeats` | 验证当前 YAML 默认配置的 5 个独立 SS seed | `IDM_subset/results/following_default_repeats/`、`IDM_subset/results/cutin_current_default_repeats/` | 否，只读取既有的配对 MC summary |

二者不能混合：冻结 OAT 的布局、manifest 和配置快照保持不可变；默认配置在未来可以更新，
但新配置不能被写入或复用在冻结 OAT 目录中。

## 当前默认配置与冻结快照

当前默认配置来自基础 YAML；`default-repeats` 使用的正是这些值。

| 事件 | 基础 YAML | 当前 SS 默认值 | 配对 MC |
| --- | --- | --- | --- |
| following | `IDM_subset/scripts/configs/latent_subset_following.yaml` | `N=3000, p0=0.20, proposal_std=0.12, context_refresh_prob=0.70, retries=6, max_levels=8, adaptive_stop=false` | `IDM_subset/results/monte_carlo_following/`，200,000 samples |
| cut-in | `IDM_subset/scripts/configs/latent_subset_cutin.yaml` | `N=1000, p0=0.10, proposal_std=0.10, context_refresh_prob=0.50, retries=4, max_levels=8, adaptive_stop=true` | `IDM_subset/results/monte_carlo_cutin/`，20,000 samples |

冻结 OAT 仍保留其建立时的设计。其 following 与 cut-in 默认快照分别与当前基础 YAML
配置相同；冻结设计和当前默认重复的统计目的不同，二者仍须分开解释。

## 工作流 A：冻结 OAT 敏感性设计

该设计以 `sensitivity_spec.py` 中的 `GRID`、`DEFAULT_SEEDS=(101,202,303,404,505)` 和
`SETTING_SEEDS=(101,202,303)` 为唯一规范：每个事件有 5 个默认设置 seed，以及其余 OAT
设置各 3 个 seed，共 64 个 SS 单元。

变化的参数只有 `num_samples`、`p0`、`proposal_std` 和 `context_refresh_prob`；其余风险
定义、highD 条件分布、扩散 checkpoint、IDM policy、阈值与执行协议保持冻结。

```bash
conda activate tread

# 建立/校验冻结 manifest 与 run plan，不执行仿真
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow frozen-oat --dry-run

# 执行或续跑所有冻结的 SS 单元，并在需要时先运行计划内 MC
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow frozen-oat --event all --run-reference-mc

# 选择一个已冻结的单元；--overwrite 才会重跑该单元
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow frozen-oat --event following --setting default --seed 101
```

完成后，`summarize_results.py` 会生成冻结 OAT 的表、图和 `summary_status.json`。它只从
`experiment_manifest.json` 的 `oat_grid.*.defaults` 读取冻结 OAT 基准，不会用当前基础 YAML
重新定义默认值。阅读或引用该批结果前，必须先检查 `summary_status.json`，而不是只读取某次
seed 的 summary。

### 路径可移植性

从统一运行器新写入的 manifest、`run_plan.csv`、seed 表和 JSON summary 中，仓库内路径均为
相对路径并统一使用 POSIX 分隔符 `/`；YAML 的 `output_dir` 也相对各自基础 YAML 解析。因此，
在保持相同仓库目录结构的前提下，可在 Windows 与 Linux 间迁移。汇总器同时兼容历史
`run_plan.csv` 中的 Windows 分隔符 `\`，但不要手工写入绝对路径或混用两种分隔符。

## 工作流 B：当前默认配置的 5 种子重复

此工作流要求显式选择一个事件，并固定使用 5 个预注册 seed：101、202、303、404、505。

```bash
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow default-repeats --event following

python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow default-repeats --event cutin
```

它不接受 `--event all` 或 `--setting`，也拒绝 `--run-reference-mc`。MC 应分别通过
`IDM_subset/scripts/run_monte_carlo_following.py` 与
`IDM_subset/scripts/run_monte_carlo_cutin.py` 独立运行。运行器仅在汇总时读取下面两个
canonical MC summary：

```text
IDM_subset/results/monte_carlo_following/latent_monte_carlo_summary.json
IDM_subset/results/monte_carlo_cutin/latent_monte_carlo_summary.json
```

5 个 seed 按顺序执行；`--workers` 仅用于冻结 OAT 中不同设置的并发调度。可使用
`--seed {101,202,303,404,505}` 选择一个 seed，或使用 `--overwrite` 有意重跑已选 seed。
对已有结果，运行器比较操作性 SS 配置哈希（忽略输出位置、seed、MC 设置和历史元数据）；
不匹配时会拒绝复用，防止把不同的当前默认配置混入同一重复目录。

每次执行会在对应结果根写入或更新：

```text
default_repeat_manifest.json
seed_results.csv
summary.json
seed_101/ ... seed_505/
```

其中 `summary.json` 汇总跨 seed 均值、样本标准差、95% t 区间、平均闭环评估数，以及与
存在的 MC summary 的差异和区间重叠。根目录中若仍保留早期迁移的 `manifest.json`，它是
历史校准记录，不是统一运行器的 canonical manifest。

## 执行参数

两个工作流都可指定以下执行参数；它们影响计算组织而不是 OAT 因子：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--workers` | 4 | 冻结 OAT 中并发的独立设置数；默认重复仍串行。 |
| `--rollout-workers` | 2 | 每个 SS 任务的 CPU 闭环 rollout worker 数。 |
| `--mcmc-batch-size` | 64 | 每批 GPU 解码的独立 MH proposal 数。 |
| `--population-batch-size` | 64 | 初始种群、MCMC 与 MC 的 GPU 解码批量。 |
| `--rollout-prefetch-batches` | 2 | CPU rollout 期间保留的已解码批次数。 |

`--mcmc-batch-size` 只接受 1 或 64；`--workers` 只接受 1、2 或 4。

## 结果边界

- `results/ss_sensitivity/` 仅是冻结 OAT 结果；其 cut-in 快照不能替换当前默认 cut-in 重复。
- `results/*_current/` 是单次常规 SS 输出，适合回放与诊断，不能代替跨 seed 不确定性评估。
- `results/*_default_repeats/` 是当前默认配置的 5 个独立 SS 结果；其配对 MC 仅位于
  `results/monte_carlo_following/` 或 `results/monte_carlo_cutin/`。
- 不要编辑任何 seed 下的 `effective_config.json`；它是运行时配置快照和复用校验的依据。

各结果根目录的实际用途和保留策略见 [`../../results/README.md`](../../results/README.md)。
