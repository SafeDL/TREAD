# highD IDM 子集模拟参数敏感性实验

本目录实现 IDM 在 highD `following` 与 `cutin` 事件上的子集模拟（SS）
one-factor-at-a-time（OAT）参数敏感性实验。它只负责冻结实验设计、调度独立
seed 和汇总结果；SS、扩散采样与闭环仿真均复用 `IDM_subset/src/` 的共享实现。

## 目录内容

- `sensitivity_spec.py`：唯一的实验规格来源，包括事件、OAT 网格、随机种子、
  MC 参考样本量、结果目录和正式并行配置。
- `run_experiments.py`：创建或验证冻结 manifest，执行可选的独立 MC 参考，并按
  设置/seed 调度 SS 单元。
- `summarize_results.py`：读取运行计划与逐 seed 摘要，生成 CSV 表格、PNG 图和
  `summary_status.json`。

事件差异仅来自基础 YAML，不存在 following/cut-in 的重复 runner：

| 事件 | 基础配置 | 默认 SS 参数 | 独立 MC 样本量 |
| --- | --- | --- | ---: |
| following | `IDM_subset/scripts/configs/latent_subset_following.yaml` | `N=3000, p0=0.20, sigma=0.12, r_c=0.70, retries=6, max_levels=8` | 200,000 |
| cutin | `IDM_subset/scripts/configs/latent_subset_cutin.yaml` | `N=1000, p0=0.10, sigma=0.10, r_c=0.50, retries=4, max_levels=8` | 20,000 |

## 冻结设计

默认设置使用 5 个独立 seed：`101, 202, 303, 404, 505`；每个非默认设置使用
`101, 202, 303`。每个事件有 32 个计划 SS 单元，两个事件共 64 个。

| 参数 | following | cutin |
| --- | --- | --- |
| `num_samples` | 1000, 3000, 5000 | 500, 1000, 2000 |
| `p0` | 0.10, 0.20, 0.30 | 0.05, 0.10, 0.20 |
| `proposal_std` | 0.06, 0.12, 0.24 | 0.05, 0.10, 0.20 |
| `context_refresh_prob` | 0.30, 0.50, 0.70, 0.90 | 0.25, 0.50, 0.75, 0.90 |

非默认设置若连续两次出现执行失败或可靠性失败，剩余 seed 会明确标记为
`skipped_after_two_quality_failures`，而不是被静默删除。已完成单元默认跳过；
manifest 会校验基础配置、输入哈希和 OAT 网格，阻止把不兼容输入混入同一批结果。

## 运行

从仓库根目录运行。正式默认并行配置为：4 个外层任务、每个任务 2 个 CPU rollout
worker、GPU 种群/MCMC 批量均为 64、预取深度为 2。

```bash
conda activate tread
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --event all --run-reference-mc
```

常用命令：

```bash
# 只建立/核验 manifest 与 64 行运行计划，不执行仿真
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py --dry-run

# 仅执行一个冻结单元；已完成单元默认跳过
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --event following --setting default --seed 101

# 仅在明确需要时覆盖所选单元或 MC 参考
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --event cutin --setting proposal_std_0p2 --seed 101 --overwrite
```

| 参数 | 可选值 / 默认值 | 含义 |
| --- | --- | --- |
| `--workers` | `1, 2, 4` / `4` | 并发的独立 SS 设置数 |
| `--rollout-workers` | `0, 1, 2, 4` / `2` | 每个 SS 任务的 CPU 闭环 rollout worker 数 |
| `--mcmc-batch-size` | `1, 64` / `64` | 一次 GPU 解码的独立 MH 提议数 |
| `--population-batch-size` | 正整数 / `64` | 初始种群、MCMC 与 MC 的 GPU 解码批量 |
| `--rollout-prefetch-batches` | 正整数 / `2` | CPU rollout 执行期间保留的已解码批次数 |

非 `--dry-run` 的运行结束后会自动调用汇总器。也可在不重跑仿真的情况下单独重建表图：

```bash
python IDM_subset/experiments/highd_ss_sensitivity/summarize_results.py
```

## 路径与可移植性

基础 YAML 使用相对路径。运行时会在当前检出目录解析实际文件；写入
`effective_config.json`、manifest、状态、摘要和运行计划时，所有文件引用均保存为
POSIX 相对路径：配置输出路径相对其基础 YAML，结果引用相对仓库或结果根目录。因此
结果目录可随仓库在 Windows/Linux 间移动，不依赖盘符、用户名或 Conda 安装位置。

## 结果产物

所有结果写入 `IDM_subset/results/ss_sensitivity/`：

```text
experiment_manifest.json                 # 冻结范围、输入哈希、网格、seed 与执行配置
run_plan.csv                             # 64 个 seed 级单元及其状态
references/{following,cutin}/            # 独立 MC 参考及其配置、摘要、样本
runs/{event}/{setting}/seed_{seed}/      # 单次 SS 配置、状态、摘要、日志、原始样本
tables/
  ss_sensitivity_seed_level_results.csv
  ss_sensitivity_setting_level_summary.csv
  ss_sensitivity_paper_conclusion_table.csv
figures/
  probability_vs_parameter.png
  closed_loop_evaluations_vs_parameter.png
  acceptance_and_diversity_vs_parameter.png
summary_status.json                      # 结果、表格和图是否有效
```

阅读结果时，先检查 `summary_status.json`，再阅读 `tables/ss_sensitivity_paper_conclusion_table.csv`。
默认配置只有在 5 次重复均通过可靠性诊断，且 SS 跨 seed 95% 区间与当前 MC Wilson 95%
区间重叠时，才被标记为 `robust_by_predeclared_rule=true`。
