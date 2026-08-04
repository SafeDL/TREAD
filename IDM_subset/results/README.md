# IDM 结果目录

本目录将三类不同证据明确隔离：单次常规运行、当前默认配置的跨 seed 重复，以及冻结的
OAT 参数敏感性实验。它们不能按目录名相近而混合为同一统计样本。

| 目录 | 生成入口 | 内容与解释 |
| --- | --- | --- |
| `following_current/` | `scripts/run_subset_following.py` | 当前 following YAML 的单次常规 SS 输出和最终层回放。 |
| `cutin_current/` | `scripts/run_subset_cutin.py` | 当前 cut-in YAML 的单次常规 SS 输出。 |
| `following_default_repeats/` | `experiments/highd_ss_sensitivity/run_experiments.py --workflow default-repeats --event following` | 当前 following 默认配置的 5 个独立 seed。 |
| `cutin_default_repeats/` | `experiments/highd_ss_sensitivity/run_experiments.py --workflow default-repeats --event cutin` | 当前 cut-in 默认配置的 5 个独立 seed。 |
| `monte_carlo_following/` | `scripts/run_monte_carlo_following.py` | following 的 canonical 200,000-sample 独立 MC 参考。 |
| `monte_carlo_cutin/` | `scripts/run_monte_carlo_cutin.py` | cut-in 的 canonical 20,000-sample 独立 MC 参考。 |
| `ss_sensitivity/` | `experiments/highd_ss_sensitivity/run_experiments.py --workflow frozen-oat` | 已冻结的 IDM SS OAT 参数敏感性设计、其 MC 参考和汇总图表。 |

## 使用规则

1. `*_current/` 的单次 SS 结果用于诊断、案例和回放，不作为跨随机种子不确定性的唯一证据。
2. 默认配置的比较应读取对应 `*_default_repeats/seed_results.csv` 与 `summary.json`；
   `summary.json` 的 MC 比较只指向上表所列 canonical MC 目录。
3. `ss_sensitivity/` 是单独冻结的 OAT 实验。其 cut-in 基准是校准前快照
   （`N=1000, p0=0.10, adaptive_stop=true`），不能与当前 cut-in 默认重复
   （`N=2000, p0=0.05, adaptive_stop=false`）合并。
4. `default-repeats` 只运行 SS；运行前应先用相应的 `run_monte_carlo_*.py` 准备 canonical
   MC。它对已有 seed 进行配置校验，发现不兼容时会拒绝复用结果。

## 结果根内的文件

当前统一重复运行器在 `*_default_repeats/` 下管理：

```text
default_repeat_manifest.json     # 当前操作性 SS 配置、seed 集、执行参数和 canonical MC 路径
seed_results.csv                 # 逐 seed 的状态、概率、层数、评估量和接受率
summary.json                     # 跨 seed 统计与可用时的 MC 比较
seed_101/ ... seed_505/          # 原始 SS summary、有效配置、状态、统计和样本
```

早期迁移或校准留下的 `manifest.json` 仅保留历史追溯价值；统一运行器的 canonical 文件名是
`default_repeat_manifest.json`。同样，某些迁移来的 `effective_config.json` 可保留原输出路径、
旧的 MC 设置或历史 metadata；复用时运行器只比较实际执行 SS 所需的操作性配置。

`ss_sensitivity/` 的阅读顺序是 `summary_status.json`、`experiment_manifest.json`、
`tables/ss_sensitivity_paper_conclusion_table.csv` 和 `run_plan.csv`。它的具体目录结构和
保留规则见 [`ss_sensitivity/README.md`](ss_sensitivity/README.md)。

## 版本控制与保留

JSON、CSV、README、manifest 和 PNG 应保留，以便审计结果来源。可再生且体积较大的
`*_samples.npz`、回放媒体与日志通常不纳入 Git，但不应因其未跟踪而被误当作无用数据删除。
不要编辑 seed 目录中的 `effective_config.json`；它是运行快照与兼容性校验的证据。
