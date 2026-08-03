# IDM 结果目录

本目录存放 IDM baseline 与高维 SS 参数敏感性实验的结果。各子目录由对应脚本直接
写入，彼此不应混用。

| 目录 | 生成入口 | 内容 |
| --- | --- | --- |
| `following/` | `IDM_subset/scripts/run_subset_following.py` | following 默认 SS 结果与 final-level 回放 |
| `cutin/` | `IDM_subset/scripts/run_subset_cutin.py` | cut-in 默认 SS 结果与 final-level 回放 |
| `monte_carlo_following/` | `IDM_subset/scripts/run_monte_carlo_following.py` | following 独立 MC 基准 |
| `monte_carlo_cutin/` | `IDM_subset/scripts/run_monte_carlo_cutin.py` | cut-in 独立 MC 基准 |
| `ss_sensitivity/` | `IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py` | 完整 highD IDM SS OAT 参数敏感性实验 |

默认 baseline 目录用于常规 IDM 对照；`ss_sensitivity/` 是单独冻结的参数敏感性设计，
包含独立 MC 参考、逐 seed SS 审计、汇总表和图。详细阅读顺序及保留策略见
`ss_sensitivity/README.md`。

所有结果中的持久化文件引用均为相对路径。除非明确放弃复核与复现，不要删除
`ss_sensitivity/runs/`、`ss_sensitivity/references/` 或各 baseline 的原始样本与回放文件。

## 版本控制

`.gitignore` 会提交轻量的 README、JSON、CSV 和 PNG，从而保留实验规格、逐 seed
审计摘要、表格和图；大且可再生成的 `*_samples.npz`、模型权重、日志和回放媒体保持在
本地结果目录，不纳入 Git。忽略规则不会改变已经被 Git 跟踪的历史文件。
