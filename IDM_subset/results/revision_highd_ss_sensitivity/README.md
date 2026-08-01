# highD IDM 子集模拟参数敏感性：结果索引

这是 `IDM_subset/experiments/highd_ss_sensitivity/` 的唯一结果根目录。它只包含
highD、IDM、following 与 cut-in 的 SS 参数敏感性实验；不混入多 ADS、模块消融、
IS 或新数据集实验。

## 先看哪里

| 目的 | 文件或目录 | 何时出现 |
| --- | --- | --- |
| 确认实验边界、输入资产、哈希和网格 | `experiment_manifest.json` | 初始化后 |
| 查看 64 个 seed 级单元的执行状态 | `run_plan.csv` | 初始化后，逐运行更新 |
| 判断当前汇总是否可用于论文 | `summary_status.json` | 每次汇总后 |
| 查看重新运行的 MC 参考值 | `references/{following,cutin}/` | MC 完成后 |
| 追溯某个 SS 单元的完整配置、日志和原始产物 | `runs/{event}/{setting_id}/seed_{seed}/` | SS 开始后 |
| 读取 seed/setting/论文级汇总 | `tables/` | 有任一执行记录后 |
| 使用参数敏感性图 | `figures/` | 有任一有效 SS 估计后 |

## 命名规则

- `event` 使用工程既有名称：`following`、`cutin`。
- `setting_id=default` 是共享主设置；每事件运行五个 seed。
- 其他 `setting_id` 采用 `<parameter>_<value>`，例如 `p0_0p1` 表示
  `p0 = 0.10`，`context_refresh_prob_0p75` 表示 `0.75`。小数点写为 `p`，
  使路径在各种平台和脚本中安全。
- 每个实际运行目录均保存 `effective_config.json`、`run_status.json`、`run.log`
  及共享 runner 生成的原始 SS 结果。

## 当前状态的正确解读

请以 `summary_status.json` 为准：`has_valid_ss_result=false` 时，任何表格或图表都
不能视为实验结论。汇总器会在全体单元仍为 `pending` 时清理过期的空表/空图；一旦
出现失败或已完成单元，会保留表格以便审计，图表则至少需要一个有效 SS 估计才生成。

该结果根目录对应当前的首个可审计版本。若 checkpoint、尾部条件分布、EVT 模型或
OAT 网格需要改变，应创建一个新的带版本后缀的结果根目录，而不能混入本目录。
