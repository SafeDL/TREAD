# highD IDM SS 参数敏感性结果

本目录只保存 IDM 在 highD `following` 和 `cutin` 事件上的冻结 OAT 子集模拟
参数敏感性实验。它不包含多 ADS、模块消融、重要性采样、城市数据或风险指标替换实验。

## 结果阅读顺序

1. `summary_status.json`：确认 SS 结果、表格和图是否有效。
2. `experiment_manifest.json`：固定基础配置、输入 SHA-256、OAT 网格、seed、MC
   预算和正式执行配置。
3. `tables/ss_sensitivity_paper_conclusion_table.csv`：默认参数的预注册结论。
4. `tables/ss_sensitivity_setting_level_summary.csv`：所有设置的跨 seed 汇总。
5. `run_plan.csv`：64 个计划 seed 单元的状态、耗时与失败原因。

默认配置仅当 5 个独立 seed 均通过可靠性诊断，且 SS 跨 seed 95% 区间与当前独立 MC
Wilson 95% 区间重叠时，才在结论表中标为 `robust_by_predeclared_rule=true`。

## 目录结构

```text
experiment_manifest.json
run_plan.csv
summary_status.json
references/{following,cutin}/
  effective_config.json
  run_status.json
  latent_monte_carlo_summary.json
  latent_monte_carlo_stats.csv
  latent_monte_carlo_top_cases.json
  latent_monte_carlo_samples.npz  # 可再生的原始 MC 样本；非汇总所需
runs/{event}/{setting}/seed_{seed}/
  effective_config.json
  run_status.json
  latent_subset_summary.json
  latent_subset_level_stats.csv
  latent_subset_top_cases.json
  global_risk_exposure_comparison.{json,csv}
  latent_subset_samples.npz       # 可再生的原始 SS 样本；非汇总所需
  run.log                          # 可再生的运行日志；非汇总所需
tables/
figures/
```

`references/` 保存独立 MC 基准；`runs/` 保存每个 SS 重复的完整审计材料。`tables/` 的
3 张 CSV 和 `figures/` 的 3 张 PNG 都可由
`IDM_subset/experiments/highd_ss_sensitivity/summarize_results.py` 重新生成。

## 保留与可移植性

本目录保留逐 seed 的 JSON/CSV 诊断、manifest 和汇总图表，以支持复核与复现。原始 `.npz`
样本和 `run.log` 是可再生中间产物，汇总器不会读取它们；在 summary、CSV 和 PNG 已生成后
可以删除。统一运行器新写入的持久化路径均为使用 `/` 的 POSIX 相对路径：配置输出路径相对
基础 YAML，结果引用相对仓库或结果根目录。因此在同一仓库结构下可直接迁移到 Windows 或
Linux。`summarize_results.py` 还会规范化历史 `run_plan.csv` 中遗留的 Windows 分隔符 `\`；
新增或手工编辑的 metadata 不应使用绝对路径。
