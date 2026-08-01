# highD 子集模拟参数敏感性实验

本目录只实现修订计划 `docs/revision_highd_experiments_goal.md` 第 5 节：在冻结的
highD 尾部场景条件分布、EVT 阈值、扩散 checkpoint 和 IDM ego policy 下，对
subset simulation (SS) 的四个超参数做 one-factor-at-a-time (OAT) 敏感性分析。
不包含多 ADS、模块消融、城市数据、风险指标替换或 IS 实验。

## 目录定位

```text
IDM_subset/
├── src/                         # 原有核心 SS、扩散和闭环实现（不在本实验中改动）
├── scripts/configs/             # 原有 following / cut-in 基础 YAML
├── experiments/highd_ss_sensitivity/
│   ├── run_experiments.py       # 64 个预注册 SS 单元与 MC 参考值的编排入口
│   ├── sensitivity_spec.py      # 唯一的网格、seed 与输出命名定义
│   ├── summarize_results.py     # 跨 seed 表格、图表和论文结论汇总
│   ├── following/README.md      # 跟驰事件的固定对象与 OAT 网格
│   └── cutin/README.md          # cut-in 事件的固定对象与 OAT 网格
└── results/revision_highd_ss_sensitivity/
    ├── README.md                # 当前实验状态、命名规则与产物导航
    ├── experiment_manifest.json # 冻结输入哈希和完整设计
    ├── run_plan.csv             # 64 个 seed 级计划单元及执行状态
    ├── references/              # MC 参考结果（运行后出现）
    ├── runs/                    # 原始 SS 结果：event/setting/seed
    ├── tables/                  # 仅在出现执行记录后生成
    └── figures/                 # 仅在出现有效 SS 结果后生成
```

## 简洁约定

- `sensitivity_spec.py` 是 OAT 网格、seed、默认设置与路径 token 的唯一来源；不要在
  事件 README、driver 或 shell 命令中复制第二套参数。
- `run_experiments.py` 直接调用已有 `latent_subset_runner`，不新增 following/cut-in
  转发脚本；两个事件子目录只说明各自的固定对象和网格差异。
- 原始样本、日志和有效配置只保存在对应 `runs/{event}/{setting}/seed_{seed}/`；汇总
  CSV/PNG 由一个汇总器按需生成。不要把临时表图或可重建的大型样本复制到其他目录。
- 当前可否形成结论只由结果根的 `summary_status.json` 决定，而非目录是否存在。

## 固定对象与网格

每个事件使用独立的基础 YAML（`IDM_subset/scripts/configs/`），只改变一个 SS
参数。默认配置共运行 five seeds `[101, 202, 303, 404, 505]`；其余设置运行
`[101, 202, 303]`。`mh_retries_per_sample`、风险阈值、输入资产、IDM 控制器和
其余 SS 参数均保持基础 YAML 的默认值。

| 参数 | following | cut-in |
| --- | --- | --- |
| `num_samples` | 1000, 3000, 5000 | 500, 1000, 2000 |
| `p0` | 0.10, 0.20, 0.30 | 0.05, 0.10, 0.20 |
| `proposal_std` | 0.06, 0.12, 0.24 | 0.05, 0.10, 0.20 |
| `context_refresh_prob` | 0.30, 0.50, 0.70, 0.90 | 0.25, 0.50, 0.75, 0.90 |

实际执行一次共享的 default 设置，汇总时再将它作为每个参数的基准行，避免重复
运行同一配置。每个事件共有 32 次 SS 运行：default 五次、九个非默认设置各三次。

## 运行

从仓库根目录执行。脚本使用 `tread` 环境；系统默认 Python 不包含本工程所需的
PyTorch。为避免占用正在使用的 GPU，正式运行前应确认有足够显存。

```bash
/home/hp/anaconda3/envs/tread/bin/python \
  IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --event all --run-reference-mc
```

可分事件断点续跑；已成功完成的 run 会被跳过：

```bash
/home/hp/anaconda3/envs/tread/bin/python \
  IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --event following
```

在全部或部分运行完成后，单独重新生成汇总表和图：

```bash
/home/hp/anaconda3/envs/tread/bin/python \
  IDM_subset/experiments/highd_ss_sensitivity/summarize_results.py
```

`--dry-run` 仅生成 manifest 与完整运行计划；`--setting` 和 `--seed` 可用于诊断
单元，例如 `--event cutin --setting proposal_std_0p2 --seed 101`。`--overwrite`
只会覆盖本实验专用结果根目录中指定 run 的产物，不能用于主实验结果目录；它不会
重写首次创建的输入哈希 manifest。若输入资产或网格发生变化，脚本会拒绝混合运行，
须创建新的结果目录。

`summary_status.json` 会记录是否已有有效 SS 结果；在至少一个 seed 通过运行前，
汇总器不会生成图表，避免把空运行计划误当成实验结果。部分失败或缺失的 setting 会在
正式图的底部以红色叉号明确标注。

## 结果与验收

所有新产物写入：

```text
IDM_subset/results/revision_highd_ss_sensitivity/
├── README.md                         # 人类阅读索引与命名规则
├── experiment_manifest.json          # 版本、输入哈希、精确网格
├── run_plan.csv                      # 64 个 SS 计划单元及状态
├── summary_status.json               # 结果是否已可用于论文
├── references/{following,cutin}/     # 本轮 MC 参考值和 Wilson 区间
├── runs/{following,cutin}/.../       # 每个 seed 的 config、log、原始 SS 产物
├── tables/                           # 出现执行记录后：seed/setting/论文结论表
└── figures/                          # 有效 SS 结果后：三组参数敏感性图
```

非默认设置若连续两个 seed 出现异常、无 elite 或 reliability failure，脚本会将
该设置剩余 seed 显式标记为 `skipped_after_two_quality_failures` 并保存原因；不会
静默删除失败单元。default 的五个 seed 始终全部尝试。结论中的“稳健”仅在 default
五次运行均通过 reliability，且跨 seed 95% 区间与本轮 MC Wilson 95% 区间重叠时
成立。单次 MCMC 的二项近似标准误只作为诊断，跨 seed 统计才用于不确定性判断。
