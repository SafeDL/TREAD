# TREAD highD 第一阶段 — 尾部风险交互事件数据集构建器

> **Project**: TREAD (Tail-Risk Extreme-value-Aware Diffusion)  
> **目标**: 从 highD 原始自然驾驶轨迹数据中自动抽取 cut-in 与 car-following 交互事件，构建标准化尾部风险轨迹数据集。

## 环境配置

```bash
conda activate jzm
```

## 数据目录

highD 原始数据文件 (`XX_tracks.csv`, `XX_tracksMeta.csv`, `XX_recordingMeta.csv`) 
位于: `../highD-dataset/Matlab/data/`

要处理的 recording 范围只在配置文件的 `recordings.include` / `recordings.exclude` 中指定；脚本不再提供单独的 `--recordings` 参数, 避免配置源冲突。

## 快速开始

### 1. 抽取候选事件

```bash
python scripts/01_extract_highd_events.py
```

输出:

```text
data/processed/intermediate/candidate_events.csv
data/processed/intermediate/invalid_events.csv
```

### 2. 固化事件主表

从 intermediate 文件生成 `events.csv`, 并添加 `risk_percentile` 与 tail labels。

```bash
python scripts/highd_events.py finalize-events
```

### 3. 构建训练数组和划分

需要模型训练张量时再执行；只做事件审计、可视化或 EVT 阈值检查时可以跳过。

```bash
python scripts/highd_events.py build-artifacts
```

输出:

```text
data/processed/trajectories.h5
data/processed/splits.json
data/processed/normalization_stats.json
```

### 4. 可视化事件

```bash
python scripts/03_visualize_highd_events.py --event_type cut_in --top_k 20 --sort_by risk_score
```

### 5. 生成质量报告

```bash
python scripts/04_generate_quality_report.py
```

## 输出文件

原则: 保留一个主事实表, 其他文件只保存模型训练必须直接读取的数组或可复现的统计量。功能性重复允许少量存在, 但不要让多个文件同时成为同一字段的权威来源。

```
data/processed/
├── events.csv                   # 事件主表: 所有事件、风险指标、尾部标签、过滤状态
├── trajectories.h5              # 训练数组: 有效事件的 states/mask/frame_ids/event_id
├── splits.json                  # 按 recording_id 划分的 train/val/test
├── normalization_stats.json     # 仅由训练集估计的归一化统计量
├── quality_report.json          # 可再生成的质量报告
└── figures/                     # 可再生成的可视化图表
    ├── risk_distribution_cut_in.png
    ├── risk_distribution_following.png
    └── ttc_drac_scatter.png
```

推荐的精简语义:

- `events.csv` 是事件元数据和风险标签的唯一主表。人工审计、统计分析、EVT 拟合阈值检查均优先读取它。
- `trajectories.h5` 面向训练数据加载。除 `event_id` 外, 不应重复保存大量可由 `events.csv` join 得到的字段。若为训练便利保留少量风险字段, 仍以 `events.csv` 为权威来源。
- `quality_report.json` 和 `figures/` 是诊断产物, 可以由 `events.csv` 再生成, 不应作为建模输入的唯一来源。
- `intermediate/candidate_events.csv` 和 `intermediate/invalid_events.csv` 只用于调试抽取过程, 正式建模不依赖它们。

## 项目结构

```
tread_highd/
├── src/
│   ├── schema.py                # 数据结构定义
│   ├── io_utils.py              # I/O 工具
│   ├── loader.py                # highD 数据读取器
│   ├── preprocess.py            # 轨迹清洗、方向统一
│   ├── lane_utils.py            # 车道几何工具
│   ├── risk_metrics.py          # TTC/THW/DRAC 风险指标
│   ├── event_extraction.py      # 事件抽取（following + cut-in）
│   ├── coordinate.py            # ego-centric 坐标转换
│   ├── windowing.py             # 固定窗口构建
│   ├── filtering.py             # 事件过滤与标签
│   ├── dataset_builder.py       # 训练数组/划分导出工具
│   ├── visualization.py         # 可视化
│   └── quality_check.py         # 质量报告
├── scripts/
│   ├── 01_extract_highd_events.py
│   ├── highd_events.py
│   ├── 03_visualize_highd_events.py
│   ├── 04_generate_quality_report.py
│   └── configs/
│       └── highd_default.yaml
└── README.md
```

## 风险指标

### Anchor 策略

事件抽取阶段应尽量保持语义中立, 先获得完整自然驾驶事件, 再把风险作为 EVT 建模标签。默认配置因此采用:

- `following.anchor_mode = "center"`: anchor 放在完整跟驰片段的中点, 不由 TTC/DRAC/risk_score 决定。
- `cutin.anchor_mode = "cross"`: anchor 放在换道跨线帧, 表示 cut-in 事件本身的语义中心。

`risk`、`min_ttc`、`max_drac` 等风险驱动 anchor 可用于诊断或对齐最危险片段, 但不作为默认事件抽取策略。

建模和 EVT 分析必须统一使用 danger-oriented 语义: 数值越大表示越危险。原始物理指标可以保留用于解释, 但进入尾部建模前需要转换成同方向的 severity 指标。

| 字段 | 公式或来源 | 建模语义 |
|------|------------|----------|
| `min_ttc` | min(gap / closing_speed) | 原始物理量, 越小越危险 |
| `min_thw` | min(gap / ego_vx) | 原始物理量, 越小越危险 |
| `max_drac` | max(closing_speed² / (2·gap)) | 原始物理量, 越大越危险 |
| `ttc_severity` | 1 / (`min_ttc` + eps) | danger-oriented, 越大越危险 |
| `thw_severity` | 1 / (`min_thw` + eps) | danger-oriented, 越大越危险 |
| `drac_severity` | `max_drac` | danger-oriented, 越大越危险 |
| `risk_score` | soft-max 聚合后的综合风险 | danger-oriented, 越大越危险 |
| `risk_percentile` | 同一 `event_type` 内的 `risk_score` 排名 | danger-oriented, 越大越危险 |
| `tail_label_90/95/99` | `risk_score` 是否超过对应分位阈值 | `True` 表示尾部高风险 |

即时风险推荐写成:

```text
S(t) = w_ttc / (TTC(t) + eps) + w_thw / (THW(t) + eps) + w_drac * DRAC(t)
R = logsumexp(lambda * S(t)) / lambda
```

但该公式只应在物理关系有效的帧上计算: target 必须位于 ego 前方、`gap > 0`, 且 closing/THW/DRAC 的定义有效。无效帧应被 mask 掉或赋安全基线, 不能通过 clip 到 0 后再取倒数。

### CUT-IN 风险窗口

CUT-IN 的 `risk_score` 不应在整段 ego-target 公共轨迹上计算。切入车辆在切入前常位于 ego 后方, 这会产生负 gap；如果 THW 被 clip 为 0, `1 / (THW + eps)` 会制造约 `1 / eps` 量级的伪尾部, 使 95/99 百分位异常巨大。

CUT-IN 推荐规则:

- 事件匹配仍可使用完整公共轨迹, 但风险聚合只使用 `cross_frame` 之后的帧, 或固定事件窗口内且 `gap > 0` 的帧。
- `min_ttc`、`min_thw`、`max_drac`、`risk_score` 应从同一个有效风险窗口中得到, 保证语义一致。
- 若风险窗口内没有有效 `gap > 0` 帧, 该事件应标记为无效风险样本或给出明确 `filter_reason`, 不参与 EVT 尾部拟合。
- CUT-IN 的 tail label 应基于修正后的 danger-oriented `risk_score`, 且仍按 `event_type` 分开计算分位数。

### 代码简洁性

另外实现了**熵权法** (参考 Efficient and Unbiased Safety Test 论文) 用于多指标的客观权重计算。

为避免后续语义漂移, 风险相关代码应保持单一、短路径实现:

- 一个函数负责从原始 TTC/THW/DRAC 映射到 danger-oriented severity。
- 一个函数负责根据事件类型选择有效风险窗口。
- 一个函数负责聚合 `risk_score` 和尾部标签。
- 不在多个模块中重复实现分位数、方向转换或 CUT-IN 特例逻辑。
- 风险评分只用于标注和尾部分析, 不作为候选事件抽取的过滤条件, 以保持自然暴露分布。


## 参考

- highD 数据集论文: *The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways*
- 熵权法: *Efficient and Unbiased Safety Test for Autonomous Driving Systems*
- Matlab 参考实现: `highD-dataset/Matlab/` (longfilter, CutInFilter, SafetyIndicator)
