# TREAD highD 第一阶段 — 尾部风险交互事件数据集构建器

> **Project**: TREAD (Tail-Risk Extreme-value-Aware Diffusion)  
> **目标**: 从 highD 原始自然驾驶轨迹数据中自动抽取 cut-in 与 car-following 交互事件，构建标准化尾部风险轨迹数据集。

## 环境配置

```bash
conda activate jzm
pip install -r requirements.txt
```

## 数据目录

highD 原始数据文件 (`XX_tracks.csv`, `XX_tracksMeta.csv`, `XX_recordingMeta.csv`) 
位于: `../highD-dataset/Matlab/data/`

## 快速开始

### 1. 抽取事件

```bash
python scripts/01_extract_highd_events.py --config configs/highd_default.yaml
```

### 2. 构建完整数据集

```bash
python scripts/02_build_highd_dataset.py --config configs/highd_default.yaml
```

### 3. 可视化事件

```bash
python scripts/03_visualize_highd_events.py \
    --config configs/highd_default.yaml \
    --event_type cut_in --top_k 20 --sort_by risk_score
```

### 4. 生成质量报告

```bash
python scripts/04_generate_quality_report.py --config configs/highd_default.yaml
```

## 输出文件

```
data/processed/tread_highd/
├── events.csv                   # 所有事件（每行一个事件）
├── trajectories.h5              # 固定长度轨迹张量 [N, 64, 2, 11]
├── splits.json                  # 按 recording_id 划分的 train/val/test
├── normalization_stats.json     # 归一化统计量
├── quality_report.json          # 质量报告
└── figures/                     # 可视化图表
    ├── risk_distribution_cut_in.png
    ├── risk_distribution_following.png
    └── ttc_drac_scatter.png
```

## 项目结构

```
tread_highd/
├── configs/
│   └── highd_default.yaml       # 默认配置
├── src/
│   └── tread_highd/
│       ├── __init__.py
│       ├── schema.py            # 数据结构定义
│       ├── io_utils.py          # I/O 工具
│       ├── loader.py            # highD 数据读取器
│       ├── preprocess.py        # 轨迹清洗、方向统一
│       ├── lane_utils.py        # 车道几何工具
│       ├── risk_metrics.py      # TTC/THW/DRAC 风险指标
│       ├── event_extraction.py  # 事件抽取（following + cut-in）
│       ├── coordinate.py        # ego-centric 坐标转换
│       ├── windowing.py         # 固定窗口构建
│       ├── filtering.py         # 事件过滤与标签
│       ├── dataset_builder.py   # 数据集构建主流程
│       ├── visualization.py     # 可视化
│       └── quality_check.py     # 质量报告
├── scripts/
│   ├── 01_extract_highd_events.py
│   ├── 02_build_highd_dataset.py
│   ├── 03_visualize_highd_events.py
│   └── 04_generate_quality_report.py
├── tests/
│   ├── test_risk_metrics.py
│   ├── test_coordinate.py
│   └── test_windowing.py
├── requirements.txt
└── README.md
```

## 风险指标

| 指标 | 公式 | 含义 |
|------|------|------|
| TTC | gap / closing_speed | 碰撞时间 |
| THW | gap / ego_vx | 跟车时距 |
| DRAC | closing_speed² / (2·gap) | 避撞所需减速度 |
| S(t) | w₁/TTC + w₂/THW + w₃·DRAC | 即时风险 |
| R | logsumexp(λ·S) / λ | 轨迹风险 (soft-max) |

另外实现了**熵权法** (参考 Efficient and Unbiased Safety Test 论文) 用于多指标的客观权重计算。

## 运行测试

```bash
conda activate jzm
cd tread_highd
python -m pytest tests/ -v
```

## 参考

- highD 数据集论文: *The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways*
- 熵权法: *Efficient and Unbiased Safety Test for Autonomous Driving Systems*
- Matlab 参考实现: `highD-dataset/Matlab/` (longfilter, CutInFilter, SafetyIndicator)
