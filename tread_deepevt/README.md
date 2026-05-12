# TREAD Phase 2：DeepEVT 条件尾部风险建模

`tread_deepevt` 是 TREAD 的第二阶段。它基于 `tread_highd` 生成的
`events.csv`，回到 highD 原始轨迹中重建固定长度窗口，并训练
DeepEVT 模型来预测条件尾部风险：

- 条件阈值 `u`
- 超阈值概率 `p`
- GPD 参数 `xi`、`beta`
- 尾部分位 `q90/q95/q99`
- Expected Shortfall `ES95/ES99`

最终导出的 `tail_conditions.csv` 是后续 diffusion / MATLAB / RoadRunner
阶段共享的条件契约。

## 当前实现状态

已实现并可通过脚本串联运行的能力：

- 从 `events.csv` 筛选有效 `following` 或 `cut_in` 事件
- 复用 `tread_highd` 的 loader / preprocess / risk_metrics 重建固定长度窗口
- 将状态转换到 ego-initial frame
- 生成 initial-context 特征和 `CanonicalScenarioContext`
- 按 recording 粒度切分 train / val / test
- 基于 train split 计算 normalization stats
- 训练 DeepEVT：GRU / TCN / MLP prefix encoder + context MLP + EVT heads
- 评估 DeepEVT、Global POT-GPD 和 QuantileOnly baseline
- 导出 `tail_conditions.csv`

本目录当前没有 `tests/` 目录；本次 review 按要求没有新增或运行单元测试。

## 输入依赖

先运行第一阶段：

```bash
python tread_highd/scripts/extract_highd_events.py \
  --config tread_highd/scripts/configs/highd_default.yaml
```

DeepEVT 默认读取：

```text
../../../data/processed/events.csv
../../../highD-dataset/Matlab/data
```

这些路径都在：

```text
tread_deepevt/scripts/configs/deepevt_following.yaml
tread_deepevt/scripts/configs/deepevt_cutin.yaml
```

中配置，并相对于配置文件所在目录解析。

## 快速开始

以下命令从 TREAD 仓库根目录运行。

```bash
# 1. 构建 following 数据集
python tread_deepevt/scripts/01_build_deepevt_dataset.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 2. 训练 following DeepEVT
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 3. 评估
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 4. 导出 tail conditions
python tread_deepevt/scripts/04_export_tail_conditions.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

切入场景使用：

```bash
python tread_deepevt/scripts/01_build_deepevt_dataset.py \
  --config tread_deepevt/scripts/configs/deepevt_cutin.yaml
```

后续训练、评估、导出脚本同样替换为 `deepevt_cutin.yaml`。

## 输出文件

默认输出目录：

```text
data/processed/deepevt/{following|cut_in}/
```

产物：

```text
dataset.npz
feature_schema.json
normalization_stats.json
train_val_test_split.json
canonical_contexts.json
model.pt
training_history.json
eval_report.json
tail_conditions.csv
figures/
```

核心文件语义：

- `dataset.npz`：模型训练数组，包括 `prefix_states`、`context_features`、`risk_score`、split index 和 ego-initial frame metadata。
- `feature_schema.json`：context feature 顺序、prefix state 通道、schema version 和 canonical mapping。
- `normalization_stats.json`：只基于 train split 计算的均值和标准差。
- `canonical_contexts.json`：每个事件的 `CanonicalScenarioContext`，供三阶段共享。
- `tail_conditions.csv`：DeepEVT 预测、经验尾部标签、context 字段、canonical 字段和坐标回投 metadata。

## 目录结构

```text
tread_deepevt/
├── src/
│   ├── scenario_frame.py       # CanonicalScenarioContext 与 ego-initial frame
│   ├── window_rebuild.py       # 从 events.csv + raw highD 重建窗口并重算风险
│   ├── features.py             # initial-context 特征与泄漏检查
│   ├── data.py                 # dataset.npz / schema / split / normalization
│   ├── model.py                # DeepEVT PyTorch 模型
│   ├── losses.py               # Pinball / BCE / GPD NLL / tail quantile / ES
│   ├── train.py                # 三阶段训练
│   ├── evaluate.py             # 测试集评估与诊断图
│   ├── inference.py            # 模型加载、预测、tail_conditions 导出
│   ├── metrics.py              # ECE / tail bin error / tail NLL / ES error
│   └── baselines.py            # Global POT-GPD 与 QuantileOnly baseline
└── scripts/
    ├── 01_build_deepevt_dataset.py
    ├── 02_train_deepevt.py
    ├── 03_evaluate_deepevt.py
    ├── 04_export_tail_conditions.py
    └── configs/
        ├── deepevt_following.yaml
        └── deepevt_cutin.yaml
```

## 场景契约

`scenario_frame.py` 定义三阶段共享契约：

```text
DeepEVT.context == Diffusion.condition == MATLAB/RoadRunner.scenario_init
```

当前 schema version：

```text
1.0.0
```

所有窗口都会被转换到 ego-initial frame：

- 原点：analysis window 第 0 帧 ego 几何中心
- `+x`：ego 初始前进方向
- actor 0：ego
- actor 1：target
- actor state feature 顺序：`x, y, vx, vy, ax, ay`

`CanonicalScenarioContext` 同时保留：

- target 几何中心：`target_center_x0`, `target_center_y0`
- 净纵向间距：`initial_gap`
- 横向偏移：`initial_lateral_offset`
- 车辆尺寸、初始速度/加速度、相对速度
- lane id、窗口时长、prefix 时长、cut-in 计划持续时间

## Context Features

当前两个配置都采用 initial-context：

```yaml
prefix:
  prefix_steps: 1
```

这意味着模型 context 全部来自窗口第 0 帧，不使用未来轨迹统计量。

### Following

| Feature | Canonical 来源 |
| --- | --- |
| `ego_v0` | `ego_v0` |
| `lead_v0` | `target_v0` |
| `relative_speed_0` | `relative_speed_0` |
| `gap_0` | `initial_gap` |
| `ego_accel_0` | `ego_ax0` |
| `lead_accel_0` | `target_ax0` |
| `thw_0` | `extras.thw_0` |

### Cut-In

| Feature | Canonical 来源 |
| --- | --- |
| `ego_v0` | `ego_v0` |
| `target_v0` | `target_v0` |
| `relative_speed_0` | `relative_speed_0` |
| `initial_dx` | `initial_gap` |
| `initial_dy` | `initial_lateral_offset` |
| `target_vy_0` | `target_vy0` |
| `target_ax_0` | `target_ax0` |
| `target_ay_0` | `target_ay0` |

禁止进入模型输入的泄漏字段定义在 `features.py:LEAKAGE_KEYS`，包括
`risk_score`、`min_ttc`、`min_thw`、`max_drac`、severity、tail label、
planned duration 和 raw duration 等。

## 训练目标

模型输出：

```text
u     条件阈值
p     P(risk > u | context)
xi    GPD shape，限制在 [xi_min, xi_max]
beta  GPD scale，softplus + beta_min 保证为正
```

训练分三阶段：

1. threshold pretrain：Pinball + Calibration
2. tail train：Pinball + Exceedance BCE + GPD NLL + Calibration + Support penalty
3. end-to-end finetune：全模型继续训练

尾部分位使用 GPD 闭式外推。若 `p <= 1 - tau`，导出时会写入
`qXX_invalid_mask=1`，提示该样本对目标分位的外推条件不足。

## 实现完整性与正确性 Review

整体判断：`tread_deepevt` 的主流程完整，能够从第一阶段事件重建窗口、
构建数据集、训练模型、评估并导出尾部条件。`CanonicalScenarioContext`
和 `feature_schema.json` 把特征顺序、坐标系和下游契约固定住，这是当前实现中
最重要的正确性保障。

已确认较完整的部分：

- `window_rebuild.py`：复用第一阶段 highD 预处理与风险函数，缺帧或异常帧会导致样本跳过。
- `features.py`：following / cut-in 的 context key 顺序固定，并做风险泄漏字段检查。
- `data.py`：以 recording 为粒度切分，normalization 只使用 train split。
- `model.py` 与 `losses.py`：EVT heads 有基本数值约束，GPD NLL 对小 `xi` 使用指数极限。
- `inference.py`：`tail_conditions.csv` 同时导出预测、context、canonical 字段和 ego-initial frame metadata。

需要注意的实现边界：

- `filter_events_by_type()` 使用 `events["is_valid"].astype(bool)`。如果 CSV 中该列被读成字符串，`"False"` 也会变成 `True`；当前 pandas 通常会把 `True/False` 读成 bool，但更稳妥的实现应和 `play_highd_events.py` 一样做字符串解析。
- `_split_by_recording()` 对 recording 数量很少的数据使用 `round()` 切分，可能产生空 val 或空 test。空 test 会让 `predict()` 在 `np.concatenate([])` 处失败。
- `recompute_window_risk()` 在固定窗口内只用 `gap > eps` 做风险帧 mask。对于 cut-in，如果固定窗口向 `cross_frame` 前扩展且 pre-cross 已有正 gap，Phase 2 的训练目标可能和 Phase 1 的 post-cross 风险语义不完全一致。
- 当前 initial-context 版本不使用真实 prefix 统计量。`prefix_encoder` 仍保留，但默认 `prefix_steps=1`，因此它本质上编码单帧初始状态。
- `baselines.py` 的 docstring 提到 `ContextGroupedPOTGPD`，但代码中当前只实现了 Global POT-GPD 和 QuantileOnly baseline。

## 与下游的接口

下游建议优先读取 `tail_conditions.csv` 中这些字段：

- 事件身份：`event_id`, `event_type`, `recording_id`, `split`
- DeepEVT 输出：`u_pred`, `p_exceed_pred`, `xi_pred`, `beta_pred`, `q90_pred`, `q95_pred`, `q99_pred`, `es95_pred`, `es99_pred`
- 无效外推诊断：`q90_invalid_mask`, `q95_invalid_mask`, `q99_invalid_mask`
- 坐标回投：`ego_origin_x`, `ego_origin_y`, `ego_rot_cos`, `ego_rot_sin`
- canonical 初始场景：所有 `canonical_*` 字段
- 模型输入回显：所有 `context_*` 字段

