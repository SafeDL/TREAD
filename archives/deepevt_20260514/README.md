# TREAD Phase 2：DeepEVT 条件尾部风险建模

`tread_deepevt` 是 TREAD 的第二阶段。它基于 `tread_highd` 生成的
`events.csv`，回到 highD 原始轨迹中重建固定长度窗口，并训练
DeepEVT 模型来预测条件尾部风险：

- 条件阈值 `u`
- 超阈值概率 `p`
- GPD 参数 `xi`、`beta`
- 尾部分位 `q90/q95`，其中 `q95` 是当前主长尾指标
- Expected Shortfall `ES95`

最终导出的 `tail_conditions.csv` 是后续 diffusion / MATLAB / RoadRunner
阶段共享的条件契约。

## 当前实现状态

已实现并可通过脚本串联运行的能力：

- 从 `events.csv` 筛选有效 `following` 或 `cut_in` 事件
- 复用 `tread_highd` 的 loader / preprocess / risk_metrics 重建固定长度窗口
- 将状态转换到 prefix 末端的 ego-current frame
- 生成 short-history context 特征和 `CanonicalScenarioContext`
- 按 recording 粒度切分 train / val / test
- 基于 train split 计算 normalization stats
- 训练 DeepEVT：actor temporal encoder + ego-target interaction token + scene Transformer encoder + EVT heads
- 评估 DeepEVT、Global POT-GPD 和 QuantileOnly baseline
- 导出 `tail_conditions.csv`

## 输入依赖

先运行第一阶段：

```bash
python tread_highd/scripts/extract_highd_events.py \
  --config tread_highd/scripts/configs/highd_default.yaml
```

DeepEVT 默认读取：

```text
../../../data/events.csv
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

# 可选：训练开始后另开一个终端查看 TensorBoard
tensorboard --logdir data/deepevt/following/runs --port 6006

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
data/deepevt/{following|cut_in}/
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
              runs/
eval_report.json
tail_conditions.csv
figures/
```

核心文件语义：

- `dataset.npz`：模型训练数组，包括 `prefix_states`、`context_features`、`risk_score`、split index、prefix/risk window frame 边界和 ego-current frame metadata。
- `feature_schema.json`：context feature 顺序、prefix state 通道、schema version 和 canonical mapping。
- `normalization_stats.json`：只基于 train split 计算的均值和标准差。
- `canonical_contexts.json`：每个事件的 `CanonicalScenarioContext`，供三阶段共享。
- `tail_conditions.csv`：DeepEVT 预测、经验尾部标签、context 字段、canonical 字段和坐标回投 metadata。

## 目录结构

```text
tread_deepevt/
├── src/
│   ├── scenario_frame.py       # CanonicalScenarioContext 与 ego-current frame
│   ├── window_rebuild.py       # 重建 prefix/risk 窗口并重算风险
│   ├── features.py             # short-history context 特征与泄漏检查
│   ├── data.py                 # dataset.npz / schema / split / normalization
│   ├── model.py                # DeepEVT PyTorch 模型
│   ├── losses.py               # Pinball / BCE / GPD NLL / numpy tail quantile / ES
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
1.1.0
```

所有 prefix 窗口都会被转换到 ego-current frame：

- 原点：prefix window 末端当前帧 ego 几何中心
- `+x`：ego 当前前进方向
- actor 0：ego
- actor 1：target
- actor state feature 顺序：`x, y, vx, vy, ax, ay`

时间边界会显式写入 `dataset.npz` 和 `tail_conditions.csv`：

- `prefix_start_frame`
- `prefix_end_frame`
- `risk_window_start_frame`
- `risk_window_end_frame`

默认定义为 `prefix_end_frame == risk_window_start_frame`，因此 `K=1` 与原单帧 current-scene 输入兼容；`risk_score` 只在 `risk_window_start_frame..risk_window_end_frame` 上重算。

`CanonicalScenarioContext` 同时保留：

- target 几何中心：`target_center_x0`, `target_center_y0`
- 净纵向间距：`initial_gap`
- 横向偏移：`initial_lateral_offset`
- 车辆尺寸、当前速度/加速度、相对速度
- lane id、窗口时长、prefix 时长、cut-in 计划持续时间

## Context Features

following 配置默认采用 short-history context：

```yaml
prefix:
  prefix_steps: 5
```

这意味着模型输入包括前 `K` 帧 ego/lead 状态、prefix 末端当前状态特征、prefix 内动态趋势特征，以及少量配置常量。模型仍不使用未来风险统计量。

### Following

| Feature | Canonical 来源 |
| --- | --- |
| `ego_vx_current` | `ego_v0` |
| `lead_vx_current` | `target_v0` |
| `relative_speed_current` | `relative_speed_0` |
| `gap_current` | `initial_gap` |
| `lateral_offset_current` | `initial_lateral_offset` |
| `ego_ax_current` | `ego_ax0` |
| `lead_ax_current` | `target_ax0` |
| `gap_change_rate` | `extras.gap_change_rate` |
| `relative_speed_trend` | `extras.relative_speed_trend` |
| `relative_acceleration` | `extras.relative_acceleration` |
| `ego_acc_mean_over_prefix` | `extras.ego_acc_mean_over_prefix` |
| `lead_acc_mean_over_prefix` | `extras.lead_acc_mean_over_prefix` |
| `lead_brake_indicator` | `extras.lead_brake_indicator` |
| `min_gap_in_prefix` | `extras.min_gap_in_prefix` |
| `max_closing_speed_in_prefix` | `extras.max_closing_speed_in_prefix` |
| `lateral_offset_change_rate` | `extras.lateral_offset_change_rate` |
| `lane_width` | `extras.lane_width` |
| `dt` | `extras.dt` |
| `horizon_steps` | `extras.horizon_steps` |
| `prefix_steps` | `extras.prefix_steps` |

### Cut-In

| Feature | Canonical 来源 |
| --- | --- |
| `ego_vx0` | `ego_v0` |
| `target_vx0` | `target_v0` |
| `relative_speed_0` | `relative_speed_0` |
| `target_center_x0` | `target_center_x0` |
| `target_center_y0` | `target_center_y0` |
| `initial_gap` | `initial_gap` |
| `initial_lateral_offset` | `initial_lateral_offset` |
| `target_vy0` | `target_vy0` |
| `target_ax0` | `target_ax0` |
| `target_ay0` | `target_ay0` |
| `lane_width` | `extras.lane_width` |
| `target_final_y` | `extras.target_final_y` |
| `dt` | `extras.dt` |
| `horizon_steps` | `extras.horizon_steps` |

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
u/xi/beta scale  参数不确定性输出，用于标记小样本尾部外推风险
```

训练分三阶段：

1. threshold pretrain：Pinball + Calibration
2. tail train：Pinball + Exceedance BCE + GPD NLL + Calibration + Support penalty + xi/beta regularization
3. end-to-end finetune：全模型继续训练

训练配置中的 `training.tensorboard: true` 只把按 stage 分开的关键 epoch
级别诊断指标写到 `output_dir/runs`，不记录 batch 曲线。重点查看
`selection_score`、`loss_q`、`exceed_rate_error`、`empirical_exceed_rate`、
`p_mean`、`loss_exc` 和 `loss_gpd`。从仓库根目录运行
`tensorboard --logdir data/deepevt/following/runs --port 6006` 后，在浏览器打开
`http://localhost:6006` 即可查看训练曲线。

当前 following 默认参数按短历史训练的稳定性做了保守化：

- `prefix_steps: 5`
- `batch_size: 256`
- `lr: 5e-4`
- `use_interaction_token: true`
- Stage 1 使用真实样本分布训练阈值，不使用 tail-balanced sampler
- Stage 2 从最佳 Stage 1 checkpoint 开始，只训练 tail heads
- `tail_lr_multiplier: 0.3`
- `encoder_lr_multiplier: 0.0`
- `threshold_lr_multiplier: 0.0`
- `selection_hard_cal_weight: 1.0`
- `tail_train_epochs: 40`
- `finetune_epochs: 0`
- `early_stopping_patience: 8`

训练保存的 `model.pt` 不是最后一个 epoch，而是验证选择分数最好的 checkpoint。
该选择分数为：

```text
loss_q + lambda_cal*loss_cal + lambda_exc*loss_exc
+ lambda_support*loss_support + selection_hard_cal_weight*exceed_rate_error
```

其中 `exceed_rate_error`
是 `|mean(risk > u) - (1 - alpha_u)|`，用于避免选中实际尾部质量严重偏离目标的
阈值 checkpoint。选择分数故意不包含可为负且容易主导曲线的 GPD NLL。
TensorBoard 中的 `stage_*/selection_score/val` 和
`stage_*/exceed_rate_error/val` 是推荐优先查看的模型选择曲线。

尾部分位使用 GPD 闭式外推。若 `p <= 1 - tau`，导出时会写入
`qXX_invalid_mask=1`，提示该样本对目标分位的外推条件不足。

## 实现完整性与正确性 Review

整体判断：`tread_deepevt` 的主流程完整，能够从第一阶段事件重建窗口、
构建数据集、训练模型、评估并导出尾部条件。`CanonicalScenarioContext`
和 `feature_schema.json` 把特征顺序、坐标系和下游契约固定住，这是当前实现中
最重要的正确性保障。

已确认较完整的部分：

- `window_rebuild.py`：复用第一阶段 highD 预处理与风险函数，显式分离 prefix window 和 risk window；缺帧或异常帧会导致样本跳过。
- `features.py`：following / cut-in 的 context key 顺序固定，并做风险泄漏字段检查。
- `data.py`：以 recording 为粒度切分，normalization 只使用 train split。
- `model.py` 与 `losses.py`：模型显式编码 ego-target interaction temporal token；EVT heads 有基本数值约束，GPD NLL 对小 `xi` 使用指数极限，并输出 `u/xi/beta` uncertainty scale。
- `inference.py`：`tail_conditions.csv` 同时导出预测、context、canonical 字段、prefix/risk frame 边界和 ego-current frame metadata。

需要注意的实现边界：

- `filter_events_by_type()` 已改为显式解析字符串布尔值，避免 `"False"` 被 `astype(bool)` 误判为有效事件。
- `_split_by_recording()` 对 recording 数量很少的数据使用 `round()` 切分，可能产生空 val 或空 test；评估阶段会对空 test split 给出明确错误。
- `recompute_window_risk()` 在固定窗口内只用 `gap > eps` 做风险帧 mask。对于 cut-in，如果固定窗口向 `cross_frame` 前扩展且 pre-cross 已有正 gap，Phase 2 的训练目标可能和 Phase 1 的 post-cross 风险语义不完全一致。
- 当前 short-history 版本使用真实 prefix 统计量。`prefix_steps=1` 可退化为单帧模式，`prefix_steps>1` 时模型用 GRU 编码每个 actor 的短历史，并额外编码 ego-target interaction temporal token，再用场景 Transformer 融合 actor、interaction 和 context tokens。
- 当前 schema 与模型结构已经改变，旧的 `dataset.npz`、`feature_schema.json` 和 `model.pt` 需要重新生成与训练；训练入口会校验 context key 顺序，避免误用旧数据。
- 如果 TensorBoard 出现训练 GPD NLL 持续下降、但验证 `loss_exc`/`loss_support` 上升的形态，应优先看 `selection_score/val`，并考虑进一步降低 `lambda_gpd`、`tail_lr_multiplier` 或缩短 Stage 2。

## 与下游的接口

下游建议优先读取 `tail_conditions.csv` 中这些字段：

- 事件身份：`event_id`, `event_type`, `recording_id`, `split`
- DeepEVT 输出：`u_pred`, `p_exceed_pred`, `xi_pred`, `beta_pred`, `u_scale_pred`, `xi_scale_pred`, `beta_scale_pred`, `q90_pred`, `q95_pred`, `es95_pred`
- 无效外推诊断：`q90_invalid_mask`, `q95_invalid_mask`
- 坐标回投：`ego_origin_x`, `ego_origin_y`, `ego_rot_cos`, `ego_rot_sin`
- canonical 当前场景：所有 `canonical_*` 字段
- 模型输入回显：所有 `context_*` 字段
