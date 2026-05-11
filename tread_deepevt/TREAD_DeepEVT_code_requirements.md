# TREAD 第二阶段代码需求文档：DeepEVT 条件极值尾部风险建模

> Project codename: **TREAD**  
> 第二阶段模块名: **DeepEVT**  
> 第一阶段输入: `tread_highd` 已抽取的 highD `events.csv` 与 highD 原始轨迹 CSV  
> 第二阶段目标: 针对 **car-following** 与 **cut-in** 两类驾驶事件分别建立条件化 EVT 模型，学习上下文相关的尾部风险分布，并导出可供后续 diffusion 条件生成使用的 `q90 / q95 / q99` 尾部风险目标。

---

## 0. 设计原则

本阶段只实现 **DeepEVT 条件极值风险建模**，不实现 diffusion 训练，不实现 MATLAB / RoadRunner 仿真。

本阶段应遵守以下原则：

1. **按事件类型分别建模**  
   第一版必须分别训练：
   - `DeepEVT-Following`
   - `DeepEVT-CutIn`

   不要用一个模型混合所有驾驶事件，也不要依赖 event-type embedding 统一建模。

2. **上下文特征分为两类**  
   - 一类由轨迹编码器自动从短时 prefix 轨迹中学习；
   - 一类由少量可解释行为特征手工构造。

3. **不要使用 `traffic_density_proxy`**  
   第一版不使用交通密度代理变量，避免定义不稳定、解释不直接。

4. **不要把风险标签泄漏进输入**  
   不要把以下变量作为 DeepEVT 输入特征：
   - `risk_score`
   - `min_ttc`
   - `min_thw`
   - `max_drac`
   - `ttc_severity`
   - `thw_severity`
   - `drac_severity`

   它们只能作为训练标签或评估指标。

5. **DeepEVT 的上下文应尽量对应可执行测试场景初始条件**  
   后续 diffusion 需要根据初始场景条件生成对抗轨迹。因此 DeepEVT 的上下文输入应优先使用在生成前可定义的变量，例如初始速度、初始间距、初始相对速度、初始横向偏移、计划切入持续时间等。

6. **风险响应变量必须来自固定长度 analysis window**  
   第一阶段 highD 事件本身可以长短不一，但 DeepEVT 训练样本必须基于固定物理时长的窗口计算风险响应变量 `R = window_risk_score`。

---

## 1. 与当前 `tread_highd` 代码的衔接

当前仓库已有第一阶段模块：

```text
TREAD-main/
  tread_highd/
    scripts/
      configs/highd_default.yaml
      extract_highd_events.py
      generate_quality_report.py
      play_highd_events.py
      visualize_risky_scores.py
    src/
      loader.py
      preprocess.py
      event_extraction.py
      risk_metrics.py
      schema.py
      ...
```

第二阶段不要重写第一阶段代码。
第二阶段应复用第一阶段函数：
---

## 2. 第二阶段输入与输出

### 2.1 输入

第二阶段至少需要以下输入：

```text
processed/events.csv
raw highD CSV files:
  XX_tracks.csv
  XX_tracksMeta.csv
  XX_recordingMeta.csv
```

其中 `events.csv` 来自第一阶段 `extract_highd_events.py`，应至少包含：

```text
event_id
event_type              # following 或 cut_in
recording_id
ego_id
target_id
start_frame
end_frame
anchor_frame
cross_frame             # cut-in 可选
cutin_start_frame       # cut-in 可选
cutin_end_frame         # cut-in 可选
risk_window_start_frame # 若第一阶段已有则优先使用
risk_window_end_frame   # 若第一阶段已有则优先使用
risk_score
min_ttc
min_thw
max_drac
initial_gap
min_gap
initial_relative_speed
post_cutin_gap
cutin_duration
is_valid
filter_reason
```

如果第一阶段尚未导出固定窗口轨迹张量，第二阶段必须能根据 `events.csv + raw highD` 重新构建固定窗口样本。

### 2.2 输出

建议输出目录：

```text
processed/deepevt/
  following/
    dataset.npz
    feature_schema.json
    normalization_stats.json
    train_val_test_split.json
    model.pt
    eval_report.json
    tail_conditions.csv
    figures/
      calibration_q95.png
      calibration_q99.png
      gpd_qq_plot.png
      tail_quantile_error.png
      predicted_vs_empirical_exceedance.png
  cut_in/
    dataset.npz
    feature_schema.json
    normalization_stats.json
    train_val_test_split.json
    model.pt
    eval_report.json
    tail_conditions.csv
    figures/
      ...
```

其中 `tail_conditions.csv` 是给 diffusion 使用的关键文件，包含：

```text
event_id
event_type
split
risk_score
u_pred
p_exceed_pred
xi_pred
beta_pred
q90_pred
q95_pred
q99_pred
es95_pred
es99_pred
empirical_risk_percentile
tail_label_90
tail_label_95
tail_label_99
```

---

## 3. DeepEVT 数据样本定义

每个 DeepEVT 样本必须包含以下字段：

```python
sample = {
    "event_id": str,
    "event_type": str,        # following 或 cut_in
    "split": str,             # train / val / test

    # 轨迹编码器输入：短时 prefix，而不是完整未来轨迹
    "prefix_states": np.ndarray,     # shape [K, A, F]

    # 可解释行为上下文特征
    "context_features": np.ndarray,  # shape [C]

    # EVT 响应变量
    "risk_score": float,             # 固定 analysis window 内的 window_risk_score

    # 仅用于评估，不作为输入
    "min_ttc": float,
    "min_thw": float,
    "max_drac": float,
}
```

### 3.1 为什么使用 prefix_states

DeepEVT 应服务于后续 diffusion 的条件生成。后续生成时可用的是**初始场景上下文**，而不是完整未来轨迹。因此 DeepEVT 的轨迹编码器不应直接编码完整 analysis window，否则模型可能泄漏未来风险信息。

推荐第一版使用：

```yaml
prefix:
  prefix_steps: 10   # 25 Hz 下约 0.4 秒
```

或者：

```yaml
prefix:
  prefix_steps: 25   # 25 Hz 下约 1.0 秒
```

如果事件窗口是 `[T, 2, F]`，则：

```python
prefix_states = states[:K]
```

这样 DeepEVT 学习的是：

> 给定初始交互状态和短时运动趋势，估计该上下文下未来固定窗口风险的尾部分布。

---

## 4. 固定窗口重建要求

如果第一阶段已经导出固定窗口轨迹张量，DeepEVT 可以直接读取。若没有，则必须在第二阶段实现 `window_rebuild.py`。

### 4.1 必须实现函数

```python
def rebuild_event_window(event_row: pd.Series, config: dict) -> dict:
    """根据 events.csv 中的一行事件和 raw highD 数据，重建固定 analysis window。"""


def get_analysis_frames(event_row: pd.Series, config: dict) -> np.ndarray:
    """优先使用 risk_window_start/end；若不存在，则围绕 anchor_frame 构建固定窗口。"""


def build_states_from_raw(recording, event_row: pd.Series, frames: np.ndarray, config: dict) -> np.ndarray:
    """返回 states, shape [T, 2, F]。actor0=ego, actor1=target。"""


def recompute_window_risk(recording, event_row: pd.Series, frames: np.ndarray, config: dict) -> dict:
    """在固定 analysis window 内重新计算 risk_score/min_ttc/min_thw/max_drac。"""
```

### 4.2 analysis window 规则

优先使用第一阶段字段：

```text
risk_window_start_frame
risk_window_end_frame
```

若不存在，则使用：

```python
pre = config["sampling"]["pre_anchor_steps"]
post = config["sampling"]["post_anchor_steps"]
frames = np.arange(anchor_frame - pre, anchor_frame + post + 1)
```

要求：

```python
len(frames) == window_length
```

若窗口不完整，则该样本不能进入 DeepEVT 训练集，记录过滤原因：

```text
insufficient_analysis_window
```

---

## 5. DeepEVT 上下文特征设计

上下文特征分为两部分：

1. `prefix_states`：由轨迹编码器自动学习；
2. `context_features`：少量可解释行为特征。

### 5.1 通用规则

禁止作为输入的字段：

```text
risk_score
min_ttc
min_thw
max_drac
ttc_severity
thw_severity
drac_severity
risk_percentile
tail_label_90/95/99
```

不建议作为核心输入的字段：

```text
source_lane
target_lane
traffic_density_proxy
```

可以保留这些字段作为元数据，但不放入 `context_features`。

---

## 6. car-following 行为上下文特征

`DeepEVT-Following` 的上下文应围绕纵向跟驰机理构造。

### 6.1 推荐输入特征

第一版建议使用以下特征：

```text
ego_v0
lead_v0
relative_speed_0          # ego_v0 - lead_v0
gap_0                     # window start frame net gap
ego_accel_0
lead_accel_0
thw_0                     # gap_0 / ego_v0
gap_slope_prefix          # prefix 内 gap 的线性趋势
closing_speed_max_prefix   # prefix 内 max(ego_v - lead_v, 0)
raw_segment_duration       # 原始 following segment 自然持续时间，单位秒
```

说明：

- `v0` 表示 analysis window 起始帧，而不是风险峰值帧；
- `raw_segment_duration` 可以作为上下文，因为自然持续时间不是风险标签；
- `min_ttc/max_drac/risk_score` 不得作为输入；
- `lead_min_acceleration` 如果来自完整未来窗口，不建议作为输入。若只从 prefix 计算，可使用 `lead_accel_min_prefix`。

### 6.2 必须实现函数

```python
def extract_following_context(states: np.ndarray, event_row: pd.Series, config: dict) -> dict:
    """从 [T,2,F] 状态张量和事件元信息中提取 following 上下文特征。"""
```

返回 dict 的 key 必须稳定，并写入 `feature_schema.json`。

---

## 7. cut-in 行为上下文特征

`DeepEVT-CutIn` 的上下文应围绕横向切入和切入前初始交互状态构造。

### 7.1 推荐输入特征

第一版建议使用以下特征：

```text
ego_v0
target_v0
relative_speed_0          # ego_v0 - target_v0
initial_dx                # target 相对 ego 的初始纵向位置
initial_dy                # target 相对 ego 的初始横向位置
target_vy_0
target_ax_0
target_ay_0
planned_cutin_duration    # 可由 cutin_duration 字段给出；后续生成时可作为可控场景参数
prefix_lateral_speed_mean # prefix 内 target 横向速度均值
raw_event_duration        # 原始 cut-in 事件自然持续时间，单位秒
```

说明：

- 不使用 `post_cutin_gap` 作为第一版核心上下文，因为它通常是切入后结果，作为生成前条件不稳定；
- 如果后续将 cut-in 场景定义为“从 cut-in end 开始生成后续风险演化”，可另行设计 post-cutin context，但第一版不采用；
- `source_lane/target_lane` 不作为核心特征。

### 7.2 必须实现函数

```python
def extract_cutin_context(states: np.ndarray, event_row: pd.Series, config: dict) -> dict:
    """从 [T,2,F] 状态张量和事件元信息中提取 cut-in 上下文特征。"""
```

---

## 8. 数据集构建脚本

### 8.1 配置示例

`configs/deepevt_following.yaml`：

```yaml
paths:
  raw_dir: "../highD-dataset/Matlab/data"
  events_csv: "../tread_highd/data/processed/events.csv"
  output_dir: "../data/processed/deepevt/following"

event:
  event_type: "following"

sampling:
  source_fps: 25
  target_fps: 25
  window_length: 128
  pre_anchor_steps: 64
  post_anchor_steps: 63

prefix:
  prefix_steps: 25

risk:
  epsilon: 1.0e-6
  max_ttc_clip: 20.0
  max_thw_clip: 10.0
  ttc_weight: 1.0
  thw_weight: 0.5
  drac_weight: 1.0
  softmax_lambda: 10.0

features:
  normalize: true
  use_prefix_encoder: true
  use_context_features: true
  forbid_risk_leakage: true

splits:
  strategy: "recording"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
```

`deepevt_cutin.yaml` 与其类似，只需：

```yaml
event:
  event_type: "cut_in"
```

### 8.2 CLI

```bash
python tread_deepevt/scripts/01_build_deepevt_dataset.py \
  --config tread_deepevt/configs/deepevt_following.yaml

python tread_deepevt/scripts/01_build_deepevt_dataset.py \
  --config tread_deepevt/configs/deepevt_cutin.yaml
```

输出：

```text
dataset.npz
feature_schema.json
normalization_stats.json
train_val_test_split.json
```

`dataset.npz` 应包含：

```text
event_id: [N]
recording_id: [N]
prefix_states: [N, K, 2, F]
context_features: [N, C]
risk_score: [N]
min_ttc: [N]
min_thw: [N]
max_drac: [N]
split_index: [N]    # 0=train, 1=val, 2=test
```

---

## 9. DeepEVT 模型结构

### 9.1 模型输入

```python
prefix_states: torch.Tensor      # [B, K, 2, F]
context_features: torch.Tensor   # [B, C]
```

### 9.2 模型输出

```python
outputs = {
    "u": Tensor[B],       # 条件阈值
    "p": Tensor[B],       # 超阈值概率，可选；默认也可使用 1-alpha_u
    "xi": Tensor[B],      # GPD shape
    "beta": Tensor[B],    # GPD scale > 0
    "q90": Tensor[B],
    "q95": Tensor[B],
    "q99": Tensor[B],
    "es95": Tensor[B],
    "es99": Tensor[B],
}
```

### 9.3 推荐架构

```text
PrefixEncoder:
  flatten actor-feature dimension: [B, K, 2*F]
  GRU or TCN
  hidden_dim = 64 or 128

ContextMLP:
  input_dim = C
  hidden_dim = 64

Fusion:
  concat(prefix_embedding, context_embedding)
  MLP hidden_dim = 128

Heads:
  u_head: linear -> positive or unconstrained scalar
  p_head: linear -> sigmoid
  xi_head: linear -> bounded xi
  beta_head: linear -> softplus + eps
```

第一版默认：

```yaml
model:
  encoder_type: "gru"       # options: gru, tcn, mlp
  hidden_dim: 128
  context_hidden_dim: 64
  dropout: 0.1
  xi_min: -0.3
  xi_max: 0.5
  beta_min: 1.0e-4
```

### 9.4 参数约束

```python
beta = softplus(beta_raw) + beta_min
xi = xi_min + (xi_max - xi_min) * sigmoid(xi_raw)
p = sigmoid(p_raw)
```

---

## 10. DeepEVT 损失函数

设固定窗口风险标签为：

```python
R = risk_score
```

### 10.1 Quantile loss

阈值头 `u` 学习条件分位数：

```yaml
training:
  alpha_u: 0.90
```

Pinball loss：

```python
def pinball_loss(R, u, alpha):
    e = R - u
    return torch.mean(torch.maximum(alpha * e, (alpha - 1.0) * e))
```

### 10.2 Exceedance loss

超阈值标签：

```python
exceed = (R > u.detach()).float()
```

BCE：

```python
loss_exc = binary_cross_entropy(p, exceed)
```

第一版允许配置关闭 `p_head`：

```yaml
training:
  use_exceedance_head: true
```

如果关闭，则默认：

```python
p = 1 - alpha_u
```

### 10.3 GPD negative log-likelihood

超额量：

```python
y = R - u.detach()
mask = y > 0
```

GPD NLL：

```python
def gpd_nll(y, xi, beta, eps=1e-6):
    # y > 0, beta > 0
    # support: 1 + xi*y/beta > 0
```

必须处理 `xi -> 0` 的指数极限：

```python
if abs(xi) < small:
    nll = log(beta) + y / beta
else:
    nll = log(beta) + (1 + 1/xi) * log(1 + xi*y/beta)
```

必须加入 support penalty：

```python
support = 1 + xi * y / beta
penalty = relu(eps - support) ** 2
```

### 10.4 Calibration loss

要求超阈值比例接近：

```python
1 - alpha_u
```

```python
soft_exceed = sigmoid((R - u) / delta)
loss_cal = (soft_exceed.mean() - (1 - alpha_u)) ** 2
```

### 10.5 总损失

```python
loss = (
    lambda_q * loss_q
    + lambda_exc * loss_exc
    + lambda_gpd * loss_gpd
    + lambda_cal * loss_cal
    + lambda_support * loss_support
)
```

推荐默认：

```yaml
loss_weights:
  lambda_q: 1.0
  lambda_exc: 0.2
  lambda_gpd: 1.0
  lambda_cal: 0.5
  lambda_support: 10.0
```

---

## 11. 训练流程

### 11.1 三阶段训练

为避免 GPD 训练不稳定，必须采用分阶段训练。

#### Stage 1: 阈值预训练

只训练 encoder、context MLP 和 `u_head`：

```yaml
pretrain_quantile_epochs: 50
```

损失：

```python
loss = loss_q + lambda_cal * loss_cal
```

#### Stage 2: 尾部分布训练

训练 `xi_head`、`beta_head`、`p_head`，encoder 可低学习率微调：

```yaml
tail_train_epochs: 100
```

损失：

```python
loss = loss_q + loss_exc + loss_gpd + loss_cal + support_penalty
```

#### Stage 3: 端到端微调

所有模块一起微调：

```yaml
finetune_epochs: 30
```

### 11.2 CLI

```bash
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/configs/deepevt_following.yaml

python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/configs/deepevt_cutin.yaml
```

---

## 12. 尾部分位与 Expected Shortfall 计算

给定：

```python
u, p, xi, beta
```

对于目标分位 `tau`，要求：

```python
tau > 1 - p
```

GPD 外推分位：

```python
q_tau = u + beta / xi * ((p / (1 - tau)) ** xi - 1)
```

当 `xi -> 0`：

```python
q_tau = u + beta * log(p / (1 - tau))
```

Expected Shortfall：

```python
ES_tau = q_tau + (beta + xi * (q_tau - u)) / (1 - xi)
```

如果 `xi >= 1`，ES 不稳定，应返回 NaN 并在报告中统计数量。模型默认约束 `xi < 0.5`。

必须实现：

```python
def tail_quantile(u, p, xi, beta, tau): ...

def expected_shortfall(u, p, xi, beta, tau): ...
```

并编写单元测试：

```text
test_tail_quantile_formula.py
```

---

## 13. Baselines

为了证明 DeepEVT 有必要，必须实现以下基线。

### 13.1 Global POT-GPD

按事件类型分别拟合固定阈值：

```python
u_global = train_risk.quantile(alpha_u)
y = train_risk[train_risk > u_global] - u_global
fit GPD(xi, beta)
```

输出：

```text
q90/q95/q99
```

### 13.2 Context-wise POT-GPD

可选。按初始速度或初始 gap 粗分组后分别拟合 GPD。

第一版可只实现 following：

```text
speed_bin: low/high ego_v0
gap_bin: small/large gap_0
```

### 13.3 Quantile-only neural baseline

只训练：

```python
u_theta(z)
```

用于比较：DeepEVT 是否优于纯分位回归。

---

## 14. 评估指标

### 14.1 Exceedance Calibration Error

对于预测分位 `q_tau(z)`：

```python
empirical_exceed_rate = mean(R > q_tau(z))
expected_exceed_rate = 1 - tau
ECE_tau = abs(empirical_exceed_rate - expected_exceed_rate)
```

分别计算：

```text
ECE_90
ECE_95
ECE_99
```

### 14.2 Tail Quantile Error

按分组统计经验分位与预测分位误差。

可按以下分组：

```text
following: gap_0 bins, relative_speed_0 bins
cut_in: initial_dx bins, relative_speed_0 bins
```

### 14.3 GPD Tail NLL

只对测试集中超过预测阈值的样本计算 GPD NLL。

### 14.4 Expected Shortfall Error

比较预测 ES 与测试集中超过 `q_tau(z)` 样本的经验均值。

### 14.5 Reliability plot

绘制：

```text
predicted exceedance probability vs empirical exceedance frequency
```

---

## 15. 评估脚本

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/configs/deepevt_following.yaml \
  --checkpoint processed/deepevt/following/model.pt

python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/configs/deepevt_cutin.yaml \
  --checkpoint processed/deepevt/cut_in/model.pt
```

输出：

```text
eval_report.json
figures/calibration_q95.png
figures/calibration_q99.png
figures/reliability_plot.png
figures/gpd_qq_plot.png
```

---

## 16. 导出给 diffusion 使用的条件文件

DeepEVT 训练完成后，必须导出给第三阶段 diffusion 使用的条件文件。

### 16.1 CLI

```bash
python tread_deepevt/scripts/04_export_tail_conditions.py \
  --config tread_deepevt/configs/deepevt_following.yaml \
  --checkpoint processed/deepevt/following/model.pt
```

### 16.2 输出 CSV

```text
tail_conditions.csv
```

字段：

```text
event_id
event_type
recording_id
split
risk_score
u_pred
p_exceed_pred
xi_pred
beta_pred
q90_pred
q95_pred
q99_pred
es95_pred
es99_pred
```

另外，为了 diffusion 条件生成，应导出对应的上下文特征：

```text
context_feature_*
```

但不要导出风险泄漏特征作为 diffusion 输入，除非明确用于评估。

---

## 17. 与 diffusion 的接口说明

第二阶段不实现 diffusion，但必须为第三阶段提供明确接口。

对于 following：

```text
Diffusion-Following input condition:
  ego_v0
  lead_v0
  relative_speed_0
  gap_0
  thw_0
  target_tail_level alpha
  q_alpha_pred from DeepEVT

Diffusion-Following output:
  lead vehicle future trajectory or acceleration/velocity profile
```

对于 cut-in：

```text
Diffusion-CutIn input condition:
  ego_v0
  target_v0
  relative_speed_0
  initial_dx
  initial_dy
  planned_cutin_duration
  target_tail_level alpha
  q_alpha_pred from DeepEVT

Diffusion-CutIn output:
  cut-in vehicle future trajectory
```

DeepEVT 的作用不是简单筛选 q99 样本，而是为 diffusion 提供：

1. 连续风险等级条件；
2. 上下文相关 q95/q99 目标；
3. 生成后校准器；
4. 后续可扩展为采样 guidance 的 tail probability 模型。

---

## 18. 单元测试要求

必须实现：

```text
tests/test_gpd_loss.py
tests/test_quantile_loss.py
tests/test_tail_quantile_formula.py
tests/test_feature_extraction_synthetic.py
```

### 18.1 `test_quantile_loss.py`

检查：

- 当预测分位数偏低时，高分位 pinball loss 更大；
- alpha=0.9 时，低估高值应受到更大惩罚。

### 18.2 `test_gpd_loss.py`

检查：

- beta 必须为正；
- support `1 + xi*y/beta` 小于 0 时必须产生 penalty；
- xi 接近 0 时使用指数极限且没有 NaN。

### 18.3 `test_tail_quantile_formula.py`

检查：

- tau 越大，q_tau 越大；
- beta 越大，q_tau 越大；
- xi 接近 0 时公式连续。

### 18.4 `test_feature_extraction_synthetic.py`

构造一个简单 following 或 cut-in synthetic window，检查：

- `gap_0` 正确；
- `relative_speed_0` 正确；
- 不出现风险泄漏字段；
- feature order 与 `feature_schema.json` 一致。

---

## 19. 代码质量要求

1. 不要修改 `tread_highd` 现有代码，除非导入路径或字段兼容必须微调。
2. 所有路径、窗口长度、prefix 长度、alpha、loss 权重必须写入 YAML。
3. 所有输出必须可复现，固定 random seed。
4. 所有训练统计量、归一化均值方差必须只用 train split 计算。
5. `val/test` 不得参与 threshold、normalization、GPD baseline 的拟合。
6. 日志使用 `logging`，长循环使用 `tqdm`。
7. 出现某些事件窗口重建失败时，不要中断全流程；记录失败原因并继续。
8. 模型训练中若某 batch 无超阈值样本，跳过 GPD NLL，但仍计算 quantile/calibration loss。
9. 所有评估图保存到 `figures/`。

---

## 20. 最小实现顺序

建议 Codex 按如下顺序实现：

1. `window_rebuild.py`：从 `events.csv + raw highD` 重建固定 analysis window。
2. `features.py`：实现 following 与 cut-in 上下文特征提取。
3. `data.py`：构建 `dataset.npz`、feature schema、normalization stats。
4. `losses.py`：实现 pinball loss、GPD NLL、tail quantile、expected shortfall。
5. `model.py`：实现 GRU/TCN prefix encoder + context MLP + EVT heads。
6. `train.py`：实现三阶段训练。
7. `baselines.py`：实现 Global POT-GPD 和 Quantile-only baseline。
8. `metrics.py` 和 `evaluate.py`：实现校准、NLL、分位误差、ES 误差。
9. `inference.py`：导出 `tail_conditions.csv`。
10. CLI scripts 与 tests。

---

## 21. 第一版验收标准

完成后应能回答：

1. following 与 cut-in 是否能分别训练 DeepEVT？
2. `u_pred` 的测试集 exceedance rate 是否接近 `1-alpha_u`？
3. DeepEVT 的 q95/q99 校准误差是否小于 Global POT-GPD？
4. DeepEVT 的 GPD Tail NLL 是否优于固定阈值 EVT？
5. `tail_conditions.csv` 是否包含每个事件的 q90/q95/q99？
6. 输出的上下文特征是否不包含 `risk_score/min_ttc/max_drac` 等泄漏变量？
7. 所有归一化统计量和 EVT baseline 是否只使用 train split？

---

## 22. 不属于第二阶段的内容

以下内容不要在第二阶段实现：

1. diffusion 模型训练；
2. diffusion guidance sampling；
3. MATLAB / RoadRunner 场景生成；
4. ACC/AEB 闭环测试；
5. CARLA 或 OpenSCENARIO 执行。

第二阶段只负责：

> 从 highD 事件数据中学习上下文相关的尾部风险分布，并导出 diffusion 可使用的条件尾部风险目标。

