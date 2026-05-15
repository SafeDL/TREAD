# DeepEVT 条件极值尾部风险建模：理论、实现与训练诊断说明

## 摘要

本文档说明 `tread_deepevt` 模块中 DeepEVT 条件尾部风险模型的建模目标、输入输出、网络结构、三阶段训练策略、损失函数、评估指标以及 TensorBoard 曲线阅读方式。文档完全依据当前代码实现撰写，主要涉及以下文件：

- `tread_deepevt/src/window_rebuild.py`：从 highD 原始轨迹与 `events.csv` 重建短历史 prefix window 与固定 risk window，并重算窗口风险。
- `tread_deepevt/src/features.py`：提取不含未来风险泄漏的 short-history context 特征。
- `tread_deepevt/src/data.py`：生成 `dataset.npz`、schema、normalization stats 与 recording 粒度划分。
- `tread_deepevt/src/model.py`：定义 ShortHistorySceneTransformer 与 DeepEVT 参数头。
- `tread_deepevt/src/losses.py`：定义 pinball loss、exceedance BCE、GPD NLL、calibration loss、尾部分位和 ES 公式。
- `tread_deepevt/src/train.py`：实现三阶段训练与 TensorBoard 记录。
- `tread_deepevt/src/evaluate.py`、`metrics.py`、`baselines.py`：实现测试集评估、Global POT-GPD 和指标计算。

DeepEVT 的核心思想是：给定交通事件前 `K` 帧短历史交互状态和当前上下文，估计未来固定时间窗内风险得分的条件尾部分布。模型不是只预测一个平均风险，而是预测一个条件阈值 `u`、超阈概率 `p` 以及广义帕累托分布参数 `xi` 和 `beta`，从而进一步得到高分位风险 `q90/q95` 与 Expected Shortfall `ES95`。这适用于交通安全中的长尾问题：少数高风险事件比平均误差更重要。

## 1. 问题定义

### 1.1 样本与目标变量

对每一个 highD 事件样本，`window_rebuild.py` 会重建两段时间窗口：模型可见的 `prefix window` 和用于监督标签的 `risk window`。当前 following 配置为：

- `window_length = 128`
- `target_fps = 25`
- risk window 时间长度约为 `128 / 25 = 5.12 s`
- `prefix_steps = 5`

默认时间边界为：

```text
prefix_start_frame .. prefix_end_frame
risk_window_start_frame .. risk_window_end_frame
prefix_end_frame == risk_window_start_frame
```

这样 `K=1` 时保持单帧 current-scene 兼容；`K>1` 时 prefix 不再误读 risk window 起点之后的未来帧。

每个样本的输入包括：

- `prefix_states`: shape 为 `[prefix_steps, actors, state_features]`，两个 actor 分别为 ego 和 target，状态通道为 `x, y, vx, vy, ax, ay`。
- `context_features`: following 场景下为 20 维短历史上下文特征。

输出标签为窗口风险得分：

```text
risk_score = window_risk_score
```

该风险不是模型输入，而是监督学习的目标变量。代码中还保存 `min_ttc`、`min_thw`、`max_drac` 作为诊断字段。

### 1.2 风险得分构造

在 `window_rebuild.py` 中，风险得分通过以下步骤得到：

1. 对窗口内每一帧计算净间距 `gap`、TTC、THW 和 DRAC。
2. 调用 `tread_highd.src.risk_metrics.compute_instant_risk` 得到逐帧瞬时风险：

```text
instant_risk = w_ttc * severity_ttc
             + w_thw * severity_thw
             + w_drac * severity_drac
```

当前配置为：

```yaml
ttc_weight: 1.0
thw_weight: 0.5
drac_weight: 1.0
```

3. 调用 `compute_trajectory_risk` 对整段窗口风险做 softmax/log-sum-exp 汇总：

```text
R = (log sum_i exp(lambda * r_i) - log T) / lambda
```

当前 `softmax_lambda = 10.0`。这相当于一个可微的“近似最大风险”：比简单平均更关注窗口内危险峰值，又比直接最大值更平滑。

### 1.3 输入特征与防止风险泄漏

当前 following 模型的 20 个 context 特征分为三组：

| 特征 | 含义 |
| --- | --- |
| `ego_vx_current` | ego 当前纵向速度 |
| `lead_vx_current` | 前车当前纵向速度 |
| `relative_speed_current` | 当前相对速度 |
| `gap_current` | 当前净纵向间距 |
| `lateral_offset_current` | 当前横向偏移 |
| `ego_ax_current` | ego 当前纵向加速度 |
| `lead_ax_current` | 前车当前纵向加速度 |
| `gap_change_rate` | prefix 内 gap 变化率 |
| `relative_speed_trend` | prefix 内相对速度趋势 |
| `relative_acceleration` | 当前相对加速度 |
| `ego_acc_mean_over_prefix` | ego prefix 平均纵向加速度 |
| `lead_acc_mean_over_prefix` | lead prefix 平均纵向加速度 |
| `lead_brake_indicator` | prefix 内前车是否明显制动 |
| `min_gap_in_prefix` | prefix 内最小净间距 |
| `max_closing_speed_in_prefix` | prefix 内最大闭合速度 |
| `lateral_offset_change_rate` | prefix 内横向偏移变化率 |
| `lane_width` | 车道宽度 |
| `dt` | 采样间隔 |
| `horizon_steps` | risk window 步数 |
| `prefix_steps` | prefix window 步数 |

这些特征只来自 prefix window 及其末端当前帧。`features.py` 明确禁止以下未来风险字段进入模型输入：

```text
risk_score, min_ttc, min_thw, max_drac,
ttc_severity, thw_severity, drac_severity,
risk_percentile, tail_label_*
```

这点非常重要。DeepEVT 的目标是从短历史条件预测未来风险尾部分布，而不是把未来已经计算出来的风险指标作为输入。

### 1.4 数据划分

`data.py` 按 recording 粒度划分 train/val/test，而不是按样本随机划分。这样可以降低同一录像内相似轨迹造成的数据泄漏。当前 following 数据为：

| split | 样本数 |
| --- | ---: |
| train | 50,802 |
| val | 13,260 |
| test | 9,261 |

风险分布在 train/val/test 上较接近。test split 的经验分位约为：

| 分位 | risk_score |
| --- | ---: |
| q50 | 0.407 |
| q85 | 0.751 |
| q90 | 0.857 |
| q95 | 1.041 |
| q99 | 1.533 |

这说明当前划分没有明显的整体风险分布漂移。

## 2. 模型结构

### 2.1 总体结构

`model.py` 中的 DeepEVT 模型由三部分组成：

1. `ShortHistorySceneTransformer`
2. `fusion` MLP
3. EVT 参数预测 heads

输入为：

```text
prefix_states      [batch, prefix_steps, actors, state_features]
context_features   [batch, context_features]
```

当前 following 配置中：

```text
prefix_steps = 5
actor 数 = 2
state_features = 6
context_features = 20
hidden_dim = 128
Transformer layers = 2
attention heads = 4
dropout = 0.2
use_interaction_token = true
```

### 2.2 ShortHistorySceneTransformer

模型包含三个编码环节。首先，每个 actor 的 `[prefix_steps, state_features]` 短历史序列通过 GRU 时间编码器压缩为一个 actor temporal token；其次，从 ego 与 target 的短历史序列中构造显式交互序列：

```text
target_x - ego_x
ego_vx - target_vx
ego_ax - target_ax
target_y - ego_y
```

该序列再经单独的 GRU 编码为一个 ego-target interaction token。最后，场景级 Transformer 对以下 token 做 self-attention：

- 一个可学习的 `cls_token`
- ego/target 两个 actor temporal token
- 一个 ego-target interaction temporal token
- 20 个 context scalar token

每个 actor 的短历史状态先经 `state_proj`、time positional embedding 和 GRU 编码；interaction 序列经 `interaction_proj`、同一 time positional embedding 和独立 GRU 编码；每个 context scalar 经 `context_value_proj` 投影到 hidden_dim；再加上 actor type、interaction type 和 context type embedding。Transformer encoder 输出 `cls_token` 对应的场景表示 `z_scene`。

这种结构的意义是：模型既可以学习“ego 是否持续加速、lead 是否持续减速”等单车短历史动态，也可以直接看到“gap 是否持续缩小、closing speed 是否持续增大、横向偏移是否变化”等交互动态，再通过 self-attention 融合 ego、lead、interaction 与 context 特征，而不是只做简单拼接回归。

### 2.3 Fusion 与 EVT heads

`fusion` 是两层 MLP，将 `z_scene` 映射为共享隐变量 `z`。随后分别预测：

| head | 输出 | 约束 | 含义 |
| --- | --- | --- | --- |
| `u_head` | `u` | unconstrained | 条件阈值，近似 `alpha_u` 分位 |
| `p_head` | `p` | sigmoid 到 `(0,1)` | 超过阈值 `u` 的条件概率 |
| `xi_head` | `xi` | sigmoid 映射到 `[xi_min, xi_max]` | GPD shape |
| `beta_head` | `beta` | softplus + `beta_min` | GPD scale |
| `u_log_scale_head` | `u_log_scale` | clamp 到 `[-5,5]` | 阈值不确定性辅助项 |
| `xi_log_scale_head` | `xi_log_scale` | clamp 到 `[-5,5]` | 尾部参数不确定性辅助项 |
| `beta_log_scale_head` | `beta_log_scale` | clamp 到 `[-5,5]` | 尾部参数不确定性辅助项 |

当前配置：

```yaml
alpha_u: 0.85
xi_min: -0.3
xi_max: 0.5
beta_min: 1.0e-4
use_exceedance_head: true
```

因此 `u` 的目标是条件 85% 分位附近的阈值，而 q90/q95 是在这个阈值之上通过 GPD 外推得到。

## 3. 极值理论与 DeepEVT 建模思想

### 3.1 为什么不只训练普通回归模型

交通风险建模中，高风险事件数量少、影响大。普通 MSE/MAE 回归通常会偏向样本密集区域，也就是中低风险区域；这会导致模型对高风险尾部低估。DeepEVT 的目标不是精确拟合所有样本的均值，而是建模：

```text
P(Y > y | X = x)
```

尤其是高分位区域，例如 q90/q95。

### 3.2 Peaks Over Threshold 与 GPD

极值理论中的 Peaks Over Threshold 思路是：当阈值 `u` 足够高时，超额量

```text
Z = Y - u, 条件为 Y > u
```

可近似服从广义帕累托分布：

```text
Z | Z > 0 ~ GPD(xi, beta)
```

其中：

- `xi` 控制尾部形状。`xi > 0` 表示重尾；`xi = 0` 对应指数尾；`xi < 0` 表示有有限上界的短尾。
- `beta > 0` 控制尺度。

DeepEVT 的条件化体现在：`u(x)`、`p(x)`、`xi(x)`、`beta(x)` 都由神经网络根据短历史条件 `x` 预测。

### 3.3 条件尾部分位公式

在 `losses.py` 中，tail quantile 的 numpy 版本为 `tail_quantile_np`。对目标尾部分位 `tau`，模型计算：

```text
p_min = 1 - tau
frac = p / (1 - tau)
```

若 `xi` 不接近 0：

```text
q_tau = u + beta / xi * (frac^xi - 1)
```

若 `xi` 接近 0，则使用指数极限：

```text
q_tau = u + beta * log(frac)
```

若 `p <= 1 - tau`，则严格来说从阈值 `u` 外推到 `tau` 不再合理，因为预测的超阈概率本身小于目标尾部概率。代码会将 `p` clamp 到 `1 - tau + eps` 以保证数值稳定，并可通过 `tail_quantile_invalid_mask` 标记这些样本。

### 3.4 Expected Shortfall

Expected Shortfall 表示超过给定分位后的平均风险，当前用于 `tau >= 0.95`：

```text
ES_tau = E[Y | Y > q_tau]
```

代码实现为：

```text
ES = q + (beta + xi * (q - u)) / (1 - xi)
```

当 `xi >= 1` 时 ES 理论上发散，因此代码会返回 `NaN`。

## 4. 为什么分成 Stage 1、Stage 2、Stage 3

当前训练由 `train.py` 实现，分为三阶段：

```text
Stage 1: threshold pretrain
Stage 2: tail train
Stage 3: end-to-end finetune
```

这不是随意拆分，而是为了让一个较难的尾部分布学习问题变得稳定。

### 4.1 Stage 1：先学稳定的条件阈值 `u`

Stage 1 只训练：

```text
encoder + fusion + u_head
```

使用的主要损失为：

```text
pinball loss + calibration loss + u uncertainty auxiliary loss
```

不训练 `p`、`xi`、`beta`，也不计算 GPD NLL。

这样做的原因是：GPD 只对 `Y > u` 的超额量有意义。如果 `u` 一开始完全不可靠，那么“哪些样本是超阈值样本”也不可靠，GPD 的训练目标会非常混乱。先让 `u` 接近 `alpha_u=0.85` 条件分位，相当于先确定尾部建模的入口。

用更直观的话说：

```text
Stage 1 先回答：什么样的短历史交互状态会导致较高风险阈值？
```

当前训练结果显示 Stage 1 是有效的：

| 指标 | train | val |
| --- | ---: | ---: |
| `loss_q` 初始 | 0.0548 | 0.0196 |
| `loss_q` Stage 1 末尾 | 0.0237 | 0.0153 |
| `loss_q` val 最低 | - | 0.0131 |

因此 Stage 1 学到了比较稳定的条件分位结构。

### 4.2 Stage 2：固定较低 encoder 学习率，集中训练尾部 heads

Stage 2 训练：

```text
encoder: lr * 0.1
u_head:  lr * 0.1
tail heads: lr
```

损失扩展为：

```text
pinball + exceedance BCE + GPD NLL + calibration + support penalty + regularization
```

这时模型开始学习：

- `p(x)`：超过阈值的概率
- `xi(x)`：尾部形状
- `beta(x)`：尾部尺度

为什么 encoder 和 `u_head` 用低学习率？因为 Stage 1 已经学到了比较有意义的条件阈值。如果 Stage 2 一开始就用同样大学习率更新 encoder 和 `u_head`，尾部损失可能会破坏已经学好的阈值表示，导致训练震荡。

Stage 2 的直观目标是：

```text
在一个相对稳定的阈值 u 上，学习超过阈值后的尾部分布。
```

### 4.3 Stage 3：端到端小学习率联合微调

Stage 3 用统一的较小学习率 `finetune_lr=2e-4` 训练所有模块：

```text
encoder + fusion + u_head + p_head + xi_head + beta_head + uncertainty heads
```

此阶段允许阈值、场景表示和尾部参数相互适配。其意义是：Stage 2 中尾部 heads 在较稳定的表示上先收敛；Stage 3 再让整个模型以较小步长做联合修正。

如果 Stage 3 过长或验证指标不再改善，就可能过拟合尾部噪声。当前 following 训练正是出现了这个现象：Stage 3 最好的验证 `loss_total` 在 epoch 3 左右，而最终 epoch 30 已经明显退化。

### 4.4 三阶段训练的必要性总结

三阶段训练对应一个由易到难的课程学习过程：

| 阶段 | 学习对象 | 难度 | 目的 |
| --- | --- | --- | --- |
| Stage 1 | 条件阈值 `u` | 较低 | 稳定确定尾部入口 |
| Stage 2 | `p, xi, beta` | 较高 | 在已成形的阈值上学习尾部分布 |
| Stage 3 | 全模型联合 | 最高 | 细调阈值、表示和尾部参数之间的一致性 |

对于 DeepEVT 这类模型，直接从随机初始化开始同时训练所有 head，常见问题包括：

- `u` 不稳定导致超阈样本集合频繁变化。
- `GPD NLL` 对极端样本敏感，早期梯度噪声大。
- `xi` 和 `beta` 容易走到边界或异常尺度。
- 高分位校准可能看似改善，但条件分箱误差变差。

因此，三阶段训练是一种工程上稳健的折中。

## 5. 损失函数设计

### 5.1 Pinball loss：分位数回归

`pinball_loss` 用于训练 `u` 接近 `alpha_u` 条件分位。令真实风险为 `y`，预测阈值为 `u`，误差为：

```text
e = y - u
```

则：

```text
L_q = mean(max(alpha * e, (alpha - 1) * e))
```

当 `alpha = 0.85` 时，模型被鼓励让约 15% 的样本高于 `u`。

### 5.2 Calibration loss：软超阈率校准

代码中 calibration loss 为：

```text
soft_exceed = sigmoid((y - u) / delta)
L_cal = (mean(soft_exceed) - (1 - alpha))^2
```

其中 `delta = 0.05`。这不是逐样本误差，而是批量层面的软校准约束，要求平均超阈比例接近 `1 - alpha`。它能减少 `u` 虽然 pinball loss 低但整体超阈率偏离目标的问题。

### 5.3 Exceedance BCE：超阈概率头

`p_head` 预测：

```text
p(x) = P(Y > u(x) | x)
```

训练标签为：

```text
1{y > u.detach()}
```

这里 `u.detach()` 很关键：BCE 训练 `p`，但不让 BCE 的梯度反向修改 `u`。否则模型可能通过移动 `u` 来让分类任务变容易，破坏分位阈值的意义。

### 5.4 GPD NLL：超额量的尾部分布似然

对超阈样本：

```text
z = y - u.detach(), z > 0
```

若 `xi` 不接近 0，GPD 的负对数似然为：

```text
L_gpd = log(beta) + (1 + 1 / xi) * log(1 + xi * z / beta)
```

若 `xi` 接近 0，则使用指数分布极限：

```text
L_gpd = log(beta) + z / beta
```

这里也对 `u` 使用 `detach()`，使 GPD 项主要训练尾部形状和尺度，而不是让 `u` 被 GPD NLL 拉动。

### 5.5 Support penalty：保证 GPD 支撑域合法

GPD 要求：

```text
1 + xi * z / beta > 0
```

如果该项小于等于 0，概率密度不合法。代码中加入：

```text
support_penalty = relu(eps - support)^2
```

并以 `lambda_support = 10.0` 加权。这个项通常应接近 0；如果明显升高，说明 `xi/beta` 与真实超额量不兼容。

### 5.6 参数正则项

当前代码还包括：

```text
loss_xi_reg   = mean((xi - xi_prior)^2)
loss_beta_reg = mean(log(beta / beta_ref)^2)
```

配置为：

```yaml
xi_prior: 0.0
beta_ref: 1.0
lambda_xi_reg: 0.05
lambda_beta_reg: 0.01
```

它们用于避免 `xi` 和 `beta` 任意漂移。但若正则太强或参数边界设置不合适，也可能导致模型偏向某些边界解。当前训练中 `xi_mean` 几乎贴到 `xi_min=-0.3`，这就是需要重点关注的信号。

### 5.7 不确定性辅助项

`loss_u_unc` 的形式为：

```text
q_per / u_scale + u_log_scale
```

其中：

```text
u_scale = exp(u_log_scale)
```

这类似异方差建模中的负对数似然结构。它允许模型对不同样本学习不同尺度的不确定性。注意，因为包含 `log(scale)`，该项可以为负数。该项不是普通意义上的“错误率”。

尾部不确定性项 `loss_tail_unc` 目前在记录中为 0，说明 `xi_log_scale` 和 `beta_log_scale` 没有触发最小置信度惩罚。

### 5.8 总损失

总损失由多个加权项组成：

```text
L_total =
  lambda_q       * L_q
+ lambda_cal     * L_cal
+ lambda_u_unc   * L_u_unc
+ lambda_exc     * L_exc
+ lambda_gpd     * L_gpd
+ lambda_support * L_support
+ lambda_xi_reg  * L_xi_reg
+ lambda_beta_reg* L_beta_reg
+ lambda_tail_unc* L_tail_unc
```

当前配置：

```yaml
lambda_q: 1.0
lambda_exc: 0.2
lambda_gpd: 1.0
lambda_cal: 0.5
lambda_support: 10.0
lambda_u_unc: 0.05
lambda_xi_reg: 0.05
lambda_beta_reg: 0.01
lambda_tail_unc: 0.01
```

## 6. 为什么有些 loss 是负数

这是一个非常容易困惑但很正常的问题。当前训练中负数主要来自两个来源。

### 6.1 GPD NLL 可以为负

负对数似然不一定总是正数。对连续型概率密度来说，密度值可以大于 1，因此：

```text
-log density < 0
```

例如 GPD 或正态分布在尺度很小时，概率密度峰值可以大于 1。代码中的 `loss_gpd` 是连续密度的 negative log-likelihood，因此为负并不代表“错误”或“训练崩了”。

需要关注的是：

- train 和 val 的相对走势
- 是否持续恶化
- 是否伴随 `support_penalty` 升高
- 是否伴随 `xi` 贴边、`beta` 异常缩小或增大

### 6.2 不确定性项可以为负

`loss_u_unc = q_per / scale + log(scale)`。当 `scale < 1` 时，`log(scale)` 为负；如果 `q_per / scale` 不大，整体就可能为负。

因此：

```text
loss_u_unc < 0
```

是数学形式允许的，不应按普通误差理解。

### 6.3 总损失可以为负

因为 `loss_total` 包含 `loss_gpd` 和 `loss_u_unc`，而这两个项都可能为负，所以 `loss_total` 也可以为负。重要的是曲线是否稳定、验证集是否同步改善，以及最终校准指标是否可靠。

当前 following 训练中，Stage 2 开始时 `val_loss_total = -2.7075`，后续退化到正值附近。这并不是因为“负数不好，正数好”，而是因为验证集目标在训练过程中恶化了，尤其 `loss_gpd` 从 `-2.77` 退化到约 `-0.74`，同时 `support` 和校准项波动加大。

## 7. TensorBoard 曲线阅读指南

训练脚本只记录按 stage 拆分的 epoch 级 scalar：

```text
stage_{stage}/{metric}/{split}
```

其中 `{stage}` 为 `1`、`2` 或 `3`，`{split}` 为 `train` 或 `val`。例如：

```text
stage_1/exceed_rate_error/val
stage_2/selection_score/val
stage_2/p_mean/val
```

这样 Stage 1 的阈值校准、Stage 2/3 的尾部训练和最终模型选择不会混在同一张曲线里。不同 stage 的 `loss_total` 组成不同，因此不再作为 TensorBoard 默认关注项；它仍保存在 `training_history.json` 中，只能在同一 stage 内做趋势比较。

### 7.1 只看 epoch 曲线，不看 batch 曲线

训练脚本不再向 TensorBoard 写 batch-level scalar。batch 曲线天然很抖，原因包括：

- 尾部样本稀少，不同 batch 中超阈样本数量差异大。
- 使用了 `tail_balanced_sampling`，训练 batch 中高风险样本被加权采样。
- GPD NLL 只在 `y > u` 的样本上计算，超阈样本数量变化会放大波动。
- batch-level calibration 是批量均值约束，小 batch 下波动更明显。

因此阅读顺序建议为：

```text
先看各 stage 的 val 曲线，再看同一 stage 的 train 曲线。
```

### 7.2 `stage_*/loss_q/*`

含义：分位数阈值 `u` 的 pinball loss。

阅读方法：

- 越低通常越好。
- Stage 1 中最重要。
- 如果 train 降、val 升，说明阈值预测开始过拟合。
- 如果 Stage 2/3 中 `loss_q` 大幅恶化，说明尾部训练破坏了阈值 head。

当前 following 结果：

- Stage 1 `val/loss_q` 从 `0.0196` 降到最低 `0.0131`，说明阈值预训练有效。
- Stage 2/3 后期 `val/loss_q` 约 `0.0164`，没有灾难性崩坏，但已不如 Stage 1 最优。

### 7.3 `stage_*/loss_cal/*`

含义：软超阈率是否接近 `1 - alpha_u`。

当前 TensorBoard 默认更推荐看 `stage_*/exceed_rate_error/*`。它是 hard calibration：

```text
exceed_rate_error = |mean(risk > u) - (1 - alpha_u)|
```

该指标直接反映阈值 `u` 选出来的实际尾部质量是否接近目标。

当前 `alpha_u=0.85`，目标超阈率为：

```text
1 - alpha_u = 0.15
```

阅读方法：

- 越低越好。
- 过高说明 `u` 的整体位置与目标分位不一致。
- 它不是 q90/q95 的最终校准指标，而是 `u` 阈值层面的辅助约束。

当前结果中，val `loss_cal` 有明显波动，最终约 `0.0287`，说明后期阈值校准并不稳定。

### 7.4 `stage_2/loss_exc/*` 与 `stage_3/loss_exc/*`

含义：`p_head` 对 `Y > u` 的超阈事件分类 BCE。

阅读方法：

- 只在 Stage 2/3 出现。
- 越低通常越好。
- 如果 train 低而 val 高，说明超阈概率头过拟合。
- 需要结合 `p_mean` 和 q90/q95 的 exceedance rate 判断。

当前最终 test 上：

```text
p_mean = 0.0945
```

这低于 `alpha_u=0.85` 对应的理论超阈概率 `0.15`。这说明 `p_head` 倾向于预测较低的超阈概率，导致 q90 外推时出现较多无效样本。

### 7.5 `stage_2/loss_gpd/*` 与 `stage_3/loss_gpd/*`

含义：超阈样本的 GPD negative log-likelihood。

阅读方法：

- 可以为负。
- 不能简单理解为“越负越好到无限”，要结合 validation。
- 如果 train 改善而 val 恶化，尾部参数可能过拟合。
- 如果伴随 `xi` 贴边或 `support` 升高，要格外警惕。

当前结果：

- Stage 2 第 1 个 epoch `val/loss_gpd = -2.7714`
- Stage 2 末尾退化到约 `-0.7358`
- Stage 3 最终约 `-0.9549`

说明后期尾部分布拟合在验证集上明显退化。

### 7.6 `stage_2/loss_support/*` 与 `stage_3/loss_support/*`

含义：GPD 支撑域非法时的惩罚。

阅读方法：

- 理想情况下接近 0。
- 如果显著上升，说明 `1 + xi*z/beta` 接近或低于 0，GPD 参数和超额量不兼容。

当前最终 val `loss_support` 约 `0.0832`，不是灾难性数值，但高于 Stage 2 初期的 0，说明后期尾部参数变得更紧张。

### 7.7 `loss_xi_reg`

含义：`xi` 偏离先验 `xi_prior=0` 的正则。

阅读方法：

- 不是越低就一定越好；它只是约束参数不要漂移。
- 如果该值接近某个固定值且 `xi` 贴边，说明 shape head 可能被推到边界。

当前最终 test：

```text
xi_mean = -0.299996
xi_min  = -0.3
```

这意味着 `xi` 几乎贴住下界。理论上这表示模型认为风险尾部有有限上界，但工程上也可能意味着参数边界或训练权重导致的饱和。

### 7.8 `loss_beta_reg`

含义：`beta` 相对 `beta_ref=1.0` 的 log-scale 正则。

阅读方法：

- 用于防止 `beta` 过大或过小。
- 如果长期偏大，说明 `beta` 与参考尺度差异大。
- 需结合实际 `beta_mean` 以及 q95 校准判断。

当前最终 test：

```text
beta_mean = 0.0724
```

beta 明显小于 `beta_ref=1.0`，因此 `loss_beta_reg` 较大是可以理解的。这也提示 `beta_ref` 是否应根据风险得分尺度重新设置。

### 7.9 `loss_u_unc` 与 `loss_tail_unc`

这些项仍保存在 `training_history.json` 中，但不再默认写入 TensorBoard。它们是训练辅助项，不应作为主要效果指标。TensorBoard 默认聚焦 `selection_score`、`loss_q`、`exceed_rate_error`、`empirical_exceed_rate`、`p_mean`、`loss_exc` 和 `loss_gpd`。

### 7.10 `training_history.json` 中的 `loss_total`

含义：所有 loss 加权和。

阅读方法：

- 只能在同一 stage 内比较。
- Stage 1 和 Stage 2/3 的 `loss_total` 组成不同，不应跨 stage 直接比较。
- 它不再默认写入 TensorBoard；如果需要排查数值问题，可从 `training_history.json` 中读取。

当前 following 结果：

| 位置 | val/loss_total |
| --- | ---: |
| Stage 1 末尾 | -0.1469 |
| Stage 2 epoch 1 | -2.7075 |
| Stage 2 末尾 | 0.1192 |
| Stage 3 epoch 3 | -1.4204 |
| Stage 3 最终 | -0.0757 |

这表明最终保存的 `model.pt` 不是验证最优 checkpoint。当前代码只保存最终 epoch，没有 early stopping 或 best checkpoint。

## 8. 训练波动为什么明显

当前 TensorBoard 中波动明显是预期现象，但并非所有波动都可以忽略。

### 8.1 正常波动来源

1. 尾部样本少。q95 附近只有约 5% 样本，batch 里有效尾部样本数量少。
2. GPD NLL 只对 `y > u` 计算，超阈集合会随 `u` 改变。
3. 使用 tail-balanced sampling，高风险样本在训练 batch 中权重更大，batch 分布与 val/test 分布不同。
4. `xi`、`beta` 对极端样本敏感，少数样本可以显著改变 NLL。
5. Stage 切换时 loss 组成和学习率同时改变，因此需要看对应 stage 内的曲线。

### 8.2 需要警惕的波动

以下现象更像训练质量问题：

- `stage_2/loss_total/val` 或 `stage_3/loss_total/val` 在各自阶段内持续恶化。
- `stage_2/loss_gpd/val` 或 `stage_3/loss_gpd/val` 明显退化，而 train 不同步退化。
- `loss_support` 从 0 升高并维持较高水平。
- `xi` 贴住 `xi_min` 或 `xi_max`。
- `p_mean` 显著偏离预期超阈率。
- q90/q95 的 empirical exceed rate 偏离目标。
- 分箱误差显示某些场景条件下系统性低估高风险。

当前 following 训练属于：阈值预训练稳定，最终 q95 全局校准尚可，但尾部参数训练不稳，且 `xi` 贴下界、条件分箱下存在系统性低估。

## 9. 评估指标含义

### 9.1 ECE：exceedance calibration error

`metrics.py` 中定义：

```text
empirical_exceed_rate = mean(y > q_pred)
expected_exceed_rate  = 1 - tau
ECE = abs(empirical_exceed_rate - expected_exceed_rate)
```

例如 `tau=0.95` 时，理想情况下约 5% 样本应超过预测 q95。

当前 test 结果：

| tau | 模型 | empirical exceed | 目标 | ECE |
| --- | --- | ---: | ---: | ---: |
| 0.90 | DeepEVT | 0.1124 | 0.1000 | 0.0124 |
| 0.90 | Global POT-GPD | 0.1106 | 0.1000 | 0.0106 |
| 0.95 | DeepEVT | 0.0531 | 0.0500 | 0.0031 |
| 0.95 | Global POT-GPD | 0.0553 | 0.0500 | 0.0053 |

结论：最终 checkpoint 的全局 q95 校准不错，并略优于 Global POT-GPD；q90 稍差于 Global POT-GPD。

### 9.2 Tail quantile bin error

该指标按某个 context feature 分箱。following 中默认使用 `gap_current`。每个 bin 内比较：

```text
empirical_quantile = quantile(y_bin, tau)
predicted_mean     = mean(q_pred_bin)
abs_error          = abs(empirical_quantile - predicted_mean)
```

它回答的问题是：

```text
模型是否在不同当前间距条件下都校准？
```

当前 q95 分箱结果：

| gap_current bin | n | empirical q95 | predicted q95 mean | abs error |
| --- | ---: | ---: | ---: | ---: |
| 3.82 - 28.44 | 2315 | 1.347 | 0.992 | 0.355 |
| 28.44 - 45.11 | 2315 | 0.787 | 0.617 | 0.170 |
| 45.11 - 79.15 | 2315 | 0.655 | 0.412 | 0.242 |
| 79.15 - 281.09 | 2316 | 0.561 | 0.265 | 0.297 |

这说明模型学到了“当前间距越小，风险分位越高”的趋势，但各 bin 内均存在低估，尤其最小 gap bin 的高风险尾部低估较明显。

### 9.3 GPD tail NLL

`gpd_tail_nll` 只在 `risk > u` 的样本上计算 GPD NLL。它衡量尾部超额量分布拟合，但不直接等价于 q95 校准。一个模型可能 NLL 较好但某个分位校准不好，也可能反过来。

当前 test：

```text
gpd_tail_nll = -0.9530
```

该值为负是允许的，解释见第 6 节。

### 9.4 Expected Shortfall error

`expected_shortfall_error` 比较预测 ES 与真实超过预测分位样本的平均风险：

```text
error = abs(mean(y | y > q_pred) - mean(ES_pred | y > q_pred))
```

当前 test q95：

```text
ES95 error = 0.0449
```

该指标越小越好，但它依赖 q95 超过样本的数量与稳定性，应该和 q95 ECE、分箱误差一起看。

### 9.5 Global POT-GPD baseline

Global POT-GPD 使用 train split 上的全局阈值与全局 GPD 参数，不根据场景条件变化。当前拟合结果：

```text
u = 0.7288
xi = 0.1253
beta = 0.2437
p = 0.1500
```

它是一个重要基线。如果 DeepEVT 不能显著优于 Global POT-GPD，则说明条件化建模的收益不足，或者当前神经网络/训练策略还没有稳定利用条件信息。

当前结果中，DeepEVT 在全局 q95 ECE 上优于 Global POT-GPD，但在 q90 ECE 上略差，且分箱误差显示条件校准仍有不足。

## 10. 当前 following 训练效果诊断

基于 `data/deepevt/following/training_history.json`、TensorBoard event 文件和补跑的 `eval_report.json`，当前结果可总结为：

### 10.1 优点

1. 数据划分合理：按 recording 划分，train/val/test 风险分布接近。
2. Stage 1 阈值预训练有效：`val/loss_q` 明显下降。
3. 最终 test q95 全局校准较好：ECE 为 `0.0031`。
4. q95 ECE 优于 Global POT-GPD：`0.0031` vs `0.0053`。
5. 模型能学习 gap_current 与风险分位之间的单调趋势。

### 10.2 主要问题

1. Stage 2/3 验证集尾部损失退化明显。
2. 当前代码保存最终 epoch，而不是验证最优 checkpoint。
3. `xi` 几乎贴住 `xi_min=-0.3`，存在边界饱和。
4. `p_mean=0.0945`，低于 `alpha_u=0.85` 对应的理论超阈概率 `0.15`。
5. q95 分箱中预测均低于经验分位，存在系统性低估。
6. q90 外推中有较多 `p <= 1 - tau` 的无效样本，需要重点检查。

### 10.3 总体评价

当前模型不是失败的。它已经学到了合理的阈值结构，并在 test q95 全局校准上达到可用水平。但是，从学术或工程安全角度，不建议直接把最终 `model.pt` 当作最佳模型。更准确的评价是：

```text
当前 DeepEVT following 模型具备初步尾部风险建模能力，
但尾部参数训练稳定性不足，条件校准仍需改进。
```

## 11. 改进建议

### 11.1 保存 best checkpoint

当前 `train.py` 训练结束后只保存最终模型。建议新增：

- `best_model.pt`
- `best_metric.json`
- `best_stage_epoch`

可选 best 指标：

1. `val/loss_total`，实现简单，但跨 stage 不完全可比。
2. `val/loss_gpd + support penalty`，偏重尾部分布。
3. 定期在 val 上计算 q95 ECE 和分箱误差，最符合目标但成本更高。

建议至少保存 Stage 2/3 中 `val/loss_total` 最低的 checkpoint。

### 11.2 Early stopping

Stage 2/3 后期验证集退化明显。建议：

- 对 Stage 2 设置 patience。
- 对 Stage 3 设置更短 epoch 或 early stopping。
- 当 `val/loss_gpd` 连续恶化时提前停止。

### 11.3 调整 tail head 学习率

当前 Stage 2 tail heads 使用 `lr=1e-3`，可能偏大。可尝试：

```yaml
lr: 5.0e-4
finetune_lr: 1.0e-4
```

或在 Stage 2 中为 tail heads 单独设置更小学习率。

### 11.4 重新检查 `xi_min` 与 `xi` 正则

当前 `xi` 贴住 `-0.3`。可尝试：

- 放宽 `xi_min`，例如 `-0.5`，观察是否仍贴边。
- 缩小 `lambda_xi_reg`，观察是否由正则引起。
- 增加 `xi` 分布监控曲线，而不仅记录 `loss_xi_reg`。
- 在评估报告中保存 `xi` 的分位数。

### 11.5 重新设定 `beta_ref`

当前 test `beta_mean=0.0724`，但 `beta_ref=1.0`。如果风险得分尺度本身在 0 到 2 附近，`beta_ref=1.0` 可能偏大。建议根据 train split 超额量尺度设定 `beta_ref`，例如使用：

```text
beta_ref = mean(y - q85_train | y > q85_train)
```

### 11.6 监控 q90/q95 invalid rate

对每个 `tau` 记录：

```text
mean(p <= 1 - tau)
```

如果 q90 invalid rate 很高，说明 `p_head` 与 `u_head` 不一致。当前最终 test 中 q90 invalid rate 较高，需要进一步处理。

### 11.7 增加条件校准指标

仅看全局 ECE 不够。建议将以下指标作为常规输出：

- 按 `gap_current` 分箱的 q90/q95 误差。
- 按 `relative_speed_current` 分箱的 q90/q95 误差。
- 按 `ego_vx_current` 或 `lead_vx_current` 分箱的 q90/q95 误差。
- 每个 bin 的 empirical exceed rate。

这样可以发现全局校准良好但局部场景低估的问题。

## 12. 面向论文写作的表述建议

可以将当前方法描述为：

```text
We formulate safety-critical trajectory risk prediction as a conditional tail
distribution estimation problem. Given an initial traffic scene represented in
an ego-current coordinate frame, the proposed DeepEVT model predicts a
conditional threshold and generalized Pareto tail parameters, enabling the
estimation of high-level risk quantiles and expected shortfall.
```

对应中文表述：

```text
本文将安全关键交通风险预测建模为条件尾部分布估计问题。给定 ego-current 坐标系
下的交通场景短历史状态，DeepEVT 模型同时预测条件阈值、超阈概率以及广义帕累托
尾部分布参数，从而估计高分位风险和 Expected Shortfall。
```

三阶段训练可以表述为：

```text
为缓解尾部分布学习中的阈值不稳定和极端样本稀疏问题，本文采用课程式三阶段
优化策略。首先通过 pinball loss 预训练条件阈值；随后在较低 encoder 学习率下
训练超阈概率和 GPD 参数；最后以较小学习率端到端联合微调全部模块。
```

损失函数可以表述为：

```text
总目标函数由条件分位数损失、软校准损失、超阈概率二元交叉熵、GPD 负对数似然、
支撑域惩罚和参数正则项组成。该目标同时约束阈值位置、超阈概率、尾部形状和
尺度参数。
```

评估指标可以表述为：

```text
模型主要通过 exceedance calibration error、条件分箱尾部分位误差、GPD tail NLL
和 Expected Shortfall error 进行评估。其中 exceedance calibration error 衡量
预测 q_tau 后真实样本超过该阈值的比例是否接近理论值 1 - tau。
```

## 13. 实践阅读清单

如果需要快速判断一次训练是否可用，建议按以下顺序检查：

1. `stage_1/loss_q/val`：Stage 1 是否稳定下降。
2. `stage_2/loss_total/val`、`stage_3/loss_total/val`：Stage 2/3 是否在各自阶段内明显退化。
3. `stage_2/loss_gpd/val`、`stage_3/loss_gpd/val`：尾部 NLL 是否在验证集上恶化。
4. `stage_2/loss_support/val`、`stage_3/loss_support/val`：GPD 支撑域是否稳定。
5. `eval_report.json` 中 q90/q95 ECE：全局校准是否达标。
6. `tail_quantile_bins`：不同场景条件下是否系统性低估。
7. `deepevt.xi_mean`、`beta_mean`、`p_mean`：参数是否贴边或偏离理论预期。

对当前 following 结果，最短结论为：

```text
Stage 1 阈值学习有效；最终 q95 全局校准可用；但 Stage 2/3 尾部训练波动和
验证集退化明显，xi 贴下界，条件分箱存在低估。建议加入 best checkpoint、
early stopping，并重新检查 tail head 学习率、xi/beta 正则和 q90/q95 条件校准。
```
