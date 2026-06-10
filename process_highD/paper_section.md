# 高维自然驾驶数据的尾事件提取与极值建模

## 1. 数据预处理

### 1.1 highD 轨迹加载

highD 数据集包含 60 段德国高速公路无人机航拍轨迹记录。每段 recording 由
`tracks.csv`、`tracksMeta.csv` 和 `recordingMeta.csv` 三类文件组成，原始帧率通常为
25 Hz。逐帧轨迹包含车辆位置、速度、加速度、车道 ID、前车 ID、后车 ID 以及左右相邻车道车辆
ID 等字段。

对 recording $r$，代码构造 `HighDRecording` 对象，并以 `(vehicle_id, frame)` 为复合索引保存
轨迹表：

```math
\mathcal{T}^{(r)} =
\{\tau_i^{(r)}\}_{i=1}^{N_r}, \qquad
\tau_i =
\{(x_{i,t}, y_{i,t}, v^x_{i,t}, v^y_{i,t},
a^x_{i,t}, a^y_{i,t}, \ell_{i,t}, p_{i,t})\}_{t=t_i^0}^{t_i^1}.
```

其中 $\ell_{i,t}$ 为车道 ID，$p_{i,t}$ 为 highD 的 `precedingId`。原始 highD 使用 0 表示
无效车辆 ID；加载时统一替换为 -1，便于后续筛选。

### 1.2 坐标与行驶方向归一化

highD 原始 `x,y` 表示 bounding box 左上角。预处理首先根据 `tracksMeta` 中的车辆尺寸将其转换为
车辆几何中心：

```math
x_{i,t} \leftarrow x_{i,t} + \frac{w_i}{2}, \qquad
y_{i,t} \leftarrow y_{i,t} + \frac{h_i}{2}.
```

随后统一纵向行驶方向。对 `drivingDirection == 1` 的车辆，代码翻转纵向位置、纵向速度和纵向加速度：

```math
x_{i,t} \leftarrow -x_{i,t}, \qquad
v^x_{i,t} \leftarrow -v^x_{i,t}, \qquad
a^x_{i,t} \leftarrow -a^x_{i,t}.
```

若存在 `precedingXVelocity`，也按同一车辆行驶方向翻转。实现不翻转横向坐标 $y$，也不重编码
lane ID；后续 lane 关系直接使用 highD 原始车道 ID 和 `recordingMeta` 中的 lane marking。
归一化后，所有车辆的纵向前进方向均为 positive x。

### 1.3 异常帧标记

预处理使用物理规则标记异常帧，并在事件提取阶段排除含异常帧的 ego-target 事件窗口。默认规则为：

- 纵向加速度异常：$|a^x_{i,t}| > 8\ \mathrm{m/s^2}$。
- jerk 异常：对 $a^x$ 做逐车差分，若存在 $a^y$ 则同时检查横向 jerk；
  默认阈值为 $30\ \mathrm{m/s^3}$。
- 纵向位置跳跃：相邻帧中心点 $x$ 差值超过 $5\ \mathrm{m}$。
- 车辆尺寸缺失或非正。
- 可选低速过滤：默认 `min_vehicle_speed=0.0`，因此不启用。

这些规则写入 `_abnormal` 标记列，而不是立即删除整车轨迹。帧连续性检查会记录诊断信息；当前实现
不单独因帧不连续标记异常。

### 1.4 帧率处理

默认配置 `sampling.target_fps=25`，与 highD 原始帧率一致，因此通常不发生重采样。若目标帧率不同，
`resample_recording()` 按 `source_fps / target_fps` 的步长对每辆车轨迹抽帧，并更新 recording
metadata 中的 `frameRate`。本文后续默认时间步为
$\Delta t = 1/25 = 0.04\ \mathrm{s}$。

---

## 2. 交互事件提取

### 2.1 跟车片段

对每辆小客车 ego 扫描 `precedingId` 序列，提取连续前车不变的片段。片段必须满足：

1. ego 和 lead 均为 `class == car`；
2. `precedingId` 连续时长不少于 128 帧，即 $5.12\ \mathrm{s}$；
3. ego 在片段内不换道；
4. ego 与 lead 同车道比例不少于 0.80；
5. 两车净间距的中位数大于 $0.5\ \mathrm{m}$；
6. ego 或 lead 在该窗口内没有 `_abnormal` 帧。

净间距按车辆中心和纵向尺寸计算：

```math
g_t = x_{\mathrm{lead},t} - x_{\mathrm{ego},t}
- \frac{L_{\mathrm{ego}} + L_{\mathrm{lead}}}{2}.
```

每个有效片段写为一个 `EventRecord`，包含 recording、ego/target ID、片段起止帧和片段中点
`anchor_frame`。事件提取阶段还写出两类 following cache：`following_event_contexts.npz` 保存
用于风险评分和 tail selection 的事件级 context；`following_event_segments.npz` 保存完整
following 片段，供 diffusion 训练阶段按 stride 重切固定长度窗口。

### 2.2 切入事件

cut-in 提取以目标车换道为入口，分为三步。

**车道变化检测。** 对 `numLaneChanges >= 1` 且 `class == car` 的车辆，使用
`lane_utils.detect_lane_changes()` 检测 lane ID 离散转移，并要求换道前后存在至少
`min_lane_stable_steps=10` 帧稳定车道。若 recording 提供 lane marking 信息，则只保留相邻车道
之间的变化。

**受扰 ego 匹配。** 对每次换道，优先使用 target 在稳定进入目标车道后或 cross frame 的
`followingId`；若不可用，则在目标车道中搜索位于 target 后方的最近非 truck 车辆。候选 ego 必须
与 target 同处目标车道、target 位于 ego 前方，并且两者之间不存在其他车辆。

**语义窗口筛选。** cut-in 的语义起点使用换道前稳定源车道段的起点，终点使用稳定进入目标车道后的
帧，并在横向速度未明显收敛时向后延伸。有效 cut-in 必须满足：

- 换道持续时间不少于 `min_cutin_duration_steps=3` 帧；
- `cross_frame` 位于 ego-target 公共帧范围内；
- cross 后至少 `min_post_cutin_duration_seconds=2.0` 秒存在共同轨迹；
- post-cutin 净间距中位数在 $[0,150]\ \mathrm{m}$ 内；
- post-cutin 同车道比例不少于 0.70；
- 默认 `require_immediate_leader=true` 时，cross 后最小持续窗口内 target 必须是 ego 的最近前车；
- 语义窗口内 ego 或 target 没有 `_abnormal` 帧。

默认 `cutin.anchor_phase="cross"`，因此 `EventRecord.anchor_frame` 为 lane-boundary crossing frame。
风险与 diffusion 使用的 context 不直接等同于该字段：EVT/tail context 的 anchor 从
$t_{\mathrm{cross}}-\delta$ 中按事件确定性选择，
$\delta \in \{15,20,25,30,35,45,50\}$ 帧，并保存固定 100 个 future steps。

---

## 3. 风险变量

### 3.1 跟车风险 $Y_{\mathrm{long}}$

following 风险在与 diffusion 对齐的固定 125 步 context 上计算。原始 following 事件仍以
不等长自然片段 $[t_s,t_e]$ 保存，并额外写入完整片段 cache 供 diffusion 训练滑窗使用；
EVT 使用的是 `context_anchor_frame` 后的 125 个 future steps。
核心纵向指标包括时距、碰撞时间和避免碰撞所需减速度：

```math
\mathrm{THW}_t = \frac{g_t}{\max(v_{\mathrm{ego},t}, v_{\min})},
```

```math
\mathrm{TTC}_t =
\frac{g_t}{\max(v_{\mathrm{ego},t} - v_{\mathrm{lead},t}, \varepsilon)},
```

```math
\mathrm{DRAC}_t =
\frac{(v_{\mathrm{ego},t} - v_{\mathrm{lead},t})_+^2}{2g_t}.
```

这些时间序列指标在 `tools/highd_longitudinal.py` 中被转换为统一方向的风险量，并聚合为
$Y_{\mathrm{long}}$。后续 EVT 模型将其映射为极值分数：

```math
S_{\mathrm{EVT}}(y) = -\log \Pr(Y_{\mathrm{long}} > y).
```

### 3.2 切入风险 $Y_{\mathrm{cutin}}$

cut-in 风险在固定 100 步 context 上计算。context anchor 为
$t_{\mathrm{cross}}-\delta$，其中
$\delta \in \{15,20,25,30,35,45,50\}$ 帧，并要求窗口覆盖 cross 后至少 2 秒。实现中记录
`risk_start_index`，对应 fixed context 中的 cross-relative 起点；纵向 TTC、DRAC、安全距离亏空和
post-cutin 最小 gap 等指标只在该索引后的窗口内计算。

切入风险变量在纵向风险基础上加入横向接近和切入语义：

```math
Y_{\mathrm{cutin}} =
f(Y_{\mathrm{long}}, \Delta y, \mathrm{LTG}, v_{\mathrm{app}}, d_{\mathrm{safe}}).
```

其中 $\Delta y$ 为相对横向偏移，LTG 为 lateral time gap，$v_{\mathrm{app}}$ 为横向接近速度，
$d_{\mathrm{safe}}$ 为安全距离亏空。若轨迹未通过 overlap、横向接近、进入后保持和
front-cutin 等语义门控，默认配置下令 $Y_{\mathrm{cutin}}=0$。只有 `is_cutin >= 0.5` 的
语义 cut-in 进入 cut-in EVT、tail condition 建模和 diffusion 生成流程。

---

## 4. 极值理论建模

### 4.1 独立峰值解聚

为了避免同一交互对的相邻高风险样本重复计入，代码对风险峰值做 5 秒 run-length
declustering。following 按 `(recording_id, ego_id)` 分组；cut-in 按
`(recording_id, ego_id, target_id)` 分组。每个 cluster 保留风险最大的代表事件：

```math
\mathcal{P}_{\mathrm{ind}} =
\{\max_{t \in C_k} Y_t : C_k \text{ separated by at least } 5\ \mathrm{s}\}.
```

### 4.2 POT/GPD 拟合

对独立峰值采用 Peak-Over-Threshold 方法。给定阈值 $u$，超出量
$e_i = Y_i-u \mid Y_i>u$ 拟合广义帕累托分布：

```math
\Pr(e \le z \mid Y>u) =
1 - \left(1+\xi\frac{z}{\sigma}\right)^{-1/\xi},
\qquad z>0.
```

阈值选择由 `process_highD/src/evt_fitting.py` 完成：扫描候选阈值和对应的超出量个数，对每个候选
拟合 GPD，并选择形状参数 $\xi$ 在尾部区间内最稳定的阈值。最终参数
$(\hat{\xi},\hat{\sigma})$ 使用 `scipy.stats.genpareto.fit` 拟合。

### 4.3 重现水平与极值分数

若 $N$ 表示独立峰值计数的重现期，重现水平 $z_N$ 满足 $\Pr(Y>z_N)=1/N$。在 POT/GPD 模型下：

```math
z_N =
\begin{cases}
u + \dfrac{\sigma}{\xi}\left[(N\lambda_u)^\xi - 1\right],
& \xi \ne 0, \\[8pt]
u + \sigma \log(N\lambda_u),
& \xi = 0,
\end{cases}
```

其中 $\lambda_u=\Pr(Y>u)$ 为独立峰值的阈值超出率。模型同时提供 survival-based EVT score
$S_{\mathrm{EVT}}(y)=-\log\Pr(Y>y)$，用于统一比较 following 和 cut-in 长尾强度。

### 4.4 诊断输出

EVT 拟合输出模型 JSON、summary JSON 和诊断图。following 还输出 return-level-vs-distance 图，
用于把 GPD 尾部分布与 highD following ego 暴露量连接起来。典型诊断包括超出量直方图、
经验 survival 与 GPD survival 对比、阈值稳定性、mean excess、QQ 和 PP 图。

---

## 5. 暴露量估计

### 5.1 following 暴露量

following 暴露量以有效 following 片段内 ego 的累计行驶距离和时长为分母：

```math
E_{\mathrm{follow}} =
\sum_r \sum_{e \in \mathcal{F}^{(r)}} \int_{t \in e}
v_{\mathrm{ego}}(t)\,dt.
```

`estimate_following_exposure.py` 同时报告 all-vehicle 口径的对照指标，但主要
tail peak rate 和 safety-critical intensity 使用 `following_ego_miles` 和
`following_ego_hours`。

### 5.2 cut-in 暴露量

cut-in 暴露量使用 highD 全车辆累计行驶距离和时长：

```math
E_{\mathrm{cutin}} =
\sum_r \sum_i \int_{t_i^0}^{t_i^1} v_i(t)\,dt.
```

该口径把 cut-in 视为全交通流中的稀有事件，而不是只在已识别 cut-in 的局部交互窗口内归一化。

### 5.3 尾事件率和安全关键强度

独立尾峰值率为：

```math
\lambda_{\mathrm{tail}} =
\frac{N_{\mathrm{ind}}(Y>u)}{E}.
```

若安全关键水平 $x_c$ 高于阈值 $u$，则使用 GPD 条件 survival 外推：

```math
\lambda_{\mathrm{crit}} =
\lambda_{\mathrm{tail}} \cdot
\Pr(Y \ge x_c \mid Y>u).
```

following 与 cut-in 默认均使用 $x_c=5.0$，表示以碰撞 bonus 对应的 raw risk 水平作为
safety-critical reference。若 $x_c \le u$，实现退回到独立峰值经验概率口径，避免在阈值以下使用
GPD 外推。

---

## 6. 自然驾驶 Diffusion 数据集

`build_natural_dataset.py` 根据配置调用 `diffusion.src.data.build_action_dataset()`。输出包括
`dataset.npz`、`dataset_normalized.npz`、`normalization_stats.json`、
`train_val_test_split.json` 和 `feature_schema.json`。模型直接条件为 `scenario_conditions`；
`initial_states` 用于动作积分、评价、tail 重构和回放。

### 6.1 following 数据集

following 使用 125 步，即 5 秒窗口。数据构建优先读取 `following_event_segments.npz` 中的完整片段，
默认 train stride 为 5 帧，validation/test stride 为 25 帧，每个事件最多保留 12 个窗口。条件向量为：

```math
\mathbf{c}_{\mathrm{follow}} =
\left[
v^x_{\mathrm{ego},0},\ g_0,\ \Delta v_0,\ a^x_{\mathrm{lead},0},\
\Delta v^x_{\mathrm{lead},0:H},\
\min_t a^x_{\mathrm{lead},t},\
T_{\mathrm{brake}}
\right].
```

其中 $\Delta v_0=v^x_{\mathrm{ego},0}-v^x_{\mathrm{lead},0}$。动作目标是 lead 车纵向 jerk。
实现先对 lead 速度做 Savitzky-Golay 平滑，再差分为加速度和 jerk，并按配置裁剪加速度和 jerk。

### 6.2 cut-in 数据集

cut-in 使用 100 步，即 4 秒窗口。数据集只使用 `cutin_event_scores.csv` 中
`is_cutin >= 0.5` 的语义事件。anchor 由
$t_{\mathrm{cross}}-\delta$ 给出，$\delta \in \{15,20,25,30,35,45,50\}$ 帧；窗口必须包含
cross 后至少 2 秒，但默认不要求 `cutin_end_frame` 在窗口内。条件向量为：

```math
\mathbf{c}_{\mathrm{cutin}} =
\left[
v^x_{\mathrm{ego},0},\ g_0,\ \Delta y_0,\ \Delta v^x_0,\
a^x_{\mathrm{target},0},\ v^y_{\mathrm{target},0},\
a^y_{\mathrm{target},0},\ y_{\mathrm{final}},\
t_{\mathrm{cross}},\ \Delta v^x_{\mathrm{target},0:H}
\right].
```

动作目标是 target 车在 ego-anchor 坐标系下的 $(a_x,a_y)$ 序列；实现裁剪纵向/横向加速度、
jerk、yaw rate 和 lane width 相关异常样本。风险、EVT 和 tail context 使用与 diffusion 对齐的
fixed-window 口径；原始不等长 cut-in 事件只作为事件发现和训练窗口来源。

---

## 7. 长尾场景背景空间

### 7.1 tail scenario condition

长尾背景空间直接对齐 diffusion 的 `scenario_conditions`。following 使用 7 维条件：

```text
ego_vx_0, initial_gap, initial_delta_v, lead_ax_0,
lead_speed_change, lead_min_ax, lead_braking_duration
```

cut-in 使用 10 维条件：

```text
ego_vx_0, initial_gap, initial_lateral_offset, initial_delta_vx,
target_ax_0, target_vy_0, target_ay_0, final_lateral_offset,
time_to_cross, target_speed_change
```

Gaussian copula 拟合时对 gap 维度使用 `log_initial_gap`，以改善正态空间中的边缘形态；写入
diffusion 条件时恢复为实际 gap。

### 7.2 经验尾事件背景

EVT exposure 脚本输出 `highd_independent_tail_peaks.csv`。tail selection 使用这些独立尾峰值反查
event context，得到经验 tail contexts。following 的经验 source type 为
`highd_independent_tail_peak`；cut-in 的经验 source type 为 `highd_evt_independent_tail_peak`。

### 7.3 Gaussian copula 合成背景

为缓解经验尾样本量不足，tail selection 对经验 tail condition 拟合 Gaussian copula：

```math
u_{i,j} = \hat{F}_j(c_{i,j}), \qquad
z_{i,j} = \Phi^{-1}(u_{i,j}),
```

```math
\hat{\mathbf{R}} =
\operatorname{corr}(\mathbf{z}) + \lambda \mathbf{I},
\qquad \lambda=10^{-4}.
```

采样后，代码在标准化特征空间中查找最近的经验 tail 事件，用其几何尺寸和未建模状态作为 base，
再用采样得到的初始 gap、相对速度、横向偏移、初始加速度等变量重构 `initial_states`。重构后的
`initial_states` 用于动作积分和 playback，不作为 diffusion denoiser 的条件输入。

following 和 cut-in 都保存 `scenario_condition_distribution.npz`，其中包含 copula correlation、
variable mask、empirical marginal values 和 tail metadata。following 默认保留全部 empirical
contexts，并额外生成 5000 个 `highd_tail_gaussian_copula` synthetic contexts；cut-in 默认通过
cut-in diffusion prior 生成 5000 个 target 轨迹，若语义 hard mask 后数量不足，则从同一 copula
分布补采样并继续解码。

### 7.4 输出与闭环 ego 响应

following 输出 `scenario_condition_distribution.npz`、`tail_contexts.npz` 和
`diffusion_generated_scenarios.npz`。后者包含采样条件、重构初始状态、diffusion jerk 动作、
积分得到的 lead 加速度和 lead 轨迹。

cut-in 输出 `scenario_condition_distribution.npz`、`tail_contexts.npz` 和
`diffusion_generated_scenarios.npz`。`tail_contexts.npz` 保存 empirical independent tail peaks 和
采样 condition 的重构状态，供 `subset/` 通过最近邻 base context 重构初始状态；generated scenarios
包含采样条件、重构初始状态、diffusion 动作、target 轨迹、语义筛选标志、质量指标和回放所需的
`base_event_id`。

两个 select 脚本只生成 adversary 轨迹：following 为 lead，cut-in 为 target。闭环 ego 响应由
`play_following_tail_events.py` 和 `play_cutin_tail_events.py` 在回放阶段调用 highway-env IDM
生成；同一 IDM 参数由 `tools/idm_ego.yaml` 管理。`base_event_id` 仅用于回放时反查原始 highD
recording、对齐背景车并排除原始 ego/target，不用于生成 ego 轨迹。

上述随机 condition 采样、扩散积分和与 highD 长尾事件的分布对比用于验证条件扩散模型在给定
scenario condition 下的场景复现能力；安全关键概率估计由 `subset/` 在相同
scenario-condition 联合分布和 diffusion latent 空间上执行。
