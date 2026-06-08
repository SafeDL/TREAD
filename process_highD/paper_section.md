# 高维自然驾驶数据的尾事件提取与极值建模

## 1. 数据预处理

### 1.1 highD 轨迹数据加载

highD 数据集包含 60 段德国高速公路无人机航拍轨迹记录，每段记录包含三个 CSV 文件：`tracks.csv`（逐帧车辆轨迹）、`tracksMeta.csv`（车辆元数据）和 `recordingMeta.csv`（记录元数据）。每段记录以 25 Hz 的原始帧率采集，包含位置 $ (x, y) $、速度 $ (v_x, v_y) $、加速度 $ (a_x, a_y) $、车道 ID、前车 ID 等信息。

对于每段记录 $r \in \{1, \dots, 60\}$，加载模块构造 `HighDRecording` 对象，其中轨迹表以 `(vehicle_id, frame)` 为复合索引：

```math
\mathcal{T}^{(r)} = \{\tau_i^{(r)}\}_{i=1}^{N_r}, \quad \tau_i = \{(\mathbf{x}_{i,t}, \mathbf{v}_{i,t}, \mathbf{a}_{i,t}, \ell_{i,t}, p_{i,t})\}_{t=t_0}^{t_1}
```

其中 $\mathbf{x}_{i,t} = (x_{i,t}, y_{i,t})$ 为位置，$\mathbf{v}_{i,t} = (v^x_{i,t}, v^y_{i,t})$ 为速度，$\mathbf{a}_{i,t} = (a^x_{i,t}, a^y_{i,t})$ 为加速度，$\ell_{i,t}$ 为车道 ID，$p_{i,t}$ 为前车 ID。

### 1.2 行车方向归一化

highD 数据集包含双向行驶的车辆。为统一处理，对行驶方向为上行的车辆（`drivingDirection == 1`），将其坐标和速度向量绕原点旋转 $180^\circ$，使其与下行方向一致：

```math
\mathbf{x}_{i,t} \leftarrow -\mathbf{x}_{i,t}, \quad \mathbf{v}_{i,t} \leftarrow -\mathbf{v}_{i,t}, \quad \ell_{i,t} \leftarrow -\ell_{i,t}
```

### 1.3 异常轨迹过滤

对每条车辆轨迹进行物理可行性检验：

**加速度异常检测**：若任一时间步满足 $|\mathbf{a}_{i,t}| > a_{\max}$（默认 $a_{\max} = 8\ \text{m/s}^2$），则该车辆被整体剔除。

**加加速度异常检测**：通过后向差分计算加加速度：

```math
j_{i,t} = \frac{a_{i,t} - a_{i,t-1}}{\Delta t}
```

若 $\max_t |j_{i,t}| > j_{\max}$（默认 $j_{\max} = 30\ \text{m/s}^3$），则剔除该车辆。

**位置跳跃检测**：若相邻帧间位移超过 $\Delta x_{\max} = 5\ \text{m}$，则剔除该车辆。

### 1.4 重采样

将所有记录统一重采样至目标帧率 $f_{\text{target}} = 25\ \text{Hz}$。重采样采用最近邻插值，时间步长 $\Delta t = 1 / f_{\text{target}} = 0.04\ \text{s}$。

---

## 2. 交互事件提取

### 2.1 跟车片段提取

对每辆小客车（`class == Car`），扫描其 `precedingId` 时间序列，提取连续的前车相同且满足以下条件的片段：

1. **持续时间**：$\Delta t_{\text{seg}} \geq t_{\min} = 5.12\ \text{s}$（128 帧）
2. **同车道比例**：$\frac{|\{t : \ell_{\text{ego},t} = \ell_{\text{lead},t}\}|}{|\text{segment}|} \geq 0.80$
3. **正车间距**：$\text{median}_t(g_t) > 0.5\ \text{m}$，其中：

```math
g_t = x_{\text{lead},t} - x_{\text{ego},t} - \frac{L_{\text{ego}} + L_{\text{lead}}}{2}
```

对每个有效片段创建 `EventRecord`，记录起止帧、锚定帧（片段中点）及两车身份信息。

### 2.2 切入事件提取

切入事件提取分为三个步骤：

**步骤一：车道变换检测**。对每辆具有车道变换记录的车辆（`numLaneChanges >= 1`），通过 `lane_utils.detect_lane_changes()` 检测车道 ID 的离散转移。对于每个转移点，向前和向后搜索稳定车道段（至少 $\ell_{\text{stable}}$ 帧在同一车道），确定换道起止点。

**步骤二：受扰车主车匹配**。对每次检测到的车道变换，首先尝试通过 `followingId` 寻找紧随的受扰车辆；若不存在，则搜索目标车道中位于变道车辆后方的最近车辆。

**步骤三：切入起止估计**。利用横向速度阈值 $\bar{v}_y = 0.10\ \text{m/s}$ 精确确定切入运动的起止帧：

```math
\begin{aligned}
t_{\text{start}} &= \arg\min_t \{ t : |v_{y,t}| > \bar{v}_y \text{ 且持续至少 } k_{\min} \text{ 帧} \} \\
t_{\text{end}} &= \arg\max_t \{ t : |v_{y,t}| > \bar{v}_y \text{ 且持续至少 } k_{\min} \text{ 帧} \}
\end{aligned}
```

**筛选条件**：
- 切入后车间距 $g_{t_{\text{end}}} \in (0, 150]\ \text{m}$
- 切入后同车道比例 $\geq 0.70$
- 若要求紧邻前车（`require_immediate_leader`），则验证受扰车与切入车之间不存在其他车辆

---

## 3. 风险变量构造

### 3.1 纵向风险变量 $Y_{\text{long}}$

对每个跟车事件，在锚定帧 $t_a$ 处计算复合纵向风险指标。核心指标包括：

**时距（THW）**：

```math
\text{THW}_t = \frac{g_t}{\max(v_{\text{ego},t}, v_{\min})}
```

**碰撞时间（TTC）**：

```math
\text{TTC}_t = \frac{g_t}{\max(v_{\text{ego},t} - v_{\text{lead},t}, \varepsilon)}
```

**避免碰撞所需减速度（DRAC）**：

```math
\text{DRAC}_t = \frac{(v_{\text{ego},t} - v_{\text{lead},t})_+^2}{2 g_t}
```

通过 softmax 池化聚合时间序列风险：

```math
Y_{\text{long}} = \frac{\sum_t w_t \cdot r_t \cdot e^{\beta r_t}}{\sum_t e^{\beta r_t}}
```

其中 $r_t$ 为各指标的归一化风险值，$\beta$ 为池化温度参数。

最终风险得分映射到极值标度：

```math
S_{\text{EVT}}(y) = -\log \mathbb{P}(Y_{\text{long}} > y)
```

### 3.2 切入风险变量 $Y_{\text{cutin}}$

切入风险变量在纵向风险基础上引入横向维度：

```math
Y_{\text{cutin}} = f(Y_{\text{long}}, \Delta y, \text{LTG}, v_{\text{app}}, d_{\text{safe}})
```

其中 $\Delta y$ 为横向偏移，LTG（Lateral Time Gap）为横向时距，$v_{\text{app}}$ 为接近速度，$d_{\text{safe}}$ 为安全距离亏空。

---

## 4. 极值理论建模

### 4.1 峰值解聚

为避免同一事件的多个连续帧被重复计入，对每个 $(recording\_id, ego\_id)$ 组内的峰值进行 $5\ \text{s}$ 窗口的解聚（declustering）：

```math
\mathcal{P}_{\text{ind}} = \left\{ \max_{t \in C_k} Y_t : C_k \text{ 为时间间隔 } \geq 5\text{s 的聚类} \right\}
```

### 4.2 POT 阈值选择

采用 Peak-Over-Threshold (POT) 方法。阈值 $u$ 的选择通过加权稳定性扫描实现。

对于候选阈值 $u_k$（对应 $k$ 个超出量，$k \in [k_{\min}, k_{\max}]$），超出量 $\{Y_i - u_k : Y_i > u_k\}$ 拟合广义帕累托分布（GPD）：

```math
H(y; \xi, \sigma) = 1 - \left(1 + \xi \frac{y - u}{\sigma}\right)^{-1/\xi}, \quad y > u
```

其中 $\xi \in \mathbb{R}$ 为形状参数，$\sigma > 0$ 为尺度参数。

选择使形状参数 $\xi$ 的加权方差最小的阈值：

```math
u^* = \arg\min_{u_i} \frac{\sum_{j=s}^{i} w_j (\xi_j - \bar{\xi}_i)^2}{\sum_{j=s}^{i} w_j}, \quad w_j = j^{0.25}
```

### 4.3 GPD 拟合

对选定的阈值 $u^*$，超出量通过最大似然估计拟合 GPD：

```math
(\hat{\xi}, \hat{\sigma}) = \arg\max_{\xi, \sigma} \sum_{i: y_i > u^*} \log h(y_i - u^*; \xi, \sigma)
```

其中 $h(\cdot)$ 为 GPD 密度函数。使用 `scipy.stats.genpareto.fit` 实现。

### 4.4 重现水平

对于重现期 $N$（以独立峰值计数），重现水平 $z_N$ 满足：

```math
\mathbb{P}(Y > z_N) = \frac{1}{N}
```

由 GPD 模型导出：

```math
z_N = \begin{cases}
u + \dfrac{\sigma}{\xi} \left[(N \cdot \lambda_u)^\xi - 1\right], & \xi \neq 0 \\[10pt]
u + \sigma \log(N \cdot \lambda_u), & \xi = 0
\end{cases}
```

其中 $\lambda_u = \mathbb{P}(Y > u)$ 为阈值超出率。通过 200 次 Bootstrap 重采样计算 90% 置信区间。

### 4.5 诊断绘图

生成以下诊断图以验证 EVT 模型拟合质量：
- 超出量直方图与拟合 GPD 密度叠加
- 经验生存函数与 GPD 生存函数对比
- 阈值稳定性图（$\xi$ 和修正尺度随阈值变化）
- 平均超出量图（mean excess plot）
- QQ 图和 PP 图

---

## 5. 暴露量估计

### 5.1 行驶暴露量

对于跟车场景，暴露量 $E_{\text{follow}}$ 为所有受扰车（ego）在跟车片段内的累计行驶距离：

```math
E_{\text{follow}} = \sum_{r} \sum_{i \in \mathcal{E}^{(r)}} \int_{t \in \mathcal{S}_{i}^{(r)}} v_{\text{ego}}(t) \, dt
```

其中 $\mathcal{E}^{(r)}$ 为记录 $r$ 中所有受扰车集合，$\mathcal{S}_{i}^{(r)}$ 为其跟车时间区间。

对于切入场景，暴露量 $E_{\text{cutin}}$ 为记录中所有车辆的累计行驶距离：

```math
E_{\text{cutin}} = \sum_{r} \sum_{i} \int_{0}^{T_r} v_i(t) \, dt
```

### 5.2 尾部峰值率

尾部峰值率（每单位暴露的独立尾事件数）：

```math
\lambda_{\text{tail}} = \frac{N_{\text{ind}}(Y > u^*)}{E}
```

以每英里和每小时两种单位报告。安全关键强度为：

```math
\lambda_{\text{crit}} = \lambda_{\text{tail}} \cdot \mathbb{P}(Y > x_c \mid Y > u^*) = \lambda_{\text{tail}} \cdot H(x_c - u^*; \hat{\xi}, \hat{\sigma})
```

其中 $x_c = 5.0$（跟车）或 $10.0$（切入）为碰撞临界水平。

---

## 6. 长尾场景背景空间构建

### 6.1 尾部场景条件向量

长尾背景空间直接对齐 diffusion 的 anchor-frame `scenario_conditions`。following 使用
7 维条件：

```math
\mathbf{c}_{\text{follow}} =
\left[v_{x,\text{ego},0}, g_0, \Delta v_0, a_{x,\text{lead},0},
\Delta v_{\text{lead},0:H}, \min_t a_{x,\text{lead},t}, T_{\text{brake}}\right]
```

cut-in 使用 10 维条件：

```math
\mathbf{c}_{\text{cutin}} =
\left[v_{x,\text{ego},0}, g_0, \Delta y_0, \Delta v_{x,0},
v_{y,\text{target},0}, a_{y,\text{target},0}, y_{\text{final}},
t_{\text{cross}}, \Delta v_{\text{target},0:H},
\left.\frac{dy}{dx}\right|_{\text{cross}}\right]
```

### 6.2 经验尾事件背景

直接选取所有独立尾部峰值（$Y > u^*$）对应的背景状态，标记为 `highd_independent_tail_peak`。

### 6.3 合成尾事件背景（Gaussian Copula）

为解决经验尾事件样本量不足的问题，生成合成尾事件背景：

**步骤一：边缘分布变换**。对每个条件维度使用经验 CDF 得到伪观测，并映射到标准正态空间：

```math
\mathbf{u}_{i,j} = \hat{F}_j(c_{i,j}), \qquad
\mathbf{z}_{i,j} = \Phi^{-1}(\mathbf{u}_{i,j})
```

**步骤二：联合相关结构拟合**。在正态得分空间拟合相关矩阵：

```math
\hat{\mathbf{R}} = \operatorname{corr}(\mathbf{z}) + \lambda \mathbf{I}
```

其中 $\lambda = 10^{-4}$ 用于数值正则化，边缘分布裁剪默认使用 0.01 分位数。

**步骤三：采样与物理重构**。从 Gaussian copula 采样新的条件向量，并在标准化条件空间中寻找最近邻经验尾样本，以其初始状态为基础重构物理一致的背景：

```math
b^* = \arg\min_b \|\tilde{\mathbf{c}}' - \tilde{\mathbf{c}}_b\|_2
```

重构规则：
- 直接写入采样后的 `scenario_conditions`
- 由初始 gap、ego 速度、相对速度和横向偏移重建 `initial_states`
- following 额外写入 `lead_ax_0`；cut-in 额外写入 `target_vy_0`、`target_ay_0`、`final_lateral_offset` 和 cross 相关条件
- 未来窗口摘要条件只作为 diffusion 条件先验，不反向构造历史轨迹

following 默认保留全部 empirical tail contexts，并额外生成 5000 个
`highd_tail_gaussian_copula` contexts。cut-in 默认保存 Gaussian copula 条件分布，并通过
diffusion prior 生成 5000 个满足语义筛选的 cut-in scenarios。

### 6.4 背景数据集输出

following 输出 `tail_contexts.npz`，包含：
- `scenario_conditions`: diffusion 条件向量，包含初始关系和 125 步参考窗口摘要
- `initial_states`: $(N, 2, 6)$ anchor-frame 初始状态，用于闭环积分和回放
- 每背景的元数据（事件 ID、记录 ID、风险得分、车间距、TTC、车道信息等）

cut-in 输出 `scenario_condition_distribution.npz` 和
`diffusion_generated_scenarios.npz`。后者包含采样条件、重构初始状态、扩散动作、
target 轨迹、语义筛选标志和回放所需的 `base_event_id`。两类场景都会在
`generated/figures/` 输出条件分布、机动指标和轨迹族对比图，并在
`generated/event_playbacks/` 输出抽样 GIF。
