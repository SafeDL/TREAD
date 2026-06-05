# 基于扩散潜在空间的子集模拟与安全关键概率估计

## 1. 问题形式化

### 1.1 失效概率估计

本模块旨在估计自动驾驶系统在长尾场景下的闭环失效概率。设 $Y = g(\mathbf{c}, \mathbf{z})$ 为通过闭环仿真获得的风险评分，其中 $\mathbf{c}$ 为场景背景（从 highD 尾部背景集中采样），$\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ 为扩散模型的潜在变量。定义失效事件 $\mathcal{F} = \{Y \geq y_{\text{crit}}\}$，其中 $y_{\text{crit}}$ 为预设的安全关键阈值。目标是估计：

```math
P_f = \mathbb{P}_{\mathbf{c} \sim \mathcal{U}(\mathcal{C}_{\text{tail}}),\; \mathbf{z} \sim \mathcal{N}(0, \mathbf{I})} \left[ g(\mathbf{c}, \mathbf{z}) \geq y_{\text{crit}} \right]
```

### 1.2 确定性映射

由于扩散模型采用 DDIM 确定性采样，动作序列是背景和潜在变量的确定性函数：

```math
\mathbf{a} = \mathcal{D}(\mathbf{c}, \mathbf{z}; \theta), \quad \mathcal{D}: \mathcal{C} \times \mathcal{Z} \to \mathcal{A}
```

这确保了子集模拟中对 $(\mathbf{c}, \mathbf{z})$ 的操作在动作空间中有明确定义。

### 1.3 闭环仿真

动作序列 $\mathbf{a}$ 在 highway-env 仿真环境中执行，产生闭环轨迹。仿真采用模型预测控制（MPC）风格的滚动规划：

```math
\mathbf{a}^{(p)} = \mathcal{D}(\mathbf{c}^{(p)}, \mathbf{z}^{(p)}), \quad p = 1, 2, \dots, \lceil T_{\text{episode}} / T_{\text{commit}} \rceil
```

其中 $\mathbf{c}^{(p)}$ 为第 $p$ 次规划时的观测背景，$T_{\text{commit}} = 25$ 步为每次规划的执行步数，$T_{\text{episode}} = 150$ 步为总仿真长度。总潜在变量维度为：

```math
\dim(\mathbf{Z}) = \lceil T_{\text{episode}} / T_{\text{commit}} \rceil \times T_{\text{horizon}} \times d_{\text{action}}
```

---

## 2. 子集模拟算法

### 2.1 标准子集模拟

子集模拟（Au & Beck, 2001）通过构造一系列嵌套的中间失效域，将小概率 $P_f$ 分解为条件概率的乘积：

```math
\mathcal{F} = \mathcal{F}_m \subset \mathcal{F}_{m-1} \subset \cdots \subset \mathcal{F}_1 \subset \mathcal{F}_0 = \Omega
```

```math
P_f = \mathbb{P}(\mathcal{F}_m) = \mathbb{P}(\mathcal{F}_1) \prod_{i=1}^{m-1} \mathbb{P}(\mathcal{F}_{i+1} \mid \mathcal{F}_i)
```

其中每个中间失效域定义为 $\mathcal{F}_i = \{Y \geq y_i\}$，阈值 $y_i$ 自适应确定。

### 2.2 算法流程

**第 0 层（初始采样）**：

从 $\mathbf{c} \sim \mathcal{U}(\mathcal{C}_{\text{tail}})$ 和 $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$ 中独立采样 $N$ 个样本（跟车 $N = 10000$，切入 $N = 1000$）：

```math
\{(\mathbf{c}_j^{(0)}, \mathbf{z}_j^{(0)})\}_{j=1}^{N}, \quad Y_j^{(0)} = g(\mathbf{c}_j^{(0)}, \mathbf{z}_j^{(0)})
```

设置阈值 $y_1$ 为样本得分的 $(1 - p_0)$ 分位数（$p_0 = 0.1$）：

```math
y_1 = Q_{1-p_0}(\{Y_j^{(0)}\})
```

选取精英样本（得分 $\geq y_1$ 的前 $\lceil N \cdot p_0 \rceil$ 个）：

```math
\mathcal{E}_1 = \{(\mathbf{c}_j^{(0)}, \mathbf{z}_j^{(0)}) : Y_j^{(0)} \geq y_1\}
```

**第 $i$ 层（$i \geq 1$）**：

从精英集 $\mathcal{E}_i$ 出发，通过马尔可夫链蒙特卡洛（MCMC）生成 $N$ 个服从条件分布 $\pi_i(\mathbf{c}, \mathbf{z}) \propto \pi_0(\mathbf{c}, \mathbf{z}) \cdot \mathbf{1}[g(\mathbf{c}, \mathbf{z}) \geq y_i]$ 的样本：

1. 将 $|\mathcal{E}_i|$ 个精英样本均匀分配为多条马尔可夫链的初始状态
2. 对每条链，重复以下步骤直至生成 $N$ 个样本：
   - 从当前状态 $(\mathbf{c}, \mathbf{z})$ 出发，生成提议 $(\mathbf{c}', \mathbf{z}')$
   - 评估 $Y' = g(\mathbf{c}', \mathbf{z}')$
   - 若 $Y' < y_i$，拒绝提议，保留当前状态
   - 否则，按 Metropolis-Hastings 准则接受：

```math
\alpha = \min\left(1, \frac{\pi_0(\mathbf{c}', \mathbf{z}') \cdot q(\mathbf{c}, \mathbf{z} \mid \mathbf{c}', \mathbf{z}')}{\pi_0(\mathbf{c}, \mathbf{z}) \cdot q(\mathbf{c}', \mathbf{z}' \mid \mathbf{c}, \mathbf{z})}\right)
```

设置新阈值 $y_{i+1} = Q_{1-p_0}(\{Y_j^{(i)}\})$，选取新精英集 $\mathcal{E}_{i+1}$。

**终止条件**：

若 $y_{i+1} \geq y_{\text{crit}}$（阈值超过安全关键水平），或达到最大层数（$m_{\max} = 8$），则停止。

**失效概率估计**：

```math
\hat{P}_f = p_0^{m-1} \cdot \hat{p}_m, \quad \hat{p}_m = \frac{1}{N} \sum_{j=1}^{N} \mathbf{1}\left[Y_j^{(m)} \geq y_{\text{crit}}\right]
```

### 2.3 Metropolis-Hastings 提议机制

**潜在空间扰动**（概率 $1 - \rho_{\text{refresh}}$）：

```math
\mathbf{z}' \sim \mathcal{N}(\mathbf{z}, \sigma_p^2 \mathbf{I}), \quad \mathbf{c}' = \mathbf{c}
```

其中 $\sigma_p = 0.08$（跟车）或 $0.10$（切入）。提议比为：

```math
\frac{q(\mathbf{z} \mid \mathbf{z}')}{q(\mathbf{z}' \mid \mathbf{z})} = \frac{\exp\left(-\frac{1}{2\sigma_p^2}\|\mathbf{z} - \mathbf{z}'\|^2\right)}{\exp\left(-\frac{1}{2\sigma_p^2}\|\mathbf{z}' - \mathbf{z}\|^2\right)} = 1
```

先验比为：

```math
\frac{\pi_0(\mathbf{z}')}{\pi_0(\mathbf{z})} = \exp\left(-\frac{1}{2}\sum_k (z_k'^2 - z_k^2)\right)
```

**背景刷新**（概率 $\rho_{\text{refresh}}$）：

```math
\mathbf{c}' \sim \mathcal{U}(\mathcal{C}_{\text{tail}}), \quad \mathbf{z}' \sim \mathcal{N}(0, \mathbf{I})
```

若新样本得分低于阈值则拒绝。背景刷新可增强链的混合性，防止陷入局部模式。

**重试机制**：每条链每次迭代最多尝试 $R$ 次提议（跟车 $R=2$，切入 $R=4$），若全部失败则保留当前状态。

### 2.4 去重与唯一性诊断

通过状态键 `(context_index, latent_bytes_hash)` 跟踪唯一状态。报告指标包括：
- 每层唯一背景数 $N_{\text{unique}}^{\text{context}}$
- 每层唯一状态数 $N_{\text{unique}}^{\text{state}}$
- 最大单状态份额 $\max_j p_j$
- MH 接受率

可靠性阈值：$N_{\text{unique}}^{\text{context}} \geq N_{\text{crit}}$（跟车 50，切入 80），接受率 $\geq 0.10$。

---

## 3. 闭环仿真引擎

### 3.1 仿真环境

基于 `highway-env` 构建闭环高速公路仿真。道路配置：
- 跟车：单车道，限速 $50\ \text{m/s}$
- 切入：双车道，限速 $50\ \text{m/s}$

仿真状态更新频率 $f = 25\ \text{Hz}$（$\Delta t = 0.04\ \text{s}$）。

### 3.2 车辆动力学模型

支持三种动力学模型：

**纵向模型（longitudinal）**：

```math
\begin{aligned}
a_{x}^{\text{applied}} &= \text{clip}\left(a_{x}^{\text{cmd}}, a_{x,\min}, a_{x,\max}\right) \\
v_{x,t+1} &= \max\left(v_{x,t} + a_{x}^{\text{applied}} \cdot \Delta t, 0\right) \\
x_{t+1} &= x_t + v_{x,t} \cdot \Delta t + \frac{1}{2} a_{x}^{\text{applied}} \cdot \Delta t^2
\end{aligned}
```

**运动学自行车模型（kinematic_bicycle）**：

```math
\begin{aligned}
v_{t+1} &= v_t + a_x^{\text{applied}} \cdot \Delta t \\
\psi_{t+1} &= \psi_t + \frac{v_t \cdot \tan(\delta_t)}{L} \cdot \Delta t \\
x_{t+1} &= x_t + v_t \cdot \cos(\psi_t) \cdot \Delta t \\
y_{t+1} &= y_t + v_t \cdot \sin(\psi_t) \cdot \Delta t
\end{aligned}
```

其中 $L = 5.0\ \text{m}$ 为轴距，$\delta_t$ 为转向角。

**质点模型（point_mass）**（切入场景）：

```math
\begin{aligned}
\mathbf{a}_t^{\text{applied}} &= \text{clip}\left(\mathbf{a}_t^{\text{cmd}}, \mathbf{a}_{\min}, \mathbf{a}_{\max}\right) \\
\mathbf{v}_{t+1} &= \mathbf{v}_t + \mathbf{a}_t^{\text{applied}} \cdot \Delta t \\
\mathbf{x}_{t+1} &= \mathbf{x}_t + \mathbf{v}_t \cdot \Delta t + \frac{1}{2} \mathbf{a}_t^{\text{applied}} \cdot \Delta t^2
\end{aligned}
```

### 3.3 物理可行性约束

施加以下物理约束，并累积物理惩罚（约束违反量的平方和）：

**加加速度约束**：

```math
|a_{x,t} - a_{x,t-1}| \leq j_{\max} \cdot \Delta t, \quad j_{\max} = 12.0\ \text{m/s}^3
```

**加速度边界**（速度相关）：

```math
a_{x,\max}(v) = \min\left(a_{x,\max}, \frac{v_{\max} - v_t}{\Delta t}\right), \quad a_{x,\min}(v) = \max\left(a_{x,\min}, \frac{-v_t}{\Delta t}\right)
```

**横向约束**（切入场景）：

```math
|a_y| \leq 4.0\ \text{m/s}^2, \quad |j_y| \leq 8.0\ \text{m/s}^3, \quad |\dot{\delta}| \leq 1.0\ \text{rad/s}
```

总物理惩罚：

```math
\Phi = \frac{1}{T}\sum_{t} \left( \phi_{a_x,t}^2 + \phi_{j,t}^2 + \phi_{a_y,t}^2 + \phi_{j_y,t}^2 + \phi_{\dot{\delta},t}^2 + \phi_{v,t}^2 \right)
```

仿真被视为物理可行当 $\Phi \leq 10^{-8}$ 且无违规计数。

---

## 4. 闭环风险评分

### 4.1 跟车场景风险

从闭环轨迹序列计算交互指标并与 highD 尾部 EVT 模型校准：

**碰撞时间（TTC）**：

```math
\text{TTC}_t = \frac{g_t}{\max(v_{\text{ego},t} - v_{\text{lead},t}, \varepsilon)}, \quad \varepsilon = 10^{-6}
```

**时距（THW）**：

```math
\text{THW}_t = \frac{g_t}{\max(v_{\text{ego},t}, \varepsilon)}
```

**碰撞临界指标**：$\text{TTC}_{\min} < 1.5\ \text{s}$ 标记为近碰撞（near-collision），$\text{TTC}_{\min} < 0$ 标记为碰撞。

**DRAC 聚合**（softmax 池化）：

```math
\text{DRAC}_t = \frac{(v_{\text{ego},t} - v_{\text{lead},t})_+^2}{2 g_t + \varepsilon}, \quad Y_{\text{raw}} = \frac{\sum_t w_t \cdot \text{DRAC}_t \cdot e^{\beta \cdot \text{DRAC}_t}}{\sum_t e^{\beta \cdot \text{DRAC}_t}}
```

其中 $\beta = 8.0$ 为池化温度。

**奖惩项**：

```math
Y = Y_{\text{raw}} + \alpha_{\text{collision}} \cdot \mathbf{1}[\text{collision}] + \alpha_{\text{near}} \cdot \mathbf{1}[\text{near\_collision}] + \alpha_{\text{brake}} \cdot \mathbf{1}[\text{hard\_brake}]
```

其中 $\alpha_{\text{collision}} = 5.0$，$\alpha_{\text{near}} = 1.0$，$\alpha_{\text{brake}} = 1.0$。

**EVT 评分映射**：

```math
S_{\text{EVT}}(y) = -\log H(y - u; \hat{\xi}, \hat{\sigma})
```

其中 $H(\cdot)$ 为 GPD 生存函数。

### 4.2 切入场景风险

切入风险评分在纵向风险基础上引入横向维度：

```math
Y_{\text{cutin}} = f(Y_{\text{long}}, \Delta y_t, \text{LTG}_t, v_{\text{app},t}, \Delta d_{\text{safe},t})
```

其中成分包括：
- **横向重叠**：当 $|\Delta y_t| < d_{\text{threshold}}$ 时认为存在横向侵入
- **横向时距（LTG）**：$\text{LTG}_t = |\Delta y_t| / |v_{y,\text{target},t}|$
- **安全距离亏空**：$\Delta d_{\text{safe},t} = g_t - \left(v_{\text{ego},t} \cdot t_r + \frac{v_{\text{ego},t}^2}{2 a_{\text{decel}}}\right)$，其中反应时间 $t_r = 0.2\ \text{s}$，舒适减速度 $a_{\text{decel}} = 6.0\ \text{m/s}^2$
- **切入后 DRAC**：切入完成后的窗口内最大 DRAC

合成风险通过切入 EVT 模型映射为最终得分 $S_{\text{EVT}}(Y_{\text{cutin}})$。

### 4.3 安全关键阈值

默认失效阈值为：
- 跟车：$y_{\text{crit}} = S_{\text{EVT}}(5.0)$，对应于 $Y_{\text{long}} = 5.0$ 的工程临界水平
- 切入：$y_{\text{crit}} = S_{\text{EVT}}(10.0)$，对应于 $Y_{\text{cutin}} = 10.0$ 的工程临界水平

也可配置为基于重现期的阈值（从 EVT 模型的返回水平计算）。

---

## 5. 冻结扩散采样器

### 5.1 确定性动作解码

`FrozenDiffusionSampler` 将预训练的扩散模型封装为评估模式下的确定性解码器。从潜在变量 $\mathbf{z}$ 解码动作：

```math
\mathbf{a} = \text{DDIM\_deterministic}(\mathbf{c}, \mathbf{z}; \theta, \tau)
```

其中 $\tau = 50$ 为推理去噪步数（从完整 $K=100$ 步中均匀子采样）。

关键性质：模型参数冻结（无梯度计算），采样为纯确定性函数：

```math
\frac{\partial \mathbf{a}}{\partial \mathbf{z}} = 0 \quad \text{（在子集模拟上下文中不通过扩散模型反向传播）}
```

### 5.2 批量采样与广播机制

支持背景张量的广播：当多个潜在变量对应同一背景时（常见于同层内的多样化提议），背景张量沿第 0 维复制以匹配潜在变量批量大小。

---

## 6. 不确定性量化

### 6.1 二项比例标准误

最终层失效分数的标准误按二项比例计算（忽略 MCMC 相关性）：

```math
\text{SE}(\hat{p}_m) = \sqrt{\frac{\hat{p}_m (1 - \hat{p}_m)}{N}}
```

失效概率的对数尺度 95% 置信区间：

```math
\log \hat{P}_f \pm 1.96 \cdot \frac{\text{SE}(\hat{p}_m)}{\hat{p}_m}
```

### 6.2 可靠性评估

子集模拟结果的可靠性基于以下指标分级：
- **通过（PASS）**：$N_{\text{unique}}^{\text{context}} \geq N_{\text{crit}}$，接受率 $\geq 0.10$，最终层 $\hat{p}_m > 0.02$
- **警告（WARN）**：部分指标接近阈值
- **失败（FAIL）**：关键指标严重低于阈值

---

## 7. 里程重现期分析

### 7.1 事件强度

将子集模拟概率与暴露分析中的尾部峰值率结合，计算安全关键事件的强度：

```math
\lambda_{\text{crit}} = \lambda_{\text{tail}} \cdot \hat{P}_f
```

其中 $\lambda_{\text{tail}}$ 为每英里（或每小时）的独立尾部峰值率。

### 7.2 重现期

```math
T_{\text{miles}} = \frac{1}{\lambda_{\text{crit}}}, \quad T_{\text{hours}} = \frac{1}{\lambda_{\text{crit}} \cdot \bar{v}}
```

其中 $\bar{v}$ 为平均行驶速度。

### 7.3 人类驾驶员基线对比

与纯 highD 人类驾驶员的尾部峰值重现期对比：

```math
\text{Ratio} = \frac{T_{\text{human}}}{T_{\text{ADS}}}
```

### 7.4 解释有效性约束

里程重现期报告需满足以下条件以保障解释有效性：
- 子集模拟可靠性评估为通过
- 背景集中仅包含独立尾部峰值（不含合成背景）
- EVT 模型阈值与背景尾部阈值匹配

---

## 8. 最终层回放与可视化

### 8.1 高分案例抽取

从子集模拟最终层选取得分最高的 $k$ 个案例（默认 $k = 5$）。可选过滤为唯一背景（每个背景仅展示最高分案例）。使用预存的动作序列精确回放仿真。

### 8.2 概述图

对每个案例生成三面板 PNG：
1. **位置面板**：本车和前车的 $x$ 位置随时间变化
2. **车间距面板**：净间距 $g_t$ 随时间的演化
3. **TTC/加速度面板**：TTC 和加速度的时间序列

### 8.3 动画渲染

生成 GIF 动画，包含：
- 带有车道线的道路
- 本车（红）和前车（蓝）的旋转矩形表示
- 两车的轨迹尾迹
- 逐帧 TTC 标注
- 碰撞时刻的 "COLLISION" 标注
- 相机跟踪两车中点

配置参数包括视野宽度（120 m）、尾迹长度（50 帧）和播放速度（1×）。

---

## 9. 输出与数据产品

子集模拟执行后生成以下输出：

| 文件 | 内容 |
|------|------|
| `latent_subset_samples.npz` | 所有层的 $(\mathbf{c}, \mathbf{z})$、得分、动作、指标和轨迹 |
| `latent_subset_level_stats.csv` | 每层统计（最小/平均/最大得分、失效分数、接受率、唯一性） |
| `latent_subset_summary.json` | 失效概率、不确定性、可靠性、重现期分析的汇总 |
| `latent_subset_top_cases.json` | 最高分的 $k$ 个案例的元数据 |
| `figures/subset_score_histograms.png` | 各层得分分布的直方图 |
