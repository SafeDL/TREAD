# 自然驾驶动作扩散先验

`diffusion/` 负责训练和评估 highD 跟驰场景下的自然驾驶动作先验。
它只学习“给定最近 ego-lead 交互历史，前车未来动作在自然数据中
应该是什么样”。它不负责对抗攻击、风险引导或闭环 ADS 测试。

当前默认口径是：

```text
训练: DDPM 噪声预测模型
测试/采样: DDIM deterministic reverse process
```

也就是说，DDIM 只改变推理阶段的反向采样过程，不改变训练损失、
checkpoint 格式或数据集格式。

## 功能范围

- 场景类型：`following`
- 输入：ego/lead 历史状态、相对历史特征、当前上下文特征
- 输出：前车未来动作序列，当前默认是 jerk
- 训练目标：DDPM noise MSE，加可选 `x0` 重建和平滑辅助损失
- 默认推理：DDIM deterministic sampler

风险筛选、EVT 标定和 highway-env 闭环概率估计属于 `process_highD/` 或
`subset/`，不放在自然先验训练路径中。
如果评估脚本需要读取 NPZ 或进行归一化适配，应优先使用根目录 `utils/`
中的公共工具，避免在 diffusion 内重复实现跨模块逻辑。

## 关键文件

```text
diffusion/
  scripts/
    configs/natural_following.yaml
    train_natural_diffusion.py
    evaluate_natural_prior.py
    sample_natural_rollouts.py
  src/
    data.py
    features.py
    normalization.py
    types.py
    model.py
    train.py
    kinematics.py
    utils.py
```

## 环境

本工程默认使用：

```bash
conda activate tread
```

也可以显式调用：

```bash
conda run -n tread python <script>
```

## 推荐运行顺序

### 1. 构建自然数据集

```bash
conda run -n tread python process_highD/scripts/build_natural_dataset.py
```

默认输出目录：

```text
results/diffusion_natural/following/
```

主要数组：

```text
context_states
future_states
context_features
relative_history
actions
split_index
recording_id
event_id
anchor_frame
ego_length
adv_length
```

其中 `future_states` 的形状是 `[N, horizon_steps, 2, 6]`。
actor `0` 是 highD 中真实 ego future，actor `1` 是真实 lead future。

### 2. 训练 DDPM 噪声预测模型

```bash
conda run -n tread python diffusion/scripts/train_natural_diffusion.py
```

如果需要强制重新构建数据集后再训练，修改
`diffusion/scripts/train_natural_diffusion.py` 顶部的 `SCRIPT_DEFAULTS["rebuild_dataset"]`。

训练阶段仍是标准 DDPM 训练：

```text
x_0 = natural highD action
epsilon ~ N(0, I)
x_t = sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) epsilon
model predicts epsilon
loss = noise MSE + optional x0/smooth auxiliary losses
```

DDIM 不改变这一步，因为 DDIM 是推理采样算法，不是训练目标。

训练输出：

```text
checkpoints/best.pt
checkpoints/best_noise_mse.pt
checkpoints/last.pt
training_history.csv
training_history.json
training_summary.json
```

### 3. 使用 DDIM 评估自然先验

```bash
conda run -n tread python diffusion/scripts/evaluate_natural_prior.py
```

评估脚本会用 DDIM deterministic sampler 生成动作，然后和 highD
真实动作、轨迹和交互指标做统计对比。

输出：

```text
naturalness_summary.json
naturalness_metrics.csv
diversity_summary.json
natural_prior_plots/ax_distribution_real_vs_generated.png
natural_prior_plots/jerk_distribution_real_vs_generated.png
natural_prior_plots/speed_distribution_real_vs_generated.png
natural_prior_plots/example_rollouts.png
```

评估内容包括：

- 配置 split 的 denoising/reconstruction/smoothness 指标，默认是 `test`
- acceleration 和 jerk 分布统计
- Wasserstein、KS、histogram L1 距离
- action clip、speed、jerk、acceleration 和轨迹跳变违规率
- lead speed、final speed、displacement 与 highD 真实 future 的对比
- 使用真实 highD ego future 计算交互自然性：gap、TTC、THW 等
- 多样性指标：同一 context 多次采样的动作和轨迹方差

### 4. 使用 DDIM 生成自然 rollout 样本

```bash
conda run -n tread python diffusion/scripts/sample_natural_rollouts.py
```

输出：

```text
natural_rollouts.npz
natural_rollouts_summary.json
```

`natural_rollouts_summary.json` 中会记录：

```text
sampler = "ddim"
```

生成内容包括：

```text
sample_index
actions
acceleration
lead_trajectory
```

## DDIM 推理

训练得到的是噪声预测网络：

```text
epsilon_theta(x_t, t, context)
```

DDIM deterministic 采样不额外注入 reverse noise，当前实现固定使用确定性反向过程：

```text
x_{t-1}
= sqrt(alpha_{t-1}) * x0_hat
  + sqrt(1 - alpha_{t-1}) * epsilon_theta
```

给定同一个 context 和初始噪声 `x_T`，每次都会得到同一条轨迹。代码不支持
在反向步骤中重新注入噪声。

因此在 DDIM 下：

```text
同一个初始 latent z = x_T -> 同一条动作轨迹
不同 latent z -> 不同自然动作轨迹
```

这正是 `subset/` 中 latent-space subset simulation 需要的确定性映射。

## 与统一风险评分的关系

`diffusion/` 不训练安全得分，也不输出用于优化的危险标签。自然先验评估中如需报告
gap、TTC、THW 等交互统计，只作为自然性诊断；闭环危险评分由 `subset/`
通过 `utils/risk.py` 统一计算。

项目不把 RSS margin 放入 EVT、tail context 筛选或 subset score。RSS 基础函数
仍保留在 `utils/rss.py`，仅作为显式 RSS 诊断或历史实验工具。

## 与 subset simulation 的关系

`subset/` 直接把随机变量定义为 diffusion latent：

```text
z ~ N(0, I)
actions = DDIM(context, z)
score = S_EVT(Y_long_sim)
```

所以 `subset/scripts/run_latent_subset_simulation.py` 会自动受益于这里的 DDIM
deterministic sampler。推荐顺序是：

```bash
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python diffusion/scripts/train_natural_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_natural_prior.py
conda run -n tread python process_highD/scripts/fit_longitudinal_evt.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py
conda run -n tread python subset/scripts/run_latent_subset_simulation.py
```

## 注意事项

- 测试和采样固定使用 DDIM deterministic reverse process。
- jerk 动作会先积分成 acceleration，再调用运动学积分。
