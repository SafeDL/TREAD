# EVT 条件化动作扩散模型

本目录是新的 diffusion 研究方向，和旧的 DeepEVT 条件分位数回归代码分开。这里不再让模型预测条件风险分位数，而是让扩散模型学习自然驾驶动作先验，并在训练/推理时使用风险强度作为条件。

当前第一阶段用于 **car-following**，目标是训练一个可靠的 highD 自然驾驶动作扩散模型：

- 事件类型：`following`
- 对抗车辆：前车 / lead car
- 输入：ego 与 lead car 的短历史交互状态 `o_t`，relative-history stream，以及风险条件 `r_t`
- 输出：lead car 未来短时域动作序列，默认是 jerk `[jx(t+1), ..., jx(t+H)]`
- 风险标签：默认用当前 ego 状态的 constant-velocity rollout 与真实 lead future 计算，不再依赖 highD human ego future
- 训练目标：DDPM noise MSE + `x0` reconstruction L1 + smoothness auxiliary loss

第一阶段暂不接入 EVT return level、`z_m`、RoadRunner/MATLAB 闭环 ADS 测试。模型先学自然、连续、可执行且风险可控的短时域 lead-car 动作。

## 关键实现

- 动作标签不再直接读取 highD `xAcceleration`；默认从平滑后的 lead velocity 差分得到 acceleration，再转换为 jerk。
- 数据集中保存 `risk_raw`、`risk_log`、`risk_percentile` 和归一化后的 `risk_condition`。
- 验证/测试窗口默认使用更稀疏 stride，训练窗口按事件内风险分层保留。
- 模型在每层 Transformer block 使用 FiLM 条件调制，并加入 `risk_dropout_prob` 支持 classifier-free guidance。
- 训练 loader 默认启用 risk-stratified sampling，提升高风险条件曝光频率。

## 配置文件说明

`diffusion_following.yaml` 是 car-following 的正式训练配置，用于完整数据集和正式训练。

两个入口脚本顶部都有：

```python
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "diffusion_following.yaml"
```

默认运行时会读取这一路径。你可以直接在代码里把它改成自己的配置文件，例如：

```python
DEFAULT_CONFIG_PATH = Path("/your/config/path.yaml")
```

命令行里的 `--config` 仍然保留，只作为临时覆盖使用。

## 环境

base conda 环境当前没有 PyTorch。沙盒外使用：

```bash
conda activate jzm
```

该环境具备 CUDA/PyTorch 条件，可用于 diffusion 训练。

如果在沙盒内直接运行，也可以显式调用：

```bash
/home/hp/anaconda3/envs/jzm/bin/python
```

## 构建 Car-Following 数据集

正式数据构建：

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/build_action_dataset.py
```

## 训练 Car-Following Diffusion

正式训练：

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/train_action_diffusion.py
```

训练脚本默认复用已有 `dataset_normalized.npz`。只有数据 schema、动作表示、risk 标签或窗口策略变化时，才需要重建：

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/train_action_diffusion.py --rebuild-dataset
```

也可以先单独构建数据集，再正常训练；两者不需要重复执行。

## 离线评价

训练后运行：

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/evaluate_action_diffusion.py
```

评价结果写入：

```text
data/diffusion/following/evaluation_summary.json
data/diffusion/following/evaluation_plots/action_distribution.png
data/diffusion/following/evaluation_plots/trajectory_profiles.png
data/diffusion/following/evaluation_plots/risk_control.png
```

主要检查：

- 真实/生成 `ax` 和 jerk 分布；
- hard-braking ratio、采样后动作 clip rate；
- 动作积分后的 gap/risk；
- 固定同一 `o_t` 时，p50/p80/p90/p95 风险条件下生成风险是否单调升高；
- zero/shuffled risk condition ablation。

## TensorBoard

当前可视化训练和验证的核心指标：

- `loss/*_loss`
- `loss/*_noise_mse`
- `loss/*_x0_l1`
- `loss/*_smooth`

暂时不写入大而杂的指标面板，避免 TensorBoard 被无关信息淹没。
