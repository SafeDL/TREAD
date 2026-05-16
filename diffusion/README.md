# EVT 条件化动作扩散模型

本目录是新的 diffusion 研究方向，和旧的 DeepEVT 条件分位数回归代码分开。这里不再让模型预测条件风险分位数，而是让扩散模型学习自然驾驶动作先验，并在训练/推理时使用风险强度作为条件。

当前已经实现的第一版用于 **car-following**：

- 事件类型：`following`
- 对抗车辆：前车 / lead car
- 输入：ego 与 lead car 的短历史交互状态 `o_t`，以及未来窗口风险条件 `r_t`
- 输出：lead car 未来短时域纵向加速度序列 `[ax(t+1), ..., ax(t+H)]`
- 训练目标：DDPM 噪声预测 MSE，即在归一化动作序列上预测扩散噪声

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

## TensorBoard

当前只可视化最关键的训练指标：

- `loss/train_noise_mse`
- `loss/val_noise_mse`

暂时不写入大而杂的指标面板，避免 TensorBoard 被无关信息淹没。
