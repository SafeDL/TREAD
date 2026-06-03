# diffusion：自然驾驶动作先验

`diffusion/` 训练 highD 场景下 adversarial vehicle 的自然动作先验。它只建模
“给定 ego-target 历史，未来 target 动作在自然数据中如何分布”，不负责风险筛选、
EVT 标定或闭环安全概率估计。

默认口径：

```text
训练: DDPM noise prediction
推理: DDIM deterministic sampling
following 动作: jerk
cut-in 动作: jerk + steering_rate
```

## 运行顺序

训练入口按事件类型拆分，不通过 CLI 切换配置。数据构建和评估仍显式传入对应配置。

following：

```bash
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_following.yaml
conda run -n tread python diffusion/scripts/train_following_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_natural_prior.py \
  --config diffusion/scripts/configs/natural_following.yaml
```

cut-in：

```bash
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
conda run -n tread python diffusion/scripts/train_cutin_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_natural_prior.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
```

cut-in 默认使用 25 帧历史，约 1 s at 25 Hz，用于覆盖切入前的纵向和横向运动趋势。
cut-in 评估除纵向 gap、TTC、speed、jerk 外，还输出 steering-rate、横向位移、横向
速度/加速度、yaw-rate、轨迹重构误差和横向 phase-space 对比；following 评估保留纵向
自然性与交互指标。

## 输入特征

两类模型都输入 `context_states`、`context_features`、`relative_history` 和未来动作。
`context_states` 是 ego 坐标系下的历史状态：

```text
[history_steps, ego/target, x, y, vx, vy, ax, ay]
```

following 使用 10 帧历史。`context_features` 保留 8 个纵向摘要：

```text
ego_vx_current, lead_vx_current, gap_current,
ego_ax_current, lead_ax_current, gap_change_rate,
min_gap_in_prefix, max_closing_speed_in_prefix
```

following 的 `relative_history` 只保留每帧：

```text
gap, delta_v
```

cut-in 使用 25 帧历史。`context_features` 保留 18 个纵向/横向摘要：

```text
ego_vx_current, target_vx_current, target_vy_current,
gap_current, lateral_offset_current,
ego_ax_current, target_ax_current, target_ay_current,
gap_change_rate, lateral_offset_change_rate,
relative_vx_trend, relative_vy_trend,
target_yaw_rate_current, target_lateral_jerk_current,
min_gap_in_prefix, min_abs_lateral_offset_in_prefix,
max_abs_target_lateral_velocity_in_prefix,
lateral_offset_range_in_prefix
```

cut-in 的 `relative_history` 保留每帧：

```text
gap, lateral_offset, delta_vx, delta_vy,
target_lateral_velocity, target_lateral_accel
```

TTC、THW、当前相对速度、当前相对加速度等可由上述字段或历史状态直接推导的字段不再
作为扩散模型输入。

可选生成样本：

```bash
conda run -n tread python diffusion/scripts/sample_natural_rollouts.py
```

训练过程会写入精简 TensorBoard 标量：

```bash
tensorboard --logdir results/diffusion_natural
```

记录项只包含 `loss`、`noise_mse`、`learning_rate` 和当前 best validation
`noise_mse`。

## 主要文件

```text
diffusion/scripts/configs/natural_following.yaml
diffusion/scripts/configs/natural_cutin.yaml
diffusion/scripts/train_following_diffusion.py
diffusion/scripts/train_cutin_diffusion.py
diffusion/scripts/evaluate_natural_prior.py
diffusion/scripts/sample_natural_rollouts.py
diffusion/src/data.py
diffusion/src/model.py
diffusion/src/train.py
diffusion/src/kinematics.py
```

## 主要输出

```text
results/diffusion_natural/following/dataset.npz
results/diffusion_natural/following/dataset_normalized.npz
results/diffusion_natural/following/feature_schema.json
results/diffusion_natural/following/normalization_stats.json
results/diffusion_natural/following/train_val_test_split.json
results/diffusion_natural/following/checkpoints/best_noise_mse.pt
results/diffusion_natural/following/training_summary.json
results/diffusion_natural/following/tensorboard/
results/diffusion_natural/following/naturalness_summary.json

results/diffusion_natural/cutin/
```

`dataset_normalized.npz` 只保存训练必需数组；`dataset.npz` 保存评估和样本回放需要的
原始 reference。`feature_schema.json`、`normalization_stats.json` 和
`best_noise_mse.pt` 是 subset simulation 的必要输入。

## 与 subset 的关系

DDIM deterministic sampler 保证：

```text
same context + same latent z -> same action trajectory
```

因此 `subset/` 可以直接把 diffusion latent 作为随机变量，执行 latent-space subset simulation：

```text
context ~ Uniform(tail-feature contexts)
z ~ N(0, I)
score = S_EVT(Y_sim)
```
