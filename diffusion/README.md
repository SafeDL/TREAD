# diffusion：anchor-frame 自然动作扩散先验

`diffusion/` 训练 highD following 和 cut-in 两条独立自然动作先验。模型只输入
anchor-frame `scenario_conditions`，不再输入 rolling history、
`context_features` 或 `relative_history`。

```text
following: p(j_lead_0:T | c_cf_0), T = 125 steps at 25 Hz
cut-in:    p(ax_target_0:T, ay_target_0:T | c_cutin_0), T = 100 steps at 25 Hz
```

## 运行顺序

同一个配置文件通过 `split.mode` 切换两种训练方式：

```text
train_val_test  按 recording 划分 train/val/test，用于模型选择和 held-out 评估
all_train       全部 recording 进入 train，用于最终生成 prior 和 subset simulation
```

训练脚本会把 `split.mode` 写入 checkpoint 文件名：

```text
checkpoints/best_noise_mse_train_val_test.pt
checkpoints/best_noise_mse_all_train.pt
```

推荐先使用默认 `train_val_test` 完成评估；确认模型质量后，将对应
`natural_*.yaml` 中 `split.mode` 改为 `all_train`，并在首次切换时设置
`dataset.rebuild: true` 重建归一化数据，再训练最终生成权重。`subset/` 默认读取
`best_noise_mse_all_train.pt`。

following：

```bash
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_following.yaml
conda run -n tread python diffusion/scripts/train_following_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_following_prior.py
```

cut-in：

```bash
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
conda run -n tread python diffusion/scripts/train_cutin_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_cutin_prior.py
```

## 输入与输出

`dataset.npz` 保存：

```text
scenario_conditions
initial_states
future_states
actions
split_index
recording_id, event_id, anchor_frame
ego_length, adv_length
```

`dataset_normalized.npz` 只保存训练必需字段：

```text
scenario_conditions
actions
split_index
```

following 条件向量：

```text
ego_vx_0, initial_gap, initial_delta_v, lead_ax_0,
lead_speed_change, lead_min_ax, lead_braking_duration
```

cut-in 条件向量：

```text
ego_vx_0, initial_gap, initial_lateral_offset, initial_delta_vx,
target_ax_0, target_vy_0, target_ay_0,
final_lateral_offset, time_to_cross, target_speed_change
```

动作表示保持不变：

```text
following: lead jx
cut-in:    target ax, ay
```

`feature_schema.json` 使用 `conditioning_mode: "anchor_scenario"` 和
`condition_keys`。旧的
history-conditioned dataset 和 checkpoint 不兼容，需要重建。

## 与 subset 的关系

DDIM deterministic sampler 保证：

```text
same scenario condition + same latent z -> same 125-step action trajectory
```

其中 following 的确定性轨迹长度为 125 步，cut-in 为 100 步。

因此 `subset/` 在 `(scenario_conditions, z)` 空间中做一次性 latent subset simulation，
不再进行 rolling reconditioning。

为避免长尾测试空间被 held-out split 人为缩小，`subset/` 默认使用全量训练得到的
扩散权重：

```text
results/diffusion_natural/following/checkpoints/best_noise_mse_all_train.pt
results/diffusion_natural/cutin/checkpoints/best_noise_mse_all_train.pt
```

## 工程组织

`diffusion/` 只保留模型、数据加载、训练和 prior 评估逻辑。跨模块 IO、归一化适配、
风险配置和论文图样式从 `tools/` 引入，例如 `tools/io.py`、`tools/diffusion_adapter.py`
和 `tools/plot_style.py`。旧根目录 `utils/` 已重命名为 `tools/`，不要再新增兼容 wrapper。
