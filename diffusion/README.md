# diffusion：anchor-frame 自然动作扩散先验

`diffusion/` 训练 highD following 和 cut-in 两条独立自然动作先验。模型只输入
anchor-frame `scenario_conditions`，不再输入 rolling history、
`context_features` 或 `relative_history`。

```text
following: p(j_lead_0:T | c_cf_0), T = 125 steps at 25 Hz
cut-in:    p(ax_target_0:T, ay_target_0:T | c_cutin_0), T = 100 steps at 25 Hz
```

## 运行顺序

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
target_vy_0, target_ay_0,
final_lateral_offset, time_to_cross, target_speed_change,
target_slope_at_cross
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
