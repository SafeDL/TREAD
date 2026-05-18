# adversaray

`adversaray/` 是本项目中面向 car-following 场景的闭环对抗生成模块。默认主线已经切换为 **Prior-Regularized Guided Diffusion in highway-env**：

- 冻结 `diffusion/` Stage 1 highD 自然驾驶 diffusion prior
- 学习一个小的 guidance residual policy `g_phi(x_t, t, c)`
- 在 DDPM reverse step 中执行 `mu_guided = mu_theta + posterior_var * g_phi`
- 用真实 highway-env 车辆动力学和 IDM surrogate ego 构造直路 car-following 闭环 reward
- 用 prior KL / guidance norm 和物理约束保持自然性

## 目录结构

```text
adversaray/
  scripts/
    configs/
      prior_guided_following.yaml
    train_prior_guided_policy.py
    sample_prior_guided_diffusion.py
    evaluate_prior_guided_policy.py
  src/
    __init__.py
    diffusion_adapter.py
    normalization_adapter.py
    guidance_policy.py
    prior_guided_sampler.py
    closed_loop_runner.py
    prior_guided_train.py
    guidance_losses.py
    rss.py
    torch_kinematics.py
```

## 默认方法

Stage 1 diffusion prior 仍然由 `diffusion/` 训练和保存。本目录通过 `DiffusionPriorAdapter` 加载 checkpoint，并冻结全部 prior 参数。可学习部分只有 `GuidancePolicy`。

一次闭环训练 episode：

```text
dataset context
→ PriorGuidedDiffusionSampler 生成 lead jerk/acceleration plan
→ ClosedLoopFollowingRunner 在手工构造的 highway-env 直路 car-following road 中执行
→ IDM surrogate ego 响应 lead 动作
→ 记录 collision / min TTC / min gap / RSS margin / hard braking / physics / naturalness
→ REINFORCE 更新 guidance policy
```

默认训练使用完整 DDPM 100-step reverse chain，并执行完整 50-step plan。这样 `trajectory_log_prob`、`prior_kl` 和闭环 reward 的信用分配是一致的。评估仍可通过 `--commit-steps 1` 使用 rolling MPC 式重规划。runner 同时记录 `prior_kl_sum`、`prior_kl_per_plan` 和 `prior_kl_per_step`；训练损失和 naturalness gate 默认使用 `prior_kl_per_plan`，避免不同 commit 策略下 KL 惩罚不可比。

`adversaray` 不再提供内部 fallback simulator。若 `HighwayEnv/highway_env` 不能导入，训练和评估会直接报错；这是为了保证实验确实使用 highway-env vehicle dynamics，而不是退化成简化替代实现。

自然性来自 frozen prior 的 transition KL：

```text
KL(q_phi || p_theta) ~= 0.5 * posterior_var * ||g_phi||^2
```

## 常用命令

项目环境中默认 `python` 可能没有 PyTorch。建议使用已有的 `jzm` 环境：

```bash
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/train_prior_guided_policy.py
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/sample_prior_guided_diffusion.py
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/evaluate_prior_guided_policy.py
```

同一批 validation contexts 上比较 frozen prior 和 guided policy：

```bash
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/evaluate_prior_guided_policy.py \
  --compare-frozen-prior
```

只评估 frozen diffusion prior：

```bash
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/evaluate_prior_guided_policy.py \
  --disable-guidance
```

快速 smoke test：

```bash
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/train_prior_guided_policy.py \
  --epochs 1 \
  --max-train-contexts 2
```

训练输出默认写入：

```text
data/adversaray/following/prior_guided/
  checkpoints/best_reward.pt
  checkpoints/last.pt
  training_history.csv
  training_summary.json
```

## 关键模块

- `guidance_policy.py`：小型 residual guidance network，用 GRU 编码 history / relative history，用 MLP 编码 context features，输出 `g_phi`。
- `prior_guided_sampler.py`：learnable guidance DDPM sampler，记录 `trajectory_log_prob`、`prior_kl` 和 `guidance_norm`。
- `closed_loop_runner.py`：highway-env vehicle dynamics + IDM surrogate ego 的手工直路跟驰 runner；可选按 highD event metadata 重建初始 history。
- `prior_guided_train.py`：黑盒 REINFORCE 训练循环，优化闭环风险 reward 与 prior KL。
- `guidance_losses.py` / `rss.py` / `torch_kinematics.py`：物理、RSS 和纵向运动学工具。

`sample_prior_guided_diffusion.py` 只做 open-loop action sampling，适合检查生成动作分布和 KL/guidance 规模；最终风险指标以 `evaluate_prior_guided_policy.py` 的 closed-loop rollout 为准。

## 配置

默认配置：

```text
adversaray/scripts/configs/prior_guided_following.yaml
```

主要字段：

```text
paths      Stage 1 natural dataset、diffusion checkpoint、policy output/checkpoint
policy     guidance network 大小、active timestep window、guidance norm clip
env        highway-env rollout horizon、IDM ego、commit_steps_max
reward     collision/TTC/gap/RSS/hard-brake/physics 权重和 RSS 尺度截断
sampling   train/eval diffusion inference steps
training   REINFORCE 训练参数、EMA baseline、评估频率
```

训练 contexts 默认使用 `training.context_sampling: stratified`，会按可用的 `recording_id` / `event_id` / initial gap / closing speed 做轻量分层抽样；metadata 不足时退回 seeded random sampling，不再取 train split 的前缀样本。

局部坐标转换与 Stage 1 保持一致：每次闭环重规划都会用当前 ego 末帧构造 ego-current frame，再调用 `compute_ego_frame()` / `world_to_ego_states()` 重建 `context_states`。highD 预处理已把原始 bbox top-left 转为车辆中心；这里把 `tracksMeta.width` 明确作为纵向车辆长度使用，`height` 是横向宽度，因此 dataset 字段 `ego_length` / `adv_length` 继续来自 `width`。

注意：DDIM / subsampled DDPM transition 尚未实现，训练阶段不要把 `train_diffusion_steps` 改成小于 Stage 1 prior diffusion steps。代码默认会阻止这种配置，避免把不一致的跳步采样当作原始 diffusion prior。

如果 `context_features` 维度或语义发生变化，必须确保以下文件来自同一版 Stage 1 数据构建和训练：

```text
dataset.npz
dataset_normalized.npz
feature_schema.json
normalization_stats.json
checkpoints/best_noise_mse.pt
```

否则需要先重建并重训 natural diffusion prior，再训练 `adversaray/` policy。

评估 summary 会同时报告风险指标和自然性指标，包括 collision / near-collision、TTC/gap/RSS p05、hard-brake rate、acceleration / jerk / speed 分布、action clip rate、jerk violation rate、speed negative rate、prior KL 和 guidance norm。若 `dataset.npz` 包含 `future_states`，还会附带 recorded highD future 的真实 gap/TTC/lead dynamics 对照，以及 frozen prior / guided rollout 相对真实 highD 的 Wasserstein / KS 分布距离。

默认 reward 使用较温和的 collision bonus，并启用 naturalness gate。训练日志会拆分 `collision_reward`、`ttc_reward`、`gap_reward`、`rss_reward`、`hard_brake_reward`、`physics_penalty_reward`、`risk_reward` 和 `prior_kl_penalty`，方便判断 policy 是否只是在追逐碰撞奖励。

`training.lambda_prior` 默认是 `0.1`，建议做 Pareto sweep：

```text
0.05, 0.1, 0.2
```
