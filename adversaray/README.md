# adversaray

`adversaray/` 是本项目中面向 car-following 场景的闭环对抗生成模块。默认主线已经切换为 **Prior-Regularized Guided Diffusion in highway-env**：

- 冻结 `diffusion/` Stage 1 highD 自然驾驶 diffusion prior
- 学习一个小的 guidance residual policy `g_phi(x_t, t, c)`
- 在 DDPM reverse step 中执行 `mu_guided = mu_theta + posterior_var * g_phi`
- 用 highway-env car-following 闭环 rollout 给出风险 reward
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
→ ClosedLoopFollowingRunner 在 highway-env 直路 car-following 场景执行
→ IDM surrogate ego 响应 lead 动作
→ 记录 collision / min TTC / min gap / RSS margin / hard braking / physics
→ REINFORCE 更新 guidance policy
```

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

快速 smoke test：

```bash
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/train_prior_guided_policy.py \
  --epochs 1 \
  --max-train-contexts 2 \
  --episode-steps 2
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

- `guidance_policy.py`：小型 residual guidance network，输入 noisy action、timestep 和 history context，输出 `g_phi`。
- `prior_guided_sampler.py`：learnable guidance DDPM sampler，记录 `trajectory_log_prob`、`prior_kl` 和 `guidance_norm`。
- `closed_loop_runner.py`：highway-env car-following 闭环 runner，lead 执行生成动作，ego 使用 IDM surrogate。
- `prior_guided_train.py`：黑盒 REINFORCE 训练循环，优化闭环风险 reward 与 prior KL。
- `guidance_losses.py` / `rss.py` / `torch_kinematics.py`：物理、RSS 和纵向运动学工具。

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
reward     collision/TTC/gap/RSS/hard-brake/physics/prior 权重
training   REINFORCE 训练参数、EMA baseline、评估频率
```
