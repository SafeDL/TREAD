# adversaray

`adversaray/` 是本项目中面向 car-following 场景的对抗生成模块。它不重新训练自然先验，也不做 PPO/RL 策略训练，而是在已经训练好的 `diffusion/` highD 自然驾驶先验基础上，实现：

- 自然性判别器 `Dpsi`
- RSS + `Dpsi` + 物理约束联合引导的 diffusion reverse denoising
- 每帧闭环执行的 rolling guided diffusion actor

Stage 1 自然先验仍然属于 `diffusion/`。本目录通过 `src/diffusion_adapter.py` 加载冻结的 diffusion checkpoint，并调用其 denoiser、schedule buffer 和 `predict_start_from_noise()`，不修改 `diffusion/src/model.py` 的标准采样接口。

## 目录结构

```text
adversaray/
  README.md
  scripts/
    configs/
      discriminator_following.yaml
      guided_sampling_following.yaml
    build_discriminator_dataset.py
    train_discriminator.py
    evaluate_discriminator.py
    sample_rss_guided_diffusion.py
    evaluate_guided_samples.py
    run_rolling_guided_actor.py
  src/
    diffusion_adapter.py
    normalization_adapter.py
    future_features.py
    torch_kinematics.py
    rss.py
    discriminator.py
    discriminator_data.py
    negative_generators.py
    discriminator_train.py
    discriminator_eval.py
    guidance_schedule.py
    guidance_losses.py
    guided_sampler.py
    rolling_actor.py
    metrics.py
```

根目录不保留旧版 wrapper 脚本。所有可执行入口都在 `adversaray/scripts/` 中。

## 数据关系

`diffusion/` 已有的数据集用于训练自然先验 `pi_theta`，主要字段包括：

```text
context_states
context_features
relative_history
actions
```

`adversaray/` 需要额外构建判别器数据集，因为 `Dpsi` 是监督式二分类/软标签模型，需要正负样本和派生特征：

```text
future_actions
future_action_features
summary_features
labels
soft_labels
sample_weights
source_type
```

正样本来自 highD 真实 future lead actions，负样本来自随机扰动、规则急刹和 RSS 过度引导样本。也就是说，判别器数据集不是替代 highD 或 diffusion 数据，而是在现有 diffusion natural dataset 上派生出的 Stage 2 训练数据。

默认输出路径：

```text
data/adversaray/following/discriminator/
data/adversaray/following/guided_sampling/
```

## 常用命令

项目环境中默认 `python` 可能没有 PyTorch。建议使用已有的 `jzm` 环境：

```bash
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/build_discriminator_dataset.py
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/train_discriminator.py
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/evaluate_discriminator.py
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/sample_rss_guided_diffusion.py
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/evaluate_guided_samples.py
/home/hp/anaconda3/envs/jzm/bin/python adversaray/scripts/run_rolling_guided_actor.py
```

配置文件位于：

```text
adversaray/scripts/configs/discriminator_following.yaml
adversaray/scripts/configs/guided_sampling_following.yaml
```

## 模块说明

- `future_features.py`：从 future lead actions 构造 jerk、acceleration、velocity、displacement 和 gap proxy 特征。
- `torch_kinematics.py`：可微的纵向运动学积分，用于判别器特征和 guidance loss。
- `rss.py`：可微 RSS safe distance、actual gap、RSS margin 和 RSS criticality objective。
- `discriminator.py`：自然性判别器 `Dpsi(history, future) -> logit`。
- `discriminator_data.py`：从 diffusion natural dataset 派生 Stage 2 判别器数据集。
- `negative_generators.py`：生成 random perturb、rule brake、RSS-over-guided 负样本。
- `diffusion_adapter.py`：加载并冻结 Stage 1 diffusion prior。
- `guidance_losses.py`：negative speed、acceleration bound、jerk bound、trajectory discontinuity 等物理惩罚。
- `guided_sampler.py`：在 reverse denoising 中联合使用 RSS、`log Dpsi` 和物理约束引导。
- `rolling_actor.py`：类 MPC 的逐帧闭环 rolling actor。

## 方法边界

当前版本明确不做以下事情：

```text
不训练 PPO 或其他 RL policy
不把 highway-env 纳入第一版主流程
不把 TTC / DRAC / hard braking 作为 denoising 主目标
不把“生成多条候选再筛选”作为主方法
不做开环执行 K 帧
```

`rolling_actor.commit_steps_max` 的含义是：一个 diffusion plan 最多连续复用多少个仿真 tick。每个 tick 只执行当前一帧 lead action，ego/ADS 必须在同一个 tick 中响应；达到 `commit_steps_max` 或触发 replanning 后重新生成计划。

