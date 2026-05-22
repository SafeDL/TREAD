# adversaray

`adversaray/` 是 car-following 场景的 KING-guided adversarial generation 模块。当前主线是：

```text
frozen highD diffusion prior
→ sample natural lead jerk plan j0
→ KING-style closed-loop optimization with highway-env IDM ego response
→ highway-env closed-loop validation
```

KING guidance 优化 adversary 控制序列时直接 replay 当前 adversary 轨迹，
并调用 highway-env `IDMVehicle` 得到 detached ego trace。风险仍用固定尺度
归一化后的 RSS / TTC / DRAC / gap 分量加权求和，但梯度只回到 adversary
控制序列，ego 响应不反传。

## Key Commands

项目默认 `python` 可能没有 PyTorch，建议使用 `tread` 环境：

```bash
conda run -n tread python adversaray/scripts/prepare_king_guided_contexts.py
conda run -n tread python adversaray/scripts/sample_king_guided_diffusion.py
conda run -n tread python adversaray/scripts/evaluate_king_guided_samples.py
conda run -n tread python adversaray/scripts/replay_king_guided_rollout.py
```

前三条是默认 KING 主流程：按危险得分选取 highD 自然驾驶尾部 contexts，
采样 frozen-prior/KING plans，再把 saved `prior_actions` 与 `king_actions`
重新放入 highway-env fixed-plan rollouts 中评估。脚本参数固定在各自的
`SCRIPT_DEFAULTS` 和 YAML 中，不通过 CLI 覆盖。`replay_king_guided_rollout.py`
是可选人工 replay。当前默认不运行去噪引导 workflow。

## Main Files

- `src/king_gradient_guidance.py`：KING-style risk objective 和 raw jerk plan optimizer。
- `src/frozen_diffusion_sampler.py`：frozen Stage 1 diffusion prior sampler。
- `src/context_utils.py`：context NPZ 读取、单条 context 包装和 batch observation 构建。
- `scripts/prepare_king_guided_contexts.py`：计算危险得分并直接导出 highD 自然驾驶尾部 contexts，不再生成 EVT synthetic contexts。
- `scripts/sample_king_guided_diffusion.py`：采样 frozen prior plan 并保存 KING-guided plan dataset。
- `scripts/evaluate_king_guided_samples.py`：读取 saved samples，用 highway-env 闭环评估 prior plan 与 KING plan，并保存 closed-loop 图。
- `scripts/replay_king_guided_rollout.py`：可选 highway-env graphics replay，用于人工查看单个 prior/KING rollout。
- `src/adversary_dynamics.py` / `src/rss.py`：可微 adversary 动力学、highway-env IDM ego trace 接入和 RSS 目标。
- `src/trajectory_constraints.py`：KING 自适应约束优化使用的自然性指标。
- `src/closed_loop_runner.py`：highway-env car-following validator。

默认 RSS 参数固定在 config/code 中：`response_time=0.458`、`ego_max_accel=2.389`、`ego_min_brake=2.136`、`lead_max_brake=7.625`。YAML 中制动参数写正数幅值，因为 `rss_safe_distance()` 已按正制动幅值计算分母。旧的 highD RSS calibration 入口和外部 RSS 覆盖路径已经移除。

默认 highway-env IDM 参数位于 `scripts/configs/king_guided_following.yaml`
的 `idm` block；`desired_speed` 跟当前 `env.ego_target_speed=30.0` 对齐。

默认 KING risk 尺度位于 `king_gradient` block：`rss_scale=100.0`、
`ttc_scale=1.0`、`drac_scale=5.0`、`gap_scale=20.0`。
`sample_king_guided_diffusion.py` 日志中的 `risk before -> after` 使用
highway-env IDM ego trace；`evaluate_king_guided_samples.py` 日志中的
`closed-loop risk` 是重新 rollout 后由 collision / gap / TTC / RSS /
physics 指标组成的验证风险。二者都不是碰撞概率。

默认 KING optimizer 使用固定范数的梯度方向步进：`num_steps=50`、`step_size=2.0`、`grad_clip_norm=1.0`、`lambda_nat=0.0`、`lambda_phys=0.2`。这是 adversarial-test 设置，主要依靠 jerk/acceleration/physics bounds 控制动作可行性。
