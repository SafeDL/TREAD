# adversaray：KING-guided 对抗轨迹生成

`adversaray/` 是 car-following 场景的 KING-guided adversarial generation 模块。当前主线是：

```text
frozen highD diffusion prior
→ sample natural lead jerk plan j0
→ KING-style closed-loop optimization with highway-env IDM ego response
→ highway-env closed-loop validation
```

KING guidance 优化 adversary 控制序列时直接 replay 当前 adversary 轨迹，
并调用 highway-env `IDMVehicle` 得到 detached ego trace。风险评分统一由
`utils/risk.py` 提供，主要由 TTC、DRAC、gap、相对 prior 的 delta RSS 和
RSS improper response 组成。梯度只回到 adversary 控制序列，ego 响应不反传。

## 运行命令

项目默认 `python` 可能没有 PyTorch，建议使用 `tread` 环境：

```bash
conda run -n tread python process_highD/scripts/select_tail_contexts.py
conda run -n tread python adversaray/scripts/sample_king_guided_diffusion.py
conda run -n tread python adversaray/scripts/evaluate_king_guided_samples.py
```

三条命令是默认 KING 主流程：先读取共享长尾 highD contexts，
再采样 frozen-prior/KING plans，并把 saved `prior_actions` 与 `king_actions`
重新放入 highway-env fixed-plan rollouts 中评估。脚本参数固定在各自的
`SCRIPT_DEFAULTS` 和 YAML 中，不通过 CLI 覆盖。

## 主要文件

- `src/king_gradient_guidance.py`：KING-style risk objective 和 jerk plan optimizer。
- `src/frozen_diffusion_sampler.py`：frozen Stage 1 DDIM diffusion prior sampler。
- `scripts/sample_king_guided_diffusion.py`：采样 frozen prior plan 并保存 KING-guided plan dataset。
- `scripts/evaluate_king_guided_samples.py`：读取 saved samples，
  用 highway-env 闭环评估 prior plan 与 KING plan，并保存 closed-loop 图。
- `src/adversary_dynamics.py`：可微 adversary 动力学。
- `src/trajectory_constraints.py`：KING 自适应约束优化使用的自然性指标。
- `src/closed_loop_runner.py`：highway-env car-following validator。

`adversaray/` 不保留仅做转发的历史兼容 wrapper。context、RSS、归一化和
diffusion adapter 均直接从根目录 `utils/` 引入。

默认 RSS 参数固定在 config/code 中：`response_time=0.458`、
`ego_max_accel=2.389`、`ego_min_brake=2.136`、
`lead_max_brake=7.625`。优化和闭环评分不使用 RSS 绝对 margin，而使用
delta RSS 与 improper response。

默认 highway-env IDM 参数位于 `scripts/configs/king_guided_following.yaml`
的 `idm` block；`desired_speed` 跟当前 `env.ego_target_speed=30.0` 对齐。

默认 KING 优化权重和尺度位于 `risk_scoring` block，包含 delta RSS 与
improper response。闭环验证权重位于 `closed_loop_risk_scoring` block，
与 `subset/` 的闭环事件风险一致，不包含这两个 adversarial RSS 权重。
默认 prior 采样固定使用确定性 DDIM。
`sample_king_guided_diffusion.py` 日志中的 `risk before -> after` 使用
highway-env IDM ego trace；`evaluate_king_guided_samples.py` 日志中的
`closed-loop risk` 是重新 rollout 后由 collision、near collision、gap、TTC
和 DRAC 等指标组成的验证风险。二者都不是碰撞概率。

默认 KING optimizer 使用固定范数的梯度方向步进：
`num_steps=50`、`step_size=2.0`、`grad_clip_norm=1.0`。这是
adversarial-test 设置，主要依靠 jerk、acceleration 和 speed 的硬边界
投影控制动作可行性。
