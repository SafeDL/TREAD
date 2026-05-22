# 目标文档：在现有 KING 主线上实现 Late-Step Risk-Tilted Diffusion

## 1. 总目标


当前 `adversaray` 已经形成一条可运行、可验证的 KING-guided 主线：

```text
highD 自然驾驶长尾 context
→ frozen highD diffusion prior 采样 prior_actions
→ KING post-denoising gradient refinement 生成 king_actions
→ highway-env closed-loop fixed-plan validation
→ 输出 closed_loop_risk / gap / TTC / RSS / collision / trajectory 图
```

下一阶段目标是在不破坏这条主线的基础上，实现并评估：

```text
Late-Step Risk-Tilted Diffusion Guidance
```

新的完整比较对象应为：

```text
1. frozen prior diffusion sample        → prior_actions
2. late-step risk-tilted diffusion      → tilted_actions
3. KING post-denoising refinement       → king_actions
```

重要限定：

- 不训练新网络。
- 不做 REINFORCE。
- 不训练 denoising controller。
- diffusion prior 保持 frozen。
- guidance 只在 DDPM reverse sampling 的后期步骤中作为 plug-and-play gradient mean shift 使用。
- 最终结论必须来自 highway-env closed-loop validation，不以 open-loop proxy risk 作为安全性结论。

---

## 2. 当前代码事实

### 2.1 已验证主流程

当前主流程命令：

```bash
conda run -n tread python adversaray/scripts/prepare_king_guided_contexts.py
conda run -n tread python adversaray/scripts/sample_king_guided_diffusion.py
conda run -n tread python adversaray/scripts/evaluate_king_guided_samples.py
```

可选诊断：

```bash
conda run -n tread python adversaray/scripts/visualize_king_guided_samples.py
conda run -n tread python adversaray/scripts/replay_king_guided_rollout.py
```

当前输出：

```text
data/adversaray/following/king_guided/
  tail_natural_contexts.npz
  king_guided_samples.npz
  king_guided_samples_summary.json
  king_guided_eval_summary.json
  figures/
    king_guided_closed_loop_histograms.png
    king_guided_closed_loop_case_XXXX.png
    king_guided_sample_histograms.png
```

### 2.2 Frozen diffusion sampler

文件：

```text
adversaray/src/frozen_diffusion_sampler.py
```

当前采样逻辑：

```python
eps = self.prior.predict_eps(x_t, t, context_states, context_features, relative_history)
x0_hat = self.prior.predict_x0(x_t, t, eps)
posterior_mean, _posterior_var, posterior_log_var = self.prior.posterior_mean_variance(x_t, t, x0_hat)
x_t = posterior_mean + diffusion_noise
```

当前默认路径在 `torch.no_grad()` 中运行。新增 risk-tilted guidance 后，必须保证：

```text
risk_tilted_diffusion.enabled: false
```

时默认采样行为不变。

只有 late guided steps 才允许 autograd 通过：

```text
x_t → eps → x0_hat → raw_actions_hat → differentiable rollout → KING proxy risk
```

### 2.3 已有可微风险与动力学

可复用文件：

```text
adversaray/src/king_gradient_guidance.py
adversaray/src/torch_kinematics.py
adversaray/src/physics_losses.py
adversaray/src/rss.py
```

应复用：

```python
compute_king_risk(kin, config)
integrate_following_actions_torch(...)
physical_violation_penalty(...)
```

这样 `tilted_actions` 和 `king_actions` 的 open-loop proxy risk 有相同定义，比较才有意义。

### 2.4 当前 closed-loop evaluator

文件：

```text
adversaray/scripts/evaluate_king_guided_samples.py
```

当前已经支持：

```text
prior_actions
king_actions
```

并行执行 highway-env fixed-plan rollouts，默认：

```python
SCRIPT_DEFAULTS["num_workers"] = 8
```

新增 risk-tilted 后，建议新建 evaluator，而不是强行把 KING evaluator 变复杂。

---

## 3. 方法定义：Late-Step Risk-Tilted Diffusion

目标分布：

```text
p_beta(τ | c) ∝ p_0(τ | c) · exp(beta · Q(τ, c))
```

其中：

```text
p_0(τ | c) = frozen natural diffusion prior
Q(τ, c)    = differentiable risk objective
beta       = guidance strength
```

实现近似：

```text
DDPM reverse step + late gradient mean shift
```

在 guided reverse step 中：

```python
x_in = x_t.detach().requires_grad_(True)

eps = prior.predict_eps(x_in, t, context_states, context_features, relative_history)
x0_hat = prior.predict_x0(x_in, t, eps)
raw_actions_hat = prior.decode_actions(x0_hat)

kin = integrate_following_actions_torch(
    raw_actions_hat,
    raw_context,
    ego_length,
    adv_length,
    prior.schema,
    config,
)
risk, risk_diag = compute_king_risk(kin, config)
physics, physics_diag = physical_violation_penalty(kin, physics_config)

action_l2 = raw_actions_hat.square().flatten(1).mean(dim=1)
objective = risk - lambda_phys * physics - lambda_action_l2 * action_l2

grad = torch.autograd.grad(objective.sum(), x_in)[0]
grad = normalize_or_clip(grad)

posterior_mean, _posterior_var, posterior_log_var = prior.posterior_mean_variance(x_in, t, x0_hat)
posterior_mean = posterior_mean + guidance_scale_t * grad.detach()

x_next = posterior_mean + diffusion_noise
```

非 guided step 必须保持当前 `no_grad` 路径。

---

## 4. 需要实现的代码改动

### 4.1 配置：新增 `risk_tilted_diffusion`

修改：

```text
adversaray/scripts/configs/king_guided_following.yaml
```

新增默认关闭配置：

```yaml
risk_tilted_diffusion:
  enabled: false

  late_fraction: 0.30
  num_late_steps: 0

  guidance_scale: 0.05
  scale_schedule: "linear_ramp"

  max_grad_norm: 1.0
  normalize_grad: true

  apply_at_t0: false

  lambda_phys: 0.2
  lambda_action_l2: 0.0

  min_grad_norm: 1.0e-12
  nan_to_num: true

  save_guidance_diagnostics: true
```

默认必须为 `enabled: false`，避免影响现有 KING workflow。

### 4.2 修改 `FrozenDiffusionSampler.sample(...)`

文件：

```text
adversaray/src/frozen_diffusion_sampler.py
```

新增参数：

```python
def sample(
    ...,
    risk_tilted: bool | None = None,
    risk_tilted_config: dict[str, Any] | None = None,
) -> FrozenDiffusionSampleResult:
```

默认逻辑：

```python
if risk_tilted is None:
    risk_tilted = bool(self.config.get("risk_tilted_diffusion", {}).get("enabled", False))
```

新增 helper：

```python
def _risk_tilted_config(self, override: dict[str, Any] | None = None) -> dict[str, Any]:
    defaults = {
        "enabled": False,
        "late_fraction": 0.30,
        "num_late_steps": 0,
        "guidance_scale": 0.05,
        "scale_schedule": "linear_ramp",
        "max_grad_norm": 1.0,
        "normalize_grad": True,
        "apply_at_t0": False,
        "lambda_phys": 0.2,
        "lambda_action_l2": 0.0,
        "min_grad_norm": 1e-12,
        "nan_to_num": True,
        "save_guidance_diagnostics": True,
    }
    cfg = dict(self.config.get("risk_tilted_diffusion", {}))
    defaults.update(cfg)
    if override:
        defaults.update(override)
    return defaults
```

新增 late-step 选择：

```python
def _guided_loop_indices(self, timesteps: list[int], cfg: dict[str, Any]) -> set[int]:
    if not cfg["enabled"]:
        return set()
    if int(cfg.get("num_late_steps", 0)) > 0:
        late_count = min(len(timesteps), int(cfg["num_late_steps"]))
    else:
        late_count = max(1, int(round(len(timesteps) * float(cfg.get("late_fraction", 0.30)))))
    return set(range(len(timesteps) - late_count, len(timesteps)))
```

采样循环应变为：

```python
raw_context = self.prior.decode_context_states(context_states).detach()
guided_loop_indices = self._guided_loop_indices(timesteps, tilted_cfg)

for loop_idx, step in enumerate(timesteps):
    is_guided = risk_tilted and loop_idx in guided_loop_indices
    if is_guided:
        x_t, guided_diag = self._risk_guided_reverse_step(...)
    else:
        with torch.no_grad():
            # current exact sampling branch
```

### 4.3 新增 `_risk_guided_reverse_step(...)`

建议作为 `FrozenDiffusionSampler` private method。

函数签名：

```python
def _risk_guided_reverse_step(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    *,
    context_states: torch.Tensor,
    context_features: torch.Tensor,
    relative_history: torch.Tensor,
    raw_context: torch.Tensor,
    ego_length: torch.Tensor | None,
    adv_length: torch.Tensor | None,
    generators: torch.Generator | list[torch.Generator] | None,
    loop_idx: int,
    guided_loop_indices: set[int],
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

实现要求：

1. `x_in = x_t.detach().requires_grad_(True)`
2. 使用 `torch.enable_grad()`。
3. 通过 frozen denoiser 计算 `x0_hat`。
4. decode `x0_hat` 为 raw jerk actions。
5. 用 `integrate_following_actions_torch()` 和 `compute_king_risk()` 计算风险。
6. 加入 `physical_violation_penalty()`。
7. 对 `x_in` 求梯度。
8. 使用 `torch.nan_to_num()` 清理异常梯度。
9. 按样本归一化或裁剪梯度。
10. 按 `constant` 或 `linear_ramp` 计算 guidance scale。
11. 只在允许时对 `posterior_mean` 加 `scale * grad`。
12. 按原 DDPM 逻辑加噪声。
13. 返回 `x_next.detach()` 和 detached diagnostics。

diagnostics 至少包括：

```text
guidance_objective
guidance_risk
guidance_physics
guidance_action_l2
guidance_grad_norm
guidance_scale
```

### 4.4 修改 `sample_batch(...)`

新增透传参数：

```python
def sample_batch(
    ...,
    risk_tilted: bool | None = None,
    risk_tilted_config: dict[str, Any] | None = None,
) -> FrozenDiffusionSampleResult:
```

并传给 `sample(...)`。

### 4.5 新增采样脚本

新增：

```text
adversaray/scripts/sample_risk_tilted_diffusion.py
```

职责：

```text
same tail contexts + same seeds
→ prior_actions
→ tilted_actions
→ optional king_actions baseline
→ save risk_tilted_samples.npz
```

默认值：

```python
SCRIPT_DEFAULTS = {
    "split": "val",
    "num_contexts": 256,
    "batch_size": 16,
    "seed": 42,
    "output_name": "risk_tilted_samples.npz",
    "run_king_baseline": True,
    "log_level": "INFO",
}
```

采样要求：

```python
seeds = [base_seed + start + pos for pos in range(len(prepared_contexts))]

prior_sample = sampler.sample_batch(batch, seed=seeds, risk_tilted=False)
tilted_sample = sampler.sample_batch(batch, seed=seeds, risk_tilted=True)
```

这样 prior 和 tilted 尽可能共享初始噪声和扩散噪声流。

必须显式重新评估最终 action 的 open-loop proxy diagnostics，不能只依赖 sampler 内部 guidance diagnostics。

输出数组至少包括：

```text
context_states
ego_length
adv_length
dataset_index
source_name

prior_actions
tilted_actions
king_actions

prior_risk_objective
tilted_risk_objective
king_risk_objective

prior_min_gap
tilted_min_gap
king_min_gap

prior_min_ttc
tilted_min_ttc
king_min_ttc

prior_min_rss_margin
tilted_min_rss_margin
king_min_rss_margin

tilted_guidance_steps
tilted_guidance_risk_mean
tilted_guidance_physics_mean
tilted_guidance_grad_norm_mean
tilted_guidance_scale_mean
```

输出 summary：

```text
risk_tilted_samples_summary.json
```

至少包含：

```text
prior_risk_mean
tilted_risk_mean
king_risk_mean
tilted_minus_prior_risk_mean
king_minus_prior_risk_mean
prior_min_gap_mean
tilted_min_gap_mean
king_min_gap_mean
tilted_action_l2_from_prior_mean
king_action_l2_from_prior_mean
tilted_physics_penalty_mean
king_physics_penalty_mean
guidance settings used
```

### 4.6 新增 closed-loop evaluator

新增：

```text
adversaray/scripts/evaluate_risk_tilted_samples.py
```

读取：

```text
risk_tilted_samples.npz
```

评估字段：

```python
PLAN_FIELDS = [
    ("prior", "prior_actions"),
    ("tilted", "tilted_actions"),
    ("king", "king_actions"),
]
```

缺失字段跳过。

每个 plan field 对每个 context 执行：

```python
runner.rollout_pre_sampled_plan(ctx, samples[action_key][case_id])
```

默认也应并行执行 highway-env fixed-plan rollouts，参考 `evaluate_king_guided_samples.py` 的 `num_workers` 设计。

输出：

```text
risk_tilted_eval_summary.json
figures/risk_tilted_closed_loop_histograms.png
figures/risk_tilted_case_XXXX.png
```

summary 结构：

```json
{
  "num_contexts": 256,
  "prior": {},
  "tilted": {},
  "king": {},
  "tilted_minus_prior": {},
  "king_minus_prior": {},
  "tilted_vs_king": {}
}
```

必须包含：

```text
closed_loop_risk_mean / p05 / p95
collision_rate
collision_valid_rate
invalid_collision_rate
near_collision_rate
min_gap_mean / p05
min_ttc_mean / p05
min_rss_margin_mean / p05
lead_physics_penalty_mean
jerk_violation_rate
speed_negative_rate
action_clip_rate
hard_brake_rate
```

图中 label 使用：

```text
prior
risk-tilted
KING
```

---

## 5. 正确性约束

### 5.1 默认行为不变

当：

```yaml
risk_tilted_diffusion:
  enabled: false
```

时，现有 KING 主线必须继续运行：

```bash
conda run -n tread python adversaray/scripts/sample_king_guided_diffusion.py
conda run -n tread python adversaray/scripts/evaluate_king_guided_samples.py
```

### 5.2 不训练 diffusion 权重

diffusion denoiser 参数必须保持 frozen。允许 autograd 通过 denoiser operations 求 `dQ/dx_t`，但不允许更新模型参数。

检查：

```python
assert not any(p.requires_grad for p in sampler.prior.model.parameters())
```

若当前模型参数默认 `requires_grad=True` 但处于 eval/no optimizer 状态，则应在 risk-tilted sampler 初始化或 guided branch 前显式：

```python
for p in self.prior.model.parameters():
    p.requires_grad_(False)
```

### 5.3 guided branch 不能被 `torch.no_grad()` 包住

非 guided branch 保持 `torch.no_grad()`。

guided branch 使用：

```python
with torch.enable_grad():
    ...
```

### 5.4 不保留计算图

每个 guided reverse step 后必须：

```python
x_t = x_next.detach()
```

diagnostics 只保存 detached tensors。

### 5.5 数值稳定

必须处理：

- NaN gradient
- zero gradient
- very large gradient
- physics penalty blow-up
- `t == 0`

使用：

```python
torch.nan_to_num
per-sample norm clamp
min_grad_norm
small guidance_scale
apply_at_t0: false
```

---

## 6. 评价标准

### 6.1 Open-loop proxy 指标

tilted 相比 prior 期望：

```text
tilted_minus_prior_risk_mean > 0
tilted_min_gap_mean < prior_min_gap_mean
tilted_min_ttc_mean < prior_min_ttc_mean
tilted_min_rss_margin_mean < prior_min_rss_margin_mean
```

但不能只靠 physics violation 提高 risk。

### 6.2 Closed-loop 指标

主要评价来自 highway-env：

```text
tilted_closed_loop_risk_mean > prior_closed_loop_risk_mean
tilted_near_collision_rate > prior_near_collision_rate
tilted_collision_valid_rate >= prior_collision_valid_rate
tilted_min_gap_p05 < prior_min_gap_p05
tilted_min_ttc_p05 < prior_min_ttc_p05
tilted_min_rss_margin_p05 < prior_min_rss_margin_p05
```

### 6.3 可行性约束

降低优先级或拒绝的结果：

```text
tilted_invalid_collision_rate 明显增加
tilted_jerk_violation_rate > king_jerk_violation_rate
tilted_speed_negative_rate > 0.05
tilted_lead_physics_penalty_mean 明显高于 KING
tilted_action_l2_from_prior_mean 高于 KING 但 closed-loop 风险没有更好
```

### 6.4 早停目标

若某次运行满足以下条件，可提前停止 10 轮搜索：

```text
closed_loop_risk_mean 相比 prior 提升 >= 10%
near_collision_rate 提升 >= 10 个百分点
min_gap_p05 下降 >= 1.0 m
min_ttc_p05 下降 >= 0.5 s
physics violations 不差于 KING baseline
tilted_action_l2_from_prior_mean < king_action_l2_from_prior_mean
```

若过严，则完成 10 轮并报告最优 trade-off。

---

## 7. 十轮实验计划

每轮流程：

```text
1. 设置 risk_tilted_diffusion 配置。
2. 运行 sample_risk_tilted_diffusion.py。
3. 运行 evaluate_risk_tilted_samples.py。
4. 保存 samples、summary、figures。
5. 追加 leaderboard。
6. 比较 prior / tilted / KING。
7. 决定下一轮配置。
```

建议前期小规模：

```text
num_contexts = 64
batch_size = 8 或 16
```

最佳 2 个配置再跑：

```text
num_contexts = 256
```

建议输出结构：

```text
data/adversaray/following/king_guided/risk_tilted_runs/
  run_01/
    config_used.yaml
    risk_tilted_samples.npz
    risk_tilted_samples_summary.json
    risk_tilted_eval_summary.json
    figures/
  run_02/
  ...
  leaderboard.csv
  best_run_summary.json
```

### 7.1 初始 10 个配置

Run 01：

```yaml
late_fraction: 0.20
guidance_scale: 0.02
scale_schedule: "linear_ramp"
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
lambda_action_l2: 0.0
```

Run 02：

```yaml
late_fraction: 0.20
guidance_scale: 0.05
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
```

Run 03：

```yaml
late_fraction: 0.20
guidance_scale: 0.10
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
```

Run 04：

```yaml
late_fraction: 0.30
guidance_scale: 0.05
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
```

Run 05：

```yaml
late_fraction: 0.30
guidance_scale: 0.10
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
```

Run 06：

```yaml
late_fraction: 0.30
guidance_scale: 0.10
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.5
```

Run 07：

```yaml
late_fraction: 0.30
guidance_scale: 0.05
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: true
lambda_phys: 0.2
```

Run 08：

```yaml
late_fraction: 0.40
guidance_scale: 0.05
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
```

Run 09：

```yaml
late_fraction: 0.30
guidance_scale: 0.02
normalize_grad: false
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
```

Run 10：

```yaml
late_fraction: 0.30
guidance_scale: 0.10
normalize_grad: true
max_grad_norm: 1.0
apply_at_t0: false
lambda_phys: 0.2
lambda_action_l2: 0.001
```

---

## 8. 运行命令

新增 risk-tilted workflow：

```bash
conda run -n tread python adversaray/scripts/prepare_king_guided_contexts.py
conda run -n tread python adversaray/scripts/sample_risk_tilted_diffusion.py
conda run -n tread python adversaray/scripts/evaluate_risk_tilted_samples.py
```

现有 KING baseline 必须继续可用：

```bash
conda run -n tread python adversaray/scripts/sample_king_guided_diffusion.py
conda run -n tread python adversaray/scripts/evaluate_king_guided_samples.py
```

---


完成 10 轮后写：

```text
best_run_summary.json
```

至少说明：

- closed-loop risk 最优 run。
- risk / naturalness trade-off 最优 run。
- near-collision 增加最明显 run。
- 是否达到早停目标。
- 是否值得测试 hybrid。

---
