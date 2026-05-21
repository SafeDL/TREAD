# adversaray

`adversaray/` 是 car-following 场景的 prior-guided adversarial generation 模块。当前主线是：

```text
frozen highD diffusion prior
→ sample natural lead jerk plan j0
→ KING-style differentiable longitudinal risk optimization
→ highway-env closed-loop validation
```

KING guidance 只在可微纵向 proxy 上优化动作序列；最终风险仍以 highway-env rollout 为准。

## Key Commands

项目默认 `python` 可能没有 PyTorch，建议使用 `tread` 环境：

```bash
conda run -n tread python adversaray/scripts/sample_king_guided_diffusion.py
conda run -n tread python adversaray/scripts/evaluate_prior_guided_policy.py
```

可选工具：

```bash
conda run -n tread python adversaray/scripts/build_tail_context_scores.py
conda run -n tread python adversaray/scripts/build_evt_synthetic_contexts.py
conda run -n tread python adversaray/scripts/calibrate_rss_on_highd.py
```

这些脚本现在把 KING 主线路径和默认执行参数写在脚本内，直接运行即可。关键输入缺失时会直接报错；例如 synthetic contexts 必须包含 `split_index`，不要依赖全量 fallback。

可视化诊断：

```bash
conda run -n tread python adversaray/scripts/plot_king_eval_cases.py
conda run -n tread python adversaray/scripts/visualize_king_rollout.py
```

## Main Files

- `src/king_gradient_guidance.py`：KING-style risk objective 和 raw jerk plan optimizer。
- `scripts/sample_king_guided_diffusion.py`：采样 frozen prior plan 并保存 KING-guided plan dataset。
- `scripts/evaluate_prior_guided_policy.py`：`--king-gradient` 闭环评估，比较 prior plan 与 KING plan。
- `scripts/plot_king_eval_cases.py`：绘制 prior/KING proxy 汇总直方图；可显式指定少量 case 输出曲线。
- `scripts/visualize_king_rollout.py`：用 highway-env graphics replay prior/KING rollout，支持窗口、GIF 和 PNG 帧。
- `src/torch_kinematics.py` / `src/rss.py`：可微纵向动力学和 RSS 目标。
- `src/closed_loop_runner.py`：highway-env car-following validator。

`guidance_policy.py` 和 learned guidance sampler 仍保留为 legacy ablation 支撑，但训练、distillation、CEM expert search 入口已经移除。
