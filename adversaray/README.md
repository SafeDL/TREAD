# adversaray

`adversaray/` 是 car-following 场景的 prior-guided adversarial generation 模块。当前主线是：

```text
frozen highD diffusion prior
→ tail-conditioned synthetic contexts
→ shared Stage 1 proposal policy over IDM ego surrogates
→ reusable scenario bank
```

旧的逐样本 action-gradient 路线已经移除公开入口；当前只保留共享 Stage 1 proposal policy 主线。

## Key Commands

项目默认 `python` 可能没有 PyTorch，建议使用 `tread` 环境：

```bash
conda run -n tread python adversaray/scripts/stage1/prepare_tail_contexts.py
conda run -n tread python adversaray/scripts/stage1/train_shared_proposal_policy.py
conda run -n tread python adversaray/scripts/stage1/build_stage1_scenario_bank.py
conda run -n tread python adversaray/scripts/stage1/diagnose_stage1_scenario_bank.py
```

可选工具：

```bash
conda run -n tread python adversaray/scripts/stage1/build_tail_context_scores.py
conda run -n tread python adversaray/scripts/stage1/calibrate_rss_on_highd.py
```

这些脚本把 Stage 1 主线路径和默认执行参数写在脚本内，直接运行即可。关键输入缺失时会直接报错；synthetic contexts 必须包含 train/val/test `split_index`，不要依赖全量 fallback。

可视化诊断：

```bash
conda run -n tread python adversaray/scripts/visualize/plot_stage1_scenario_bank.py
```

## Main Files

- `src/proxy_risk.py`：Stage 1 可微 proxy risk。
- `src/shared_proposal_policy.py`：共享低维扰动模板 policy。
- `src/ego_surrogate.py`：IDM ego surrogate 参数采样和张量工具。
- `scripts/stage1/prepare_tail_contexts.py`：构建 tail-conditioned synthetic train/val/test contexts。
- `scripts/stage1/train_shared_proposal_policy.py`：训练共享 Stage 1 proposal policy。
- `scripts/stage1/build_stage1_scenario_bank.py`：构建并筛选 reusable scenario bank。
- `scripts/stage1/diagnose_stage1_scenario_bank.py`：诊断 scenario bank 覆盖、多样性、自然性和物理性。
- `src/torch_kinematics.py` / `src/rss.py`：可微纵向动力学和 RSS 目标。
