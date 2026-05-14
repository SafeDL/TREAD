# DeepEVT Following Optimization Report

日期: 2026-05-14  
运行环境: `/home/hp/anaconda3/envs/jzm/bin/python`, PyTorch `1.13.1+cu117`, CUDA device `NVIDIA GeForce RTX 4090 D`  
说明: Codex 默认沙箱看不到 GPU；训练/评估均使用沙箱外的 `jzm` CUDA 环境执行。

## 1. Baseline

Baseline 文件已保留在:

- `data/deepevt/following/baseline_before_optimization/training_history.json`
- `data/deepevt/following/baseline_before_optimization/eval_report.json`
- `data/deepevt/following/baseline_before_optimization/eval_report_enhanced.json`
- `data/deepevt/following/baseline_before_optimization/model.pt`
- `data/deepevt/following/baseline_before_optimization/runs/`

Baseline 命令:

```bash
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

增强版 baseline 诊断使用修改后的评估代码重新评估原始 checkpoint:

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/baseline_before_optimization/model.pt \
  --no-quantile-baseline
```

## 2. 最终命令与 Checkpoint

最终训练命令:

```bash
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

最终评估命令:

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/best_model.pt \
  --no-quantile-baseline
```

最终 checkpoint:

- Best checkpoint: `data/deepevt/following/best_model.pt`
- Final checkpoint: `data/deepevt/following/final_model.pt`
- Backward-compatible alias: `data/deepevt/following/model.pt`
- Summary: `data/deepevt/following/checkpoint_summary.json`

## 3. 修改摘要

修改文件:

- `tread_deepevt/src/train.py`
- `tread_deepevt/src/losses.py`
- `tread_deepevt/src/evaluate.py`
- `tread_deepevt/scripts/configs/deepevt_following.yaml`

关键修改:

- 保存 `best_model.pt` 与 `final_model.pt`，并让 `model.pt` 指向 best checkpoint。
- Stage 1 增加 early stopping，避免 threshold pretrain 后期漂移。
- 训练日志增加 q90/q95 ECE、empirical exceed、invalid rate。
- checkpoint selection 增加 q90/q95 ECE 与 invalid rate 项。
- Stage 2 增加 q90/q95 tail quantile pinball loss 和 invalid extrapolation penalty。
- `eval_report.json` 增加 `tail_quantile_diagnostics`、`p/xi/beta/u` 分布、`p_reliability_bins` 和 `empirical_exceed_u`。

## 4. 指标对比

| run | p_mean | xi_mean | xi_q05/q50/q95 | beta_mean | beta_q05/q50/q95 | q90_ece | q90_emp | q90_invalid | q95_ece | q95_emp | q95_invalid | q95_bin_mae |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0503 | 0.1282 | 0.106/0.124/0.166 | 0.1410 | 0.105/0.138/0.191 | 0.0659 | 0.1659 | 1.0000 | 0.1138 | 0.1638 | 0.5534 | 0.2554 |
| round1_best | 0.0508 | 0.1405 | 0.117/0.136/0.180 | 0.1427 | 0.106/0.140/0.193 | 0.0659 | 0.1659 | 1.0000 | 0.1134 | 0.1634 | 0.5393 | 0.2546 |
| round2_best | 0.0685 | 0.0866 | 0.057/0.076/0.150 | 0.0978 | 0.053/0.088/0.181 | 0.0659 | 0.1659 | 0.9998 | 0.0079 | 0.0421 | 0.0000 | 0.2331 |
| round3_best | 0.0411 | 0.1273 | 0.103/0.122/0.164 | 0.1150 | 0.079/0.110/0.171 | 0.0659 | 0.1659 | 1.0000 | 0.1151 | 0.1651 | 0.8643 | 0.2650 |
| round4_best | 0.1154 | 0.0867 | 0.057/0.076/0.150 | 0.0974 | 0.053/0.088/0.181 | 0.0114 | 0.0886 | 0.0000 | 0.0422 | 0.0078 | 0.0000 | 0.1817 |
| round4_final | 0.1182 | 0.0970 | 0.065/0.085/0.166 | 0.1022 | 0.056/0.094/0.186 | 0.0275 | 0.0725 | 0.0000 | 0.0427 | 0.0073 | 0.0000 | 0.1742 |

主要改善:

- q90 ECE: `0.0659 -> 0.0114`, 改善约 82.7%。
- q95 ECE: `0.1138 -> 0.0422`, 改善约 62.9%。
- q95 tail quantile bin mean abs error: `0.2554 -> 0.1817`, 改善约 28.9%。
- q90 invalid rate: `1.0000 -> 0.0000`。
- q95 invalid rate: `0.5534 -> 0.0000`。

## 5. Best vs Final

| checkpoint | q90_ece | q90_emp | q95_ece | q95_emp | q90_invalid | q95_invalid | q95_bin_mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_model.pt | 0.0114 | 0.0886 | 0.0422 | 0.0078 | 0.0000 | 0.0000 | 0.1817 |
| final_model.pt | 0.0275 | 0.0725 | 0.0427 | 0.0073 | 0.0000 | 0.0000 | 0.1742 |

Best checkpoint 明显优于 final checkpoint 的核心点是 q90 calibration；因此最终采用 `best_model.pt`。

## 6. 每轮记录

### Round 1

假设: 当前问题首先是 checkpoint 和诊断契约不完整，训练过程无法可靠选择 q90/q95 更好的点。

修改:

- 增加 `best_model.pt`、`final_model.pt`、`checkpoint_summary.json`。
- 增加 Stage 1 early stopping。
- 训练与评估增加 q90/q95 ECE 和 invalid rate 诊断。

结果:

- q90 ECE 基本不变，q95 ECE 略微改善。
- best 明显优于 final，确认 checkpoint 机制有必要。
- 仍存在 q90 invalid 100%、q95 invalid 约 54%。

### Round 2

假设: p 太低导致 q90/q95 外推无效，需要直接约束 tail quantile 与 invalid rate。

修改:

- 增加 q90/q95 tail quantile pinball loss。
- 增加 invalid extrapolation penalty。
- 打开 `lambda_tail_quantile=0.5`, `lambda_tail_invalid=20.0`。

结果:

- q95 ECE 大幅改善到 `0.0079`，q95 invalid 归零。
- q90 仍几乎完全 invalid，q90 ECE 未改善。

### Round 3

假设: p 低估可通过 batch 级 p-rate calibration 修复。

修改:

- 增加可选 `lambda_p_rate_cal`。
- 本轮设置为 `50.0`。

结果:

- 验证集和测试集明显恶化，q95 invalid 回升。
- 按任务要求回滚本轮配置；最终配置中保留代码但 `lambda_p_rate_cal=0.0`。

### Round 4

假设: Round 2 的方向正确，但 invalid penalty 和 checkpoint selection 对 q90 不够敏感。

修改:

- `lambda_tail_invalid: 20.0 -> 200.0`
- `selection_invalid_weight: 0.1 -> 1.0`

结果:

- q90/q95 invalid rate 均归零。
- q90 ECE 改善到 `0.0114`。
- q95 ECE 改善到 `0.0422`。
- q95 分箱 MAE 改善到 `0.1817`。

## 7. Stage 2/3 退化

本实验 `finetune_epochs=0`，未运行 Stage 3。

Stage 1:

- Baseline 中 Stage 1 后期明显漂移，epoch 50 的验证超阈率恶化。
- 优化后 Stage 1 在 epoch 11 early stopping，并恢复 epoch 3。

Stage 2:

- Round 4 最佳点为 Stage 2 epoch 9。
- Stage 2 在 epoch 17 early stopping，最终 checkpoint 相比 best 的 q90 ECE 更差，因此采用 best checkpoint。

## 8. 停止条件

已满足 formal goal 的组合停止条件:

- q95 ECE 不劣于 baseline: `0.0422 < 0.1138`。
- q90 ECE 改善至少 10%: `0.0659 -> 0.0114`。
- q95 tail_quantile_bins 平均 abs error 改善至少 10%: `0.2554 -> 0.1817`。
- best checkpoint 明显优于 final checkpoint: q90 ECE `0.0114 < 0.0275`。
- `eval_report.json` 已包含 invalid rate、xi/beta/p 分布诊断。

## 9. 未解决问题

- q95 test empirical exceed rate 为 `0.0078`，低于目标 `0.05`，当前 q95 偏保守；但相对 baseline 的严重低估已经显著改善。
- p_mean 从 `0.0503` 提升到 `0.1154`，仍低于实际 `risk > u` 的 `0.1659`。
- xi/beta 为了满足 q90/q95 校准有所下降，后续需要继续验证 EVT 参数的物理解释和跨场景稳定性。
- Global POT-GPD 在全局 ECE 上仍很强；DeepEVT 的价值更应在条件分箱、场景筛选和下游 tail condition 可用性中体现。

## 10. 下一阶段建议

- 将 p calibration 从 batch mean 改成未加权验证分布上的温度/Platt calibration，避免 Round 3 那种训练内耦合恶化。
- 尝试轻微解冻 threshold head，例如 `threshold_lr_multiplier=0.02~0.05`，但需约束 `empirical_exceed_u` 不偏离 `1-alpha_u`。
- 对 q90 和 q95 使用不同的 invalid penalty，避免 q90 修复后 q95 过保守。
- 将 Global POT-GPD 与 DeepEVT 的条件分箱指标一起写入论文结果表，而不是只比较全局 ECE。
