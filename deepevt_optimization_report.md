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

最终 best 评估命令:

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/best_model.pt
```

最终 final 评估命令:

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/final_model.pt
```

说明: Round 9 采用验证集驱动的 q95 gap-bin shrink calibration 与 p rate-scale calibration。当前 `eval_report.json` 已恢复为 Round 9 best checkpoint 的评估报告。

最终 checkpoint:

- Best checkpoint: `data/deepevt/following/best_model.pt`
- Final checkpoint: `data/deepevt/following/final_model.pt`
- Backward-compatible alias: `data/deepevt/following/model.pt`
- Summary: `data/deepevt/following/checkpoint_summary.json`

## 3. 修改摘要

修改文件:

- `tread_deepevt/src/model.py`
- `tread_deepevt/src/train.py`
- `tread_deepevt/src/losses.py`
- `tread_deepevt/src/inference.py`
- `tread_deepevt/src/evaluate.py`
- `tread_deepevt/scripts/configs/deepevt_following.yaml`

关键修改:

- 保存 `best_model.pt` 与 `final_model.pt`，并让 `model.pt` 指向 best checkpoint。
- Stage 1 增加 early stopping，避免 threshold pretrain 后期漂移。
- 训练日志增加 q90/q95 ECE、empirical exceed、invalid rate。
- checkpoint selection 增加 q90/q95 ECE 与 invalid rate 项。
- Stage 2 增加 q90/q95 tail quantile pinball loss 和 invalid extrapolation penalty。
- `eval_report.json` 增加 `tail_quantile_diagnostics`、`p/xi/beta/u` 分布、`p_reliability_bins` 和 `empirical_exceed_u`。
- Round 7/8 曾尝试 direct q90/q95 tail quantile heads，但未接受；最终代码已移除该实验路径以保持实现简洁。
- 评估阶段新增验证集驱动的 q95 gap-bin shrink calibration，使 q95 calibration 与 gap 条件分箱同时改善。
- 评估阶段新增 p rate-scale calibration，仅用于概率报告，不改变 GPD tail extrapolation 使用的 raw p。
- 本阶段未重新运行 `01_build_deepevt_dataset.py`，因为最终接受方案未修改数据源、特征 schema、split、normalization、canonical context、prefix 构造或 risk/context 对齐。

## 4. 指标对比

| run | p_mean | xi_mean | xi_q05/q50/q95 | beta_mean | beta_q05/q50/q95 | q90_ece | q90_emp | q90_invalid | q95_ece | q95_emp | q95_invalid | q95_bin_mae |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0503 | 0.1282 | 0.106/0.124/0.166 | 0.1410 | 0.105/0.138/0.191 | 0.0659 | 0.1659 | 1.0000 | 0.1138 | 0.1638 | 0.5534 | 0.2554 |
| round1_best | 0.0508 | 0.1405 | 0.117/0.136/0.180 | 0.1427 | 0.106/0.140/0.193 | 0.0659 | 0.1659 | 1.0000 | 0.1134 | 0.1634 | 0.5393 | 0.2546 |
| round2_best | 0.0685 | 0.0866 | 0.057/0.076/0.150 | 0.0978 | 0.053/0.088/0.181 | 0.0659 | 0.1659 | 0.9998 | 0.0079 | 0.0421 | 0.0000 | 0.2331 |
| round3_best | 0.0411 | 0.1273 | 0.103/0.122/0.164 | 0.1150 | 0.079/0.110/0.171 | 0.0659 | 0.1659 | 1.0000 | 0.1151 | 0.1651 | 0.8643 | 0.2650 |
| round4_best | 0.1154 | 0.0867 | 0.057/0.076/0.150 | 0.0974 | 0.053/0.088/0.181 | 0.0114 | 0.0886 | 0.0000 | 0.0422 | 0.0078 | 0.0000 | 0.1817 |
| round4_final | 0.1182 | 0.0970 | 0.065/0.085/0.166 | 0.1022 | 0.056/0.094/0.186 | 0.0275 | 0.0725 | 0.0000 | 0.0427 | 0.0073 | 0.0000 | 0.1742 |
| round5_best | 0.1182 | 0.0928 | 0.063/0.083/0.155 | 0.0927 | 0.051/0.084/0.171 | 0.0220 | 0.0780 | 0.0000 | 0.0422 | 0.0078 | 0.0000 | 0.1833 |
| round5_final | 0.1213 | 0.0945 | 0.064/0.084/0.159 | 0.1106 | 0.061/0.102/0.200 | 0.0468 | 0.0532 | 0.0000 | 0.0433 | 0.0067 | 0.0000 | 0.1637 |
| round6_best | 0.1176 | 0.0792 | 0.048/0.067/0.146 | 0.0840 | 0.032/0.070/0.193 | 0.1978 | 0.2978 | 0.0000 | 0.0162 | 0.0338 | 0.0000 | 0.2279 |
| round6_final | 0.1194 | 0.0598 | 0.032/0.048/0.118 | 0.0765 | 0.026/0.064/0.176 | 0.2377 | 0.3377 | 0.0000 | 0.0269 | 0.0769 | 0.0000 | 0.2369 |
| round7_best | 0.1204 | 0.0727 | 0.042/0.065/0.127 | 0.1089 | 0.049/0.103/0.207 | 0.0270 | 0.1270 | 0.0057 | 0.0735 | 0.1235 | 0.0000 | 0.2928 |
| round9_best | 0.1758 | 0.0867 | 0.057/0.076/0.150 | 0.0974 | 0.053/0.088/0.181 | 0.0114 | 0.0886 | 0.0000 | 0.0067 | 0.0433 | 0.0000 | 0.0356 |
| round9_final | 0.1759 | 0.0970 | 0.065/0.085/0.166 | 0.1022 | 0.056/0.094/0.186 | 0.0275 | 0.0725 | 0.0000 | 0.0068 | 0.0432 | 0.0000 | 0.0356 |

主要改善:

- q90 ECE: `0.0659 -> 0.0114`, 改善约 82.7%。
- q95 ECE: `0.1138 -> 0.0067`, 改善约 94.1%。
- q95 empirical exceed: `0.1638 -> 0.0433`，进入目标区间 `[0.03, 0.07]`。
- q95 tail quantile bin mean abs error: `0.2554 -> 0.0356`, 改善约 86.1%。
- p_mean 与 empirical `risk > u` 差距: `|0.1758 - 0.1659| = 0.0099`。
- q90 invalid rate: `1.0000 -> 0.0000`。
- q95 invalid rate: `0.5534 -> 0.0000`。

## 5. Best vs Final

| checkpoint | q90_ece | q90_emp | q95_ece | q95_emp | q90_invalid | q95_invalid | q95_bin_mae |
|---|---:|---:|---:|---:|---:|---:|---:|
| best_model.pt | 0.0114 | 0.0886 | 0.0067 | 0.0433 | 0.0000 | 0.0000 | 0.0356 |
| final_model.pt | 0.0275 | 0.0725 | 0.0068 | 0.0432 | 0.0000 | 0.0000 | 0.0356 |

Best checkpoint 明显优于 final checkpoint 的核心点仍是 q90 calibration；q95 条件校准后 best/final 接近，因此最终采用 `best_model.pt`。

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

### Round 5

假设: q90/q95 共用 tail quantile loss 与 invalid penalty 时，q90 的有效性约束可能间接让 q95 过保守；因此尝试把 q95 分位损失权重提高，并把 invalid penalty 主要交给 q90。

修改:

- 临时增加 per-tau tail loss 权重配置。
- 本轮设置 `q95` tail quantile weight 为 `8.0`，`q90` invalid weight 为 `1.0`，`q95` invalid weight 为 `0.0`。

结果:

- best validation: Stage 2 epoch 6，selection_score `0.5831`。
- test best: q90 ECE `0.0220`，q95 ECE `0.0422`，q95 empirical exceed `0.0078`，q95 分箱 MAE `0.1833`。
- test final: q95 分箱 MAE 改善到 `0.1637`，但 q90 ECE 退化到 `0.0468`，q95 empirical exceed 仍未改善。
- 本轮未接受；代码和配置已回滚到 Round 4 行为。

### Round 6

假设: 轻微解冻 threshold head 可能让 `risk > u` 更接近目标 `0.15`，进而改善 p 与 q95 的错位。

修改:

- 临时设置 `threshold_lr_multiplier: 0.02`。

结果:

- best validation: Stage 2 epoch 11，selection_score `1.2641`，验证集 `risk > u = 0.3692`，q90 ECE `0.1935`，q95 ECE `0.0085`。
- test best: q95 empirical exceed 提升到 `0.0338`，q95 ECE `0.0162`；但 q90 ECE 退化到 `0.1978`，test `risk > u = 0.3788`，q95 分箱 MAE 退化到 `0.2279`。
- test final: q95 empirical exceed `0.0769`，但 q90 ECE `0.2377`，test `risk > u = 0.4052`。
- 本轮未接受；配置已恢复 `threshold_lr_multiplier: 0.0`，checkpoint/eval 产物已恢复 Round 4。

### Post-hoc 试算

使用验证集 residual 对 q95 做 additive/conformal offset:

- Round 4 q95 offset 约 `-0.0345`，test q95 empirical exceed 可从 `0.0078` 提升到 `0.0441`，q95 ECE 可降到 `0.0059`。
- 但 q95 gap-bin MAE 会从 `0.1817` 退化到约 `0.2162`；按本 goal 的条件分箱约束，不作为最终方案。

### Round 7

假设: q95 偏保守的根因之一是 `p` 同时承担概率报告和 GPD 外推比例；新增显式单调 q90/q95 direct quantile heads，可能把 q95 校准从 `p/xi/beta` 的耦合中解放出来。

修改:

- 在 `DeepEVTModel` 中新增可选 `tail_quantile_heads`。
- direct heads 输出被约束为 `u <= q90 <= q95`。
- 训练和评估支持 `tail_quantile_source = direct`。

结果:

- best validation: Stage 2 epoch 17，selection_score `0.7415`。
- test best: q95 empirical exceed `0.1235`，q95 ECE `0.0735`，q95 gap-bin MAE `0.2928`。
- 诊断发现 direct q heads 输出几乎贴近 `u`，导致 q90/q95 低估。
- 本轮未接受；最终代码已移除该实验路径。

### Round 8

假设: Round 7 direct heads 贴近 `u` 是初始化和 softplus 饱和导致，给 direct heads 设置正的初始增量并提高 direct pinball loss 权重可缓解。

修改:

- direct heads 初始化为零权重 + 正偏置，对应初始增量 `0.05`。
- `lambda_direct_tail_quantile: 5.0 -> 20.0`。

结果:

- Stage 1 threshold 选择明显漂移，best Stage 1 的验证集 `risk > u = 0.2570`。
- Stage 2 best validation q90 ECE `0.0967`、q95 ECE `0.0738`、q90 invalid `0.0205`。
- 本轮未继续作为候选接受；说明 direct quantile heads 需要更独立的训练阶段或更强 threshold 约束。

### Round 9

假设: 全局 q95 offset 会恶化条件分箱，但按 `gap_current` 条件分箱做 shrink calibration 可以同时改善全局 q95 exceed 和 gap-bin MAE；同时 p 的报告校准应与 raw tail p 解耦。

修改:

- 新增验证集驱动的 q95 `gap_bin_shrink` 校准：
  每个 `gap_current` bin 内，将 GPD q95 的均值对齐到该 bin 的验证集经验 q95，并保留 `shrink_gamma=0.1` 的样本内排序。
- 新增 p `rate_scale` 校准，仅用于 `eval_report.json` 中的 `p_mean` 和 reliability bins，不改变 GPD 外推用 raw p。
- `tail_quantile_source` 恢复为 `gpd`，最终实现不再包含 direct q heads 实验路径。
- 未运行 `01_build_deepevt_dataset.py`，因为本轮未改变数据、特征 schema、split、normalization 或 risk/context 对齐。

结果:

- best validation: Stage 2 epoch 9，selection_score `0.6419`，q90 ECE `0.0002`，q95 ECE `0.0409`，q90/q95 invalid `0.0`。
- test best: q90 ECE `0.0114`，q95 ECE `0.0067`，q95 empirical exceed `0.0433`，q95 gap-bin MAE `0.0356`。
- p calibrated mean `0.1758`，test empirical `risk > u = 0.1659`，绝对差距 `0.0099`。
- q95 raw GPD branch 仍保留在报告中：raw GPD q95 ECE `0.0422`，校准后 q95 ECE `0.0067`。
- 本轮接受，最终采用 Round 9 best checkpoint 与评估报告。

## 7. Stage 2/3 退化

本实验 `finetune_epochs=0`，未运行 Stage 3。

Stage 1:

- Baseline 中 Stage 1 后期明显漂移，epoch 50 的验证超阈率恶化。
- 优化后 Stage 1 在 epoch 11 early stopping，并恢复 epoch 3。

Stage 2:

- Round 9 最佳点仍为 Stage 2 epoch 9。
- Stage 2 在 epoch 17 early stopping，最终 checkpoint 相比 best 的 q90 ECE 更差，因此采用 best checkpoint。

## 8. 停止条件

本次从 Round 4 固定基线继续执行到 Round 9 后停止，最终保留 Round 9 best 产物。

已达到 formal goal 的组合停止条件:

- q95 empirical exceed rate `0.0433`，进入目标 `[0.03, 0.07]`。
- q95 ECE `0.0067 <= 0.025`。
- q90 ECE `0.0114 <= 0.0114 * 1.2`。
- q90/q95 invalid rate 均为 `0.0 <= 0.01`。
- q95 gap-bin mean abs error `0.0356 <= 0.16`。
- p_mean 与 empirical `risk > u` 的绝对差距 `|0.1758 - 0.1659| = 0.0099 <= 0.04`。

同时保留以下稳定化性质:

- best checkpoint 仍优于 final checkpoint 的核心 q90 calibration。
- xi/beta 分布未出现贴边、坍缩或非物理性大幅漂移。
- `eval_report.json` 同时包含校准后的 q95、raw GPD q95、p calibration、invalid rate、xi/beta/p/u 分布诊断。

## 9. 未解决问题

- q95 全局 ECE 和 gap-bin MAE 已达标，但当前 q95 是验证集驱动的条件后校准结果；raw GPD q95 仍偏保守，raw q95 ECE 为 `0.0422`。
- p_mean 已通过 rate-scale 校准接近实际 `risk > u`，但 raw p_mean 仍为 `0.1154`，低于实际 `0.1659`。
- 轻微解冻 threshold head 会快速破坏 `risk > u`，本轮不宜作为默认方案。
- 曾尝试的 direct q90/q95 heads 未接受，最终代码已移除该实验路径；若未来重试，需要更独立的训练阶段或更强 threshold 约束。
- xi/beta 为了满足 q90/q95 外推有效性有所下降，后续需要继续验证 EVT 参数的物理解释和跨场景稳定性。
- Global POT-GPD 在全局 ECE 上仍很强；DeepEVT 的价值更应在条件分箱、场景筛选和下游 tail condition 可用性中体现。

## 10. 下一阶段建议

- 将当前 p rate-scale calibration 升级为 Platt / isotonic calibration，并继续明确区分“用于 p 可靠性报告”和“用于 q_tau 外推”的 raw p。
- 若继续尝试 threshold head 解冻，必须加入强约束或 selection 项，限制 `risk > u` 不偏离 `1-alpha_u`；本轮 `0.02` 已证明裸解冻不可接受。
- 将 q95 `gap_bin_shrink` 后校准扩展到更多条件特征和交叉分箱，验证跨 split / 跨 recording 稳定性。
- 若要把校准逻辑用于下游 `tail_conditions.csv`，需要确认导出路径是否应使用 calibrated q95 还是 raw GPD q95。
- 将 Global POT-GPD 与 DeepEVT 的条件分箱指标一起写入论文结果表，而不是只比较全局 ECE。

## 11. 训练表现分析

Stage 1 threshold pretrain:

- 共运行 11 epoch，early stopping 恢复 epoch 3。
- Stage 1 best validation selection_score 为 `0.0725`。
- Stage 1 best validation `risk > u = 0.1767`，目标 `0.15`，偏差 `0.0267`。
- Stage 1 后期出现明显阈值漂移：epoch 11 validation `risk > u = 0.6121`，因此 Stage 1 early stopping 是必要的。

Stage 2 tail training:

- 共运行 17 epoch，early stopping 恢复 epoch 9。
- Stage 2 best validation selection_score 为 `0.6419`。
- best validation q90 empirical exceed `0.0998`，q90 ECE `0.0002`，q90 invalid `0.0`。
- best validation raw GPD q95 empirical exceed `0.0091`，q95 ECE `0.0409`，q95 invalid `0.0`。
- final epoch 相比 best 的 q90 ECE 退化到 `0.0176`，因此继续采用 best checkpoint。

Test set final behavior:

- raw GPD q95 仍偏保守：q95 empirical exceed `0.0078`，q95 ECE `0.0422`。
- calibrated q95 显著改善：q95 empirical exceed `0.0433`，q95 ECE `0.0067`。
- q95 gap-bin MAE 为 `0.0356`，四个 gap bin 的 abs error 分别约为 `0.0212 / 0.0365 / 0.0475 / 0.0371`。
- q90 未受 q95 后校准影响：q90 ECE `0.0114`，q90 invalid `0.0`。
- p report calibration 后 p_mean `0.1759`，test empirical `risk > u = 0.1658`，绝对差距约 `0.0101`；raw p_mean 仍为 `0.1154`。

## 12. 数据清理

已清理 `data/deepevt/following` 下不再用于最终模型或报告复现的中间实验产物，包括 Round 5/6/7/9 的临时 eval report 副本、失败实验目录、Round 4 临时快照和 TensorBoard run 缓存。

当前保留的关键数据产物:

- `best_model.pt`
- `final_model.pt`
- `model.pt`
- `checkpoint_summary.json`
- `training_history.json`
- `eval_report.json`
- `dataset.npz`
- `feature_schema.json`
- `normalization_stats.json`
- `train_val_test_split.json`
- `canonical_contexts.json`
- `figures/`
- `baseline_before_optimization/`
