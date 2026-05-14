# DeepEVT 条件尾部分布校准优化任务 Goal

## 任务名称

DeepEVT following 场景下的 q95 保守性修正、p 概率校准与条件分箱校准优化。

---

## 任务说明

本任务面向 `tread_deepevt` 模块中 following 场景下的 DeepEVT 条件极值尾部风险模型。

上一阶段已经完成训练稳定化与尾部外推有效性修复。当前代码中的以下改动视为已验证有效的稳定化基础，不应回退：

```text
1. best_model.pt / final_model.pt / model.pt checkpoint 契约；
2. Stage 1 early stopping；
3. Stage 2/3 训练日志中的 q90/q95 ECE 与 invalid rate；
4. checkpoint selection 中的 q90/q95 ECE 与 invalid rate 项；
5. eval_report.json 中的 invalid rate、q_tau 分布、p/xi/beta/u 分布诊断；
6. q90/q95 tail quantile loss 与 invalid extrapolation penalty。
```

当前主训练入口为：

```bash
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

当前主评估入口为：

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/best_model.pt
```

运行环境说明：

```text
conda activate jzm 本身确实有 CUDA；
GPU: NVIDIA GeForce RTX 4090 D；
Codex 默认沙箱看不到 NVIDIA 设备；
训练/评估需要使用沙箱外的 jzm CUDA 环境执行。
```

---

## 当前固定基线

当前最佳模型为：

```text
data/deepevt/following/best_model.pt
```

当前最终产物应保留：

```text
data/deepevt/following/best_model.pt
data/deepevt/following/final_model.pt
data/deepevt/following/model.pt
data/deepevt/following/checkpoint_summary.json
data/deepevt/following/training_history.json
data/deepevt/following/eval_report.json
data/deepevt/following/baseline_before_optimization/
deepevt_optimization_report.md
```

当前 best checkpoint 的测试集核心指标：

```text
u empirical exceed = 0.1659，目标为 0.15；
p_mean = 0.1154，仍低于实际 risk > u 的 0.1659；
q90 ECE = 0.0114；
q90 empirical exceed = 0.0886，目标为 0.10；
q90 invalid rate = 0.0；
q95 ECE = 0.0422；
q95 empirical exceed = 0.0078，目标为 0.05；
q95 invalid rate = 0.0；
q95 gap-bin mean abs error = 0.1817。
```

---

## 当前主要问题

当前模型已经从“尾部外推不可用”修复为“可用但 q95 偏保守”的条件尾部分布模型。后续优化应聚焦以下问题：

```text
1. q95 过保守：
   q95 empirical exceed = 0.0078，明显低于目标 0.05。

2. p 概率校准不足：
   p_mean = 0.1154，但实际 risk > u = 0.1659。
   p 当前更像被 invalid penalty 推到可外推区间，而不是真正校准的超阈概率。

3. 条件分箱仍存在低估：
   q95 gap-bin mean abs error 已从 baseline 的 0.2554 降到 0.1817，
   但各 gap bin 的 predicted q95 mean 仍低于 empirical q95。

4. q90/q95 权衡不够精细：
   强 invalid penalty 解决了 q90/q95 invalid rate，
   但可能把 q95 推向过保守。

5. Global POT-GPD 全局 ECE 仍很强：
   DeepEVT 的后续价值应体现在条件分箱、场景筛选和下游 tail condition 可用性。
```

---

## 总目标

在保持 DeepEVT 条件极值建模框架、`risk_score` 定义和上一阶段稳定化代码不变的前提下，进一步改善 following 场景下的概率校准和条件尾部分布质量。

优化重点不是单纯降低 `loss_total`，而是同时改善：

```text
1. q95 empirical exceed rate 向目标 0.05 靠近；
2. p_mean 与实际 risk > u 的差距缩小；
3. q95 条件分箱 mean abs error 继续降低；
4. q90 ECE 不明显回退；
5. q90/q95 invalid rate 继续保持 0 或接近 0；
6. best checkpoint 仍优于或不劣于 final checkpoint；
7. xi/beta 不出现贴边、坍缩或非物理性大幅漂移。
```

---

## 优先优化方向

建议优先尝试以下小范围策略，每轮只针对一个主要问题：

```text
1. p 后校准：
   使用验证集上的温度缩放、Platt calibration 或 monotonic calibration，
   不直接把 batch mean p 强行拉到 batch exceed mean。

2. q90/q95 分离权重：
   对 q90 与 q95 使用不同的 tail quantile loss / invalid penalty，
   避免为了修 q90 invalid 而让 q95 过度保守。

3. q95 exceed calibration：
   增加只针对 q95 empirical exceed 的温和校准项，
   目标是将 q95 empirical exceed 从 0.0078 拉近 0.05，
   但不得牺牲 q90 invalid rate。

4. 轻微解冻 threshold head：
   可尝试 threshold_lr_multiplier = 0.02 到 0.05，
   但必须监控 risk > u 是否仍接近 0.15。

5. 条件分箱校准：
   对 gap_current 分箱的 q95 误差加入验证集选择指标，
   或在评估报告中加入更多条件特征分箱。
```

已尝试且不建议作为默认方案：

```text
batch 级 lambda_p_rate_cal = 50.0 会导致 q95 invalid 回升和整体退化；
可以保留代码开关，但默认应为 0.0。
```

---

## 每轮验证命令

每轮修改后至少运行：

```bash
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/best_model.pt

python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/final_model.pt
```

每轮必须记录：

```text
1. best_validation stage/epoch/selection_score；
2. q90/q95 ECE；
3. q90/q95 empirical exceed rate；
4. q90/q95 invalid rate；
5. q95 gap-bin mean abs error；
6. p_mean 与 empirical risk > u；
7. xi/beta mean 与分位数；
8. best checkpoint 与 final checkpoint 对比。
```

---

## 迭代预算

最多执行 8 轮优化。

每轮要求：

```text
1. 只针对一个主要问题做小范围修改；
2. 修改前说明本轮假设；
3. 修改后运行训练和 best/final 评估；
4. 与当前固定基线和上一轮比较；
5. 如果 q90/q95 invalid 明显回升，优先修复；
6. 如果 q90 ECE 或 q95 分箱误差明显恶化，优先回滚；
7. 若连续 2 轮无法改善 q95 保守性，则停止并转入后校准方案。
```

---

## 停止条件

满足以下任一条件后停止：

```text
1. 完成 8 轮优化。

2. 达到以下组合目标：
   - q95 empirical exceed rate 提升到 [0.03, 0.07]；
   - q95 ECE <= 0.025；
   - q90 ECE <= 当前固定基线的 0.0114 * 1.2；
   - q90/q95 invalid rate 均 <= 0.01；
   - q95 gap-bin mean abs error <= 0.16；
   - p_mean 与 empirical risk > u 的绝对差距 <= 0.04。

3. 连续 2 轮核心指标提升不足。

4. 后续改动需要大规模重构模型结构，例如新的 temporal prefix encoder、
   mixture tail model、闭环 ADS 接口等；此时停止当前校准任务。
```

---

## 最终交付

最终输出或更新 `deepevt_optimization_report.md`，至少包含：

```text
1. 当前固定基线指标；
2. 最终训练和评估命令；
3. 最终 best/final checkpoint 路径；
4. 每轮假设、修改内容、修改文件和指标变化；
5. q90/q95 ECE 对比；
6. q90/q95 empirical exceed rate 对比；
7. q90/q95 invalid rate 对比；
8. q95 条件分箱误差对比；
9. p_mean 与 empirical risk > u 对比；
10. xi/beta 分布对比；
11. best checkpoint 与 final checkpoint 对比；
12. 是否达到停止条件；
13. 未解决问题；
14. 下一阶段建议。
```

---

## 一句话目标

在固定上一阶段已验证有效的 DeepEVT 稳定化代码基础上，把 following 模型从“q90/q95 外推有效但 q95 偏保守”进一步优化为“q90 不回退、q95 empirical exceed 更接近目标、p 更像真实超阈概率、条件分箱误差更小”的可报告条件尾部分布模型。
