# DeepEVT 条件尾部分布校准优化任务 Goal

## 任务名称

DeepEVT following 场景下的 q95 保守性修正、p 概率校准与条件分箱校准优化。

---

## 任务说明

本任务面向 `tread_deepevt` 模块中 following 场景下的 DeepEVT 条件极值尾部风险模型。

本文件已升级为下一阶段优化目标：允许更大尺度修改，包括但不限于模型结构、训练目标、校准机制、checkpoint selection、条件分箱指标、推理/评估接口，以及必要时重新构建 DeepEVT 数据集。

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

必要时允许重新构建数据集，主数据构建入口为：

```bash
python tread_deepevt/scripts/01_build_deepevt_dataset.py \
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

在保持 DeepEVT 条件极值建模目标、`risk_score` 定义和 following 场景任务语义不变的前提下，进一步改善 following 场景下的概率校准和条件尾部分布质量。

本阶段不再限制为小范围 loss/config 调参；允许模型层面、训练流程、数据管线和后校准机制的较大尺度改进。上一阶段稳定化代码原则上不应无理由回退，但如果新的模型/数据设计能更好满足指标，可以替换其实现方式，同时必须在报告中说明替换原因和回归验证结果。

允许的改进范围包括：

```text
1. 模型结构改进：
   - 新的 temporal prefix encoder；
   - context-conditioned tail head；
   - p/u 与 xi/beta 的解耦 head；
   - mixture / multi-regime tail model；
   - monotonic 或 constrained tail parameterization；
   - 显式 q90/q95 auxiliary heads 与 GPD tail head 联合建模。

2. 校准机制改进：
   - 验证集 Platt / temperature / isotonic calibration；
   - 条件化 conformal calibration；
   - gap_current / p-bin / risk-bin 条件后校准；
   - calibrated_p_for_report 与 p_for_tail_extrapolation 解耦。

3. 训练与选择机制改进：
   - 将 q95 empirical exceed、q95 gap-bin MAE、p calibration error 加入 checkpoint selection；
   - 引入 validation-driven 多目标 early stopping；
   - q90/q95 分离 loss、分离 invalid penalty、分离 calibration target；
   - 针对条件分箱的 differentiable 或 validation-side objective。

4. 数据与特征改进：
   - 必要时重新运行 `01_build_deepevt_dataset.py`；
   - 调整 context feature schema；
   - 增加 prefix temporal statistics；
   - 增加 gap / relative speed / TTC / DRAC 的分箱辅助特征；
   - 修复或重建 normalization_stats、feature_schema、canonical_contexts。

5. 评估与报告改进：
   - 增加更多条件分箱；
   - 分别报告 raw p、calibrated p、tail p；
   - 增加 per-bin q90/q95 ECE；
   - 增加 validation 与 test 的校准迁移诊断。
```

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

建议优先尝试以下方向。前 1-2 轮仍可优先选择低风险方案；若连续无法改善，则允许进入模型结构或数据重建级别修改。

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

threshold head 裸解冻 threshold_lr_multiplier = 0.02 已观察到会显著破坏 risk > u 和 q90 校准；
除非加入强 selection/constraint，否则不应作为默认方案。

全局 q95 additive/conformal offset 可改善 q95 全局 ECE，但会恶化 gap-bin MAE；
若使用后校准，应优先做条件化后校准，而不是全局 offset。
```

---

## 每轮验证命令

每轮修改后至少运行训练和 best/final 评估：

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

若本轮修改涉及数据源、特征 schema、split、normalization、canonical context、prefix 构造或 risk/context 对齐，则必须先运行：

```bash
python tread_deepevt/scripts/01_build_deepevt_dataset.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

若未运行 dataset rebuild，需要在报告中说明理由。

每轮必须记录：

```text
1. best_validation stage/epoch/selection_score；
2. q90/q95 ECE；
3. q90/q95 empirical exceed rate；
4. q90/q95 invalid rate；
5. q95 gap-bin mean abs error；
6. p_mean 与 empirical risk > u；
7. xi/beta mean 与分位数；
8. best checkpoint 与 final checkpoint 对比；
9. 若重建数据集，记录 rebuild 命令、输入数据、样本数、split 变化、feature_schema 变化与 normalization 变化。
```

---

## 迭代预算

最多执行 10 轮优化。

每轮要求：

```text
1. 只针对一个主要问题做可解释修改；允许大尺度修改，但必须控制变量；
2. 修改前说明本轮假设；
3. 如涉及数据或特征，先运行 dataset rebuild；
4. 修改后运行训练和 best/final 评估；
5. 与当前固定基线和上一轮比较；
6. 如果 q90/q95 invalid 明显回升，优先修复；
7. 如果 q90 ECE、risk > u 或 q95 分箱误差明显恶化，优先回滚或增加约束；
8. 若连续 2 轮小范围策略无法改善 q95 保守性，允许升级为模型结构、数据特征或条件校准方案，而不是直接停止。
```

---

## 停止条件

满足以下任一条件后停止：

```text
1. 完成 10 轮优化。

2. 达到以下组合目标：
   - q95 empirical exceed rate 提升到 [0.03, 0.07]；
   - q95 ECE <= 0.025；
   - q90 ECE <= 当前固定基线的 0.0114 * 1.2；
   - q90/q95 invalid rate 均 <= 0.01；
   - q95 gap-bin mean abs error <= 0.16；
   - p_mean 与 empirical risk > u 的绝对差距 <= 0.04。

3. 连续 3 轮核心指标提升不足，且已经尝试过至少一种模型结构或条件校准级别方案。

4. 后续改动已经超出 DeepEVT 条件尾部分布模型本身，例如闭环 ADS 接口、
   diffusion planner 联调、MATLAB 场景生成接口重构等；此时停止当前 DeepEVT 校准任务。

5. 连续 3 轮大尺度方案仍无法在 q95 empirical exceed、q90 ECE 和 q95 gap-bin MAE 之间取得净改善。
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
14. 下一阶段建议；
15. 若发生模型结构改动，说明新旧结构差异、参数约束、兼容性与推理接口变化；
16. 若发生数据集重建，说明 rebuild 原因、命令、样本数变化、split 变化、feature schema 变化、normalization 变化，以及是否影响论文结果可比性。
```

---

## 一句话目标

在保留 DeepEVT 条件尾部分布建模目标和 `risk_score` 语义的基础上，允许通过模型结构、校准机制、训练选择准则和必要的数据重建，把 following 模型从“q90/q95 外推有效但 q95 偏保守”进一步优化为“q90 不回退、q95 empirical exceed 更接近目标、p 更像真实超阈概率、条件分箱误差更小”的可报告条件尾部分布模型。
