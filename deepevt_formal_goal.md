# DeepEVT 条件尾部分布校准优化任务 Goal

## 任务名称

DeepEVT following 场景下的 New Round 4 tail-quantile 结果固化、raw GPD 改进与剩余 3 轮再优化。

---

## 任务说明

本任务面向 `tread_deepevt` 模块中 following 场景下的 DeepEVT 条件极值尾部风险模型。

当前代码与产物已经完成 New Round 1-5 优化尝试，其中 New Round 1/2/4 已接受，New Round 3/5 已拒绝并回滚。下一阶段的首要要求是：**固化当前 New Round 4 tail-quantile accepted 结果为新基线，再以该基线为唯一比较对象执行剩余 3 轮有记录的优化**。

本阶段允许继续修改模型结构、训练目标、校准机制、checkpoint selection、评估接口和报告指标；但任何修改都必须证明优于或不劣于当前 New Round 4 accepted baseline。若新方案未能带来净改善，应回滚配置或不接受该轮 checkpoint。

conda activate jzm环境中有cuda,但是codex沙盒中没有,所以你可以在训练的时候移动到沙盒外执行

---

## 当前代码固化要求

当前实现视为已验证有效的稳定版本。以下契约不得在没有明确替代方案和回归验证的情况下回退：

```text
1. best_model.pt / final_model.pt / model.pt checkpoint 契约；
2. Stage 1 early stopping；
3. Stage 2/3 训练日志中的 q90/q95 ECE、empirical exceed 与 invalid rate；
4. checkpoint selection 中的 q90/q95 ECE 与 invalid rate 项；
5. eval_report.json 中的 invalid rate、q_tau 分布、raw/calibrated p、xi/beta/u 分布诊断；
6. q90/q95 tail quantile loss 与 invalid extrapolation penalty；
7. 验证集驱动的 p calibration，其中 New Round 2 当前采用 isotonic_auto:decreasing；
8. 验证集驱动的 q95 gap-bin shrink calibration；
9. New Round 4 接受的 lambda_tail_quantile = 2.0；
10. best/final 评估命令与报告字段兼容性。
```

固化含义：

```text
1. 当前代码、配置和 data/deepevt/following 下的 best 产物作为新 baseline；
2. 后续剩余 3 轮优化前必须记录当前 baseline 指标；
3. 每轮只接受相对当前 baseline 或上一接受轮有净改善的改动；
4. 未接受的实验可以写入报告，但不得覆盖当前 best 产物；
5. 若修改评估口径，必须同时保留旧口径，避免与当前 baseline 失去可比性。
```

---

## 主运行入口

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

必要时允许重新构建数据集：

```bash
python tread_deepevt/scripts/01_build_deepevt_dataset.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

运行环境说明：

```text
conda activate jzm 本身确实有 CUDA；
GPU: NVIDIA GeForce RTX 4090 D；
Codex 默认沙箱看不到 NVIDIA 设备；
训练/评估需要使用沙箱外的 jzm CUDA 环境执行。
```

---

## 当前固定基线：New Round 4 Tail-Quantile Accepted

当前固定基线模型为：

```text
data/deepevt/following/best_model.pt
```

当前固定基线快照已保存在：

```text
data/deepevt/following/accepted_round2_isotonic/
data/deepevt/following/accepted_round4_tailq2/
```

当前固定基线产物应保留：

```text
data/deepevt/following/best_model.pt
data/deepevt/following/final_model.pt
data/deepevt/following/model.pt
data/deepevt/following/checkpoint_summary.json
data/deepevt/following/training_history.json
data/deepevt/following/eval_report.json
data/deepevt/following/eval_report_final.json
data/deepevt/following/accepted_round2_isotonic/
data/deepevt/following/accepted_round4_tailq2/
data/deepevt/following/rejected_round3_p_mean_target/
data/deepevt/following/rejected_round5_shrink0/
data/deepevt/following/baseline_before_optimization/
deepevt_optimization_report.md
```

当前 best checkpoint 的测试集核心指标：

```text
n_test = 9120；
u empirical exceed = 0.1658，目标为 0.15；
raw p_mean = 0.1154；
calibrated p_mean = 0.1842；
calibrated p_mean 与 empirical risk > u 的绝对差距 = 0.0183；
p calibration method = isotonic_auto:decreasing；
p reliability mean abs error = 0.0177；
p reliability Spearman = 1.0；
p reliability monotonic violations = 0；

q90 ECE = 0.0114；
q90 empirical exceed = 0.0886，目标为 0.10；
q90 invalid rate = 0.0；

q95 calibrated ECE = 0.0066；
q95 calibrated empirical exceed = 0.0434，目标为 0.05；
q95 invalid rate = 0.0；
q95 gap-bin mean abs error = 0.0356；

q95 raw GPD ECE = 0.0419；
q95 raw GPD empirical exceed = 0.0081；

xi_mean = 0.0935；
beta_mean = 0.0946。
```

当前 `checkpoint_summary.json` 中的 best validation：

```text
stage = 2；
epoch = 13；
selection_score = 0.6406；
validation empirical exceed = 0.1767；
validation q90 ECE = 0.0003；
validation q95 ECE = 0.0405；
validation q90 invalid rate = 0.0；
validation q95 invalid rate = 0.0。
```

New Round 2 相对 Round 9 best 的固化改善：

```text
p reliability mean abs error = 0.0949 -> 0.0190；
p reliability Spearman = -1.0 -> 1.0；
p reliability monotonic violations = 4 -> 0；
q90/q95 tail 指标保持不变；
raw GPD 指标保持不变；
该轮只改变 p_report calibration，不改变 p_for_tail_extrapolation。

New Round 4 相对 New Round 2 accepted 的固化改善：

```text
lambda_tail_quantile = 0.5 -> 2.0；
q95 raw GPD ECE = 0.0422 -> 0.0419；
q95 calibrated ECE = 0.0067 -> 0.0066；
p reliability mean abs error = 0.0190 -> 0.0177；
q90 ECE 保持 0.0114；
raw p_mean 基本未改善，仍为后续问题。
```

---

## 当前尚存问题

当前模型已经从“尾部外推不可用”和“q95 明显偏保守”推进到“校准后 q95 可报告、条件分箱误差显著降低、p reliability 排序已修复”的状态。后续优化不再以 p 条件后校准为主，而是聚焦以下剩余问题：

```text
1. raw GPD q95 仍然过保守：
   q95 raw GPD empirical exceed = 0.0078，明显低于目标 0.05；
   q95 raw GPD ECE = 0.0422。
   当前可报告 q95 依赖 gap-bin shrink 后校准。

2. raw p_mean 仍低于真实超阈概率：
   raw p_mean = 0.1154；
   empirical risk > u = 0.1658；
   二者差距约 0.0504。

3. calibrated p 的条件排序已经显著改善，但全局均值略偏高：
   p reliability Spearman = 1.0；
   p reliability monotonic violations = 0；
   p reliability mean abs error = 0.0190；
   calibrated p_mean = 0.1847，高于 empirical risk > u = 0.1658。

4. q90 略保守：
   q90 empirical exceed = 0.0886，目标 0.10；
   q90 ECE = 0.0114，已可接受但不能继续回退。

5. q95 gap-bin MAE 已很低，但仍有改进空间：
   当前 q95 gap-bin mean abs error = 0.0356；
   四个 gap bin 仍全部轻微低估 empirical q95。

6. 校准迁移风险需要继续监控：
   q95 gap-bin shrink calibration 在 validation 上拟合、test 上应用；
   New Round 1 已增加 validation/test 迁移诊断；
   后续修改仍需避免后校准过拟合。

7. Global POT-GPD 的全局 q95 ECE 仍更低：
   global q95 ECE = 0.0048；
   DeepEVT 的价值应继续体现在条件分箱、场景筛选、tail condition 可用性和下游生成接口。
```

---

## 总目标

在保持 DeepEVT 条件极值建模目标、`risk_score` 定义和 following 场景任务语义不变的前提下，以 New Round 4 tail-quantile accepted 为固定基线，执行剩余 3 轮优化，使模型从“p_report 条件可靠性已修复、raw q95 仅小幅改善、raw p 仍低估”的状态进一步变成“raw GPD 更可信、raw p 更接近真实超阈概率、校准迁移更稳定”的条件尾部分布模型。

本阶段优化重点不是单纯降低 `loss_total`，而是同时改善：

```text
1. raw GPD q95 empirical exceed 向 0.05 靠近；
2. raw p_mean 与 empirical risk > u 的差距缩小；
3. calibrated p 的 reliability bins 保持单调且不过度牺牲全局均值；
4. q95 calibrated ECE 保持或优于当前 0.0067；
5. q95 gap-bin mean abs error 保持或优于当前 0.0356；
6. q90 ECE 不超过当前 0.0114 的 1.2 倍；
7. q90/q95 invalid rate 继续保持 0 或接近 0；
8. xi/beta 不出现贴边、坍缩或非物理性大幅漂移；
9. best checkpoint 仍优于或不劣于 final checkpoint；
10. validation 到 test 的校准迁移误差可解释、可报告。
```

---

## 剩余 3 轮优化计划

New Round 1/2/4 已完成并接受；New Round 3/5 已拒绝并回滚。本阶段后续固定执行剩余 3 轮优化。每轮只围绕一个主假设展开；允许记录失败实验，但只有满足接受条件的结果才能覆盖当前 best 产物。

### Round 1：固化基线与评估口径

状态：已完成并接受。

方法：

```text
1. 不修改模型结构；
2. 增加或确认报告字段：raw p、calibrated p、raw GPD q90/q95、calibrated q95；
3. 增加 validation/test 同口径指标对比；
4. 增加 p reliability bins 的单调性或 rank-correlation 诊断；
5. 将当前 New Round 2 accepted 指标写入优化报告。
```

目标：

```text
1. 保证当前 best 结果可复现；
2. 明确 raw GPD 与 calibrated q95 的区别；
3. 为后续 7 轮建立稳定比较表。
```

接受条件：

```text
不改变核心指标，或所有核心指标与当前 baseline 数值一致到可解释误差范围内。
```

### Round 2：p 条件后校准

状态：已完成并接受。当前采用 `isotonic_auto:decreasing` 作为 p_report calibration。

方法：

```text
1. 在验证集上比较 rate-scale、Platt、temperature、isotonic calibration；
2. p_for_report 与 p_for_tail_extrapolation 继续解耦；
3. 优先选择能改善 reliability bins 的校准方式；
4. 不直接使用 test 信息拟合校准参数。
```

目标：

```text
1. calibrated p_mean 与 empirical risk > u 差距 <= 0.020；
2. p reliability bins 的 empirical exceed 趋势不再反向；
3. q90/q95 calibrated 指标不回退。
```

接受条件：

```text
q95 calibrated ECE <= 0.008；
q95 gap-bin MAE <= 0.040；
q90 ECE <= 0.014；
p reliability bins 明显优于 Round 9 baseline，且满足当前硬约束。
```

### Round 3：raw p 与 tail p 解耦训练

状态：已执行但未接受。`lambda_p_mean_target=20.0, p_mean_target=0.15` 只将 raw p_mean 提升到约 `0.124`，但 q90 ECE 退化到 `0.0342`，违反硬约束。

方法：

```text
1. 保留 p_report calibration；
2. 新增或调整 tail_p / exceedance_p 的训练目标；
3. 使用验证集超阈标签约束 raw p_mean；
4. 避免 batch 级强行拉齐导致 invalid 回升。
```

目标：

```text
1. raw p_mean 从 0.1154 提升到 >= 0.135；
2. raw p_mean 与 empirical risk > u 差距降到 <= 0.035；
3. q90/q95 invalid rate 保持 0。
```

接受条件：

```text
raw p_mean 改善且 q90 ECE <= 0.014；
q95 raw GPD ECE 不劣于 0.0422；
q95 calibrated ECE <= 0.008。
```

### Round 4：raw GPD q95 轻量校准项

状态：已完成并接受。当前固定基线采用 `lambda_tail_quantile=2.0`。

方法：

```text
1. 增加 validation-driven raw q95 exceed calibration objective；
2. 对 q90 和 q95 使用分离权重；
3. 控制 q95 raw quantile 不再被 invalid penalty 推得过高；
4. 不使用全局 test offset。
```

目标：

```text
1. q95 raw GPD empirical exceed 从 0.0078 提升到 >= 0.020；
2. q95 raw GPD ECE 从 0.0422 降到 <= 0.030；
3. q90 ECE 不超过 0.014。
```

接受条件：

```text
raw GPD q95 明显改善；
calibrated q95 ECE <= 0.008；
q95 gap-bin MAE <= 0.040；
q90/q95 invalid rate <= 0.005。
```

### Round 5：条件分箱校准泛化

状态：已执行但未接受。`shrink_gamma=0.0` 使 q95 ECE 退化到 `0.0107`，违反硬约束；当前保留 `shrink_gamma=0.1`。

方法：

```text
1. 对 gap_current、relative speed、TTC/THW/DRAC 派生特征增加分箱诊断；
2. 比较 gap-bin shrink、multi-feature bin shrink、conditional conformal calibration；
3. 优先降低 validation/test 迁移差异，而不是只压低 test MAE。
```

目标：

```text
1. q95 gap-bin MAE <= 0.030；
2. 各 gap bin 不再系统性低估；
3. validation/test q95 calibration gap 可报告且不过大。
```

接受条件：

```text
q95 calibrated ECE <= 0.007；
q95 gap-bin MAE <= 0.0356；
q90 ECE <= 0.014；
不引入明显过拟合迹象。
```

### Round 6：tail head 参数化改进

方法：

```text
1. 尝试 p/u 与 xi/beta 更彻底解耦；
2. 尝试 context-conditioned tail head 或 multi-regime tail head；
3. 保持 xi/beta 有界约束；
4. 不接受 xi/beta 贴边或异常漂移的方案。
```

目标：

```text
1. raw GPD q95 ECE <= 0.025；
2. raw GPD q95 empirical exceed >= 0.025；
3. xi_mean 保持在约 0.05 到 0.15；
4. beta_mean 保持在约 0.06 到 0.14。
```

接受条件：

```text
raw GPD q95、p calibration 或 q95 bin MAE 至少一项净改善；
q90 ECE <= 0.014；
q90/q95 invalid rate <= 0.005。
```

### Round 7：prefix/context 特征增强

方法：

```text
1. 若前 6 轮未充分改善 raw GPD 或 p 条件可靠性，允许重建数据集；
2. 增加 prefix temporal statistics；
3. 增加 gap、relative speed、TTC、THW、DRAC 的统计特征；
4. 重新生成 normalization_stats、feature_schema、canonical_contexts。
```

目标：

```text
1. p reliability bins 的排序质量保持或改善；
2. q95 gap-bin MAE <= 0.030；
3. raw GPD q95 ECE <= 0.025。
```

接受条件：

```text
若重建数据集，必须说明样本数、split、feature_schema 和 normalization 变化；
新数据方案必须优于当前 baseline，且论文可比性需要单独说明。
```

### Round 8：最终选择、回归验证与报告收敛

方法：

```text
1. 在所有接受轮中选择最终 best；
2. 重新运行 best/final 评估；
3. 输出最终对比表；
4. 明确 accepted / rejected rounds；
5. 若最终未优于 New Round 4 accepted baseline，则保留当前 New Round 4 accepted baseline。
```

目标：

```text
1. 最终产物不劣于当前 baseline；
2. 清楚说明是否真正改善 raw GPD、p 条件校准和 q95 条件分箱；
3. 给出下一阶段是否需要更大模型或数据重建的判断。
```

接受条件：

```text
最终 best 至少满足一项主改进目标，且所有硬约束不回退。
```

---

## 硬约束

任一接受轮必须满足：

```text
1. q90 ECE <= 0.014；
2. q90 empirical exceed 位于 [0.075, 0.115]；
3. q90 invalid rate <= 0.005；
4. q95 calibrated ECE <= 0.008；
5. q95 calibrated empirical exceed 位于 [0.040, 0.060]；
6. q95 invalid rate <= 0.005；
7. q95 gap-bin mean abs error <= 0.040；
8. calibrated p_mean 与 empirical risk > u 差距 <= 0.020；
9. xi/beta 分布不贴边、不坍缩、不出现非物理性大幅漂移；
10. best checkpoint 不明显劣于 final checkpoint。
```

---

## 优先优化方向

建议按风险从低到高尝试：

```text
1. 评估与报告固化（已完成）：
   先补齐 raw/calibrated 指标和 validation/test 迁移诊断。

2. p 条件后校准（已完成）：
   使用验证集 Platt、temperature、isotonic 或分箱校准；
   目标不是只拉齐全局 p_mean，而是改善 reliability bins。

3. raw p 与 tail p 解耦：
   calibrated_p_for_report、p_for_tail_extrapolation、raw exceedance head 分开报告和选择。

4. q95 raw GPD 温和训练约束：
   提升 raw GPD q95 empirical exceed，但不能牺牲 q90。

5. 条件化 q95 后校准泛化：
   从单一 gap-bin shrink 发展到多特征条件校准或 conformal calibration。

6. tail head 结构调整：
   p/u 与 xi/beta 解耦，或使用 multi-regime tail head。

7. 数据与特征增强：
   只有当前 6 轮不足以改善 raw GPD/p 条件可靠性时，才重建数据集。
```

已尝试且不建议作为默认方案：

```text
batch 级 lambda_p_rate_cal = 50.0 会导致 q95 invalid 回升和整体退化；
可以保留代码开关，但默认应为 0.0。

threshold head 裸解冻 threshold_lr_multiplier = 0.02 已观察到会显著破坏 risk > u 和 q90 校准；
除非加入强 selection/constraint，否则不应作为默认方案。

全局 q95 additive/conformal offset 可改善 q95 全局 ECE，但会恶化 gap-bin MAE；
若使用后校准，应优先做条件化后校准，而不是全局 offset。

direct q90/q95 heads 曾出现输出贴近 u 或条件分箱退化；
若重新尝试，必须改变初始化、约束和 selection 机制，并单独验证。
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
1. 本轮假设和方法；
2. 修改文件；
3. 是否接受该轮；
4. best_validation stage/epoch/selection_score；
5. q90/q95 calibrated ECE；
6. q90/q95 empirical exceed rate；
7. q90/q95 invalid rate；
8. q95 raw GPD ECE 与 empirical exceed；
9. q95 gap-bin mean abs error；
10. raw p_mean、calibrated p_mean 与 empirical risk > u；
11. p reliability bins；
12. xi/beta mean 与分位数；
13. best checkpoint 与 final checkpoint 对比；
14. validation/test 校准迁移诊断；
15. 若重建数据集，记录 rebuild 命令、输入数据、样本数、split 变化、feature_schema 变化与 normalization 变化。
```

---

## 迭代预算

原计划固定执行 8 轮优化；New Round 1/2/4 已完成并接受，New Round 3/5 已拒绝并回滚。后续固定执行剩余 3 轮优化，从 Round 6 开始。

每轮要求：

```text
1. 只针对一个主要问题做可解释修改；
2. 修改前说明本轮假设；
3. 如涉及数据或特征，先运行 dataset rebuild；
4. 修改后运行训练和 best/final 评估；
5. 与当前 New Round 4 accepted baseline 和上一接受轮比较；
6. 如果 q90/q95 invalid 明显回升，优先修复；
7. 如果 q90 ECE、risk > u、q95 calibrated ECE 或 q95 分箱误差明显恶化，优先回滚或增加约束；
8. 未接受轮不得覆盖当前 best 产物；
9. Round 8 必须收敛报告，不再开启新的大尺度实验。
```

---

## 停止条件

满足以下任一条件后停止：

```text
1. 完成剩余 3 轮优化，即总计完成 8 轮优化。

2. 提前达到以下强目标，并完成至少 3 轮验证：
   - q95 calibrated empirical exceed 位于 [0.045, 0.055]；
   - q95 calibrated ECE <= 0.005；
   - q95 gap-bin mean abs error <= 0.030；
   - q90 ECE <= 0.012；
   - q90/q95 invalid rate 均 <= 0.005；
   - calibrated p_mean 与 empirical risk > u 的绝对差距 <= 0.015；
   - p reliability Spearman >= 0.8；
   - p reliability monotonic violations = 0；
   - raw p_mean 与 empirical risk > u 的绝对差距 <= 0.035；
   - q95 raw GPD ECE <= 0.030。

3. 连续 3 轮核心指标提升不足，且已经尝试过至少一种 p 条件校准方案和一种 raw GPD 改进方案。

4. 后续改动已经超出 DeepEVT 条件尾部分布模型本身，例如闭环 ADS 接口、
   diffusion planner 联调、MATLAB 场景生成接口重构等；此时停止当前 DeepEVT 校准任务。

5. 连续 2 轮大尺度方案导致硬约束失效，且无法通过 selection 或 calibration 修复。
```

---

## 最终交付

最终输出或更新 `deepevt_optimization_report.md`，至少包含：

```text
1. New Round 4 accepted 固定基线指标；
2. 当前代码固化说明；
3. 最终训练和评估命令；
4. 最终 best/final checkpoint 路径；
5. 8 轮中每轮假设、方法、修改内容、修改文件、是否接受，其中 New Round 1/2 已完成并接受；
6. q90/q95 calibrated ECE 对比；
7. q90/q95 empirical exceed rate 对比；
8. q90/q95 invalid rate 对比；
9. q95 raw GPD ECE 与 empirical exceed 对比；
10. q95 条件分箱误差对比；
11. raw p、calibrated p 与 empirical risk > u 对比；
12. p reliability bins 对比；
13. xi/beta 分布对比；
14. best checkpoint 与 final checkpoint 对比；
15. validation/test 校准迁移诊断；
16. 是否达到停止条件；
17. 未解决问题；
18. 下一阶段建议；
19. 若发生模型结构改动，说明新旧结构差异、参数约束、兼容性与推理接口变化；
20. 若发生数据集重建，说明 rebuild 原因、命令、样本数变化、split 变化、feature schema 变化、normalization 变化，以及是否影响论文结果可比性。
```

---

## 一句话目标

以当前 New Round 4 tail-quantile accepted 为固定基线，执行剩余 3 轮受控优化，在不牺牲 q90、不破坏 q95 calibrated 表现、条件分箱质量和 p reliability 单调性的前提下，重点修复 raw GPD q95 过保守与 raw p 低估，使 DeepEVT following 模型从“后校准与 p_report 可报告”推进到“raw tail、更可信 raw p、条件校准迁移都更稳”的可报告条件尾部分布模型。
