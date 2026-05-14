# DeepEVT 训练稳定性与尾部校准优化任务 Goal

## 任务名称

DeepEVT following 场景下的训练稳定性、尾部校准与评估诊断优化。

---

## 任务说明

本任务面向 `tread_deepevt` 模块中 following 场景下的 DeepEVT 条件极值尾部风险模型，重点改善训练稳定性、checkpoint 选择机制、尾部参数合理性和 q90/q95 条件校准质量。

conda activate jzm 本身确实有 CUDA, 它是 NVIDIA GeForce RTX 4090 D。请将训练/评估用沙箱外执行。

当前主训练入口为：

```bash
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

当前主评估入口为：

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

必要时可先重新构建数据集：

```bash
python tread_deepevt/scripts/01_build_deepevt_dataset.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

---

## 背景

DeepEVT 的目标是：根据初始交通场景条件预测未来固定窗口内 `risk_score` 的条件尾部分布。模型输出条件阈值 `u`、超阈概率 `p`、GPD 参数 `xi/beta`，并进一步计算高分位风险 `q90/q95` 和 Expected Shortfall `ES95`。

当前实现采用三阶段训练：

```text
Stage 1: threshold pretrain
Stage 2: tail train
Stage 3: end-to-end finetune
```

其中：

```text
Stage 1 主要训练条件阈值 u；
Stage 2 主要训练超阈概率 p 和 GPD 尾部参数 xi/beta；
Stage 3 以较小学习率联合微调整体模型。
```

当前 following 训练结果显示的主要问题包括：

```text
DeepEVT 的 u 作为 85% 条件阈值还算接近：test 上 risk > u 是 16.6%，目标是 15%。真正坏掉的是尾部外推：

p_mean = 0.0503，但实际 risk > u 是 0.1658。
q90 时 p <= 0.1 的 invalid rate 是 100%，所以 q90 基本退化成 u。
q95 时 invalid rate 是 55.4%，大量样本也几乎退化成 u。
beta_mean = 0.141 偏小，尾部残差尺度也偏保守。
```

---

## 总目标

在不改变 DeepEVT 基本建模含义、不改变 `risk_score` 定义、不删除 EVT 关键功能的前提下，提升 following DeepEVT 的训练稳定性和尾部校准质量。

本任务的优化重点不是单纯降低 `loss_total`，而是同时改善以下方面：

```text
1. best checkpoint 选择机制
2. early stopping 或训练停止机制
3. q90/q95 ECE
4. q95 条件分箱误差
5. xi/beta/p 参数合理性
6. invalid q90/q95 rate
7. Stage 2/3 验证集退化问题
```

---

## 基线测量

在进行任何修改前，必须先运行当前未修改代码，得到 baseline。

需要保存或记录以下文件：

```text
data/deepevt/following/training_history.json
data/deepevt/following/eval_report.json
data/deepevt/following/runs/
```

---

## 迭代预算

最多执行 10 轮优化。

每轮必须满足以下要求：

```text
1. 只针对一个主要问题做小范围修改。
2. 修改前说明本轮假设。
3. 修改后运行训练和评估命令。
4. 记录本轮指标。
5. 与 baseline 和上一轮进行比较。
6. 如果训练失败、评估失败或指标明显恶化，优先修复。
7. 如果无法快速修复，则回滚本轮。
```

---

## 每轮验证命令

每轮修改后至少运行：

```bash
python tread_deepevt/scripts/02_train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml
```

如果实现了 best checkpoint，评估时还必须额外运行或确认：

```bash
python tread_deepevt/scripts/03_evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/best_model.pt
```

---

## 停止条件

满足以下任一条件后停止优化：

```text
1. 完成 10 轮优化。

2. 达到以下组合目标：
   - q95 ECE 不劣于 baseline；
   - q90 ECE 改善至少 10%；
   - q95 tail_quantile_bins 平均 abs error 改善至少 10%；
   - best checkpoint 明显优于 final checkpoint；
   - eval_report.json 包含 invalid rate、xi/beta/p 分布诊断。

3. 连续 2 轮核心指标提升不足。

4. 后续改动需要大规模重构模型结构，例如 temporal prefix encoder、闭环 ADS 接口等；
   此时停止当前稳定化任务，并将其列为下一阶段需求。
```

---

## 最终交付

最终输出一份 `deepevt_optimization_report.md`，至少包含以下内容：

```text
1. baseline 命令和 baseline 指标表
2. 最终使用的训练命令和评估命令
3. 最终 checkpoint 路径
4. best checkpoint 与 final checkpoint 的指标对比
5. 每轮优化假设、修改内容、修改文件、指标变化
6. q90/q95 ECE 对比
7. q90/q95 empirical exceed rate 对比
8. q95 tail_quantile_bins 对比
9. xi_mean、xi 分位数、beta_mean、beta 分位数、p_mean 对比
10. q90/q95 invalid rate 对比
11. Stage 2/3 val loss 是否退化的对比
12. 是否达到停止条件
13. 未解决问题
14. 下一阶段建议
```
---

## 一句话目标

在保持当前 DeepEVT 条件极值建模框架和 risk_score 定义不变的前提下，通过 best checkpoint、early stopping、评估诊断增强和有限超参数优化，改善 following 场景下 q90/q95 尾部校准、条件分箱低估、xi 贴边和 p_mean 偏低等问题，并输出完整的可复现实验报告。
