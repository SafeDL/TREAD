# TREAD Phase 2: DeepEVT Direct Tail Quantile Model

`tread_deepevt` 是 TREAD 的第二阶段。它读取 `tread_highd` 初筛出的
`data/highd_events/events.csv`，回到 highD 原始轨迹中重建训练窗口，并训练
一个 prefix-conditioned direct quantile 模型：

```text
prefix_states[0:12] -> q85, q90, q95
```

当前默认目标不是 GPD/POT 外推，也不训练 `p/xi/beta`。模型的主用途是：

- 用 raw direct q85/q90/q95 做高风险测试场景排序；
- 生成 `tail_conditions.csv`，供 diffusion / MATLAB / RoadRunner 消费；
- 在 DeepEVT 真实 train/val/test 数据集上生成风险分布和 tail survival 图。

## 输入

先运行第一阶段事件初筛：

```bash
python tread_highd/scripts/extract_highd_events.py \
  --config tread_highd/scripts/configs/highd_default.yaml
```

DeepEVT 默认读取：

```text
data/highd_events/events.csv
highD-dataset/Matlab/data
```

## 快速开始

```bash
# 1. 构建 following DeepEVT 数据集
python tread_deepevt/scripts/build_deepevt_dataset.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 2. 完整训练 direct q85/q90/q95；不 early stop
python tread_deepevt/scripts/train_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml

# 3. 评估 raw direct quantiles
python tread_deepevt/scripts/evaluate_deepevt.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/best_model.pt

# 4. 导出给下游生成/仿真的条件文件
python tread_deepevt/scripts/export_tail_conditions.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --checkpoint data/deepevt/following/best_model.pt
```

## 稳健性验证

长尾样本数量有限，单次 recording split 只能作为开发实验。更接近真实部署的验证应同时运行多次 recording split 和 leave-location-out：

```bash
# 多次 recording-level split + 自动 leave-location-out
python tread_deepevt/scripts/run_deepevt_protocols.py \
  --config tread_deepevt/scripts/configs/deepevt_following.yaml \
  --protocols repeated,leave-location \
  --stages build,train,evaluate
```

默认输出到：

```text
data/deepevt/following_protocols/
```

每个子目录都是独立实验，包含自己的 `dataset.npz`、normalization stats、
`best_model.pt` 和 `eval_report.json`。总表写入：

```text
data/deepevt/following_protocols/protocol_summary.json
```

## 输出

默认输出目录：

```text
data/deepevt/following/
```

主要产物：

```text
dataset.npz
feature_schema.json
normalization_stats.json
train_val_test_split.json
best_model.pt
final_model.pt
training_history.json
checkpoint_summary.json
runs/
figures/training_key_metrics.png
eval_report.json
figures/quantile_calibration_comparison.png
tail_conditions.csv
```

重要语义：

- `best_model.pt`：完整训练过程中 validation selection 最优的 checkpoint。
- `final_model.pt`：最后一个 epoch 的 checkpoint，用于观察完整训练末端结果。
- `training_history.json`：压缩后的全 epoch 关键训练指标，只保留 `loss_q`、`selection_score`、各分位数 ECE/empirical exceed 等监控信号。
- `runs/`：极简 TensorBoard 日志，只记录 `loss_q/train`、`loss_q/val`、`selection_score/val` 和 `q85/q90/q95_ece`。
- `figures/training_key_metrics.png`：训练期唯一默认关键图，集中展示 train/val quantile loss、validation selection score 和 q85/q90/q95 ECE。
- `eval_report.json`：包含 raw direct quantile、global empirical baseline、ranking AUC / top-k enrichment。
- `figures/quantile_calibration_comparison.png`：在 test split 上同时比较 q85/q90/q95 的 raw direct quantile 与 global empirical baseline 校准效果。
- `tail_conditions.csv`：只导出下游需要的事件定位、ego 坐标系 metadata、`risk_score` 离线参考，以及 raw direct `q85/q90/q95_raw_pred` 与同值的 `q85/q90/q95_pred`。

## 模型与损失

当前模型是：

```text
ShortHistorySceneTransformer
  + ego-target interaction token
  + monotone direct quantile head
```

quantile head 使用非负增量保证：

```text
q85 <= q90 <= q95
```

训练目标：

```text
multi_quantile_pinball_loss(q85, q90, q95)
+ lambda_cal  * soft_exceedance_calibration_loss
+ lambda_rank * pairwise_ranking_loss
```

`lambda_rank` 用来直接强化“把更危险测试场景排在前面”的能力。
