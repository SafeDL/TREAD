# process_highD：highD 事件与长尾空间

`process_highD/` 从 highD 原始 CSV 生成 following/cut-in 事件，缓存风险与
context，拟合 peak-level EVT，并输出 subset simulation 使用的长尾 context 空间。

## 运行顺序

following 主流程：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python process_highD/scripts/fit_following_peak_evt.py
conda run -n tread python process_highD/scripts/estimate_following_exposure.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py
```

`build_natural_dataset.py` 不带参数时默认构建 following diffusion 数据集。训练入口按
事件类型拆分；cut-in 分支需要先生成 cut-in score/context cache，再显式使用 cut-in
配置构建数据集：

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py \
  --config diffusion/scripts/configs/natural_cutin.yaml
conda run -n tread python diffusion/scripts/train_cutin_diffusion.py
conda run -n tread python process_highD/scripts/fit_cutin_peak_evt.py
conda run -n tread python process_highD/scripts/estimate_cutin_exposure.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py --scenario cut_in
```

可选回放：

```bash
conda run -n tread python process_highD/scripts/play_highd_events.py
```

## 主要文件

```text
process_highD/scripts/configs/highd_default.yaml
process_highD/scripts/extract_highd_events.py
process_highD/scripts/build_natural_dataset.py
process_highD/scripts/fit_following_peak_evt.py
process_highD/scripts/fit_cutin_peak_evt.py
process_highD/scripts/estimate_following_exposure.py
process_highD/scripts/estimate_cutin_exposure.py
process_highD/scripts/select_tail_contexts.py
process_highD/src/event_extraction.py
process_highD/src/loader.py
process_highD/src/preprocess.py
```

## 主要输出

```text
results/highd_events/events.csv
results/highd_events/following_event_scores.csv
results/highd_events/following_event_contexts.npz
results/highd_events/cutin_event_scores.csv
results/highd_events/cutin_event_contexts.npz

results/highd_following_tail/evt/longitudinal_peak_evt_model.json
results/highd_following_tail/evt/longitudinal_peak_evt_summary.json
results/highd_following_tail/exposure/highd_exposure_summary.json
results/highd_following_tail/exposure/highd_independent_tail_peaks.csv
results/highd_following_tail/contexts/tail_contexts.npz
results/highd_following_tail/contexts/tail_context_summary.json

results/highd_cutin_tail/evt/cutin_peak_evt_model.json
results/highd_cutin_tail/evt/cutin_peak_evt_summary.json
results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json
results/highd_cutin_tail/exposure/highd_independent_tail_peaks.csv
results/highd_cutin_tail/contexts/tail_contexts.npz
results/highd_cutin_tail/contexts/tail_context_summary.json
```

cut-in 输出只有在 cut-in cache 和 EVT 分支运行后才会出现。

## 风险口径

- `extract_highd_events.py` 只做自然事件抽取和质量过滤，不用风险分数筛选候选事件。
- `fit_following_peak_evt.py` 在 decluster 后的 following `Y_long` peaks 上拟合 POT/GPD。
- `fit_cutin_peak_evt.py` 在 decluster 后的 cut-in `Y_cutin` peaks 上拟合 POT/GPD。
- `estimate_following_exposure.py` 计算 following exposure、tail peak rate 和 highD 人类驾驶基线。
- `estimate_cutin_exposure.py` 使用 highD 全车辆里程/时长计算 cut-in tail peak rate 和人类驾驶基线。

following 的统一风险尺度是 `S_EVT(Y_long)`；cut-in 的统一风险尺度是
`S_EVT(Y_cutin)`。

## Tail Contexts

`select_tail_contexts.py` 默认不是只保存有限 empirical peaks，而是在全部
declustered tail peaks 上构造平滑长尾测试空间：

```text
context_source = independent_tail_peaks
empirical_context_limit = null
context_generation_method = tail_feature_kde_knn
include_empirical_contexts = true
num_synthetic_contexts = 1000
```

输出包含两类 context：

```text
highd_independent_tail_peak    empirical highD independent tail peak context
highd_tail_feature_kde_knn     tail feature 微弱扰动后的重构 context
```

低维 tail feature 来自第 10 帧历史状态：

```text
log_initial_gap, ego_speed, adv_speed, closing_speed,
ego_accel, adv_accel, lateral_offset
```

following 和 cut-in 使用同一套微扰机制。cut-in 中第二辆车按 adversarial/target
vehicle 处理，速度扰动按速度向量缩放，保留横向速度信息。

输出 metadata 用于追踪微扰来源：

```text
synthetic_context, context_model_method, base_context_index,
base_event_id, context_feature_distance
```

若只需要经验长尾峰集合，可把 `context_generation_method` 改为 `empirical`，并保留
`empirical_context_limit = null`。
