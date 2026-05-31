# process_highD：highD 事件与长尾空间

`process_highD/` 从 highD 原始 CSV 生成 following/cut-in 事件，缓存 following 风险与 context，拟合 highD peak EVT，并输出 subset simulation 使用的长尾 context 空间。

## 运行顺序

```bash
conda run -n tread python process_highD/scripts/extract_highd_events.py
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python process_highD/scripts/fit_longitudinal_peak_evt.py
conda run -n tread python process_highD/scripts/estimate_highd_exposure.py
conda run -n tread python process_highD/scripts/select_tail_contexts.py
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
process_highD/scripts/fit_longitudinal_peak_evt.py
process_highD/scripts/estimate_highd_exposure.py
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
results/highd_following_tail/evt/longitudinal_peak_evt_model.json
results/highd_following_tail/evt/longitudinal_peak_evt_summary.json
results/highd_following_tail/exposure/highd_exposure_summary.json
results/highd_following_tail/exposure/highd_independent_tail_peaks.csv
results/highd_following_tail/contexts/tail_contexts.npz
results/highd_following_tail/contexts/tail_context_summary.json
```

## 口径

- `extract_highd_events.py` 只做自然事件抽取和质量过滤，不用风险分数筛选候选事件。
- `fit_longitudinal_peak_evt.py` 在 decluster 后的 following `Y_long` peaks 上拟合 POT/GPD。
- `estimate_highd_exposure.py` 计算 following exposure、tail peak rate 和 highD 人类驾驶基线。
- `select_tail_contexts.py` 默认使用全部 `independent_tail_peaks`，即 `num_contexts = 0`。

`tail_contexts.npz` 是 subset simulation 的测试空间输入；`longitudinal_peak_evt_model.json` 提供统一风险尺度 `S_EVT(Y_long)`。
