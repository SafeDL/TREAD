# process_highD：highD 典型驾驶事件初筛

`process_highD/` 负责从 highD 原始轨迹中抽取两类自然驾驶交互事件：
`following` 与 `cut_in`。事件抽取本身只做初筛和质量审计；后续脚本会基于
有效 following 事件拟合纵向风险 EVT 模型，并为 subset simulation 提供共享
风险尺度和 highD tail contexts。

## 当前实现状态

已实现并可通过脚本串联运行的能力：

- 读取 highD 三类 CSV：`XX_tracks.csv`、`XX_tracksMeta.csv`、`XX_recordingMeta.csv`
- 将行驶方向统一到 `+x`，并将 highD top-left bbox 坐标转换为车辆几何中心
- 标记异常轨迹帧：加速度、jerk、位置跳变、尺寸异常、低速
- 按配置重采样 recording
- 抽取稳定跟驰片段与切入事件
- 输出 `events.csv`、中间审计 CSV 和质量报告
- 一次性缓存 valid following events 的 `y_long`、风险分量和 `context_states`
- 将有效事件渲染为 MP4 回放
- 从全部有效 highD following events 提取事件级 `y_long`，用 POT/GPD 拟合
  自然驾驶纵向风险尾部分布

## 环境与数据

```bash
conda activate tread
```

默认配置文件：

```text
process_highD/scripts/configs/highd_default.yaml
```

默认 highD 原始数据目录相对于配置文件解析为：

```text
../../../highD_dataset/Matlab/data
```

目录中应包含：

```text
XX_tracks.csv
XX_tracksMeta.csv
XX_recordingMeta.csv
```

要处理的 recording 范围只由配置中的 `recordings.include` 和
`recordings.exclude` 控制；`extract_highd_events.py` 没有单独的
`--recordings` 参数。

## 快速开始

以下命令可从 TREAD 仓库根目录运行。

```bash
# 1. 抽取 following / cut-in 事件，并生成质量报告
python process_highD/scripts/extract_highd_events.py

# 2. 渲染事件回放 MP4
python process_highD/scripts/play_highd_events.py

# 3. 构建 diffusion 自然先验使用的固定长度数据集
python process_highD/scripts/build_natural_dataset.py

# 4. 拟合 highD 纵向风险 EVT 模型
python process_highD/scripts/fit_longitudinal_evt.py

# 5. 选择 highD tail contexts
python process_highD/scripts/select_tail_contexts.py
```

`play_highd_events.py` 当前只导出单个 MP4 文件，依赖本机可用的 ffmpeg。
默认渲染 `cut_in`，如需改变事件类型、帧数或输出名，修改脚本顶部 `SCRIPT_DEFAULTS`。

## 输出文件

默认输出目录相对于配置文件解析为 `../../../results/highd_events`：

```text
results/highd_events/
├── events.csv
├── candidate_events.csv
├── invalid_events.csv
├── following_event_scores.csv
├── following_event_contexts.npz
├── following_event_cache_summary.json
├── quality_report.json
└── figures/event_playbacks/events_<event_type>.mp4
```

语义约定：

- `events.csv` 是本阶段的主产物，包含事件元数据、有效性标记和过滤原因。
- `candidate_events.csv` 只包含 `is_valid=True` 的事件；
  `invalid_events.csv` 只包含无效事件。
- `quality_report.json` 与事件回放是可再生成的质量诊断产物。
- `following_event_scores.csv` 和 `following_event_contexts.npz` 在
  `extract_highd_events.py` 第一次读取 raw highD 时同步生成；后续 EVT 拟合和
  tail context 选择必须读取这两个缓存；缓存缺失会直接报错，避免重复重建 recording。
- `fit_longitudinal_evt.py` 输出 highD 自然驾驶纵向风险的 POT/GPD 模型；
  EVT 估计的是 `P_highD(Y_long > y)`，不是 ADS collision probability。
  同时输出 survival 拟合点表、阈值稳定性表、return level 置信区间和诊断图。
- `scripts/select_tail_contexts.py` 读取 `following_event_contexts.npz`，
  并要求 EVT 模型已经存在，然后输出 `risk_score = S_EVT(y_long)`。
  该筛选不使用 RSS；默认按 EVT 拟合得到的 POT 阈值 `u` 保留
  `y_long > u` 的 tail events。当前为便于 subset 测试，默认从 tail 集合中
  按固定随机种子抽取 `500` 个 context；若将 `num_contexts` 改为 `0`，
  则保留全部 tail contexts。输出 metadata 中的 EVT return level 由脚本内
  `evt_return_period` 指定，应和 subset 实验配置保持一致。
- `fit_longitudinal_evt.py` 和 `select_tail_contexts.py` 不回退 raw highD
  重建；所有 event suffix 与 RSS-free `y_long` 都由 `extract_highd_events.py`
  通过 `utils/highd_longitudinal.py` 一次性缓存。

## 代码结构

```text
process_highD/
├── src/
│   ├── loader.py              # highD CSV 读取与 HighDRecording 查询
│   ├── preprocess.py          # 坐标中心化、方向统一、异常帧标记、重采样
│   ├── lane_utils.py          # 车道线解析、相邻车道判断、换道检测
│   ├── event_extraction.py    # following 与 cut-in 抽取
│   ├── filtering.py           # EventRecord 列表转 DataFrame
│   ├── quality_check.py       # quality_report.json
│   ├── schema.py              # EventRecord dataclass
│   └── io_utils.py            # YAML / JSON / 路径 / recording id 工具
└── scripts/
    ├── extract_highd_events.py
    ├── build_natural_dataset.py
    ├── fit_longitudinal_evt.py
    ├── select_tail_contexts.py
    ├── play_highd_events.py
    └── configs/highd_default.yaml
```

## 抽取流水线

`extract_highd_events.py` 对每个 recording 执行：

```text
load_recording()
  -> normalize_driving_direction()
  -> filter_abnormal_tracks()
  -> resample_recording()
  -> extract_following_segments()
  -> extract_cutin_events()
  -> events_to_dataframe()
```

### Following

`extract_following_segments()` 基于连续相同 `precedingId` 分段，并筛选：

- ego 与 lead 不是 truck
- 公共帧数满足 `filters.min_segment_seconds` 或 `following.min_same_preceding_steps`
- segment 内 ego 不换道，ego/lead 同车道比例至少 80%
- ego 与 lead 没有 `_abnormal=True` 帧
- median gap 大于 `filters.min_positive_gap`

默认 anchor 为完整跟驰片段中心。

### Cut-In

`extract_cutin_events()` 遍历所有小汽车的相邻车道变化，并筛选：

- 换道前后车道稳定，且 `from_lane` / `to_lane` 相邻
- 优先在稳定进入目标车道后的帧和 cross frame 使用 `followingId`
  匹配被切入 ego；若不满足条件，再在这两个时刻于目标车道后方寻找最近小汽车。
- cross frame 必须在 ego 与 target 公共轨迹中
- cross frame 后 target 与 ego 同车道比例至少 70%
- post window 中 target 必须是 ego 前方最近同车道车辆，默认检查 cross frame 后 0.6 秒
- post median gap 位于 `(0, max_post_cutin_gap]`
- ego 与 target 没有 `_abnormal=True` 帧

默认 anchor 为 `cross_frame`。

## 实现完整性与正确性 Review

整体判断：`process_highD` 已经实现了从 highD 原始 CSV 到事件级候选数据集的主流程，
并且模块边界清晰。事件抽取仅使用语义、轨迹质量和必要的几何关系；风险变量和
EVT 模型由后续脚本从有效事件中重建计算。

已确认较完整的部分：

- `loader.py`：文件存在性检查、无效 ID `0 -> -1`、MultiIndex 构建、
  lane markings 解析完整。
- `event_extraction.py`：following 和 cut-in 都没有用危险分数做候选过滤，
  保留了自然暴露分布。
- `quality_check.py`：围绕 `events.csv` 生成可再生质量诊断产物。

需要注意的实现边界：

- `filter_abnormal_tracks()` 记录了 `_discontinuous_ids`，
  但没有把不连续车辆帧写入 `_abnormal=True`。
  Phase 2 会用缺帧检查再次过滤，第一阶段 `events.csv` 仍可能保留这类事件。
- `normalize_driving_direction()` 先执行坐标中心化，
  再按需翻转 `drivingDirection == 1` 车辆，因此所有 recording 都会得到中心坐标。
- `play_highd_events.py` 使用中心坐标作为视窗中心。
- 已清理当前代码路径中未使用的可视化辅助函数、批量加载函数、
  熵权法函数和未调用的过滤函数，保留脚本实际使用的入口。

## 与后续模块的接口

`diffusion/` 读取本阶段的事件表来构建自然先验数据集：

```text
results/highd_events/events.csv
results/highd_events/following_event_scores.csv
results/highd_events/following_event_contexts.npz
```

`subset/` 读取 EVT 模型和 highD tail contexts 来执行闭环极端风险概率估计：

```text
results/highd_evt/following/longitudinal_evt_model.json
results/highd_tail_contexts/following/tail_contexts.npz
```

这些脚本会回到 raw highD 中重建固定长度窗口或 event suffix。因此
`events.csv` 中至少需要保留：

- `event_id`, `event_type`, `recording_id`
- `ego_id`, `target_id`
- `start_frame`, `end_frame`, `anchor_frame`
- cut-in 专用的 `cross_frame`, `source_lane`, `target_lane`, `cutin_start_frame`, `cutin_end_frame`
- `is_valid`

## EVT 诊断输出

`fit_longitudinal_evt.py` 默认写入：

```text
results/highd_evt/following/
├── longitudinal_evt_model.json
├── longitudinal_evt_model.npz
├── longitudinal_evt_scores.csv
├── longitudinal_evt_summary.json
├── threshold_stability.csv
├── evt_survival_diagnostic_points.csv
└── figures/
    ├── evt_y_long_histogram.png
    ├── evt_survival_fit.png
    ├── evt_threshold_stability.png
    └── evt_return_levels.png
```

其中 `longitudinal_evt_summary.json` 包含 `u, xi, beta, z20, z50, z100`、
return level CI、exceedance 数量、GPD excess CDF RMSE 等诊断值。图用于快速检查：
`y_long` 分布和阈值位置、经验 survival 与 EVT tail 是否贴合、阈值稳定性，以及
return level 外推和置信区间。
