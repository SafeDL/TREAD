# TREAD highD 第一阶段：交互事件初筛

`tread_highd` 负责从 highD 原始轨迹中抽取两类自然驾驶交互事件：
`following` 与 `cut_in`。本阶段只做事件级初筛和质量审计；固定长度训练窗口、
数据长尾可视化、分位数建模和 calibration 由 `tread_deepevt` 继续完成。

## 当前实现状态

已实现并可通过脚本串联运行的能力：

- 读取 highD 三类 CSV：`XX_tracks.csv`、`XX_tracksMeta.csv`、`XX_recordingMeta.csv`
- 将行驶方向统一到 `+x`，并将 highD top-left bbox 坐标转换为车辆几何中心
- 标记异常轨迹帧：加速度、jerk、位置跳变、尺寸异常、低速
- 按配置重采样 recording
- 抽取稳定跟驰片段与切入事件
- 计算 `TTC`、`THW`、`DRAC`、severity 与综合 `risk_score`
- 输出 `events.csv`、中间审计 CSV 和质量报告
- 将有效事件渲染为 MP4 回放

## 环境与数据

```bash
conda activate jzm
```

默认配置文件：

```text
tread_highd/scripts/configs/highd_default.yaml
```

默认 highD 原始数据目录相对于配置文件解析为：

```text
../../../highD-dataset/Matlab/data
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
# 1. 抽取 following / cut-in 事件
python tread_highd/scripts/extract_highd_events.py \
  --config tread_highd/scripts/configs/highd_default.yaml

# 2. 渲染事件回放 MP4
python tread_highd/scripts/play_highd_events.py \
  --config tread_highd/scripts/configs/highd_default.yaml \
  --event_type cut_in \
  --max_events 50

# 3. 生成质量报告
python tread_highd/scripts/generate_quality_report.py \
  --config tread_highd/scripts/configs/highd_default.yaml
```

`play_highd_events.py` 当前只导出单个 MP4 文件，依赖本机可用的 ffmpeg。
默认 `--event_type cut_in`，可显式传入 `following` 或 `all`。

## 输出文件

默认输出目录相对于配置文件解析为 `../../../data/highd_events`：

```text
data/highd_events/
├── events.csv
├── intermediate/
│   ├── candidate_events.csv
│   └── invalid_events.csv
├── quality_report.json
└── figures/event_playbacks/events_<event_type>.mp4
```

语义约定：

- `events.csv` 是本阶段的主产物，包含事件元数据、风险指标、有效性标记和过滤原因。
- `intermediate/candidate_events.csv` 只包含 `is_valid=True` 的事件；`invalid_events.csv` 只包含无效事件。
- `quality_report.json` 与事件回放是可再生成的质量诊断产物。
- `risk_score`、`ttc_severity`、`thw_severity`、`drac_severity` 是描述性风险指标，不是 EVT 标签。
- 长尾分布图、survival 曲线和 tail diagnostics 不在本阶段生成；这些诊断必须基于 `tread_deepevt` 真正训练用的 train/val/test 数据集生成。

## 代码结构

```text
tread_highd/
├── src/
│   ├── loader.py              # highD CSV 读取与 HighDRecording 查询
│   ├── preprocess.py          # 坐标中心化、方向统一、异常帧标记、重采样
│   ├── lane_utils.py          # 车道线解析、相邻车道判断、换道检测
│   ├── risk_metrics.py        # gap / TTC / THW / DRAC / risk_score
│   ├── event_extraction.py    # following 与 cut-in 抽取
│   ├── filtering.py           # EventRecord 列表转 DataFrame
│   ├── quality_check.py       # quality_report.json
│   ├── schema.py              # EventRecord dataclass
│   └── io_utils.py            # YAML / JSON / 路径 / recording id 工具
└── scripts/
    ├── extract_highd_events.py
    ├── play_highd_events.py
    ├── generate_quality_report.py
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

默认 anchor 为完整跟驰片段中心；也支持 `min_ttc`、`max_drac` 和 `risk`。

### Cut-In

`extract_cutin_events()` 遍历所有小汽车的相邻车道变化，并筛选：

- 换道前后车道稳定，且 `from_lane` / `to_lane` 相邻
- 优先在稳定进入目标车道后的帧和 cross frame 使用 `followingId` 匹配被切入 ego，失败时在这两个时刻于目标车道后方寻找最近小汽车
- cross frame 必须在 ego 与 target 公共轨迹中
- cross frame 后 target 与 ego 同车道比例至少 70%
- post window 中 target 必须是 ego 前方最近同车道车辆，默认检查 cross frame 后 0.6 秒
- post median gap 位于 `(0, max_post_cutin_gap]`
- ego 与 target 没有 `_abnormal=True` 帧

默认 anchor 为 `cross_frame`。

## 风险指标

基础物理量：

```text
gap  = x_target - x_ego - 0.5 * (length_target + length_ego)
TTC  = gap / (ego_vx - target_vx), only when closing > 0 and gap > 0
THW  = gap / ego_vx, only when ego_vx > 0 and gap > 0
DRAC = closing^2 / (2 * gap), only when closing > 0 and gap > 0
```

danger-oriented severity：

```text
ttc_severity  = 1 / (min_ttc + eps)
thw_severity  = 1 / (min_thw + eps)
drac_severity = max_drac
```

综合风险：

```text
S(t) = w_ttc / (TTC(t) + eps) + w_thw / (THW(t) + eps) + w_drac * DRAC(t)
R    = [logsumexp(lambda * S(t)) - log(N)] / lambda
```

其中 `N` 是有效风险帧数。`cut_in` 的第一阶段风险窗口从 `cross_frame`
开始，避免切入前 target 位于 ego 后方时产生负 gap 伪尾部。

## 实现完整性与正确性 Review

整体判断：`tread_highd` 已经实现了从 highD 原始 CSV 到事件级风险数据集的主流程，
并且模块边界清晰。风险计算保持 danger-oriented 语义，cut-in 风险窗口从
`cross_frame` 开始，这是当前实现里最关键、也最正确的一点。

已确认较完整的部分：

- `loader.py`：文件存在性检查、无效 ID `0 -> -1`、MultiIndex 构建、lane markings 解析完整。
- `event_extraction.py`：following 和 cut-in 都没有用风险分数做候选过滤，保留了自然暴露分布。
- `risk_metrics.py`：TTC/THW/DRAC 对无效几何关系做安全处理，`risk_score` 做长度归一化 soft maximum。
- `quality_check.py`：围绕 `events.csv` 生成可再生质量诊断产物。

需要注意的实现边界：

- `filter_abnormal_tracks()` 记录了 `_discontinuous_ids`，但没有把不连续车辆帧写入 `_abnormal=True`。Phase 2 会用缺帧检查再次过滤，第一阶段 `events.csv` 仍可能保留这类事件。
- `normalize_driving_direction()` 先执行坐标中心化，再按需翻转 `drivingDirection == 1` 车辆，因此即使某个 recording 不需要方向翻转，也会得到中心坐标。
- `play_highd_events.py` 使用中心坐标作为视窗中心；旧实现中额外加 `width / 2` 的偏移已移除。
- 已清理当前代码路径中未使用的可视化辅助函数、批量加载函数、熵权法函数和未调用的过滤函数，保留脚本实际使用的入口。

## 与 DeepEVT 的接口

`tread_deepevt` 读取本阶段的：

```text
data/highd_events/events.csv
```

并回到 raw highD 中重建固定长度窗口。因此 `events.csv` 中至少需要保留：

- `event_id`, `event_type`, `recording_id`
- `ego_id`, `target_id`
- `start_frame`, `end_frame`, `anchor_frame`
- cut-in 专用的 `cross_frame`, `source_lane`, `target_lane`, `cutin_start_frame`, `cutin_end_frame`
- `is_valid`
- 风险字段：`risk_score`, `min_ttc`, `min_thw`, `max_drac`, `risk_window_start_frame`, `risk_window_end_frame`
