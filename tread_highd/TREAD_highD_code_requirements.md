# TREAD 第一阶段代码需求文档：highD 尾部风险事件数据集构建器

> Project codename: **TREAD**  
> Full name: **Tail-Risk Extreme-value-Aware Diffusion for Naturalistic Adversarial Trajectory Generation**  
> 第一阶段实现: 新建一个文件夹，将这一阶段的代码工作都放在这个子文件夹下
> 目标：从 highD 原始自然驾驶轨迹数据中自动抽取 cut-in 与 car-following / hard-braking 交互事件，完成轨迹清洗、方向统一、ego-centric 坐标转换、TTC/THW/DRAC 风险指标计算、固定长度窗口构建、风险分层标签生成，并导出可供 Deep EVT 与 diffusion 模型训练的数据集。

---

## 1. 背景与目标

本研究后续将构建 **条件深度 EVT 尾部风险模型** 与 **EVT-guided diffusion 对抗轨迹生成模型**。这两个模型的前提是：必须先从 highD 自然驾驶轨迹数据中构建一个干净、稳定、可复现、可解释的尾部风险交互轨迹数据集。

第一阶段代码不训练 Deep EVT，不训练 diffusion，不做 MATLAB / RoadRunner 仿真。它只负责完成从 highD 原始 CSV 到标准化训练数据的完整数据工程和事件抽取。

### 1.1 第一阶段核心目标

输入 highD 原始数据：

```text
XX_tracks.csv，例如 01_tracks.csv
XX_tracksMeta.csv，例如 01_tracksMeta.csv
XX_recordingMeta.csv，例如 01_recordingMeta.csv
```

输出标准化数据集：

```text
processed/
  events.csv
  trajectories.h5
  splits.json
  normalization_stats.json
  quality_report.json
  figures/
    risk_distribution_cutin.png
    risk_distribution_following.png
    example_cutin_*.png
    example_following_*.png
```

其中：

- `events.csv`：每个交互事件一行，包含事件类型、车辆 ID、起止帧、风险指标、切入/跟驰统计量、过滤状态等。
- `trajectories.h5`：固定长度 ego-centric 轨迹张量，供后续 Deep EVT 与 diffusion 使用。
- `splits.json`：训练/验证/测试划分，必须按 `recording_id` 划分，禁止简单随机样本划分。
- `normalization_stats.json`：后续模型训练所需的均值、标准差、上下界。
- `quality_report.json`：样本数量、过滤原因、风险分位数、异常比例等质量报告。

---

## 2. 总体代码结构要求

建议实现为一个模块化 Python package，而不是一个 notebook。推荐项目结构如下：

```text
tread_highd/
  README.md
  pyproject.toml 或 requirements.txt
  configs/
    highd_default.yaml
  src/
    tread_highd/
      __init__.py
      loader.py
      preprocess.py
      lane_utils.py
      event_extraction.py
      coordinate.py
      risk_metrics.py
      windowing.py
      filtering.py
      dataset_builder.py
      visualization.py
      quality_check.py
      io_utils.py
      schema.py
  scripts/
    01_extract_highd_events.py
    02_build_highd_dataset.py
    03_visualize_highd_events.py
    04_generate_quality_report.py
  tests/
    test_risk_metrics.py
    test_coordinate.py
    test_windowing.py
    test_event_extraction_synthetic.py
```

如果在 `RobertKrajewski/highD-dataset` 仓库中实现，请将上述 `tread_highd/` 作为仓库根目录下的新文件夹创建，不要改动原仓库的 `src/main.py`、`src/data_management/read_csv.py`、`src/visualization/visualize_frame.py`、`src/utils/plot_utils.py` 等示例工具。Codex 可以直接实现本需求中的独立读取器；如需复用原仓库读取逻辑，只允许在 `io_utils.py` 中做可选 adapter，并保证没有原仓库工具时本阶段代码仍可直接读取 highD CSV。


---

## 3. 配置文件需求

所有关键参数必须写入 YAML 配置，避免硬编码。

示例：`configs/highd_default.yaml`

```yaml
paths:
  raw_dir: "data/raw/highD"
  processed_dir: "data/processed"

recordings:
  include: "all"
  exclude: []

sampling:
  source_fps: 25
  target_fps: 10
  window_length: 64
  pre_anchor_steps: 32
  post_anchor_steps: 31

filters:
  min_segment_seconds: 4.0
  max_abs_accel: 8.0
  max_abs_jerk: 20.0
  max_position_jump: 5.0
  min_positive_gap: 0.5
  max_ttc_clip: 20.0
  max_thw_clip: 10.0
  min_vehicle_speed: 0.0

cutin:
  min_lane_stable_steps: 5
  lateral_velocity_threshold: 0.15
  max_post_cutin_gap: 120.0
  min_post_cutin_duration_steps: 10
  anchor_mode: "risk"  # options: cross, end, risk

following:
  min_same_preceding_steps: 40
  anchor_mode: "risk"  # options: min_ttc, max_drac, risk

risk:
  epsilon: 1.0e-6
  ttc_weight: 1.0
  thw_weight: 0.5
  drac_weight: 1.0
  softmax_lambda: 10.0
  tail_quantiles: [0.90, 0.95, 0.99]

splits:
  strategy: "recording"  # options: recording, location_if_available
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42

output:
  save_csv: true
  save_h5: true
  save_figures: true
  max_examples_per_type: 20
```

---

## 4. 数据结构与 schema 要求

### 4.1 轨迹事件 EventRecord schema

建议在 `schema.py` 中定义 dataclass 或 TypedDict。

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EventRecord:
    event_id: str
    event_type: str  # "cut_in" or "following"
    recording_id: int
    ego_id: int
    target_id: int

    start_frame: int
    end_frame: int
    anchor_frame: int

    # optional for cut-in
    cross_frame: Optional[int] = None
    cutin_start_frame: Optional[int] = None
    cutin_end_frame: Optional[int] = None
    source_lane: Optional[int] = None
    target_lane: Optional[int] = None

    # core risk metrics
    min_ttc: float = float("nan")
    min_thw: float = float("nan")
    max_drac: float = float("nan")
    risk_score: float = float("nan")

    # scenario statistics
    initial_gap: float = float("nan")
    min_gap: float = float("nan")
    initial_relative_speed: float = float("nan")
    post_cutin_gap: float = float("nan")
    cutin_duration: float = float("nan")

    # quality
    is_valid: bool = True
    filter_reason: str = ""
```

### 4.2 轨迹张量 schema

建议 `trajectories.h5` 中包含：

```text
/events/event_id: string array, shape [N]
/events/event_type: string array, shape [N]
/events/recording_id: int array, shape [N]
/events/ego_id: int array, shape [N]
/events/target_id: int array, shape [N]

/trajectories/states: float32 array, shape [N, T, A, F]
/trajectories/mask: bool array, shape [N, T, A]
/trajectories/frame_ids: int array, shape [N, T]

/risk/risk_score: float32 array, shape [N]
/risk/min_ttc: float32 array, shape [N]
/risk/min_thw: float32 array, shape [N]
/risk/max_drac: float32 array, shape [N]
/risk/risk_percentile: float32 array, shape [N]
/risk/tail_label_90: bool array, shape [N]
/risk/tail_label_95: bool array, shape [N]
/risk/tail_label_99: bool array, shape [N]
```

第一版 actor 数 `A=2`：

```text
actor 0 = ego vehicle
actor 1 = target vehicle, i.e., cut-in vehicle or lead vehicle
```

状态特征 `F` 建议为：

```text
0  dx                target/ego relative x; ego actor dx = 0
1  dy                relative y; ego actor dy = 0
2  dvx               relative longitudinal velocity
3  dvy               relative lateral velocity
4  vx                absolute longitudinal velocity after direction normalization
5  vy                lateral velocity
6  ax                longitudinal acceleration
7  ay                lateral acceleration
8  lane_id_normalized
9  length
10 width
```

因此：

```text
states shape = [N, 64, 2, 11]
```

---

## 5. 模块级代码需求

## 5.1 `loader.py`：highD 数据读取器

### 目标

读取 highD 三类 CSV 文件，并构建便于按车辆和按帧查询的数据结构。

### 必须实现的类与函数

```python
class HighDRecording:
    def __init__(self, recording_id: int, tracks: pd.DataFrame,
                 tracks_meta: pd.DataFrame, recording_meta: dict): ...

    def get_vehicle_track(self, vehicle_id: int) -> pd.DataFrame: ...

    def get_frame(self, frame_id: int) -> pd.DataFrame: ...

    def vehicle_ids(self) -> list[int]: ...

    def frame_ids(self) -> list[int]: ...
```

```python
def load_recording(raw_dir: str, recording_id: int) -> HighDRecording: ...

def load_all_recordings(raw_dir: str, include="all", exclude=None) -> list[HighDRecording]: ...
```

### 处理要求

1. 自动匹配 highD 官方命名文件，例如 `01_tracks.csv`、`01_tracksMeta.csv`、`01_recordingMeta.csv`；实现时使用 `{recording_id:02d}` 格式化。
2. 对 `precedingId`、`followingId`、`leftPrecedingId`、`rightPrecedingId` 等 ID 字段中无效值统一处理为 `-1`。
3. 对 tracks 建立 MultiIndex：`(id, frame)`。
4. 对每个 recording 建立缓存字典：`vehicle_id -> track_df`，`frame_id -> frame_df`。
5. loader 不能修改原始坐标，只负责读取和索引。

### 验收标准

- 能够加载任一 recording。
- 能够通过 `get_vehicle_track(id)` 返回该车完整轨迹。
- 能够通过 `get_frame(frame)` 返回该帧所有车辆。
- 读取后车辆数与 `tracksMeta` 车辆数一致。

---

## 5.2 `preprocess.py`：轨迹清洗、方向统一与重采样

### 目标

将 highD 原始轨迹转换为统一方向、统一帧率、异常已过滤的数据。

### 必须实现函数

```python
def check_frame_continuity(track: pd.DataFrame) -> bool: ...

def normalize_driving_direction(recording: HighDRecording) -> HighDRecording: ...

def filter_abnormal_tracks(recording: HighDRecording, config: dict) -> HighDRecording: ...

def resample_recording(recording: HighDRecording, target_fps: int) -> HighDRecording: ...
```

### 方向统一要求

highD 中不同方向车道的车辆可能沿相反 x 方向行驶。输出中必须统一为：

```text
ego forward direction = positive x
```

若某车辆原始纵向速度方向为负，则转换：

```text
x  -> -x
vx -> -vx
ax -> -ax
```

横向坐标是否反向需根据 highD lane coordinate 定义谨慎处理。第一版可以保留 y 不变，但必须保证 lane_id 与左右关系后续使用时一致。

### 异常过滤规则

至少实现：

1. 帧不连续的轨迹标记，不一定立即删除，但窗口构建时必须过滤。
2. `abs(xVelocity)`、`abs(xAcceleration)` 超出配置阈值的轨迹片段过滤。
3. 相邻帧位置跳变超过 `max_position_jump` 的轨迹片段过滤。
4. 车辆宽度、长度缺失或为 0 的样本过滤。

---

## 5.3 `lane_utils.py`：车道几何工具

### 目标

解析 `recordingMeta` 中车道线信息，支持车道中心线、车道宽度、相邻车道和车道变化检测。

### 必须实现函数

```python
def parse_lane_markings(recording_meta: dict) -> dict: ...

def get_lane_center(lane_id: int, lane_info: dict) -> float: ...

def get_lane_width(lane_id: int, lane_info: dict) -> float: ...

def are_adjacent_lanes(lane_a: int, lane_b: int, lane_info: dict) -> bool: ...

def detect_lane_changes(track: pd.DataFrame, min_stable_steps: int) -> list[dict]: ...
```

### lane change 输出格式

```python
{
  "vehicle_id": int,
  "from_lane": int,
  "to_lane": int,
  "cross_frame": int,
  "stable_before_start": int,
  "stable_after_end": int
}
```

### 注意事项

- highD 的 laneId 变化帧可作为 `cross_frame` 的初始近似。
- 后续 cut-in start/end 可以由横向速度进一步修正。
- 如果 lane marking 解析失败，应抛出带 recording_id 的清晰异常。

---

## 5.4 `event_extraction.py`：事件抽取

该模块包含两个核心抽取器：following 与 cut-in。

---

### 5.4.1 following / hard-braking 事件抽取

#### 目标

基于 `precedingId` 自动识别 ego-lead 跟驰片段，并以风险最高时刻为锚点构建事件窗口。

#### 必须实现函数

```python
def extract_following_segments(recording: HighDRecording, config: dict) -> list[dict]: ...

def build_following_event(recording: HighDRecording, segment: dict, config: dict) -> EventRecord | None: ...
```

#### segment 判定逻辑

对每辆 ego 车辆，遍历其轨迹中的 `precedingId`：

1. `precedingId != -1`。
2. ego 与 lead 在同一车道。
3. ego 与 lead 行驶方向一致。
4. lead 在 ego 前方，gap > `min_positive_gap`。
5. 该关系连续存在时间不低于 `min_same_preceding_steps`。
6. 该 segment 内 ego 不发生 lane change。

#### anchor frame

默认使用组合风险最高帧：

```text
anchor_frame = argmax_t S(t)
```

也支持配置：

```text
anchor_mode = min_ttc 或 max_drac
```

#### 输出 EventRecord 必填字段

```text
event_type = "following"
recording_id
ego_id
target_id = lead_id
start_frame
end_frame
anchor_frame
min_ttc
min_thw
max_drac
risk_score
initial_gap
min_gap
initial_relative_speed
```

---

### 5.4.2 cut-in 事件抽取

#### 目标

从 lane change 车辆中识别切入 ego 前方的事件。

#### 必须实现函数

```python
def extract_cutin_events(recording: HighDRecording, config: dict) -> list[EventRecord]: ...

def match_cutin_ego(recording: HighDRecording, lane_change: dict, config: dict) -> int | None: ...

def estimate_cutin_start_end(track: pd.DataFrame, cross_frame: int, config: dict) -> tuple[int, int]: ...
```

#### cut-in 判定逻辑

对每一个 lane change：

1. 目标车辆从相邻车道进入目标车道。
2. 在 lane change 完成后，目标车辆前方/后方关系稳定。
3. 在 `cross_frame` 或 `cutin_end_frame` 后，目标车辆位于某 ego 前方。
4. ego 与 cut-in vehicle 在同一车道且同方向。
5. 切入后 gap > `min_positive_gap`。
6. 切入后至少持续 `min_post_cutin_duration_steps`。

#### ego 匹配优先级

优先使用 highD 已有邻接 ID：

```text
cut-in vehicle 的 followingId at cutin_end_frame
```

若不可用，则在同车道后方车辆中找距离最近者：

```text
ego = argmin_j x_cutin - x_j, subject to x_cutin - x_j > 0
```

#### start/end frame 估计

`cross_frame` 由 laneId 改变帧确定。  
`cutin_start_frame` 向前搜索横向速度绝对值首次超过阈值的帧。  
`cutin_end_frame` 向后搜索 laneId 稳定且横向速度回落的帧。

#### 输出 EventRecord 必填字段

```text
event_type = "cut_in"
recording_id
ego_id
target_id = cutin_id
start_frame
end_frame
anchor_frame
cross_frame
cutin_start_frame
cutin_end_frame
source_lane
target_lane
cutin_duration
post_cutin_gap
min_ttc
min_thw
max_drac
risk_score
```

---

## 5.5 `risk_metrics.py`：风险指标计算

### 目标

计算 TTC、THW、DRAC 以及轨迹级综合风险分数。

### 必须实现函数

```python
def compute_gap(ego: pd.DataFrame, target: pd.DataFrame,
                ego_length: float, target_length: float) -> np.ndarray: ...

def compute_ttc(gap: np.ndarray, ego_vx: np.ndarray, target_vx: np.ndarray,
                max_ttc: float, eps: float) -> np.ndarray: ...

def compute_thw(gap: np.ndarray, ego_vx: np.ndarray,
                max_thw: float, eps: float) -> np.ndarray: ...

def compute_drac(gap: np.ndarray, ego_vx: np.ndarray, target_vx: np.ndarray,
                 eps: float) -> np.ndarray: ...

def compute_instant_risk(ttc: np.ndarray, thw: np.ndarray, drac: np.ndarray,
                         weights: dict, eps: float) -> np.ndarray: ...

def compute_trajectory_risk(instant_risk: np.ndarray, softmax_lambda: float) -> float: ...
```

### 公式要求

纵向净间距：

```text
gap = x_target - x_ego - 0.5 * (length_target + length_ego)
```

TTC：

```text
closing_speed = ego_vx - target_vx
if closing_speed > 0 and gap > 0:
    TTC = gap / closing_speed
else:
    TTC = max_ttc
```

THW：

```text
THW = gap / max(ego_vx, eps)
```

DRAC：

```text
if closing_speed > 0 and gap > 0:
    DRAC = closing_speed**2 / (2 * gap)
else:
    DRAC = 0
```

即时风险：

```text
S(t) = w_ttc/(TTC+eps) + w_thw/(THW+eps) + w_drac*DRAC
```

轨迹风险：

```text
R = logsumexp(lambda * S) / lambda
```

### 单元测试要求

必须编写 `test_risk_metrics.py`：

1. ego 比 target 慢时，TTC 应为 `max_ttc`。
2. gap 越小，TTC 越小。
3. closing speed 越大，DRAC 越大。
4. softmax risk 应接近 max risk。

---

## 5.6 `coordinate.py`：ego-centric 坐标转换

### 目标

将原始 highD 坐标转换为以 ego 为参考的相对坐标，输出固定维度状态张量。

### 必须实现函数

```python
def to_ego_centric(ego_track: pd.DataFrame, target_track: pd.DataFrame,
                   frames: np.ndarray) -> np.ndarray: ...

def build_state_tensor(event: EventRecord, recording: HighDRecording,
                       config: dict) -> tuple[np.ndarray, np.ndarray]: ...
```

### 输出要求

`states` shape:

```text
[T, A, F] = [64, 2, 11]
```

其中：

```text
actor 0 = ego
actor 1 = target
```

ego actor 的相对坐标：

```text
dx = 0
dy = 0
dvx = 0
dvy = 0
```

target actor 的相对坐标：

```text
dx = x_target - x_ego
dy = y_target - y_ego
dvx = vx_target - vx_ego
dvy = vy_target - vy_ego
```

### 注意

第一版 highD 高速公路可以近似为直路，不强制做 heading rotation。若后续扩展到 INTERACTION/nuPlan，再加入旋转矩阵。

---

## 5.7 `windowing.py`：固定长度窗口构建

### 目标

以事件 anchor frame 为中心，截取固定长度窗口。

### 必须实现函数

```python
def get_window_frames(anchor_frame: int, pre_steps: int, post_steps: int) -> np.ndarray: ...

def validate_window(recording: HighDRecording, ego_id: int, target_id: int,
                    frames: np.ndarray, config: dict) -> tuple[bool, str]: ...
```

### 验证条件

窗口必须满足：

1. ego 与 target 在所有窗口帧中均存在。
2. 帧连续，无缺失。
3. gap 在有效帧中不出现大量负值。
4. 车辆速度、加速度不超过配置阈值。
5. 对 following，ego 不发生 lane change。
6. 对 cut-in，窗口应覆盖切入过程或风险最高过程。

若失败，返回：

```python
(False, "reason_string")
```

过滤原因必须进入 quality report。

---

## 5.8 `dataset_builder.py`：数据集构建主流程

### 目标

整合 loader、preprocess、event extraction、risk metrics、coordinate、windowing，批量处理所有 recording。

### 必须实现类

```python
class HighDTailRiskDatasetBuilder:
    def __init__(self, config_path: str): ...

    def run(self) -> None: ...

    def process_recording(self, recording_id: int) -> list[EventRecord]: ...

    def build_trajectory_arrays(self, events: list[EventRecord]) -> dict: ...

    def assign_risk_percentiles(self, events_df: pd.DataFrame) -> pd.DataFrame: ...

    def build_splits(self, events_df: pd.DataFrame) -> dict: ...

    def export(self, events_df: pd.DataFrame, arrays: dict, splits: dict) -> None: ...
```

### 主流程伪代码

```python
for recording_id in recordings:
    recording = load_recording(raw_dir, recording_id)
    recording = normalize_driving_direction(recording)
    recording = filter_abnormal_tracks(recording, config)
    recording = resample_recording(recording, target_fps)

    following_events = extract_following_segments(recording, config)
    cutin_events = extract_cutin_events(recording, config)

    for event in following_events + cutin_events:
        frames = get_window_frames(event.anchor_frame, pre_steps, post_steps)
        valid, reason = validate_window(recording, event.ego_id, event.target_id, frames, config)
        if not valid:
            mark invalid and store reason
            continue
        states, mask = build_state_tensor(event, recording, config)
        risk_metrics = compute event risk
        store event and states

assign risk percentiles separately for cut_in and following
build recording-level train/val/test splits
export events csv, trajectories h5, splits json, stats json, quality report
```

---

## 5.9 `quality_check.py` 与 `visualization.py`

### 目标

生成可诊断的数据质量报告。该模块必须能帮助研究者判断事件抽取是否正确。

### 必须实现函数

```python
def generate_quality_report(events_df: pd.DataFrame, output_dir: str) -> dict: ...

def plot_event_trajectory(event: EventRecord, states: np.ndarray, save_path: str) -> None: ...

def plot_risk_timeseries(event: EventRecord, risk_series: dict, save_path: str) -> None: ...

def plot_risk_distribution(events_df: pd.DataFrame, event_type: str, save_path: str) -> None: ...

def plot_ttc_drac_scatter(events_df: pd.DataFrame, save_path: str) -> None: ...
```

### quality_report 内容

```json
{
  "num_recordings": 60,
  "num_candidate_cutin": 0,
  "num_valid_cutin": 0,
  "num_candidate_following": 0,
  "num_valid_following": 0,
  "filter_reasons": {
    "missing_frames": 0,
    "invalid_gap": 0,
    "abnormal_acceleration": 0
  },
  "risk_quantiles": {
    "cut_in": {"q50": 0, "q90": 0, "q95": 0, "q99": 0},
    "following": {"q50": 0, "q90": 0, "q95": 0, "q99": 0}
  }
}
```

---

## 6. CLI 脚本需求

### 6.1 抽取事件

```bash
python scripts/01_extract_highd_events.py \
  --config configs/highd_default.yaml \
  --recordings all
```

输出：

```text
processed/intermediate/candidate_events.csv
processed/intermediate/invalid_events.csv
```

### 6.2 构建训练数据集

```bash
python scripts/02_build_highd_dataset.py \
  --config configs/highd_default.yaml
```

输出：

```text
processed/events.csv
processed/trajectories.h5
processed/splits.json
processed/normalization_stats.json
```

### 6.3 可视化事件

```bash
python scripts/03_visualize_highd_events.py \
  --config configs/highd_default.yaml \
  --event_type cut_in \
  --top_k 20 \
  --sort_by risk_score
```

### 6.4 生成质量报告

```bash
python scripts/04_generate_quality_report.py \
  --config configs/highd_default.yaml
```

---

## 7. 训练/验证/测试划分要求

禁止按样本随机划分，因为同一个 recording 中存在高度相似的车辆交互。必须按 recording_id 划分。

建议实现：

```python
def split_by_recording(events_df, train_ratio, val_ratio, test_ratio, seed): ...
```

输出：

```json
{
  "train_recordings": [1, 2, 3],
  "val_recordings": [4],
  "test_recordings": [5],
  "train_event_ids": [],
  "val_event_ids": [],
  "test_event_ids": []
}
```

要求：

1. 同一 `recording_id` 的事件不能同时出现在 train 和 test。
2. cut-in 和 following 在三个集合中都尽量保留。
3. 输出 split summary。

---

## 8. 归一化统计量需求

后续 Deep EVT 与 diffusion 训练需要固定归一化参数。因此第一阶段必须输出训练集统计量。

```json
{
  "feature_names": ["dx", "dy", "dvx", "dvy", "vx", "vy", "ax", "ay", "lane_id", "length", "width"],
  "mean": [...],
  "std": [...],
  "min": [...],
  "max": [...],
  "risk": {
    "cut_in": {"q90": 0.0, "q95": 0.0, "q99": 0.0},
    "following": {"q90": 0.0, "q95": 0.0, "q99": 0.0}
  }
}
```

重要：风险分位数应只用训练集计算，然后应用到 val/test，避免数据泄漏。

---
## 10. 代码质量要求

1. 所有核心函数必须有 docstring。
2. 所有路径、阈值、窗口长度必须从 config 读取。
3. 禁止在核心逻辑中写绝对路径。
4. 对每个过滤原因必须有明确字符串标记。
5. 使用 `logging` 输出处理进度，不要只使用 `print`。
6. 大规模循环使用 `tqdm`。
7. 输出文件必须可重复生成。相同配置和随机种子应得到相同 split。
8. 如果某个 recording 处理失败，不应使整个 pipeline 崩溃；应记录错误并继续处理后续 recording。

---

## 11. 第一阶段验收标准

第一阶段完成后，应能稳定回答并输出以下结果：

1. highD 中共抽取多少个候选 cut-in 事件？多少个有效 cut-in 事件？
2. highD 中共抽取多少个候选 following 事件？多少个有效 following 事件？
3. 每类事件的 min TTC、min THW、max DRAC、risk_score 分布如何？
4. 风险最高的前 20 个 cut-in 事件在可视化上是否确实表现为切入冲突？
5. 风险最高的前 20 个 following 事件是否确实表现为低 TTC 或高 DRAC 跟驰风险？
6. 所有输出轨迹是否统一为 `[N, 64, 2, 11]`？
7. train/val/test 是否按 recording_id 分开？
8. 是否存在大量车辆重叠、坐标反向、车道错误、轨迹跳变？若有，quality report 中必须说明。

---

## 12. 不属于第一阶段的内容

以下内容不要在第一阶段实现，以免范围失控：

1. Deep EVT 网络训练。
2. GPD 参数头、pinball loss、EVT NLL。
3. diffusion 模型训练。
4. EVT guidance 采样。
5. MATLAB Automated Driving Toolbox 批量仿真。
6. RoadRunner 场景导出。
7. CARLA 或 OpenSCENARIO 执行。

第一阶段只负责把 highD 数据变成后续模型可用的 **tail-risk trajectory dataset**。

---

## 13. 推荐实现顺序

建议按以下顺序实现：

1. `loader.py`：读取 highD 文件并建立索引。
2. `risk_metrics.py`：实现 TTC、THW、DRAC 与单元测试。
3. `windowing.py`：实现固定窗口截取与验证。
4. `lane_utils.py`：实现 lane change 检测。
5. `event_extraction.py`：先实现 following，再实现 cut-in。
6. `coordinate.py`：实现 ego-centric 状态张量。
7. `dataset_builder.py`：串联全流程。
8. `visualization.py` 与 `quality_check.py`：生成图和报告。
9. CLI scripts。
10. 完成测试与 README。

---

## 14. 最终交付物

Codex 实现完成后，仓库应至少包含：

```text
configs/highd_default.yaml
src/tread_highd/*.py
scripts/01_extract_highd_events.py
scripts/02_build_highd_dataset.py
scripts/03_visualize_highd_events.py
scripts/04_generate_quality_report.py
tests/*.py
README.md
```

运行：

```bash
python scripts/02_build_highd_dataset.py --config configs/highd_default.yaml
```

应生成：

```text
processed/events.csv
processed/trajectories.h5
processed/splits.json
processed/normalization_stats.json
processed/quality_report.json
processed/figures/*.png
```

这即为 TREAD 研究的第一阶段数据基础。
