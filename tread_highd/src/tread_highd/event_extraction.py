"""
event_extraction.py — 事件抽取
===============================
从 highD recording 中提取 following 和 cut-in 交互事件。
参考: Matlab longfilter_onlycar.m, CutInFilter.m

EVT 方法论:
  本模块中所有过滤条件均为 语义/运动学规则:
    - 车辆类型 (排除卡车)
    - 同车道 (precedingId 一致 + laneId 匹配)
    - 连续帧数 (min_same_preceding_steps)
    - 间距合理性 (median gap > min_positive_gap)
    - 单次换道 + 相邻车道 (cut-in)
  风险指标 (TTC/THW/DRAC) 仅作为事件属性计算并记录，
  不用于筛选事件，以保持自然暴露分布的无偏完整性。
"""
from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from .loader import HighDRecording
from .schema import EventRecord
from .lane_utils import detect_lane_changes, are_adjacent_lanes, parse_lane_markings
from .risk_metrics import (compute_gap, compute_ttc, compute_thw,
                           compute_drac, compute_instant_risk, compute_trajectory_risk)

logger = logging.getLogger(__name__)


def _get_track_array(recording, vid, col):
    """从 tracks 中安全取出某车辆某列的 ndarray."""
    t = recording.get_vehicle_track(vid)
    return t[col].values if col in t.columns else np.array([])


def _align_frames(rec, ego_id, target_id, frame_range=None):
    """对齐 ego 与 target 的公共帧范围，返回 (common_frames, ego_df, target_df)."""
    ego_t = rec.get_vehicle_track(ego_id)
    tgt_t = rec.get_vehicle_track(target_id)

    # 提取到 frame 级别的 index
    if isinstance(ego_t.index, pd.MultiIndex) and "id" in ego_t.index.names:
        ego_t = ego_t.xs(ego_id, level="id") if ego_id in ego_t.index.get_level_values("id") else ego_t.droplevel("id")
    if isinstance(tgt_t.index, pd.MultiIndex) and "id" in tgt_t.index.names:
        tgt_t = tgt_t.xs(target_id, level="id") if target_id in tgt_t.index.get_level_values("id") else tgt_t.droplevel("id")

    ef = set(ego_t.index)
    tf = set(tgt_t.index)
    common = sorted(ef & tf)
    if frame_range is not None:
        common = [f for f in common if frame_range[0] <= f <= frame_range[1]]
    if not common:
        return np.array([]), pd.DataFrame(), pd.DataFrame()
    common = np.array(common)
    ego_df = ego_t.loc[common]
    tgt_df = tgt_t.loc[common]
    return common, ego_df, tgt_df


def _compute_event_risk(ego_df, tgt_df, ego_length, tgt_length, config):
    """为对齐的 ego/target DataFrame 计算风险指标，返回 dict."""
    risk_cfg = config.get("risk", {})
    filt_cfg = config.get("filters", {})
    eps = risk_cfg.get("epsilon", 1e-6)
    gap = compute_gap(ego_df["x"].values, tgt_df["x"].values, ego_length, tgt_length)
    ego_vx = ego_df["xVelocity"].values
    tgt_vx = tgt_df["xVelocity"].values
    ttc = compute_ttc(gap, ego_vx, tgt_vx, filt_cfg.get("max_ttc_clip", 20.0), eps)
    thw = compute_thw(gap, ego_vx, filt_cfg.get("max_thw_clip", 10.0), eps)
    drac = compute_drac(gap, ego_vx, tgt_vx, eps)
    instant = compute_instant_risk(ttc, thw, drac, risk_cfg, eps)
    traj_risk = compute_trajectory_risk(instant, risk_cfg.get("softmax_lambda", 10.0))
    return {
        "gap": gap, "ttc": ttc, "thw": thw, "drac": drac,
        "instant_risk": instant, "trajectory_risk": traj_risk,
        "min_ttc": float(np.min(ttc)) if len(ttc) > 0 else float("nan"),
        "min_thw": float(np.min(thw)) if len(thw) > 0 else float("nan"),
        "max_drac": float(np.max(drac)) if len(drac) > 0 else float("nan"),
    }


# ══════════════════════════════════════════════════════════
# Following 事件抽取
# ══════════════════════════════════════════════════════════

def extract_following_segments(recording, config):
    """提取所有跟驰事件段。

    逻辑 (参考 longfilter_onlycar.m):
    1. 遍历每辆 ego，找 precedingId != -1 的连续段
    2. ego 与 lead 在同一车道、同方向
    3. gap > min_positive_gap
    4. 连续帧 >= min_same_preceding_steps
    5. ego 不发生 lane change
    """
    fol_cfg = config.get("following", {})
    filt_cfg = config.get("filters", {})
    min_steps = fol_cfg.get("min_same_preceding_steps", 40)
    min_gap = filt_cfg.get("min_positive_gap", 0.5)
    anchor_mode = fol_cfg.get("anchor_mode", "risk")

    meta = recording.tracks_meta
    events = []
    event_counter = 0

    for ego_id in meta.index:
        ego_meta = meta.loc[ego_id]
        # 跳过卡车和换道车辆
        if str(ego_meta.get("class", "")).lower() == "truck":
            continue
        if ego_meta.get("numLaneChanges", 0) > 0:
            continue

        try:
            ego_track = recording.get_vehicle_track(ego_id)
        except KeyError:
            continue

        if isinstance(ego_track.index, pd.MultiIndex):
            ego_track = ego_track.droplevel("id")

        if "precedingId" not in ego_track.columns:
            continue

        prec_ids = ego_track["precedingId"].values
        frames = ego_track.index.values
        lane_ids = ego_track["laneId"].values if "laneId" in ego_track.columns else None

        # 找连续相同 precedingId 的段
        segments = []
        seg_start = 0
        for i in range(1, len(prec_ids)):
            if prec_ids[i] != prec_ids[seg_start] or prec_ids[i] == -1:
                if prec_ids[seg_start] != -1 and (i - seg_start) >= min_steps:
                    segments.append((seg_start, i - 1, int(prec_ids[seg_start])))
                seg_start = i
        # 最后一段
        if prec_ids[seg_start] != -1 and (len(prec_ids) - seg_start) >= min_steps:
            segments.append((seg_start, len(prec_ids) - 1, int(prec_ids[seg_start])))

        for s_start, s_end, lead_id in segments:
            # 检查 lead 车辆是否也是卡车
            if lead_id in meta.index and str(meta.loc[lead_id].get("class", "")).lower() == "truck":
                continue

            seg_frames = frames[s_start:s_end + 1]
            fr_range = (int(seg_frames[0]), int(seg_frames[-1]))

            common_f, ego_df, tgt_df = _align_frames(recording, ego_id, lead_id, fr_range)
            if len(common_f) < min_steps:
                continue

            # 检查同车道
            if lane_ids is not None and "laneId" in tgt_df.columns:
                ego_lanes = ego_df["laneId"].values if "laneId" in ego_df.columns else None
                tgt_lanes = tgt_df["laneId"].values
                if ego_lanes is not None and not np.all(ego_lanes == tgt_lanes):
                    # 允许少量不一致
                    same_rate = np.mean(ego_lanes == tgt_lanes)
                    if same_rate < 0.8:
                        continue

            # 获取车辆尺寸 (highD width = 车长)
            ego_len = float(meta.loc[ego_id].get("width", 4.5))
            tgt_len = float(meta.loc[lead_id].get("width", 4.5)) if lead_id in meta.index else 4.5

            # 计算风险
            risk = _compute_event_risk(ego_df, tgt_df, ego_len, tgt_len, config)

            # 检查间距
            if np.median(risk["gap"]) < min_gap:
                continue

            # 确定 anchor frame
            if anchor_mode == "min_ttc":
                anchor_idx = int(np.argmin(risk["ttc"]))
            elif anchor_mode == "max_drac":
                anchor_idx = int(np.argmax(risk["drac"]))
            else:  # "risk"
                anchor_idx = int(np.argmax(risk["instant_risk"]))
            anchor_frame = int(common_f[anchor_idx])

            event_counter += 1
            ev = EventRecord(
                event_id=f"fol_{recording.recording_id:02d}_{event_counter:05d}",
                event_type="following",
                recording_id=recording.recording_id,
                ego_id=ego_id,
                target_id=lead_id,
                start_frame=int(common_f[0]),
                end_frame=int(common_f[-1]),
                anchor_frame=anchor_frame,
                min_ttc=risk["min_ttc"],
                min_thw=risk["min_thw"],
                max_drac=risk["max_drac"],
                risk_score=risk["trajectory_risk"],
                initial_gap=float(risk["gap"][0]),
                min_gap=float(np.min(risk["gap"])),
                initial_relative_speed=float(ego_df["xVelocity"].values[0] - tgt_df["xVelocity"].values[0]),
            )
            events.append(ev)

    logger.info("Recording %02d: 提取 %d 个 following 事件",
                recording.recording_id, len(events))
    return events


# ══════════════════════════════════════════════════════════
# Cut-in 事件抽取
# ══════════════════════════════════════════════════════════

def match_cutin_ego(recording, lane_change, config):
    """为换道事件匹配被切入的 ego 车辆。
    
    参考 CutInFilter.m: 优先使用 leftFollowingId/rightFollowingId，
    否则在同车道后方找最近车辆。
    """
    cutin_id = lane_change["vehicle_id"]
    cross_frame = lane_change["cross_frame"]
    target_lane = lane_change["to_lane"]
    meta = recording.tracks_meta

    try:
        cutin_track = recording.get_vehicle_track(cutin_id)
    except KeyError:
        return None

    if isinstance(cutin_track.index, pd.MultiIndex):
        cutin_track = cutin_track.droplevel("id")

    # 在 cross_frame 或附近帧找 followingId
    end_frame = lane_change.get("stable_after_end", cross_frame)

    for check_frame in [end_frame, cross_frame]:
        if check_frame in cutin_track.index:
            row = cutin_track.loc[check_frame]
            fid = int(row.get("followingId", -1))
            if fid != -1 and fid in meta.index:
                if str(meta.loc[fid].get("class", "")).lower() != "truck":
                    return fid

    # 回退: 在同车道后方找最近车辆
    try:
        frame_df = recording.get_frame(cross_frame)
    except Exception:
        return None

    cutin_x = None
    if cross_frame in cutin_track.index:
        cutin_x = float(cutin_track.loc[cross_frame, "x"])

    if cutin_x is None:
        return None

    # 在 frame_df 中找同车道且在后方的车辆
    # frame_df 有 MultiIndex (id, frame)
    candidates = []
    if isinstance(frame_df.index, pd.MultiIndex):
        vids_in_frame = frame_df.index.get_level_values("id").unique()
        for vid in vids_in_frame:
            if vid == cutin_id:
                continue
            try:
                row = frame_df.loc[(vid, cross_frame)]
            except KeyError:
                continue
            if "laneId" in frame_df.columns:
                lane_val = int(row["laneId"]) if not isinstance(row["laneId"], pd.Series) else int(row["laneId"].iloc[0])
                if lane_val != target_lane:
                    continue
            if vid in meta.index and str(meta.loc[vid].get("class", "")).lower() == "truck":
                continue
            vx = float(row["x"]) if not isinstance(row["x"], pd.Series) else float(row["x"].iloc[0])
            if vx < cutin_x:  # 后方
                candidates.append((vid, cutin_x - vx))
    else:
        for idx in frame_df.index:
            vid = idx
            if vid == cutin_id:
                continue
            row = frame_df.loc[idx]
            if "laneId" in frame_df.columns:
                if int(row["laneId"]) != target_lane:
                    continue
            if vid in meta.index and str(meta.loc[vid].get("class", "")).lower() == "truck":
                continue
            vx = float(row["x"])
            if vx < cutin_x:
                candidates.append((vid, cutin_x - vx))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def estimate_cutin_start_end(track, cross_frame, config):
    """估计切入起始和结束帧 (基于横向速度)。
    
    参考 CutInFilter.m 行 24-25:
    - start: 向前搜索 |yVelocity| >= threshold 的首帧
    - end: 向后搜索 |yVelocity| <= threshold 的首帧
    """
    cutin_cfg = config.get("cutin", {})
    lat_thresh = cutin_cfg.get("lateral_velocity_threshold", 0.15)

    if isinstance(track.index, pd.MultiIndex):
        track = track.droplevel("id")

    frames = track.index.values
    cross_idx = np.searchsorted(frames, cross_frame)
    cross_idx = min(cross_idx, len(frames) - 1)

    yvel = track["yVelocity"].values if "yVelocity" in track.columns else np.zeros(len(frames))

    # 向前搜索 start
    start_idx = cross_idx
    for i in range(cross_idx - 1, -1, -1):
        if abs(yvel[i]) < lat_thresh * 0.3:  # 低于阈值30%视为未开始
            start_idx = i + 1
            break
        start_idx = i

    # 向后搜索 end
    end_idx = min(cross_idx + 1, len(frames) - 1)
    for i in range(cross_idx + 1, len(frames)):
        if abs(yvel[i]) < lat_thresh * 0.3:
            end_idx = i
            break
        end_idx = i

    return int(frames[start_idx]), int(frames[end_idx])


def extract_cutin_events(recording, config):
    """提取所有 cut-in 事件。

    逻辑 (参考 CutInFilter.m):
    1. 遍历发生 1 次换道的小汽车
    2. 找到换道过程、目标车道后车
    3. 计算风险指标
    """
    cutin_cfg = config.get("cutin", {})
    filt_cfg = config.get("filters", {})
    min_gap = filt_cfg.get("min_positive_gap", 0.5)
    min_post_steps = cutin_cfg.get("min_post_cutin_duration_steps", 10)
    anchor_mode = cutin_cfg.get("anchor_mode", "risk")
    min_stable = cutin_cfg.get("min_lane_stable_steps", 5)

    meta = recording.tracks_meta
    lane_info = None
    try:
        lane_info = parse_lane_markings(recording.recording_meta)
    except Exception as e:
        logger.warning("Recording %02d: 车道解析失败: %s", recording.recording_id, e)

    events = []
    event_counter = 0

    for vid in meta.index:
        vm = meta.loc[vid]
        if vm.get("numLaneChanges", 0) != 1:
            continue
        if str(vm.get("class", "")).lower() != "car":
            continue

        try:
            track = recording.get_vehicle_track(vid)
        except KeyError:
            continue

        lc_list = detect_lane_changes(track, vid, min_stable)
        if not lc_list:
            continue

        for lc in lc_list:
            # 检查相邻车道
            if lane_info and not are_adjacent_lanes(lc["from_lane"], lc["to_lane"], lane_info):
                continue

            # 匹配 ego
            ego_id = match_cutin_ego(recording, lc, config)
            if ego_id is None:
                continue

            # 检查 ego 不是卡车
            if ego_id in meta.index and str(meta.loc[ego_id].get("class", "")).lower() == "truck":
                continue

            # 估算 cutin start/end
            cutin_start, cutin_end = estimate_cutin_start_end(track, lc["cross_frame"], config)

            # 对齐帧
            common_f, ego_df, tgt_df = _align_frames(recording, ego_id, vid)
            if len(common_f) < min_post_steps:
                continue

            ego_len = float(meta.loc[ego_id].get("width", 4.5))
            tgt_len = float(meta.loc[vid].get("width", 4.5))

            risk = _compute_event_risk(ego_df, tgt_df, ego_len, tgt_len, config)

            # 跳过间距全为负的
            if np.all(risk["gap"] < 0):
                continue

            # anchor frame
            # 只在 cross_frame 之后计算
            cross_idx = np.searchsorted(common_f, lc["cross_frame"])
            if cross_idx >= len(common_f):
                cross_idx = len(common_f) - 1

            post_risk = risk["instant_risk"][cross_idx:]
            if len(post_risk) < min_post_steps:
                continue

            if anchor_mode == "cross":
                anchor_frame = lc["cross_frame"]
            elif anchor_mode == "end":
                anchor_frame = cutin_end
            else:  # "risk"
                anchor_idx = cross_idx + int(np.argmax(post_risk))
                anchor_frame = int(common_f[anchor_idx])

            # post cutin gap
            post_gap = risk["gap"][cross_idx:]
            post_cutin_gap = float(np.mean(post_gap)) if len(post_gap) > 0 else float("nan")

            fps = int(recording.recording_meta.get("frameRate", 25))
            duration = (cutin_end - cutin_start) / fps if fps > 0 else 0.0

            event_counter += 1
            ev = EventRecord(
                event_id=f"cin_{recording.recording_id:02d}_{event_counter:05d}",
                event_type="cut_in",
                recording_id=recording.recording_id,
                ego_id=ego_id,
                target_id=vid,
                start_frame=int(common_f[0]),
                end_frame=int(common_f[-1]),
                anchor_frame=anchor_frame,
                cross_frame=lc["cross_frame"],
                cutin_start_frame=cutin_start,
                cutin_end_frame=cutin_end,
                source_lane=lc["from_lane"],
                target_lane=lc["to_lane"],
                min_ttc=risk["min_ttc"],
                min_thw=risk["min_thw"],
                max_drac=risk["max_drac"],
                risk_score=risk["trajectory_risk"],
                initial_gap=float(risk["gap"][0]),
                min_gap=float(np.min(risk["gap"])),
                initial_relative_speed=float(ego_df["xVelocity"].values[0] - tgt_df["xVelocity"].values[0]),
                post_cutin_gap=post_cutin_gap,
                cutin_duration=duration,
            )
            events.append(ev)

    logger.info("Recording %02d: 提取 %d 个 cut-in 事件",
                recording.recording_id, len(events))
    return events
