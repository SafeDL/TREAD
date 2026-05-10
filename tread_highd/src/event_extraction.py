"""
event_extraction.py — 事件抽取
===============================
从 highD recording 中提取 following 和 cut-in 交互事件。
参考: Matlab longfilter_onlycar.m, CutInFilter.m

EVT 方法论:
  所有过滤条件均为语义/运动学规则，风险指标仅作为事件属性记录，
  不用于筛选事件，以保持自然暴露分布的无偏完整性。
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from .loader import HighDRecording
from .schema import EventRecord
from .lane_utils import detect_lane_changes, are_adjacent_lanes, parse_lane_markings
from .risk_metrics import (compute_gap, compute_ttc, compute_thw,
                           compute_drac, compute_instant_risk, compute_trajectory_risk)

logger = logging.getLogger(__name__)


def _align_frames(rec, ego_id, target_id, frame_range=None):
    """对齐 ego 与 target 的公共帧范围。"""
    ego_t = rec.get_vehicle_track(ego_id)
    tgt_t = rec.get_vehicle_track(target_id)
    common = sorted(set(ego_t.index) & set(tgt_t.index))
    if frame_range is not None:
        common = [f for f in common if frame_range[0] <= f <= frame_range[1]]
    if not common:
        return np.array([]), pd.DataFrame(), pd.DataFrame()
    common = np.array(common)
    return common, ego_t.loc[common], tgt_t.loc[common]


def _compute_event_risk(ego_df, tgt_df, ego_length, tgt_length, config,
                        frames=None, risk_start_frame=None):
    """计算同一有效风险窗口内的 danger-oriented 指标。"""
    risk_cfg = config.get("risk", {})
    filt_cfg = config.get("filters", {})
    eps = risk_cfg.get("epsilon", 1e-6)
    gap = compute_gap(ego_df["x"].values, tgt_df["x"].values, ego_length, tgt_length)
    ego_vx = ego_df["xVelocity"].values
    tgt_vx = tgt_df["xVelocity"].values
    ttc = compute_ttc(gap, ego_vx, tgt_vx, filt_cfg.get("max_ttc_clip", 20.0), eps)
    thw = compute_thw(gap, ego_vx, filt_cfg.get("max_thw_clip", 10.0), eps)
    drac = compute_drac(gap, ego_vx, tgt_vx, eps)

    risk_mask = gap > eps
    if frames is not None and risk_start_frame is not None:
        risk_mask &= np.asarray(frames) >= risk_start_frame
    risk_indices = np.flatnonzero(risk_mask)

    if len(risk_indices) == 0:
        return {
            "gap": gap, "ttc": ttc, "thw": thw, "drac": drac,
            "instant_risk": np.array([]), "risk_indices": risk_indices,
            "trajectory_risk": float("nan"),
            "min_ttc": float("nan"), "min_thw": float("nan"), "max_drac": float("nan"),
            "ttc_severity": float("nan"), "thw_severity": float("nan"),
            "drac_severity": float("nan"), "valid_risk_frames": 0,
            "risk_window_start_frame": None, "risk_window_end_frame": None,
            "risk_min_gap": float("nan"),
        }

    r_ttc = ttc[risk_indices]
    r_thw = thw[risk_indices]
    r_drac = drac[risk_indices]
    instant = compute_instant_risk(r_ttc, r_thw, r_drac, risk_cfg, eps)
    traj_risk = compute_trajectory_risk(instant, risk_cfg.get("softmax_lambda", 10.0))

    min_ttc = float(np.min(r_ttc))
    min_thw = float(np.min(r_thw))
    max_drac = float(np.max(r_drac))
    frame_values = np.asarray(frames)[risk_indices] if frames is not None else risk_indices
    return {
        "gap": gap, "ttc": ttc, "thw": thw, "drac": drac,
        "instant_risk": instant, "risk_indices": risk_indices,
        "trajectory_risk": traj_risk,
        "min_ttc": min_ttc,
        "min_thw": min_thw,
        "max_drac": max_drac,
        "ttc_severity": float(1.0 / (min_ttc + eps)),
        "thw_severity": float(1.0 / (min_thw + eps)),
        "drac_severity": max_drac,
        "valid_risk_frames": int(len(risk_indices)),
        "risk_window_start_frame": int(frame_values[0]),
        "risk_window_end_frame": int(frame_values[-1]),
        "risk_min_gap": float(np.min(gap[risk_indices])),
    }


# ══════════════════════════════════════════════════════════
# Following 事件抽取
# ══════════════════════════════════════════════════════════

def extract_following_segments(recording, config):
    """提取所有跟驰事件段。

    筛选规则 (语义/运动学):
    1. ego 和 lead 均为小汽车
    2. ego 无换道
    3. precedingId 连续 >= min_same_preceding_steps
    4. 两车同车道 (>= 80%)
    5. median gap > min_positive_gap
    """
    fol_cfg = config.get("following", {})
    filt_cfg = config.get("filters", {})
    min_steps = fol_cfg.get("min_same_preceding_steps", 40)
    min_gap = filt_cfg.get("min_positive_gap", 0.5)
    anchor_mode = fol_cfg.get("anchor_mode", "center")

    meta = recording.tracks_meta
    events = []
    event_counter = 0

    for ego_id in meta.index:
        ego_meta = meta.loc[ego_id]
        if str(ego_meta.get("class", "")).lower() == "truck":
            continue
        if ego_meta.get("numLaneChanges", 0) > 0:
            continue

        ego_track = recording.get_vehicle_track(ego_id)
        prec_ids = ego_track["precedingId"].values
        frames = ego_track.index.values

        # 找连续相同 precedingId 的段
        segments = []
        seg_start = 0
        for i in range(1, len(prec_ids)):
            if prec_ids[i] != prec_ids[seg_start] or prec_ids[i] == -1:
                if prec_ids[seg_start] != -1 and (i - seg_start) >= min_steps:
                    segments.append((seg_start, i - 1, int(prec_ids[seg_start])))
                seg_start = i
        if prec_ids[seg_start] != -1 and (len(prec_ids) - seg_start) >= min_steps:
            segments.append((seg_start, len(prec_ids) - 1, int(prec_ids[seg_start])))

        for s_start, s_end, lead_id in segments:
            if lead_id in meta.index and str(meta.loc[lead_id].get("class", "")).lower() == "truck":
                continue

            seg_frames = frames[s_start:s_end + 1]
            fr_range = (int(seg_frames[0]), int(seg_frames[-1]))
            common_f, ego_df, tgt_df = _align_frames(recording, ego_id, lead_id, fr_range)
            if len(common_f) < min_steps:
                continue

            # 同车道检查
            ego_lanes = ego_df["laneId"].values
            tgt_lanes = tgt_df["laneId"].values
            if np.mean(ego_lanes == tgt_lanes) < 0.8:
                continue

            ego_len = float(meta.loc[ego_id]["width"])
            tgt_len = float(meta.loc[lead_id]["width"])
            risk = _compute_event_risk(ego_df, tgt_df, ego_len, tgt_len,
                                       config, frames=common_f)

            if np.median(risk["gap"]) < min_gap:
                continue
            if risk["valid_risk_frames"] == 0:
                event_counter += 1
                events.append(EventRecord(
                    event_id=f"fol_{recording.recording_id:02d}_{event_counter:05d}",
                    event_type="following",
                    recording_id=recording.recording_id,
                    ego_id=ego_id, target_id=lead_id,
                    start_frame=int(common_f[0]), end_frame=int(common_f[-1]),
                    anchor_frame=int(common_f[0]),
                    initial_gap=float(risk["gap"][0]),
                    initial_relative_speed=float(ego_df["xVelocity"].values[0] - tgt_df["xVelocity"].values[0]),
                    is_valid=False,
                    filter_reason="no_valid_risk_frames",
                ))
                continue

            # anchor frame
            if anchor_mode == "center":
                anchor_idx = len(common_f) // 2
            elif anchor_mode == "min_ttc":
                anchor_idx = int(risk["risk_indices"][np.argmin(risk["ttc"][risk["risk_indices"]])])
            elif anchor_mode == "max_drac":
                anchor_idx = int(risk["risk_indices"][np.argmax(risk["drac"][risk["risk_indices"]])])
            else:
                anchor_idx = int(risk["risk_indices"][np.argmax(risk["instant_risk"])])

            event_counter += 1
            events.append(EventRecord(
                event_id=f"fol_{recording.recording_id:02d}_{event_counter:05d}",
                event_type="following",
                recording_id=recording.recording_id,
                ego_id=ego_id, target_id=lead_id,
                start_frame=int(common_f[0]), end_frame=int(common_f[-1]),
                anchor_frame=int(common_f[anchor_idx]),
                min_ttc=risk["min_ttc"], min_thw=risk["min_thw"], max_drac=risk["max_drac"],
                ttc_severity=risk["ttc_severity"], thw_severity=risk["thw_severity"],
                drac_severity=risk["drac_severity"],
                risk_score=risk["trajectory_risk"],
                risk_window_start_frame=risk["risk_window_start_frame"],
                risk_window_end_frame=risk["risk_window_end_frame"],
                valid_risk_frames=risk["valid_risk_frames"],
                initial_gap=float(risk["gap"][0]), min_gap=risk["risk_min_gap"],
                initial_relative_speed=float(ego_df["xVelocity"].values[0] - tgt_df["xVelocity"].values[0]),
            ))

    logger.info("Recording %02d: 提取 %d 个 following 事件",
                recording.recording_id, len(events))
    return events


# ══════════════════════════════════════════════════════════
# Cut-in 事件抽取
# ══════════════════════════════════════════════════════════

def match_cutin_ego(recording, lane_change, config):
    """为换道事件匹配被切入的 ego 车辆。

    优先使用 followingId，否则在目标车道后方找最近小汽车。
    """
    cutin_id = lane_change["vehicle_id"]
    cross_frame = lane_change["cross_frame"]
    target_lane = lane_change["to_lane"]
    meta = recording.tracks_meta
    cutin_track = recording.get_vehicle_track(cutin_id)

    # 优先: followingId
    end_frame = lane_change.get("stable_after_end", cross_frame)
    for check_frame in [end_frame, cross_frame]:
        if check_frame in cutin_track.index:
            fid = int(cutin_track.loc[check_frame, "followingId"])
            if fid != -1 and fid in meta.index:
                if str(meta.loc[fid].get("class", "")).lower() != "truck":
                    return fid

    # 回退: 在目标车道后方找最近小汽车
    if cross_frame not in cutin_track.index:
        return None
    cutin_x = float(cutin_track.loc[cross_frame, "x"])

    frame_df = recording.get_frame(cross_frame)
    vids = frame_df.index.get_level_values("id").unique()
    candidates = []
    for vid in vids:
        if vid == cutin_id:
            continue
        row = frame_df.loc[(vid, cross_frame)]
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
    """估计切入起始和结束帧 (基于横向速度)。"""
    lat_thresh = config.get("cutin", {}).get("lateral_velocity_threshold", 0.15)
    frames = track.index.values
    cross_idx = min(np.searchsorted(frames, cross_frame), len(frames) - 1)
    yvel = track["yVelocity"].values

    start_idx = cross_idx
    for i in range(cross_idx - 1, -1, -1):
        if abs(yvel[i]) < lat_thresh * 0.3:
            start_idx = i + 1
            break
        start_idx = i

    end_idx = min(cross_idx + 1, len(frames) - 1)
    for i in range(cross_idx + 1, len(frames)):
        if abs(yvel[i]) < lat_thresh * 0.3:
            end_idx = i
            break
        end_idx = i

    return int(frames[start_idx]), int(frames[end_idx])


def extract_cutin_events(recording, config):
    """提取所有 cut-in 事件。

    筛选规则 (语义/运动学):
    1. 目标车: 恰好 1 次换道的小汽车，相邻车道
    2. 切入后 target 在 ego 前方 (median post_gap > 0)
    3. 切入后两车同车道 (>= 70%)
    4. cutin 持续时间 >= min_cutin_duration_steps
    5. 间距不全为负 (排除数据对齐错误)
    6. cross_frame 后帧数 >= min_post_cutin_duration_steps
    """
    cutin_cfg = config.get("cutin", {})
    filt_cfg = config.get("filters", {})
    min_post_steps = cutin_cfg.get("min_post_cutin_duration_steps", 10)
    min_cutin_duration_steps = cutin_cfg.get("min_cutin_duration_steps", 5)
    anchor_mode = cutin_cfg.get("anchor_mode", "risk")
    min_stable = cutin_cfg.get("min_lane_stable_steps", 5)

    meta = recording.tracks_meta
    lane_info = parse_lane_markings(recording.recording_meta)

    events = []
    event_counter = 0

    for vid in meta.index:
        vm = meta.loc[vid]
        if vm.get("numLaneChanges", 0) != 1 or str(vm.get("class", "")).lower() != "car":
            continue

        track = recording.get_vehicle_track(vid)
        lc_list = detect_lane_changes(track, vid, min_stable)
        if not lc_list:
            continue

        for lc in lc_list:
            if lane_info and not are_adjacent_lanes(lc["from_lane"], lc["to_lane"], lane_info):
                continue

            ego_id = match_cutin_ego(recording, lc, config)
            if ego_id is None:
                continue
            if str(meta.loc[ego_id].get("class", "")).lower() == "truck":
                continue

            cutin_start, cutin_end = estimate_cutin_start_end(track, lc["cross_frame"], config)
            if (cutin_end - cutin_start) < min_cutin_duration_steps:
                continue

            common_f, ego_df, tgt_df = _align_frames(recording, ego_id, vid)
            if len(common_f) < min_post_steps:
                continue

            # 切入后 target 在 ego 前方 + 同车道
            cross_idx = min(np.searchsorted(common_f, lc["cross_frame"]), len(common_f) - 1)
            post_gap = tgt_df["x"].values[cross_idx:] - ego_df["x"].values[cross_idx:]
            if len(post_gap) < min_post_steps:
                continue
            if len(post_gap) > 0 and np.median(post_gap) < 0:
                continue

            post_ego_lanes = ego_df["laneId"].values[cross_idx:]
            post_tgt_lanes = tgt_df["laneId"].values[cross_idx:]
            if len(post_ego_lanes) > 0 and np.mean(post_ego_lanes == post_tgt_lanes) < 0.7:
                continue

            ego_len = float(meta.loc[ego_id]["width"])
            tgt_len = float(meta.loc[vid]["width"])
            risk = _compute_event_risk(ego_df, tgt_df, ego_len, tgt_len,
                                       config, frames=common_f,
                                       risk_start_frame=lc["cross_frame"])
            if risk["valid_risk_frames"] == 0:
                event_counter += 1
                events.append(EventRecord(
                    event_id=f"cin_{recording.recording_id:02d}_{event_counter:05d}",
                    event_type="cut_in",
                    recording_id=recording.recording_id,
                    ego_id=ego_id, target_id=vid,
                    start_frame=int(common_f[0]), end_frame=int(common_f[-1]),
                    anchor_frame=int(lc["cross_frame"]),
                    cross_frame=lc["cross_frame"],
                    cutin_start_frame=cutin_start, cutin_end_frame=cutin_end,
                    source_lane=lc["from_lane"], target_lane=lc["to_lane"],
                    initial_gap=float(risk["gap"][0]),
                    initial_relative_speed=float(ego_df["xVelocity"].values[0] - tgt_df["xVelocity"].values[0]),
                    is_valid=False,
                    filter_reason="no_valid_risk_frames",
                ))
                continue

            if anchor_mode == "cross":
                anchor_frame = lc["cross_frame"]
            elif anchor_mode == "end":
                anchor_frame = cutin_end
            else:
                anchor_idx = int(risk["risk_indices"][np.argmax(risk["instant_risk"])])
                anchor_frame = int(common_f[anchor_idx])

            post_cutin_gap = float(np.mean(risk["gap"][risk["risk_indices"]]))
            fps = int(recording.recording_meta.get("frameRate", 25))
            duration = (cutin_end - cutin_start) / fps

            event_counter += 1
            events.append(EventRecord(
                event_id=f"cin_{recording.recording_id:02d}_{event_counter:05d}",
                event_type="cut_in",
                recording_id=recording.recording_id,
                ego_id=ego_id, target_id=vid,
                start_frame=int(common_f[0]), end_frame=int(common_f[-1]),
                anchor_frame=anchor_frame,
                cross_frame=lc["cross_frame"],
                cutin_start_frame=cutin_start, cutin_end_frame=cutin_end,
                source_lane=lc["from_lane"], target_lane=lc["to_lane"],
                min_ttc=risk["min_ttc"], min_thw=risk["min_thw"], max_drac=risk["max_drac"],
                ttc_severity=risk["ttc_severity"], thw_severity=risk["thw_severity"],
                drac_severity=risk["drac_severity"],
                risk_score=risk["trajectory_risk"],
                risk_window_start_frame=risk["risk_window_start_frame"],
                risk_window_end_frame=risk["risk_window_end_frame"],
                valid_risk_frames=risk["valid_risk_frames"],
                initial_gap=float(risk["gap"][0]), min_gap=risk["risk_min_gap"],
                initial_relative_speed=float(ego_df["xVelocity"].values[0] - tgt_df["xVelocity"].values[0]),
                post_cutin_gap=post_cutin_gap,
                cutin_duration=duration,
            ))

    logger.info("Recording %02d: 提取 %d 个 cut-in 事件",
                recording.recording_id, len(events))
    return events
