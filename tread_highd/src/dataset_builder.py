"""
dataset_builder.py — 数据集构建主流程
======================================
整合所有模块，批量处理所有 recording，导出标准化数据集
"""
from __future__ import annotations
import logging
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

from .io_utils import load_config, ensure_dir, resolve_data_path, save_json, resolve_recording_ids
from .loader import load_recording
from .preprocess import normalize_driving_direction, filter_abnormal_tracks, resample_recording
from .event_extraction import extract_following_segments, extract_cutin_events
from .coordinate import build_state_tensor
from .windowing import get_window_frames, get_window_from_track, validate_window
from .filtering import events_to_dataframe, filter_events
from .quality_check import generate_quality_report
from .visualization import (plot_risk_distribution, plot_ttc_drac_scatter,
                            plot_event_trajectory)
from .schema import EventRecord

logger = logging.getLogger(__name__)


class HighDTailRiskDatasetBuilder:
    """TREAD highD 数据集构建器。"""

    def __init__(self, config_path: str):
        self.config_path = str(Path(config_path).resolve())
        self.config = load_config(config_path)
        self.raw_dir = str(resolve_data_path(
            self.config["paths"]["raw_dir"], self.config_path))
        self.output_dir = str(resolve_data_path(
            self.config["paths"]["processed_dir"], self.config_path))

    def run(self):
        """执行完整的数据集构建流程。"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        logger.info("=" * 60)
        logger.info("TREAD highD 数据集构建开始")
        logger.info("原始数据: %s", self.raw_dir)
        logger.info("输出目录: %s", self.output_dir)
        logger.info("=" * 60)

        ensure_dir(self.output_dir)

        # 收集所有事件
        all_events: List[EventRecord] = []

        ids = resolve_recording_ids(self.raw_dir, self.config.get("recordings", {}))
        logger.info("将处理 recording IDs: %s", ids)

        for rid in tqdm(ids, desc="Processing recordings"):
            try:
                events = self.process_recording(rid)
                all_events.extend(events)
            except Exception as e:
                logger.error("Recording %02d 处理失败: %s", rid, e, exc_info=True)
                continue

        if not all_events:
            logger.warning("未提取到任何事件!")
            return

        # 转 DataFrame
        events_df = events_to_dataframe(all_events)
        events_df = filter_events(events_df, self.config)

        # 构建轨迹张量
        logger.info("构建轨迹张量 ...")
        arrays = self.build_trajectory_arrays(all_events, events_df)

        # 划分数据集
        splits = self.build_splits(events_df)

        # 导出
        self.export(events_df, arrays, splits)

        # 可视化
        self._generate_visualizations(events_df)

        # 质量报告
        generate_quality_report(events_df, self.output_dir)

        logger.info("=" * 60)
        logger.info("TREAD highD 数据集构建完成!")
        logger.info("有效 following: %d", ((events_df["event_type"] == "following") & events_df["is_valid"]).sum())
        logger.info("有效 cut_in: %d", ((events_df["event_type"] == "cut_in") & events_df["is_valid"]).sum())
        logger.info("=" * 60)

    def process_recording(self, recording_id: int) -> List[EventRecord]:
        """处理单个 recording。"""
        cfg = self.config
        sampling = cfg.get("sampling", {})
        target_fps = sampling.get("target_fps", 10)

        recording = load_recording(self.raw_dir, recording_id)
        recording = normalize_driving_direction(recording)
        recording = filter_abnormal_tracks(recording, cfg)
        recording = resample_recording(recording, target_fps)

        # 抽取事件
        fol_events = extract_following_segments(recording, cfg)
        cin_events = extract_cutin_events(recording, cfg)

        # 窗口验证 — 使用实际帧索引
        pre = sampling.get("pre_anchor_steps", 32)
        post = sampling.get("post_anchor_steps", 31)

        valid_events = []
        for ev in fol_events + cin_events:
            # 从实际轨迹帧中提取窗口
            frames = get_window_from_track(
                recording, ev.ego_id, ev.target_id,
                ev.anchor_frame, pre, post,
            )
            if frames is None:
                ev.is_valid = False
                ev.filter_reason = "insufficient_window_frames"
                valid_events.append(ev)
                continue

            ok, reason = validate_window(
                recording, ev.ego_id, ev.target_id, frames, cfg)
            if not ok:
                ev.is_valid = False
                ev.filter_reason = reason
            valid_events.append(ev)

        return valid_events

    def build_trajectory_arrays(self, events, events_df):
        """为所有有效事件构建轨迹张量。"""
        valid_mask = events_df["is_valid"].values
        n_valid = int(valid_mask.sum())

        sampling = self.config.get("sampling", {})
        pre = sampling.get("pre_anchor_steps", 32)
        post = sampling.get("post_anchor_steps", 31)
        T = pre + 1 + post

        states_all = np.zeros((n_valid, T, 2, 11), dtype=np.float32)
        masks_all = np.zeros((n_valid, T, 2), dtype=bool)
        frame_ids_all = np.zeros((n_valid, T), dtype=np.int32)
        event_ids_all = []

        idx = 0
        # 按 recording 分组加载
        rec_cache = {}
        for i, ev in enumerate(events):
            if not valid_mask[i]:
                continue
            rid = ev.recording_id
            if rid not in rec_cache:
                try:
                    rec = load_recording(self.raw_dir, rid)
                    rec = normalize_driving_direction(rec)
                    rec = filter_abnormal_tracks(rec, self.config)
                    target_fps = sampling.get("target_fps", 10)
                    rec = resample_recording(rec, target_fps)
                    rec_cache[rid] = rec
                except Exception as e:
                    logger.error("重新加载 recording %02d 失败: %s", rid, e)
                    continue

            rec = rec_cache.get(rid)
            if rec is None:
                continue

            try:
                frames = get_window_from_track(
                    rec, ev.ego_id, ev.target_id,
                    ev.anchor_frame, pre, post,
                )
                if frames is None:
                    continue
                states, mask = build_state_tensor(ev, rec, self.config, frames=frames)
                states_all[idx] = states
                masks_all[idx] = mask
                frame_ids_all[idx] = frames
                event_ids_all.append(ev.event_id)
                idx += 1
            except Exception as e:
                logger.warning("事件 %s 张量构建失败: %s", ev.event_id, e)

        # 截断到实际数量
        states_all = states_all[:idx]
        masks_all = masks_all[:idx]
        frame_ids_all = frame_ids_all[:idx]

        return {
            "states": states_all,
            "masks": masks_all,
            "frame_ids": frame_ids_all,
            "event_ids": np.asarray(event_ids_all, dtype="S"),
        }

    def build_splits(self, events_df):
        """按 recording_id 划分 train/val/test。"""
        split_cfg = self.config.get("splits", {})
        train_r = split_cfg.get("train_ratio", 0.70)
        val_r = split_cfg.get("val_ratio", 0.15)
        seed = split_cfg.get("random_seed", 42)

        rng = np.random.RandomState(seed)
        rec_ids = sorted(events_df["recording_id"].unique())
        rng.shuffle(rec_ids)

        n = len(rec_ids)
        n_train = max(1, int(n * train_r))
        n_val = max(1, int(n * val_r))

        train_recs = list(map(int, rec_ids[:n_train]))
        val_recs = list(map(int, rec_ids[n_train:n_train + n_val]))
        test_recs = list(map(int, rec_ids[n_train + n_val:]))

        splits = {
            "train_recordings": train_recs,
            "val_recordings": val_recs,
            "test_recordings": test_recs,
            "train_event_ids": events_df[events_df["recording_id"].isin(train_recs)]["event_id"].tolist(),
            "val_event_ids": events_df[events_df["recording_id"].isin(val_recs)]["event_id"].tolist(),
            "test_event_ids": events_df[events_df["recording_id"].isin(test_recs)]["event_id"].tolist(),
        }

        logger.info("数据划分: train=%d recs, val=%d recs, test=%d recs",
                     len(train_recs), len(val_recs), len(test_recs))
        return splits

    def export(self, events_df, arrays, splits):
        """导出所有输出文件。"""
        out = Path(self.output_dir)
        ensure_dir(out)

        # events.csv
        if self.config.get("output", {}).get("save_csv", True):
            events_df.to_csv(out / "events.csv", index=False)
            logger.info("已保存 events.csv")

        # trajectories.h5
        if self.config.get("output", {}).get("save_h5", True):
            h5_path = out / "trajectories.h5"

            with h5py.File(h5_path, "w") as f:
                f.attrs["metadata_source"] = "events.csv"
                eg = f.create_group("events")
                eg.create_dataset("event_id", data=arrays["event_ids"])

                # trajectories group
                tg = f.create_group("trajectories")
                tg.create_dataset("states", data=arrays["states"], compression="gzip")
                tg.create_dataset("mask", data=arrays["masks"])
                tg.create_dataset("frame_ids", data=arrays["frame_ids"])

            logger.info("已保存 trajectories.h5, shape=%s", arrays["states"].shape)

        # splits.json
        save_json(splits, out / "splits.json")

        # normalization_stats.json
        self._save_normalization_stats(events_df, arrays, splits, out)

    def _save_normalization_stats(self, events_df, arrays, splits, out):
        """保存归一化统计量 (仅用训练集)。"""
        from .schema import FEATURE_NAMES

        train_ids = set(splits.get("train_event_ids", []))
        event_ids = np.asarray([eid.decode("utf-8") if isinstance(eid, bytes) else str(eid)
                                for eid in arrays.get("event_ids", [])])
        train_mask = np.isin(event_ids, list(train_ids))
        train_indices = np.where(train_mask)[0]

        states = arrays["states"]
        if len(train_indices) > 0 and len(states) > 0:
            train_states = states[train_indices]
            # 展平为 (N*T*A, F)
            flat = train_states.reshape(-1, train_states.shape[-1])
        else:
            flat = states.reshape(-1, states.shape[-1]) if len(states) > 0 else np.zeros((1, 11))

        stats = {
            "feature_names": FEATURE_NAMES,
            "mean": flat.mean(axis=0).tolist(),
            "std": flat.std(axis=0).tolist(),
            "min": flat.min(axis=0).tolist(),
            "max": flat.max(axis=0).tolist(),
        }

        # 风险分位数 (仅训练集)
        valid_df = events_df[events_df["is_valid"]]
        risk_q = {}
        for etype in ["cut_in", "following"]:
            sub = valid_df[(valid_df["event_type"] == etype) & valid_df["event_id"].isin(train_ids)]
            if len(sub) > 0:
                scores = sub["risk_score"].replace([np.inf, -np.inf], np.nan).dropna()
                if len(scores) > 0:
                    risk_q[etype] = {
                        "q90": float(scores.quantile(0.90)),
                        "q95": float(scores.quantile(0.95)),
                        "q99": float(scores.quantile(0.99)),
                    }
                else:
                    risk_q[etype] = {"q90": 0.0, "q95": 0.0, "q99": 0.0}
            else:
                risk_q[etype] = {"q90": 0.0, "q95": 0.0, "q99": 0.0}
        stats["risk"] = risk_q

        save_json(stats, out / "normalization_stats.json")

    def _generate_visualizations(self, events_df):
        """生成可视化图表。"""
        if not self.config.get("output", {}).get("save_figures", True):
            return

        fig_dir = Path(self.output_dir) / "figures"
        ensure_dir(fig_dir)

        # 风险分布图
        for etype in ["cut_in", "following"]:
            plot_risk_distribution(events_df, etype, str(fig_dir / f"risk_distribution_{etype}.png"))

        # TTC vs DRAC 散点图
        plot_ttc_drac_scatter(events_df, str(fig_dir / "ttc_drac_scatter.png"))

        logger.info("可视化图表已保存至 %s", fig_dir)
