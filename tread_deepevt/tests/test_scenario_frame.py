"""Unit tests for scenario_frame canonical schema and transforms."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tread_deepevt.src.scenario_frame import (  # noqa: E402
    CUTIN_CONTEXT_TO_CANONICAL,
    FOLLOWING_CONTEXT_TO_CANONICAL,
    SCENARIO_CONTEXT_SCHEMA_VERSION,
    build_canonical_context,
    compute_ego_initial_frame,
    ego_to_world_xy,
    world_to_ego_states,
)


def _make_state(x, y, vx, vy, ax=0.0, ay=0.0):
    return np.array([x, y, vx, vy, ax, ay], dtype=np.float32)


def test_world_to_ego_zero_heading_is_translation():
    # ego heading already aligned with +x -> no rotation, only translation
    states_world = np.zeros((3, 2, 6), dtype=np.float32)
    states_world[:, 0] = _make_state(100.0, 5.0, 20.0, 0.0)
    states_world[:, 1] = _make_state(120.0, 5.0, 18.0, 0.0)
    frame = compute_ego_initial_frame(states_world[0, 0])
    ego_frame_states = world_to_ego_states(states_world, frame)
    assert np.allclose(ego_frame_states[0, 0, :2], 0.0)
    assert np.isclose(ego_frame_states[0, 1, 0], 20.0)
    assert np.isclose(ego_frame_states[0, 1, 1], 0.0)
    # velocities are not translated
    assert np.isclose(ego_frame_states[0, 0, 2], 20.0)
    assert np.isclose(ego_frame_states[0, 1, 2], 18.0)


def test_world_to_ego_roundtrip_with_rotation():
    states_world = np.zeros((1, 2, 6), dtype=np.float32)
    states_world[0, 0] = _make_state(10.0, -2.0, 0.0, 0.0)
    states_world[0, 1] = _make_state(13.0, 2.0, 0.0, 0.0)
    frame = compute_ego_initial_frame(
        states_world[0, 0], world_heading_x=0.0, world_heading_y=1.0,
    )
    ego_states = world_to_ego_states(states_world, frame)
    # ego at origin
    assert np.allclose(ego_states[0, 0, :2], 0.0, atol=1e-6)
    # target in ego frame: rotation aligns world +y to ego +x
    expected_x = 2.0 - (-2.0)
    expected_y = -(13.0 - 10.0)
    assert np.isclose(ego_states[0, 1, 0], expected_x, atol=1e-5)
    assert np.isclose(ego_states[0, 1, 1], expected_y, atol=1e-5)

    # roundtrip
    target_back_world = ego_to_world_xy(ego_states[0, 1, :2], frame)
    assert np.allclose(target_back_world, states_world[0, 1, :2], atol=1e-5)


def test_build_canonical_context_following_fields():
    states = np.zeros((128, 2, 6), dtype=np.float32)
    # ego at origin moving +x at 20m/s
    states[:, 0, 2] = 20.0
    # target 25m ahead at 18m/s
    states[:, 1, 0] = 25.0
    states[:, 1, 2] = 18.0
    ctx = build_canonical_context(
        event_id="fol_test", event_type="following",
        states_ego_frame=states,
        ego_length=4.5, ego_width=1.8,
        target_length=4.5, target_width=1.8,
        fps=25.0, prefix_steps=1,
        source_lane=2, target_lane=2,
    )
    assert ctx.schema_version == SCENARIO_CONTEXT_SCHEMA_VERSION
    assert ctx.ego_x0 == 0.0 and ctx.ego_y0 == 0.0
    assert np.isclose(ctx.ego_v0, 20.0)
    assert np.isclose(ctx.target_v0, 18.0)
    # target_center_x0 = 25.0 (actual center x in ego-initial frame)
    assert np.isclose(ctx.target_center_x0, 25.0)
    assert np.isclose(ctx.target_center_y0, 0.0)
    # initial_gap = 25.0 - 0.5*(4.5 + 4.5) = 20.5
    assert np.isclose(ctx.initial_gap, 20.5)
    assert np.isclose(ctx.initial_lateral_offset, 0.0)
    # target_dx0 kept for backward compat = initial_gap
    assert np.isclose(ctx.target_dx0, 20.5)
    assert np.isclose(ctx.relative_speed_0, 2.0)
    assert np.isclose(ctx.time_horizon_s, 128 / 25.0)
    assert np.isclose(ctx.prefix_horizon_s, 1.0 / 25.0)
    assert ctx.same_lane_initial is True


def test_canonical_mapping_keys_unique_and_cover_features():
    fol = FOLLOWING_CONTEXT_TO_CANONICAL
    cin = CUTIN_CONTEXT_TO_CANONICAL
    # all canonical references must be a path: "<field>" or "extras.<key>"
    for k, v in fol.items():
        assert v.startswith("extras.") or v in {
            "ego_v0", "target_v0", "relative_speed_0",
            "initial_gap", "ego_ax0", "target_ax0",
        }, f"following key {k} has unexpected canonical path {v}"
    for k, v in cin.items():
        assert v.startswith("extras.") or v in {
            "ego_v0", "target_v0", "relative_speed_0",
            "initial_gap", "initial_lateral_offset",
            "target_vy0", "target_ax0", "target_ay0",
        }, f"cut_in key {k} has unexpected canonical path {v}"


def test_zero_heading_vector_falls_back_to_identity():
    s = _make_state(7.0, 3.0, 0.0, 0.0)
    frame = compute_ego_initial_frame(s, world_heading_x=0.0, world_heading_y=0.0)
    # degenerate heading -> rotation defaults to identity
    assert np.isclose(frame["rot_cos"], 1.0)
    assert np.isclose(frame["rot_sin"], 0.0)


def test_context_to_scenario_consistency():
    """target_center_x0 = initial_gap + 0.5*(L_ego + L_target)"""
    ego_len, tgt_len = 4.5, 4.5
    states = np.zeros((10, 2, 6), dtype=np.float32)
    states[:, 1, 0] = 30.0
    ctx = build_canonical_context(
        event_id="test", event_type="following",
        states_ego_frame=states,
        ego_length=ego_len, ego_width=1.8,
        target_length=tgt_len, target_width=1.8,
        fps=25.0, prefix_steps=1,
    )
    expected_gap = 30.0 - 0.5 * (ego_len + tgt_len)
    assert np.isclose(ctx.initial_gap, expected_gap)
    assert np.isclose(ctx.target_center_x0, ctx.initial_gap + 0.5 * (ego_len + tgt_len))
    assert ctx.ego_x0 == 0.0 and ctx.ego_y0 == 0.0
    assert np.isclose(ctx.relative_speed_0, ctx.ego_v0 - ctx.target_v0)
