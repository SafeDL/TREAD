"""
test_risk_metrics.py — 风险指标单元测试
========================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from tread_highd.risk_metrics import (
    compute_gap, compute_ttc, compute_thw, compute_drac,
    compute_instant_risk, compute_trajectory_risk, entropy_weight_method,
)


class TestComputeGap:
    def test_positive_gap(self):
        ego_x = np.array([0.0, 1.0, 2.0])
        target_x = np.array([10.0, 11.0, 12.0])
        gap = compute_gap(ego_x, target_x, 4.0, 4.0)
        np.testing.assert_allclose(gap, [6.0, 6.0, 6.0])

    def test_zero_gap(self):
        gap = compute_gap(np.array([0.0]), np.array([4.0]), 4.0, 4.0)
        np.testing.assert_allclose(gap, [0.0])


class TestComputeTTC:
    def test_ego_slower_than_target(self):
        """ego 比 target 慢 → TTC = max_ttc"""
        gap = np.array([10.0])
        ego_vx = np.array([20.0])
        target_vx = np.array([25.0])  # target 更快
        ttc = compute_ttc(gap, ego_vx, target_vx, max_ttc=20.0)
        assert ttc[0] == 20.0

    def test_smaller_gap_smaller_ttc(self):
        """gap 越小 → TTC 越小"""
        ego_vx = np.array([30.0, 30.0])
        target_vx = np.array([20.0, 20.0])
        gap_large = np.array([20.0, 20.0])
        gap_small = np.array([5.0, 5.0])
        ttc_large = compute_ttc(gap_large, ego_vx, target_vx)
        ttc_small = compute_ttc(gap_small, ego_vx, target_vx)
        assert np.all(ttc_small < ttc_large)

    def test_zero_gap(self):
        """gap = 0 → gap ≤ eps → TTC = max_ttc (无有效碰撞计算)"""
        ttc = compute_ttc(np.array([0.0]), np.array([30.0]), np.array([20.0]))
        assert ttc[0] == 20.0  # gap not positive → max_ttc


class TestComputeDRAC:
    def test_higher_closing_speed_higher_drac(self):
        gap = np.array([10.0, 10.0])
        ego_vx_fast = np.array([35.0, 35.0])
        ego_vx_slow = np.array([25.0, 25.0])
        target_vx = np.array([20.0, 20.0])
        drac_fast = compute_drac(gap, ego_vx_fast, target_vx)
        drac_slow = compute_drac(gap, ego_vx_slow, target_vx)
        assert np.all(drac_fast > drac_slow)

    def test_no_closing(self):
        drac = compute_drac(np.array([10.0]), np.array([20.0]), np.array([25.0]))
        assert drac[0] == 0.0


class TestTrajectoryRisk:
    def test_softmax_close_to_max(self):
        """softmax risk 应接近 max risk"""
        instant = np.array([0.1, 0.5, 2.0, 0.3])
        risk = compute_trajectory_risk(instant, softmax_lambda=50.0)
        assert risk > 1.5  # should be close to 2.0
        assert risk <= 2.0 + 0.5  # bounded

    def test_empty(self):
        assert compute_trajectory_risk(np.array([])) == 0.0


class TestEntropyWeight:
    def test_equal_columns(self):
        data = np.ones((10, 3))
        w = entropy_weight_method(data)
        np.testing.assert_allclose(w, [1/3, 1/3, 1/3], atol=0.01)

    def test_sum_to_one(self):
        rng = np.random.RandomState(42)
        data = rng.rand(50, 4) + 0.1
        w = entropy_weight_method(data)
        assert abs(w.sum() - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
