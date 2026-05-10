"""
test_coordinate.py — 坐标转换单元测试
=======================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
from tread_highd.coordinate import to_ego_centric


class TestEgoCentric:
    def _make_track(self, frames, x, vx, y=0.0, vy=0.0):
        """辅助: 构建简单 track DataFrame。"""
        n = len(frames)
        return pd.DataFrame({
            "x": np.full(n, x) if isinstance(x, (int, float)) else x,
            "y": np.full(n, y),
            "xVelocity": np.full(n, vx) if isinstance(vx, (int, float)) else vx,
            "yVelocity": np.full(n, vy),
            "xAcceleration": np.zeros(n),
            "yAcceleration": np.zeros(n),
            "laneId": np.ones(n),
            "width": np.full(n, 4.5),
            "height": np.full(n, 1.8),
        }, index=pd.Index(frames, name="frame"))

    def test_ego_relative_zero(self):
        """ego 的相对坐标应为 0"""
        frames = np.arange(1, 11)
        ego = self._make_track(frames, 100.0, 30.0)
        tgt = self._make_track(frames, 120.0, 25.0)
        states, mask = to_ego_centric(ego, tgt, frames)
        # ego dx, dy, dvx, dvy 应为 0
        np.testing.assert_allclose(states[:, 0, :4], 0.0)
        assert np.all(mask[:, 0])

    def test_target_relative(self):
        frames = np.arange(1, 6)
        ego = self._make_track(frames, 100.0, 30.0)
        tgt = self._make_track(frames, 120.0, 25.0)
        states, mask = to_ego_centric(ego, tgt, frames)
        # target dx = 120 - 100 = 20
        np.testing.assert_allclose(states[:, 1, 0], 20.0)
        # target dvx = 25 - 30 = -5
        np.testing.assert_allclose(states[:, 1, 2], -5.0)

    def test_output_shape(self):
        frames = np.arange(1, 65)
        ego = self._make_track(frames, 100.0, 30.0)
        tgt = self._make_track(frames, 120.0, 25.0)
        states, mask = to_ego_centric(ego, tgt, frames)
        assert states.shape == (64, 2, 11)
        assert mask.shape == (64, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
