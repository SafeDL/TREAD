"""
test_windowing.py — 窗口构建单元测试
======================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from tread_highd.windowing import get_window_frames


class TestGetWindowFrames:
    def test_basic(self):
        frames = get_window_frames(100, 32, 31)
        assert len(frames) == 64
        assert frames[0] == 68
        assert frames[-1] == 131
        assert frames[32] == 100  # anchor at center

    def test_consecutive(self):
        frames = get_window_frames(50, 10, 9)
        diffs = np.diff(frames)
        assert np.all(diffs == 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
