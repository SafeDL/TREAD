"""Event post-processing for process_highD."""
from __future__ import annotations
import pandas as pd
from dataclasses import asdict


def events_to_dataframe(events):
    """将 EventRecord 列表转换为 DataFrame"""
    if not events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in events])
