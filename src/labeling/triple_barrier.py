from __future__ import annotations

import pandas as pd


def triple_barrier_label(
    close: pd.Series,
    start_pos: int,
    upper_pct: float,
    lower_pct: float,
    max_bars: int,
) -> int:
    """Simple closed-candle triple-barrier helper used for diagnostics."""
    entry = float(close.iloc[start_pos])
    upper = entry * (1 + upper_pct)
    lower = entry * (1 - lower_pct)
    horizon = close.iloc[start_pos + 1 : start_pos + max_bars + 1]
    for value in horizon:
        if value >= upper:
            return 1
        if value <= lower:
            return -1
    return 0

