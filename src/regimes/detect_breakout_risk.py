from __future__ import annotations

import pandas as pd


def detect_breakout_risk(features: pd.DataFrame) -> pd.Series:
    trend_alignment = features.get("trend_alignment_score", pd.Series(index=features.index, dtype=float)).abs()
    range_expansion = features.get("range_expansion_ratio", pd.Series(index=features.index, dtype=float))
    candle_z = features.get("candle_range_zscore", pd.Series(index=features.index, dtype=float))
    vol_ratio = features.get("realized_volatility_ratio", pd.Series(index=features.index, dtype=float))
    position = features.get("price_position_in_range", pd.Series(index=features.index, dtype=float))
    boundary_pressure = position.ge(0.90) | position.le(0.10)

    return (
        trend_alignment.ge(0.95)
        | range_expansion.ge(2.5)
        | candle_z.ge(3.0)
        | vol_ratio.ge(2.0)
        | boundary_pressure
    ).fillna(False)

