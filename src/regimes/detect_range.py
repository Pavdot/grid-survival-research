from __future__ import annotations

import pandas as pd


def detect_range_regime(features: pd.DataFrame) -> pd.Series:
    position = features.get("price_position_in_range", pd.Series(index=features.index, dtype=float))
    adx_15m = features.get("adx_15m", pd.Series(index=features.index, dtype=float))
    adx_1h = features.get("adx_1h", pd.Series(index=features.index, dtype=float))
    vol_ratio = features.get("realized_volatility_ratio", pd.Series(index=features.index, dtype=float))
    trend_alignment = features.get("trend_alignment_score", pd.Series(index=features.index, dtype=float))

    return (
        position.between(0.15, 0.85)
        & adx_15m.lt(28)
        & adx_1h.lt(30)
        & vol_ratio.lt(1.60)
        & trend_alignment.abs().lt(0.75)
    ).fillna(False)

