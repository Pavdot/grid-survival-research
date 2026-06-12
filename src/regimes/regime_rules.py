from __future__ import annotations

import pandas as pd

from src.regimes.detect_breakout_risk import detect_breakout_risk
from src.regimes.detect_range import detect_range_regime


def add_regime_columns(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["is_range_regime"] = detect_range_regime(out).astype(int)
    out["breakout_risk"] = detect_breakout_risk(out).astype(int)
    out["regime_allows_grid"] = ((out["is_range_regime"] == 1) & (out["breakout_risk"] == 0)).astype(int)
    return out

