from __future__ import annotations

import numpy as np
import pandas as pd


def add_volume_features(df: pd.DataFrame, range_features: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    volume_mean = df["volume"].rolling(96, min_periods=96).mean()
    volume_std = df["volume"].rolling(96, min_periods=96).std()
    features["volume_zscore"] = (df["volume"] - volume_mean) / volume_std
    features["relative_volume"] = df["volume"] / volume_mean
    short_volume = df["volume"].rolling(12, min_periods=12).mean()
    long_volume = df["volume"].rolling(96, min_periods=96).mean()
    features["volume_expansion_ratio"] = short_volume / long_volume
    features["price_move_per_volume"] = df["close"].pct_change().abs() / (df["volume"] + 1e-12)

    near_boundary = (
        range_features["price_position_in_range"].ge(0.80)
        | range_features["price_position_in_range"].le(0.20)
    )
    boundary_volume = df["volume"].where(near_boundary).rolling(96, min_periods=12).mean()
    features["volume_at_range_boundary"] = boundary_volume / long_volume
    return features.replace([np.inf, -np.inf], np.nan)

