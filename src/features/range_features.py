from __future__ import annotations

import numpy as np
import pandas as pd


def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    high_1h = df["high"].rolling(12, min_periods=12).max()
    low_1h = df["low"].rolling(12, min_periods=12).min()
    high_4h = df["high"].rolling(48, min_periods=48).max()
    low_4h = df["low"].rolling(48, min_periods=48).min()

    features["rolling_high_1h"] = high_1h
    features["rolling_low_1h"] = low_1h
    features["rolling_high_4h"] = high_4h
    features["rolling_low_4h"] = low_4h

    range_width = high_4h - low_4h
    features["price_position_in_range"] = (df["close"] - low_4h) / range_width
    features["distance_to_range_high"] = (high_4h - df["close"]) / df["close"]
    features["distance_to_range_low"] = (df["close"] - low_4h) / df["close"]

    upper_near = features["price_position_in_range"] >= 0.80
    lower_near = features["price_position_in_range"] <= 0.20
    features["closes_near_upper_bound_count"] = upper_near.rolling(48, min_periods=48).sum()
    features["closes_near_lower_bound_count"] = lower_near.rolling(48, min_periods=48).sum()

    body = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    features["rejection_wick_score"] = (upper_wick + lower_wick) / (body + 1e-12)

    prior_high = high_4h.shift(1)
    prior_low = low_4h.shift(1)
    failed_up = (df["high"] > prior_high) & (df["close"] < prior_high)
    failed_down = (df["low"] < prior_low) & (df["close"] > prior_low)
    features["failed_breakout_count"] = (failed_up | failed_down).rolling(48, min_periods=48).sum()
    features["time_spent_top_quartile"] = (features["price_position_in_range"] >= 0.75).rolling(48, min_periods=48).mean()
    features["time_spent_bottom_quartile"] = (features["price_position_in_range"] <= 0.25).rolling(48, min_periods=48).mean()
    return features.replace([np.inf, -np.inf], np.nan)

