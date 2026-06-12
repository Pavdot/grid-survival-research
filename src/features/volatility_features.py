from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window, min_periods=window).mean()


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window, min_periods=window).std() * np.sqrt(window)


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    tr = true_range(df)
    features["atr_5m"] = atr(df, 14)
    features["atr_short"] = tr.rolling(12, min_periods=12).mean()
    features["atr_long"] = tr.rolling(96, min_periods=96).mean()
    features["atr_short_over_long"] = features["atr_short"] / features["atr_long"]
    features["realized_volatility_1h"] = realized_volatility(df["close"], 12)
    features["realized_volatility_4h"] = realized_volatility(df["close"], 48)
    features["realized_volatility_ratio"] = (
        features["realized_volatility_1h"] / features["realized_volatility_4h"]
    )

    middle = df["close"].rolling(20, min_periods=20).mean()
    std = df["close"].rolling(20, min_periods=20).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    features["bollinger_bandwidth"] = (upper - lower) / middle

    candle_range = df["high"] - df["low"]
    features["range_expansion_ratio"] = candle_range / candle_range.rolling(48, min_periods=48).median()
    range_mean = candle_range.rolling(96, min_periods=96).mean()
    range_std = candle_range.rolling(96, min_periods=96).std()
    features["candle_range_zscore"] = (candle_range - range_mean) / range_std
    return features.replace([np.inf, -np.inf], np.nan)


def add_timeframe_atr(
    base_features: pd.DataFrame,
    tf_df: pd.DataFrame,
    timeframe: str,
    target_index: pd.Index,
) -> pd.DataFrame:
    tf_features = pd.DataFrame(index=tf_df.index)
    tf_features[f"atr_{timeframe}"] = atr(tf_df, 14)
    return base_features.join(tf_features.reindex(target_index, method="ffill"))

