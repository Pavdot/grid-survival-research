from __future__ import annotations

import numpy as np
import pandas as pd


def ema_slope(close: pd.Series, span: int = 20, periods: int = 3) -> pd.Series:
    ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
    return ema.pct_change(periods)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=window).sum()
    plus_di = 100 * plus_dm.rolling(window, min_periods=window).sum() / atr
    minus_di = 100 * minus_dm.rolling(window, min_periods=window).sum() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window, min_periods=window).mean()


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["ema_slope_5m"] = ema_slope(df["close"], span=20, periods=3)
    ema_20 = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    ema_100 = df["close"].ewm(span=100, adjust=False, min_periods=100).mean()
    features["price_distance_to_ema_20"] = (df["close"] - ema_20) / ema_20
    features["price_distance_to_ema_100"] = (df["close"] - ema_100) / ema_100

    rolling_high = df["high"].rolling(24, min_periods=24).max()
    rolling_low = df["low"].rolling(24, min_periods=24).min()
    features["higher_highs_count"] = (df["high"] >= rolling_high.shift(1)).rolling(24, min_periods=24).sum()
    features["lower_lows_count"] = (df["low"] <= rolling_low.shift(1)).rolling(24, min_periods=24).sum()
    return features.replace([np.inf, -np.inf], np.nan)


def timeframe_trend_features(tf_df: pd.DataFrame, timeframe: str, target_index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=tf_df.index)
    out[f"ema_slope_{timeframe}"] = ema_slope(tf_df["close"], span=20, periods=3)
    if timeframe in {"15m", "1h"}:
        out[f"adx_{timeframe}"] = adx(tf_df, 14)
    return out.reindex(target_index, method="ffill").replace([np.inf, -np.inf], np.nan)


def add_trend_alignment(features: pd.DataFrame) -> pd.DataFrame:
    slope_cols = [col for col in ["ema_slope_5m", "ema_slope_15m", "ema_slope_30m", "ema_slope_1h"] if col in features]
    signed = pd.concat([np.sign(features[col]) for col in slope_cols], axis=1)
    features["trend_alignment_score"] = signed.mean(axis=1)
    return features

