from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(market: pd.DataFrame, window: int = 14) -> pd.Series:
    if "atr_5m" in market:
        return market["atr_5m"].astype(float)
    return _true_range(market).rolling(window, min_periods=window).mean()


def _future_rolling_sum(flag: pd.Series, horizon_bars: int) -> pd.Series:
    return flag.shift(-1).iloc[::-1].rolling(horizon_bars, min_periods=horizon_bars).sum().iloc[::-1]


def _future_rolling_max(values: pd.Series, horizon_bars: int) -> pd.Series:
    return values.shift(-1).iloc[::-1].rolling(horizon_bars, min_periods=horizon_bars).max().iloc[::-1]


def _future_rolling_min(values: pd.Series, horizon_bars: int) -> pd.Series:
    return values.shift(-1).iloc[::-1].rolling(horizon_bars, min_periods=horizon_bars).min().iloc[::-1]


def _bars_for_hours(hours: float, bar_minutes: float) -> int:
    if hours <= 0:
        raise ValueError("range_break_label horizons must be positive")
    if bar_minutes <= 0:
        raise ValueError("range_break_label bar_minutes must be positive")
    return max(1, int(round(hours * 60.0 / bar_minutes)))


def build_range_break_labels(market: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    label_config = config.get("range_break_label", config)
    index = market.index
    if index.tz is None:
        raise ValueError("range-break label index must be timezone-aware")
    for column in ["high", "low", "close"]:
        if column not in market.columns:
            raise ValueError(f"range-break labels require market column: {column}")

    lookback = int(label_config.get("range_lookback_bars", 48))
    if lookback <= 1:
        raise ValueError("range_break_label.range_lookback_bars must be > 1")
    atr_buffer = float(label_config.get("breakout_atr_buffer", 0.25))
    extension_atr = float(label_config.get("extension_atr", 0.75))
    min_persistent_closes = int(label_config.get("min_persistent_closes", 2))
    if min_persistent_closes <= 0:
        raise ValueError("range_break_label.min_persistent_closes must be positive")
    bar_minutes = float(label_config.get("bar_minutes", 5))
    horizons = [float(value) for value in label_config.get("diagnostic_horizons_hours", [6, 12, 24])]
    primary = float(label_config.get("primary_horizon_hours", 6))
    if primary not in horizons:
        horizons = [primary, *horizons]

    close = market["close"].astype(float)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    atr = _atr(market)
    prior_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low = low.rolling(lookback, min_periods=lookback).min().shift(1)
    buffer = atr * atr_buffer
    break_up_now = close.gt(prior_high + buffer)
    break_down_now = close.lt(prior_low - buffer)

    trend_alignment = market.get("trend_alignment_score", pd.Series(0.0, index=index)).fillna(0.0).astype(float)
    range_expansion = market.get("range_expansion_ratio", pd.Series(0.0, index=index)).fillna(0.0).astype(float)
    realized_volatility_ratio = (
        market.get("realized_volatility_ratio", pd.Series(0.0, index=index)).fillna(0.0).astype(float)
    )
    trend_threshold = float(label_config.get("min_abs_trend_alignment", 0.65))
    range_threshold = float(label_config.get("min_range_expansion_ratio", 1.25))
    vol_threshold = float(label_config.get("min_realized_volatility_ratio", 1.25))
    up_confirmation = (
        trend_alignment.ge(trend_threshold) | range_expansion.ge(range_threshold) | realized_volatility_ratio.ge(vol_threshold)
    )
    down_confirmation = (
        trend_alignment.le(-trend_threshold)
        | range_expansion.ge(range_threshold)
        | realized_volatility_ratio.ge(vol_threshold)
    )

    out = pd.DataFrame(
        {
            "range_break_prior_high": prior_high,
            "range_break_prior_low": prior_low,
            "range_break_buffer": buffer,
            "range_break_up_now": break_up_now.fillna(False).astype(int),
            "range_break_down_now": break_down_now.fillna(False).astype(int),
        },
        index=index,
    )

    base_valid = prior_high.notna() & prior_low.notna() & atr.notna()
    for hours in horizons:
        horizon_bars = _bars_for_hours(hours, bar_minutes)
        suffix = f"{int(hours)}h" if float(hours).is_integer() else f"{hours:g}h"
        future_break_up_count = _future_rolling_sum(break_up_now.fillna(False), horizon_bars)
        future_break_down_count = _future_rolling_sum(break_down_now.fillna(False), horizon_bars)
        future_high = _future_rolling_max(high, horizon_bars)
        future_low = _future_rolling_min(low, horizon_bars)
        future_up_confirm = _future_rolling_sum(up_confirmation.fillna(False), horizon_bars).ge(1)
        future_down_confirm = _future_rolling_sum(down_confirmation.fillna(False), horizon_bars).ge(1)
        up_extension = future_high.gt(prior_high + buffer + atr * extension_atr)
        down_extension = future_low.lt(prior_low - buffer - atr * extension_atr)
        full_future = future_high.notna() & future_low.notna()
        valid = base_valid & full_future
        short_danger = (
            (future_break_up_count.ge(min_persistent_closes) | up_extension.fillna(False)) & future_up_confirm.fillna(False)
        )
        long_danger = (
            (future_break_down_count.ge(min_persistent_closes) | down_extension.fillna(False))
            & future_down_confirm.fillna(False)
        )
        out[f"range_break_danger_short_{suffix}"] = np.where(valid, short_danger.astype(int), np.nan)
        out[f"range_break_danger_long_{suffix}"] = np.where(valid, long_danger.astype(int), np.nan)
        out[f"range_break_label_valid_{suffix}"] = valid.astype(int)

    out.index.name = market.index.name
    return out


def label_column_for_side(side: str, horizon_hours: float) -> str:
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    suffix = f"{int(horizon_hours)}h" if float(horizon_hours).is_integer() else f"{horizon_hours:g}h"
    return f"range_break_danger_{side}_{suffix}"
