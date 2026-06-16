from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _true_range(df: pd.DataFrame) -> pd.Series:
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _atr(market: pd.DataFrame, window: int = 14) -> pd.Series:
    if "atr_5m" in market:
        return market["atr_5m"].astype(float)
    return _true_range(market).rolling(window, min_periods=window).mean()


def _fallback_trend_alignment(close: pd.Series) -> pd.Series:
    fast = close.ewm(span=20, adjust=False, min_periods=20).mean()
    slow = close.ewm(span=100, adjust=False, min_periods=100).mean()
    return np.sign(fast - slow).fillna(0.0)


def build_trend_escape_components(market: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    trend_config = config.get("trend_escape", config)
    index = market.index
    if index.tz is None:
        raise ValueError("trend escape index must be timezone-aware")
    lookback = int(trend_config.get("range_lookback_bars", 48))
    confirmation_bars = int(trend_config.get("confirmation_bars", 2))
    min_confirmations = int(trend_config.get("min_confirmations", 1))
    delay_bars = int(trend_config.get("delay_bars", 1))
    propagation_bars = int(trend_config.get("propagation_bars", 12))
    if lookback <= 1:
        raise ValueError("trend_escape.range_lookback_bars must be > 1")
    if confirmation_bars <= 0 or min_confirmations <= 0:
        raise ValueError("trend_escape confirmation settings must be positive")
    if min_confirmations > confirmation_bars:
        raise ValueError("trend_escape.min_confirmations cannot exceed confirmation_bars")
    if delay_bars < 0:
        raise ValueError("trend_escape.delay_bars must be non-negative")
    if propagation_bars <= 0:
        raise ValueError("trend_escape.propagation_bars must be positive")

    close = market["close"].astype(float)
    atr = _atr(market)
    prior_high = market["high"].astype(float).rolling(lookback, min_periods=lookback).max().shift(1)
    prior_low = market["low"].astype(float).rolling(lookback, min_periods=lookback).min().shift(1)
    buffer = atr * float(trend_config.get("breakout_atr_buffer", 0.25))
    break_up = close.gt(prior_high + buffer)
    break_down = close.lt(prior_low - buffer)
    direction = pd.Series(0, index=index, dtype=int)
    direction[break_up] = 1
    direction[break_down] = -1

    range_width_pct = ((prior_high - prior_low) / close).replace([np.inf, -np.inf], np.nan)
    if bool(trend_config.get("require_compression", False)):
        compression_lookback = int(trend_config.get("compression_lookback_bars", max(lookback * 3, lookback + 1)))
        if compression_lookback <= lookback:
            raise ValueError("trend_escape.compression_lookback_bars must exceed range_lookback_bars")
        quantile = float(trend_config.get("compression_quantile", 0.35))
        if quantile <= 0 or quantile >= 1:
            raise ValueError("trend_escape.compression_quantile must be within (0, 1)")
        compression_threshold = range_width_pct.rolling(compression_lookback, min_periods=compression_lookback).quantile(
            quantile
        ).shift(1)
        compression_ok = range_width_pct.le(compression_threshold)
    else:
        compression_ok = pd.Series(True, index=index)

    trend_alignment = market.get("trend_alignment_score", _fallback_trend_alignment(close)).astype(float)
    range_expansion = market.get("range_expansion_ratio", pd.Series(0.0, index=index)).fillna(0.0).astype(float)
    realized_volatility_ratio = (
        market.get("realized_volatility_ratio", pd.Series(0.0, index=index)).fillna(0.0).astype(float)
    )
    direction_aligned = (direction * np.sign(trend_alignment.fillna(0.0))).gt(0)
    trend_confirmed = (
        direction_aligned
        | trend_alignment.abs().ge(float(trend_config.get("min_abs_trend_alignment", 0.75)))
        | range_expansion.ge(float(trend_config.get("min_range_expansion_ratio", 1.5)))
        | realized_volatility_ratio.ge(float(trend_config.get("min_realized_volatility_ratio", 1.5)))
    )
    raw_break = (direction.ne(0) & compression_ok.fillna(False) & trend_confirmed.fillna(False)).astype(bool)
    if bool(trend_config.get("include_existing_breakout_risk", False)) and "breakout_risk" in market:
        raw_break = raw_break | market["breakout_risk"].fillna(0).astype(int).eq(1)

    confirmed = raw_break.astype(int).rolling(confirmation_bars, min_periods=confirmation_bars).sum().ge(min_confirmations)
    delayed = confirmed.shift(delay_bars).fillna(False).astype(bool)
    mask = delayed.astype(int).rolling(propagation_bars, min_periods=1).max().astype(bool)
    out = pd.DataFrame(
        {
            "prior_range_high": prior_high,
            "prior_range_low": prior_low,
            "range_width_pct": range_width_pct,
            "trend_escape_direction": direction,
            "range_break_up": break_up.fillna(False).astype(int),
            "range_break_down": break_down.fillna(False).astype(int),
            "trend_escape_raw": raw_break.astype(int),
            "trend_escape_confirmed": confirmed.fillna(False).astype(int),
            "trend_escape": mask.astype(int),
            "trend_escape_compression_ok": compression_ok.fillna(False).astype(int),
            "trend_escape_trend_confirmed": trend_confirmed.fillna(False).astype(int),
        },
        index=index,
    )
    out.index.name = market.index.name
    return out


def build_trend_escape_mask(market: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    components = build_trend_escape_components(market, config)
    mask = components["trend_escape"].astype(bool)
    mask.name = "trend_escape"
    return mask
