from __future__ import annotations

import pandas as pd


def ms_to_utc_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, unit="ms", utc=True)


def close_time_to_timestamp(close_time: pd.Series) -> pd.Series:
    """Convert Binance close time in ms to the next exact closed-candle boundary."""
    return ms_to_utc_datetime(close_time) + pd.Timedelta(milliseconds=1)


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index.name = "timestamp"
    return out


def timeframe_to_minutes(timeframe: str) -> int:
    units = {"m": 1, "h": 60, "d": 1440}
    suffix = timeframe[-1]
    if suffix not in units:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(timeframe[:-1]) * units[suffix]


def timeframe_to_pandas_freq(timeframe: str) -> str:
    minutes = timeframe_to_minutes(timeframe)
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"

