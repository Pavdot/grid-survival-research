from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.asset_paths import processed_ohlcv_path
from src.utils.logging import get_logger
from src.utils.time_utils import ensure_utc_index, timeframe_to_minutes


LOGGER = get_logger(__name__)
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume", "open_time", "close_time"}


def validate_ohlcv(df: pd.DataFrame, timeframe: str = "5m") -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be monotonically increasing")
    if df.index.has_duplicates:
        raise ValueError("OHLCV index contains duplicate timestamps")
    if (df[["open", "high", "low", "close", "volume"]] < 0).any().any():
        raise ValueError("OHLCV values must be non-negative")
    if ((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))).any():
        raise ValueError("OHLC values are internally inconsistent")
    if len(df) > 2:
        expected = pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
        median_delta = df.index.to_series().diff().dropna().median()
        if median_delta != expected:
            raise ValueError(f"Unexpected median candle spacing: {median_delta}, expected {expected}")


def load_processed(path: Path) -> pd.DataFrame:
    return ensure_utc_index(pd.read_parquet(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed OHLCV data.")
    parser.add_argument("--asset", default="btcusdt")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    path = Path(args.path) if args.path else processed_ohlcv_path(args.asset, args.timeframe)
    df = load_processed(path)
    validate_ohlcv(df, timeframe=args.timeframe)
    LOGGER.info("Validated %s rows in %s", len(df), path)


if __name__ == "__main__":
    main()
