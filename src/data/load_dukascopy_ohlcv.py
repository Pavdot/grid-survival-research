from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.data.validate_data import validate_ohlcv
from src.utils.asset_paths import normalize_asset_id, processed_ohlcv_path
from src.utils.logging import get_logger
from src.utils.time_utils import timeframe_to_minutes


LOGGER = get_logger(__name__)

DUKASCOPY_TIMEFRAME_TOKENS = {
    "5m": "m5",
    "15m": "m15",
    "30m": "m30",
    "1h": "h1",
}

REQUIRED_DUKASCOPY_COLUMNS = {
    "datetime_utc",
    "open",
    "high",
    "low",
    "close",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "volume",
    "spread_close",
    "spread_avg",
    "spread_bps_close",
    "tick_count",
}

TIMEZONE_MARKER = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


def _find_source(input_path: Path, asset_id: str, timeframe: str) -> Path:
    if input_path.is_file():
        return input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Dukascopy input path does not exist: {input_path}")
    token = DUKASCOPY_TIMEFRAME_TOKENS.get(timeframe)
    if token is None:
        raise ValueError(f"Unsupported Dukascopy timeframe: {timeframe}")
    patterns = [
        f"{asset_id}_{token}_*.parquet",
        f"{asset_id}_{token}_*.csv",
        f"*{asset_id}*{token}*.parquet",
        f"*{asset_id}*{token}*.csv",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(input_path.glob(pattern)))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(f"No {asset_id} {timeframe} Dukascopy CSV/Parquet found in {input_path}")
    return matches[0]


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported Dukascopy file extension: {path.suffix}")


def _parse_utc(series: pd.Series) -> pd.Series:
    as_text = series.astype(str)
    if not as_text.map(lambda value: bool(TIMEZONE_MARKER.search(value))).all():
        raise ValueError("datetime_utc must contain timezone-aware UTC timestamps")
    return pd.to_datetime(as_text, utc=True, errors="raise")


def normalize_dukascopy_ohlcv(df: pd.DataFrame, timeframe: str = "5m", limit: int | None = None) -> pd.DataFrame:
    missing = REQUIRED_DUKASCOPY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing Dukascopy columns: {sorted(missing)}")
    if limit is not None:
        df = df.iloc[:limit].copy()
    else:
        df = df.copy()
    if df.empty:
        raise ValueError("No Dukascopy rows provided")

    numeric_columns = [column for column in df.columns if column not in {"datetime_utc"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    open_datetime = _parse_utc(df["datetime_utc"])
    minutes = timeframe_to_minutes(timeframe)
    close_datetime = open_datetime + pd.Timedelta(minutes=minutes)
    df["open_datetime"] = open_datetime
    df["close_datetime"] = close_datetime
    df["open_time"] = (open_datetime.astype("int64") // 1_000_000).astype("int64")
    df["close_time"] = (close_datetime.astype("int64") // 1_000_000 - 1).astype("int64")

    df = df.set_index(close_datetime)
    df.index.name = "timestamp"
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("Dukascopy OHLCV contains duplicate timestamps")

    required_numeric = ["open", "high", "low", "close", "volume", "spread_close", "spread_avg", "spread_bps_close"]
    if df[required_numeric].isna().any().any():
        bad_cols = df[required_numeric].columns[df[required_numeric].isna().any()].tolist()
        raise ValueError(f"Dukascopy OHLCV contains NaNs after normalization: {bad_cols}")
    if (df[["spread_close", "spread_avg", "spread_bps_close"]] <= 0).any().any():
        raise ValueError("Dukascopy spread columns must be positive")

    now_utc = pd.Timestamp.now("UTC")
    df = df[df.index <= now_utc]
    validate_ohlcv(df, timeframe=timeframe)
    return df


def load_dukascopy_ohlcv(
    input_path: Path,
    asset_id: str = "xauusd",
    timeframe: str = "5m",
    limit: int | None = None,
) -> pd.DataFrame:
    asset = normalize_asset_id(asset_id)
    source = _find_source(input_path, asset, timeframe)
    LOGGER.info("Loading Dukascopy %s %s data from %s", asset, timeframe, source)
    return normalize_dukascopy_ohlcv(_read_source(source), timeframe=timeframe, limit=limit)


def write_processed(df: pd.DataFrame, asset_id: str, timeframe: str) -> Path:
    output = processed_ohlcv_path(asset_id, timeframe)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    LOGGER.info("Wrote %s rows to %s", len(df), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local Dukascopy OHLCV with bid/ask spread columns.")
    parser.add_argument("--asset", default="xauusd")
    parser.add_argument("--input", required=True, help="Dukascopy CSV/Parquet file or directory.")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    df = load_dukascopy_ohlcv(Path(args.input), asset_id=args.asset, timeframe=args.timeframe, limit=args.limit)
    write_processed(df, args.asset, args.timeframe)


if __name__ == "__main__":
    main()
