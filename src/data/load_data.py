from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.utils.config_loader import configured_path, load_settings, project_path
from src.utils.logging import get_logger
from src.utils.time_utils import close_time_to_timestamp, ms_to_utc_datetime


LOGGER = get_logger(__name__)

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _unwrap_kline(row: Any) -> list[Any]:
    if isinstance(row, dict) and "value" in row:
        return list(row["value"])
    if isinstance(row, (list, tuple)):
        return list(row)
    raise ValueError(f"Unsupported Binance kline row: {type(row)!r}")


def read_binance_json(path: Path) -> list[list[Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of klines: {path}")
    return [_unwrap_kline(row) for row in payload]


def normalize_klines(rows: Iterable[list[Any]], limit: int | None = None) -> pd.DataFrame:
    records = list(rows)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError("No kline rows provided")

    df = pd.DataFrame(records, columns=KLINE_COLUMNS[: len(records[0])])
    missing = [col for col in KLINE_COLUMNS[:7] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required kline columns: {missing}")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["open_time"] = pd.to_numeric(df["open_time"], errors="raise").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="raise").astype("int64")
    df["open_datetime"] = ms_to_utc_datetime(df["open_time"])
    df["close_datetime"] = ms_to_utc_datetime(df["close_time"])
    df["timestamp"] = close_time_to_timestamp(df["close_time"])

    now_utc = pd.Timestamp.now("UTC")
    df = df[df["timestamp"] <= now_utc]
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")
    df.index.name = "timestamp"

    required = ["open", "high", "low", "close", "volume"]
    if df[required].isna().any().any():
        bad_cols = df[required].columns[df[required].isna().any()].tolist()
        raise ValueError(f"OHLCV contains NaNs after normalization: {bad_cols}")
    return df


def load_local_or_raw(limit: int | None = None) -> pd.DataFrame:
    settings = load_settings()
    raw_config_path = Path(settings["paths"]["local_kline_json"])
    source = raw_config_path if raw_config_path.is_absolute() else project_path(raw_config_path)
    if not source.exists():
        alternative = project_path("data/raw/btcusdt_5m_klines.json")
        if alternative.exists():
            source = alternative
        else:
            raise FileNotFoundError(
                "No local kline JSON found. Expected either "
                f"{source} or {alternative}. Run src.data.download_binance first."
            )
    LOGGER.info("Loading Binance 5m klines from %s", source)
    return normalize_klines(read_binance_json(source), limit=limit)


def write_processed(df: pd.DataFrame, symbol: str = "btcusdt", timeframe: str = "5m") -> Path:
    output = configured_path("processed_dir", f"{symbol.lower()}_{timeframe}.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output)
    LOGGER.info("Wrote %s rows to %s", len(df), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize local Binance BTCUSDT 5m klines.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke runs.")
    args = parser.parse_args()
    df = load_local_or_raw(limit=args.limit)
    write_processed(df)


if __name__ == "__main__":
    main()
