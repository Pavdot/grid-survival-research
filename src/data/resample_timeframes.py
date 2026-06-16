from __future__ import annotations

import argparse

import pandas as pd

from src.data.validate_data import load_processed, validate_ohlcv
from src.utils.asset_paths import processed_ohlcv_path
from src.utils.config_loader import load_settings
from src.utils.logging import get_logger
from src.utils.time_utils import timeframe_to_pandas_freq


LOGGER = get_logger(__name__)


def resample_closed_ohlcv(df_5m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    freq = timeframe_to_pandas_freq(timeframe)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_asset_volume": "sum",
        "number_of_trades": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
        "open_datetime": "first",
        "close_datetime": "last",
        "open_time": "first",
        "close_time": "last",
        "bid_open": "first",
        "bid_high": "max",
        "bid_low": "min",
        "bid_close": "last",
        "bid_volume": "sum",
        "ask_open": "first",
        "ask_high": "max",
        "ask_low": "min",
        "ask_close": "last",
        "ask_volume": "sum",
        "spread_open": "first",
        "spread_high": "max",
        "spread_low": "min",
        "spread_close": "last",
        "spread_avg": "mean",
        "spread_bps_close": "last",
        "tick_count": "sum",
    }
    available_agg = {key: value for key, value in agg.items() if key in df_5m.columns}
    resampled = (
        df_5m.sort_index()
        .resample(freq, label="right", closed="right", origin="start_day")
        .agg(available_agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    resampled.index.name = "timestamp"
    return resampled


def rebuild_timeframes(asset: str = "btcusdt", limit: int | None = None) -> dict[str, pd.DataFrame]:
    settings = load_settings()
    source = processed_ohlcv_path(asset, "5m")
    df_5m = load_processed(source)
    if limit is not None:
        df_5m = df_5m.iloc[:limit]
    validate_ohlcv(df_5m, "5m")

    outputs: dict[str, pd.DataFrame] = {}
    for timeframe in settings["data"]["higher_timeframes"]:
        frame = resample_closed_ohlcv(df_5m, timeframe)
        validate_ohlcv(frame, timeframe)
        output = processed_ohlcv_path(asset, timeframe)
        frame.to_parquet(output)
        LOGGER.info("Wrote %s rows to %s", len(frame), output)
        outputs[timeframe] = frame
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild higher timeframes from closed 5m candles.")
    parser.add_argument("--asset", default="btcusdt")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rebuild_timeframes(asset=args.asset, limit=args.limit)


if __name__ == "__main__":
    main()
