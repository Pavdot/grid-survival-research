from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.validate_data import load_processed
from src.features.range_features import add_range_features
from src.features.session_features import add_session_features
from src.features.trend_features import add_trend_alignment, add_trend_features, timeframe_trend_features
from src.features.volatility_features import add_timeframe_atr, add_volatility_features
from src.features.volume_features import add_volume_features
from src.regimes.regime_rules import add_regime_columns
from src.utils.asset_paths import feature_path, processed_ohlcv_path
from src.utils.config_loader import load_settings
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def load_timeframe(timeframe: str, asset: str = "btcusdt", limit: int | None = None) -> pd.DataFrame:
    df = load_processed(processed_ohlcv_path(asset, timeframe))
    if limit is not None:
        df = df.iloc[:limit]
    return df


def build_feature_frame(asset: str = "btcusdt", limit: int | None = None) -> pd.DataFrame:
    settings = load_settings()
    df_5m = load_timeframe("5m", asset=asset, limit=limit)

    vol = add_volatility_features(df_5m)
    trend = add_trend_features(df_5m)
    ranges = add_range_features(df_5m)
    volume = add_volume_features(df_5m, ranges)
    session = add_session_features(df_5m.index)

    features = pd.concat([vol, trend, ranges, volume, session], axis=1)

    for timeframe in settings["data"]["higher_timeframes"]:
        tf_df = load_timeframe(timeframe, asset=asset)
        if limit is not None:
            tf_df = tf_df.loc[tf_df.index <= df_5m.index[-1]]
        features = add_timeframe_atr(features, tf_df, timeframe, df_5m.index)
        features = features.join(timeframe_trend_features(tf_df, timeframe, df_5m.index))

    features = add_trend_alignment(features)
    features = add_regime_columns(features)
    features = features.replace([np.inf, -np.inf], np.nan)
    features.index.name = "timestamp"
    return features


def write_features(features: pd.DataFrame, asset: str = "btcusdt") -> str:
    output = feature_path(asset)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output)
    LOGGER.info("Wrote %s feature rows and %s columns to %s", len(features), len(features.columns), output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe grid survival features.")
    parser.add_argument("--asset", default="btcusdt")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    write_features(build_feature_frame(asset=args.asset, limit=args.limit), asset=args.asset)


if __name__ == "__main__":
    main()
