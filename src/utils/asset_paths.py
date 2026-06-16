from __future__ import annotations

from pathlib import Path

from src.utils.config_loader import configured_path


DEFAULT_ASSET_ID = "btcusdt"


def normalize_asset_id(asset_id: str | None) -> str:
    value = (asset_id or DEFAULT_ASSET_ID).strip().lower()
    if not value:
        raise ValueError("asset_id must not be empty")
    return value


def processed_ohlcv_path(asset_id: str | None, timeframe: str) -> Path:
    asset = normalize_asset_id(asset_id)
    return configured_path("processed_dir", f"{asset}_{timeframe}.parquet")


def feature_filename(asset_id: str | None) -> str:
    asset = normalize_asset_id(asset_id)
    if asset == DEFAULT_ASSET_ID:
        return "grid_features.parquet"
    return f"{asset}_grid_features.parquet"


def feature_path(asset_id: str | None) -> Path:
    return configured_path("features_dir", feature_filename(asset_id))
