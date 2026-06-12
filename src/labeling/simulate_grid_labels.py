from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.validate_data import load_processed
from src.labeling.grid_engine import simulate_grid_from_index
from src.labeling.grid_risk import validate_strategy_config
from src.utils.config_loader import configured_path, load_strategy_config
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _prepare_market(limit: int | None = None) -> pd.DataFrame:
    market = load_processed(configured_path("processed_dir", "btcusdt_5m.parquet"))
    features_path = configured_path("features_dir", "grid_features.parquet")
    if features_path.exists():
        features = pd.read_parquet(features_path)
        wanted = [
            "atr_5m",
            "breakout_risk",
            "range_expansion_ratio",
            "realized_volatility_ratio",
        ]
        market = market.join(features[[col for col in wanted if col in features.columns]], how="left")
    if "atr_5m" not in market:
        tr = pd.concat(
            [
                market["high"] - market["low"],
                (market["high"] - market["close"].shift(1)).abs(),
                (market["low"] - market["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        market["atr_5m"] = tr.rolling(14, min_periods=14).mean()
    market["breakout_risk"] = market.get("breakout_risk", 0).fillna(0).astype(int)
    market["volatility_shock"] = (
        market.get("range_expansion_ratio", pd.Series(0, index=market.index)).fillna(0).ge(2.5)
        | market.get("realized_volatility_ratio", pd.Series(0, index=market.index)).fillna(0).ge(2.0)
    ).astype(int)
    if limit is not None:
        market = market.iloc[:limit]
    return market


def simulate_labels(limit: int | None = None) -> pd.DataFrame:
    risk = validate_strategy_config(load_strategy_config())
    market = _prepare_market(limit=limit)
    horizon_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    end = max(0, len(market) - horizon_bars)
    rows = []
    for start_pos in range(end):
        if not np.isfinite(float(market.iloc[start_pos].get("atr_5m", np.nan))):
            continue
        result = simulate_grid_from_index(market, start_pos, risk)
        rows.append(result.to_dict())
    if not rows:
        raise ValueError("No labels generated; ensure enough rows and ATR history are available")
    labels = pd.DataFrame(rows).set_index("start_timestamp").sort_index()
    labels.index.name = "timestamp"
    return labels


def write_labels(labels: pd.DataFrame) -> str:
    output = configured_path("labels_dir", "grid_labels.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output)
    LOGGER.info("Wrote %s label rows to %s", len(labels), output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate grid labels for every eligible timestamp.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    write_labels(simulate_labels(limit=args.limit))


if __name__ == "__main__":
    main()

