from __future__ import annotations

import argparse

import pandas as pd

from src.backtesting.backtest_grid import run_grid_backtest
from src.backtesting.metrics import calculate_metrics
from src.data.validate_data import load_processed
from src.labeling.grid_risk import validate_strategy_config
from src.utils.config_loader import configured_path, load_strategy_config
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _prepare_market(limit: int | None = None) -> pd.DataFrame:
    market = load_processed(configured_path("processed_dir", "btcusdt_5m.parquet"))
    features = pd.read_parquet(configured_path("features_dir", "grid_features.parquet"))
    cols = [
        "atr_5m",
        "breakout_risk",
        "regime_allows_grid",
        "range_expansion_ratio",
        "realized_volatility_ratio",
    ]
    market = market.join(features[[col for col in cols if col in features.columns]], how="left")
    market["breakout_risk"] = market.get("breakout_risk", 0).fillna(0).astype(int)
    market["volatility_shock"] = (
        market.get("range_expansion_ratio", pd.Series(0, index=market.index)).fillna(0).ge(2.5)
        | market.get("realized_volatility_ratio", pd.Series(0, index=market.index)).fillna(0).ge(2.0)
    ).astype(int)
    if limit is not None:
        market = market.iloc[:limit]
    return market


def _load_scores(index: pd.Index) -> pd.Series | None:
    path = configured_path("model_reports_dir", "grid_survival_predictions.parquet")
    if not path.exists():
        LOGGER.warning("No model predictions found at %s; ML backtests will be skipped", path)
        return None
    predictions = pd.read_parquet(path)
    if "grid_survival_score" not in predictions.columns:
        return None
    return predictions["grid_survival_score"].reindex(index)


def run_comparative_backtests(limit: int | None = None) -> pd.DataFrame:
    config = load_strategy_config()
    risk = validate_strategy_config(config)
    market = _prepare_market(limit=limit)
    scores = _load_scores(market.index)

    strategies = {
        "constant_grid_no_ml": {
            "constant_size": True,
            "allow_open": pd.Series(True, index=market.index),
        },
        "light_progressive_no_ml": {
            "constant_size": False,
            "allow_open": pd.Series(True, index=market.index),
        },
        "regime_filtered_grid": {
            "constant_size": False,
            "allow_open": market.get("regime_allows_grid", pd.Series(False, index=market.index)).astype(bool),
        },
    }
    if scores is not None:
        strategies["ml_filtered_grid"] = {
            "constant_size": False,
            "allow_open": pd.Series(True, index=market.index),
            "scores": scores,
            "min_open_score": float(config["model_filter"]["min_survival_probability_open"]),
            "add_level_min_score": float(config["model_filter"]["min_survival_probability_add"]),
        }
        strategies["ml_filtered_grid_kill_switch"] = {
            "constant_size": False,
            "allow_open": pd.Series(True, index=market.index),
            "scores": scores,
            "min_open_score": float(config["model_filter"]["min_survival_probability_open"]),
            "add_level_min_score": float(config["model_filter"]["min_survival_probability_add"]),
            "kill_switch_threshold": float(config["model_filter"]["force_exit_below"]),
        }

    output_dir = configured_path("backtests_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    baseline_count = 0
    for name, kwargs in strategies.items():
        result = run_grid_backtest(market, risk, **kwargs)
        trades = result.trades
        trades.to_csv(output_dir / f"{name}_trades.csv", index=False)
        result.equity_curve.to_frame().to_csv(output_dir / f"{name}_equity.csv")
        metrics = calculate_metrics(result.equity_curve, trades)
        if name == "light_progressive_no_ml":
            baseline_count = int(metrics.get("number_of_grids", 0))
        metrics["strategy"] = name
        if name.startswith("ml_"):
            metrics["number_of_grids_saved_by_ml_filter"] = baseline_count - int(metrics.get("number_of_grids", 0))
        else:
            metrics["number_of_grids_saved_by_ml_filter"] = 0
        summary_rows.append(metrics)
        LOGGER.info("Backtested %s with %s grids", name, len(trades))

    summary = pd.DataFrame(summary_rows).set_index("strategy")
    summary.to_csv(output_dir / "backtest_summary.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comparative bounded-grid backtests.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_comparative_backtests(limit=args.limit)


if __name__ == "__main__":
    main()

