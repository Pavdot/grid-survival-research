from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.backtest_grid import run_grid_backtest
from src.backtesting.metrics import calculate_metrics
from src.backtesting.walk_forward import temporal_train_validation_test_split
from src.data.validate_data import load_processed
from src.labeling.grid_engine import GridSimulationResult, simulate_grid_from_index
from src.labeling.grid_risk import GridRiskConfig, validate_strategy_config
from src.utils.config_loader import load_settings, load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SEARCH_SPLIT = "validation"
EVALUATION_SPLITS = {"validation", "test"}
FORCED_COLUMNS = [
    "stopped_by_regime_break",
    "stopped_by_max_loss",
    "stopped_by_max_holding",
    "stopped_by_volatility_shock",
    "stopped_by_exposure",
    "stopped_by_kill_switch",
]


@dataclass(frozen=True)
class EconomyCandidate:
    take_profit_spacing_multiplier: float
    survival_min_net_profit_pct: float


def make_economy_candidates(config: dict[str, Any]) -> list[EconomyCandidate]:
    search = config["search"]
    candidates: list[EconomyCandidate] = []
    for tp_multiplier in search["take_profit_spacing_multipliers"]:
        tp_multiplier = float(tp_multiplier)
        if tp_multiplier <= 0:
            raise ValueError("take_profit_spacing_multipliers must be positive")
        for min_profit in search["survival_min_net_profit_pcts"]:
            min_profit = float(min_profit)
            if min_profit < 0:
                raise ValueError("survival_min_net_profit_pcts must be non-negative")
            candidates.append(EconomyCandidate(tp_multiplier, min_profit))
    if not candidates:
        raise ValueError("No economy candidates generated")
    return candidates


def prepare_market() -> pd.DataFrame:
    market = load_processed(project_path("data/processed/btcusdt_5m.parquet"))
    features_path = project_path("data/features/grid_features.parquet")
    if features_path.exists():
        features = pd.read_parquet(features_path)
        cols = [
            "atr_5m",
            "breakout_risk",
            "regime_allows_grid",
            "range_expansion_ratio",
            "realized_volatility_ratio",
        ]
        market = market.join(features[[col for col in cols if col in features.columns]], how="left")
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
    return market


def split_positions(
    market: pd.DataFrame,
    risk: GridRiskConfig,
    stride_bars: int,
    max_positions_per_split: int | None = None,
) -> dict[str, list[int]]:
    if stride_bars <= 0:
        raise ValueError("stride_bars must be positive")
    settings = load_settings()
    split = temporal_train_validation_test_split(
        market.index,
        train_fraction=float(settings["validation"]["train_fraction"]),
        validation_fraction=float(settings["validation"]["validation_fraction"]),
        embargo_bars=int(settings["validation"]["embargo_bars"]),
    )
    max_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    by_split: dict[str, list[int]] = {}
    for split_name, split_index in {"validation": split.validation, "test": split.test}.items():
        if len(split_index) <= max_bars:
            raise ValueError(f"{split_name} split is too short for max holding horizon")
        split_start = int(market.index.searchsorted(split_index.min(), side="left"))
        split_end = int(market.index.searchsorted(split_index.max(), side="right") - 1)
        positions = list(range(split_start, split_end - max_bars + 1, stride_bars))
        positions = [pos for pos in positions if np.isfinite(float(market.iloc[pos].get("atr_5m", np.nan)))]
        if max_positions_per_split is not None:
            if max_positions_per_split <= 0:
                raise ValueError("max_positions_per_split must be positive")
            positions = positions[:max_positions_per_split]
        if not positions:
            raise ValueError(f"No eligible positions for {split_name}")
        by_split[split_name] = positions
    return by_split


def simulate_candidate_sample(
    market: pd.DataFrame,
    positions: list[int],
    risk: GridRiskConfig,
    candidate: EconomyCandidate,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pos in positions:
        result: GridSimulationResult = simulate_grid_from_index(
            market,
            pos,
            risk,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            survival_min_realized_pnl=candidate.survival_min_net_profit_pct,
        )
        row = result.to_dict()
        row.update({**asdict(candidate), "split": split})
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows simulated for {split}")
    return pd.DataFrame(rows)


def summarize_simulations(frame: pd.DataFrame, baseline_grids: int | None = None) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot summarize empty simulation frame")
    pnl = frame["realized_pnl"].astype(float)
    equity = 1.0 + pnl.cumsum()
    drawdown = equity / equity.cummax() - 1.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    take_profit = frame[frame["exit_reason"] == "take_profit"]
    forced = frame[FORCED_COLUMNS].fillna(0).astype(int).max(axis=1)
    baseline = len(frame) if baseline_grids is None else int(baseline_grids)
    return {
        "number_of_grids": int(len(frame)),
        "baseline_grids": baseline,
        "grids_removed": int(max(0, baseline - len(frame))),
        "expectancy": float(pnl.mean()),
        "realized_pnl": float(pnl.sum()),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
        "winrate": float(pnl.gt(0).mean()),
        "economic_survival_rate": float(frame["grid_survived"].mean()),
        "take_profit_count": int(len(take_profit)),
        "positive_take_profit_rate": float(take_profit["realized_pnl"].gt(0).mean()) if not take_profit.empty else 0.0,
        "number_of_forced_exits": int(forced.sum()),
        "number_of_regime_exits": int(frame["stopped_by_regime_break"].sum()),
        "number_of_max_exposure_exits": int(frame["stopped_by_exposure"].sum()),
        "fees_paid": float(frame["fees_paid"].sum()),
        "slippage_paid": float(frame["slippage_paid"].sum()),
        "average_holding_time": float(frame["time_to_exit"].mean()),
        "exposure_time": float(frame["time_to_exit"].sum()),
        "max_unrealized_drawdown": float(frame["unrealized_drawdown_max"].max()),
        "average_levels_reached": float(frame["number_of_levels_filled"].mean()),
        "max_levels_reached": int(frame["number_of_levels_filled"].max()),
        "worst_grid": float(pnl.min()),
        "best_grid": float(pnl.max()),
    }


def search_validation_economics(
    market: pd.DataFrame,
    positions: dict[str, list[int]],
    risk: GridRiskConfig,
    candidates: list[EconomyCandidate],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = len(positions[SEARCH_SPLIT])
    for candidate in candidates:
        simulations = simulate_candidate_sample(market, positions[SEARCH_SPLIT], risk, candidate, SEARCH_SPLIT)
        rows.append({**asdict(candidate), "split": SEARCH_SPLIT, **summarize_simulations(simulations, baseline)})
    return pd.DataFrame(rows)


def select_best_candidate(validation_summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if validation_summary.empty:
        raise ValueError("Validation summary is empty")
    if set(validation_summary["split"].unique()) != {SEARCH_SPLIT}:
        raise ValueError("Economy candidate selection must use validation rows only")
    search = config["search"]
    baseline = int(validation_summary["baseline_grids"].max())
    minimum = min(int(search["min_grid_absolute_cap"]), int(np.ceil(baseline * float(search["min_grid_fraction_baseline"]))))
    eligible = validation_summary[validation_summary["number_of_grids"] >= minimum].copy()
    if eligible.empty:
        eligible = validation_summary.copy()
    eligible = eligible.sort_values(
        by=["expectancy", "profit_factor", "max_drawdown", "number_of_forced_exits"],
        ascending=[False, False, False, True],
    )
    selected = eligible.iloc[0].to_dict()
    selected["minimum_grid_constraint"] = int(minimum)
    selected["selected_from_validation_only"] = True
    return selected


def split_market(market: pd.DataFrame, split_name: str) -> pd.DataFrame:
    settings = load_settings()
    split = temporal_train_validation_test_split(
        market.index,
        train_fraction=float(settings["validation"]["train_fraction"]),
        validation_fraction=float(settings["validation"]["validation_fraction"]),
        embargo_bars=int(settings["validation"]["embargo_bars"]),
    )
    split_index = split.validation if split_name == "validation" else split.test
    return market.loc[(market.index >= split_index.min()) & (market.index <= split_index.max())]


def run_exact_backtest_for_split(
    market: pd.DataFrame,
    risk: GridRiskConfig,
    selected: dict[str, Any],
    split_name: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    split_frame = split_market(market, split_name)
    result = run_grid_backtest(
        split_frame,
        risk,
        allow_open=pd.Series(True, index=split_frame.index),
        constant_size=False,
        take_profit_spacing_multiplier=float(selected["take_profit_spacing_multiplier"]),
        survival_min_realized_pnl=float(selected["survival_min_net_profit_pct"]),
    )
    metrics = calculate_metrics(result.equity_curve, result.trades)
    if not result.trades.empty:
        metrics.update(summarize_simulations(result.trades, baseline_grids=len(result.trades)))
    metrics.update(
        {
            "split": split_name,
            "take_profit_spacing_multiplier": float(selected["take_profit_spacing_multiplier"]),
            "survival_min_net_profit_pct": float(selected["survival_min_net_profit_pct"]),
        }
    )
    return metrics, result.trades, result.equity_curve


def apply_cost_sensitivity(
    market: pd.DataFrame,
    positions: dict[str, list[int]],
    base_risk: GridRiskConfig,
    selected: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    candidate = EconomyCandidate(
        take_profit_spacing_multiplier=float(selected["take_profit_spacing_multiplier"]),
        survival_min_net_profit_pct=float(selected["survival_min_net_profit_pct"]),
    )
    rows: list[dict[str, Any]] = []
    for fee_multiplier in config["cost_sensitivity"]["fee_multipliers"]:
        for slippage_bps in config["cost_sensitivity"]["slippage_bps"]:
            risk = replace(
                base_risk,
                taker_fee=base_risk.taker_fee * float(fee_multiplier),
                maker_fee=base_risk.maker_fee * float(fee_multiplier),
                slippage_bps=float(slippage_bps),
            )
            sample = simulate_candidate_sample(market, positions["test"], risk, candidate, "test")
            rows.append(
                {
                    "fee_multiplier": float(fee_multiplier),
                    "slippage_bps": float(slippage_bps),
                    **asdict(candidate),
                    **summarize_simulations(sample, baseline_grids=len(positions["test"])),
                }
            )
    return pd.DataFrame(rows)


def pnl_quantile_table(frame: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, split_frame in frame.groupby("split", observed=False):
        quantile_values = split_frame["realized_pnl"].quantile(quantiles)
        row = {"split": split}
        row.update({f"pnl_q_{q:g}": float(value) for q, value in quantile_values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def decide_economy_outcome(test_metrics: dict[str, Any]) -> str:
    grids = int(test_metrics.get("number_of_grids", 0))
    if grids == 0:
        return "no-trade / no-edge"
    if float(test_metrics.get("expectancy", 0.0)) > 0 and float(test_metrics.get("realized_pnl", 0.0)) > 0:
        if float(test_metrics.get("positive_take_profit_rate", 0.0)) >= 0.5:
            return "economic grid viable"
        return "needs ML after relabeling"
    return "strategy economics still negative"


def write_report(
    output_dir: Path,
    selected: dict[str, Any],
    validation_exact: dict[str, Any],
    test_exact: dict[str, Any],
    decision: str,
) -> Path:
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 002 - Economy First Research",
        "",
        "## Decision",
        f"`{decision}`",
        "",
        "## Selected Economic Parameters",
        f"- Take-profit spacing multiplier: `{selected['take_profit_spacing_multiplier']}`",
        f"- Net-profit label threshold: `{selected['survival_min_net_profit_pct']}`",
        f"- Selected from validation only: `{selected['selected_from_validation_only']}`",
        f"- Minimum grid constraint: `{selected['minimum_grid_constraint']}`",
        "",
        "## Validation Search Metrics",
        "```json",
        json.dumps(selected, indent=2, default=str),
        "```",
        "",
        "## Exact Sequential Backtest - Validation",
        "```json",
        json.dumps(validation_exact, indent=2, default=str),
        "```",
        "",
        "## Exact Sequential Backtest - Test",
        "```json",
        json.dumps(test_exact, indent=2, default=str),
        "```",
        "",
        "## Interpretation",
        "This iteration changes only experimental take-profit and economic success labeling. Base risk limits, fees, slippage, max exposure, regime stops, and no-live-trading constraints remain active.",
        "",
        "## Next Step",
        "If the test result is economically viable, retrain the survival model on the net-profitable target. If it remains negative, redesign the grid payoff profile before adding more ML complexity.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(config_path: str, max_positions_per_split: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_risk = validate_strategy_config(load_strategy_config())
    market = prepare_market()
    search_positions = split_positions(
        market,
        base_risk,
        stride_bars=int(config["search"]["search_entry_stride_bars"]),
        max_positions_per_split=max_positions_per_split,
    )
    candidates = make_economy_candidates(config)
    validation_summary = search_validation_economics(market, search_positions, base_risk, candidates)
    validation_summary.to_csv(output_dir / "economy_search_validation.csv", index=False)
    selected = select_best_candidate(validation_summary, config)

    selected_candidate = EconomyCandidate(
        take_profit_spacing_multiplier=float(selected["take_profit_spacing_multiplier"]),
        survival_min_net_profit_pct=float(selected["survival_min_net_profit_pct"]),
    )
    validation_sample = simulate_candidate_sample(
        market, search_positions["validation"], base_risk, selected_candidate, "validation"
    )
    test_sample = simulate_candidate_sample(market, search_positions["test"], base_risk, selected_candidate, "test")
    sample_results = pd.concat([validation_sample, test_sample], ignore_index=True)
    pnl_quantile_table(sample_results, [float(q) for q in config["diagnostics"]["pnl_quantiles"]]).to_csv(
        output_dir / "selected_sample_pnl_quantiles.csv", index=False
    )
    test_sample_metrics = summarize_simulations(test_sample, baseline_grids=len(search_positions["test"]))
    pd.DataFrame([{**asdict(selected_candidate), **test_sample_metrics}]).to_csv(
        output_dir / "selected_sample_test_metrics.csv", index=False
    )

    validation_exact, validation_trades, validation_equity = run_exact_backtest_for_split(
        market, base_risk, selected, "validation"
    )
    test_exact, test_trades, test_equity = run_exact_backtest_for_split(market, base_risk, selected, "test")
    pd.DataFrame([validation_exact]).to_csv(output_dir / "selected_exact_validation_metrics.csv", index=False)
    pd.DataFrame([test_exact]).to_csv(output_dir / "selected_exact_test_metrics.csv", index=False)
    validation_trades.to_csv(output_dir / "selected_exact_validation_trades.csv", index=False)
    test_trades.to_csv(output_dir / "selected_exact_test_trades.csv", index=False)
    validation_equity.to_frame().to_csv(output_dir / "selected_exact_validation_equity.csv")
    test_equity.to_frame().to_csv(output_dir / "selected_exact_test_equity.csv")

    sensitivity = apply_cost_sensitivity(market, search_positions, base_risk, selected, config)
    sensitivity.to_csv(output_dir / "cost_sensitivity_test_sample.csv", index=False)

    decision = decide_economy_outcome(test_exact)
    payload = {
        "decision": decision,
        "selected_economic_parameters": selected,
        "validation_exact": validation_exact,
        "test_exact": test_exact,
        "test_sample": test_sample_metrics,
        "candidate_count": len(candidates),
        "search_positions": {key: len(value) for key, value in search_positions.items()},
        "max_positions_per_split": max_positions_per_split,
    }
    (output_dir / "selected_economy_parameters.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = write_report(output_dir, selected, validation_exact, test_exact, decision)
    LOGGER.info("Wrote economy-first iteration outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run economy-first grid calibration research iteration.")
    parser.add_argument("--config", default="config/research_iteration_economy.yaml")
    parser.add_argument("--max-positions-per-split", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(args.config, max_positions_per_split=args.max_positions_per_split)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

