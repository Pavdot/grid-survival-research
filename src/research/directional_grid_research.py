from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtesting.backtest_grid import run_grid_backtest
from src.backtesting.metrics import calculate_metrics
from src.labeling.grid_risk import validate_strategy_config
from src.research.economy_first_research import (
    FORCED_COLUMNS,
    prepare_market,
    split_market,
    split_positions,
    summarize_simulations,
)
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SEARCH_SPLIT = "validation"


@dataclass(frozen=True)
class DirectionalGridCandidate:
    side: str
    spacing_atr_multiplier: float
    max_levels: int
    take_profit_spacing_multiplier: float
    stop_on_regime_break: bool
    stop_on_volatility_shock: bool
    survival_min_net_profit_pct: float


def make_candidates(config: dict[str, Any]) -> list[DirectionalGridCandidate]:
    search = config["search"]
    candidates: list[DirectionalGridCandidate] = []
    for side in search["sides"]:
        if side not in {"long", "short"}:
            raise ValueError("sides must contain only long or short")
        for spacing in search["spacing_atr_multipliers"]:
            if float(spacing) <= 0:
                raise ValueError("spacing_atr_multipliers must be positive")
            for max_levels in search["max_levels"]:
                if int(max_levels) <= 0:
                    raise ValueError("max_levels must be positive")
                for tp in search["take_profit_spacing_multipliers"]:
                    if float(tp) <= 0:
                        raise ValueError("take_profit_spacing_multipliers must be positive")
                    for stop_regime in search["stop_on_regime_break"]:
                        for stop_vol in search["stop_on_volatility_shock"]:
                            for min_profit in search["survival_min_net_profit_pcts"]:
                                candidates.append(
                                    DirectionalGridCandidate(
                                        side=str(side),
                                        spacing_atr_multiplier=float(spacing),
                                        max_levels=int(max_levels),
                                        take_profit_spacing_multiplier=float(tp),
                                        stop_on_regime_break=bool(stop_regime),
                                        stop_on_volatility_shock=bool(stop_vol),
                                        survival_min_net_profit_pct=float(min_profit),
                                    )
                                )
    if not candidates:
        raise ValueError("No directional grid candidates generated")
    return candidates


def _risk_for_candidate(base_risk, candidate: DirectionalGridCandidate):
    return replace(
        base_risk,
        spacing_atr_multiplier=candidate.spacing_atr_multiplier,
        max_levels=candidate.max_levels,
        sizing_sequence=base_risk.sizing_sequence[: candidate.max_levels],
        stop_on_regime_break=candidate.stop_on_regime_break,
        stop_on_volatility_shock=candidate.stop_on_volatility_shock,
    )


def simulate_candidate_sample(market: pd.DataFrame, positions: list[int], base_risk, candidate: DirectionalGridCandidate, split: str) -> pd.DataFrame:
    from src.labeling.grid_engine import simulate_grid_from_index

    risk = _risk_for_candidate(base_risk, candidate)
    rows: list[dict[str, object]] = []
    for pos in positions:
        result = simulate_grid_from_index(
            market,
            pos,
            risk,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            survival_min_realized_pnl=candidate.survival_min_net_profit_pct,
            side=candidate.side,
        )
        row = result.to_dict()
        row.update({**asdict(candidate), "split": split})
        rows.append(row)
    if not rows:
        raise ValueError(f"No simulations for {split}")
    return pd.DataFrame(rows)


def search_validation(market: pd.DataFrame, positions: dict[str, list[int]], base_risk, candidates: list[DirectionalGridCandidate]) -> pd.DataFrame:
    baseline = len(positions[SEARCH_SPLIT])
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        sample = simulate_candidate_sample(market, positions[SEARCH_SPLIT], base_risk, candidate, SEARCH_SPLIT)
        rows.append({**asdict(candidate), "split": SEARCH_SPLIT, **summarize_simulations(sample, baseline)})
    return pd.DataFrame(rows)


def select_best(validation_summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if validation_summary.empty:
        raise ValueError("Validation summary is empty")
    if set(validation_summary["split"].unique()) != {SEARCH_SPLIT}:
        raise ValueError("Directional candidate selection must use validation rows only")
    search = config["search"]
    baseline = int(validation_summary["baseline_grids"].max())
    minimum = min(int(search["min_grid_absolute_cap"]), int((baseline * float(search["min_grid_fraction_baseline"])) + 0.9999))
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


def candidate_from_selected(selected: dict[str, Any]) -> DirectionalGridCandidate:
    return DirectionalGridCandidate(
        side=str(selected["side"]),
        spacing_atr_multiplier=float(selected["spacing_atr_multiplier"]),
        max_levels=int(selected["max_levels"]),
        take_profit_spacing_multiplier=float(selected["take_profit_spacing_multiplier"]),
        stop_on_regime_break=bool(selected["stop_on_regime_break"]),
        stop_on_volatility_shock=bool(selected["stop_on_volatility_shock"]),
        survival_min_net_profit_pct=float(selected["survival_min_net_profit_pct"]),
    )


def run_exact_backtest(market: pd.DataFrame, base_risk, candidate: DirectionalGridCandidate, split_name: str) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    split_frame = split_market(market, split_name)
    risk = _risk_for_candidate(base_risk, candidate)
    result = run_grid_backtest(
        split_frame,
        risk,
        allow_open=pd.Series(True, index=split_frame.index),
        constant_size=False,
        take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
        survival_min_realized_pnl=candidate.survival_min_net_profit_pct,
        side=candidate.side,
    )
    metrics = calculate_metrics(result.equity_curve, result.trades)
    if not result.trades.empty:
        metrics.update(summarize_simulations(result.trades, baseline_grids=len(result.trades)))
    metrics.update({**asdict(candidate), "split": split_name})
    return metrics, result.trades, result.equity_curve


def decide(test_metrics: dict[str, Any]) -> str:
    if int(test_metrics.get("number_of_grids", 0)) == 0:
        return "no-trade / no-edge"
    if float(test_metrics.get("expectancy", 0.0)) > 0 and float(test_metrics.get("realized_pnl", 0.0)) > 0:
        return "directional grid candidate viable"
    return "directional grid not viable"


def write_report(output_dir: Path, selected: dict[str, Any], validation_exact: dict[str, Any], test_exact: dict[str, Any], decision: str) -> Path:
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 003 - Directional Grid Research",
        "",
        "## Decision",
        f"`{decision}`",
        "",
        "## Selected Candidate",
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
        "This iteration searches long and short bounded grids with wider spacing and optional regime stop removal. Selection is validation-only; test is used once for the selected candidate.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(config_path: str, max_positions_per_split: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_risk = validate_strategy_config(load_strategy_config())
    market = prepare_market()
    positions = split_positions(
        market,
        base_risk,
        stride_bars=int(config["search"]["search_entry_stride_bars"]),
        max_positions_per_split=max_positions_per_split or int(config["search"]["max_positions_per_split"]),
    )
    candidates = make_candidates(config)
    validation_summary = search_validation(market, positions, base_risk, candidates)
    validation_summary.to_csv(output_dir / "directional_search_validation.csv", index=False)
    selected = select_best(validation_summary, config)
    candidate = candidate_from_selected(selected)

    test_sample = simulate_candidate_sample(market, positions["test"], base_risk, candidate, "test")
    test_sample_metrics = summarize_simulations(test_sample, baseline_grids=len(positions["test"]))
    pd.DataFrame([{**asdict(candidate), **test_sample_metrics}]).to_csv(output_dir / "selected_sample_test_metrics.csv", index=False)

    validation_exact, validation_trades, validation_equity = run_exact_backtest(market, base_risk, candidate, "validation")
    test_exact, test_trades, test_equity = run_exact_backtest(market, base_risk, candidate, "test")
    pd.DataFrame([validation_exact]).to_csv(output_dir / "selected_exact_validation_metrics.csv", index=False)
    pd.DataFrame([test_exact]).to_csv(output_dir / "selected_exact_test_metrics.csv", index=False)
    validation_trades.to_csv(output_dir / "selected_exact_validation_trades.csv", index=False)
    test_trades.to_csv(output_dir / "selected_exact_test_trades.csv", index=False)
    validation_equity.to_frame().to_csv(output_dir / "selected_exact_validation_equity.csv")
    test_equity.to_frame().to_csv(output_dir / "selected_exact_test_equity.csv")

    decision = decide(test_exact)
    payload = {
        "decision": decision,
        "selected_candidate": selected,
        "validation_exact": validation_exact,
        "test_exact": test_exact,
        "test_sample": test_sample_metrics,
        "candidate_count": len(candidates),
        "search_positions": {key: len(value) for key, value in positions.items()},
    }
    (output_dir / "selected_directional_candidate.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, selected, validation_exact, test_exact, decision)
    LOGGER.info("Wrote directional grid iteration outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run directional grid research iteration.")
    parser.add_argument("--config", default="config/research_iteration_directional.yaml")
    parser.add_argument("--max-positions-per-split", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(args.config, max_positions_per_split=args.max_positions_per_split)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
