from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import drawdown_series
from src.data.validate_data import load_processed
from src.labeling.grid_risk import validate_strategy_config
from src.research.economy_first_research import prepare_market
from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    SignalGridBacktestResult,
    build_side_signal,
    candidate_from_row,
    choose_candidate_subset,
    evaluate_exact_candidates,
    make_candidates,
    monthly_return_from_equity,
    risk_for_candidate,
    search_validation_sample,
    select_best_exact,
    summarize_exact,
    top_sample_candidates,
)
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
CANDIDATE_COLUMNS = [
    "name",
    "side_mode",
    "entry_mode",
    "entry_cooldown_hours",
    "rsi_window",
    "rsi_low",
    "rsi_high",
    "spacing_atr_multiplier",
    "take_profit_spacing_multiplier",
    "max_levels",
    "base_position_size_pct",
    "progression_multiplier",
    "max_total_exposure_pct",
    "fee_rate",
    "slippage_bps",
    "max_grid_loss_pct",
    "max_holding_hours",
    "stop_on_regime_break",
    "stop_on_volatility_shock",
]


@dataclass(frozen=True)
class WalkForwardWindow:
    fold_id: int
    train: pd.Index
    test: pd.Index


def _median_bar_duration(index: pd.Index) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("index must contain at least two timestamps")
    delta = pd.Series(index).diff().dropna().median()
    if pd.isna(delta) or delta <= pd.Timedelta(0):
        raise ValueError("cannot infer bar duration from index")
    return delta


def make_walk_forward_windows(
    index: pd.Index,
    train_days: float,
    test_days: float,
    step_days: float,
    embargo_bars: int,
    max_folds: int | None = None,
) -> list[WalkForwardWindow]:
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be non-negative")
    if max_folds is not None and max_folds <= 0:
        raise ValueError("max_folds must be positive")
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("walk-forward days must be positive")
    if index.tz is None:
        raise ValueError("walk-forward index must be timezone-aware")
    sorted_index = pd.Index(index).sort_values()
    embargo_delta = _median_bar_duration(sorted_index) * embargo_bars
    train_delta = pd.Timedelta(days=float(train_days))
    test_delta = pd.Timedelta(days=float(test_days))
    step_delta = pd.Timedelta(days=float(step_days))
    latest = sorted_index.max()
    start = sorted_index.min()
    windows: list[WalkForwardWindow] = []
    while True:
        train_start = start
        train_end = train_start + train_delta
        test_start = train_end + embargo_delta
        test_end = test_start + test_delta
        if test_end > latest:
            break
        train = sorted_index[(sorted_index >= train_start) & (sorted_index < train_end)]
        test = sorted_index[(sorted_index >= test_start) & (sorted_index < test_end)]
        if train.empty or test.empty:
            break
        windows.append(WalkForwardWindow(fold_id=len(windows) + 1, train=train, test=test))
        if max_folds is not None and len(windows) >= max_folds:
            break
        start = start + step_delta
    if not windows:
        raise ValueError("walk-forward configuration produced no folds")
    return windows


def candidate_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in CANDIDATE_COLUMNS)


def same_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return candidate_signature(left) == candidate_signature(right)


def split_frame_from_index(market: pd.DataFrame, split_index: pd.Index) -> pd.DataFrame:
    if split_index.empty:
        raise ValueError("split index is empty")
    return market.loc[(market.index >= split_index.min()) & (market.index <= split_index.max())]


def run_exact_candidate_on_index(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidate: MonthlyMartingaleCandidate,
    split_index: pd.Index,
    split_name: str,
) -> tuple[dict[str, Any], SignalGridBacktestResult]:
    from src.research.monthly_target_martingale_research import run_signal_grid_backtest

    split_frame = split_frame_from_index(market, split_index)
    risk = risk_for_candidate(base_risk, candidate)
    side_signal = build_side_signal(market, signal_frame, candidate)
    result = run_signal_grid_backtest(split_frame, risk, side_signal, candidate)
    return summarize_exact(result.equity_curve, result.trades, candidate, split_name), result


def run_fold(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidates: list[MonthlyMartingaleCandidate],
    window: WalkForwardWindow,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.Series]:
    indexes = {"validation": window.train}
    sample_summary = search_validation_sample(market, signal_frame, indexes, base_risk, candidates, config)
    top_rows = top_sample_candidates(sample_summary, config)
    validation_exact_frame, _validation_results = evaluate_exact_candidates(
        market,
        signal_frame,
        indexes,
        base_risk,
        top_rows,
        "validation",
    )
    selected = select_best_exact(validation_exact_frame, config)
    selected_candidate = candidate_from_row(selected)
    test_metrics, test_result = run_exact_candidate_on_index(
        market,
        signal_frame,
        base_risk,
        selected_candidate,
        window.test,
        "test",
    )
    if not same_candidate(selected, test_metrics):
        raise ValueError("selected candidate changed before fold test evaluation")
    validation_exact = validation_exact_frame[validation_exact_frame["name"] == selected_candidate.name].iloc[0].to_dict()
    fold_row = {
        "fold_id": window.fold_id,
        "train_start": window.train.min(),
        "train_end": window.train.max(),
        "test_start": window.test.min(),
        "test_end": window.test.max(),
        "selected_name": selected_candidate.name,
        "validation_monthly_return": float(validation_exact["monthly_return"]),
        "validation_total_return": float(validation_exact["total_return"]),
        "validation_max_drawdown": float(validation_exact["max_drawdown"]),
        "validation_profit_factor": float(validation_exact["profit_factor"]),
        "validation_target_reached": bool(float(validation_exact["monthly_return"]) >= float(config["target"]["monthly_return"])),
        "test_monthly_return": float(test_metrics["monthly_return"]),
        "test_total_return": float(test_metrics["total_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_profit_factor": float(test_metrics["profit_factor"]),
        "test_number_of_grids": int(test_metrics["number_of_grids"]),
        "test_positive": bool(float(test_metrics["total_return"]) > 0),
        "test_target_reached": bool(float(test_metrics["monthly_return"]) >= float(config["target"]["monthly_return"])),
        "test_equity_ruined": bool(test_result.equity_curve.min() <= 0),
    }
    selected_with_fold = {"fold_id": window.fold_id, **selected}
    return fold_row, selected_with_fold, test_result.trades, test_result.equity_curve


def stitch_oos_equity(fold_equities: list[tuple[int, pd.Series]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    current_equity = 1.0
    for fold_id, equity in fold_equities:
        if equity.empty:
            continue
        stitched = current_equity * equity.astype(float)
        rows.append(pd.DataFrame({"fold_id": fold_id, "equity": stitched}, index=stitched.index))
        current_equity = float(stitched.iloc[-1])
    if not rows:
        raise ValueError("cannot stitch empty fold equities")
    stitched_frame = pd.concat(rows).sort_index()
    stitched_frame = stitched_frame[~stitched_frame.index.duplicated(keep="last")]
    return stitched_frame


def summarize_walk_forward(fold_summary: pd.DataFrame, oos_equity: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    if fold_summary.empty:
        raise ValueError("fold summary is empty")
    target = float(config["target"]["monthly_return"])
    max_drawdown_limit = float(config["target"]["max_drawdown"])
    positive_fold_rate = float(fold_summary["test_positive"].mean())
    target_fold_rate = float(fold_summary["test_target_reached"].mean())
    validation_target_rate = float(fold_summary["validation_target_reached"].mean())
    aggregate_monthly = monthly_return_from_equity(oos_equity)
    aggregate_total = float(oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) if float(oos_equity.iloc[0]) != 0 else -1.0
    max_drawdown = float(drawdown_series(oos_equity).min())
    equity_ruined = bool((oos_equity <= 0).any() or fold_summary["test_equity_ruined"].any())
    return {
        "fold_count": int(len(fold_summary)),
        "aggregate_total_return": aggregate_total,
        "aggregate_monthly_return": aggregate_monthly,
        "aggregate_max_drawdown": max_drawdown,
        "positive_fold_rate": positive_fold_rate,
        "target_fold_rate": target_fold_rate,
        "validation_target_rate": validation_target_rate,
        "equity_ruined": equity_ruined,
        "target_monthly_return": target,
        "max_drawdown_constraint": max_drawdown_limit,
        "min_positive_fold_rate": float(config["target"]["min_positive_fold_rate"]),
        "min_target_fold_rate": float(config["target"]["min_target_fold_rate"]),
    }


def decide_walk_forward(summary: dict[str, Any]) -> str:
    if bool(summary["equity_ruined"]) or float(summary["aggregate_max_drawdown"]) < float(summary["max_drawdown_constraint"]):
        return "rejected by drawdown"
    if (
        float(summary["aggregate_monthly_return"]) >= float(summary["target_monthly_return"])
        and float(summary["positive_fold_rate"]) >= float(summary["min_positive_fold_rate"])
        and float(summary["target_fold_rate"]) >= float(summary["min_target_fold_rate"])
    ):
        return "walk-forward martingale target viable"
    if (
        float(summary["validation_target_rate"]) >= float(summary["min_target_fold_rate"])
        and float(summary["target_fold_rate"]) < float(summary["min_target_fold_rate"])
    ):
        return "validation overfit"
    if float(summary["aggregate_monthly_return"]) > 0:
        return "below monthly target"
    return "not viable"


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 006 - Walk-Forward Martingale Research",
        "",
        "## Decision",
        f"`{payload['decision']}`",
        "",
        "## Aggregate OOS Summary",
        "```json",
        json.dumps(payload["aggregate_summary"], indent=2, default=str),
        "```",
        "",
        "## Fold Summary",
        f"- Fold count: `{payload['aggregate_summary']['fold_count']}`",
        f"- Positive fold rate: `{payload['aggregate_summary']['positive_fold_rate']:.3f}`",
        f"- Target fold rate: `{payload['aggregate_summary']['target_fold_rate']:.3f}`",
        f"- Validation target rate: `{payload['aggregate_summary']['validation_target_rate']:.3f}`",
        "",
        "## Interpretation",
        "This walk-forward run reselects the martingale-grid parameters on each rolling selection window and evaluates the selected candidate once on the following out-of-sample window. The verdict is based on stitched OOS equity across folds, not on the best fold.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_progress_outputs(
    output_dir: Path,
    fold_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    fold_equities: list[tuple[int, pd.Series]],
    config: dict[str, Any],
    final: bool = False,
) -> dict[str, Any]:
    fold_summary = pd.DataFrame(fold_rows)
    selected_candidates = pd.DataFrame(selected_rows)
    oos_equity_frame = stitch_oos_equity(fold_equities)
    aggregate_summary = summarize_walk_forward(fold_summary, oos_equity_frame["equity"], config)
    decision = decide_walk_forward(aggregate_summary)

    fold_summary.to_csv(output_dir / "walk_forward_fold_summary.csv", index=False)
    selected_candidates.to_csv(output_dir / "walk_forward_selected_candidates.csv", index=False)
    oos_equity_frame.to_csv(output_dir / "walk_forward_oos_equity.csv")
    payload = {
        "decision": decision,
        "aggregate_summary": aggregate_summary,
        "fold_count": int(len(fold_rows)),
        "is_final": bool(final),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def run_iteration(
    config_path: str,
    max_folds: int | None = None,
    max_candidates: int | None = None,
    exact_top_n: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if exact_top_n is not None:
        config["search"]["exact_top_n"] = int(exact_top_n)

    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = output_dir / "selected_fold_trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    for old_trade_file in trades_dir.glob("fold_*_trades.csv"):
        old_trade_file.unlink()

    base_risk = validate_strategy_config(load_strategy_config())
    market = prepare_market()
    signal_frame = load_processed(project_path("data/processed/btcusdt_1h.parquet"))
    wf = config["walk_forward"]
    windows = make_walk_forward_windows(
        market.index,
        train_days=float(wf["train_days"]),
        test_days=float(wf["test_days"]),
        step_days=float(wf["step_days"]),
        embargo_bars=int(wf.get("embargo_bars", 0)),
        max_folds=max_folds,
    )

    candidates = make_candidates(config)
    total_candidate_count = len(candidates)
    sample_size = max_candidates if max_candidates is not None else config["search"].get("candidate_sample_size")
    sample_seed = config["search"].get("candidate_sample_seed")
    candidates = choose_candidate_subset(
        candidates,
        None if sample_size is None else int(sample_size),
        None if sample_seed is None else int(sample_seed),
    )

    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_equities: list[tuple[int, pd.Series]] = []
    for window in windows:
        LOGGER.info("Running walk-forward fold %s/%s", window.fold_id, len(windows))
        fold_row, selected_row, trades, equity = run_fold(market, signal_frame, base_risk, candidates, window, config)
        fold_rows.append(fold_row)
        selected_rows.append(selected_row)
        fold_equities.append((window.fold_id, equity))
        trades.to_csv(trades_dir / f"fold_{window.fold_id:03d}_trades.csv", index=False)
        write_progress_outputs(output_dir, fold_rows, selected_rows, fold_equities, config, final=False)

    payload = write_progress_outputs(output_dir, fold_rows, selected_rows, fold_equities, config, final=True)
    payload.update(
        {
        "fold_count": len(windows),
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
        "exact_top_n": int(config["search"]["exact_top_n"]),
        }
    )
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, payload)
    LOGGER.info("Wrote walk-forward martingale outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling walk-forward bounded martingale-grid research.")
    parser.add_argument("--config", default="config/research_iteration_walk_forward_martingale.yaml")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--exact-top-n", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        max_folds=args.max_folds,
        max_candidates=args.max_candidates,
        exact_top_n=args.exact_top_n,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
