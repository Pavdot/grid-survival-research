from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.validate_data import load_processed
from src.fundamentals.event_blackout import build_blackout_bundle
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
    risk_for_candidate,
    run_signal_grid_backtest,
    search_validation_sample,
    summarize_exact,
    top_sample_candidates,
)
from src.research.walk_forward_martingale_research import (
    WalkForwardWindow,
    make_walk_forward_windows,
    same_candidate,
    split_frame_from_index,
    stitch_oos_equity,
    summarize_walk_forward,
)
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SEARCH_SPLIT = "validation"
VALID_EVENT_MODES = {"realistic", "oracle", "both"}


def select_best_exact_no_drawdown(validation_exact: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if validation_exact.empty:
        raise ValueError("Validation exact summary is empty")
    if set(validation_exact["split"].unique()) != {SEARCH_SPLIT}:
        raise ValueError("Fundamental blackout selection must use validation rows only")
    target = float(config["target"]["monthly_return"])
    min_grids = int(config["target"]["min_exact_grids"])
    eligible = validation_exact[validation_exact["number_of_grids"].astype(int) >= min_grids].copy()
    if eligible.empty:
        eligible = validation_exact.copy()
    eligible["target_reached"] = eligible["monthly_return"].astype(float) >= target
    eligible = eligible.sort_values(
        by=["target_reached", "monthly_return", "profit_factor", "number_of_forced_exits"],
        ascending=[False, False, False, True],
    )
    selected = eligible.iloc[0].to_dict()
    selected["selected_from_validation_only"] = True
    selected["selection_uses_drawdown"] = False
    selected["target_monthly_return"] = target
    selected["min_exact_grids"] = min_grids
    return selected


def _blackout_for_split(
    blackout_series: pd.Series | None,
    split_frame: pd.DataFrame,
) -> pd.Series | None:
    if blackout_series is None:
        return None
    return blackout_series.reindex(split_frame.index).fillna(False).astype(bool)


def _add_blackout_metrics(
    metrics: dict[str, Any],
    trades: pd.DataFrame,
    blackout_series: pd.Series | None,
) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["fundamental_exit_count"] = (
        int(trades["exit_reason"].eq("fundamental_blackout").sum())
        if not trades.empty and "exit_reason" in trades.columns
        else 0
    )
    metrics["blackout_time_fraction"] = 0.0 if blackout_series is None else float(blackout_series.mean())
    return metrics


def run_exact_candidate_on_index(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidate: MonthlyMartingaleCandidate,
    split_index: pd.Index,
    split_name: str,
    blackout_series: pd.Series | None = None,
) -> tuple[dict[str, Any], SignalGridBacktestResult]:
    split_frame = split_frame_from_index(market, split_index)
    split_blackout = _blackout_for_split(blackout_series, split_frame)
    risk = risk_for_candidate(base_risk, candidate)
    side_signal = build_side_signal(market, signal_frame, candidate)
    result = run_signal_grid_backtest(split_frame, risk, side_signal, candidate, blackout_series=split_blackout)
    metrics = summarize_exact(result.equity_curve, result.trades, candidate, split_name)
    return _add_blackout_metrics(metrics, result.trades, split_blackout), result


def run_fold_for_mode(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidates: list[MonthlyMartingaleCandidate],
    window: WalkForwardWindow,
    config: dict[str, Any],
    mode: str,
    blackout_series: pd.Series | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.Series]:
    indexes = {SEARCH_SPLIT: window.train}
    LOGGER.info("%s fold %s: sampling validation candidates", mode, window.fold_id)
    sample_summary = search_validation_sample(
        market,
        signal_frame,
        indexes,
        base_risk,
        candidates,
        config,
        blackout_series=blackout_series,
    )
    top_rows = top_sample_candidates(sample_summary, config)
    LOGGER.info("%s fold %s: exact validation on top %s candidates", mode, window.fold_id, len(top_rows))
    validation_exact_frame, _validation_results = evaluate_exact_candidates(
        market,
        signal_frame,
        indexes,
        base_risk,
        top_rows,
        SEARCH_SPLIT,
        blackout_series=blackout_series,
    )
    selected = select_best_exact_no_drawdown(validation_exact_frame, config)
    selected_candidate = candidate_from_row(selected)
    LOGGER.info("%s fold %s: testing selected candidate", mode, window.fold_id)
    test_metrics, test_result = run_exact_candidate_on_index(
        market,
        signal_frame,
        base_risk,
        selected_candidate,
        window.test,
        "test",
        blackout_series=blackout_series,
    )
    if not same_candidate(selected, test_metrics):
        raise ValueError("selected candidate changed before fold test evaluation")
    validation_exact = validation_exact_frame[validation_exact_frame["name"] == selected_candidate.name].iloc[0].to_dict()
    train_frame = split_frame_from_index(market, window.train)
    test_frame = split_frame_from_index(market, window.test)
    train_blackout = _blackout_for_split(blackout_series, train_frame)
    test_blackout = _blackout_for_split(blackout_series, test_frame)
    fold_row = {
        "mode": mode,
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
        "test_blackout_time_fraction": 0.0 if test_blackout is None else float(test_blackout.mean()),
        "train_blackout_time_fraction": 0.0 if train_blackout is None else float(train_blackout.mean()),
        "fundamental_exit_count": int(test_metrics["fundamental_exit_count"]),
    }
    selected_with_fold = {"mode": mode, "fold_id": window.fold_id, **selected}
    return fold_row, selected_with_fold, test_result.trades, test_result.equity_curve


def summarize_mode(
    mode: str,
    fold_rows: list[dict[str, Any]],
    oos_equity: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    fold_summary = pd.DataFrame(fold_rows)
    summary = summarize_walk_forward(fold_summary, oos_equity, config)
    summary["mode"] = mode
    summary["blackout_time_fraction"] = float(fold_summary["test_blackout_time_fraction"].mean())
    summary["fundamental_exit_count"] = int(fold_summary["fundamental_exit_count"].sum())
    summary["test_grid_count"] = int(fold_summary["test_number_of_grids"].sum())
    summary["worst_fold_total_return"] = float(fold_summary["test_total_return"].min())
    summary["worst_fold_monthly_return"] = float(fold_summary["test_monthly_return"].min())
    summary["report_only_max_drawdown"] = summary["aggregate_max_drawdown"]
    return summary


def mode_order(event_mode: str) -> list[str]:
    if event_mode not in VALID_EVENT_MODES:
        raise ValueError("event_mode must be realistic, oracle, or both")
    modes = ["baseline"]
    if event_mode in {"realistic", "both"}:
        modes.append("realistic")
    if event_mode in {"oracle", "both"}:
        modes.append("oracle")
    return modes


def decide_fundamental_blackout(comparison: pd.DataFrame, config: dict[str, Any]) -> str:
    target = float(config["target"]["monthly_return"])
    min_positive = float(config["target"]["min_positive_fold_rate"])
    min_target = float(config["target"]["min_target_fold_rate"])
    baseline = comparison[comparison["mode"] == "baseline"].iloc[0]

    def passes(row: pd.Series) -> bool:
        return (
            float(row["aggregate_monthly_return"]) >= target
            and float(row["positive_fold_rate"]) >= min_positive
            and float(row["target_fold_rate"]) >= min_target
            and float(row["aggregate_monthly_return"]) > float(baseline["aggregate_monthly_return"])
        )

    realistic_rows = comparison[comparison["mode"] == "realistic"]
    oracle_rows = comparison[comparison["mode"] == "oracle"]
    realistic_pass = not realistic_rows.empty and passes(realistic_rows.iloc[0])
    oracle_pass = not oracle_rows.empty and passes(oracle_rows.iloc[0])
    if realistic_pass:
        return "fundamental blackout viable"
    if oracle_pass:
        return "oracle-only edge"

    candidate_rows = comparison[comparison["mode"].isin(["realistic", "oracle"])]
    if candidate_rows.empty:
        return "no fundamental edge"
    improves_monthly = candidate_rows["aggregate_monthly_return"].astype(float).max() > float(
        baseline["aggregate_monthly_return"]
    )
    reduces_worst_loss = candidate_rows["worst_fold_total_return"].astype(float).max() > float(
        baseline["worst_fold_total_return"]
    )
    improves_drawdown_report = candidate_rows["aggregate_max_drawdown"].astype(float).max() > float(
        baseline["aggregate_max_drawdown"]
    )
    if improves_monthly or reduces_worst_loss or improves_drawdown_report:
        return "risk events insufficient"
    return "no fundamental edge"


def write_mode_outputs(
    output_dir: Path,
    mode: str,
    fold_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    fold_equities: list[tuple[int, pd.Series]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_summary = pd.DataFrame(fold_rows)
    selected_candidates = pd.DataFrame(selected_rows)
    equity_frame = stitch_oos_equity(fold_equities)
    suffix = "baseline" if mode == "baseline" else mode
    fold_summary.to_csv(output_dir / f"walk_forward_fold_summary_{suffix}.csv", index=False)
    selected_candidates.to_csv(output_dir / f"walk_forward_selected_candidates_{suffix}.csv", index=False)
    equity_frame.to_csv(output_dir / f"walk_forward_oos_equity_{suffix}.csv")
    return fold_summary, equity_frame


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    comparison = pd.DataFrame(payload["comparison"])
    lines = [
        "# Iteration 007 - Fundamental Blackout Walk-Forward",
        "",
        "## Decision",
        f"`{payload['decision']}`",
        "",
        "## OOS Comparison",
        markdown_table(
            comparison[
                [
                    "mode",
                    "aggregate_monthly_return",
                    "positive_fold_rate",
                    "target_fold_rate",
                    "blackout_time_fraction",
                    "aggregate_max_drawdown",
                    "fundamental_exit_count",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "Selection is rolling and validation-only. Drawdown is reported in the table but is not used as an optimization criterion or verdict veto for this iteration. Realistic blackout only reacts to surprise events after the configured known time; oracle blackout also cuts before surprise events and is therefore a non-tradable upper bound.",
        "",
        "## Event Sources",
        "- Fed FOMC calendar: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        "- BLS CPI schedule: https://www.bls.gov/schedule/news_release/cpi.htm",
        "- GDELT: https://www.gdeltproject.org/",
        "",
        "## Caveat",
        "The MVP event set is a curated local seed, not an exhaustive fundamental database. It is meant to test whether event-aware cutoffs can plausibly change the walk-forward result before wiring richer news ingestion.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(
    config_path: str,
    max_folds: int | None = None,
    max_candidates: int | None = None,
    exact_top_n: int | None = None,
    event_mode: str = "both",
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if exact_top_n is not None:
        config["search"]["exact_top_n"] = int(exact_top_n)

    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = output_dir / "selected_fold_trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    for old_trade_file in trades_dir.glob("*_fold_*_trades.csv"):
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

    events, windows_frame, masks = build_blackout_bundle(market.index, config)
    events.to_csv(output_dir / "fundamental_events.csv", index=False)
    windows_frame.to_csv(output_dir / "blackout_windows.csv", index=False)

    candidates = make_candidates(config)
    total_candidate_count = len(candidates)
    sample_size = max_candidates if max_candidates is not None else config["search"].get("candidate_sample_size")
    sample_seed = config["search"].get("candidate_sample_seed")
    candidates = choose_candidate_subset(
        candidates,
        None if sample_size is None else int(sample_size),
        None if sample_seed is None else int(sample_seed),
    )

    comparison_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    for mode in mode_order(event_mode):
        LOGGER.info("Running iteration 007 mode: %s", mode)
        mode_blackout = None if mode == "baseline" else masks[mode]
        fold_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        fold_equities: list[tuple[int, pd.Series]] = []
        for window in windows:
            LOGGER.info("Running %s fold %s/%s", mode, window.fold_id, len(windows))
            fold_row, selected_row, trades, equity = run_fold_for_mode(
                market,
                signal_frame,
                base_risk,
                candidates,
                window,
                config,
                mode,
                blackout_series=mode_blackout,
            )
            fold_rows.append(fold_row)
            selected_rows.append(selected_row)
            all_selected_rows.append(selected_row)
            fold_equities.append((window.fold_id, equity))
            trades.to_csv(trades_dir / f"{mode}_fold_{window.fold_id:03d}_trades.csv", index=False)
        fold_summary, equity_frame = write_mode_outputs(output_dir, mode, fold_rows, selected_rows, fold_equities)
        comparison_rows.append(summarize_mode(mode, fold_rows, equity_frame["equity"], config))

    comparison = pd.DataFrame(comparison_rows)
    baseline_monthly = float(comparison[comparison["mode"] == "baseline"]["aggregate_monthly_return"].iloc[0])
    comparison["baseline_improvement_monthly"] = comparison["aggregate_monthly_return"].astype(float) - baseline_monthly
    comparison.to_csv(output_dir / "walk_forward_comparison.csv", index=False)
    pd.DataFrame(all_selected_rows).to_csv(output_dir / "walk_forward_selected_candidates.csv", index=False)

    decision = decide_fundamental_blackout(comparison, config)
    payload = {
        "decision": decision,
        "comparison": comparison.to_dict("records"),
        "fold_count": len(windows),
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
        "exact_top_n": int(config["search"]["exact_top_n"]),
        "event_mode": event_mode,
        "event_count": int(len(events)),
        "blackout_window_count": int(len(windows_frame)),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, payload)
    LOGGER.info("Wrote fundamental blackout martingale outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fundamental-blackout walk-forward martingale research.")
    parser.add_argument("--config", default="config/research_iteration_fundamental_blackout_martingale.yaml")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--exact-top-n", type=int, default=None)
    parser.add_argument("--event-mode", choices=sorted(VALID_EVENT_MODES), default="both")
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        max_folds=args.max_folds,
        max_candidates=args.max_candidates,
        exact_top_n=args.exact_top_n,
        event_mode=args.event_mode,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
