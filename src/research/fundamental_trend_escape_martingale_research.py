from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.validate_data import load_processed
from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market
from src.research.fundamental_blackout_martingale_research import markdown_table, select_best_exact_no_drawdown
from src.research.fundamental_blackout_ablation_research import trade_attribution_by_fold
from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    SignalGridBacktestResult,
    build_side_signal,
    candidate_from_row,
    choose_candidate_subset,
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
DEFAULT_VARIANTS = [
    "baseline",
    "trend_escape_entry_only",
    "trend_escape_close",
    "fundamental_event_entry_only",
    "fundamental_trend_escape_entry_only",
    "fundamental_trend_escape_close",
    "oracle_fundamental_trend_escape_close",
]


def _align_mask(mask: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if mask is None:
        return None
    return mask.reindex(index).fillna(False).astype(bool)


def validate_variants(config: dict[str, Any]) -> list[str]:
    variants = list(config.get("trend_escape_ablation", {}).get("variants", DEFAULT_VARIANTS))
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported trend escape variants: {unknown}")
    if "baseline" not in variants:
        raise ValueError("trend escape ablation must include baseline")
    return variants


def build_variant_masks(
    variant: str,
    trend_escape: pd.Series,
    blackout_masks: dict[str, pd.Series],
) -> tuple[pd.Series | None, pd.Series | None, str]:
    realistic_event = blackout_masks["realistic"].astype(bool)
    oracle_event = blackout_masks["oracle"].astype(bool)
    realistic_trend = (trend_escape & realistic_event).astype(bool)
    oracle_trend = (trend_escape & oracle_event).astype(bool)
    if variant == "baseline":
        return None, None, "none"
    if variant == "trend_escape_entry_only":
        return trend_escape, None, "trend_escape"
    if variant == "trend_escape_close":
        return trend_escape, trend_escape, "trend_escape"
    if variant == "fundamental_event_entry_only":
        return realistic_event, None, "fundamental_blackout"
    if variant == "fundamental_trend_escape_entry_only":
        return realistic_trend, None, "fundamental_trend_escape"
    if variant == "fundamental_trend_escape_close":
        return realistic_trend, realistic_trend, "fundamental_trend_escape"
    if variant == "oracle_fundamental_trend_escape_close":
        return oracle_trend, oracle_trend, "oracle_fundamental_trend_escape"
    raise ValueError(f"Unsupported trend escape variant: {variant}")


def _add_mask_metrics(
    metrics: dict[str, Any],
    trades: pd.DataFrame,
    entry_mask: pd.Series | None,
    exit_mask: pd.Series | None,
) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["entry_mask_time_fraction"] = 0.0 if entry_mask is None else float(entry_mask.mean())
    metrics["exit_mask_time_fraction"] = 0.0 if exit_mask is None else float(exit_mask.mean())
    metrics["trend_escape_exit_count"] = (
        int(trades["exit_reason"].astype(str).str.contains("trend_escape", regex=False).sum())
        if not trades.empty and "exit_reason" in trades.columns
        else 0
    )
    metrics["fundamental_exit_count"] = (
        int(trades["exit_reason"].astype(str).str.contains("fundamental", regex=False).sum())
        if not trades.empty and "exit_reason" in trades.columns
        else 0
    )
    return metrics


def run_exact_candidate_on_index(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidate: MonthlyMartingaleCandidate,
    split_index: pd.Index,
    split_name: str,
    entry_mask: pd.Series | None = None,
    exit_mask: pd.Series | None = None,
    exit_reason: str = "trend_escape",
) -> tuple[dict[str, Any], SignalGridBacktestResult]:
    split_frame = split_frame_from_index(market, split_index)
    split_entry = _align_mask(entry_mask, split_frame.index)
    split_exit = _align_mask(exit_mask, split_frame.index)
    risk = risk_for_candidate(base_risk, candidate)
    side_signal = build_side_signal(market, signal_frame, candidate)
    result = run_signal_grid_backtest(
        split_frame,
        risk,
        side_signal,
        candidate,
        entry_blackout_series=split_entry,
        exit_blackout_series=split_exit,
        exit_blackout_reason=exit_reason,
    )
    metrics = summarize_exact(result.equity_curve, result.trades, candidate, split_name)
    return _add_mask_metrics(metrics, result.trades, split_entry, split_exit), result


def evaluate_exact_candidates_for_variant(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidate_rows: list[dict[str, Any]],
    split_index: pd.Index,
    split_name: str,
    entry_mask: pd.Series | None,
    exit_mask: pd.Series | None,
    exit_reason: str,
) -> tuple[pd.DataFrame, dict[str, SignalGridBacktestResult]]:
    rows: list[dict[str, Any]] = []
    results: dict[str, SignalGridBacktestResult] = {}
    for row in candidate_rows:
        candidate = candidate_from_row(row)
        metrics, result = run_exact_candidate_on_index(
            market,
            signal_frame,
            base_risk,
            candidate,
            split_index,
            split_name,
            entry_mask=entry_mask,
            exit_mask=exit_mask,
            exit_reason=exit_reason,
        )
        rows.append(metrics)
        results[candidate.name] = result
    if not rows:
        raise ValueError(f"No exact rows evaluated for {split_name}")
    return pd.DataFrame(rows), results


def run_fold_for_variant(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidates: list[MonthlyMartingaleCandidate],
    window: WalkForwardWindow,
    config: dict[str, Any],
    variant: str,
    entry_mask: pd.Series | None,
    exit_mask: pd.Series | None,
    exit_reason: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.Series]:
    indexes = {SEARCH_SPLIT: window.train}
    LOGGER.info("%s fold %s: sampling validation candidates", variant, window.fold_id)
    sample_summary = search_validation_sample(
        market,
        signal_frame,
        indexes,
        base_risk,
        candidates,
        config,
        entry_blackout_series=entry_mask,
        exit_blackout_series=exit_mask,
    )
    top_rows = top_sample_candidates(sample_summary, config)
    validation_exact_frame, _validation_results = evaluate_exact_candidates_for_variant(
        market,
        signal_frame,
        base_risk,
        top_rows,
        window.train,
        SEARCH_SPLIT,
        entry_mask,
        exit_mask,
        exit_reason,
    )
    selected = select_best_exact_no_drawdown(validation_exact_frame, config)
    selected_candidate = candidate_from_row(selected)
    test_metrics, test_result = run_exact_candidate_on_index(
        market,
        signal_frame,
        base_risk,
        selected_candidate,
        window.test,
        "test",
        entry_mask=entry_mask,
        exit_mask=exit_mask,
        exit_reason=exit_reason,
    )
    if not same_candidate(selected, test_metrics):
        raise ValueError("selected candidate changed before fold test evaluation")

    validation_exact = validation_exact_frame[validation_exact_frame["name"] == selected_candidate.name].iloc[0].to_dict()
    train_frame = split_frame_from_index(market, window.train)
    test_frame = split_frame_from_index(market, window.test)
    train_entry = _align_mask(entry_mask, train_frame.index)
    train_exit = _align_mask(exit_mask, train_frame.index)
    test_entry = _align_mask(entry_mask, test_frame.index)
    test_exit = _align_mask(exit_mask, test_frame.index)
    fold_row = {
        "variant": variant,
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
        "train_entry_mask_time_fraction": 0.0 if train_entry is None else float(train_entry.mean()),
        "test_entry_mask_time_fraction": 0.0 if test_entry is None else float(test_entry.mean()),
        "train_exit_mask_time_fraction": 0.0 if train_exit is None else float(train_exit.mean()),
        "test_exit_mask_time_fraction": 0.0 if test_exit is None else float(test_exit.mean()),
        "trend_escape_exit_count": int(test_metrics["trend_escape_exit_count"]),
        "fundamental_exit_count": int(test_metrics["fundamental_exit_count"]),
    }
    selected_with_fold = {"variant": variant, "fold_id": window.fold_id, **selected}
    return fold_row, selected_with_fold, test_result.trades, test_result.equity_curve


def summarize_variant(
    variant: str,
    fold_rows: list[dict[str, Any]],
    oos_equity: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    fold_summary = pd.DataFrame(fold_rows)
    summary = summarize_walk_forward(fold_summary, oos_equity, config)
    summary["variant"] = variant
    summary["entry_mask_time_fraction"] = float(fold_summary["test_entry_mask_time_fraction"].mean())
    summary["exit_mask_time_fraction"] = float(fold_summary["test_exit_mask_time_fraction"].mean())
    summary["trend_escape_exit_count"] = int(fold_summary["trend_escape_exit_count"].sum())
    summary["fundamental_exit_count"] = int(fold_summary["fundamental_exit_count"].sum())
    summary["test_grid_count"] = int(fold_summary["test_number_of_grids"].sum())
    summary["worst_fold_total_return"] = float(fold_summary["test_total_return"].min())
    summary["worst_fold_monthly_return"] = float(fold_summary["test_monthly_return"].min())
    summary["report_only_max_drawdown"] = summary["aggregate_max_drawdown"]
    return summary


def decide_trend_escape(comparison: pd.DataFrame, config: dict[str, Any]) -> str:
    baseline = comparison[comparison["variant"] == "baseline"].iloc[0]
    target = float(config["target"]["monthly_return"])
    min_positive = float(config["target"]["min_positive_fold_rate"])
    min_target = float(config["target"]["min_target_fold_rate"])
    non_baseline = comparison[comparison["variant"] != "baseline"].copy()
    non_baseline["monthly_delta"] = non_baseline["aggregate_monthly_return"].astype(float) - float(
        baseline["aggregate_monthly_return"]
    )
    best = non_baseline.sort_values(
        ["aggregate_monthly_return", "positive_fold_rate", "worst_fold_total_return"],
        ascending=[False, False, False],
    ).iloc[0]

    passes_target = (
        float(best["aggregate_monthly_return"]) >= target
        and float(best["positive_fold_rate"]) >= min_positive
        and float(best["target_fold_rate"]) >= min_target
        and float(best["monthly_delta"]) > 0
    )
    if passes_target and "fundamental" in str(best["variant"]):
        return "fundamental trend-escape viable"
    if passes_target:
        return "trend-escape viable without fundamental edge"

    realistic = non_baseline[~non_baseline["variant"].astype(str).str.startswith("oracle")]
    best_realistic = realistic.sort_values("monthly_delta", ascending=False).iloc[0]
    if float(best_realistic["monthly_delta"]) > 0:
        if "fundamental_trend_escape" in str(best_realistic["variant"]):
            return "fundamental trend filter helps but below target"
        if "trend_escape" in str(best_realistic["variant"]):
            return "price trend filter helps but below target"
        return "event-only filter helps but below target"

    oracle = non_baseline[non_baseline["variant"].astype(str).str.startswith("oracle")]
    if not oracle.empty and float(oracle["monthly_delta"].max()) > 0:
        return "oracle-only trend edge"
    return "no range-to-trend edge"


def write_variant_outputs(
    output_dir: Path,
    variant: str,
    fold_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    fold_equities: list[tuple[int, pd.Series]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_summary = pd.DataFrame(fold_rows)
    selected_candidates = pd.DataFrame(selected_rows)
    equity_frame = stitch_oos_equity(fold_equities)
    fold_summary.to_csv(output_dir / f"walk_forward_fold_summary_{variant}.csv", index=False)
    selected_candidates.to_csv(output_dir / f"walk_forward_selected_candidates_{variant}.csv", index=False)
    equity_frame.to_csv(output_dir / f"walk_forward_oos_equity_{variant}.csv")
    return fold_summary, equity_frame


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    comparison = pd.DataFrame(payload["comparison"])
    attribution = pd.DataFrame(payload["trade_attribution"])
    iteration_name = str(payload.get("iteration_name", "iteration_016_fundamental_trend_escape_martingale"))
    lines = [
        f"# {iteration_name} - Fundamental Trend-Escape Martingale",
        "",
        "## Decision",
        f"`{payload['decision']}`",
        "",
        "## OOS Comparison",
        markdown_table(
            comparison[
                [
                    "variant",
                    "aggregate_monthly_return",
                    "baseline_improvement_monthly",
                    "positive_fold_rate",
                    "target_fold_rate",
                    "entry_mask_time_fraction",
                    "exit_mask_time_fraction",
                    "test_grid_count",
                    "trend_escape_exit_count",
                    "aggregate_max_drawdown",
                    "worst_fold_monthly_return",
                ]
            ]
        ),
        "",
        "## Trade Attribution",
        markdown_table(
            attribution.groupby("variant", as_index=False)[
                ["baseline_only_pnl", "variant_only_pnl", "trade_set_pnl_delta", "common_pnl_delta", "total_pnl_delta"]
            ].sum()
        )
        if not attribution.empty
        else "No non-baseline attribution rows.",
        "",
        "## Interpretation",
        "This iteration tests whether range-to-trend escape detection can remove the catastrophic tail that breaks the high-return martingale grid. Selection remains rolling and validation-only; drawdown is reported, but the main verdict is based on OOS monthly return, fold hit rates, and improvement versus baseline.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(
    config_path: str,
    max_folds: int | None = None,
    max_candidates: int | None = None,
    exact_top_n: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if exact_top_n is not None:
        config["search"]["exact_top_n"] = int(exact_top_n)
    variants = validate_variants(config)
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

    events, windows_frame, blackout_masks = build_blackout_bundle(market.index, config)
    trend_components = build_trend_escape_components(market, config)
    trend_escape = trend_components["trend_escape"].astype(bool)
    events.to_csv(output_dir / "fundamental_events.csv", index=False)
    windows_frame.to_csv(output_dir / "blackout_windows.csv", index=False)
    trend_components.to_csv(output_dir / "trend_escape_components.csv")

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
    selected_rows_all: list[dict[str, Any]] = []
    variant_trade_frames: dict[str, dict[int, pd.DataFrame]] = {}
    variant_selected_by_fold: dict[str, dict[int, str]] = {}
    mask_time_rows: list[dict[str, Any]] = []

    for variant in variants:
        LOGGER.info("Running trend-escape variant: %s", variant)
        entry_mask, exit_mask, exit_reason = build_variant_masks(variant, trend_escape, blackout_masks)
        fold_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        fold_equities: list[tuple[int, pd.Series]] = []
        variant_trade_frames[variant] = {}
        variant_selected_by_fold[variant] = {}
        for window in windows:
            LOGGER.info("Running %s fold %s/%s", variant, window.fold_id, len(windows))
            fold_row, selected_row, trades, equity = run_fold_for_variant(
                market,
                signal_frame,
                base_risk,
                candidates,
                window,
                config,
                variant,
                entry_mask,
                exit_mask,
                exit_reason,
            )
            fold_rows.append(fold_row)
            selected_rows.append(selected_row)
            selected_rows_all.append(selected_row)
            fold_equities.append((window.fold_id, equity))
            variant_trade_frames[variant][window.fold_id] = trades
            variant_selected_by_fold[variant][window.fold_id] = str(fold_row["selected_name"])
            mask_time_rows.append(
                {
                    "variant": variant,
                    "fold_id": window.fold_id,
                    "train_entry_mask_time_fraction": fold_row["train_entry_mask_time_fraction"],
                    "test_entry_mask_time_fraction": fold_row["test_entry_mask_time_fraction"],
                    "train_exit_mask_time_fraction": fold_row["train_exit_mask_time_fraction"],
                    "test_exit_mask_time_fraction": fold_row["test_exit_mask_time_fraction"],
                }
            )
            trades.to_csv(trades_dir / f"{variant}_fold_{window.fold_id:03d}_trades.csv", index=False)
        _fold_summary, equity_frame = write_variant_outputs(output_dir, variant, fold_rows, selected_rows, fold_equities)
        comparison_rows.append(summarize_variant(variant, fold_rows, equity_frame["equity"], config))

    comparison = pd.DataFrame(comparison_rows)
    baseline = comparison[comparison["variant"] == "baseline"].iloc[0]
    baseline_monthly = float(baseline["aggregate_monthly_return"])
    baseline_grid_count = int(baseline["test_grid_count"])
    comparison["baseline_improvement_monthly"] = comparison["aggregate_monthly_return"].astype(float) - baseline_monthly
    comparison["grids_removed_vs_baseline"] = baseline_grid_count - comparison["test_grid_count"].astype(int)

    attribution_rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant == "baseline":
            continue
        for window in windows:
            attribution_rows.append(
                trade_attribution_by_fold(
                    variant_trade_frames["baseline"][window.fold_id],
                    variant_trade_frames[variant][window.fold_id],
                    variant,
                    window.fold_id,
                    same_selected_candidate=variant_selected_by_fold["baseline"][window.fold_id]
                    == variant_selected_by_fold[variant][window.fold_id],
                )
            )
    attribution = pd.DataFrame(attribution_rows)
    mask_time = pd.DataFrame(mask_time_rows)
    comparison.to_csv(output_dir / "walk_forward_trend_escape_comparison.csv", index=False)
    pd.DataFrame(selected_rows_all).to_csv(output_dir / "walk_forward_selected_candidates.csv", index=False)
    attribution.to_csv(output_dir / "trade_attribution_by_fold.csv", index=False)
    mask_time.to_csv(output_dir / "mask_time_by_variant.csv", index=False)

    decision = decide_trend_escape(comparison, config)
    payload = {
        "decision": decision,
        "iteration_name": str(config["iteration"].get("name", "iteration_016_fundamental_trend_escape_martingale")),
        "comparison": comparison.to_dict("records"),
        "trade_attribution": attribution.to_dict("records"),
        "fold_count": len(windows),
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
        "exact_top_n": int(config["search"]["exact_top_n"]),
        "event_count": int(len(events)),
        "blackout_window_count": int(len(windows_frame)),
        "trend_escape_time_fraction": float(trend_escape.mean()),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, payload)
    LOGGER.info("Wrote fundamental trend-escape martingale outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fundamental trend-escape martingale walk-forward research.")
    parser.add_argument("--config", default="config/research_iteration_fundamental_trend_escape_martingale.yaml")
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
