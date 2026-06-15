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
from src.research.fundamental_blackout_martingale_research import (
    markdown_table,
    select_best_exact_no_drawdown,
)
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
VARIANTS = [
    "baseline",
    "realistic_entry_only",
    "realistic_close_on_blackout",
    "realistic_regime_entry_only",
    "oracle_entry_only",
]


def _align_mask(mask: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if mask is None:
        return None
    return mask.reindex(index).fillna(False).astype(bool)


def build_regime_danger_mask(market: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    regime_config = config.get("regime_danger", {})
    lookback = int(regime_config.get("lookback_bars", 12))
    if lookback <= 0:
        raise ValueError("regime_danger.lookback_bars must be positive")
    vol_threshold = float(regime_config.get("realized_volatility_ratio_threshold", 2.0))
    range_threshold = float(regime_config.get("range_expansion_ratio_threshold", 2.5))
    raw = (
        market.get("volatility_shock", pd.Series(0, index=market.index)).fillna(0).astype(int).eq(1)
        | market.get("breakout_risk", pd.Series(0, index=market.index)).fillna(0).astype(int).eq(1)
        | market.get("realized_volatility_ratio", pd.Series(0, index=market.index)).fillna(0).astype(float).ge(vol_threshold)
        | market.get("range_expansion_ratio", pd.Series(0, index=market.index)).fillna(0).astype(float).ge(range_threshold)
    )
    delayed = raw.shift(1).fillna(False).astype(int)
    mask = delayed.rolling(lookback, min_periods=1).max().astype(bool)
    mask.name = "regime_danger"
    return mask


def validate_variants(config: dict[str, Any]) -> list[str]:
    variants = list(config.get("ablation", {}).get("variants", VARIANTS))
    if variants != VARIANTS:
        raise ValueError(f"Iteration 008 must compare variants exactly in this order: {VARIANTS}")
    return variants


def variant_masks(
    variant: str,
    blackout_masks: dict[str, pd.Series],
    regime_danger: pd.Series,
) -> tuple[pd.Series | None, pd.Series | None]:
    if variant == "baseline":
        return None, None
    if variant == "realistic_entry_only":
        return blackout_masks["realistic"], None
    if variant == "realistic_close_on_blackout":
        return blackout_masks["realistic"], blackout_masks["realistic"]
    if variant == "realistic_regime_entry_only":
        return (blackout_masks["realistic"] & regime_danger).astype(bool), None
    if variant == "oracle_entry_only":
        return blackout_masks["oracle"], None
    raise ValueError(f"Unsupported ablation variant: {variant}")


def _add_variant_metrics(
    metrics: dict[str, Any],
    trades: pd.DataFrame,
    entry_mask: pd.Series | None,
    exit_mask: pd.Series | None,
) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["entry_blackout_time_fraction"] = 0.0 if entry_mask is None else float(entry_mask.mean())
    metrics["exit_blackout_time_fraction"] = 0.0 if exit_mask is None else float(exit_mask.mean())
    metrics["fundamental_exit_count"] = (
        int(trades["exit_reason"].eq("fundamental_blackout").sum())
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
    entry_blackout_series: pd.Series | None = None,
    exit_blackout_series: pd.Series | None = None,
) -> tuple[dict[str, Any], SignalGridBacktestResult]:
    split_frame = split_frame_from_index(market, split_index)
    split_entry = _align_mask(entry_blackout_series, split_frame.index)
    split_exit = _align_mask(exit_blackout_series, split_frame.index)
    risk = risk_for_candidate(base_risk, candidate)
    side_signal = build_side_signal(market, signal_frame, candidate)
    result = run_signal_grid_backtest(
        split_frame,
        risk,
        side_signal,
        candidate,
        entry_blackout_series=split_entry,
        exit_blackout_series=split_exit,
    )
    metrics = summarize_exact(result.equity_curve, result.trades, candidate, split_name)
    return _add_variant_metrics(metrics, result.trades, split_entry, split_exit), result


def run_fold_for_variant(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidates: list[MonthlyMartingaleCandidate],
    window: WalkForwardWindow,
    config: dict[str, Any],
    variant: str,
    entry_blackout_series: pd.Series | None,
    exit_blackout_series: pd.Series | None,
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
        entry_blackout_series=entry_blackout_series,
        exit_blackout_series=exit_blackout_series,
    )
    top_rows = top_sample_candidates(sample_summary, config)
    LOGGER.info("%s fold %s: exact validation on top %s candidates", variant, window.fold_id, len(top_rows))
    validation_exact_frame, _validation_results = evaluate_exact_candidates(
        market,
        signal_frame,
        indexes,
        base_risk,
        top_rows,
        SEARCH_SPLIT,
        entry_blackout_series=entry_blackout_series,
        exit_blackout_series=exit_blackout_series,
    )
    selected = select_best_exact_no_drawdown(validation_exact_frame, config)
    selected_candidate = candidate_from_row(selected)
    LOGGER.info("%s fold %s: testing selected candidate", variant, window.fold_id)
    test_metrics, test_result = run_exact_candidate_on_index(
        market,
        signal_frame,
        base_risk,
        selected_candidate,
        window.test,
        "test",
        entry_blackout_series=entry_blackout_series,
        exit_blackout_series=exit_blackout_series,
    )
    if not same_candidate(selected, test_metrics):
        raise ValueError("selected candidate changed before fold test evaluation")

    validation_exact = validation_exact_frame[validation_exact_frame["name"] == selected_candidate.name].iloc[0].to_dict()
    train_frame = split_frame_from_index(market, window.train)
    test_frame = split_frame_from_index(market, window.test)
    train_entry = _align_mask(entry_blackout_series, train_frame.index)
    train_exit = _align_mask(exit_blackout_series, train_frame.index)
    test_entry = _align_mask(entry_blackout_series, test_frame.index)
    test_exit = _align_mask(exit_blackout_series, test_frame.index)
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
        "train_entry_blackout_time_fraction": 0.0 if train_entry is None else float(train_entry.mean()),
        "test_entry_blackout_time_fraction": 0.0 if test_entry is None else float(test_entry.mean()),
        "train_exit_blackout_time_fraction": 0.0 if train_exit is None else float(train_exit.mean()),
        "test_exit_blackout_time_fraction": 0.0 if test_exit is None else float(test_exit.mean()),
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
    summary["entry_blackout_time_fraction"] = float(fold_summary["test_entry_blackout_time_fraction"].mean())
    summary["exit_blackout_time_fraction"] = float(fold_summary["test_exit_blackout_time_fraction"].mean())
    summary["fundamental_exit_count"] = int(fold_summary["fundamental_exit_count"].sum())
    summary["test_grid_count"] = int(fold_summary["test_number_of_grids"].sum())
    summary["worst_fold_total_return"] = float(fold_summary["test_total_return"].min())
    summary["worst_fold_monthly_return"] = float(fold_summary["test_monthly_return"].min())
    summary["report_only_max_drawdown"] = summary["aggregate_max_drawdown"]
    return summary


def _trade_groups(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    frame = trades.copy()
    frame["_trade_key"] = frame["start_timestamp"].astype(str) + "|" + frame["side"].astype(str)
    return frame.groupby("_trade_key")["realized_pnl"].sum().astype(float)


def trade_attribution_by_fold(
    baseline_trades: pd.DataFrame,
    variant_trades: pd.DataFrame,
    variant: str,
    fold_id: int,
    same_selected_candidate: bool,
) -> dict[str, Any]:
    baseline = _trade_groups(baseline_trades)
    other = _trade_groups(variant_trades)
    baseline_keys = set(baseline.index)
    other_keys = set(other.index)
    common = baseline_keys & other_keys
    baseline_only = baseline_keys - other_keys
    variant_only = other_keys - baseline_keys
    baseline_only_pnl = float(baseline.loc[list(baseline_only)].sum()) if baseline_only else 0.0
    variant_only_pnl = float(other.loc[list(variant_only)].sum()) if variant_only else 0.0
    common_baseline_pnl = float(baseline.loc[list(common)].sum()) if common else 0.0
    common_variant_pnl = float(other.loc[list(common)].sum()) if common else 0.0
    return {
        "variant": variant,
        "fold_id": fold_id,
        "same_selected_candidate": bool(same_selected_candidate),
        "effect_scope": "policy_only" if same_selected_candidate else "policy_plus_selection",
        "baseline_trade_count": int(len(baseline_trades)),
        "variant_trade_count": int(len(variant_trades)),
        "baseline_only_count": int(len(baseline_only)),
        "variant_only_count": int(len(variant_only)),
        "common_count": int(len(common)),
        "baseline_only_pnl": baseline_only_pnl,
        "variant_only_pnl": variant_only_pnl,
        "common_baseline_pnl": common_baseline_pnl,
        "common_variant_pnl": common_variant_pnl,
        "trade_set_pnl_delta": variant_only_pnl - baseline_only_pnl,
        "common_pnl_delta": common_variant_pnl - common_baseline_pnl,
        "total_pnl_delta": float(other.sum() - baseline.sum()),
    }


def decide_ablation(comparison: pd.DataFrame) -> str:
    baseline = comparison[comparison["variant"] == "baseline"].iloc[0]
    entry = comparison[comparison["variant"] == "realistic_entry_only"].iloc[0]
    close = comparison[comparison["variant"] == "realistic_close_on_blackout"].iloc[0]
    regime = comparison[comparison["variant"] == "realistic_regime_entry_only"].iloc[0]
    oracle = comparison[comparison["variant"] == "oracle_entry_only"].iloc[0]
    baseline_monthly = float(baseline["aggregate_monthly_return"])
    if float(regime["aggregate_monthly_return"]) > baseline_monthly:
        return "regime filter helps"
    if float(entry["aggregate_monthly_return"]) > baseline_monthly:
        return "entry-only helps"
    if float(oracle["aggregate_monthly_return"]) > baseline_monthly:
        return "oracle-only edge"
    if float(close["aggregate_monthly_return"]) < float(entry["aggregate_monthly_return"]):
        return "close policy harmful"
    return "no event filter edge"


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
    lines = [
        "# Iteration 008 - Fundamental Blackout Ablation",
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
                    "positive_fold_rate",
                    "target_fold_rate",
                    "entry_blackout_time_fraction",
                    "exit_blackout_time_fraction",
                    "grids_removed_vs_baseline",
                    "fundamental_exit_count",
                    "aggregate_max_drawdown",
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
        "This ablation keeps the same martingale engine, event seed, folds, and capped candidate budget as the Iteration 007 validation run. Drawdown is report-only. Entry-only variants block new grids but do not force-close open grids; close-on-blackout matches the Iteration 007 policy.",
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
    regime_danger = build_regime_danger_mask(market, config)
    events.to_csv(output_dir / "fundamental_events.csv", index=False)
    windows_frame.to_csv(output_dir / "blackout_windows.csv", index=False)
    regime_danger.to_frame().to_csv(output_dir / "regime_danger_mask.csv")

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
    variant_trade_frames: dict[str, dict[int, pd.DataFrame]] = {}
    variant_selected_by_fold: dict[str, dict[int, str]] = {}
    blackout_time_rows: list[dict[str, Any]] = []

    for variant in variants:
        LOGGER.info("Running iteration 008 variant: %s", variant)
        entry_mask, exit_mask = variant_masks(variant, blackout_masks, regime_danger)
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
            )
            fold_rows.append(fold_row)
            selected_rows.append(selected_row)
            all_selected_rows.append(selected_row)
            fold_equities.append((window.fold_id, equity))
            variant_trade_frames[variant][window.fold_id] = trades
            variant_selected_by_fold[variant][window.fold_id] = str(fold_row["selected_name"])
            blackout_time_rows.append(
                {
                    "variant": variant,
                    "fold_id": window.fold_id,
                    "train_entry_blackout_time_fraction": fold_row["train_entry_blackout_time_fraction"],
                    "test_entry_blackout_time_fraction": fold_row["test_entry_blackout_time_fraction"],
                    "train_exit_blackout_time_fraction": fold_row["train_exit_blackout_time_fraction"],
                    "test_exit_blackout_time_fraction": fold_row["test_exit_blackout_time_fraction"],
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
    blackout_time = pd.DataFrame(blackout_time_rows)
    comparison.to_csv(output_dir / "walk_forward_ablation_comparison.csv", index=False)
    pd.DataFrame(all_selected_rows).to_csv(output_dir / "walk_forward_selected_candidates.csv", index=False)
    attribution.to_csv(output_dir / "trade_attribution_by_fold.csv", index=False)
    blackout_time.to_csv(output_dir / "blackout_time_by_variant.csv", index=False)

    decision = decide_ablation(comparison)
    payload = {
        "decision": decision,
        "comparison": comparison.to_dict("records"),
        "trade_attribution": attribution.to_dict("records"),
        "fold_count": len(windows),
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
        "exact_top_n": int(config["search"]["exact_top_n"]),
        "event_count": int(len(events)),
        "blackout_window_count": int(len(windows_frame)),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, payload)
    LOGGER.info("Wrote fundamental blackout ablation outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fundamental-blackout ablation walk-forward research.")
    parser.add_argument("--config", default="config/research_iteration_blackout_ablation_martingale.yaml")
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
