from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import drawdown_series
from src.data.validate_data import load_processed
from src.research.momentum_switch_research import (
    MomentumCandidate,
    backtest_signal,
    build_signal,
    candidate_from_row,
    make_candidates,
    monthly_return_from_equity,
    summarize,
)
from src.research.walk_forward_martingale_research import WalkForwardWindow, make_walk_forward_windows, split_frame_from_index
from src.utils.asset_paths import normalize_asset_id, processed_ohlcv_path
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SEARCH_SPLIT = "validation"


def choose_candidate_subset(
    candidates: list[MomentumCandidate],
    max_candidates: int | None,
    seed: int | None,
) -> list[MomentumCandidate]:
    if max_candidates is None or max_candidates >= len(candidates):
        return candidates
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    rng = np.random.default_rng(seed)
    indexes = sorted(int(value) for value in rng.choice(len(candidates), size=max_candidates, replace=False))
    return [candidates[index] for index in indexes]


def _candidate_max_position(config: dict[str, Any], candidate: MomentumCandidate) -> float:
    return float(candidate.params.get("max_position_pct", config["execution"]["max_position_pct"]))


def _evaluate_candidate_with_signal(
    base_frame: pd.DataFrame,
    signal: pd.Series,
    candidate: MomentumCandidate,
    split_index: pd.Index,
    split_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    split_frame = split_frame_from_index(base_frame, split_index)
    equity, trades = backtest_signal(
        split_frame,
        signal,
        config,
        max_position_pct=_candidate_max_position(config, candidate),
    )
    metrics = {
        "name": candidate.name,
        "signal_type": candidate.signal_type,
        "params": json.dumps(candidate.params, sort_keys=True),
        "max_position_pct": _candidate_max_position(config, candidate),
        "split": split_name,
        **summarize(equity, trades),
    }
    return metrics, equity, trades


def evaluate_candidate_on_index(
    base_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    candidate: MomentumCandidate,
    split_index: pd.Index,
    split_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.Series, pd.DataFrame]:
    signal = build_signal(signal_frame, candidate)
    return _evaluate_candidate_with_signal(base_frame, signal, candidate, split_index, split_name, config)


def _validation_subwindows(index: pd.Index, window_count: int) -> list[pd.Index]:
    if window_count <= 1:
        return []
    sorted_index = pd.Index(index).sort_values()
    if len(sorted_index) < window_count:
        raise ValueError("validation stability windows exceed available bars")
    windows: list[pd.Index] = []
    for positions in np.array_split(np.arange(len(sorted_index)), window_count):
        if len(positions) == 0:
            raise ValueError("empty validation stability window")
        windows.append(sorted_index.take(positions))
    return windows


def _add_stability_metrics(
    metrics: dict[str, Any],
    subwindow_metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not subwindow_metrics:
        return metrics
    monthly_returns = np.array([float(row["monthly_return"]) for row in subwindow_metrics], dtype=float)
    drawdowns = np.array([float(row["max_drawdown"]) for row in subwindow_metrics], dtype=float)
    trades = np.array([int(row["trades"]) for row in subwindow_metrics], dtype=float)
    target_monthly = float(config["target"]["monthly_return"])
    metrics.update(
        {
            "stability_windows": int(len(subwindow_metrics)),
            "stability_avg_monthly_return": float(monthly_returns.mean()),
            "stability_median_monthly_return": float(np.median(monthly_returns)),
            "stability_min_monthly_return": float(monthly_returns.min()),
            "stability_return_std": float(monthly_returns.std(ddof=0)),
            "stability_positive_rate": float((monthly_returns > 0).mean()),
            "stability_target_rate": float((monthly_returns >= target_monthly).mean()),
            "stability_worst_drawdown": float(drawdowns.min()),
            "stability_min_trades": int(trades.min()),
        }
    )
    return metrics


def search_validation_candidates(
    base_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    candidates: list[MomentumCandidate],
    window: WalkForwardWindow,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stability_windows = int(config.get("selection", {}).get("stability_windows", 0) or 0)
    subwindows = _validation_subwindows(window.train, stability_windows) if stability_windows > 1 else []
    for candidate in candidates:
        signal = build_signal(signal_frame, candidate)
        metrics, _equity, _trades = _evaluate_candidate_with_signal(
            base_frame,
            signal,
            candidate,
            window.train,
            SEARCH_SPLIT,
            config,
        )
        if subwindows:
            subwindow_rows = []
            for subwindow in subwindows:
                sub_metrics, _sub_equity, _sub_trades = _evaluate_candidate_with_signal(
                    base_frame,
                    signal,
                    candidate,
                    subwindow,
                    "validation_stability",
                    config,
                )
                subwindow_rows.append(sub_metrics)
            metrics = _add_stability_metrics(metrics, subwindow_rows, config)
        rows.append(metrics)
    if not rows:
        raise ValueError("No validation candidate rows generated")
    return pd.DataFrame(rows)


def select_best_validation(validation_summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if validation_summary.empty:
        raise ValueError("Validation summary is empty")
    if set(validation_summary["split"].unique()) != {SEARCH_SPLIT}:
        raise ValueError("Momentum walk-forward selection must use validation rows only")
    target = config["target"]
    max_drawdown = float(target["max_drawdown"])
    min_trades = int(target.get("min_trades", 1))
    eligible = validation_summary[
        (validation_summary["max_drawdown"].astype(float) >= max_drawdown)
        & (validation_summary["trades"].astype(int) >= min_trades)
        & (validation_summary["risk_kill_triggered"].astype(int) == 0)
    ].copy()
    if eligible.empty:
        eligible = validation_summary.copy()
    selection = config.get("selection", {})
    primary_metric = str(selection.get("primary_metric", "monthly_return"))
    if primary_metric == "risk_adjusted_monthly":
        penalty = float(selection.get("drawdown_penalty", 0.5))
        eligible["selection_score"] = eligible["monthly_return"].astype(float) + penalty * eligible["max_drawdown"].astype(float)
        sort_columns = ["selection_score", "monthly_return", "max_drawdown", "total_return", "trades"]
        ascending = [False, False, False, False, True]
    elif primary_metric == "stability_adjusted_monthly":
        required = {
            "stability_median_monthly_return",
            "stability_min_monthly_return",
            "stability_positive_rate",
            "stability_worst_drawdown",
        }
        missing = required - set(eligible.columns)
        if missing:
            raise ValueError(f"stability_adjusted_monthly requires stability metrics: {sorted(missing)}")
        min_positive_rate = float(selection.get("min_stability_positive_rate", 0.0))
        stable = eligible[eligible["stability_positive_rate"].astype(float) >= min_positive_rate].copy()
        if stable.empty:
            stable = eligible
        monthly_weight = float(selection.get("monthly_weight", 0.45))
        median_weight = float(selection.get("stability_median_weight", 0.35))
        min_weight = float(selection.get("stability_min_weight", 0.10))
        positive_weight = float(selection.get("stability_positive_weight", 0.02))
        drawdown_penalty = float(selection.get("drawdown_penalty", 0.50))
        stable["selection_score"] = (
            monthly_weight * stable["monthly_return"].astype(float)
            + median_weight * stable["stability_median_monthly_return"].astype(float)
            + min_weight * stable["stability_min_monthly_return"].astype(float)
            + positive_weight * stable["stability_positive_rate"].astype(float)
            + drawdown_penalty * stable["max_drawdown"].astype(float)
        )
        eligible = stable
        sort_columns = [
            "selection_score",
            "stability_positive_rate",
            "stability_median_monthly_return",
            "monthly_return",
            "max_drawdown",
            "trades",
        ]
        ascending = [False, False, False, False, False, True]
    elif primary_metric == "monthly_return":
        sort_columns = ["monthly_return", "max_drawdown", "total_return", "trades"]
        ascending = [False, False, False, True]
    else:
        raise ValueError(
            "selection.primary_metric must be monthly_return, risk_adjusted_monthly, or stability_adjusted_monthly"
        )
    eligible = eligible.sort_values(
        by=sort_columns,
        ascending=ascending,
    )
    selected = eligible.iloc[0].to_dict()
    selected["selected_from_validation_only"] = True
    selected["target_monthly_return"] = float(target["monthly_return"])
    return selected


def same_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        str(left["name"]) == str(right["name"])
        and str(left["signal_type"]) == str(right["signal_type"])
        and json.loads(left["params"]) == json.loads(right["params"])
    )


def run_fold(
    base_frame: pd.DataFrame,
    signal_frame: pd.DataFrame,
    candidates: list[MomentumCandidate],
    window: WalkForwardWindow,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.Series, pd.DataFrame, pd.DataFrame]:
    validation_summary = search_validation_candidates(base_frame, signal_frame, candidates, window, config)
    selected = select_best_validation(validation_summary, config)
    selected_candidate = candidate_from_row(selected)
    test_metrics, test_equity, test_trades = evaluate_candidate_on_index(
        base_frame,
        signal_frame,
        selected_candidate,
        window.test,
        "test",
        config,
    )
    if not same_candidate(selected, test_metrics):
        raise ValueError("Selected momentum candidate changed before fold test evaluation")
    fold_row = {
        "fold_id": window.fold_id,
        "train_start": window.train.min(),
        "train_end": window.train.max(),
        "test_start": window.test.min(),
        "test_end": window.test.max(),
        "selected_name": selected_candidate.name,
        "validation_monthly_return": float(selected["monthly_return"]),
        "validation_total_return": float(selected["total_return"]),
        "validation_max_drawdown": float(selected["max_drawdown"]),
        "validation_trades": int(selected["trades"]),
        "validation_target_reached": bool(float(selected["monthly_return"]) >= float(config["target"]["monthly_return"])),
        "test_monthly_return": float(test_metrics["monthly_return"]),
        "test_total_return": float(test_metrics["total_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_trades": int(test_metrics["trades"]),
        "test_positive": bool(float(test_metrics["total_return"]) > 0),
        "test_target_reached": bool(float(test_metrics["monthly_return"]) >= float(config["target"]["monthly_return"])),
        "test_stretch_reached": bool(
            float(test_metrics["monthly_return"]) >= float(config["target"].get("stretch_monthly_return", 0.20))
        ),
        "test_equity_ruined": bool((test_equity <= 0).any()),
        "test_risk_kill_triggered": int(test_metrics["risk_kill_triggered"]),
        "test_exposure_time": float(test_metrics["exposure_time"]),
        "test_average_position": float(test_metrics["average_position"]),
        "test_costs": float(test_metrics["fees_slippage_paid"]),
    }
    selected_with_fold = {"fold_id": window.fold_id, **selected}
    return fold_row, selected_with_fold, test_equity, test_trades, validation_summary


def stitch_oos_equity(fold_equities: list[tuple[int, pd.Series]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    current_equity = 1.0
    for fold_id, equity in fold_equities:
        equity = equity.dropna().astype(float)
        if equity.empty:
            continue
        normalized = equity / float(equity.iloc[0])
        stitched = current_equity * normalized
        rows.append(pd.DataFrame({"fold_id": fold_id, "equity": stitched}, index=stitched.index))
        current_equity = float(stitched.iloc[-1])
    if not rows:
        raise ValueError("cannot stitch empty fold equities")
    stitched_frame = pd.concat(rows).sort_index()
    return stitched_frame[~stitched_frame.index.duplicated(keep="last")]


def summarize_walk_forward(fold_summary: pd.DataFrame, oos_equity: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    if fold_summary.empty:
        raise ValueError("fold summary is empty")
    target = config["target"]
    aggregate_total = float(oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) if float(oos_equity.iloc[0]) != 0 else -1.0
    return {
        "fold_count": int(len(fold_summary)),
        "aggregate_total_return": aggregate_total,
        "aggregate_monthly_return": monthly_return_from_equity(oos_equity),
        "aggregate_max_drawdown": float(drawdown_series(oos_equity).min()),
        "positive_fold_rate": float(fold_summary["test_positive"].mean()),
        "target_fold_rate": float(fold_summary["test_target_reached"].mean()),
        "stretch_fold_rate": float(fold_summary["test_stretch_reached"].mean()),
        "validation_target_rate": float(fold_summary["validation_target_reached"].mean()),
        "risk_kill_rate": float(fold_summary["test_risk_kill_triggered"].mean()),
        "average_exposure_time": float(fold_summary["test_exposure_time"].mean()),
        "average_position": float(fold_summary["test_average_position"].mean()),
        "total_costs": float(fold_summary["test_costs"].sum()),
        "equity_ruined": bool((oos_equity <= 0).any() or fold_summary["test_equity_ruined"].any()),
        "target_monthly_return": float(target["monthly_return"]),
        "stretch_monthly_return": float(target.get("stretch_monthly_return", 0.20)),
        "max_drawdown_constraint": float(target["max_drawdown"]),
        "min_positive_fold_rate": float(target["min_positive_fold_rate"]),
        "min_target_fold_rate": float(target["min_target_fold_rate"]),
    }


def decide(summary: dict[str, Any]) -> str:
    if bool(summary["equity_ruined"]) or float(summary["aggregate_max_drawdown"]) < float(summary["max_drawdown_constraint"]):
        return "rejected by drawdown"
    if (
        float(summary["aggregate_monthly_return"]) >= float(summary["stretch_monthly_return"])
        and float(summary["positive_fold_rate"]) >= float(summary["min_positive_fold_rate"])
    ):
        return "momentum switch stretch target viable"
    if (
        float(summary["aggregate_monthly_return"]) >= float(summary["target_monthly_return"])
        and float(summary["positive_fold_rate"]) >= float(summary["min_positive_fold_rate"])
        and float(summary["target_fold_rate"]) >= float(summary["min_target_fold_rate"])
    ):
        return "momentum switch robust edge"
    if (
        float(summary["validation_target_rate"]) >= float(summary["min_target_fold_rate"])
        and float(summary["target_fold_rate"]) < float(summary["min_target_fold_rate"])
    ):
        return "validation overfit"
    if float(summary["aggregate_monthly_return"]) > 0:
        return "positive but below robust target"
    return "no momentum edge"


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    summary = payload["aggregate_summary"]
    family_frame = pd.DataFrame(payload["family_summary"])
    family_text = family_frame.to_csv(index=False).strip() if not family_frame.empty else "No family summary."
    iteration_name = payload.get("iteration_name", "momentum_switch_walk_forward")
    lines = [
        f"# {iteration_name} - Momentum Switch Walk-Forward",
        "",
        "## Decision",
        f"`{payload['decision']}`",
        "",
        "## Aggregate OOS Summary",
        "```json",
        json.dumps(summary, indent=2, default=str),
        "```",
        "",
        "## Best Families",
        "```csv",
        family_text,
        "```",
        "",
        "## Interpretation",
        "This iteration reselects the bounded momentum switch on each rolling validation window and evaluates the selected candidate once on the next out-of-sample window. It optimizes signal family and position size, but does not use test folds for selection.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(
    config_path: str,
    max_folds: int | None = None,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    asset_id = normalize_asset_id(config.get("asset", {}).get("id", "btcusdt"))
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    base_frame = load_processed(processed_ohlcv_path(asset_id, config["asset"].get("base_timeframe", "5m")))
    signal_frame = load_processed(processed_ohlcv_path(asset_id, config["asset"].get("signal_timeframe", "1h")))
    wf = config["walk_forward"]
    windows = make_walk_forward_windows(
        base_frame.index,
        train_days=float(wf["train_days"]),
        test_days=float(wf["test_days"]),
        step_days=float(wf["step_days"]),
        embargo_bars=int(wf.get("embargo_bars", 0)),
        max_folds=max_folds,
    )

    candidates = make_candidates(config)
    total_candidate_count = len(candidates)
    sample_size = max_candidates if max_candidates is not None else config["search"].get("candidate_sample_size")
    candidates = choose_candidate_subset(
        candidates,
        None if sample_size is None else int(sample_size),
        None if config["search"].get("candidate_sample_seed") is None else int(config["search"]["candidate_sample_seed"]),
    )

    fold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_equities: list[tuple[int, pd.Series]] = []
    validation_frames: list[pd.DataFrame] = []
    trades_dir = output_dir / "selected_fold_trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    for old_file in trades_dir.glob("fold_*_trades.csv"):
        old_file.unlink()

    for window in windows:
        LOGGER.info("Running momentum fold %s/%s", window.fold_id, len(windows))
        fold_row, selected_row, equity, trades, validation_summary = run_fold(
            base_frame,
            signal_frame,
            candidates,
            window,
            config,
        )
        fold_rows.append(fold_row)
        selected_rows.append(selected_row)
        fold_equities.append((window.fold_id, equity))
        validation_summary = validation_summary.copy()
        validation_summary["fold_id"] = window.fold_id
        validation_frames.append(validation_summary)
        trades.to_csv(trades_dir / f"fold_{window.fold_id:03d}_trades.csv")

    fold_summary = pd.DataFrame(fold_rows)
    selected_candidates = pd.DataFrame(selected_rows)
    validation_search = pd.concat(validation_frames, ignore_index=True)
    oos_equity = stitch_oos_equity(fold_equities)
    aggregate_summary = summarize_walk_forward(fold_summary, oos_equity["equity"], config)
    decision = decide(aggregate_summary)

    family_summary = (
        fold_summary.assign(family=fold_summary["selected_name"].str.extract(r"^([a-z]+(?:_[a-z]+)?)", expand=False))
        .groupby("family", as_index=False)
        .agg(
            folds=("fold_id", "count"),
            avg_test_monthly_return=("test_monthly_return", "mean"),
            avg_test_drawdown=("test_max_drawdown", "mean"),
            positive_rate=("test_positive", "mean"),
        )
        .sort_values(["avg_test_monthly_return", "positive_rate"], ascending=[False, False])
        .to_dict("records")
    )

    fold_summary.to_csv(output_dir / "momentum_walk_forward_fold_summary.csv", index=False)
    selected_candidates.to_csv(output_dir / "momentum_walk_forward_selected_candidates.csv", index=False)
    validation_search.to_csv(output_dir / "momentum_walk_forward_validation_search.csv", index=False)
    oos_equity.to_csv(output_dir / "momentum_walk_forward_oos_equity.csv")

    payload = {
        "decision": decision,
        "iteration_name": str(config["iteration"].get("name", "momentum_switch_walk_forward")),
        "asset_id": asset_id,
        "aggregate_summary": aggregate_summary,
        "family_summary": family_summary,
        "fold_count": len(windows),
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
    }
    (output_dir / "momentum_walk_forward_payload.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = write_report(output_dir, payload)
    LOGGER.info("Wrote momentum walk-forward outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded momentum switch walk-forward research.")
    parser.add_argument("--config", default="config/research_iteration_momentum_walk_forward.yaml")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(args.config, max_folds=args.max_folds, max_candidates=args.max_candidates)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
