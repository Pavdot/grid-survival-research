from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.backtesting.backtest_grid import run_grid_backtest
from src.backtesting.metrics import calculate_metrics
from src.backtesting.walk_forward import temporal_train_validation_test_split
from src.data.validate_data import load_processed
from src.labeling.grid_risk import validate_strategy_config
from src.utils.config_loader import load_model_config, load_settings, load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
VALID_SEARCH_SPLIT = "validation"
VALID_EVALUATION_SPLITS = {"validation", "test"}


@dataclass(frozen=True)
class ThresholdCandidate:
    open_threshold: float
    add_threshold: float
    kill_switch_threshold: float | None


def _round_threshold(value: float) -> float:
    return round(float(value) + 1e-12, 2)


def threshold_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold_step must be positive")
    if start > stop:
        raise ValueError("open_threshold_start must be <= open_threshold_stop")
    values: list[float] = []
    current = float(start)
    while current <= float(stop) + 1e-12:
        values.append(_round_threshold(current))
        current += float(step)
    return values


def make_threshold_candidates(config: dict[str, Any]) -> list[ThresholdCandidate]:
    search = config["threshold_search"]
    open_values = threshold_range(
        float(search["open_threshold_start"]),
        float(search["open_threshold_stop"]),
        float(search["threshold_step"]),
    )
    add_floor = float(search["minimum_add_threshold_floor"])
    kill_values = search["kill_switch_thresholds"]
    candidates: list[ThresholdCandidate] = []
    for open_threshold in open_values:
        add_start = max(open_threshold, add_floor)
        for add_threshold in threshold_range(add_start, float(search["open_threshold_stop"]), float(search["threshold_step"])):
            for kill_threshold in kill_values:
                if kill_threshold is not None and float(kill_threshold) >= open_threshold:
                    continue
                candidates.append(
                    ThresholdCandidate(
                        open_threshold=open_threshold,
                        add_threshold=add_threshold,
                        kill_switch_threshold=None if kill_threshold is None else _round_threshold(float(kill_threshold)),
                    )
                )
    if not candidates:
        raise ValueError("No threshold candidates generated")
    return candidates


def validate_prediction_frame(predictions: pd.DataFrame) -> None:
    required = {"grid_survival_score", "dataset_split"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")
    evaluation = predictions[predictions.get("dataset_split", pd.Series(index=predictions.index, dtype=object)).isin(VALID_EVALUATION_SPLITS)]
    if evaluation.empty:
        raise ValueError("Predictions do not contain validation/test rows")
    if evaluation["grid_survival_score"].isna().any():
        raise ValueError("Validation/test predictions contain missing grid_survival_score values")


def load_iteration_inputs(max_rows_per_split: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(project_path("data/features/grid_features.parquet"))
    labels = pd.read_parquet(project_path("data/labels/grid_labels.parquet"))
    predictions_path = project_path("reports/model_reports/grid_survival_predictions.parquet")
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing model predictions: {predictions_path}")
    predictions = pd.read_parquet(predictions_path)
    validate_prediction_frame(predictions)

    common = labels.index.intersection(predictions.index).intersection(features.index)
    labels = labels.loc[common].sort_index()
    predictions = predictions.loc[common].sort_index()
    features = features.loc[common].sort_index()
    scored = labels.join(predictions[["grid_survival_score", "dataset_split"]], how="inner")
    scored = scored[scored["dataset_split"].isin(VALID_EVALUATION_SPLITS)]
    if max_rows_per_split is not None:
        if max_rows_per_split <= 0:
            raise ValueError("max_rows_per_split must be positive")
        keep = []
        for split in sorted(VALID_EVALUATION_SPLITS):
            keep.append(scored[scored["dataset_split"] == split].head(max_rows_per_split))
        scored = pd.concat(keep).sort_index()
        features = features.loc[scored.index]
        labels = labels.loc[scored.index]
        predictions = predictions.loc[scored.index]
    return features, labels, predictions


def add_intratrade_score_diagnostics(scored: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    full_scores = predictions["grid_survival_score"].astype(float).sort_index()
    full_index = full_scores.index
    score_values = full_scores.to_numpy(dtype=float)
    out["entry_score"] = out["grid_survival_score"].astype(float)

    min_scores: list[float] = []
    exit_splits: list[str | None] = []
    split_max = predictions.groupby("dataset_split", observed=False).apply(lambda frame: frame.index.max()).to_dict()
    for start_ts, row in out.iterrows():
        start_pos = int(full_index.searchsorted(start_ts, side="left"))
        exit_ts = pd.Timestamp(row["exit_timestamp"])
        exit_pos = int(full_index.searchsorted(exit_ts, side="right") - 1)
        if start_pos < 0 or exit_pos < start_pos or start_pos >= len(full_index):
            min_scores.append(float(row["entry_score"]))
        else:
            window = score_values[start_pos : min(exit_pos + 1, len(score_values))]
            finite_window = window[np.isfinite(window)]
            min_scores.append(float(finite_window.min()) if len(finite_window) else float(row["entry_score"]))

        split_name = str(row["dataset_split"])
        split_end = split_max.get(split_name)
        exit_splits.append(split_name if split_end is not None and exit_ts <= split_end else None)

    out["intratrade_min_score"] = min_scores
    out["exit_within_entry_split"] = pd.Series(exit_splits, index=out.index)
    return out


def score_decile_analysis(scored: pd.DataFrame, deciles: int, worst_pnl_quantile: float) -> pd.DataFrame:
    frame = scored.copy()
    frame["score_decile"] = pd.qcut(frame["entry_score"], q=deciles, labels=False, duplicates="drop")
    worst_cutoff = frame["realized_pnl"].quantile(worst_pnl_quantile)
    forced_cols = [
        "stopped_by_regime_break",
        "stopped_by_max_loss",
        "stopped_by_max_holding",
        "stopped_by_volatility_shock",
        "stopped_by_exposure",
        "stopped_by_kill_switch",
    ]
    frame["dangerous_grid"] = (
        frame["grid_survived"].astype(int).eq(0)
        | frame["realized_pnl"].le(worst_cutoff)
        | frame[forced_cols].fillna(0).astype(int).max(axis=1).eq(1)
    )
    return (
        frame.groupby(["dataset_split", "score_decile"], observed=False)
        .agg(
            rows=("grid_survived", "size"),
            min_score=("entry_score", "min"),
            max_score=("entry_score", "max"),
            survival_rate=("grid_survived", "mean"),
            realized_pnl_mean=("realized_pnl", "mean"),
            realized_pnl_sum=("realized_pnl", "sum"),
            mae_mean=("max_adverse_excursion", "mean"),
            avg_levels=("number_of_levels_filled", "mean"),
            regime_exits=("stopped_by_regime_break", "sum"),
            forced_exits=("dangerous_grid", "sum"),
        )
        .reset_index()
    )


def _candidate_mask(frame: pd.DataFrame, candidate: ThresholdCandidate) -> pd.Series:
    mask = frame["entry_score"].ge(candidate.open_threshold)
    add_blocked = frame["number_of_levels_filled"].gt(1) & frame["entry_score"].lt(candidate.add_threshold)
    mask = mask & ~add_blocked
    if candidate.kill_switch_threshold is not None:
        mask = mask & frame["intratrade_min_score"].ge(candidate.kill_switch_threshold)
    return mask


def evaluate_candidate_on_split(
    scored: pd.DataFrame,
    candidate: ThresholdCandidate,
    split: str,
    worst_pnl_quantile: float,
) -> dict[str, Any]:
    frame = scored[(scored["dataset_split"] == split) & (scored["exit_within_entry_split"] == split)]
    if frame.empty:
        raise ValueError(f"No complete rows available for split {split}")
    baseline_grids = len(frame)
    mask = _candidate_mask(frame, candidate)
    selected = frame[mask].sort_index()
    worst_cutoff = frame["realized_pnl"].quantile(worst_pnl_quantile)
    forced_cols = [
        "stopped_by_regime_break",
        "stopped_by_max_loss",
        "stopped_by_max_holding",
        "stopped_by_volatility_shock",
        "stopped_by_exposure",
        "stopped_by_kill_switch",
    ]
    if selected.empty:
        metrics = {
            "number_of_grids": 0,
            "expectancy": 0.0,
            "realized_pnl": 0.0,
            "max_drawdown": 0.0,
            "winrate": 0.0,
            "survival_rate": 0.0,
            "number_of_forced_exits": 0,
            "number_of_regime_exits": 0,
            "fees_paid": 0.0,
            "slippage_paid": 0.0,
            "exposure_time": 0.0,
            "dangerous_authorized_grids": 0,
        }
    else:
        pnl = selected["realized_pnl"].astype(float)
        equity = 1.0 + pnl.cumsum()
        drawdown = equity / equity.cummax() - 1.0
        dangerous = (
            selected["grid_survived"].astype(int).eq(0)
            | selected["realized_pnl"].le(worst_cutoff)
            | selected[forced_cols].fillna(0).astype(int).max(axis=1).eq(1)
        )
        metrics = {
            "number_of_grids": int(len(selected)),
            "expectancy": float(pnl.mean()),
            "realized_pnl": float(pnl.sum()),
            "max_drawdown": float(drawdown.min()),
            "winrate": float(pnl.gt(0).mean()),
            "survival_rate": float(selected["grid_survived"].mean()),
            "number_of_forced_exits": int(selected[forced_cols].fillna(0).astype(int).max(axis=1).sum()),
            "number_of_regime_exits": int(selected["stopped_by_regime_break"].sum()),
            "fees_paid": float(selected["fees_paid"].sum()),
            "slippage_paid": float(selected["slippage_paid"].sum()),
            "exposure_time": float(selected["time_to_exit"].sum()),
            "dangerous_authorized_grids": int(dangerous.sum()),
        }
    return {
        **asdict(candidate),
        "split": split,
        "baseline_grids": int(baseline_grids),
        "grids_saved_by_filter": int(baseline_grids - metrics["number_of_grids"]),
        **metrics,
    }


def search_validation_thresholds(scored: pd.DataFrame, candidates: list[ThresholdCandidate], config: dict[str, Any]) -> pd.DataFrame:
    worst_q = float(config["diagnostics"]["worst_pnl_quantile"])
    rows = [evaluate_candidate_on_split(scored, candidate, VALID_SEARCH_SPLIT, worst_q) for candidate in candidates]
    return pd.DataFrame(rows)


def select_best_candidate(validation_summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if validation_summary.empty:
        raise ValueError("Validation summary is empty")
    if set(validation_summary["split"].unique()) != {VALID_SEARCH_SPLIT}:
        raise ValueError("Threshold selection must use validation rows only")
    search = config["threshold_search"]
    baseline_grids = int(validation_summary["baseline_grids"].max())
    minimum = min(int(search["min_grid_absolute_cap"]), int(np.ceil(baseline_grids * float(search["min_grid_fraction_baseline"]))))
    eligible = validation_summary[validation_summary["number_of_grids"] >= minimum].copy()
    if eligible.empty:
        eligible = validation_summary.copy()
    eligible = eligible.sort_values(
        by=["expectancy", "max_drawdown", "number_of_forced_exits"],
        ascending=[False, False, True],
    )
    selected = eligible.iloc[0].to_dict()
    selected["minimum_grid_constraint"] = int(minimum)
    selected["selected_from_validation_only"] = True
    return selected


def _prepare_market_for_exact_backtest() -> pd.DataFrame:
    market = load_processed(project_path("data/processed/btcusdt_5m.parquet"))
    features = pd.read_parquet(project_path("data/features/grid_features.parquet"))
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
    return market


def run_exact_selected_backtest(
    selected: dict[str, Any],
    predictions: pd.DataFrame,
    split: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    risk = validate_strategy_config(load_strategy_config())
    market = _prepare_market_for_exact_backtest()
    split_index = predictions[predictions["dataset_split"] == split].index
    if split_index.empty:
        raise ValueError(f"No predictions for exact backtest split {split}")
    market = market.loc[(market.index >= split_index.min()) & (market.index <= split_index.max())]
    scores = predictions["grid_survival_score"].astype(float).reindex(market.index)
    result = run_grid_backtest(
        market,
        risk,
        allow_open=pd.Series(True, index=market.index),
        scores=scores,
        min_open_score=float(selected["open_threshold"]),
        add_level_min_score=float(selected["add_threshold"]),
        kill_switch_threshold=None
        if pd.isna(selected["kill_switch_threshold"])
        else float(selected["kill_switch_threshold"]),
        constant_size=False,
    )
    metrics = calculate_metrics(result.equity_curve, result.trades)
    metrics.update(
        {
            "split": split,
            "open_threshold": float(selected["open_threshold"]),
            "add_threshold": float(selected["add_threshold"]),
            "kill_switch_threshold": None
            if pd.isna(selected["kill_switch_threshold"])
            else float(selected["kill_switch_threshold"]),
        }
    )
    return metrics, result.trades, result.equity_curve


def run_permutation_importance(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    perm_cfg = config["permutation_importance"]
    if not bool(perm_cfg["enabled"]):
        return pd.DataFrame()
    bundle_path = project_path("reports/model_reports/best_grid_survival_model.joblib")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing best model bundle: {bundle_path}")
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    validation_index = predictions[predictions["dataset_split"] == VALID_SEARCH_SPLIT].index
    validation_index = validation_index.intersection(features.index).intersection(labels.index)
    if validation_index.empty:
        raise ValueError("No validation rows available for permutation importance")
    sample_rows = min(int(perm_cfg["sample_rows"]), len(validation_index))
    sample_index = validation_index.to_series().sample(
        n=sample_rows,
        random_state=int(perm_cfg["random_state"]),
    ).sort_values()
    result = permutation_importance(
        model,
        features.loc[sample_index, feature_columns],
        labels.loc[sample_index, "grid_survived"].astype(int),
        n_repeats=int(perm_cfg["n_repeats"]),
        random_state=int(perm_cfg["random_state"]),
        scoring=str(perm_cfg["scoring"]),
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def decide_iteration_outcome(exact_test_metrics: dict[str, Any], scored: pd.DataFrame) -> str:
    if int(exact_test_metrics.get("number_of_grids", 0)) == 0:
        return "no-trade / no-edge"
    if float(exact_test_metrics.get("expectancy", 0.0)) > 0 and float(exact_test_metrics.get("realized_pnl", 0.0)) > 0:
        return "ML filter viable"
    take_profit = scored[scored["exit_reason"] == "take_profit"]
    positive_take_profit_rate = float(take_profit["realized_pnl"].gt(0).mean()) if not take_profit.empty else 0.0
    if positive_take_profit_rate < 0.25:
        return "needs relabeling"
    return "no-trade / no-edge"


def write_report(
    output_dir: Path,
    selected: dict[str, Any],
    validation_exact: dict[str, Any],
    test_exact: dict[str, Any],
    decision: str,
    config: dict[str, Any],
) -> Path:
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 001 - ML First Threshold Research",
        "",
        "## Decision",
        f"`{decision}`",
        "",
        "## Selected Thresholds",
        f"- Open threshold: `{selected['open_threshold']}`",
        f"- Add threshold: `{selected['add_threshold']}`",
        f"- Kill switch threshold: `{selected['kill_switch_threshold']}`",
        f"- Selected from validation only: `{selected['selected_from_validation_only']}`",
        f"- Minimum grid constraint: `{selected['minimum_grid_constraint']}`",
        "",
        "## Validation Label-Replay Selection Metrics",
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
        "## Caveat",
        "The threshold grid is selected with fast label replay on validation, then checked once with the sequential grid engine. The engine, labels, fees, slippage, and risk rules are unchanged in this iteration.",
        "",
        "## Next Step",
        "If the test expectancy remains negative, move to the economy-first iteration: net-profitable labels, wider take-profit logic after costs, and TP/SL sensitivity.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(config_path: str, max_rows_per_split: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, predictions = load_iteration_inputs(max_rows_per_split=max_rows_per_split)
    scored = labels.join(predictions[["grid_survival_score", "dataset_split"]], how="inner")
    scored = scored[scored["dataset_split"].isin(VALID_EVALUATION_SPLITS)]
    scored = add_intratrade_score_diagnostics(scored, predictions)

    candidates = make_threshold_candidates(config)
    validation_summary = search_validation_thresholds(scored, candidates, config)
    validation_summary.to_csv(output_dir / "threshold_search_validation.csv", index=False)

    selected = select_best_candidate(validation_summary, config)
    test_label_replay = evaluate_candidate_on_split(
        scored,
        ThresholdCandidate(
            open_threshold=float(selected["open_threshold"]),
            add_threshold=float(selected["add_threshold"]),
            kill_switch_threshold=None
            if pd.isna(selected["kill_switch_threshold"])
            else float(selected["kill_switch_threshold"]),
        ),
        "test",
        float(config["diagnostics"]["worst_pnl_quantile"]),
    )
    pd.DataFrame([test_label_replay]).to_csv(output_dir / "selected_threshold_test_label_replay.csv", index=False)

    score_deciles = score_decile_analysis(
        scored,
        deciles=int(config["diagnostics"]["score_deciles"]),
        worst_pnl_quantile=float(config["diagnostics"]["worst_pnl_quantile"]),
    )
    score_deciles.to_csv(output_dir / "score_deciles.csv", index=False)

    importance = run_permutation_importance(features, labels, predictions, config)
    if not importance.empty:
        importance.to_csv(output_dir / "permutation_importance.csv", index=False)

    validation_exact, validation_trades, validation_equity = run_exact_selected_backtest(selected, predictions, "validation")
    test_exact, test_trades, test_equity = run_exact_selected_backtest(selected, predictions, "test")
    pd.DataFrame([validation_exact]).to_csv(output_dir / "selected_threshold_exact_validation_metrics.csv", index=False)
    pd.DataFrame([test_exact]).to_csv(output_dir / "selected_threshold_exact_test_metrics.csv", index=False)
    validation_trades.to_csv(output_dir / "selected_threshold_exact_validation_trades.csv", index=False)
    test_trades.to_csv(output_dir / "selected_threshold_exact_test_trades.csv", index=False)
    validation_equity.to_frame().to_csv(output_dir / "selected_threshold_exact_validation_equity.csv")
    test_equity.to_frame().to_csv(output_dir / "selected_threshold_exact_test_equity.csv")

    decision = decide_iteration_outcome(test_exact, scored)
    payload = {
        "decision": decision,
        "selected_thresholds": selected,
        "validation_exact": validation_exact,
        "test_exact": test_exact,
        "label_replay_test": test_label_replay,
        "candidate_count": len(candidates),
        "max_rows_per_split": max_rows_per_split,
    }
    (output_dir / "selected_thresholds.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, selected, validation_exact, test_exact, decision, config)
    LOGGER.info("Wrote ML threshold iteration outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML-first threshold research iteration.")
    parser.add_argument("--config", default="config/research_iteration_ml.yaml")
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(args.config, max_rows_per_split=args.max_rows_per_split)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
