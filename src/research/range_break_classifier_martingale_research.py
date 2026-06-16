from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.data.validate_data import load_processed
from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.range_break_labels import build_range_break_labels, label_column_for_side
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market
from src.research.fundamental_blackout_ablation_research import trade_attribution_by_fold
from src.research.fundamental_blackout_martingale_research import markdown_table, select_best_exact_no_drawdown
from src.research.monthly_target_martingale_research import (
    MaskLike,
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
from src.research.monte_carlo_oos_robustness import MonteCarloConfig, run_monte_carlo
from src.research.walk_forward_martingale_research import (
    WalkForwardWindow,
    make_walk_forward_windows,
    same_candidate,
    split_frame_from_index,
    stitch_oos_equity,
    summarize_walk_forward,
)
from src.utils.asset_paths import feature_path
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SEARCH_SPLIT = "validation"
VARIANTS = [
    "baseline",
    "fundamental_trend_escape_entry_only",
    "classifier_entry_only",
    "classifier_entry_add_block",
    "classifier_entry_add_emergency_exit_high",
]
MODEL_FEATURE_COLUMNS = [
    "atr_5m",
    "atr_short_over_long",
    "realized_volatility_ratio",
    "bollinger_bandwidth",
    "range_expansion_ratio",
    "candle_range_zscore",
    "ema_slope_5m",
    "ema_slope_15m",
    "ema_slope_30m",
    "ema_slope_1h",
    "adx_15m",
    "adx_1h",
    "price_distance_to_ema_20",
    "price_distance_to_ema_100",
    "higher_highs_count",
    "lower_lows_count",
    "price_position_in_range",
    "distance_to_range_high",
    "distance_to_range_low",
    "closes_near_upper_bound_count",
    "closes_near_lower_bound_count",
    "failed_breakout_count",
    "time_spent_top_quartile",
    "time_spent_bottom_quartile",
    "rejection_wick_score",
    "volume_zscore",
    "relative_volume",
    "volume_expansion_ratio",
    "price_move_per_volume",
    "volume_at_range_boundary",
    "trend_alignment_score",
    "breakout_risk",
    "regime_allows_grid",
    "trend_escape_direction",
    "range_break_up",
    "range_break_down",
    "trend_escape_raw",
    "trend_escape_confirmed",
    "trend_escape",
    "trend_escape_compression_ok",
    "trend_escape_trend_confirmed",
    "event_realistic_blackout",
    "hours_to_next_scheduled_event",
    "hours_since_last_known_event",
    "known_event_recent_24h",
    "side_is_short",
    "side_sign",
    "side_adverse_boundary_pressure",
    "side_adverse_distance_to_boundary",
    "side_favorable_distance_to_boundary",
    "side_adverse_trend_alignment",
]


@dataclass(frozen=True)
class InternalSplit:
    model_train: pd.Index
    selection: pd.Index


@dataclass(frozen=True)
class RangeBreakModel:
    estimator: Any
    feature_columns: list[str]
    model_type: str

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_type == "constant":
            return np.full(len(frame), float(self.estimator), dtype=float)
        proba = self.estimator.predict_proba(frame[self.feature_columns])
        return proba[:, 1].astype(float)


def validate_variants(config: dict[str, Any]) -> list[str]:
    variants = list(config.get("range_break_ablation", {}).get("variants", VARIANTS))
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported range-break variants: {unknown}")
    if "baseline" not in variants or "fundamental_trend_escape_entry_only" not in variants:
        raise ValueError("range-break ablation must include baseline and fundamental_trend_escape_entry_only")
    return variants


def threshold_grid(config: dict[str, Any]) -> list[float]:
    search = config.get("range_break_classifier", {}).get("threshold_search", {})
    start = float(search.get("start", 0.50))
    stop = float(search.get("stop", 0.95))
    step = float(search.get("step", 0.05))
    if not (0 <= start <= stop <= 1) or step <= 0:
        raise ValueError("range_break_classifier threshold_search is incoherent")
    values = np.arange(start, stop + step / 2, step)
    return [round(float(value), 4) for value in values if value <= stop + 1e-12]


def split_internal_train(window: WalkForwardWindow, config: dict[str, Any]) -> InternalSplit:
    split_config = config.get("range_break_classifier", {}).get("internal_split", {})
    model_train_days = float(split_config.get("model_train_days", 120))
    selection_days = float(split_config.get("selection_days", 60))
    if model_train_days <= 0 or selection_days <= 0:
        raise ValueError("internal split days must be positive")
    start = window.train.min()
    model_end = start + pd.Timedelta(days=model_train_days)
    selection_end = model_end + pd.Timedelta(days=selection_days)
    model_train = window.train[(window.train >= start) & (window.train < model_end)]
    selection = window.train[(window.train >= model_end) & (window.train < selection_end)]
    if model_train.empty or selection.empty:
        raise ValueError(f"fold {window.fold_id} produced empty model_train or selection split")
    if model_train.max() >= selection.min() or selection.max() >= window.test.min():
        raise ValueError("internal split must be chronological and before test")
    return InternalSplit(model_train=model_train, selection=selection)


def load_feature_frame(asset: str, market: pd.DataFrame) -> pd.DataFrame:
    path = feature_path(asset)
    if not path.exists():
        raise FileNotFoundError(f"feature file not found: {path}")
    features = pd.read_parquet(path).reindex(market.index)
    return features.replace([np.inf, -np.inf], np.nan)


def build_event_features(index: pd.Index, events: pd.DataFrame, realistic_mask: pd.Series) -> pd.DataFrame:
    if index.tz is None:
        raise ValueError("event feature index must be timezone-aware")
    frame = pd.DataFrame(index=index)
    frame["event_realistic_blackout"] = realistic_mask.reindex(index).fillna(False).astype(int)
    idx_ns = pd.Series(index).astype("int64").to_numpy()

    scheduled = events[events["is_scheduled"].astype(bool)].sort_values("event_time_utc")
    if scheduled.empty:
        frame["hours_to_next_scheduled_event"] = 999.0
    else:
        event_ns = pd.to_datetime(scheduled["event_time_utc"], utc=True).astype("int64").to_numpy()
        pos = np.searchsorted(event_ns, idx_ns, side="left")
        hours = np.full(len(index), 999.0, dtype=float)
        valid = pos < len(event_ns)
        hours[valid] = (event_ns[pos[valid]] - idx_ns[valid]) / 3_600_000_000_000
        frame["hours_to_next_scheduled_event"] = np.clip(hours, 0.0, 999.0)

    known_ns = pd.to_datetime(events["known_time_utc"], utc=True).sort_values().astype("int64").to_numpy()
    if len(known_ns) == 0:
        frame["hours_since_last_known_event"] = 999.0
    else:
        pos = np.searchsorted(known_ns, idx_ns, side="right") - 1
        hours = np.full(len(index), 999.0, dtype=float)
        valid = pos >= 0
        hours[valid] = (idx_ns[valid] - known_ns[pos[valid]]) / 3_600_000_000_000
        frame["hours_since_last_known_event"] = np.clip(hours, 0.0, 999.0)
    frame["known_event_recent_24h"] = frame["hours_since_last_known_event"].le(24).astype(int)
    return frame


def build_model_base_frame(
    market: pd.DataFrame,
    features: pd.DataFrame,
    trend_components: pd.DataFrame,
    event_features: pd.DataFrame,
) -> pd.DataFrame:
    base = market.join(features, how="left", rsuffix="_feature")
    for column in features.columns:
        feature_column = f"{column}_feature"
        if feature_column in base.columns and column not in base.columns:
            base[column] = base[feature_column]
    base = base.join(trend_components, how="left", rsuffix="_trend")
    base = base.join(event_features, how="left")
    base = base.replace([np.inf, -np.inf], np.nan)
    return base


def _side_features(base: pd.DataFrame, side: str) -> pd.DataFrame:
    out = pd.DataFrame(index=base.index)
    side_sign = 1.0 if side == "long" else -1.0
    position = base.get("price_position_in_range", pd.Series(np.nan, index=base.index)).astype(float)
    trend = base.get("trend_alignment_score", pd.Series(0.0, index=base.index)).fillna(0.0).astype(float)
    distance_high = base.get("distance_to_range_high", pd.Series(np.nan, index=base.index)).astype(float)
    distance_low = base.get("distance_to_range_low", pd.Series(np.nan, index=base.index)).astype(float)
    out["side_is_short"] = 1 if side == "short" else 0
    out["side_sign"] = side_sign
    out["side_adverse_boundary_pressure"] = 1.0 - position if side == "long" else position
    out["side_adverse_distance_to_boundary"] = distance_low if side == "long" else distance_high
    out["side_favorable_distance_to_boundary"] = distance_high if side == "long" else distance_low
    out["side_adverse_trend_alignment"] = -side_sign * trend
    return out


def build_side_rows(
    base: pd.DataFrame,
    labels: pd.DataFrame | None,
    index: pd.Index,
    feature_columns: list[str],
    horizon_hours: float,
    sides: tuple[str, ...] = ("long", "short"),
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    labels_out: list[pd.Series] = []
    meta: list[pd.DataFrame] = []
    base_split = base.reindex(index)
    for side in sides:
        side_frame = base_split.copy()
        side_frame = side_frame.join(_side_features(base_split, side))
        rows.append(side_frame[feature_columns])
        meta.append(pd.DataFrame({"timestamp": index, "side": side}, index=index))
        if labels is not None:
            label_col = label_column_for_side(side, horizon_hours)
            labels_out.append(labels.reindex(index)[label_col])
    X = pd.concat(rows, axis=0)
    y = pd.concat(labels_out, axis=0) if labels is not None else None
    meta_frame = pd.concat(meta, axis=0)
    return X, y, meta_frame


def training_cutoff(index: pd.Index, horizon_hours: float) -> pd.Timestamp:
    return index.max() - pd.Timedelta(hours=float(horizon_hours))


def train_range_break_model(
    base: pd.DataFrame,
    labels: pd.DataFrame,
    split: InternalSplit,
    config: dict[str, Any],
) -> tuple[RangeBreakModel, dict[str, Any]]:
    classifier_config = config.get("range_break_classifier", {})
    horizon = float(config.get("range_break_label", {}).get("primary_horizon_hours", 6))
    feature_columns = [column for column in MODEL_FEATURE_COLUMNS if column in base.columns or column.startswith("side_")]
    train_index = split.model_train[split.model_train <= training_cutoff(split.model_train, horizon)]
    X, y, _meta = build_side_rows(base, labels, train_index, feature_columns, horizon)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    if X.empty:
        raise ValueError("range-break model training frame is empty")
    positive_rate = float(y.mean())
    if y.nunique() < 2:
        model = RangeBreakModel(positive_rate, feature_columns, "constant")
        return model, {"model_type": "constant", "train_rows": int(len(y)), "train_positive_rate": positive_rate}

    try:
        estimator: Any = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=int(classifier_config.get("max_iter", 120)),
                        learning_rate=float(classifier_config.get("learning_rate", 0.06)),
                        max_leaf_nodes=int(classifier_config.get("max_leaf_nodes", 15)),
                        l2_regularization=float(classifier_config.get("l2_regularization", 0.05)),
                        random_state=int(classifier_config.get("random_state", 42)),
                    ),
                ),
            ]
        )
        counts = y.value_counts()
        weights = y.map({0: len(y) / (2 * counts[0]), 1: len(y) / (2 * counts[1])}).astype(float)
        estimator.fit(X[feature_columns], y, model__sample_weight=weights)
        model_type = "hist_gradient_boosting"
    except Exception as exc:
        LOGGER.warning("HistGradientBoosting failed, using LogisticRegression fallback: %s", exc)
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=int(classifier_config.get("random_state", 42)),
                    ),
                ),
            ]
        )
        estimator.fit(X[feature_columns], y)
        model_type = "logistic_regression_fallback"
    model = RangeBreakModel(estimator, feature_columns, model_type)
    metrics = {"model_type": model_type, "train_rows": int(len(y)), "train_positive_rate": positive_rate}
    return model, metrics


def predict_scores_by_side(model: RangeBreakModel, base: pd.DataFrame) -> dict[str, pd.Series]:
    scores: dict[str, pd.Series] = {}
    horizon = 6.0
    for side in ("long", "short"):
        X, _y, _meta = build_side_rows(base, None, base.index, model.feature_columns, horizon, sides=(side,))
        scores[side] = pd.Series(model.predict_probability(X), index=base.index, name=f"range_break_score_{side}")
    return scores


def score_masks(scores: dict[str, pd.Series], threshold: float) -> dict[str, pd.Series]:
    return {side: series.ge(float(threshold)).fillna(False).astype(bool) for side, series in scores.items()}


def combine_mask_and_scores(mask: pd.Series, scores: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return {side: (series.index.to_series().map(mask).fillna(False).astype(bool) & series.ge(-1)).astype(bool) for side, series in scores.items()}


def fundamental_trend_mask(trend_escape: pd.Series, blackout_masks: dict[str, pd.Series]) -> pd.Series:
    return (trend_escape.astype(bool) & blackout_masks["realistic"].astype(bool)).astype(bool)


def mask_time_fraction(mask: MaskLike, index: pd.Index) -> float:
    if mask is None:
        return 0.0
    if isinstance(mask, dict):
        values = [series.reindex(index).fillna(False).astype(bool).mean() for series in mask.values()]
        return float(np.mean(values)) if values else 0.0
    return float(mask.reindex(index).fillna(False).astype(bool).mean())


def _add_variant_metrics(
    metrics: dict[str, Any],
    trades: pd.DataFrame,
    entry_mask: MaskLike,
    add_block_mask: MaskLike,
    emergency_exit_mask: MaskLike,
) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["range_break_emergency_exit_count"] = (
        int(trades["exit_reason"].eq("range_break_emergency").sum())
        if not trades.empty and "exit_reason" in trades.columns
        else 0
    )
    metrics["entry_mask_time_fraction"] = 0.0 if entry_mask is None else np.nan
    metrics["add_block_time_fraction"] = 0.0 if add_block_mask is None else np.nan
    metrics["emergency_exit_time_fraction"] = 0.0 if emergency_exit_mask is None else np.nan
    return metrics


def run_exact_candidate_on_index(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidate: MonthlyMartingaleCandidate,
    split_index: pd.Index,
    split_name: str,
    entry_mask: MaskLike = None,
    add_block_mask: MaskLike = None,
    emergency_exit_mask: MaskLike = None,
) -> tuple[dict[str, Any], SignalGridBacktestResult]:
    split_frame = split_frame_from_index(market, split_index)
    risk = risk_for_candidate(base_risk, candidate)
    side_signal = build_side_signal(market, signal_frame, candidate)
    result = run_signal_grid_backtest(
        split_frame,
        risk,
        side_signal,
        candidate,
        entry_blackout_series=entry_mask,
        add_block_series=add_block_mask,
        emergency_exit_series=emergency_exit_mask,
        emergency_exit_reason="range_break_emergency",
    )
    metrics = summarize_exact(result.equity_curve, result.trades, candidate, split_name)
    return _add_variant_metrics(metrics, result.trades, entry_mask, add_block_mask, emergency_exit_mask), result


def evaluate_exact_candidates_for_variant(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidate_rows: list[dict[str, Any]],
    split_index: pd.Index,
    split_name: str,
    entry_mask: MaskLike,
    add_block_mask: MaskLike,
    emergency_exit_mask: MaskLike,
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
            add_block_mask=add_block_mask,
            emergency_exit_mask=emergency_exit_mask,
        )
        rows.append(metrics)
        results[candidate.name] = result
    if not rows:
        raise ValueError(f"No exact rows evaluated for {split_name}")
    return pd.DataFrame(rows), results


def masks_for_variant(
    variant: str,
    threshold: float | None,
    scores: dict[str, pd.Series] | None,
    fundamental_entry_mask: pd.Series,
    emergency_threshold: float,
) -> tuple[MaskLike, MaskLike, MaskLike]:
    if variant == "baseline":
        return None, None, None
    if variant == "fundamental_trend_escape_entry_only":
        return fundamental_entry_mask, None, None
    if scores is None or threshold is None:
        raise ValueError(f"classifier variant {variant} requires scores and threshold")
    entry = score_masks(scores, threshold)
    add_block = None
    emergency = None
    if variant in {"classifier_entry_add_block", "classifier_entry_add_emergency_exit_high"}:
        add_block = score_masks(scores, threshold)
    if variant == "classifier_entry_add_emergency_exit_high":
        emergency = score_masks(scores, emergency_threshold)
    if variant == "classifier_entry_only":
        return entry, None, None
    if variant in {"classifier_entry_add_block", "classifier_entry_add_emergency_exit_high"}:
        return entry, add_block, emergency
    raise ValueError(f"Unsupported variant: {variant}")


def run_selection_for_masks(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidates: list[MonthlyMartingaleCandidate],
    selection_index: pd.Index,
    config: dict[str, Any],
    entry_mask: MaskLike,
    add_block_mask: MaskLike,
    emergency_exit_mask: MaskLike,
    preselected_candidate_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], MonthlyMartingaleCandidate, pd.DataFrame]:
    if preselected_candidate_rows is None:
        indexes = {SEARCH_SPLIT: selection_index}
        sample_summary = search_validation_sample(
            market,
            signal_frame,
            indexes,
            base_risk,
            candidates,
            config,
            entry_blackout_series=entry_mask,
        )
        top_rows = top_sample_candidates(sample_summary, config)
    else:
        top_rows = preselected_candidate_rows
    validation_exact_frame, _results = evaluate_exact_candidates_for_variant(
        market,
        signal_frame,
        base_risk,
        top_rows,
        selection_index,
        SEARCH_SPLIT,
        entry_mask,
        add_block_mask,
        emergency_exit_mask,
    )
    selected = select_best_exact_no_drawdown(validation_exact_frame, config)
    return selected, candidate_from_row(selected), validation_exact_frame


def select_variant_threshold(
    variant: str,
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidates: list[MonthlyMartingaleCandidate],
    selection_index: pd.Index,
    config: dict[str, Any],
    scores: dict[str, pd.Series] | None,
    fundamental_entry_mask: pd.Series,
    preselected_candidate_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], MonthlyMartingaleCandidate, pd.DataFrame]:
    emergency_threshold = float(config.get("range_break_classifier", {}).get("emergency_exit_threshold", 0.90))
    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    thresholds = [None] if variant in {"baseline", "fundamental_trend_escape_entry_only"} else threshold_grid(config)
    for threshold in thresholds:
        entry_mask, add_block_mask, emergency_exit_mask = masks_for_variant(
            variant,
            threshold,
            scores,
            fundamental_entry_mask,
            emergency_threshold,
        )
        selected, candidate, validation_exact = run_selection_for_masks(
            market,
            signal_frame,
            base_risk,
            candidates,
            selection_index,
            config,
            entry_mask,
            add_block_mask,
            emergency_exit_mask,
            preselected_candidate_rows=preselected_candidate_rows,
        )
        selected = dict(selected)
        selected["variant"] = variant
        selected["range_break_threshold"] = np.nan if threshold is None else float(threshold)
        selected["range_break_emergency_threshold"] = emergency_threshold if emergency_exit_mask is not None else np.nan
        selected["selected_candidate_name"] = candidate.name
        rows.append(selected)
        validation_exact = validation_exact.copy()
        validation_exact["variant"] = variant
        validation_exact["range_break_threshold"] = np.nan if threshold is None else float(threshold)
        frames.append(validation_exact)

    frame = pd.DataFrame(rows)
    frame["target_reached"] = frame["monthly_return"].astype(float) >= float(config["target"]["monthly_return"])
    frame = frame.sort_values(
        ["target_reached", "monthly_return", "profit_factor", "number_of_forced_exits"],
        ascending=[False, False, False, True],
    )
    best = frame.iloc[0].to_dict()
    best["threshold_selected_on_selection_only"] = True
    return best, candidate_from_row(best), pd.concat(frames, ignore_index=True)


def run_fold_for_variant(
    variant: str,
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk,
    candidates: list[MonthlyMartingaleCandidate],
    window: WalkForwardWindow,
    internal_split: InternalSplit,
    config: dict[str, Any],
    scores: dict[str, pd.Series] | None,
    fundamental_entry_mask: pd.Series,
    preselected_candidate_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame]:
    selected, candidate, validation_exact = select_variant_threshold(
        variant,
        market,
        signal_frame,
        base_risk,
        candidates,
        internal_split.selection,
        config,
        scores,
        fundamental_entry_mask,
        preselected_candidate_rows=preselected_candidate_rows,
    )
    threshold = selected.get("range_break_threshold")
    threshold_value = None if pd.isna(threshold) else float(threshold)
    emergency_threshold = float(config.get("range_break_classifier", {}).get("emergency_exit_threshold", 0.90))
    entry_mask, add_block_mask, emergency_exit_mask = masks_for_variant(
        variant,
        threshold_value,
        scores,
        fundamental_entry_mask,
        emergency_threshold,
    )
    test_metrics, test_result = run_exact_candidate_on_index(
        market,
        signal_frame,
        base_risk,
        candidate,
        window.test,
        "test",
        entry_mask=entry_mask,
        add_block_mask=add_block_mask,
        emergency_exit_mask=emergency_exit_mask,
    )
    if not same_candidate(selected, test_metrics):
        raise ValueError("selected candidate changed before fold test evaluation")

    train_frame = split_frame_from_index(market, internal_split.model_train)
    selection_frame = split_frame_from_index(market, internal_split.selection)
    test_frame = split_frame_from_index(market, window.test)
    fold_row = {
        "variant": variant,
        "fold_id": window.fold_id,
        "model_train_start": internal_split.model_train.min(),
        "model_train_end": internal_split.model_train.max(),
        "selection_start": internal_split.selection.min(),
        "selection_end": internal_split.selection.max(),
        "test_start": window.test.min(),
        "test_end": window.test.max(),
        "selected_name": candidate.name,
        "selected_threshold": threshold_value if threshold_value is not None else np.nan,
        "selected_emergency_threshold": emergency_threshold if emergency_exit_mask is not None else np.nan,
        "selection_monthly_return": float(selected["monthly_return"]),
        "selection_total_return": float(selected["total_return"]),
        "selection_max_drawdown": float(selected["max_drawdown"]),
        "selection_profit_factor": float(selected["profit_factor"]),
        "selection_target_reached": bool(float(selected["monthly_return"]) >= float(config["target"]["monthly_return"])),
        "validation_target_reached": bool(float(selected["monthly_return"]) >= float(config["target"]["monthly_return"])),
        "test_monthly_return": float(test_metrics["monthly_return"]),
        "test_total_return": float(test_metrics["total_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_profit_factor": float(test_metrics["profit_factor"]),
        "test_number_of_grids": int(test_metrics["number_of_grids"]),
        "test_positive": bool(float(test_metrics["total_return"]) > 0),
        "test_target_reached": bool(float(test_metrics["monthly_return"]) >= float(config["target"]["monthly_return"])),
        "test_equity_ruined": bool(test_result.equity_curve.min() <= 0),
        "model_train_entry_mask_time_fraction": mask_time_fraction(entry_mask, train_frame.index),
        "selection_entry_mask_time_fraction": mask_time_fraction(entry_mask, selection_frame.index),
        "test_entry_mask_time_fraction": mask_time_fraction(entry_mask, test_frame.index),
        "selection_add_block_time_fraction": mask_time_fraction(add_block_mask, selection_frame.index),
        "test_add_block_time_fraction": mask_time_fraction(add_block_mask, test_frame.index),
        "selection_emergency_exit_time_fraction": mask_time_fraction(emergency_exit_mask, selection_frame.index),
        "test_emergency_exit_time_fraction": mask_time_fraction(emergency_exit_mask, test_frame.index),
        "range_break_emergency_exit_count": int(test_metrics["range_break_emergency_exit_count"]),
    }
    selected_row = {"variant": variant, "fold_id": window.fold_id, **selected}
    return fold_row, selected_row, test_result.trades, test_result.equity_curve, validation_exact


def evaluate_model_on_split(
    model: RangeBreakModel,
    base: pd.DataFrame,
    labels: pd.DataFrame,
    split_index: pd.Index,
    split_name: str,
    fold_id: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    horizon = float(config.get("range_break_label", {}).get("primary_horizon_hours", 6))
    X, y, meta = build_side_rows(base, labels, split_index, model.feature_columns, horizon)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    meta = meta.loc[valid]
    if X.empty:
        raise ValueError(f"empty model evaluation split: {split_name}")
    score = pd.Series(model.predict_probability(X), index=X.index)
    metrics: dict[str, Any] = {
        "fold_id": fold_id,
        "split": split_name,
        "model_type": model.model_type,
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "average_precision": np.nan,
        "roc_auc": np.nan,
    }
    if y.nunique() > 1:
        metrics["average_precision"] = float(average_precision_score(y, score))
        metrics["roc_auc"] = float(roc_auc_score(y, score))
    deciles = pd.DataFrame({"score": score.to_numpy(), "label": y.to_numpy(), "side": meta["side"].to_numpy()})
    deciles["fold_id"] = fold_id
    deciles["split"] = split_name
    try:
        deciles["score_decile"] = pd.qcut(deciles["score"], 10, labels=False, duplicates="drop")
    except ValueError:
        deciles["score_decile"] = 0
    grouped = (
        deciles.groupby(["fold_id", "split", "score_decile"], as_index=False)
        .agg(rows=("label", "size"), avg_score=("score", "mean"), danger_rate=("label", "mean"))
        .sort_values(["fold_id", "split", "score_decile"])
    )
    return metrics, grouped


def compute_permutation_importance_rows(
    model: RangeBreakModel,
    base: pd.DataFrame,
    labels: pd.DataFrame,
    split_index: pd.Index,
    fold_id: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if model.model_type == "constant":
        return []
    importance_config = config.get("range_break_classifier", {}).get("permutation_importance", {})
    if not bool(importance_config.get("enabled", True)):
        return []
    max_folds = importance_config.get("max_folds")
    if max_folds is not None and int(fold_id) > int(max_folds):
        return []
    sample_size = int(importance_config.get("sample_size", 3000))
    repeats = int(importance_config.get("n_repeats", 3))
    horizon = float(config.get("range_break_label", {}).get("primary_horizon_hours", 6))
    X, y, _meta = build_side_rows(base, labels, split_index, model.feature_columns, horizon)
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    if X.empty or y.nunique() < 2:
        return []
    if len(X) > sample_size:
        rng = np.random.default_rng(int(config.get("range_break_classifier", {}).get("random_state", 42)))
        positions = np.sort(rng.choice(len(X), size=sample_size, replace=False))
        X = X.iloc[positions]
        y = y.iloc[positions]
    result = permutation_importance(
        model.estimator,
        X[model.feature_columns],
        y,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=int(config.get("range_break_classifier", {}).get("random_state", 42)),
    )
    rows: list[dict[str, Any]] = []
    for feature, mean, std in zip(model.feature_columns, result.importances_mean, result.importances_std):
        rows.append(
            {
                "fold_id": fold_id,
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
        )
    return rows


def summarize_variant(variant: str, fold_rows: list[dict[str, Any]], oos_equity: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    fold_summary = pd.DataFrame(fold_rows)
    summary = summarize_walk_forward(fold_summary, oos_equity, config)
    summary["variant"] = variant
    summary["entry_mask_time_fraction"] = float(fold_summary["test_entry_mask_time_fraction"].mean())
    summary["add_block_time_fraction"] = float(fold_summary["test_add_block_time_fraction"].mean())
    summary["emergency_exit_time_fraction"] = float(fold_summary["test_emergency_exit_time_fraction"].mean())
    summary["range_break_emergency_exit_count"] = int(fold_summary["range_break_emergency_exit_count"].sum())
    summary["test_grid_count"] = int(fold_summary["test_number_of_grids"].sum())
    summary["worst_fold_total_return"] = float(fold_summary["test_total_return"].min())
    summary["worst_fold_monthly_return"] = float(fold_summary["test_monthly_return"].min())
    summary["report_only_max_drawdown"] = summary["aggregate_max_drawdown"]
    return summary


def decide_range_break(comparison: pd.DataFrame, config: dict[str, Any]) -> str:
    baseline = comparison[comparison["variant"] == "baseline"].iloc[0]
    fundamental = comparison[comparison["variant"] == "fundamental_trend_escape_entry_only"].iloc[0]
    classifier = comparison[comparison["variant"].astype(str).str.startswith("classifier_")].copy()
    classifier["fundamental_delta"] = classifier["aggregate_monthly_return"].astype(float) - float(
        fundamental["aggregate_monthly_return"]
    )
    best = classifier.sort_values(
        ["aggregate_monthly_return", "positive_fold_rate", "worst_fold_total_return"],
        ascending=[False, False, False],
    ).iloc[0]
    target = float(config["target"]["monthly_return"])
    min_positive = float(config["target"]["min_positive_fold_rate"])
    min_target = float(config["target"]["min_target_fold_rate"])
    if (
        float(best["aggregate_monthly_return"]) >= target
        and float(best["positive_fold_rate"]) >= min_positive
        and float(best["target_fold_rate"]) >= min_target
        and float(best["fundamental_delta"]) > 0
    ):
        return "range-break classifier target viable"
    if float(best["fundamental_delta"]) > 0:
        return "range-break classifier improves but below target"
    if float(best["aggregate_max_drawdown"]) > float(fundamental["aggregate_max_drawdown"]):
        return "range-break classifier risk reduction only"
    if float(fundamental["aggregate_monthly_return"]) > float(baseline["aggregate_monthly_return"]):
        return "fundamental trend escape remains best"
    return "no range-break classifier edge"


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    comparison = pd.DataFrame(payload["comparison"])
    model_metrics = pd.DataFrame(payload["model_metrics"])
    lines = [
        "# iteration_018_range_break_classifier_martingale",
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
                    "fundamental_improvement_monthly",
                    "positive_fold_rate",
                    "target_fold_rate",
                    "entry_mask_time_fraction",
                    "add_block_time_fraction",
                    "emergency_exit_time_fraction",
                    "test_grid_count",
                    "aggregate_max_drawdown",
                    "worst_fold_monthly_return",
                ]
            ]
        ),
        "",
        "## Classifier Diagnostics",
        markdown_table(
            model_metrics.groupby(["split", "model_type"], as_index=False)[
                ["rows", "positive_rate", "average_precision", "roc_auc"]
            ].mean(numeric_only=True)
        )
        if not model_metrics.empty
        else "No classifier metrics.",
        "",
        "## Interpretation",
        "The classifier is trained on the first internal window, thresholds and candidates are selected only on the selection window, and the selected policy is applied unchanged to the OOS test fold. Drawdown remains report-only.",
    ]
    output_dir.joinpath("iteration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_iteration(
    config_path: str,
    max_folds: int | None = None,
    max_candidates: int | None = None,
    exact_top_n: int | None = None,
    skip_monte_carlo: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if exact_top_n is not None:
        config["search"]["exact_top_n"] = int(exact_top_n)
    variants = validate_variants(config)
    asset = str(config.get("asset", "btcusdt"))
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = output_dir / "selected_fold_trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    for old_trade_file in trades_dir.glob("*_fold_*_trades.csv"):
        old_trade_file.unlink()

    base_risk = validate_strategy_config(load_strategy_config())
    market = prepare_market(asset)
    features = load_feature_frame(asset, market)
    signal_frame = load_processed(project_path(f"data/processed/{asset}_1h.parquet"))
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
    labels = build_range_break_labels(market.join(features, how="left", rsuffix="_feature"), config)
    event_features = build_event_features(market.index, events, blackout_masks["realistic"])
    model_base = build_model_base_frame(market, features, trend_components, event_features)
    trend_escape = trend_components["trend_escape"].astype(bool)
    fundamental_entry = fundamental_trend_mask(trend_escape, blackout_masks)
    events.to_csv(output_dir / "fundamental_events.csv", index=False)
    windows_frame.to_csv(output_dir / "blackout_windows.csv", index=False)
    trend_components.to_csv(output_dir / "trend_escape_components.csv")
    labels.to_csv(output_dir / "range_break_labels.csv")

    candidates = make_candidates(config)
    total_candidate_count = len(candidates)
    sample_size = max_candidates if max_candidates is not None else config["search"].get("candidate_sample_size")
    candidates = choose_candidate_subset(
        candidates,
        None if sample_size is None else int(sample_size),
        int(config["search"].get("candidate_sample_seed", 42)),
    )

    comparison_rows: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    selected_threshold_rows: list[dict[str, Any]] = []
    validation_exact_rows: list[pd.DataFrame] = []
    model_metric_rows: list[dict[str, Any]] = []
    decile_rows: list[pd.DataFrame] = []
    importance_rows: list[dict[str, Any]] = []
    variant_trade_frames: dict[str, dict[int, pd.DataFrame]] = {variant: {} for variant in variants}
    variant_selected_by_fold: dict[str, dict[int, str]] = {variant: {} for variant in variants}

    model_by_fold: dict[int, RangeBreakModel] = {}
    scores_by_fold: dict[int, dict[str, pd.Series]] = {}
    internal_splits: dict[int, InternalSplit] = {}
    classifier_candidate_rows_by_fold: dict[int, list[dict[str, Any]]] = {}
    for window in windows:
        LOGGER.info("Preparing range-break model fold %s/%s", window.fold_id, len(windows))
        split = split_internal_train(window, config)
        internal_splits[window.fold_id] = split
        LOGGER.info("Fold %s: training classifier", window.fold_id)
        model, train_metrics = train_range_break_model(model_base, labels, split, config)
        model_by_fold[window.fold_id] = model
        LOGGER.info("Fold %s: predicting side-aware scores", window.fold_id)
        scores_by_fold[window.fold_id] = predict_scores_by_side(model, model_base)
        LOGGER.info("Fold %s: preselecting candidate rows on selection window", window.fold_id)
        classifier_sample = search_validation_sample(
            market,
            signal_frame,
            {SEARCH_SPLIT: split.selection},
            base_risk,
            candidates,
            config,
        )
        classifier_candidate_rows_by_fold[window.fold_id] = top_sample_candidates(classifier_sample, config)
        train_metrics.update({"fold_id": window.fold_id, "split": "model_train"})
        model_metric_rows.append(train_metrics)
        for split_name, split_index in {"selection": split.selection, "test": window.test}.items():
            LOGGER.info("Fold %s: evaluating classifier on %s", window.fold_id, split_name)
            metrics, deciles = evaluate_model_on_split(model, model_base, labels, split_index, split_name, window.fold_id, config)
            model_metric_rows.append(metrics)
            decile_rows.append(deciles)
        LOGGER.info("Fold %s: computing permutation importance if enabled", window.fold_id)
        importance_rows.extend(
            compute_permutation_importance_rows(model, model_base, labels, split.selection, window.fold_id, config)
        )
        LOGGER.info("Fold %s: preparation complete", window.fold_id)

    for variant in variants:
        LOGGER.info("Running range-break variant: %s", variant)
        fold_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        fold_equities: list[tuple[int, pd.Series]] = []
        for window in windows:
            LOGGER.info("Running %s fold %s/%s", variant, window.fold_id, len(windows))
            scores = scores_by_fold[window.fold_id] if variant.startswith("classifier_") else None
            fold_row, selected_row, trades, equity, validation_exact = run_fold_for_variant(
                variant,
                market,
                signal_frame,
                base_risk,
                candidates,
                window,
                internal_splits[window.fold_id],
                config,
                scores,
                fundamental_entry,
                preselected_candidate_rows=classifier_candidate_rows_by_fold[window.fold_id]
                if variant.startswith("classifier_")
                else None,
            )
            fold_rows.append(fold_row)
            selected_rows.append(selected_row)
            selected_rows_all.append(selected_row)
            selected_threshold_rows.append(
                {
                    "variant": variant,
                    "fold_id": window.fold_id,
                    "selected_threshold": fold_row["selected_threshold"],
                    "selected_emergency_threshold": fold_row["selected_emergency_threshold"],
                    "selected_name": fold_row["selected_name"],
                    "selection_monthly_return": fold_row["selection_monthly_return"],
                    "test_monthly_return": fold_row["test_monthly_return"],
                }
            )
            validation_exact_rows.append(validation_exact.assign(fold_id=window.fold_id))
            fold_equities.append((window.fold_id, equity))
            variant_trade_frames[variant][window.fold_id] = trades
            variant_selected_by_fold[variant][window.fold_id] = str(fold_row["selected_name"])
            trades.to_csv(trades_dir / f"{variant}_fold_{window.fold_id:03d}_trades.csv", index=False)

        fold_summary = pd.DataFrame(fold_rows)
        selected_candidates = pd.DataFrame(selected_rows)
        equity_frame = stitch_oos_equity(fold_equities)
        fold_summary.to_csv(output_dir / f"walk_forward_fold_summary_{variant}.csv", index=False)
        selected_candidates.to_csv(output_dir / f"walk_forward_selected_candidates_{variant}.csv", index=False)
        equity_frame.to_csv(output_dir / f"walk_forward_oos_equity_{variant}.csv")
        comparison_rows.append(summarize_variant(variant, fold_rows, equity_frame["equity"], config))

    comparison = pd.DataFrame(comparison_rows)
    baseline_monthly = float(comparison[comparison["variant"] == "baseline"]["aggregate_monthly_return"].iloc[0])
    fundamental_monthly = float(
        comparison[comparison["variant"] == "fundamental_trend_escape_entry_only"]["aggregate_monthly_return"].iloc[0]
    )
    baseline_grids = int(comparison[comparison["variant"] == "baseline"]["test_grid_count"].iloc[0])
    comparison["baseline_improvement_monthly"] = comparison["aggregate_monthly_return"].astype(float) - baseline_monthly
    comparison["fundamental_improvement_monthly"] = comparison["aggregate_monthly_return"].astype(float) - fundamental_monthly
    comparison["grids_removed_vs_baseline"] = baseline_grids - comparison["test_grid_count"].astype(int)

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

    comparison.to_csv(output_dir / "walk_forward_range_break_comparison.csv", index=False)
    pd.DataFrame(selected_rows_all).to_csv(output_dir / "walk_forward_selected_candidates.csv", index=False)
    pd.DataFrame(selected_threshold_rows).to_csv(output_dir / "walk_forward_selected_thresholds.csv", index=False)
    pd.concat(validation_exact_rows, ignore_index=True).to_csv(output_dir / "validation_exact_by_threshold.csv", index=False)
    pd.DataFrame(model_metric_rows).to_csv(output_dir / "range_break_model_metrics.csv", index=False)
    pd.concat(decile_rows, ignore_index=True).to_csv(output_dir / "score_deciles.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_dir / "permutation_importance.csv", index=False)
    pd.DataFrame(attribution_rows).to_csv(output_dir / "trade_attribution_by_fold.csv", index=False)

    decision = decide_range_break(comparison, config)
    monte_carlo_payloads: list[dict[str, Any]] = []
    if not skip_monte_carlo:
        best_variant = (
            comparison[comparison["variant"].astype(str).str.startswith("classifier_")]
            .sort_values("aggregate_monthly_return", ascending=False)
            .iloc[0]["variant"]
        )
        mc_iterations = int(config.get("monte_carlo", {}).get("iterations", 5000))
        for variant in ["baseline", "fundamental_trend_escape_entry_only", str(best_variant)]:
            monte_carlo_payloads.append(
                run_monte_carlo(
                    MonteCarloConfig(
                        iteration_dir=output_dir,
                        variant=variant,
                        iterations=mc_iterations,
                        seed=int(config.get("monte_carlo", {}).get("seed", 42)),
                        target_monthly_return=float(config["target"]["monthly_return"]),
                        output_dir=output_dir / "monte_carlo",
                    )
                )
            )

    payload = {
        "decision": decision,
        "iteration_name": str(config["iteration"].get("name", "iteration_018_range_break_classifier_martingale")),
        "comparison": comparison.to_dict("records"),
        "model_metrics": pd.DataFrame(model_metric_rows).to_dict("records"),
        "trade_attribution": attribution_rows,
        "monte_carlo": monte_carlo_payloads,
        "fold_count": len(windows),
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
        "exact_top_n": int(config["search"]["exact_top_n"]),
        "threshold_grid": threshold_grid(config),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(output_dir, payload)
    LOGGER.info("Wrote range-break classifier martingale outputs to %s", output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run range-break classifier martingale walk-forward research.")
    parser.add_argument("--config", default="config/research_iteration_range_break_classifier_martingale.yaml")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--exact-top-n", type=int, default=None)
    parser.add_argument("--skip-monte-carlo", action="store_true")
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        max_folds=args.max_folds,
        max_candidates=args.max_candidates,
        exact_top_n=args.exact_top_n,
        skip_monte_carlo=bool(args.skip_monte_carlo),
    )
    compact = {
        "decision": payload["decision"],
        "fold_count": payload["fold_count"],
        "candidate_count": payload["candidate_count"],
        "exact_top_n": payload["exact_top_n"],
        "comparison": [
            {
                "variant": row["variant"],
                "aggregate_monthly_return": row["aggregate_monthly_return"],
                "positive_fold_rate": row["positive_fold_rate"],
                "target_fold_rate": row["target_fold_rate"],
                "aggregate_max_drawdown": row["aggregate_max_drawdown"],
            }
            for row in payload["comparison"]
        ],
    }
    print(json.dumps(compact, indent=2, default=str))


if __name__ == "__main__":
    main()
