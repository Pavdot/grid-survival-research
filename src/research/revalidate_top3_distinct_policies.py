from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.metrics import calculate_metrics, drawdown_series
from src.data.validate_data import load_processed, validate_ohlcv
from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.range_break_labels import build_range_break_labels
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market, summarize_simulations
from src.research.execution_worst_case_audit import (
    AuditScenario,
    run_signal_grid_backtest_audit,
    sample_positions_for_scenario,
    simulate_candidate_sample_for_scenario,
)
from src.research.fundamental_blackout_martingale_research import (
    markdown_table,
    select_best_exact_no_drawdown,
)
from src.research.monthly_target_martingale_research import (
    MaskLike,
    MonthlyMartingaleCandidate,
    build_side_signal,
    candidate_from_row,
    monthly_return_from_equity,
    risk_for_candidate,
    run_signal_grid_backtest,
    sample_positions,
    simulate_candidate_sample,
    summarize_exact,
)
from src.research.range_break_classifier_martingale_research import (
    build_event_features,
    build_model_base_frame,
    fundamental_trend_mask,
    load_feature_frame,
    masks_for_variant,
    predict_scores_by_side,
    split_internal_train,
    train_range_break_model,
)
from src.research.walk_forward_martingale_research import (
    WalkForwardWindow,
    make_walk_forward_windows,
    split_frame_from_index,
    stitch_oos_equity,
)
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_revalidate_top3_distinct_policies.yaml"
SEARCH_SPLIT = "validation"
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
FINGERPRINT_COLUMNS = [
    "start_timestamp",
    "exit_timestamp",
    "side",
    "exit_reason",
    "number_of_levels_filled",
    "realized_pnl",
    "fees_paid",
    "slippage_paid",
    "name",
]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    source_iteration_dir: Path
    source_variant: str
    policy_kind: str


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    engine: str
    entry_execution_mode: str
    signal_lag_bars: int = 0
    mask_lag_bars: int = 0
    fee_rate: float | None = None
    slippage_bps: float | None = None


@dataclass(frozen=True)
class EvaluationContext:
    entry_mask: MaskLike
    add_block_mask: MaskLike
    emergency_exit_mask: MaskLike


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_path(value)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_utc_index(frame: pd.DataFrame, label: str) -> None:
    if frame.index.tz is None:
        raise ValueError(f"{label} index must be timezone-aware UTC")
    if str(frame.index.tz) not in {"UTC", "datetime64[ns, UTC]"}:
        converted = pd.DatetimeIndex(frame.index).tz_convert("UTC")
        if not converted.equals(frame.index):
            raise ValueError(f"{label} index must be UTC")


def _open_datetime(frame: pd.DataFrame) -> pd.Series:
    if "open_datetime" in frame.columns:
        return pd.to_datetime(frame["open_datetime"], utc=True)
    if "open_time" in frame.columns:
        return pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return pd.Series(pd.DatetimeIndex(frame.index).tz_convert("UTC"), index=frame.index)


def full_signal_hours(market_5m: pd.DataFrame, signal_1h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market_open = _open_datetime(market_5m)
    hour_open = market_open.dt.floor("h")
    counts = hour_open.value_counts().sort_index()
    full_hours = counts[counts.eq(12)].index
    signal_open = _open_datetime(signal_1h).dt.floor("h")
    full_signal = signal_1h.loc[signal_open.isin(full_hours)].copy()
    filtered_market = market_5m.loc[hour_open.isin(full_hours)].copy()
    excluded_signal = int(len(signal_1h) - len(full_signal))
    excluded_5m = int(len(market_5m) - len(filtered_market))
    details = {
        "full_signal_hours": int(len(full_hours)),
        "excluded_incomplete_1h_bars": excluded_signal,
        "excluded_5m_bars_in_incomplete_hours": excluded_5m,
        "last_full_hour_open_utc": str(full_hours.max()) if len(full_hours) else None,
    }
    if full_signal.empty or filtered_market.empty:
        raise ValueError("full-hour filtering removed all data")
    return filtered_market, full_signal, details


def audit_data_files(
    market_path: Path,
    signal_path: Path,
    min_coverage_rate: float,
    bar_minutes: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market = load_processed(market_path)
    signal = load_processed(signal_path)
    require_utc_index(market, "5m market")
    require_utc_index(signal, "1h signal")
    validate_ohlcv(market, "5m")
    validate_ohlcv(signal, "1h")
    if not market.index.is_monotonic_increasing or not signal.index.is_monotonic_increasing:
        raise ValueError("market and signal indexes must be monotonic")
    if market.index.has_duplicates or signal.index.has_duplicates:
        raise ValueError("market and signal indexes must not contain duplicates")

    expected_index = pd.date_range(market.index.min(), market.index.max(), freq=f"{bar_minutes}min", tz="UTC")
    observed_unique = pd.DatetimeIndex(market.index.unique())
    missing = expected_index.difference(observed_unique)
    coverage = 1.0 - (len(missing) / len(expected_index)) if len(expected_index) else 0.0
    deltas = market.index.to_series().diff().dropna()
    gap_count = int(deltas.gt(pd.Timedelta(minutes=bar_minutes)).sum())
    duplicate_count = int(market.index.duplicated().sum())
    if coverage < float(min_coverage_rate):
        raise ValueError(f"5m coverage {coverage:.4f} below minimum {min_coverage_rate:.4f}")

    filtered_market, filtered_signal, full_hour_details = full_signal_hours(market, signal)
    audit = {
        "market_5m_path": str(market_path),
        "signal_1h_path": str(signal_path),
        "market_5m_sha256": sha256_file(market_path),
        "signal_1h_sha256": sha256_file(signal_path),
        "market_5m_size_bytes": int(market_path.stat().st_size),
        "signal_1h_size_bytes": int(signal_path.stat().st_size),
        "market_5m_mtime_utc": datetime.fromtimestamp(market_path.stat().st_mtime, timezone.utc).isoformat(),
        "signal_1h_mtime_utc": datetime.fromtimestamp(signal_path.stat().st_mtime, timezone.utc).isoformat(),
        "market_5m_start_utc": str(market.index.min()),
        "market_5m_end_utc": str(market.index.max()),
        "signal_1h_start_utc": str(signal.index.min()),
        "signal_1h_end_utc": str(signal.index.max()),
        "expected_5m_bars": int(len(expected_index)),
        "observed_5m_bars": int(len(market)),
        "coverage_rate": float(coverage),
        "gap_count": gap_count,
        "duplicate_5m_count": duplicate_count,
        "duplicate_1h_count": int(signal.index.duplicated().sum()),
        "filtered_5m_bars": int(len(filtered_market)),
        "filtered_1h_bars": int(len(filtered_signal)),
        **full_hour_details,
    }
    return filtered_market, filtered_signal, audit


def split_pre_holdout(
    market: pd.DataFrame,
    train_days: float,
    test_days: float,
    step_days: float,
    embargo_bars: int,
    holdout_days: float,
    max_folds: int | None = None,
) -> tuple[list[WalkForwardWindow], pd.Index, pd.Timestamp]:
    holdout_start_raw = market.index.max() - pd.Timedelta(days=float(holdout_days))
    holdout_start_pos = int(market.index.searchsorted(holdout_start_raw, side="left"))
    if holdout_start_pos <= 0:
        raise ValueError("holdout split leaves no pre-holdout data")
    holdout_start = market.index[holdout_start_pos]
    pre_holdout_index = market.index[market.index < holdout_start]
    holdout_index = market.index[market.index >= holdout_start]
    if holdout_index.empty:
        raise ValueError("holdout index is empty")
    windows = make_walk_forward_windows(
        pre_holdout_index,
        train_days=float(train_days),
        test_days=float(test_days),
        step_days=float(step_days),
        embargo_bars=int(embargo_bars),
        max_folds=max_folds,
    )
    return windows, holdout_index, holdout_start


def load_policy_specs(config: dict[str, Any]) -> list[PolicySpec]:
    specs: list[PolicySpec] = []
    for row in list(config["policies"]["primary"]) + list(config["policies"].get("fallback_order", [])):
        specs.append(
            PolicySpec(
                name=str(row["name"]),
                source_iteration_dir=resolve_path(row["source_iteration_dir"]),
                source_variant=str(row["source_variant"]),
                policy_kind=str(row["policy_kind"]),
            )
        )
    return specs


def load_scenarios(config: dict[str, Any]) -> list[ScenarioSpec]:
    scenarios: list[ScenarioSpec] = []
    for row in config["scenarios"]:
        scenario = ScenarioSpec(
            name=str(row["name"]),
            engine=str(row["engine"]),
            entry_execution_mode=str(row.get("entry_execution_mode", "current_close")),
            signal_lag_bars=int(row.get("signal_lag_bars", 0)),
            mask_lag_bars=int(row.get("mask_lag_bars", 0)),
            fee_rate=None if row.get("fee_rate") is None else float(row["fee_rate"]),
            slippage_bps=None if row.get("slippage_bps") is None else float(row["slippage_bps"]),
        )
        if scenario.engine not in {"legacy", "audit"}:
            raise ValueError(f"Unsupported scenario engine: {scenario.engine}")
        scenarios.append(scenario)
    return scenarios


def selected_candidates_path(policy: PolicySpec) -> Path:
    specific = policy.source_iteration_dir / f"walk_forward_selected_candidates_{policy.source_variant}.csv"
    if specific.exists():
        return specific
    generic = policy.source_iteration_dir / "walk_forward_selected_candidates.csv"
    if generic.exists():
        return generic
    raise FileNotFoundError(f"No selected candidates found for {policy.name} in {policy.source_iteration_dir}")


def load_selected_candidates(policy: PolicySpec) -> pd.DataFrame:
    path = selected_candidates_path(policy)
    frame = pd.read_csv(path)
    if "variant" in frame.columns:
        frame = frame[frame["variant"].astype(str).eq(policy.source_variant)].copy()
    missing = [column for column in CANDIDATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing candidate columns: {missing}")
    if frame.empty:
        raise ValueError(f"No selected rows for {policy.name}")
    if "fold_id" in frame.columns:
        frame["fold_id"] = frame["fold_id"].astype(int)
    else:
        frame["fold_id"] = np.arange(1, len(frame) + 1)
    frame["source_path"] = str(path)
    return frame.sort_values("fold_id").reset_index(drop=True)


def unique_candidate_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    threshold_columns = ["range_break_threshold", "range_break_emergency_threshold"]
    rows = frame.copy()
    for column in threshold_columns:
        if column not in rows.columns:
            rows[column] = np.nan
    dedupe_cols = CANDIDATE_COLUMNS + threshold_columns
    rows = rows.drop_duplicates(dedupe_cols, keep="first")
    return rows.to_dict("records")


def candidate_for_scenario(row: dict[str, Any], scenario: ScenarioSpec) -> MonthlyMartingaleCandidate:
    candidate = candidate_from_row(row)
    updates: dict[str, Any] = {}
    if scenario.fee_rate is not None:
        updates["fee_rate"] = float(scenario.fee_rate)
    if scenario.slippage_bps is not None:
        updates["slippage_bps"] = float(scenario.slippage_bps)
    return replace(candidate, **updates) if updates else candidate


def threshold_from_row(row: dict[str, Any]) -> float | None:
    value = row.get("range_break_threshold")
    if value is None or pd.isna(value):
        return None
    return float(value)


def emergency_threshold_from_row(row: dict[str, Any], config: dict[str, Any]) -> float:
    value = row.get("range_break_emergency_threshold")
    if value is None or pd.isna(value):
        return float(config.get("range_break_classifier", {}).get("emergency_exit_threshold", 0.90))
    return float(value)


def _lag_mask(mask: MaskLike, bars: int) -> MaskLike:
    if mask is None or bars <= 0:
        return mask
    if isinstance(mask, dict):
        return {side: series.astype(bool).shift(bars).fillna(False).astype(bool) for side, series in mask.items()}
    return mask.astype(bool).shift(bars).fillna(False).astype(bool)


def _mask_value(mask: MaskLike, side: str, timestamp: pd.Timestamp) -> bool:
    if mask is None:
        return False
    if isinstance(mask, dict):
        series = mask.get(side)
        if series is None or timestamp not in series.index:
            return False
        return bool(series.loc[timestamp])
    if timestamp not in mask.index:
        return False
    return bool(mask.loc[timestamp])


def prefilter_side_signal_for_audit(
    side_signal: pd.Series,
    split_index: pd.Index,
    entry_mask: MaskLike,
    scenario: ScenarioSpec,
) -> pd.Series:
    signal = side_signal.reindex(split_index).shift(int(scenario.signal_lag_bars))
    if entry_mask is None:
        return signal
    offset = 0 if scenario.entry_execution_mode == "current_close" else 1
    values = signal.astype("object").copy()
    for pos, timestamp in enumerate(split_index):
        side = values.iloc[pos]
        if side not in {"long", "short"}:
            continue
        entry_pos = pos + offset
        if entry_pos >= len(split_index):
            values.iloc[pos] = pd.NA
            continue
        entry_ts = split_index[entry_pos]
        if _mask_value(entry_mask, str(side), entry_ts):
            values.iloc[pos] = pd.NA
    return values


def policy_masks(
    policy: PolicySpec,
    row: dict[str, Any],
    fundamental_entry: pd.Series,
    event_entry: pd.Series,
    trend_escape: pd.Series,
    classifier_scores: dict[str, pd.Series] | None,
    config: dict[str, Any],
) -> EvaluationContext:
    if policy.policy_kind == "fundamental_trend_escape_entry_only":
        return EvaluationContext(fundamental_entry, None, None)
    if policy.policy_kind == "fundamental_event_entry_only":
        return EvaluationContext(event_entry, None, None)
    if policy.policy_kind == "trend_escape_entry_only":
        return EvaluationContext(trend_escape, None, None)
    if policy.policy_kind in {"classifier_entry_only", "classifier_entry_add_block"}:
        if classifier_scores is None:
            raise ValueError(f"{policy.name} requires classifier scores")
        threshold = threshold_from_row(row)
        if threshold is None:
            raise ValueError(f"{policy.name} selected row has no range_break_threshold")
        return EvaluationContext(
            *masks_for_variant(
                policy.policy_kind,
                threshold,
                classifier_scores,
                fundamental_entry,
                emergency_threshold_from_row(row, config),
            )
        )
    raise ValueError(f"Unsupported policy kind: {policy.policy_kind}")


def summarize_result(
    equity: pd.Series,
    trades: pd.DataFrame,
    candidate: MonthlyMartingaleCandidate,
    split: str,
) -> dict[str, Any]:
    metrics = summarize_exact(equity, trades, candidate, split)
    metrics["equity_ruined"] = bool((equity <= 0).any())
    return metrics


def run_candidate_on_split(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    policy: PolicySpec,
    row: dict[str, Any],
    split_index: pd.Index,
    split_name: str,
    scenario: ScenarioSpec,
    context: EvaluationContext,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    split_frame = split_frame_from_index(market, split_index)
    candidate = candidate_for_scenario(row, scenario)
    risk = risk_for_candidate(base_risk, candidate)
    side_signal = build_side_signal(market, signal_frame, candidate)
    entry_mask = _lag_mask(context.entry_mask, scenario.mask_lag_bars)
    add_block_mask = _lag_mask(context.add_block_mask, scenario.mask_lag_bars)
    emergency_exit_mask = _lag_mask(context.emergency_exit_mask, scenario.mask_lag_bars)

    if scenario.engine == "legacy":
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
        trades = result.trades
        equity = result.equity_curve
    else:
        audit_signal = prefilter_side_signal_for_audit(side_signal, split_frame.index, entry_mask, scenario)
        audit_scenario = AuditScenario(
            name=scenario.name,
            variant=policy.source_variant,
            selection_policy="revalidation",
            entry_execution_mode=scenario.entry_execution_mode,
            signal_lag_bars=0,
            mask_lag_bars=0,
        )
        result = run_signal_grid_backtest_audit(
            split_frame,
            risk,
            audit_signal,
            candidate,
            audit_scenario,
            entry_mask=None,
        )
        trades = result.trades
        equity = result.equity_curve
    metrics = summarize_result(equity, trades, candidate, split_name)
    metrics.update(
        {
            "policy": policy.name,
            "source_variant": policy.source_variant,
            "policy_kind": policy.policy_kind,
            "scenario": scenario.name,
            "engine": scenario.engine,
            "entry_execution_mode": scenario.entry_execution_mode,
            "signal_lag_bars": int(scenario.signal_lag_bars),
            "mask_lag_bars": int(scenario.mask_lag_bars),
            "threshold": threshold_from_row(row),
            "range_break_emergency_threshold": emergency_threshold_from_row(row, {"range_break_classifier": {}}),
        }
    )
    return metrics, trades, equity


def select_row_on_train(
    universe: list[dict[str, Any]],
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    policy: PolicySpec,
    train_index: pd.Index,
    scenario: ScenarioSpec,
    context_by_row: dict[int, EvaluationContext],
    evaluator: Any | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stride = int((config or {}).get("search", {}).get("search_entry_stride_bars", 24))
    max_positions = int((config or {}).get("search", {}).get("max_sample_positions_per_candidate", 30))
    for idx, row in enumerate(universe):
        candidate = candidate_for_scenario(row, scenario)
        risk = risk_for_candidate(base_risk, candidate)
        side_signal = build_side_signal(market, signal_frame, candidate)
        entry_mask = _lag_mask(context_by_row[idx].entry_mask, scenario.mask_lag_bars)
        if scenario.engine == "legacy":
            positions = sample_positions(
                market,
                train_index,
                side_signal,
                risk,
                candidate.entry_cooldown_hours,
                stride,
                max_positions,
                entry_blackout_series=entry_mask,
            )
            if not positions:
                continue
            sample = simulate_candidate_sample(market, positions, risk, side_signal, candidate, SEARCH_SPLIT)
        else:
            audit_scenario = AuditScenario(
                name=scenario.name,
                variant=policy.source_variant,
                selection_policy="sampled_train_reselection",
                entry_execution_mode=scenario.entry_execution_mode,
                signal_lag_bars=0,
                mask_lag_bars=0,
            )
            shifted_signal = prefilter_side_signal_for_audit(side_signal, market.index, entry_mask, scenario)
            positions = sample_positions_for_scenario(
                market,
                train_index,
                shifted_signal,
                risk,
                candidate.entry_cooldown_hours,
                stride,
                max_positions,
                entry_mask=None,
                scenario=audit_scenario,
            )
            if not positions:
                continue
            sample = simulate_candidate_sample_for_scenario(
                market,
                positions,
                risk,
                shifted_signal,
                candidate,
                SEARCH_SPLIT,
                audit_scenario,
            )
        metrics = summarize_simulations(sample, baseline_grids=len(positions))
        metrics.update(
            {
                "split": SEARCH_SPLIT,
                "monthly_return": float(metrics.get("realized_pnl", 0.0)),
                "sample_positions": int(len(positions)),
                "selection_method": "sampled_train_only",
            }
        )
        metrics["candidate_row_index"] = idx
        rows.append(metrics)
    if not rows:
        raise ValueError(f"No train-only sampled selection rows generated for {policy.name}")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["expectancy", "realized_pnl", "profit_factor", "number_of_forced_exits"],
        ascending=[False, False, False, True],
    )
    selected = frame.iloc[0].to_dict()
    selected["selected_from_validation_only"] = True
    selected["selection_uses_drawdown"] = False
    row_index = int(selected["candidate_row_index"])
    return universe[row_index], selected


def locked_row_for_fold(selected: pd.DataFrame, fold_id: int) -> dict[str, Any]:
    rows = selected[selected["fold_id"].astype(int).eq(int(fold_id))]
    if rows.empty:
        rows = selected.tail(1)
    return rows.iloc[0].to_dict()


def locked_row_for_holdout(policy: PolicySpec, selected: pd.DataFrame, holdout_start: pd.Timestamp) -> dict[str, Any]:
    summary_path = policy.source_iteration_dir / f"walk_forward_fold_summary_{policy.source_variant}.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "fold_id" in summary.columns and "test_end" in summary.columns:
            summary["test_end"] = pd.to_datetime(summary["test_end"], utc=True)
            eligible = summary[summary["test_end"].lt(holdout_start)]
            if not eligible.empty:
                fold_id = int(eligible.sort_values("test_end").iloc[-1]["fold_id"])
                return locked_row_for_fold(selected, fold_id)
    return selected.sort_values("fold_id").iloc[-1].to_dict()


def make_classifier_scores(
    market: pd.DataFrame,
    model_base: pd.DataFrame,
    labels: pd.DataFrame,
    window: WalkForwardWindow,
    config: dict[str, Any],
) -> dict[str, pd.Series]:
    split = split_internal_train(window, config)
    model, _metrics = train_range_break_model(model_base, labels, split, config)
    score_index = pd.Index(window.train.union(window.test)).sort_values()
    scoped_base = model_base.reindex(score_index)
    return predict_scores_by_side(model, scoped_base)


def window_cache_key(window: WalkForwardWindow) -> tuple[str, str, str, str]:
    return (
        str(window.train.min()),
        str(window.train.max()),
        str(window.test.min()),
        str(window.test.max()),
    )


def candidate_contexts(
    policy: PolicySpec,
    universe: list[dict[str, Any]],
    fundamental_entry: pd.Series,
    event_entry: pd.Series,
    trend_escape: pd.Series,
    classifier_scores: dict[str, pd.Series] | None,
    config: dict[str, Any],
) -> dict[int, EvaluationContext]:
    return {
        idx: policy_masks(policy, row, fundamental_entry, event_entry, trend_escape, classifier_scores, config)
        for idx, row in enumerate(universe)
    }


def fold_metrics_row(
    mode: str,
    fold_id: int | str,
    train_index: pd.Index,
    test_index: pd.Index,
    selected_row: dict[str, Any],
    metrics: dict[str, Any],
    trades: pd.DataFrame,
    account_equity_usdt: float,
) -> dict[str, Any]:
    max_exposure = float(trades["max_exposure_pct"].max()) if not trades.empty and "max_exposure_pct" in trades else 0.0
    avg_exposure = float(trades["max_exposure_pct"].mean()) if not trades.empty and "max_exposure_pct" in trades else 0.0
    return {
        "selection_mode": mode,
        "policy": metrics["policy"],
        "scenario": metrics["scenario"],
        "fold_id": fold_id,
        "train_start": train_index.min() if len(train_index) else pd.NaT,
        "train_end": train_index.max() if len(train_index) else pd.NaT,
        "test_start": test_index.min(),
        "test_end": test_index.max(),
        "selected_name": str(selected_row["name"]),
        "total_return": float(metrics.get("total_return", 0.0)),
        "monthly_return": float(metrics.get("monthly_return", 0.0)),
        "annualized_return": float(metrics.get("annualized_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "number_of_grids": int(metrics.get("number_of_grids", 0)),
        "fees_paid": float(metrics.get("fees_paid", 0.0)),
        "slippage_paid": float(metrics.get("slippage_paid", 0.0)),
        "net_pnl": float(metrics.get("realized_pnl", 0.0)),
        "fees_usdt": float(metrics.get("fees_paid", 0.0)) * account_equity_usdt,
        "slippage_usdt": float(metrics.get("slippage_paid", 0.0)) * account_equity_usdt,
        "net_pnl_usdt": float(metrics.get("realized_pnl", 0.0)) * account_equity_usdt,
        "equity_ruined": bool(metrics.get("equity_ruined", False)),
        "max_trade_exposure_pct": max_exposure,
        "mean_trade_exposure_pct": avg_exposure,
        "max_notional_usdt": max_exposure * account_equity_usdt,
        "effective_leverage_max": max_exposure,
        "threshold": metrics.get("threshold"),
    }


def summary_for_scope(
    fold_metrics: pd.DataFrame,
    equity: pd.Series,
    trades: pd.DataFrame,
    scope: str,
    account_equity_usdt: float,
) -> dict[str, Any]:
    metrics = calculate_metrics(equity, trades if not trades.empty else None)
    if not trades.empty:
        metrics.update(summarize_simulations(trades, baseline_grids=len(trades)))
    else:
        metrics.update(
            {
                "profit_factor": 0.0,
                "number_of_grids": 0,
                "fees_paid": 0.0,
                "slippage_paid": 0.0,
                "realized_pnl": 0.0,
            }
        )
    monthly = monthly_return_from_equity(equity)
    fold_returns = fold_metrics["monthly_return"].astype(float) if not fold_metrics.empty else pd.Series(dtype=float)
    max_exposure = float(trades["max_exposure_pct"].max()) if not trades.empty and "max_exposure_pct" in trades else 0.0
    worst = fold_metrics.sort_values("monthly_return").iloc[0] if not fold_metrics.empty else None
    best = fold_metrics.sort_values("monthly_return", ascending=False).iloc[0] if not fold_metrics.empty else None
    return {
        "scope": scope,
        "fold_count": int(len(fold_metrics)),
        "total_return": float(metrics.get("total_return", 0.0)),
        "monthly_return": float(monthly),
        "annualized_return": float(metrics.get("annualized_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "positive_fold_rate": float((fold_metrics["total_return"].astype(float) > 0).mean()) if not fold_metrics.empty else 0.0,
        "fold_rate_above_3pct": float(fold_returns.ge(0.03).mean()) if not fold_returns.empty else 0.0,
        "fold_rate_above_8pct": float(fold_returns.ge(0.08).mean()) if not fold_returns.empty else 0.0,
        "fold_rate_above_12pct": float(fold_returns.ge(0.12).mean()) if not fold_returns.empty else 0.0,
        "fold_rate_above_20pct": float(fold_returns.ge(0.20).mean()) if not fold_returns.empty else 0.0,
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "number_of_grids": int(metrics.get("number_of_grids", 0)),
        "fees_paid": float(metrics.get("fees_paid", 0.0)),
        "slippage_paid": float(metrics.get("slippage_paid", 0.0)),
        "net_pnl": float(metrics.get("realized_pnl", 0.0)),
        "fees_usdt": float(metrics.get("fees_paid", 0.0)) * account_equity_usdt,
        "slippage_usdt": float(metrics.get("slippage_paid", 0.0)) * account_equity_usdt,
        "net_pnl_usdt": float(metrics.get("realized_pnl", 0.0)) * account_equity_usdt,
        "worst_fold": None if worst is None else str(worst["fold_id"]),
        "worst_fold_monthly": np.nan if worst is None else float(worst["monthly_return"]),
        "best_fold": None if best is None else str(best["fold_id"]),
        "best_fold_monthly": np.nan if best is None else float(best["monthly_return"]),
        "equity_ruined": bool(
            (equity <= 0).any()
            or float(metrics.get("max_drawdown", 0.0)) <= -1.0
            or fold_metrics.get("equity_ruined", pd.Series(dtype=bool)).astype(bool).any()
        ),
        "max_trade_exposure_pct": max_exposure,
        "max_notional_usdt": max_exposure * account_equity_usdt,
        "effective_leverage_max": max_exposure,
    }


def normalize_trades_for_fingerprint(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "empty"
    frame = trades.copy()
    for column in FINGERPRINT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[FINGERPRINT_COLUMNS].copy()
    for column in ["realized_pnl", "fees_paid", "slippage_paid"]:
        frame[column] = frame[column].astype(float).round(10)
    frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True).astype(str)
    frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True).astype(str)
    encoded = frame.sort_values(FINGERPRINT_COLUMNS[:2]).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def equity_fingerprint(equity: pd.Series) -> str:
    frame = pd.DataFrame({"timestamp": equity.index.astype(str), "equity": equity.astype(float).round(10).to_numpy()})
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def select_distinct_policies(candidates: list[dict[str, Any]], desired_count: int = 3) -> tuple[list[str], pd.DataFrame]:
    selected: list[str] = []
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (str(candidate["trades_fingerprint"]), str(candidate["equity_fingerprint"]))
        duplicate_of = None
        if key in seen:
            duplicate_of = next(row["policy"] for row in rows if row["fingerprint_key"] == str(key))
        is_selected = duplicate_of is None and len(selected) < desired_count
        if is_selected:
            selected.append(str(candidate["policy"]))
            seen.add(key)
        rows.append(
            {
                "policy": candidate["policy"],
                "source_variant": candidate["source_variant"],
                "trades_fingerprint": candidate["trades_fingerprint"],
                "equity_fingerprint": candidate["equity_fingerprint"],
                "fingerprint_key": str(key),
                "duplicate_of": duplicate_of,
                "selected": bool(is_selected),
                "legacy_monthly_return": float(candidate["legacy_monthly_return"]),
            }
        )
    return selected, pd.DataFrame(rows)


def rank_outputs(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    legacy = combined[combined["scenario"].eq("legacy")].sort_values(
        ["monthly_return", "total_return", "positive_fold_rate"], ascending=[False, False, False]
    )
    realistic = combined[combined["scenario"].eq("realistic_timing_costs")].copy()
    realistic["costs_paid"] = realistic["fees_paid"].astype(float) + realistic["slippage_paid"].astype(float)
    realistic = realistic.sort_values(
        ["monthly_return", "net_pnl", "profit_factor", "costs_paid"],
        ascending=[False, False, False, True],
    )
    robust = combined[(~combined["equity_ruined"].astype(bool)) & (combined["max_drawdown"].astype(float) > -0.35)].copy()
    robust = robust.sort_values(
        ["max_drawdown", "positive_fold_rate", "fold_rate_above_8pct", "worst_fold_monthly"],
        ascending=[False, False, False, False],
    )
    return legacy, realistic, robust


def write_figures(output_dir: Path, summary: pd.DataFrame, equity_frames: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    if not combined.empty:
        labels = combined["selection_mode"] + "\n" + combined["policy"] + "\n" + combined["scenario"]
        plt.figure(figsize=(max(10, len(combined) * 0.5), 6))
        plt.bar(np.arange(len(combined)), combined["monthly_return"].astype(float) * 100)
        plt.xticks(np.arange(len(combined)), labels, rotation=90, fontsize=7)
        plt.ylabel("Monthly return (%)")
        plt.title("Combined OOS Monthly Return by Policy / Scenario")
        plt.tight_layout()
        plt.savefig(figures / "combined_monthly_return_by_policy_scenario.png", dpi=150)
        plt.close()

    plt.figure(figsize=(12, 6))
    plotted = 0
    for key, frame in equity_frames.items():
        if "realistic_timing_costs" not in key or frame.empty:
            continue
        plt.plot(pd.to_datetime(frame["timestamp"], utc=True), frame["equity"], label=key)
        plotted += 1
    if plotted:
        plt.title("Realistic Timing + Costs Combined OOS Equity")
        plt.ylabel("Equity")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(figures / "realistic_combined_oos_equity.png", dpi=150)
    plt.close()


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    summary = pd.DataFrame(payload["summary"])
    legacy = pd.DataFrame(payload["ranking_best_legacy_return"]).head(5)
    realistic = pd.DataFrame(payload["ranking_best_realistic_net"]).head(5)
    robust = pd.DataFrame(payload["ranking_best_robustness"]).head(5)
    lines = [
        "# Top 3 Distinct Policy Revalidation",
        "",
        "## Data Audit",
        markdown_table(pd.DataFrame(payload["data_audit"])),
        "",
        "## Distinct Policies",
        markdown_table(pd.DataFrame(payload["policy_distinctness"])),
        "",
        "## Combined OOS Summary",
        markdown_table(
            summary[summary["scope"].eq("combined_oos")][
                [
                    "selection_mode",
                    "policy",
                    "scenario",
                    "monthly_return",
                    "max_drawdown",
                    "positive_fold_rate",
                    "fold_rate_above_20pct",
                    "profit_factor",
                    "number_of_grids",
                    "fees_paid",
                    "slippage_paid",
                    "net_pnl",
                    "effective_leverage_max",
                ]
            ]
        ),
        "",
        "## Rankings",
        "### Best Legacy Return",
        markdown_table(legacy[["selection_mode", "policy", "scenario", "monthly_return", "total_return", "positive_fold_rate"]])
        if not legacy.empty
        else "No legacy ranking.",
        "",
        "### Best Realistic Net",
        markdown_table(realistic[["selection_mode", "policy", "scenario", "monthly_return", "net_pnl", "profit_factor"]])
        if not realistic.empty
        else "No realistic ranking.",
        "",
        "### Best Robustness",
        markdown_table(robust[["selection_mode", "policy", "scenario", "monthly_return", "max_drawdown", "positive_fold_rate"]])
        if not robust.empty
        else "No robust candidate passed the drawdown/ruin filter.",
        "",
        "## Notes",
        "- No new broad parameter optimization is performed; train-only reselection uses only the historical candidate universe for each policy.",
        "- The final 90 days are evaluated as holdout after fold selection and are not used for reselection.",
        "- Classifier policies retrain their model on each fold train window because the historical fitted estimators are not serialized.",
    ]
    (output_dir / "revalidation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_iteration(
    config_path: str = DEFAULT_CONFIG,
    smoke: bool = False,
    max_folds: int | None = None,
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if smoke and max_folds is None:
        max_folds = 2
    if smoke:
        # Smoke remains a real end-to-end integration run, but uses short windows so it can finish locally.
        config["walk_forward"]["train_days"] = min(float(config["walk_forward"]["train_days"]), 30.0)
        config["walk_forward"]["test_days"] = min(float(config["walk_forward"]["test_days"]), 5.0)
        config["walk_forward"]["step_days"] = min(float(config["walk_forward"]["step_days"]), 5.0)
        config["walk_forward"]["holdout_days"] = min(float(config["walk_forward"]["holdout_days"]), 15.0)
        config["range_break_classifier"]["internal_split"]["model_train_days"] = 20
        config["range_break_classifier"]["internal_split"]["selection_days"] = 10
    output_root = resolve_path(config["iteration"]["output_root"])
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"revalidation_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "oos_equity").mkdir(parents=True, exist_ok=True)

    market_path = resolve_path(config["data"]["market_5m_path"])
    signal_path = resolve_path(config["data"]["signal_1h_path"])
    market, signal_frame, data_audit = audit_data_files(
        market_path,
        signal_path,
        float(config["data"]["min_coverage_rate"]),
        int(config["data"]["bar_minutes"]),
    )
    data_audit["output_dir"] = str(output_dir)
    pd.DataFrame([data_audit]).to_csv(output_dir / "data_audit.csv", index=False)
    (output_dir / "data_audit.json").write_text(json.dumps(data_audit, indent=2, default=str), encoding="utf-8")

    # prepare_market enriches the already audited OHLCV with ATR/regime features; reindex to the full-hour filtered audit set.
    enriched_market = prepare_market(str(config.get("asset", "btcusdt"))).reindex(market.index).dropna(subset=["open", "high", "low", "close"])
    signal_frame = signal_frame.reindex(signal_frame.index.intersection(signal_path and signal_frame.index))
    wf = config["walk_forward"]
    windows, holdout_index, holdout_start = split_pre_holdout(
        enriched_market,
        float(wf["train_days"]),
        float(wf["test_days"]),
        float(wf["step_days"]),
        int(wf["embargo_bars"]),
        float(wf["holdout_days"]),
        max_folds=max_folds,
    )
    holdout_train_start = holdout_start - pd.Timedelta(days=float(wf["train_days"]))
    holdout_train_index = enriched_market.index[(enriched_market.index >= holdout_train_start) & (enriched_market.index < holdout_start)]
    if holdout_train_index.empty:
        raise ValueError("holdout train window is empty")

    base_risk = validate_strategy_config(load_strategy_config())
    events, _event_windows, blackout_masks = build_blackout_bundle(enriched_market.index, config)
    trend_components = build_trend_escape_components(enriched_market, config)
    trend_escape = trend_components["trend_escape"].astype(bool)
    realistic_event = blackout_masks["realistic"].reindex(enriched_market.index).fillna(False).astype(bool)
    fundamental_entry = fundamental_trend_mask(trend_escape, blackout_masks).reindex(enriched_market.index).fillna(False).astype(bool)
    features = load_feature_frame(str(config.get("asset", "btcusdt")), enriched_market)
    labels = build_range_break_labels(enriched_market.join(features, how="left", rsuffix="_feature"), config)
    event_features = build_event_features(enriched_market.index, events, realistic_event)
    model_base = build_model_base_frame(enriched_market, features, trend_components, event_features)

    all_policy_specs = load_policy_specs(config)
    scenarios = load_scenarios(config)
    account_equity = float(config["reporting"]["account_equity_usdt"])
    selected_frames = {policy.name: load_selected_candidates(policy) for policy in all_policy_specs}
    classifier_score_cache: dict[tuple[str, str, str, str], dict[str, pd.Series]] = {}
    evaluation_cache: dict[tuple[Any, ...], tuple[dict[str, Any], pd.DataFrame, pd.Series]] = {}

    def scores_for_window(policy: PolicySpec, window: WalkForwardWindow) -> dict[str, pd.Series] | None:
        if not policy.policy_kind.startswith("classifier"):
            return None
        key = window_cache_key(window)
        if key not in classifier_score_cache:
            classifier_score_cache[key] = make_classifier_scores(enriched_market, model_base, labels, window, config)
        return classifier_score_cache[key]

    def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for column in CANDIDATE_COLUMNS + ["range_break_threshold", "range_break_emergency_threshold"]:
            value = row.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                values.append(None)
            elif isinstance(value, (np.floating, float)):
                values.append(round(float(value), 12))
            else:
                values.append(value)
        return tuple(values)

    def cached_eval(
        policy: PolicySpec,
        row: dict[str, Any],
        split_index: pd.Index,
        split_name: str,
        scenario: ScenarioSpec,
        context: EvaluationContext,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
        key = (
            policy.name,
            policy.policy_kind,
            candidate_key(row),
            str(split_index.min()),
            str(split_index.max()),
            len(split_index),
            split_name,
            scenario.name,
            scenario.engine,
            scenario.entry_execution_mode,
            scenario.signal_lag_bars,
            scenario.mask_lag_bars,
            scenario.fee_rate,
            scenario.slippage_bps,
        )
        if key not in evaluation_cache:
            evaluation_cache[key] = run_candidate_on_split(
                enriched_market,
                signal_frame,
                base_risk,
                policy,
                row,
                split_index,
                split_name,
                scenario,
                context,
            )
        metrics, trades, equity = evaluation_cache[key]
        return dict(metrics), trades.copy(), equity.copy()

    # Select the top three practical distinct policies from legacy locked replay on the configured order.
    distinct_candidates: list[dict[str, Any]] = []
    selected_policy_names: list[str] = []
    seen_fingerprints: set[tuple[str, str]] = set()
    for policy in all_policy_specs:
        selected = selected_frames[policy.name]
        fold_trades: list[pd.DataFrame] = []
        fold_equities: list[tuple[int, pd.Series]] = []
        for window in windows:
            row = locked_row_for_fold(selected, window.fold_id)
            universe = [row]
            scores = scores_for_window(policy, window)
            contexts = candidate_contexts(policy, universe, fundamental_entry, realistic_event, trend_escape, scores, config)
            metrics, trades, equity = cached_eval(
                policy,
                row,
                window.test,
                "test",
                scenarios[0],
                contexts[0],
            )
            fold_trades.append(trades)
            fold_equities.append((window.fold_id, equity))
        combined_trades = pd.concat(fold_trades, ignore_index=True) if fold_trades else pd.DataFrame()
        combined_equity = stitch_oos_equity(fold_equities)["equity"]
        trades_hash = normalize_trades_for_fingerprint(combined_trades)
        equity_hash = equity_fingerprint(combined_equity)
        distinct_candidates.append(
            {
                "policy": policy.name,
                "source_variant": policy.source_variant,
                "trades_fingerprint": trades_hash,
                "equity_fingerprint": equity_hash,
                "legacy_monthly_return": monthly_return_from_equity(combined_equity),
            }
        )
        fingerprint = (trades_hash, equity_hash)
        if fingerprint not in seen_fingerprints and len(selected_policy_names) < 3:
            selected_policy_names.append(policy.name)
            seen_fingerprints.add(fingerprint)
        if len(selected_policy_names) >= 3:
            break
    selected_policy_names, distinctness = select_distinct_policies(distinct_candidates, desired_count=3)
    if len(selected_policy_names) < 3:
        raise ValueError("fewer than three distinct policies were available")
    selected_policies = [policy for policy in all_policy_specs if policy.name in set(selected_policy_names)]
    distinctness.to_csv(output_dir / "policy_distinctness.csv", index=False)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    equity_outputs: dict[str, pd.DataFrame] = {}
    selected_rows_output: list[dict[str, Any]] = []

    windows_plus_holdout = windows + [WalkForwardWindow(fold_id=999999, train=holdout_train_index, test=holdout_index)]
    for policy in selected_policies:
        selected = selected_frames[policy.name]
        universe = unique_candidate_rows(selected)
        if smoke:
            universe = universe[: min(3, len(universe))]
        for scenario in scenarios:
            for mode in config["selection_modes"]:
                wf_fold_rows: list[dict[str, Any]] = []
                wf_trades: list[pd.DataFrame] = []
                wf_equities: list[tuple[int, pd.Series]] = []
                combined_trades: list[pd.DataFrame] = []
                combined_equities: list[tuple[int, pd.Series]] = []
                for window in windows_plus_holdout:
                    is_holdout = window.fold_id == 999999
                    fold_label: int | str = "holdout_90d" if is_holdout else int(window.fold_id)
                    scores = scores_for_window(policy, window)
                    contexts = candidate_contexts(policy, universe, fundamental_entry, realistic_event, trend_escape, scores, config)
                    if mode == "locked_historical_replay":
                        selected_row = (
                            locked_row_for_holdout(policy, selected, holdout_start)
                            if is_holdout
                            else locked_row_for_fold(selected, int(window.fold_id))
                        )
                        # map locked row to its own context even if it is not position 0 in the unique universe
                        eval_context = policy_masks(policy, selected_row, fundamental_entry, realistic_event, trend_escape, scores, config)
                        selection_metrics = {}
                    elif mode == "train_only_reselection":
                        selected_row, selection_metrics = select_row_on_train(
                            universe,
                            enriched_market,
                            signal_frame,
                            base_risk,
                            policy,
                            window.train,
                            scenario,
                            contexts,
                            evaluator=cached_eval,
                            config=config,
                        )
                        row_idx = int(selection_metrics["candidate_row_index"])
                        eval_context = contexts[row_idx]
                    else:
                        raise ValueError(f"Unsupported selection mode: {mode}")
                    metrics, trades, equity = cached_eval(
                        policy,
                        selected_row,
                        window.test,
                        "holdout" if is_holdout else "test",
                        scenario,
                        eval_context,
                    )
                    row = fold_metrics_row(
                        mode,
                        fold_label,
                        window.train,
                        window.test,
                        selected_row,
                        metrics,
                        trades,
                        account_equity,
                    )
                    row["selection_monthly_return"] = selection_metrics.get("monthly_return", np.nan)
                    row["is_holdout"] = bool(is_holdout)
                    fold_rows.append(row)
                    selected_rows_output.append(
                        {
                            "selection_mode": mode,
                            "policy": policy.name,
                            "scenario": scenario.name,
                            "fold_id": fold_label,
                            "selected_name": selected_row["name"],
                            "selected_from": "holdout_prior_180d" if is_holdout and mode == "train_only_reselection" else mode,
                            "threshold": threshold_from_row(selected_row),
                        }
                    )
                    trades = trades.copy()
                    trades.insert(0, "fold_id", fold_label)
                    trades.insert(0, "scenario", scenario.name)
                    trades.insert(0, "policy", policy.name)
                    trades.insert(0, "selection_mode", mode)
                    combined_trades.append(trades)
                    combined_equities.append((999999 if is_holdout else int(window.fold_id), equity))
                    if is_holdout:
                        holdout_rows.append(row)
                    else:
                        wf_fold_rows.append(row)
                        wf_trades.append(trades)
                        wf_equities.append((int(window.fold_id), equity))
                wf_frame = pd.DataFrame(wf_fold_rows)
                holdout_frame = pd.DataFrame([row for row in holdout_rows if row["selection_mode"] == mode and row["policy"] == policy.name and row["scenario"] == scenario.name])
                all_frame = pd.DataFrame([row for row in fold_rows if row["selection_mode"] == mode and row["policy"] == policy.name and row["scenario"] == scenario.name])
                wf_equity = stitch_oos_equity(wf_equities)["equity"]
                all_equity = stitch_oos_equity(combined_equities)["equity"]
                trades_frame = pd.concat(combined_trades, ignore_index=True) if combined_trades else pd.DataFrame()
                wf_trades_frame = pd.concat(wf_trades, ignore_index=True) if wf_trades else pd.DataFrame()
                holdout_trades = trades_frame[trades_frame["fold_id"].astype(str).eq("holdout_90d")].copy()
                holdout_equity = combined_equities[-1][1]
                for scope, frame, equity, trades in [
                    ("wf_pre_holdout", wf_frame, wf_equity, wf_trades_frame),
                    ("holdout_90d", holdout_frame, holdout_equity, holdout_trades),
                    ("combined_oos", all_frame, all_equity, trades_frame),
                ]:
                    summary = summary_for_scope(frame, equity, trades, scope, account_equity)
                    summary.update({"selection_mode": mode, "policy": policy.name, "scenario": scenario.name})
                    summary_rows.append(summary)
                equity_key = f"{mode}__{policy.name}__{scenario.name}"
                equity_frame = pd.DataFrame({"timestamp": all_equity.index.astype(str), "equity": all_equity.to_numpy()})
                equity_frame.to_csv(output_dir / "oos_equity" / f"{equity_key}.csv", index=False)
                equity_outputs[equity_key] = equity_frame

    fold_metrics = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    holdout_summary = summary[summary["scope"].eq("holdout_90d")].copy()
    selected_rows_frame = pd.DataFrame(selected_rows_output)
    legacy_rank, realistic_rank, robust_rank = rank_outputs(summary)

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "strategy_scenario_summary.csv", index=False)
    holdout_summary.to_csv(output_dir / "holdout_summary.csv", index=False)
    selected_rows_frame.to_csv(output_dir / "selected_candidates.csv", index=False)
    legacy_rank.to_csv(output_dir / "ranking_best_legacy_return.csv", index=False)
    realistic_rank.to_csv(output_dir / "ranking_best_realistic_net.csv", index=False)
    robust_rank.to_csv(output_dir / "ranking_best_robustness.csv", index=False)
    write_figures(output_dir, summary, equity_outputs)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "max_folds": max_folds,
        "selected_policies": selected_policy_names,
        "scenarios": [scenario.name for scenario in scenarios],
        "selection_modes": list(config["selection_modes"]),
        "holdout_start_utc": str(holdout_start),
        "holdout_end_utc": str(holdout_index.max()),
        "fold_count_pre_holdout": len(windows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    payload = {
        "data_audit": [data_audit],
        "policy_distinctness": distinctness.to_dict("records"),
        "summary": summary.to_dict("records"),
        "ranking_best_legacy_return": legacy_rank.to_dict("records"),
        "ranking_best_realistic_net": realistic_rank.to_dict("records"),
        "ranking_best_robustness": robust_rank.to_dict("records"),
    }
    write_report(output_dir, payload)
    return {
        "output_dir": str(output_dir),
        "manifest": manifest,
        "top_legacy": legacy_rank.head(1).to_dict("records"),
        "top_realistic": realistic_rank.head(1).to_dict("records"),
        "top_robust": robust_rank.head(1).to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Revalidate top 3 distinct BTCUSDT policies.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(args.config, smoke=args.smoke, max_folds=args.max_folds)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
