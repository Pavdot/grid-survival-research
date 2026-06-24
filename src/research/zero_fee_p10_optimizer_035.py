from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.metrics import calculate_metrics
from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.cost_budget_hypothesis_ablation_033 import (
    apply_maker_proxy,
    enrich_cost_metrics,
    order_count,
    recompute_metrics,
)
from src.research.economy_first_research import prepare_market, summarize_simulations
from src.research.execution_worst_case_audit import (
    AuditScenario,
    make_equity_curve,
    sample_positions_for_scenario,
    simulate_candidate_sample_for_scenario,
    simulate_grid_from_signal_pos,
)
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    build_side_signal,
    candidate_from_row,
    monthly_return_from_equity,
    risk_for_candidate,
)
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.research.revalidate_top3_distinct_policies import (
    EvaluationContext,
    PolicySpec,
    ScenarioSpec,
    audit_data_files,
    load_selected_candidates,
    prefilter_side_signal_for_audit,
    resolve_path,
    split_pre_holdout,
    unique_candidate_rows,
)
from src.research.walk_forward_martingale_research import WalkForwardWindow, stitch_oos_equity
from src.utils.config_loader import load_strategy_config, load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_zero_fee_p10_optimizer_035.yaml"


def scenario_spec(name: str, fee_rate: float, slippage_bps: float) -> ScenarioSpec:
    return ScenarioSpec(
        name=name,
        engine="audit",
        entry_execution_mode="next_bar_open",
        signal_lag_bars=1,
        mask_lag_bars=1,
        fee_rate=float(fee_rate),
        slippage_bps=float(slippage_bps),
    )


def sizing_sum(levels: int, progression: float) -> float:
    return float(sum(float(progression) ** level for level in range(int(levels))))


def candidate_base_size(cap: float, levels: int, progression: float, safety: float) -> float:
    return float(float(cap) / sizing_sum(int(levels), float(progression)) * float(safety))


def internal_train_selection_split(
    train_index: pd.Index,
    search_days: float,
    selection_days: float,
) -> tuple[pd.Index, pd.Index]:
    if train_index.empty:
        raise ValueError("train_index is empty")
    train_start = pd.Timestamp(train_index.min())
    selection_start = pd.Timestamp(train_index.max()) - pd.Timedelta(days=float(selection_days))
    search_end = min(train_start + pd.Timedelta(days=float(search_days)), selection_start)
    search_index = train_index[(train_index >= train_start) & (train_index < search_end)]
    selection_index = train_index[train_index >= selection_start]
    if search_index.empty or selection_index.empty:
        raise ValueError("internal search/selection split is empty")
    if pd.Timestamp(search_index.max()) >= pd.Timestamp(selection_index.min()):
        raise ValueError("internal search and selection windows overlap")
    return search_index, selection_index


def _base_template_row(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not selected_rows:
        raise ValueError("selected_rows is empty")
    row = dict(selected_rows[0])
    row["side_mode"] = "mean_reversion_dual"
    row["entry_mode"] = "hourly_extreme"
    row["rsi_window"] = int(row.get("rsi_window", 24))
    row["max_grid_loss_pct"] = float(row.get("max_grid_loss_pct", 0.35))
    row["stop_on_regime_break"] = False
    row["stop_on_volatility_shock"] = True
    return row


def generate_candidate_universe(
    selected_rows: list[dict[str, Any]],
    config: dict[str, Any],
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    search = config["candidate_search"]
    template = _base_template_row(selected_rows)
    total_requested = int(max_candidates or search["max_candidates"])
    if total_requested <= 0:
        raise ValueError("max_candidates must be positive")
    rng = np.random.default_rng(int(search.get("random_seed", 35)))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    safety = float(search["base_size_safety"])
    while len(rows) < total_requested and attempts < total_requested * 50:
        attempts += 1
        low, high = search["rsi_threshold_pairs"][int(rng.integers(0, len(search["rsi_threshold_pairs"])))]
        levels = int(rng.choice(search["max_levels"]))
        progression = float(rng.choice(search["progression_multipliers"]))
        cap = float(rng.choice(search["max_notional_caps"]))
        row = {
            **template,
            "name": "zero_fee_p10_candidate",
            "entry_cooldown_hours": float(rng.choice(search["cooldown_hours"])),
            "rsi_low": float(low),
            "rsi_high": float(high),
            "spacing_atr_multiplier": float(rng.choice(search["spacing_atr_multipliers"])),
            "take_profit_spacing_multiplier": float(rng.choice(search["take_profit_spacing_multipliers"])),
            "max_levels": levels,
            "progression_multiplier": progression,
            "max_total_exposure_pct": cap,
            "base_position_size_pct": candidate_base_size(cap, levels, progression, safety),
            "fee_rate": float(config["primary_scenario"]["fee_rate"]),
            "slippage_bps": float(config["primary_scenario"]["slippage_bps"]),
            "max_holding_hours": float(rng.choice(search["max_holding_hours"])),
            "max_grids_per_month": float(rng.choice(search["max_grids_per_month"])),
            "blackout_hours": int(rng.choice(search["blackout_hours"])),
            "min_severity": int(rng.choice(search["min_severity"])),
            "trend_propagation_bars": int(rng.choice(search["trend_propagation_bars"])),
            "breakout_atr_buffer": float(rng.choice(search["breakout_atr_buffer"])),
            "min_range_expansion_ratio": float(rng.choice(search["min_range_expansion_ratio"])),
            "pause_after_forced_loss_hours": float(rng.choice(search["pause_after_forced_loss_hours"])),
            "rolling_7d_loss_threshold": float(rng.choice(search["rolling_7d_loss_thresholds"])),
        }
        key = (
            row["entry_cooldown_hours"],
            row["rsi_low"],
            row["rsi_high"],
            row["spacing_atr_multiplier"],
            row["take_profit_spacing_multiplier"],
            row["max_levels"],
            row["progression_multiplier"],
            row["max_total_exposure_pct"],
            row["max_holding_hours"],
            row["max_grids_per_month"],
            row["blackout_hours"],
            row["min_severity"],
            row["trend_propagation_bars"],
            row["breakout_atr_buffer"],
            row["min_range_expansion_ratio"],
            row["pause_after_forced_loss_hours"],
            row["rolling_7d_loss_threshold"],
        )
        if key in seen:
            continue
        seen.add(key)
        row["candidate_uid"] = f"zf035_{len(rows):05d}"
        row["name"] = (
            f"zf035_cool{row['entry_cooldown_hours']:g}_rsi{row['rsi_low']:g}_{row['rsi_high']:g}"
            f"_sp{row['spacing_atr_multiplier']:g}_tp{row['take_profit_spacing_multiplier']:g}"
            f"_lvl{row['max_levels']}_prog{row['progression_multiplier']:g}_cap{row['max_total_exposure_pct']:g}"
            f"_hold{row['max_holding_hours']:g}_grid{row['max_grids_per_month']:g}"
        ).replace(".", "p")
        rows.append(row)
    if not rows:
        raise ValueError("candidate universe is empty")
    return rows


def config_for_candidate(base_config: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config, default=str))
    hours = int(row["blackout_hours"])
    cfg["fundamental_blackout"]["pre_event_hours"] = hours
    cfg["fundamental_blackout"]["post_event_hours"] = hours
    cfg["fundamental_blackout"]["min_severity"] = int(row["min_severity"])
    cfg["trend_escape"]["propagation_bars"] = int(row["trend_propagation_bars"])
    cfg["trend_escape"]["breakout_atr_buffer"] = float(row["breakout_atr_buffer"])
    cfg["trend_escape"]["min_range_expansion_ratio"] = float(row["min_range_expansion_ratio"])
    return cfg


def build_entry_mask_for_candidate(market: pd.DataFrame, row: dict[str, Any], config: dict[str, Any]) -> pd.Series:
    cfg = config_for_candidate(config, row)
    _events, _windows, blackout_masks = build_blackout_bundle(market.index, cfg)
    trend_components = build_trend_escape_components(market, cfg)
    return fundamental_trend_mask(trend_components["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)


def forced_loss(row: pd.Series) -> bool:
    forced_cols = [
        "stopped_by_max_loss",
        "stopped_by_max_holding",
        "stopped_by_volatility_shock",
        "stopped_by_exposure",
        "stopped_by_kill_switch",
    ]
    return bool(float(row.get("realized_pnl", 0.0)) < 0 and any(int(row.get(column, 0)) for column in forced_cols))


def run_gated_backtest(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    row: dict[str, Any],
    split_index: pd.Index,
    scenario: ScenarioSpec,
    entry_mask: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    split_frame = market.loc[split_index]
    candidate = candidate_from_row({**row, "fee_rate": scenario.fee_rate, "slippage_bps": scenario.slippage_bps})
    risk = risk_for_candidate(base_risk, candidate)
    raw_signal = build_side_signal(market, signal_frame, candidate)
    side_signal = raw_signal.reindex(split_frame.index).shift(int(scenario.signal_lag_bars))
    shifted_entry_mask = entry_mask.reindex(split_frame.index).fillna(False).astype(bool).shift(int(scenario.mask_lag_bars)).fillna(False).astype(bool)
    audit_scenario = AuditScenario(
        name=scenario.name,
        variant="fundamental_trend_escape_entry_only",
        selection_policy="zero_fee_p10_optimizer_035",
        entry_execution_mode=scenario.entry_execution_mode,
    )
    rows: list[dict[str, Any]] = []
    i = 0
    horizon_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    entry_offset = 1
    pause_until = pd.Timestamp.min.tz_localize("UTC")
    month_counts: dict[str, int] = {}
    max_grids_per_month = int(row["max_grids_per_month"])
    while i < len(split_frame) - horizon_bars - entry_offset:
        side = side_signal.iloc[i]
        if side not in {"long", "short"}:
            i += 1
            continue
        entry_pos = i + entry_offset
        if entry_pos >= len(split_frame):
            break
        entry_ts = split_frame.index[entry_pos]
        if bool(shifted_entry_mask.iloc[entry_pos]):
            i += 1
            continue
        if entry_ts < pause_until:
            i += 1
            continue
        month_key = entry_ts.strftime("%Y-%m")
        if month_counts.get(month_key, 0) >= max_grids_per_month:
            i += 1
            continue
        if rows:
            prior = pd.DataFrame(rows)
            prior["exit_timestamp"] = pd.to_datetime(prior["exit_timestamp"], utc=True)
            recent = prior[(prior["exit_timestamp"] < entry_ts) & (prior["exit_timestamp"] >= entry_ts - pd.Timedelta(days=7))]
            rolling_pnl = float(recent["realized_pnl"].sum()) if not recent.empty else 0.0
            if rolling_pnl < float(row["rolling_7d_loss_threshold"]):
                i += 1
                continue
        result, extras = simulate_grid_from_signal_pos(
            split_frame,
            i,
            risk,
            str(side),
            audit_scenario,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            survival_min_realized_pnl=0.0,
        )
        out = result.to_dict()
        out.update(asdict(candidate))
        out.update(extras)
        out.update(
            {
                "candidate_uid": row["candidate_uid"],
                "max_grids_per_month": max_grids_per_month,
                "pause_after_forced_loss_hours": float(row["pause_after_forced_loss_hours"]),
                "rolling_7d_loss_threshold": float(row["rolling_7d_loss_threshold"]),
                "blackout_hours": int(row["blackout_hours"]),
                "min_severity": int(row["min_severity"]),
                "trend_propagation_bars": int(row["trend_propagation_bars"]),
                "breakout_atr_buffer": float(row["breakout_atr_buffer"]),
                "min_range_expansion_ratio": float(row["min_range_expansion_ratio"]),
            }
        )
        rows.append(out)
        month_counts[month_key] = month_counts.get(month_key, 0) + 1
        if forced_loss(pd.Series(out)):
            pause_until = pd.Timestamp(out["exit_timestamp"]) + pd.Timedelta(hours=float(row["pause_after_forced_loss_hours"]))
        cooldown_until = pd.Timestamp(out["exit_timestamp"]) + pd.Timedelta(hours=float(candidate.entry_cooldown_hours))
        i = max(split_frame.index.get_loc(out["exit_timestamp"]) + 1, int(split_frame.index.searchsorted(cooldown_until)))
    trades = pd.DataFrame(rows)
    if trades.empty:
        trades = pd.DataFrame(columns=["realized_pnl", "exit_timestamp", "number_of_levels_filled", "fees_paid", "slippage_paid"])
    equity = make_equity_curve(split_frame.index, trades)
    metrics = summarize_gated_result(equity, trades, split_frame.index, candidate)
    metrics.update({"scenario": scenario.name, "candidate_uid": row["candidate_uid"], "selected_name": row["name"]})
    return metrics, trades, equity


def monthly_chunk_returns(equity: pd.Series, chunk_days: float = 30.4375) -> list[float]:
    equity = equity.dropna()
    if len(equity) < 2:
        return [0.0]
    start = pd.Timestamp(equity.index.min())
    end = pd.Timestamp(equity.index.max())
    out: list[float] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + pd.Timedelta(days=float(chunk_days)), end)
        chunk = equity[(equity.index >= cursor) & (equity.index <= nxt)]
        if len(chunk) >= 2:
            out.append(monthly_return_from_equity(chunk))
        cursor = nxt
    return out or [monthly_return_from_equity(equity)]


def summarize_gated_result(
    equity: pd.Series,
    trades: pd.DataFrame,
    index: pd.Index,
    candidate: MonthlyMartingaleCandidate,
) -> dict[str, Any]:
    metrics = calculate_metrics(equity, trades if not trades.empty else None)
    if not trades.empty and "unrealized_drawdown_max" in trades:
        metrics.update(summarize_simulations(trades, baseline_grids=len(trades)))
    else:
        metrics.update({"number_of_grids": 0, "realized_pnl": 0.0, "profit_factor": 0.0, "fees_paid": 0.0, "slippage_paid": 0.0, "number_of_forced_exits": 0})
    metrics.update(asdict(candidate))
    metrics["monthly_return"] = monthly_return_from_equity(equity)
    metrics["monthly_p10_chunks"] = float(np.quantile(monthly_chunk_returns(equity), 0.10))
    metrics["equity_ruined"] = bool((equity <= 0).any() or float(metrics.get("max_drawdown", 0.0)) <= -1.0)
    return enrich_cost_metrics(metrics, trades, index, 10000.0)


def selection_constraints(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    sel = config["selection"]
    reasons: list[str] = []
    if bool(metrics.get("equity_ruined", False)):
        reasons.append("ruin")
    if float(metrics.get("max_drawdown", 0.0)) <= float(sel["max_drawdown"]):
        reasons.append("drawdown")
    if float(metrics.get("grids_per_month", 0.0)) < float(sel["min_grids_per_month"]):
        reasons.append("min_grids")
    if float(metrics.get("orders_per_month", 0.0)) > float(sel["max_orders_per_month"]):
        reasons.append("max_orders")
    if bool(sel.get("require_net_pnl_per_order_positive", True)) and float(metrics.get("net_pnl_per_order", 0.0)) <= 0:
        reasons.append("net_pnl_per_order")
    return not reasons, "|".join(reasons) if reasons else "pass"


def sample_prune_candidates(
    universe: list[dict[str, Any]],
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    search_index: pd.Index,
    scenario: ScenarioSpec,
    mask_cache: dict[tuple[Any, ...], pd.Series],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stride = int(config["candidate_search"]["search_entry_stride_bars"])
    max_positions = int(config["candidate_search"]["max_sample_positions_per_candidate"])
    for idx, row in enumerate(universe):
        candidate = candidate_from_row({**row, "fee_rate": scenario.fee_rate, "slippage_bps": scenario.slippage_bps})
        risk = risk_for_candidate(base_risk, candidate)
        side_signal = build_side_signal(market, signal_frame, candidate)
        entry_mask = mask_cache[mask_key(row)].astype(bool).shift(int(scenario.mask_lag_bars)).fillna(False).astype(bool)
        shifted_signal = prefilter_side_signal_for_audit(side_signal, market.index, entry_mask, scenario)
        audit_scenario = AuditScenario(
            name=scenario.name,
            variant="fundamental_trend_escape_entry_only",
            selection_policy="zero_fee_p10_search_prune",
            entry_execution_mode=scenario.entry_execution_mode,
        )
        positions = sample_positions_for_scenario(
            market,
            search_index,
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
        sample = simulate_candidate_sample_for_scenario(market, positions, risk, shifted_signal, candidate, "search", audit_scenario)
        metrics = summarize_simulations(sample, baseline_grids=len(sample))
        metrics = enrich_cost_metrics(metrics, sample, search_index, 10000.0)
        passed, reasons = selection_constraints(metrics, config)
        metrics.update(
            {
                "candidate_row_index": idx,
                "candidate_uid": row["candidate_uid"],
                "selected_name": row["name"],
                "stage": "search_prune",
                "constraint_pass": bool(passed),
                "constraint_reasons": reasons,
                "sample_positions": len(positions),
                "monthly_p10_chunks": float(metrics.get("realized_pnl", 0.0)),
            }
        )
        rows.append(metrics)
    if not rows:
        raise ValueError("search/prune generated no rows")
    frame = pd.DataFrame(rows)
    frame["constraint_rank"] = (~frame["constraint_pass"].astype(bool)).astype(int)
    return frame.sort_values(
        ["constraint_rank", "monthly_p10_chunks", "net_pnl_per_order", "profit_factor", "orders_per_month"],
        ascending=[True, False, False, False, True],
    )


def mask_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["blackout_hours"]),
        int(row["min_severity"]),
        int(row["trend_propagation_bars"]),
        float(row["breakout_atr_buffer"]),
        float(row["min_range_expansion_ratio"]),
    )


def select_on_train(
    universe: list[dict[str, Any]],
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    train_index: pd.Index,
    scenario: ScenarioSpec,
    mask_cache: dict[tuple[Any, ...], pd.Series],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    wf = config["walk_forward"]
    search_index, selection_index = internal_train_selection_split(
        train_index,
        float(wf["internal_search_days"]),
        float(wf["internal_selection_days"]),
    )
    prune = sample_prune_candidates(universe, market, signal_frame, base_risk, search_index, scenario, mask_cache, config)
    exact_top_n = int(config["candidate_search"]["exact_top_n"])
    top_indexes = prune.head(exact_top_n)["candidate_row_index"].astype(int).tolist()
    exact_rows: list[dict[str, Any]] = []
    for row_index in top_indexes:
        row = universe[int(row_index)]
        metrics, _trades, equity = run_gated_backtest(
            market,
            signal_frame,
            base_risk,
            row,
            selection_index,
            scenario,
            mask_cache[mask_key(row)],
        )
        passed, reasons = selection_constraints(metrics, config)
        metrics.update(
            {
                "candidate_row_index": int(row_index),
                "candidate_uid": row["candidate_uid"],
                "selected_name": row["name"],
                "stage": "selection_exact",
                "constraint_pass": bool(passed),
                "constraint_reasons": reasons,
                "selection_uses_drawdown": False,
                "selected_from_train_only": True,
                "selection_start": selection_index.min(),
                "selection_end": selection_index.max(),
            }
        )
        exact_rows.append(metrics)
    exact = pd.DataFrame(exact_rows)
    if exact.empty:
        raise ValueError("selection exact generated no rows")
    exact["constraint_rank"] = (~exact["constraint_pass"].astype(bool)).astype(int)
    exact = exact.sort_values(
        ["constraint_rank", "monthly_p10_chunks", "monthly_return", "max_drawdown", "orders_per_month", "net_pnl_per_order"],
        ascending=[True, False, False, False, True, False],
    )
    selected = exact.iloc[0].to_dict()
    matrix = pd.concat([prune, exact], ignore_index=True, sort=False)
    return universe[int(selected["candidate_row_index"])], selected, matrix


def evaluate_selected(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    row: dict[str, Any],
    split_index: pd.Index,
    scenario: ScenarioSpec,
    entry_mask: pd.Series,
    maker_penalty: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    metrics, trades, equity = run_gated_backtest(market, signal_frame, base_risk, row, split_index, scenario, entry_mask)
    if maker_penalty > 0:
        trades = apply_maker_proxy(trades, maker_penalty)
        candidate = candidate_from_row({**row, "fee_rate": scenario.fee_rate, "slippage_bps": scenario.slippage_bps})
        metrics, equity = recompute_metrics(market.loc[split_index].index, trades, "test", candidate)
        metrics = enrich_cost_metrics(metrics, trades, split_index, 10000.0)
    return metrics, trades, equity


def fold_row(
    fold_id: int | str,
    scenario_name: str,
    selected_row: dict[str, Any],
    selection_metrics: dict[str, Any],
    metrics: dict[str, Any],
    train_index: pd.Index,
    test_index: pd.Index,
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "scenario": scenario_name,
        "is_holdout": str(fold_id) == "holdout_90d",
        "candidate_uid": selected_row["candidate_uid"],
        "selected_name": selected_row["name"],
        "train_start": train_index.min() if len(train_index) else pd.NaT,
        "train_end": train_index.max() if len(train_index) else pd.NaT,
        "test_start": test_index.min() if len(test_index) else pd.NaT,
        "test_end": test_index.max() if len(test_index) else pd.NaT,
        "selection_monthly_p10": float(selection_metrics.get("monthly_p10_chunks", np.nan)),
        "selection_monthly_return": float(selection_metrics.get("monthly_return", np.nan)),
        "selection_constraint_pass": bool(selection_metrics.get("constraint_pass", False)),
        "selection_constraint_reasons": selection_metrics.get("constraint_reasons", ""),
        "monthly_return": float(metrics.get("monthly_return", 0.0)),
        "total_return": float(metrics.get("total_return", 0.0)),
        "annualized_return": float(metrics.get("annualized_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "monthly_p10_chunks": float(metrics.get("monthly_p10_chunks", 0.0)),
        "equity_ruined": bool(metrics.get("equity_ruined", False)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "number_of_grids": int(metrics.get("number_of_grids", 0)),
        "orders": int(metrics.get("orders", 0)),
        "orders_per_month": float(metrics.get("orders_per_month", 0.0)),
        "grids_per_month": float(metrics.get("grids_per_month", 0.0)),
        "fees_paid": float(metrics.get("fees_paid", 0.0)),
        "slippage_paid": float(metrics.get("slippage_paid", 0.0)),
        "cost_total_per_month": float(metrics.get("cost_total_per_month", 0.0)),
        "net_pnl": float(metrics.get("net_pnl", 0.0)),
        "net_pnl_per_order": float(metrics.get("net_pnl_per_order", 0.0)),
        "expectancy_per_grid": float(metrics.get("expectancy_per_grid", 0.0)),
        "effective_leverage": float(metrics.get("effective_leverage", 0.0)),
        "max_notional": float(metrics.get("max_notional", 0.0)),
    }


def summarize_scope(
    scenario_name: str,
    frame: pd.DataFrame,
    equity: pd.Series,
    trades: pd.DataFrame,
    scope: str,
) -> dict[str, Any]:
    metrics = calculate_metrics(equity, trades if not trades.empty else None)
    if not trades.empty and "unrealized_drawdown_max" in trades:
        metrics.update(summarize_simulations(trades, baseline_grids=len(trades)))
    else:
        metrics.update({"number_of_grids": 0, "realized_pnl": 0.0, "profit_factor": 0.0, "fees_paid": 0.0, "slippage_paid": 0.0})
    returns = frame["monthly_return"].astype(float) if not frame.empty else pd.Series(dtype=float)
    return {
        "scope": scope,
        "scenario": scenario_name,
        "fold_count": int(len(frame)),
        "monthly_return": float(monthly_return_from_equity(equity)),
        "total_return": float(metrics.get("total_return", 0.0)),
        "annualized_return": float(metrics.get("annualized_return", 0.0)),
        "monthly_p10": float(returns.quantile(0.10)) if not returns.empty else 0.0,
        "monthly_median": float(returns.median()) if not returns.empty else 0.0,
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "equity_ruined": bool((equity <= 0).any() or float(metrics.get("max_drawdown", 0.0)) <= -1.0 or frame.get("equity_ruined", pd.Series(dtype=bool)).astype(bool).any()),
        "positive_fold_rate": float((frame["total_return"].astype(float) > 0).mean()) if not frame.empty else 0.0,
        "fold_rate_above_3pct": float(returns.ge(0.03).mean()) if not returns.empty else 0.0,
        "fold_rate_above_8pct": float(returns.ge(0.08).mean()) if not returns.empty else 0.0,
        "fold_rate_above_12pct": float(returns.ge(0.12).mean()) if not returns.empty else 0.0,
        "fold_rate_above_20pct": float(returns.ge(0.20).mean()) if not returns.empty else 0.0,
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "number_of_grids": int(metrics.get("number_of_grids", 0)),
        "orders": order_count(trades),
        "orders_per_month": float(frame["orders_per_month"].mean()) if not frame.empty else 0.0,
        "grids_per_month": float(frame["grids_per_month"].mean()) if not frame.empty else 0.0,
        "cost_total_per_month": float(frame["cost_total_per_month"].mean()) if not frame.empty else 0.0,
        "net_pnl": float(metrics.get("realized_pnl", 0.0)),
        "net_pnl_per_order": float(metrics.get("realized_pnl", 0.0)) / max(order_count(trades), 1),
        "expectancy_per_grid": float(metrics.get("realized_pnl", 0.0)) / max(int(metrics.get("number_of_grids", 0)), 1),
        "effective_leverage": float(frame["effective_leverage"].max()) if not frame.empty else 0.0,
        "max_notional": float(frame["max_notional"].max()) if not frame.empty else 0.0,
    }


def cpcv_process_summary(primary_folds: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    folds = primary_folds[~primary_folds["is_holdout"].astype(bool)].sort_values("fold_id").reset_index(drop=True)
    if folds.empty:
        return pd.DataFrame()
    blocks = np.array_split(np.arange(len(folds)), int(config["validation"]["cpcv_blocks"]))
    rows: list[dict[str, Any]] = []
    for combo in itertools.combinations(range(len(blocks)), int(config["validation"]["cpcv_test_blocks"])):
        test_idx = np.concatenate([blocks[index] for index in combo])
        purge = set()
        if bool(config["validation"].get("cpcv_purge_adjacent_blocks", True)):
            for index in combo:
                purge.update([index - 1, index + 1])
        train_blocks = [idx for idx in range(len(blocks)) if idx not in set(combo) and idx not in purge and 0 <= idx < len(blocks)]
        test_returns = folds.iloc[test_idx]["monthly_return"].astype(float)
        rows.append(
            {
                "test_blocks": ",".join(str(value) for value in combo),
                "train_blocks_after_purge": ",".join(str(value) for value in train_blocks),
                "test_fold_count": int(len(test_returns)),
                "monthly_median": float(test_returns.median()),
                "monthly_p25": float(test_returns.quantile(0.25)),
                "monthly_p10": float(test_returns.quantile(0.10)),
                "positive_rate": float(test_returns.gt(0).mean()),
                "ruin": bool(folds.iloc[test_idx]["equity_ruined"].astype(bool).any()),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_summary(primary_folds: pd.DataFrame, trades: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["validation"]["monte_carlo_seed"]))
    iterations = int(config["validation"]["monte_carlo_iterations"])
    folds = primary_folds[~primary_folds["is_holdout"].astype(bool)].sort_values("fold_id").reset_index(drop=True)
    fold_returns = folds["monthly_return"].astype(float).to_numpy()
    rows: list[dict[str, Any]] = []
    if len(fold_returns):
        sims = np.array([rng.choice(fold_returns, size=len(fold_returns), replace=True).mean() for _ in range(iterations)])
        rows.append(
            {
                "method": "fold_bootstrap",
                "monthly_median": float(np.median(sims)),
                "monthly_p05": float(np.quantile(sims, 0.05)),
                "positive_probability": float((sims > 0).mean()),
                "ruin_probability": float(folds["equity_ruined"].astype(bool).mean()),
            }
        )
        block_size = max(1, min(3, len(fold_returns)))
        block_sims = []
        for _ in range(iterations):
            sampled: list[float] = []
            while len(sampled) < len(fold_returns):
                start = int(rng.integers(0, max(1, len(fold_returns) - block_size + 1)))
                sampled.extend(fold_returns[start : start + block_size].tolist())
            block_sims.append(np.mean(sampled[: len(fold_returns)]))
        block_sims_arr = np.asarray(block_sims)
        rows.append(
            {
                "method": "fold_block_bootstrap",
                "monthly_median": float(np.median(block_sims_arr)),
                "monthly_p05": float(np.quantile(block_sims_arr, 0.05)),
                "positive_probability": float((block_sims_arr > 0).mean()),
                "ruin_probability": float(folds["equity_ruined"].astype(bool).mean()),
            }
        )
    if not trades.empty:
        pnl = trades["realized_pnl"].astype(float).to_numpy()
        months = max(float(len(folds)), 1.0)
        trade_sims = np.array([rng.choice(pnl, size=len(pnl), replace=True).sum() / months for _ in range(iterations)])
        rows.append(
            {
                "method": "trade_bootstrap",
                "monthly_median": float(np.median(trade_sims)),
                "monthly_p05": float(np.quantile(trade_sims, 0.05)),
                "positive_probability": float((trade_sims > 0).mean()),
                "ruin_probability": 0.0,
            }
        )
    return pd.DataFrame(rows)


def final_verdict(summary: pd.DataFrame, cpcv: pd.DataFrame, mc: pd.DataFrame, config: dict[str, Any]) -> str:
    primary = summary[(summary["scope"].eq("combined_oos")) & (summary["scenario"].eq(config["primary_scenario"]["name"]))]
    holdout = summary[(summary["scope"].eq("holdout_90d")) & (summary["scenario"].eq(config["primary_scenario"]["name"]))]
    if primary.empty:
        return "not salvageable by zero-fee optimization"
    row = primary.iloc[0]
    hold = holdout.iloc[0] if not holdout.empty else row
    validation = config["validation"]
    cpcv_pass = not cpcv.empty and float(cpcv["monthly_p25"].min()) > float(validation["cpcv_p25_min"]) and not cpcv["ruin"].astype(bool).any()
    mc_pass = not mc.empty and float(mc["monthly_p05"].min()) > float(validation["mc_p05_min"]) and float(mc["positive_probability"].min()) >= float(validation["positive_probability_min"]) and float(mc["ruin_probability"].max()) <= float(validation["ruin_probability_max"])
    wf_pass = float(row["monthly_p10"]) > float(validation["robust_monthly_p10_min"]) and not bool(row["equity_ruined"])
    holdout_pass = float(hold["total_return"]) >= float(validation["holdout_min_total_return"])
    if wf_pass and holdout_pass and cpcv_pass and mc_pass:
        return "zero-fee robust candidate"
    if float(row["monthly_return"]) > 0 and not bool(row["equity_ruined"]):
        return "zero-fee fragile but improved"
    return "not salvageable by zero-fee optimization"


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    summary = pd.DataFrame(payload["scenario_summary"])
    cpcv = pd.DataFrame(payload["cpcv_summary"])
    mc = pd.DataFrame(payload["monte_carlo_summary"])
    rankings = payload["rankings"]
    lines = [
        "# Iteration 035 - Zero-Fee P10 Optimizer",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        "## Scenario Summary",
        markdown_table(
            summary[summary["scope"].eq("combined_oos")][
                [
                    "scenario",
                    "monthly_return",
                    "monthly_p10",
                    "monthly_median",
                    "max_drawdown",
                    "equity_ruined",
                    "positive_fold_rate",
                    "orders_per_month",
                    "net_pnl_per_order",
                    "effective_leverage",
                ]
            ]
        ),
        "",
        "## Holdout",
        markdown_table(summary[summary["scope"].eq("holdout_90d")][["scenario", "monthly_return", "total_return", "max_drawdown", "equity_ruined"]]),
        "",
        "## CPCV",
        markdown_table(cpcv) if not cpcv.empty else "No CPCV rows.",
        "",
        "## Monte Carlo",
        markdown_table(mc) if not mc.empty else "No Monte Carlo rows.",
        "",
        "## Rankings",
        f"- Best P10 monthly: `{rankings.get('best_p10', '')}`",
        f"- Best holdout: `{rankings.get('best_holdout', '')}`",
        f"- Best compromise: `{rankings.get('best_compromise', '')}`",
        f"- Best 0.25bps -> 0.5bps sensitivity: `{rankings.get('best_sensitivity', '')}`",
        "",
        "## Notes",
        "- Selection target is zero-fee with 0.25bps slippage.",
        "- All selection happens on train only; the 90d holdout is selected from the preceding train window once.",
        "- CPCV is a purged process-stability check over OOS selected-fold returns.",
    ]
    (output_dir / "iteration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output_dir: Path, summary: pd.DataFrame, equity_outputs: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    if not combined.empty:
        ordered = combined.sort_values("monthly_p10", ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(ordered["scenario"], ordered["monthly_p10"].astype(float) * 100)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=70, ha="right", fontsize=8)
        plt.ylabel("P10 monthly (%)")
        plt.title("Iteration 035 P10 Monthly by Scenario")
        plt.tight_layout()
        plt.savefig(figures / "p10_monthly_by_scenario.png", dpi=150)
        plt.close()
    plt.figure(figsize=(12, 6))
    for key, frame in equity_outputs.items():
        if "combined_oos" not in key:
            continue
        plt.plot(pd.to_datetime(frame["timestamp"], utc=True), frame["equity"], label=key.replace("__combined_oos", ""))
    plt.title("Iteration 035 Combined OOS Equity")
    plt.ylabel("Equity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures / "combined_oos_equity.png", dpi=150)
    plt.close()


def run_iteration(
    config_path: str = DEFAULT_CONFIG,
    smoke: bool = False,
    max_folds: int | None = None,
    max_candidates: int | None = None,
    exact_top_n: int | None = None,
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if max_candidates is not None:
        config["candidate_search"]["max_candidates"] = int(max_candidates)
    if exact_top_n is not None:
        config["candidate_search"]["exact_top_n"] = int(exact_top_n)
    if smoke and max_folds is None:
        max_folds = 2
    if smoke:
        config["walk_forward"]["train_days"] = min(float(config["walk_forward"]["train_days"]), 30.0)
        config["walk_forward"]["test_days"] = min(float(config["walk_forward"]["test_days"]), 5.0)
        config["walk_forward"]["step_days"] = min(float(config["walk_forward"]["step_days"]), 5.0)
        config["walk_forward"]["holdout_days"] = min(float(config["walk_forward"]["holdout_days"]), 15.0)
        config["walk_forward"]["internal_search_days"] = 18.0
        config["walk_forward"]["internal_selection_days"] = 10.0
        config["candidate_search"]["max_candidates"] = min(int(config["candidate_search"]["max_candidates"]), 12)
        config["candidate_search"]["exact_top_n"] = min(int(config["candidate_search"]["exact_top_n"]), 4)
        config["candidate_search"]["max_sample_positions_per_candidate"] = 4
        config["scenario_replay"] = config["scenario_replay"][:3]
        config["validation"]["monte_carlo_iterations"] = 200

    output_root = resolve_path(config["iteration"]["output_root"])
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"zero_fee_p10_optimizer_035_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "oos_equity").mkdir(parents=True, exist_ok=True)
    (output_dir / "finalist_trades").mkdir(parents=True, exist_ok=True)

    market_path = resolve_path(config["data"]["market_5m_path"])
    signal_path = resolve_path(config["data"]["signal_1h_path"])
    market_audited, signal_frame, data_audit = audit_data_files(
        market_path,
        signal_path,
        float(config["data"]["min_coverage_rate"]),
        int(config["data"]["bar_minutes"]),
    )
    data_audit["output_dir"] = str(output_dir)
    pd.DataFrame([data_audit]).to_csv(output_dir / "data_audit.csv", index=False)
    (output_dir / "data_audit.json").write_text(json.dumps(data_audit, indent=2, default=str), encoding="utf-8")

    market = prepare_market(str(config.get("asset", "btcusdt"))).reindex(market_audited.index).dropna(subset=["open", "high", "low", "close"])
    wf = config["walk_forward"]
    windows, holdout_index, holdout_start = split_pre_holdout(
        market,
        float(wf["train_days"]),
        float(wf["test_days"]),
        float(wf["step_days"]),
        int(wf["embargo_bars"]),
        float(wf["holdout_days"]),
        max_folds=max_folds,
    )
    holdout_train_start = holdout_start - pd.Timedelta(days=float(wf["train_days"]))
    holdout_train_index = market.index[(market.index >= holdout_train_start) & (market.index < holdout_start)]
    if holdout_train_index.empty:
        raise ValueError("holdout train window is empty")

    policy = PolicySpec(
        name=str(config["source_policy"]["name"]),
        source_iteration_dir=resolve_path(config["source_policy"]["source_iteration_dir"]),
        source_variant=str(config["source_policy"]["source_variant"]),
        policy_kind=str(config["source_policy"]["policy_kind"]),
    )
    selected = load_selected_candidates(policy)
    base_rows = unique_candidate_rows(selected)
    universe = generate_candidate_universe(base_rows, config, int(config["candidate_search"]["max_candidates"]))
    pd.DataFrame(universe).to_csv(output_dir / "candidate_universe.csv", index=False)

    base_risk = validate_strategy_config(load_strategy_config())
    primary = scenario_spec(config["primary_scenario"]["name"], float(config["primary_scenario"]["fee_rate"]), float(config["primary_scenario"]["slippage_bps"]))
    scenario_rows = list(config["scenario_replay"])
    scenarios = [
        (
            scenario_spec(str(row["name"]), float(row["fee_rate"]), float(row["slippage_bps"])),
            float(row.get("maker_missed_fill_penalty", 0.0)),
        )
        for row in scenario_rows
    ]

    unique_mask_keys = {mask_key(row) for row in universe}
    mask_cache = {}
    for key in unique_mask_keys:
        sample_row = next(row for row in universe if mask_key(row) == key)
        mask_cache[key] = build_entry_mask_for_candidate(market, sample_row, config)

    windows_plus_holdout = windows + [WalkForwardWindow(fold_id=999999, train=holdout_train_index, test=holdout_index)]
    selected_by_fold: dict[int | str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    primary_fold_rows: list[dict[str, Any]] = []
    primary_trades: list[pd.DataFrame] = []
    primary_equities: list[tuple[int, pd.Series]] = []

    for window in windows_plus_holdout:
        is_holdout = window.fold_id == 999999
        fold_label: int | str = "holdout_90d" if is_holdout else int(window.fold_id)
        LOGGER.info("Selecting fold %s", fold_label)
        selected_row, selection_metrics, matrix = select_on_train(
            universe,
            market,
            signal_frame,
            base_risk,
            window.train,
            primary,
            mask_cache,
            config,
        )
        selected_by_fold[fold_label] = selected_row
        matrix = matrix.copy()
        matrix["fold_id"] = fold_label
        selection_rows.extend(matrix.to_dict("records"))
        metrics, trades, equity = evaluate_selected(
            market,
            signal_frame,
            base_risk,
            selected_row,
            window.test,
            primary,
            mask_cache[mask_key(selected_row)],
        )
        row = fold_row(fold_label, primary.name, selected_row, selection_metrics, metrics, window.train, window.test)
        primary_fold_rows.append(row)
        scoped_trades = trades.copy()
        scoped_trades.insert(0, "fold_id", fold_label)
        scoped_trades.insert(0, "scenario", primary.name)
        primary_trades.append(scoped_trades)
        if not is_holdout:
            primary_equities.append((int(window.fold_id), equity))

    selection_matrix = pd.DataFrame(selection_rows)
    selection_matrix.to_csv(output_dir / "selection_matrix.csv", index=False)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    equity_outputs: dict[str, pd.DataFrame] = {}
    scenario_trades: dict[str, list[pd.DataFrame]] = {}
    for scenario, maker_penalty in scenarios:
        LOGGER.info("Replaying scenario %s", scenario.name)
        scenario_fold_rows: list[dict[str, Any]] = []
        scenario_trades[scenario.name] = []
        wf_equities: list[tuple[int, pd.Series]] = []
        all_equities: list[tuple[int, pd.Series]] = []
        holdout_equity = pd.Series(dtype=float)
        holdout_trades = pd.DataFrame()
        for window in windows_plus_holdout:
            is_holdout = window.fold_id == 999999
            fold_label = "holdout_90d" if is_holdout else int(window.fold_id)
            selected_row = selected_by_fold[fold_label]
            metrics, trades, equity = evaluate_selected(
                market,
                signal_frame,
                base_risk,
                selected_row,
                window.test,
                scenario,
                mask_cache[mask_key(selected_row)],
                maker_penalty=maker_penalty,
            )
            row = fold_row(fold_label, scenario.name, selected_row, {}, metrics, window.train, window.test)
            scenario_fold_rows.append(row)
            fold_rows.append(row)
            scoped_trades = trades.copy()
            scoped_trades.insert(0, "fold_id", fold_label)
            scoped_trades.insert(0, "scenario", scenario.name)
            scenario_trades[scenario.name].append(scoped_trades)
            all_equities.append((999999 if is_holdout else int(window.fold_id), equity))
            if is_holdout:
                holdout_equity = equity
                holdout_trades = scoped_trades
            else:
                wf_equities.append((int(window.fold_id), equity))
        frame = pd.DataFrame(scenario_fold_rows)
        wf_frame = frame[~frame["is_holdout"].astype(bool)].copy()
        holdout_frame = frame[frame["is_holdout"].astype(bool)].copy()
        all_trades = pd.concat(scenario_trades[scenario.name], ignore_index=True) if scenario_trades[scenario.name] else pd.DataFrame()
        wf_trades = all_trades[~all_trades["fold_id"].astype(str).eq("holdout_90d")].copy() if not all_trades.empty else pd.DataFrame()
        wf_equity = stitch_oos_equity(wf_equities)["equity"] if wf_equities else pd.Series(dtype=float)
        all_equity = stitch_oos_equity(all_equities)["equity"] if all_equities else pd.Series(dtype=float)
        for scope, scope_frame, scope_equity, scope_trades in [
            ("wf_pre_holdout", wf_frame, wf_equity, wf_trades),
            ("holdout_90d", holdout_frame, holdout_equity, holdout_trades),
            ("combined_oos", frame, all_equity, all_trades),
        ]:
            summary_rows.append(summarize_scope(scenario.name, scope_frame, scope_equity, scope_trades, scope))
            if not scope_equity.empty:
                key = f"{scenario.name}__{scope}"
                equity_frame = pd.DataFrame({"timestamp": scope_equity.index.astype(str), "equity": scope_equity.to_numpy()})
                equity_frame.to_csv(output_dir / "oos_equity" / f"{key}.csv", index=False)
                equity_outputs[key] = equity_frame
        all_trades.to_csv(output_dir / "finalist_trades" / f"{scenario.name}_trades.csv", index=False)

    fold_metrics = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    primary_fold_frame = fold_metrics[(fold_metrics["scenario"].eq(primary.name))].copy()
    primary_trades_frame = pd.concat(scenario_trades.get(primary.name, []), ignore_index=True) if scenario_trades.get(primary.name) else pd.DataFrame()
    cpcv = cpcv_process_summary(primary_fold_frame, config)
    mc = monte_carlo_summary(primary_fold_frame, primary_trades_frame[~primary_trades_frame["fold_id"].astype(str).eq("holdout_90d")].copy() if not primary_trades_frame.empty else pd.DataFrame(), config)
    verdict = final_verdict(summary, cpcv, mc, config)

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    summary[summary["scope"].eq("holdout_90d")].to_csv(output_dir / "holdout_summary.csv", index=False)
    cpcv.to_csv(output_dir / "cpcv_summary.csv", index=False)
    mc.to_csv(output_dir / "monte_carlo_summary.csv", index=False)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    rankings = {
        "best_p10": str(combined.sort_values("monthly_p10", ascending=False).iloc[0]["scenario"]) if not combined.empty else "",
        "best_holdout": str(summary[summary["scope"].eq("holdout_90d")].sort_values("total_return", ascending=False).iloc[0]["scenario"]) if not summary[summary["scope"].eq("holdout_90d")].empty else "",
        "best_compromise": str(combined.assign(compromise=combined["monthly_p10"].astype(float) - combined["orders_per_month"].astype(float) / 10000 + combined["max_drawdown"].astype(float) / 10).sort_values("compromise", ascending=False).iloc[0]["scenario"]) if not combined.empty else "",
        "best_sensitivity": "zero_fee_0p25bps_vs_zero_fee_0p5bps",
    }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "max_folds": max_folds,
        "max_candidates": int(config["candidate_search"]["max_candidates"]),
        "exact_top_n": int(config["candidate_search"]["exact_top_n"]),
        "holdout_start_utc": str(holdout_start),
        "holdout_end_utc": str(holdout_index.max()),
        "fold_count_pre_holdout": len(windows),
        "verdict": verdict,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    payload = {
        "manifest": manifest,
        "data_audit": [data_audit],
        "scenario_summary": summary.to_dict("records"),
        "cpcv_summary": cpcv.to_dict("records"),
        "monte_carlo_summary": mc.to_dict("records"),
        "rankings": rankings,
        "verdict": verdict,
    }
    write_report(output_dir, payload)
    write_figures(output_dir, summary, equity_outputs)
    return {
        "output_dir": str(output_dir),
        "verdict": verdict,
        "top_combined": combined.sort_values("monthly_p10", ascending=False).head(5).to_dict("records"),
        "cpcv": cpcv.describe(include="all").to_dict() if not cpcv.empty else {},
        "monte_carlo": mc.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteration 035 zero-fee P10 optimizer.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--exact-top-n", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        smoke=args.smoke,
        max_folds=args.max_folds,
        max_candidates=args.max_candidates,
        exact_top_n=args.exact_top_n,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
