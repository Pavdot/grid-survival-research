from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
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
from src.research.economy_first_research import prepare_market, summarize_simulations
from src.research.execution_worst_case_audit import (
    AuditScenario,
    make_equity_curve,
    sample_positions_for_scenario,
    simulate_candidate_sample_for_scenario,
)
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    build_side_signal,
    candidate_from_row,
    monthly_return_from_equity,
    planned_exposure,
    risk_for_candidate,
    sizing_sequence,
)
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.research.revalidate_top3_distinct_policies import (
    CANDIDATE_COLUMNS,
    EvaluationContext,
    PolicySpec,
    ScenarioSpec,
    audit_data_files,
    candidate_for_scenario,
    load_selected_candidates,
    prefilter_side_signal_for_audit,
    resolve_path,
    run_candidate_on_split,
    split_pre_holdout,
    unique_candidate_rows,
)
from src.research.walk_forward_martingale_research import WalkForwardWindow, stitch_oos_equity
from src.utils.config_loader import load_strategy_config, load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_cost_budget_hypothesis_ablation_033.yaml"
HYPOTHESIS_COMPONENTS: dict[str, tuple[str, ...]] = {
    "baseline_realistic": (),
    "H1_cost_budget": ("H1",),
    "H2_entry_reduction": ("H2",),
    "H3_wider_tp": ("H3",),
    "H4_reduced_martingale": ("H4",),
    "H5_maker_first_proxy": ("H5",),
    "H1_H2": ("H1", "H2"),
    "H1_H3": ("H1", "H3"),
    "H1_H4": ("H1", "H4"),
    "H2_H3_H4": ("H2", "H3", "H4"),
    "H1_H2_H3_H4": ("H1", "H2", "H3", "H4"),
    "all_plus_maker_proxy": ("H1", "H2", "H3", "H4", "H5"),
}


def _hash_row(row: dict[str, Any]) -> str:
    payload = {key: row.get(key) for key in sorted(set(CANDIDATE_COLUMNS) | set(row.keys()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = _hash_row(row)
        if key in seen:
            continue
        seen.add(key)
        row = dict(row)
        row["candidate_uid"] = key
        out.append(row)
    return out


def _limit_rows(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or len(rows) <= limit:
        return rows
    indexes = np.linspace(0, len(rows) - 1, limit, dtype=int)
    return [rows[int(index)] for index in indexes]


def _row_label(row: dict[str, Any], family: str) -> str:
    return (
        f"{family}__cool{float(row['entry_cooldown_hours']):g}"
        f"__rsi{int(row['rsi_window'])}_{float(row['rsi_low']):g}_{float(row['rsi_high']):g}"
        f"__tp{float(row['take_profit_spacing_multiplier']):g}"
        f"__lvl{int(row['max_levels'])}"
        f"__base{float(row['base_position_size_pct']):g}"
        f"__prog{float(row['progression_multiplier']):g}"
        f"__cap{float(row['max_total_exposure_pct']):g}"
    ).replace(".", "p")


def _base_copy(row: dict[str, Any], family: str, components: tuple[str, ...]) -> dict[str, Any]:
    out = {column: row.get(column) for column in CANDIDATE_COLUMNS if column in row}
    for key in ["range_break_threshold", "range_break_emergency_threshold"]:
        if key in row:
            out[key] = row[key]
    out.update(
        {
            "family": family,
            "hypotheses": "+".join(components) if components else "baseline",
            "cost_budget_enabled": "H1" in components,
            "entry_reduction_enabled": "H2" in components,
            "wider_tp_enabled": "H3" in components,
            "reduced_martingale_enabled": "H4" in components,
            "maker_proxy_enabled": "H5" in components,
            "max_grids_per_month": np.nan,
            "min_expected_net_tp_to_roundtrip_cost": np.nan,
            "maker_missed_fill_penalty": 0.0,
        }
    )
    return out


def _expand_h2(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    h2 = config["hypotheses"]["H2_entry_reduction"]
    expanded: list[dict[str, Any]] = []
    for row in rows:
        for cooldown in h2["cooldown_hours"]:
            for low, high in h2["rsi_threshold_pairs"]:
                for max_grids in h2["max_grids_per_month"]:
                    out = dict(row)
                    out["entry_cooldown_hours"] = float(cooldown)
                    out["rsi_low"] = float(low)
                    out["rsi_high"] = float(high)
                    out["max_grids_per_month"] = float(max_grids)
                    expanded.append(out)
    return expanded


def _expand_h3(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    h3 = config["hypotheses"]["H3_wider_tp"]
    expanded: list[dict[str, Any]] = []
    for row in rows:
        for tp in h3["take_profit_spacing_multipliers"]:
            out = dict(row)
            out["take_profit_spacing_multiplier"] = float(tp)
            out["min_expected_net_tp_to_roundtrip_cost"] = float(h3["min_expected_net_tp_to_roundtrip_cost"])
            expanded.append(out)
    return expanded


def _expand_h4(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    h4 = config["hypotheses"]["H4_reduced_martingale"]
    safety = float(h4.get("base_size_safety", 0.95))
    expanded: list[dict[str, Any]] = []
    for row in rows:
        for cap in h4["max_notional_caps"]:
            for levels in h4["max_levels"]:
                for progression in h4["progression_multipliers"]:
                    exposure_units = sum(float(progression) ** level for level in range(int(levels)))
                    capped_base = float(cap) / exposure_units * safety
                    out = dict(row)
                    out["max_total_exposure_pct"] = float(cap)
                    out["max_levels"] = int(levels)
                    out["progression_multiplier"] = float(progression)
                    out["base_position_size_pct"] = min(float(row["base_position_size_pct"]), capped_base)
                    expanded.append(out)
    return expanded


def _expand_h5(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    h5 = config["hypotheses"]["H5_maker_first_proxy"]
    expanded: list[dict[str, Any]] = []
    for row in rows:
        for penalty in h5["missed_fill_penalties"]:
            out = dict(row)
            out["fee_rate"] = float(h5["maker_fee_rate"])
            out["slippage_bps"] = float(h5["maker_slippage_bps"])
            out["maker_missed_fill_penalty"] = float(penalty)
            expanded.append(out)
    return expanded


def generate_family_universe(
    base_rows: list[dict[str, Any]],
    family: str,
    config: dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if family not in HYPOTHESIS_COMPONENTS:
        raise ValueError(f"Unknown hypothesis family: {family}")
    components = HYPOTHESIS_COMPONENTS[family]
    rows = [_base_copy(row, family, components) for row in base_rows]
    if "H2" in components:
        rows = _expand_h2(rows, config)
    if "H3" in components:
        rows = _expand_h3(rows, config)
    if "H4" in components:
        rows = _expand_h4(rows, config)
    if "H5" in components:
        rows = _expand_h5(rows, config)
    for row in rows:
        row["name"] = _row_label(row, family)
    return _limit_rows(_dedupe_rows(rows), limit)


def scenario_for_row(base_scenario: ScenarioSpec, row: dict[str, Any]) -> ScenarioSpec:
    if bool(row.get("maker_proxy_enabled", False)):
        return ScenarioSpec(
            name=base_scenario.name,
            engine=base_scenario.engine,
            entry_execution_mode=base_scenario.entry_execution_mode,
            signal_lag_bars=base_scenario.signal_lag_bars,
            mask_lag_bars=base_scenario.mask_lag_bars,
            fee_rate=float(row["fee_rate"]),
            slippage_bps=float(row["slippage_bps"]),
        )
    return base_scenario


def apply_maker_proxy(trades: pd.DataFrame, penalty: float) -> pd.DataFrame:
    if trades.empty or penalty <= 0:
        out = trades.copy()
        out["maker_missed_fill_penalty"] = 0.0
        return out
    out = trades.copy()
    positive = out["realized_pnl"].astype(float).clip(lower=0.0)
    out["maker_missed_fill_penalty"] = positive * float(penalty)
    out["realized_pnl"] = out["realized_pnl"].astype(float) - out["maker_missed_fill_penalty"]
    return out


def recompute_metrics(
    market_index: pd.Index,
    trades: pd.DataFrame,
    split: str,
    candidate: MonthlyMartingaleCandidate,
) -> tuple[dict[str, Any], pd.Series]:
    equity = make_equity_curve(market_index, trades)
    metrics = calculate_metrics(equity, trades if not trades.empty else None)
    if not trades.empty:
        metrics.update(summarize_simulations(trades, baseline_grids=len(trades)))
    else:
        metrics.update(
            {
                "number_of_grids": 0,
                "realized_pnl": 0.0,
                "profit_factor": 0.0,
                "fees_paid": 0.0,
                "slippage_paid": 0.0,
                "number_of_forced_exits": 0,
            }
        )
    metrics.update(asdict(candidate))
    metrics["split"] = split
    metrics["monthly_return"] = monthly_return_from_equity(equity)
    metrics["equity_ruined"] = bool((equity <= 0).any() or float(metrics.get("max_drawdown", 0.0)) <= -1.0)
    return metrics, equity


def order_count(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    return int(trades["number_of_levels_filled"].fillna(0).astype(int).sum() + len(trades))


def notional_turnover_proxy(trades: pd.DataFrame) -> float:
    if trades.empty or "max_exposure_pct" not in trades:
        return 0.0
    return float(2.0 * trades["max_exposure_pct"].fillna(0.0).astype(float).sum())


def duration_months(index: pd.Index) -> float:
    if len(index) < 2:
        return 1.0
    days = max((pd.Timestamp(index.max()) - pd.Timestamp(index.min())) / pd.Timedelta(days=1), 1 / 24)
    return float(days / 30.4375)


def expected_tp_cost_ratio(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    cost = (trades.get("fees_paid", 0.0).astype(float) + trades.get("slippage_paid", 0.0).astype(float)).mean()
    positive = trades.loc[trades["realized_pnl"].astype(float) > 0, "realized_pnl"].astype(float)
    if positive.empty or cost <= 0:
        return 0.0 if positive.empty else np.inf
    return float(positive.mean() / cost)


def enrich_cost_metrics(metrics: dict[str, Any], trades: pd.DataFrame, index: pd.Index, account_equity: float) -> dict[str, Any]:
    months = duration_months(index)
    grids = int(metrics.get("number_of_grids", 0))
    orders = order_count(trades)
    fees = float(metrics.get("fees_paid", 0.0))
    slip = float(metrics.get("slippage_paid", 0.0))
    total_cost = fees + slip
    turnover = notional_turnover_proxy(trades)
    max_exposure = float(trades["max_exposure_pct"].max()) if not trades.empty and "max_exposure_pct" in trades else 0.0
    metrics.update(
        {
            "grids_per_month": grids / months,
            "orders": orders,
            "orders_per_month": orders / months,
            "notional_turnover_per_month": turnover / months,
            "fees_per_month": fees / months,
            "slippage_per_month": slip / months,
            "cost_total": total_cost,
            "cost_total_per_month": total_cost / months,
            "net_pnl": float(metrics.get("realized_pnl", 0.0)),
            "net_pnl_per_order": float(metrics.get("realized_pnl", 0.0)) / max(orders, 1),
            "expectancy_per_grid": float(metrics.get("realized_pnl", 0.0)) / max(grids, 1),
            "expected_net_tp_to_roundtrip_cost": expected_tp_cost_ratio(trades),
            "max_notional": max_exposure * account_equity,
            "effective_leverage": max_exposure,
            "max_exposure_pct": max_exposure,
        }
    )
    return metrics


def constraint_audit(row: dict[str, Any], metrics: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    reasons: list[str] = []
    if bool(row.get("cost_budget_enabled", False)):
        h1 = config["hypotheses"]["H1_cost_budget"]
        if float(metrics.get("cost_total_per_month", 0.0)) > float(h1["max_cost_pct_per_month"]):
            reasons.append("cost_budget")
        if float(metrics.get("notional_turnover_per_month", 0.0)) > float(h1["max_notional_turnover_per_month"]):
            reasons.append("turnover_budget")
    if bool(row.get("entry_reduction_enabled", False)) and pd.notna(row.get("max_grids_per_month")):
        if float(metrics.get("grids_per_month", 0.0)) > float(row["max_grids_per_month"]):
            reasons.append("max_grids_per_month")
    if bool(row.get("wider_tp_enabled", False)) and pd.notna(row.get("min_expected_net_tp_to_roundtrip_cost")):
        if float(metrics.get("expected_net_tp_to_roundtrip_cost", 0.0)) < float(row["min_expected_net_tp_to_roundtrip_cost"]):
            reasons.append("min_net_tp_cost_ratio")
    if bool(row.get("reduced_martingale_enabled", False)):
        candidate = candidate_from_row(row)
        if planned_exposure(candidate) > float(row["max_total_exposure_pct"]) + 1e-9:
            reasons.append("planned_exposure_above_cap")
    return not reasons, "|".join(reasons) if reasons else "pass"


def select_candidate_on_train(
    universe: list[dict[str, Any]],
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    policy: PolicySpec,
    train_index: pd.Index,
    base_scenario: ScenarioSpec,
    context: EvaluationContext,
    config: dict[str, Any],
    account_equity: float,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    stride = int(config["search"]["search_entry_stride_bars"])
    max_positions = int(config["search"]["max_sample_positions_per_candidate"])
    for index, row in enumerate(universe):
        scenario = scenario_for_row(base_scenario, row)
        candidate = candidate_for_scenario(row, scenario)
        risk = risk_for_candidate(base_risk, candidate)
        side_signal = build_side_signal(market, signal_frame, candidate)
        entry_mask = context.entry_mask.astype(bool).shift(int(scenario.mask_lag_bars)).fillna(False).astype(bool)
        shifted_signal = prefilter_side_signal_for_audit(side_signal, market.index, entry_mask, scenario)
        audit_scenario = AuditScenario(
            name=scenario.name,
            variant=policy.source_variant,
            selection_policy="cost_budget_hypothesis_033",
            entry_execution_mode=scenario.entry_execution_mode,
        )
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
            "selection",
            audit_scenario,
        )
        sample = apply_maker_proxy(sample, float(row.get("maker_missed_fill_penalty", 0.0)))
        metrics = summarize_simulations(sample, baseline_grids=len(sample))
        metrics["monthly_return"] = float(metrics.get("realized_pnl", 0.0))
        metrics = enrich_cost_metrics(metrics, sample, train_index, account_equity)
        passed, reasons = constraint_audit(row, metrics, config)
        metrics.update(
            {
                "candidate_row_index": index,
                "family": row["family"],
                "candidate_uid": row["candidate_uid"],
                "selected_name": row["name"],
                "sample_positions": len(positions),
                "selection_constraint_pass": bool(passed),
                "selection_constraint_reasons": reasons,
                "selection_uses_drawdown": False,
                "selected_from_validation_only": True,
            }
        )
        rows.append(metrics)
    if not rows:
        raise ValueError("No train selection rows generated")
    frame = pd.DataFrame(rows)
    frame["constraint_rank"] = (~frame["selection_constraint_pass"].astype(bool)).astype(int)
    frame = frame.sort_values(
        [
            "constraint_rank",
            "monthly_return",
            "net_pnl",
            "profit_factor",
            "cost_total_per_month",
            "number_of_forced_exits",
        ],
        ascending=[True, False, False, False, True, True],
    )
    selected = frame.iloc[0].to_dict()
    row_index = int(selected["candidate_row_index"])
    return universe[row_index], selected, frame


def evaluate_exact(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    policy: PolicySpec,
    row: dict[str, Any],
    split_index: pd.Index,
    split_name: str,
    base_scenario: ScenarioSpec,
    context: EvaluationContext,
    account_equity: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    scenario = scenario_for_row(base_scenario, row)
    metrics, trades, equity = run_candidate_on_split(
        market,
        signal_frame,
        base_risk,
        policy,
        row,
        split_index,
        split_name,
        scenario,
        context,
    )
    penalty = float(row.get("maker_missed_fill_penalty", 0.0))
    if penalty > 0:
        split_frame = market.loc[split_index]
        trades = apply_maker_proxy(trades, penalty)
        candidate = candidate_for_scenario(row, scenario)
        metrics, equity = recompute_metrics(split_frame.index, trades, split_name, candidate)
        metrics.update({"policy": policy.name, "scenario": base_scenario.name})
    metrics = enrich_cost_metrics(metrics, trades, split_index, account_equity)
    metrics.update(
        {
            "family": row["family"],
            "candidate_uid": row["candidate_uid"],
            "selected_name": row["name"],
            "maker_missed_fill_penalty": penalty,
            "fee_rate_applied": float(candidate_for_scenario(row, scenario).fee_rate),
            "slippage_bps_applied": float(candidate_for_scenario(row, scenario).slippage_bps),
        }
    )
    return metrics, trades, equity


def fold_metrics_row(
    fold_id: int | str,
    train_index: pd.Index,
    test_index: pd.Index,
    selected_row: dict[str, Any],
    selection_metrics: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "family": selected_row["family"],
        "candidate_uid": selected_row["candidate_uid"],
        "selected_name": selected_row["name"],
        "train_start": train_index.min() if len(train_index) else pd.NaT,
        "train_end": train_index.max() if len(train_index) else pd.NaT,
        "test_start": test_index.min() if len(test_index) else pd.NaT,
        "test_end": test_index.max() if len(test_index) else pd.NaT,
        "is_holdout": str(fold_id) == "holdout_90d",
        "selection_monthly_return": float(selection_metrics.get("monthly_return", np.nan)),
        "selection_constraint_pass": bool(selection_metrics.get("selection_constraint_pass", False)),
        "selection_constraint_reasons": selection_metrics.get("selection_constraint_reasons", ""),
        "total_return": float(metrics.get("total_return", 0.0)),
        "monthly_return": float(metrics.get("monthly_return", 0.0)),
        "annualized_return": float(metrics.get("annualized_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "equity_ruined": bool(metrics.get("equity_ruined", False)),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "number_of_grids": int(metrics.get("number_of_grids", 0)),
        "orders": int(metrics.get("orders", 0)),
        "fees_paid": float(metrics.get("fees_paid", 0.0)),
        "slippage_paid": float(metrics.get("slippage_paid", 0.0)),
        "cost_total": float(metrics.get("cost_total", 0.0)),
        "net_pnl": float(metrics.get("net_pnl", 0.0)),
        "grids_per_month": float(metrics.get("grids_per_month", 0.0)),
        "orders_per_month": float(metrics.get("orders_per_month", 0.0)),
        "notional_turnover_per_month": float(metrics.get("notional_turnover_per_month", 0.0)),
        "cost_total_per_month": float(metrics.get("cost_total_per_month", 0.0)),
        "net_pnl_per_order": float(metrics.get("net_pnl_per_order", 0.0)),
        "expectancy_per_grid": float(metrics.get("expectancy_per_grid", 0.0)),
        "expected_net_tp_to_roundtrip_cost": float(metrics.get("expected_net_tp_to_roundtrip_cost", 0.0)),
        "max_notional": float(metrics.get("max_notional", 0.0)),
        "effective_leverage": float(metrics.get("effective_leverage", 0.0)),
        "fee_rate_applied": float(metrics.get("fee_rate_applied", np.nan)),
        "slippage_bps_applied": float(metrics.get("slippage_bps_applied", np.nan)),
        "maker_missed_fill_penalty": float(metrics.get("maker_missed_fill_penalty", 0.0)),
    }


def summarize_scope(fold_metrics: pd.DataFrame, equity: pd.Series, trades: pd.DataFrame, scope: str, family: str) -> dict[str, Any]:
    metrics = calculate_metrics(equity, trades if not trades.empty else None)
    if not trades.empty:
        metrics.update(summarize_simulations(trades, baseline_grids=len(trades)))
    else:
        metrics.update({"number_of_grids": 0, "realized_pnl": 0.0, "profit_factor": 0.0, "fees_paid": 0.0, "slippage_paid": 0.0})
    fold_returns = fold_metrics["monthly_return"].astype(float) if not fold_metrics.empty else pd.Series(dtype=float)
    return {
        "scope": scope,
        "family": family,
        "fold_count": int(len(fold_metrics)),
        "total_return": float(metrics.get("total_return", 0.0)),
        "monthly_return": float(monthly_return_from_equity(equity)),
        "annualized_return": float(metrics.get("annualized_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "equity_ruined": bool((equity <= 0).any() or float(metrics.get("max_drawdown", 0.0)) <= -1.0 or fold_metrics.get("equity_ruined", pd.Series(dtype=bool)).astype(bool).any()),
        "positive_fold_rate": float((fold_metrics["total_return"].astype(float) > 0).mean()) if not fold_metrics.empty else 0.0,
        "fold_rate_above_3pct": float(fold_returns.ge(0.03).mean()) if not fold_returns.empty else 0.0,
        "fold_rate_above_8pct": float(fold_returns.ge(0.08).mean()) if not fold_returns.empty else 0.0,
        "fold_rate_above_12pct": float(fold_returns.ge(0.12).mean()) if not fold_returns.empty else 0.0,
        "fold_rate_above_20pct": float(fold_returns.ge(0.20).mean()) if not fold_returns.empty else 0.0,
        "monthly_p10": float(fold_returns.quantile(0.10)) if not fold_returns.empty else 0.0,
        "worst_fold_monthly": float(fold_returns.min()) if not fold_returns.empty else 0.0,
        "best_fold_monthly": float(fold_returns.max()) if not fold_returns.empty else 0.0,
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "number_of_grids": int(metrics.get("number_of_grids", 0)),
        "orders": order_count(trades),
        "fees_paid": float(metrics.get("fees_paid", 0.0)),
        "slippage_paid": float(metrics.get("slippage_paid", 0.0)),
        "cost_total": float(metrics.get("fees_paid", 0.0)) + float(metrics.get("slippage_paid", 0.0)),
        "net_pnl": float(metrics.get("realized_pnl", 0.0)),
        "grids_per_month": float(fold_metrics["grids_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "orders_per_month": float(fold_metrics["orders_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "notional_turnover_per_month": float(fold_metrics["notional_turnover_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "cost_total_per_month": float(fold_metrics["cost_total_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "net_pnl_per_order": float(metrics.get("realized_pnl", 0.0)) / max(order_count(trades), 1),
        "expectancy_per_grid": float(metrics.get("realized_pnl", 0.0)) / max(int(metrics.get("number_of_grids", 0)), 1),
        "max_notional": float(fold_metrics["max_notional"].max()) if not fold_metrics.empty else 0.0,
        "effective_leverage": float(fold_metrics["effective_leverage"].max()) if not fold_metrics.empty else 0.0,
    }


def classify_hypothesis(summary: pd.DataFrame, family: str, config: dict[str, Any]) -> str:
    row = summary[(summary["family"].eq(family)) & (summary["scope"].eq("combined_oos"))]
    base = summary[(summary["family"].eq("baseline_realistic")) & (summary["scope"].eq("combined_oos"))]
    holdout = summary[(summary["family"].eq(family)) & (summary["scope"].eq("holdout_90d"))]
    if row.empty:
        return "harmful"
    current = row.iloc[0]
    baseline = base.iloc[0] if not base.empty else current
    hold = holdout.iloc[0] if not holdout.empty else current
    target = config["target"]
    if (
        float(current["monthly_p10"]) > float(target["candidate_v3_monthly_p10"])
        and bool(current["equity_ruined"]) is False
        and float(current["max_drawdown"]) > float(target["max_drawdown"])
        and float(current["cost_total_per_month"]) <= float(target["max_cost_pct_per_month"])
        and float(hold["max_drawdown"]) > float(target["max_drawdown"])
        and float(hold["total_return"]) > float(target["holdout_min_total_return"])
    ):
        return "candidate_v3"
    monthly_improved = float(current["monthly_return"]) > float(baseline["monthly_return"])
    cost_reduced = float(current["cost_total_per_month"]) < float(baseline["cost_total_per_month"])
    trades_alive = float(current["grids_per_month"]) >= float(target["min_grids_per_month"])
    if monthly_improved and cost_reduced and trades_alive:
        return "helps"
    if cost_reduced and trades_alive:
        return "insufficient"
    return "harmful"


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    equity_outputs: dict[str, pd.DataFrame],
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(payload["hypothesis_summary"])
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    if not combined.empty:
        ordered = combined.sort_values("monthly_return", ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(ordered["family"], ordered["monthly_return"].astype(float) * 100)
        plt.xticks(rotation=75, ha="right", fontsize=8)
        plt.ylabel("Monthly return (%)")
        plt.title("Iteration 033 Realistic Net Monthly Return")
        plt.tight_layout()
        plt.savefig(figures / "net_monthly_return_by_hypothesis.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 6))
        width = 0.35
        x = np.arange(len(ordered))
        plt.bar(x - width / 2, ordered["orders_per_month"].astype(float), width=width, label="orders/month")
        plt.bar(x + width / 2, ordered["cost_total_per_month"].astype(float) * 100, width=width, label="cost/month (%)")
        plt.xticks(x, ordered["family"], rotation=75, ha="right", fontsize=8)
        plt.title("Turnover and Cost by Hypothesis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "cost_turnover_by_hypothesis.png", dpi=150)
        plt.close()

    plt.figure(figsize=(12, 6))
    for key, frame in equity_outputs.items():
        if frame.empty or "combined_oos" not in key:
            continue
        plt.plot(pd.to_datetime(frame["timestamp"], utc=True), frame["equity"], label=key.replace("__combined_oos", ""))
    plt.title("Iteration 033 Combined OOS Equity")
    plt.ylabel("Equity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures / "combined_oos_equity.png", dpi=150)
    plt.close()

    lines = [
        "# Iteration 033 - Cost Budget Hypothesis Ablation",
        "",
        "## Verdicts",
        markdown_table(pd.DataFrame(payload["best_by_hypothesis"])),
        "",
        "## Net Realistic Ranking",
        markdown_table(pd.DataFrame(payload["ranking_net_realistic"]).head(12)),
        "",
        "## Cost Efficiency Ranking",
        markdown_table(pd.DataFrame(payload["ranking_cost_efficiency"]).head(12)),
        "",
        "## Combined OOS Summary",
        markdown_table(
            combined[
                [
                    "family",
                    "decision",
                    "monthly_return",
                    "monthly_p10",
                    "max_drawdown",
                    "equity_ruined",
                    "positive_fold_rate",
                    "number_of_grids",
                    "orders_per_month",
                    "cost_total_per_month",
                    "net_pnl_per_order",
                    "effective_leverage",
                ]
            ]
        )
        if not combined.empty
        else "No summary rows.",
        "",
        "## Notes",
        "- Base signal/filter is `fundamental_trend_escape_v2`.",
        "- Scenario is next-bar open, signal/mask lag 1 bar, fee 0.0004 and slippage 2 bps unless the maker proxy explicitly lowers costs and applies a missed-fill penalty.",
        "- Holdout 90d is selected from the preceding train window only and is not used for fold selection.",
        "- Maker-first rows are proxies only; real L2 validation still belongs to the microstructure iterations.",
    ]
    (output_dir / "iteration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_iteration(
    config_path: str = DEFAULT_CONFIG,
    smoke: bool = False,
    max_folds: int | None = None,
    max_candidates_per_family: int | None = None,
    max_sample_positions: int | None = None,
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if max_candidates_per_family is not None:
        if max_candidates_per_family <= 0:
            raise ValueError("max_candidates_per_family must be positive")
        config["search"]["max_universe_per_family"] = int(max_candidates_per_family)
        config["search"]["smoke_max_universe_per_family"] = int(max_candidates_per_family)
    if max_sample_positions is not None:
        if max_sample_positions <= 0:
            raise ValueError("max_sample_positions must be positive")
        config["search"]["max_sample_positions_per_candidate"] = int(max_sample_positions)
    if smoke and max_folds is None:
        max_folds = 2
    if smoke:
        config["walk_forward"]["train_days"] = min(float(config["walk_forward"]["train_days"]), 30.0)
        config["walk_forward"]["test_days"] = min(float(config["walk_forward"]["test_days"]), 5.0)
        config["walk_forward"]["step_days"] = min(float(config["walk_forward"]["step_days"]), 5.0)
        config["walk_forward"]["holdout_days"] = min(float(config["walk_forward"]["holdout_days"]), 15.0)
        config["search"]["max_sample_positions_per_candidate"] = min(int(config["search"]["max_sample_positions_per_candidate"]), 3)
        config["search"]["smoke_max_universe_per_family"] = 1

    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = resolve_path(config["iteration"]["output_root"])
    output_dir = output_root / f"hypothesis_ablation_033_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "oos_equity").mkdir(parents=True, exist_ok=True)

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
    signal_frame = signal_frame.reindex(signal_frame.index.intersection(signal_frame.index))
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

    base_risk = validate_strategy_config(load_strategy_config())
    policy = PolicySpec(
        name=str(config["source_policy"]["name"]),
        source_iteration_dir=resolve_path(config["source_policy"]["source_iteration_dir"]),
        source_variant=str(config["source_policy"]["source_variant"]),
        policy_kind=str(config["source_policy"]["policy_kind"]),
    )
    scenario_cfg = config["scenario"]
    base_scenario = ScenarioSpec(
        name=str(scenario_cfg["name"]),
        engine=str(scenario_cfg["engine"]),
        entry_execution_mode=str(scenario_cfg["entry_execution_mode"]),
        signal_lag_bars=int(scenario_cfg["signal_lag_bars"]),
        mask_lag_bars=int(scenario_cfg["mask_lag_bars"]),
        fee_rate=float(scenario_cfg["fee_rate"]),
        slippage_bps=float(scenario_cfg["slippage_bps"]),
    )
    account_equity = float(config["reporting"]["account_equity_usdt"])

    selected = load_selected_candidates(policy)
    base_rows = unique_candidate_rows(selected)
    if smoke:
        base_rows = base_rows[: min(1, len(base_rows))]

    _events, _event_windows, blackout_masks = build_blackout_bundle(market.index, config)
    trend_components = build_trend_escape_components(market, config)
    fundamental_entry = fundamental_trend_mask(trend_components["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)
    context = EvaluationContext(fundamental_entry, None, None)

    max_universe = int(config["search"]["smoke_max_universe_per_family"] if smoke else config["search"]["max_universe_per_family"])
    families = list(config["families"])
    universe_rows: list[dict[str, Any]] = []
    universe_by_family: dict[str, list[dict[str, Any]]] = {}
    for family in families:
        rows = generate_family_universe(base_rows, family, config, max_universe)
        universe_by_family[family] = rows
        universe_rows.extend(rows)
    pd.DataFrame(universe_rows).to_csv(output_dir / "hypothesis_candidate_universe.csv", index=False)

    windows_plus_holdout = windows + [WalkForwardWindow(fold_id=999999, train=holdout_train_index, test=holdout_index)]
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    combined_trades_by_family: dict[str, list[pd.DataFrame]] = {family: [] for family in families}
    combined_equities_by_family: dict[str, list[tuple[int, pd.Series]]] = {family: [] for family in families}
    wf_trades_by_family: dict[str, list[pd.DataFrame]] = {family: [] for family in families}
    wf_equities_by_family: dict[str, list[tuple[int, pd.Series]]] = {family: [] for family in families}
    holdout_trades_by_family: dict[str, pd.DataFrame] = {}
    holdout_equity_by_family: dict[str, pd.Series] = {}
    equity_outputs: dict[str, pd.DataFrame] = {}

    for family in families:
        LOGGER.info("Evaluating hypothesis family %s with %s candidates", family, len(universe_by_family[family]))
        for window in windows_plus_holdout:
            is_holdout = window.fold_id == 999999
            fold_label: int | str = "holdout_90d" if is_holdout else int(window.fold_id)
            selected_row, selection_metrics, selection_frame = select_candidate_on_train(
                universe_by_family[family],
                market,
                signal_frame,
                base_risk,
                policy,
                window.train,
                base_scenario,
                context,
                config,
                account_equity,
            )
            selection_frame = selection_frame.copy()
            selection_frame["fold_id"] = fold_label
            selection_rows.extend(selection_frame.to_dict("records"))
            metrics, trades, equity = evaluate_exact(
                market,
                signal_frame,
                base_risk,
                policy,
                selected_row,
                window.test,
                "holdout" if is_holdout else "test",
                base_scenario,
                context,
                account_equity,
            )
            row = fold_metrics_row(fold_label, window.train, window.test, selected_row, selection_metrics, metrics)
            fold_rows.append(row)
            trades = trades.copy()
            trades.insert(0, "fold_id", fold_label)
            trades.insert(0, "family", family)
            combined_trades_by_family[family].append(trades)
            combined_equities_by_family[family].append((999999 if is_holdout else int(window.fold_id), equity))
            if is_holdout:
                holdout_trades_by_family[family] = trades
                holdout_equity_by_family[family] = equity
            else:
                wf_trades_by_family[family].append(trades)
                wf_equities_by_family[family].append((int(window.fold_id), equity))

    fold_metrics = pd.DataFrame(fold_rows)
    selection_frame = pd.DataFrame(selection_rows)
    for family in families:
        wf_fold = fold_metrics[(fold_metrics["family"].eq(family)) & (~fold_metrics["is_holdout"].astype(bool))].copy()
        holdout_fold = fold_metrics[(fold_metrics["family"].eq(family)) & (fold_metrics["is_holdout"].astype(bool))].copy()
        all_fold = fold_metrics[fold_metrics["family"].eq(family)].copy()
        wf_trades = pd.concat(wf_trades_by_family[family], ignore_index=True) if wf_trades_by_family[family] else pd.DataFrame()
        all_trades = pd.concat(combined_trades_by_family[family], ignore_index=True) if combined_trades_by_family[family] else pd.DataFrame()
        wf_equity = stitch_oos_equity(wf_equities_by_family[family])["equity"] if wf_equities_by_family[family] else pd.Series(dtype=float)
        all_equity = stitch_oos_equity(combined_equities_by_family[family])["equity"] if combined_equities_by_family[family] else pd.Series(dtype=float)
        holdout_equity = holdout_equity_by_family.get(family, pd.Series(dtype=float))
        holdout_trades = holdout_trades_by_family.get(family, pd.DataFrame())
        scopes = [
            ("wf_pre_holdout", wf_fold, wf_equity, wf_trades),
            ("holdout_90d", holdout_fold, holdout_equity, holdout_trades),
            ("combined_oos", all_fold, all_equity, all_trades),
        ]
        for scope, frame, equity, trades in scopes:
            summary_rows.append(summarize_scope(frame, equity, trades, scope, family))
            if not equity.empty:
                key = f"{family}__{scope}"
                equity_frame = pd.DataFrame({"timestamp": equity.index.astype(str), "equity": equity.to_numpy()})
                equity_frame.to_csv(output_dir / "oos_equity" / f"{key}.csv", index=False)
                equity_outputs[key] = equity_frame

    summary = pd.DataFrame(summary_rows)
    summary["decision"] = [classify_hypothesis(summary, str(row["family"]), config) for _, row in summary.iterrows()]
    holdout_summary = summary[summary["scope"].eq("holdout_90d")].copy()
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    attribution = combined.merge(
        combined[combined["family"].eq("baseline_realistic")][
            [
                "monthly_return",
                "cost_total_per_month",
                "orders_per_month",
                "notional_turnover_per_month",
                "net_pnl_per_order",
            ]
        ].rename(
            columns={
                "monthly_return": "baseline_monthly_return",
                "cost_total_per_month": "baseline_cost_total_per_month",
                "orders_per_month": "baseline_orders_per_month",
                "notional_turnover_per_month": "baseline_notional_turnover_per_month",
                "net_pnl_per_order": "baseline_net_pnl_per_order",
            }
        ),
        how="cross",
    )
    for column in ["monthly_return", "cost_total_per_month", "orders_per_month", "notional_turnover_per_month", "net_pnl_per_order"]:
        attribution[f"delta_{column}"] = attribution[column] - attribution[f"baseline_{column}"]

    best_by_hypothesis = combined.sort_values(["family", "monthly_return"], ascending=[True, False]).groupby("family", as_index=False).head(1)
    best_combined = combined[combined["decision"].eq("candidate_v3")].sort_values(["monthly_p10", "monthly_return"], ascending=[False, False])
    ranking_net = combined.sort_values(["monthly_return", "net_pnl", "profit_factor"], ascending=[False, False, False])
    ranking_cost = combined.sort_values(["net_pnl_per_order", "cost_total_per_month", "notional_turnover_per_month"], ascending=[False, True, True])
    cost_breakdown = combined[
        [
            "family",
            "number_of_grids",
            "orders",
            "grids_per_month",
            "orders_per_month",
            "notional_turnover_per_month",
            "fees_paid",
            "slippage_paid",
            "cost_total",
            "cost_total_per_month",
            "net_pnl_per_order",
            "expectancy_per_grid",
        ]
    ].copy()

    fold_metrics.to_csv(output_dir / "hypothesis_fold_metrics.csv", index=False)
    selection_frame.to_csv(output_dir / "selection_sample_metrics.csv", index=False)
    summary.to_csv(output_dir / "hypothesis_summary.csv", index=False)
    holdout_summary.to_csv(output_dir / "hypothesis_holdout_summary.csv", index=False)
    attribution.to_csv(output_dir / "hypothesis_attribution.csv", index=False)
    cost_breakdown.to_csv(output_dir / "cost_turnover_breakdown.csv", index=False)
    best_by_hypothesis.to_csv(output_dir / "best_by_hypothesis.csv", index=False)
    best_combined.to_csv(output_dir / "best_combined_candidates.csv", index=False)
    ranking_net.to_csv(output_dir / "ranking_net_realistic.csv", index=False)
    ranking_cost.to_csv(output_dir / "ranking_cost_efficiency.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "max_folds": max_folds,
        "families": families,
        "source_policy": config["source_policy"],
        "scenario": config["scenario"],
        "holdout_start_utc": str(holdout_start),
        "holdout_end_utc": str(holdout_index.max()),
        "fold_count_pre_holdout": len(windows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    payload = {
        "manifest": manifest,
        "data_audit": [data_audit],
        "hypothesis_summary": summary.to_dict("records"),
        "best_by_hypothesis": best_by_hypothesis.to_dict("records"),
        "ranking_net_realistic": ranking_net.to_dict("records"),
        "ranking_cost_efficiency": ranking_cost.to_dict("records"),
    }
    write_outputs(output_dir, payload, equity_outputs)
    return {
        "output_dir": str(output_dir),
        "top_net_realistic": ranking_net.head(3).to_dict("records"),
        "candidate_v3": best_combined.head(3).to_dict("records"),
        "decisions": best_by_hypothesis[["family", "decision", "monthly_return", "monthly_p10", "cost_total_per_month"]].to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteration 033 cost budget hypothesis ablation.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates-per-family", type=int, default=None)
    parser.add_argument("--max-sample-positions", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        smoke=args.smoke,
        max_folds=args.max_folds,
        max_candidates_per_family=args.max_candidates_per_family,
        max_sample_positions=args.max_sample_positions,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
