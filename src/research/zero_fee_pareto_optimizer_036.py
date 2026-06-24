from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.research.revalidate_top3_distinct_policies import (
    PolicySpec,
    audit_data_files,
    load_selected_candidates,
    resolve_path,
    split_pre_holdout,
    unique_candidate_rows,
)
from src.research.walk_forward_martingale_research import WalkForwardWindow, stitch_oos_equity
from src.research.zero_fee_p10_optimizer_035 import (
    build_entry_mask_for_candidate,
    cpcv_process_summary,
    evaluate_selected,
    fold_row,
    generate_candidate_universe,
    internal_train_selection_split,
    mask_key,
    monte_carlo_summary,
    monthly_chunk_returns,
    sample_prune_candidates,
    scenario_spec,
    summarize_scope,
)
from src.utils.config_loader import load_strategy_config, load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_zero_fee_pareto_optimizer_036.yaml"


def add_chunk_stats(metrics: dict[str, Any], equity: pd.Series) -> dict[str, Any]:
    chunks = monthly_chunk_returns(equity)
    metrics["monthly_p10_chunks"] = float(np.quantile(chunks, 0.10))
    metrics["monthly_median_chunks"] = float(np.median(chunks))
    return metrics


def seed_historical_candidates(base_rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    search = config["candidate_search"]
    seeds: list[dict[str, Any]] = []
    for idx, row in enumerate(base_rows):
        out = dict(row)
        out["candidate_uid"] = f"seed_historical_{idx:03d}"
        out["name"] = f"seed_historical_{idx:03d}_{str(row.get('name', 'candidate'))[:80]}"
        out["fee_rate"] = float(config["primary_scenario"]["fee_rate"])
        out["slippage_bps"] = float(config["primary_scenario"]["slippage_bps"])
        out["max_grids_per_month"] = float(max(search["max_grids_per_month"]))
        out["blackout_hours"] = 24
        out["min_severity"] = 4
        out["trend_propagation_bars"] = 36
        out["breakout_atr_buffer"] = 0.25
        out["min_range_expansion_ratio"] = 1.3
        out["pause_after_forced_loss_hours"] = 0.0
        out["rolling_7d_loss_threshold"] = -999.0
        seeds.append(out)
    return seeds


def dedupe_universe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("entry_cooldown_hours"),
            row.get("rsi_low"),
            row.get("rsi_high"),
            row.get("spacing_atr_multiplier"),
            row.get("take_profit_spacing_multiplier"),
            row.get("max_levels"),
            row.get("base_position_size_pct"),
            row.get("progression_multiplier"),
            row.get("max_total_exposure_pct"),
            row.get("max_holding_hours"),
            row.get("max_grids_per_month"),
            row.get("blackout_hours"),
            row.get("min_severity"),
            row.get("trend_propagation_bars"),
            row.get("breakout_atr_buffer"),
            row.get("min_range_expansion_ratio"),
            row.get("pause_after_forced_loss_hours"),
            row.get("rolling_7d_loss_threshold"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def pareto_constraint_audit(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    sel = config["selection"]
    reasons: list[str] = []
    if bool(metrics.get("equity_ruined", False)):
        reasons.append("ruin")
    if float(metrics.get("max_drawdown", 0.0)) <= float(sel["max_drawdown"]):
        reasons.append("drawdown")
    if float(metrics.get("monthly_return", 0.0)) < float(sel["min_monthly_return"]):
        reasons.append("min_monthly_return")
    if float(metrics.get("monthly_median_chunks", metrics.get("monthly_return", 0.0))) < float(sel["min_monthly_median"]):
        reasons.append("min_monthly_median")
    if float(metrics.get("grids_per_month", 0.0)) < float(sel["min_grids_per_month"]):
        reasons.append("min_grids")
    if float(metrics.get("grids_per_month", 0.0)) > float(sel["max_grids_per_month"]):
        reasons.append("max_grids")
    if float(metrics.get("orders_per_month", 0.0)) < float(sel["min_orders_per_month"]):
        reasons.append("min_orders")
    if float(metrics.get("orders_per_month", 0.0)) > float(sel["max_orders_per_month"]):
        reasons.append("max_orders")
    if bool(sel.get("require_net_pnl_per_order_positive", True)) and float(metrics.get("net_pnl_per_order", 0.0)) <= 0:
        reasons.append("net_pnl_per_order")
    if "stability_positive_rate" in metrics and float(metrics["stability_positive_rate"]) < float(sel.get("min_stability_positive_rate", 0.0)):
        reasons.append("stability_positive_rate")
    if "stability_worst_monthly" in metrics and float(metrics["stability_worst_monthly"]) < float(sel.get("min_stability_worst_monthly", -1.0)):
        reasons.append("stability_worst_monthly")
    return not reasons, "|".join(reasons) if reasons else "pass"


def pareto_score(metrics: dict[str, Any], config: dict[str, Any]) -> float:
    sel = config["selection"]
    weights = sel["weights"]
    min_orders = float(sel["min_orders_per_month"])
    max_orders = float(sel["max_orders_per_month"])
    target_orders = (min_orders + max_orders) / 2.0
    order_distance = abs(float(metrics.get("orders_per_month", 0.0)) - target_orders) / max(target_orders, 1.0)
    drawdown = float(metrics.get("max_drawdown", 0.0))
    stability_median = float(metrics.get("stability_monthly_median", metrics.get("monthly_median_chunks", metrics.get("monthly_return", 0.0))))
    stability_p10 = float(metrics.get("stability_monthly_p10", metrics.get("monthly_p10_chunks", 0.0)))
    return float(
        weights["monthly_return"] * float(metrics.get("monthly_return", 0.0))
        + weights["monthly_median"] * float(metrics.get("monthly_median_chunks", metrics.get("monthly_return", 0.0)))
        + weights["monthly_p10"] * float(metrics.get("monthly_p10_chunks", 0.0))
        + weights["net_pnl_per_order"] * float(metrics.get("net_pnl_per_order", 0.0))
        + weights["drawdown"] * drawdown
        + 0.20 * stability_median
        + 0.15 * stability_p10
        - weights["order_distance"] * order_distance
    )


def stability_indexes(train_index: pd.Index, slices: int) -> list[pd.Index]:
    if slices <= 1:
        return [train_index]
    parts = np.array_split(np.asarray(train_index), int(slices))
    return [pd.Index(part) for part in parts if len(part)]


def train_stability_metrics(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    row: dict[str, Any],
    train_index: pd.Index,
    primary: Any,
    mask_cache: dict[tuple[Any, ...], pd.Series],
    config: dict[str, Any],
) -> dict[str, Any]:
    returns: list[float] = []
    ruined = False
    for split_index in stability_indexes(train_index, int(config["selection"].get("stability_slices", 3))):
        metrics, _trades, _equity = evaluate_selected(
            market,
            signal_frame,
            base_risk,
            row,
            split_index,
            primary,
            mask_cache[mask_key(row)],
        )
        returns.append(float(metrics.get("monthly_return", 0.0)))
        ruined = ruined or bool(metrics.get("equity_ruined", False))
    if not returns:
        returns = [0.0]
    values = np.asarray(returns, dtype=float)
    return {
        "stability_monthly_mean": float(values.mean()),
        "stability_monthly_median": float(np.median(values)),
        "stability_monthly_p10": float(np.quantile(values, 0.10)),
        "stability_worst_monthly": float(values.min()),
        "stability_positive_rate": float((values > 0).mean()),
        "stability_ruined": bool(ruined),
    }


def select_on_train_pareto(
    universe: list[dict[str, Any]],
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    train_index: pd.Index,
    primary: Any,
    mask_cache: dict[tuple[Any, ...], pd.Series],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    search_index, selection_index = internal_train_selection_split(
        train_index,
        float(config["walk_forward"]["internal_search_days"]),
        float(config["walk_forward"]["internal_selection_days"]),
    )
    prune = sample_prune_candidates(universe, market, signal_frame, base_risk, search_index, primary, mask_cache, config)
    exact_top_n = int(config["candidate_search"]["exact_top_n"])
    top_indexes = prune.head(exact_top_n)["candidate_row_index"].astype(int).tolist()
    seed_top_n = int(config["candidate_search"].get("seed_exact_top_n", 0))
    if seed_top_n > 0 and "candidate_uid" in prune:
        seed_indexes = (
            prune[prune["candidate_uid"].astype(str).str.startswith("seed_historical_")]
            .head(seed_top_n)["candidate_row_index"]
            .astype(int)
            .tolist()
        )
        top_indexes = list(dict.fromkeys(top_indexes + seed_indexes))
    exact_rows: list[dict[str, Any]] = []
    for row_index in top_indexes:
        candidate_row = universe[int(row_index)]
        metrics, _trades, equity = evaluate_selected(
            market,
            signal_frame,
            base_risk,
            candidate_row,
            selection_index,
            primary,
            mask_cache[mask_key(candidate_row)],
        )
        metrics = add_chunk_stats(metrics, equity)
        metrics.update(
            train_stability_metrics(
                market,
                signal_frame,
                base_risk,
                candidate_row,
                train_index,
                primary,
                mask_cache,
                config,
            )
        )
        passed, reasons = pareto_constraint_audit(metrics, config)
        metrics.update(
            {
                "candidate_row_index": int(row_index),
                "candidate_uid": candidate_row["candidate_uid"],
                "selected_name": candidate_row["name"],
                "stage": "selection_exact_pareto",
                "constraint_pass": bool(passed),
                "constraint_reasons": reasons,
                "pareto_score": pareto_score(metrics, config),
                "selection_uses_drawdown": False,
                "selected_from_train_only": True,
            }
        )
        exact_rows.append(metrics)
    exact = pd.DataFrame(exact_rows)
    if exact.empty:
        raise ValueError("Pareto selection exact stage generated no rows")
    exact["constraint_rank"] = (~exact["constraint_pass"].astype(bool)).astype(int)
    exact = exact.sort_values(
        ["constraint_rank", "pareto_score", "monthly_return", "monthly_median_chunks", "monthly_p10_chunks"],
        ascending=[True, False, False, False, False],
    )
    selected = exact.iloc[0].to_dict()
    matrix = pd.concat([prune, exact], ignore_index=True, sort=False)
    return universe[int(selected["candidate_row_index"])], selected, matrix


def pareto_verdict(summary: pd.DataFrame, cpcv: pd.DataFrame, mc: pd.DataFrame, config: dict[str, Any]) -> str:
    primary = summary[(summary["scope"].eq("combined_oos")) & (summary["scenario"].eq(config["primary_scenario"]["name"]))]
    holdout = summary[(summary["scope"].eq("holdout_90d")) & (summary["scenario"].eq(config["primary_scenario"]["name"]))]
    if primary.empty:
        return "not evaluated"
    row = primary.iloc[0]
    hold = holdout.iloc[0] if not holdout.empty else row
    val = config["validation"]
    if (
        float(row["monthly_return"]) >= float(val["min_interesting_monthly"])
        and float(row["monthly_median"]) >= float(val["min_interesting_median"])
        and float(row["monthly_p10"]) >= float(val["max_p10_floor"])
        and float(hold["total_return"]) >= float(val["holdout_min_total_return"])
        and not bool(row["equity_ruined"])
    ):
        return "pareto candidate - yield preserved"
    if float(row["monthly_return"]) >= float(val["min_interesting_monthly"]) and not bool(row["equity_ruined"]):
        return "high yield but fragile"
    if float(row["monthly_return"]) > 0 and float(hold["total_return"]) >= 0:
        return "positive but too diluted"
    return "not enough edge"


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    summary = pd.DataFrame(payload["scenario_summary"])
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    holdout = summary[summary["scope"].eq("holdout_90d")].copy()
    lines = [
        "# Iteration 036 - Zero-Fee Pareto Optimizer",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        "## Combined OOS",
        markdown_table(
            combined[
                [
                    "scenario",
                    "monthly_return",
                    "monthly_p10",
                    "monthly_median",
                    "max_drawdown",
                    "positive_fold_rate",
                    "orders_per_month",
                    "grids_per_month",
                    "net_pnl_per_order",
                    "effective_leverage",
                ]
            ]
        ),
        "",
        "## Holdout 90d",
        markdown_table(holdout[["scenario", "monthly_return", "total_return", "max_drawdown", "orders_per_month", "net_pnl_per_order"]]),
        "",
        "## CPCV",
        markdown_table(pd.DataFrame(payload["cpcv_summary"])) if payload["cpcv_summary"] else "No CPCV rows.",
        "",
        "## Monte Carlo",
        markdown_table(pd.DataFrame(payload["monte_carlo_summary"])) if payload["monte_carlo_summary"] else "No Monte Carlo rows.",
        "",
        "## Interpretation",
        "- This optimizer intentionally rejects ultra-low-turnover candidates that preserve P10 by killing most trades.",
        "- The target is a Pareto compromise: preserve yield first, then reduce the left tail.",
        "- Scenario selection remains train-only; holdout is not used for candidate choice.",
    ]
    (output_dir / "iteration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output_dir: Path, summary: pd.DataFrame, equity_outputs: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    if not combined.empty:
        labels = combined["scenario"].astype(str)
        x = np.arange(len(combined))
        plt.figure(figsize=(12, 6))
        plt.bar(x - 0.2, combined["monthly_return"].astype(float) * 100, width=0.4, label="Monthly")
        plt.bar(x + 0.2, combined["monthly_p10"].astype(float) * 100, width=0.4, label="P10")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, labels, rotation=70, ha="right", fontsize=8)
        plt.ylabel("Return (%)")
        plt.title("Iteration 036 Monthly Return vs P10")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "monthly_vs_p10_by_scenario.png", dpi=150)
        plt.close()
    plt.figure(figsize=(12, 6))
    for key, frame in equity_outputs.items():
        if "combined_oos" not in key:
            continue
        plt.plot(pd.to_datetime(frame["timestamp"], utc=True), frame["equity"], label=key.replace("__combined_oos", ""))
    plt.title("Iteration 036 Combined OOS Equity")
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
        config["walk_forward"]["train_days"] = 30.0
        config["walk_forward"]["test_days"] = 5.0
        config["walk_forward"]["step_days"] = 5.0
        config["walk_forward"]["holdout_days"] = 15.0
        config["walk_forward"]["internal_search_days"] = 18.0
        config["walk_forward"]["internal_selection_days"] = 10.0
        config["candidate_search"]["max_candidates"] = min(int(config["candidate_search"]["max_candidates"]), 12)
        config["candidate_search"]["exact_top_n"] = min(int(config["candidate_search"]["exact_top_n"]), 4)
        config["candidate_search"]["max_sample_positions_per_candidate"] = 4
        config["scenario_replay"] = config["scenario_replay"][:3]
        config["validation"]["monte_carlo_iterations"] = 200

    output_root = resolve_path(config["iteration"]["output_root"])
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"zero_fee_pareto_optimizer_036_{stamp}"
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
    universe = dedupe_universe(
        seed_historical_candidates(base_rows, config)
        + generate_candidate_universe(base_rows, config, int(config["candidate_search"]["max_candidates"]))
    )
    pd.DataFrame(universe).to_csv(output_dir / "candidate_universe.csv", index=False)

    base_risk = validate_strategy_config(load_strategy_config())
    primary = scenario_spec(config["primary_scenario"]["name"], float(config["primary_scenario"]["fee_rate"]), float(config["primary_scenario"]["slippage_bps"]))
    scenarios = [
        (
            scenario_spec(str(row["name"]), float(row["fee_rate"]), float(row["slippage_bps"])),
            float(row.get("maker_missed_fill_penalty", 0.0)),
        )
        for row in config["scenario_replay"]
    ]

    mask_cache: dict[tuple[Any, ...], pd.Series] = {}
    for key in {mask_key(row) for row in universe}:
        mask_cache[key] = build_entry_mask_for_candidate(market, next(row for row in universe if mask_key(row) == key), config)

    windows_plus_holdout = windows + [WalkForwardWindow(fold_id=999999, train=holdout_train_index, test=holdout_index)]
    selected_by_fold: dict[int | str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    for window in windows_plus_holdout:
        label: int | str = "holdout_90d" if window.fold_id == 999999 else int(window.fold_id)
        LOGGER.info("Pareto selecting fold %s", label)
        selected_row, selection_metrics, matrix = select_on_train_pareto(
            universe,
            market,
            signal_frame,
            base_risk,
            window.train,
            primary,
            mask_cache,
            config,
        )
        selected_by_fold[label] = selected_row
        matrix = matrix.copy()
        matrix["fold_id"] = label
        selection_rows.extend(matrix.to_dict("records"))
    pd.DataFrame(selection_rows).to_csv(output_dir / "selection_matrix.csv", index=False)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    equity_outputs: dict[str, pd.DataFrame] = {}
    primary_trades = pd.DataFrame()
    for scenario, maker_penalty in scenarios:
        LOGGER.info("Pareto replay scenario %s", scenario.name)
        scenario_rows: list[dict[str, Any]] = []
        scenario_trades: list[pd.DataFrame] = []
        all_equities: list[tuple[int, pd.Series]] = []
        wf_equities: list[tuple[int, pd.Series]] = []
        holdout_equity = pd.Series(dtype=float)
        holdout_trades = pd.DataFrame()
        for window in windows_plus_holdout:
            is_holdout = window.fold_id == 999999
            label: int | str = "holdout_90d" if is_holdout else int(window.fold_id)
            selected_row = selected_by_fold[label]
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
            metrics = add_chunk_stats(metrics, equity)
            row = fold_row(label, scenario.name, selected_row, {}, metrics, window.train, window.test)
            scenario_rows.append(row)
            fold_rows.append(row)
            trades = trades.copy()
            trades.insert(0, "fold_id", label)
            trades.insert(0, "scenario", scenario.name)
            scenario_trades.append(trades)
            all_equities.append((999999 if is_holdout else int(window.fold_id), equity))
            if is_holdout:
                holdout_equity = equity
                holdout_trades = trades
            else:
                wf_equities.append((int(window.fold_id), equity))
        frame = pd.DataFrame(scenario_rows)
        all_trades = pd.concat(scenario_trades, ignore_index=True) if scenario_trades else pd.DataFrame()
        if scenario.name == primary.name:
            primary_trades = all_trades.copy()
        all_trades.to_csv(output_dir / "finalist_trades" / f"{scenario.name}_trades.csv", index=False)
        wf_frame = frame[~frame["is_holdout"].astype(bool)].copy()
        holdout_frame = frame[frame["is_holdout"].astype(bool)].copy()
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

    fold_metrics = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    primary_folds = fold_metrics[fold_metrics["scenario"].eq(primary.name)].copy()
    primary_wf_trades = primary_trades[~primary_trades["fold_id"].astype(str).eq("holdout_90d")].copy() if not primary_trades.empty else pd.DataFrame()
    cpcv = cpcv_process_summary(primary_folds, config)
    mc = monte_carlo_summary(primary_folds, primary_wf_trades, config)
    verdict = pareto_verdict(summary, cpcv, mc, config)

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    summary[summary["scope"].eq("holdout_90d")].to_csv(output_dir / "holdout_summary.csv", index=False)
    cpcv.to_csv(output_dir / "cpcv_summary.csv", index=False)
    mc.to_csv(output_dir / "monte_carlo_summary.csv", index=False)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    combined.to_csv(output_dir / "pareto_ranking.csv", index=False)

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
        "scenario_summary": summary.to_dict("records"),
        "cpcv_summary": cpcv.to_dict("records"),
        "monte_carlo_summary": mc.to_dict("records"),
        "verdict": verdict,
    }
    write_report(output_dir, payload)
    write_figures(output_dir, summary, equity_outputs)
    return {
        "output_dir": str(output_dir),
        "verdict": verdict,
        "top_combined": combined.sort_values(["monthly_return", "monthly_p10"], ascending=[False, False]).head(5).to_dict("records"),
        "monte_carlo": mc.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteration 036 zero-fee Pareto optimizer.")
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
