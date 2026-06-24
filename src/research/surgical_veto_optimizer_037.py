from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
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
    locked_row_for_fold,
    locked_row_for_holdout,
    resolve_path,
    split_pre_holdout,
    unique_candidate_rows,
)
from src.research.walk_forward_martingale_research import WalkForwardWindow, stitch_oos_equity
from src.research.zero_fee_p10_optimizer_035 import (
    cpcv_process_summary,
    evaluate_selected,
    fold_row,
    monte_carlo_summary,
    scenario_spec,
    summarize_scope,
)
from src.utils.config_loader import load_strategy_config, load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_surgical_veto_optimizer_037.yaml"


def generate_veto_policies(config: dict[str, Any], max_policies: int | None = None) -> list[dict[str, Any]]:
    search = config["veto_search"]
    rows: list[dict[str, Any]] = []
    for idx, values in enumerate(
        itertools.product(
            search["range_expansion_thresholds"],
            search["realized_vol_thresholds"],
            search["atr_percentile_thresholds"],
            search["use_breakout_risk"],
            search["use_regime_disallow"],
            search["max_grids_per_month"],
            search["pause_after_forced_loss_hours"],
            search["rolling_7d_loss_thresholds"],
        )
    ):
        range_threshold, vol_threshold, atr_pct, breakout, regime_disallow, max_grids, pause, rolling_loss = values
        row = {
            "veto_uid": f"veto_{idx:05d}",
            "range_expansion_threshold": float(range_threshold),
            "realized_vol_threshold": float(vol_threshold),
            "atr_percentile_threshold": float(atr_pct),
            "use_breakout_risk": bool(breakout),
            "use_regime_disallow": bool(regime_disallow),
            "max_grids_per_month": float(max_grids),
            "pause_after_forced_loss_hours": float(pause),
            "rolling_7d_loss_threshold": float(rolling_loss),
        }
        active = [
            row["range_expansion_threshold"] > 0,
            row["realized_vol_threshold"] > 0,
            row["atr_percentile_threshold"] > 0,
            row["use_breakout_risk"],
            row["use_regime_disallow"],
            row["pause_after_forced_loss_hours"] > 0,
            row["rolling_7d_loss_threshold"] > -100,
            row["max_grids_per_month"] < 999,
        ]
        if sum(active) > 3:
            continue
        row["veto_name"] = (
            f"rx{row['range_expansion_threshold']:g}_rv{row['realized_vol_threshold']:g}"
            f"_atr{row['atr_percentile_threshold']:g}_br{int(row['use_breakout_risk'])}"
            f"_reg{int(row['use_regime_disallow'])}_grid{row['max_grids_per_month']:g}"
            f"_pause{row['pause_after_forced_loss_hours']:g}_roll{row['rolling_7d_loss_threshold']:g}"
        ).replace(".", "p").replace("-", "m")
        rows.append(row)
    rows.insert(
        0,
        {
            "veto_uid": "veto_none",
            "veto_name": "veto_none",
            "range_expansion_threshold": 0.0,
            "realized_vol_threshold": 0.0,
            "atr_percentile_threshold": 0.0,
            "use_breakout_risk": False,
            "use_regime_disallow": False,
            "max_grids_per_month": 999.0,
            "pause_after_forced_loss_hours": 0.0,
            "rolling_7d_loss_threshold": -999.0,
        },
    )
    limit = int(max_policies or search["max_policies"])
    return rows[:limit]


def atr_percentile_series(market: pd.DataFrame, lookback_bars: int = 30 * 24 * 12) -> pd.Series:
    atr = market["atr_5m"].astype(float)
    return atr.rolling(lookback_bars, min_periods=max(100, lookback_bars // 10)).rank(pct=True)


def build_veto_mask(market: pd.DataFrame, policy: dict[str, Any], atr_pct: pd.Series | None = None) -> pd.Series:
    mask = pd.Series(False, index=market.index)
    if float(policy["range_expansion_threshold"]) > 0 and "range_expansion_ratio" in market:
        mask |= market["range_expansion_ratio"].astype(float).ge(float(policy["range_expansion_threshold"]))
    if float(policy["realized_vol_threshold"]) > 0 and "realized_volatility_ratio" in market:
        mask |= market["realized_volatility_ratio"].astype(float).ge(float(policy["realized_vol_threshold"]))
    if float(policy["atr_percentile_threshold"]) > 0:
        if atr_pct is None:
            atr_pct = atr_percentile_series(market)
        mask |= atr_pct.reindex(market.index).fillna(0.0).ge(float(policy["atr_percentile_threshold"]))
    if bool(policy["use_breakout_risk"]) and "breakout_risk" in market:
        mask |= market["breakout_risk"].fillna(0).astype(int).astype(bool)
    if bool(policy["use_regime_disallow"]) and "regime_allows_grid" in market:
        mask |= ~market["regime_allows_grid"].fillna(1).astype(int).astype(bool)
    return mask.fillna(False).astype(bool)


def candidate_with_veto(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.setdefault("blackout_hours", 24)
    out.setdefault("min_severity", 4)
    out.setdefault("trend_propagation_bars", 36)
    out.setdefault("breakout_atr_buffer", 0.25)
    out.setdefault("min_range_expansion_ratio", 1.3)
    out["candidate_uid"] = str(policy["veto_uid"])
    out["name"] = f"{policy['veto_name']}__{str(row.get('name', 'locked'))[:80]}"
    out["max_grids_per_month"] = float(policy["max_grids_per_month"])
    out["pause_after_forced_loss_hours"] = float(policy["pause_after_forced_loss_hours"])
    out["rolling_7d_loss_threshold"] = float(policy["rolling_7d_loss_threshold"])
    return out


def monthly_chunks_from_equity(equity: pd.Series) -> tuple[float, float]:
    from src.research.zero_fee_p10_optimizer_035 import monthly_chunk_returns

    values = monthly_chunk_returns(equity)
    return float(np.quantile(values, 0.10)), float(np.median(values))


def evaluate_veto_policy(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    locked_row: dict[str, Any],
    policy: dict[str, Any],
    split_index: pd.Index,
    scenario: Any,
    base_entry_mask: pd.Series,
    veto_mask: pd.Series,
    maker_penalty: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    row = candidate_with_veto(locked_row, policy)
    combined_mask = (base_entry_mask.reindex(market.index).fillna(False).astype(bool) | veto_mask.reindex(market.index).fillna(False).astype(bool))
    metrics, trades, equity = evaluate_selected(
        market,
        signal_frame,
        base_risk,
        row,
        split_index,
        scenario,
        combined_mask,
        maker_penalty=maker_penalty,
    )
    p10, median = monthly_chunks_from_equity(equity)
    metrics.update(
        {
            "veto_uid": policy["veto_uid"],
            "veto_name": policy["veto_name"],
            "monthly_p10_chunks": p10,
            "monthly_median_chunks": median,
        }
    )
    return metrics, trades, equity


def veto_constraint_audit(metrics: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    search = config["veto_search"]
    reasons: list[str] = []
    if float(metrics.get("monthly_return", 0.0)) < float(search["min_train_monthly_return"]):
        reasons.append("min_monthly")
    if float(metrics.get("monthly_p10_chunks", 0.0)) < float(search.get("min_train_monthly_p10", -1.0)):
        reasons.append("min_p10")
    if float(metrics.get("monthly_return", 0.0)) < float(baseline.get("monthly_return", 0.0)) * float(search["min_baseline_return_retention"]):
        reasons.append("return_retention")
    if float(metrics.get("orders_per_month", 0.0)) < float(baseline.get("orders_per_month", 0.0)) * float(search["min_baseline_order_retention"]):
        reasons.append("order_retention")
    if float(metrics.get("monthly_p10_chunks", 0.0)) < float(baseline.get("monthly_p10_chunks", 0.0)) + float(search["min_p10_improvement"]):
        reasons.append("p10_not_improved")
    if float(metrics.get("max_drawdown", 0.0)) <= float(search["max_drawdown"]):
        reasons.append("drawdown")
    if bool(metrics.get("equity_ruined", False)):
        reasons.append("ruin")
    return not reasons, "|".join(reasons) if reasons else "pass"


def veto_score(metrics: dict[str, Any], baseline: dict[str, Any]) -> float:
    return float(
        0.55 * float(metrics.get("monthly_return", 0.0))
        + 0.25 * float(metrics.get("monthly_p10_chunks", 0.0))
        + 0.10 * float(metrics.get("monthly_median_chunks", metrics.get("monthly_return", 0.0)))
        + 15.0 * float(metrics.get("net_pnl_per_order", 0.0))
        + 0.10 * (float(metrics.get("monthly_p10_chunks", 0.0)) - float(baseline.get("monthly_p10_chunks", 0.0)))
        - 0.05 * max(0.0, 1.0 - float(metrics.get("orders_per_month", 0.0)) / max(float(baseline.get("orders_per_month", 1.0)), 1.0))
    )


def select_veto_on_train(
    policies: list[dict[str, Any]],
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    locked_row: dict[str, Any],
    train_index: pd.Index,
    scenario: Any,
    base_entry_mask: pd.Series,
    veto_masks: dict[str, pd.Series],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    baseline_policy = policies[0]
    baseline_metrics, _baseline_trades, _baseline_equity = evaluate_veto_policy(
        market,
        signal_frame,
        base_risk,
        locked_row,
        baseline_policy,
        train_index,
        scenario,
        base_entry_mask,
        veto_masks[baseline_policy["veto_uid"]],
    )
    rows: list[dict[str, Any]] = []
    for policy in policies:
        metrics, _trades, _equity = evaluate_veto_policy(
            market,
            signal_frame,
            base_risk,
            locked_row,
            policy,
            train_index,
            scenario,
            base_entry_mask,
            veto_masks[policy["veto_uid"]],
        )
        passed, reasons = veto_constraint_audit(metrics, baseline_metrics, config)
        metrics.update(
            {
                **policy,
                "constraint_pass": bool(passed),
                "constraint_reasons": reasons,
                "veto_score": veto_score(metrics, baseline_metrics),
                "baseline_train_monthly_return": baseline_metrics.get("monthly_return", 0.0),
                "baseline_train_monthly_p10": baseline_metrics.get("monthly_p10_chunks", 0.0),
                "baseline_orders_per_month": baseline_metrics.get("orders_per_month", 0.0),
            }
        )
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame["constraint_rank"] = (~frame["constraint_pass"].astype(bool)).astype(int)
    frame = frame.sort_values(["constraint_rank", "veto_score", "monthly_return", "monthly_p10_chunks"], ascending=[True, False, False, False])
    selected = frame.iloc[0].to_dict()
    selected_policy = next(policy for policy in policies if policy["veto_uid"] == selected["veto_uid"])
    return selected_policy, selected, frame


def candidate_pool_from_history(selected: pd.DataFrame, max_candidates: int | None = None) -> list[dict[str, Any]]:
    rows = unique_candidate_rows(selected)
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    for idx, row in enumerate(rows):
        row["historical_pool_id"] = f"hist_{idx:03d}"
    if not rows:
        raise ValueError("historical candidate pool is empty")
    return rows


def candidate_selection_score(row: dict[str, Any]) -> float:
    return (
        float(row.get("monthly_return", 0.0))
        + 0.65 * float(row.get("monthly_p10_chunks", 0.0))
        + 0.15 * float(row.get("monthly_median_chunks", row.get("monthly_return", 0.0)))
        + 25.0 * float(row.get("net_pnl_per_order", 0.0))
        - 0.00035 * max(0.0, float(row.get("orders_per_month", 0.0)) - 90.0)
        - 0.0030 * max(0.0, float(row.get("effective_leverage", 0.0)) - 15.0)
        + 0.10 * float(row.get("max_drawdown", 0.0))
    )


def select_candidate_and_veto_on_train(
    candidate_pool: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: Any,
    train_index: pd.Index,
    scenario: Any,
    base_entry_mask: pd.Series,
    veto_masks: dict[str, pd.Series],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    best_policy: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_candidate: dict[str, Any] | None = None
    best_score = -np.inf
    for candidate_idx, locked in enumerate(candidate_pool):
        selected_policy, selected_metrics, matrix = select_veto_on_train(
            policies,
            market,
            signal_frame,
            base_risk,
            locked,
            train_index,
            scenario,
            base_entry_mask,
            veto_masks,
            config,
        )
        candidate_id = str(locked.get("historical_pool_id", f"hist_{candidate_idx:03d}"))
        selected_metrics = dict(selected_metrics)
        selected_metrics["historical_pool_id"] = candidate_id
        selected_metrics["historical_candidate_name"] = str(locked.get("name", ""))
        selected_metrics["candidate_selection_score"] = candidate_selection_score(selected_metrics)
        selected_metrics["candidate_selection_rank"] = 0 if bool(selected_metrics.get("constraint_pass", False)) else 1
        matrix = matrix.copy()
        matrix["historical_pool_id"] = candidate_id
        matrix["historical_candidate_name"] = str(locked.get("name", ""))
        matrix["candidate_selection_score"] = matrix.apply(lambda row: candidate_selection_score(row.to_dict()), axis=1)
        matrix["candidate_selection_rank"] = (~matrix["constraint_pass"].astype(bool)).astype(int)
        frames.append(matrix)
        score = float(selected_metrics["candidate_selection_score"]) - 100.0 * float(selected_metrics["candidate_selection_rank"])
        if score > best_score:
            best_score = score
            best_policy = selected_policy
            best_metrics = selected_metrics
            best_candidate = locked
    if best_policy is None or best_metrics is None or best_candidate is None:
        raise ValueError("No candidate/veto combination could be selected on train")
    full_matrix = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    full_matrix = full_matrix.sort_values(
        ["candidate_selection_rank", "constraint_rank", "candidate_selection_score", "monthly_return", "monthly_p10_chunks"],
        ascending=[True, True, False, False, False],
    )
    return best_candidate, best_policy, best_metrics, full_matrix


def surgical_verdict(summary: pd.DataFrame, config: dict[str, Any]) -> str:
    primary = summary[(summary["scope"].eq("combined_oos")) & (summary["scenario"].eq(config["primary_scenario"]["name"]))]
    holdout = summary[(summary["scope"].eq("holdout_90d")) & (summary["scenario"].eq(config["primary_scenario"]["name"]))]
    if primary.empty:
        return "not evaluated"
    row = primary.iloc[0]
    hold = holdout.iloc[0] if not holdout.empty else row
    validation = config["validation"]
    if (
        float(row["monthly_return"]) >= float(validation["min_monthly_return"])
        and float(row["monthly_p10"]) >= float(validation["robust_monthly_p10_min"])
        and float(hold["total_return"]) >= float(validation["min_holdout_total_return"])
        and not bool(row["equity_ruined"])
    ):
        return "surgical veto candidate"
    if float(row["monthly_return"]) > 0 and float(hold["total_return"]) >= 0:
        return "fragile but improved"
    return "not enough edge"


def write_report(output_dir: str, payload: dict[str, Any]) -> None:
    summary = pd.DataFrame(payload["scenario_summary"])
    combined = summary[summary["scope"].eq("combined_oos")]
    holdout = summary[summary["scope"].eq("holdout_90d")]
    lines = [
        "# Iteration 037 - Surgical Veto Optimizer",
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
        "## Holdout",
        markdown_table(holdout[["scenario", "monthly_return", "total_return", "max_drawdown", "orders_per_month", "net_pnl_per_order"]]),
        "",
        "## Notes",
        "- Grid/sizing/candidates remain locked from fundamental_trend_escape_v2.",
        "- Veto policies only block entries or apply stateful post-loss guards selected on train.",
        "- The score requires return/order retention against the baseline train replay.",
    ]
    (resolve_path(output_dir) / "iteration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output_dir: str, summary: pd.DataFrame, equity_outputs: dict[str, pd.DataFrame]) -> None:
    figures = resolve_path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = summary[summary["scope"].eq("combined_oos")]
    if not combined.empty:
        x = np.arange(len(combined))
        plt.figure(figsize=(12, 6))
        plt.bar(x - 0.2, combined["monthly_return"].astype(float) * 100, width=0.4, label="Monthly")
        plt.bar(x + 0.2, combined["monthly_p10"].astype(float) * 100, width=0.4, label="P10")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(x, combined["scenario"], rotation=70, ha="right", fontsize=8)
        plt.ylabel("Return (%)")
        plt.title("Iteration 037 Surgical Veto: Monthly vs P10")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures / "monthly_vs_p10_by_scenario.png", dpi=150)
        plt.close()
    plt.figure(figsize=(12, 6))
    for key, frame in equity_outputs.items():
        if "combined_oos" not in key:
            continue
        plt.plot(pd.to_datetime(frame["timestamp"], utc=True), frame["equity"], label=key.replace("__combined_oos", ""))
    plt.title("Iteration 037 Combined OOS Equity")
    plt.ylabel("Equity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures / "combined_oos_equity.png", dpi=150)
    plt.close()


def run_iteration(
    config_path: str = DEFAULT_CONFIG,
    smoke: bool = False,
    max_folds: int | None = None,
    max_policies: int | None = None,
    max_candidates: int | None = None,
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if max_policies is not None:
        config["veto_search"]["max_policies"] = int(max_policies)
    if max_candidates is not None:
        config.setdefault("candidate_selection", {})["max_historical_candidates"] = int(max_candidates)
    if smoke and max_folds is None:
        max_folds = 2
    if smoke:
        config["walk_forward"]["train_days"] = 30.0
        config["walk_forward"]["test_days"] = 5.0
        config["walk_forward"]["step_days"] = 5.0
        config["walk_forward"]["holdout_days"] = 15.0
        config["veto_search"]["max_policies"] = min(int(config["veto_search"]["max_policies"]), 8)
        config.setdefault("candidate_selection", {})["max_historical_candidates"] = min(
            int(config.get("candidate_selection", {}).get("max_historical_candidates", 4)),
            4,
        )
        config["scenario_replay"] = config["scenario_replay"][:3]
        config["validation"]["monte_carlo_iterations"] = 200

    output_root = resolve_path(config["iteration"]["output_root"])
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"surgical_veto_optimizer_037_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "oos_equity").mkdir(parents=True, exist_ok=True)
    (output_dir / "finalist_trades").mkdir(parents=True, exist_ok=True)

    market_path = resolve_path(config["data"]["market_5m_path"])
    signal_path = resolve_path(config["data"]["signal_1h_path"])
    market_audited, signal_frame, data_audit = audit_data_files(market_path, signal_path, float(config["data"]["min_coverage_rate"]), int(config["data"]["bar_minutes"]))
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

    policy = PolicySpec(
        name=str(config["source_policy"]["name"]),
        source_iteration_dir=resolve_path(config["source_policy"]["source_iteration_dir"]),
        source_variant=str(config["source_policy"]["source_variant"]),
        policy_kind=str(config["source_policy"]["policy_kind"]),
    )
    selected = load_selected_candidates(policy)
    candidate_cfg = config.get("candidate_selection", {})
    candidate_mode = str(candidate_cfg.get("mode", "locked"))
    historical_pool = candidate_pool_from_history(
        selected,
        int(candidate_cfg["max_historical_candidates"]) if candidate_cfg.get("max_historical_candidates") is not None else None,
    )
    base_risk = validate_strategy_config(load_strategy_config())
    primary = scenario_spec(config["primary_scenario"]["name"], float(config["primary_scenario"]["fee_rate"]), float(config["primary_scenario"]["slippage_bps"]))
    scenarios = [
        (
            scenario_spec(str(row["name"]), float(row["fee_rate"]), float(row["slippage_bps"])),
            float(row.get("maker_missed_fill_penalty", 0.0)),
        )
        for row in config["scenario_replay"]
    ]

    _events, _event_windows, blackout_masks = build_blackout_bundle(market.index, config)
    trend_components = build_trend_escape_components(market, config)
    base_entry_mask = fundamental_trend_mask(trend_components["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)
    atr_pct = atr_percentile_series(market)
    policies = generate_veto_policies(config, config["veto_search"]["max_policies"])
    veto_masks = {row["veto_uid"]: build_veto_mask(market, row, atr_pct) for row in policies}
    pd.DataFrame(policies).to_csv(output_dir / "veto_universe.csv", index=False)
    pd.DataFrame(historical_pool).to_csv(output_dir / "historical_candidate_pool.csv", index=False)

    windows_plus_holdout = windows + [WalkForwardWindow(fold_id=999999, train=holdout_train_index, test=holdout_index)]
    selected_veto_by_fold: dict[int | str, dict[str, Any]] = {}
    locked_row_by_fold: dict[int | str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    for window in windows_plus_holdout:
        label: int | str = "holdout_90d" if window.fold_id == 999999 else int(window.fold_id)
        LOGGER.info("Selecting surgical veto fold %s", label)
        if candidate_mode == "locked":
            locked = locked_row_for_holdout(policy, selected, holdout_start) if window.fold_id == 999999 else locked_row_for_fold(selected, int(window.fold_id))
            selected_veto, selection_metrics, matrix = select_veto_on_train(
                policies,
                market,
                signal_frame,
                base_risk,
                locked,
                window.train,
                primary,
                base_entry_mask,
                veto_masks,
                config,
            )
        elif candidate_mode == "historical_pool":
            locked, selected_veto, selection_metrics, matrix = select_candidate_and_veto_on_train(
                historical_pool,
                policies,
                market,
                signal_frame,
                base_risk,
                window.train,
                primary,
                base_entry_mask,
                veto_masks,
                config,
            )
        else:
            raise ValueError(f"Unsupported candidate_selection.mode: {candidate_mode}")
        locked_row_by_fold[label] = locked
        selected_veto_by_fold[label] = selected_veto
        matrix = matrix.copy()
        matrix["fold_id"] = label
        selection_rows.extend(matrix.to_dict("records"))
    pd.DataFrame(selection_rows).to_csv(output_dir / "selection_matrix.csv", index=False)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    equity_outputs: dict[str, pd.DataFrame] = {}
    primary_trades = pd.DataFrame()
    for scenario, maker_penalty in scenarios:
        scenario_fold_rows: list[dict[str, Any]] = []
        scenario_trades: list[pd.DataFrame] = []
        all_equities: list[tuple[int, pd.Series]] = []
        wf_equities: list[tuple[int, pd.Series]] = []
        holdout_equity = pd.Series(dtype=float)
        holdout_trades = pd.DataFrame()
        for window in windows_plus_holdout:
            is_holdout = window.fold_id == 999999
            label = "holdout_90d" if is_holdout else int(window.fold_id)
            selected_veto = selected_veto_by_fold[label]
            locked = locked_row_by_fold[label]
            metrics, trades, equity = evaluate_veto_policy(
                market,
                signal_frame,
                base_risk,
                locked,
                selected_veto,
                window.test,
                scenario,
                base_entry_mask,
                veto_masks[selected_veto["veto_uid"]],
                maker_penalty=maker_penalty,
            )
            row = fold_row(label, scenario.name, candidate_with_veto(locked, selected_veto), {}, metrics, window.train, window.test)
            row.update({k: selected_veto[k] for k in selected_veto})
            scenario_fold_rows.append(row)
            fold_rows.append(row)
            scoped_trades = trades.copy()
            scoped_trades.insert(0, "fold_id", label)
            scoped_trades.insert(0, "scenario", scenario.name)
            scenario_trades.append(scoped_trades)
            all_equities.append((999999 if is_holdout else int(window.fold_id), equity))
            if is_holdout:
                holdout_equity = equity
                holdout_trades = scoped_trades
            else:
                wf_equities.append((int(window.fold_id), equity))
        frame = pd.DataFrame(scenario_fold_rows)
        all_trades = pd.concat(scenario_trades, ignore_index=True) if scenario_trades else pd.DataFrame()
        all_trades.to_csv(output_dir / "finalist_trades" / f"{scenario.name}_trades.csv", index=False)
        if scenario.name == primary.name:
            primary_trades = all_trades.copy()
        wf_frame = frame[~frame["is_holdout"].astype(bool)]
        holdout_frame = frame[frame["is_holdout"].astype(bool)]
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
                equity_frame = pd.DataFrame({"timestamp": scope_equity.index.astype(str), "equity": scope_equity.to_numpy()})
                key = f"{scenario.name}__{scope}"
                equity_frame.to_csv(output_dir / "oos_equity" / f"{key}.csv", index=False)
                equity_outputs[key] = equity_frame

    fold_metrics = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    primary_folds = fold_metrics[fold_metrics["scenario"].eq(primary.name)].copy()
    primary_wf_trades = primary_trades[~primary_trades["fold_id"].astype(str).eq("holdout_90d")].copy() if not primary_trades.empty else pd.DataFrame()
    cpcv = cpcv_process_summary(primary_folds, config)
    mc = monte_carlo_summary(primary_folds, primary_wf_trades, config)
    verdict = surgical_verdict(summary, config)

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    summary[summary["scope"].eq("holdout_90d")].to_csv(output_dir / "holdout_summary.csv", index=False)
    cpcv.to_csv(output_dir / "cpcv_summary.csv", index=False)
    mc.to_csv(output_dir / "monte_carlo_summary.csv", index=False)
    summary[summary["scope"].eq("combined_oos")].to_csv(output_dir / "surgical_ranking.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "max_folds": max_folds,
        "max_policies": int(config["veto_search"]["max_policies"]),
        "candidate_mode": candidate_mode,
        "max_historical_candidates": int(candidate_cfg.get("max_historical_candidates", len(historical_pool))),
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
    write_report(str(output_dir), payload)
    write_figures(str(output_dir), summary, equity_outputs)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    return {
        "output_dir": str(output_dir),
        "verdict": verdict,
        "top_combined": combined.sort_values(["monthly_return", "monthly_p10"], ascending=[False, False]).head(5).to_dict("records"),
        "monte_carlo": mc.to_dict("records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteration 037 surgical veto optimizer.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        smoke=args.smoke,
        max_folds=args.max_folds,
        max_policies=args.max_policies,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
