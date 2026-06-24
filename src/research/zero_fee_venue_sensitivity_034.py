from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.monthly_target_martingale_research import monthly_return_from_equity
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.research.revalidate_top3_distinct_policies import (
    EvaluationContext,
    PolicySpec,
    ScenarioSpec,
    audit_data_files,
    candidate_for_scenario,
    load_selected_candidates,
    locked_row_for_fold,
    locked_row_for_holdout,
    resolve_path,
    run_candidate_on_split,
    split_pre_holdout,
)
from src.research.walk_forward_martingale_research import WalkForwardWindow, stitch_oos_equity
from src.utils.config_loader import load_strategy_config, load_yaml
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_zero_fee_venue_sensitivity_034.yaml"


@dataclass(frozen=True)
class VenueScenario:
    name: str
    fee_rate: float
    slippage_bps: float
    maker_missed_fill_penalty: float
    venue_note: str

    def to_scenario_spec(self) -> ScenarioSpec:
        return ScenarioSpec(
            name=self.name,
            engine="audit",
            entry_execution_mode="next_bar_open",
            signal_lag_bars=1,
            mask_lag_bars=1,
            fee_rate=self.fee_rate,
            slippage_bps=self.slippage_bps,
        )


def load_venue_scenarios(config: dict[str, Any]) -> list[VenueScenario]:
    scenarios: list[VenueScenario] = []
    for row in config["scenarios"]:
        scenario = VenueScenario(
            name=str(row["name"]),
            fee_rate=float(row["fee_rate"]),
            slippage_bps=float(row["slippage_bps"]),
            maker_missed_fill_penalty=float(row.get("maker_missed_fill_penalty", 0.0)),
            venue_note=str(row.get("venue_note", "")),
        )
        if scenario.fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        if scenario.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if not 0 <= scenario.maker_missed_fill_penalty <= 1:
            raise ValueError("maker_missed_fill_penalty must be between 0 and 1")
        scenarios.append(scenario)
    return scenarios


def fold_metrics_row(
    scenario: VenueScenario,
    fold_id: int | str,
    train_index: pd.Index,
    test_index: pd.Index,
    selected_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scenario": scenario.name,
        "venue_note": scenario.venue_note,
        "fold_id": fold_id,
        "is_holdout": str(fold_id) == "holdout_90d",
        "train_start": train_index.min() if len(train_index) else pd.NaT,
        "train_end": train_index.max() if len(train_index) else pd.NaT,
        "test_start": test_index.min() if len(test_index) else pd.NaT,
        "test_end": test_index.max() if len(test_index) else pd.NaT,
        "selected_name": selected_name,
        "fee_rate": scenario.fee_rate,
        "slippage_bps": scenario.slippage_bps,
        "maker_missed_fill_penalty": scenario.maker_missed_fill_penalty,
        "total_return": float(metrics.get("total_return", 0.0)),
        "monthly_return": float(metrics.get("monthly_return", 0.0)),
        "annualized_return": float(metrics.get("annualized_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "equity_ruined": bool(metrics.get("equity_ruined", False)),
        "positive_fold": bool(float(metrics.get("total_return", 0.0)) > 0),
        "profit_factor": float(metrics.get("profit_factor", 0.0)),
        "number_of_grids": int(metrics.get("number_of_grids", 0)),
        "orders": int(metrics.get("orders", 0)),
        "fees_paid": float(metrics.get("fees_paid", 0.0)),
        "slippage_paid": float(metrics.get("slippage_paid", 0.0)),
        "cost_total": float(metrics.get("cost_total", 0.0)),
        "net_pnl": float(metrics.get("net_pnl", 0.0)),
        "grids_per_month": float(metrics.get("grids_per_month", 0.0)),
        "orders_per_month": float(metrics.get("orders_per_month", 0.0)),
        "cost_total_per_month": float(metrics.get("cost_total_per_month", 0.0)),
        "net_pnl_per_order": float(metrics.get("net_pnl_per_order", 0.0)),
        "expectancy_per_grid": float(metrics.get("expectancy_per_grid", 0.0)),
        "notional_turnover_per_month": float(metrics.get("notional_turnover_per_month", 0.0)),
        "max_notional": float(metrics.get("max_notional", 0.0)),
        "effective_leverage": float(metrics.get("effective_leverage", 0.0)),
    }


def summarize_scope(
    scenario: VenueScenario,
    fold_metrics: pd.DataFrame,
    equity: pd.Series,
    trades: pd.DataFrame,
    scope: str,
) -> dict[str, Any]:
    metrics = calculate_metrics(equity, trades if not trades.empty else None)
    if not trades.empty:
        metrics.update(summarize_simulations(trades, baseline_grids=len(trades)))
    else:
        metrics.update({"number_of_grids": 0, "realized_pnl": 0.0, "profit_factor": 0.0, "fees_paid": 0.0, "slippage_paid": 0.0})
    fold_returns = fold_metrics["monthly_return"].astype(float) if not fold_metrics.empty else pd.Series(dtype=float)
    return {
        "scope": scope,
        "scenario": scenario.name,
        "venue_note": scenario.venue_note,
        "fee_rate": scenario.fee_rate,
        "slippage_bps": scenario.slippage_bps,
        "maker_missed_fill_penalty": scenario.maker_missed_fill_penalty,
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
        "orders_per_month": float(fold_metrics["orders_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "grids_per_month": float(fold_metrics["grids_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "cost_total_per_month": float(fold_metrics["cost_total_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "net_pnl_per_order": float(metrics.get("realized_pnl", 0.0)) / max(order_count(trades), 1),
        "expectancy_per_grid": float(metrics.get("realized_pnl", 0.0)) / max(int(metrics.get("number_of_grids", 0)), 1),
        "notional_turnover_per_month": float(fold_metrics["notional_turnover_per_month"].mean()) if not fold_metrics.empty else 0.0,
        "max_notional": float(fold_metrics["max_notional"].max()) if not fold_metrics.empty else 0.0,
        "effective_leverage": float(fold_metrics["effective_leverage"].max()) if not fold_metrics.empty else 0.0,
    }


def scenario_verdict(summary: pd.DataFrame, scenario: str, config: dict[str, Any]) -> str:
    combined = summary[(summary["scope"].eq("combined_oos")) & (summary["scenario"].eq(scenario))]
    holdout = summary[(summary["scope"].eq("holdout_90d")) & (summary["scenario"].eq(scenario))]
    if combined.empty:
        return "not_evaluated"
    row = combined.iloc[0]
    hold = holdout.iloc[0] if not holdout.empty else row
    decision = config["decision"]
    if (
        float(row["monthly_return"]) > float(decision["positive_monthly_min"])
        and
        float(row["monthly_p10"]) > float(decision["robust_monthly_p10_min"])
        and float(row["max_drawdown"]) > float(decision["max_drawdown_block"])
        and not bool(row["equity_ruined"])
        and float(hold["total_return"]) > float(decision["holdout_min_total_return"])
    ):
        return "zero_fee_robust_candidate"
    if float(row["monthly_return"]) > float(decision["positive_monthly_min"]) and not bool(row["equity_ruined"]):
        return "zero_fee_fragile_positive"
    return "zero_fee_not_enough"


def write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    summary = pd.DataFrame(payload["scenario_summary"])
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    holdout = summary[summary["scope"].eq("holdout_90d")].copy()
    lines = [
        "# Iteration 034 - Zero-Fee Venue Sensitivity",
        "",
        "## Combined OOS",
        markdown_table(
            combined[
                [
                    "scenario",
                    "verdict",
                    "monthly_return",
                    "monthly_p10",
                    "max_drawdown",
                    "equity_ruined",
                    "positive_fold_rate",
                    "number_of_grids",
                    "orders",
                    "fees_paid",
                    "slippage_paid",
                    "cost_total_per_month",
                    "net_pnl_per_order",
                    "effective_leverage",
                ]
            ]
        )
        if not combined.empty
        else "No combined rows.",
        "",
        "## Holdout 90d",
        markdown_table(
            holdout[
                [
                    "scenario",
                    "monthly_return",
                    "total_return",
                    "max_drawdown",
                    "equity_ruined",
                    "number_of_grids",
                    "orders",
                    "net_pnl",
                ]
            ]
        )
        if not holdout.empty
        else "No holdout rows.",
        "",
        "## Notes",
        "- Strategy and historical selected candidates are locked from `fundamental_trend_escape_v2`.",
        "- Timing remains conservative: signal/mask lag 1 bar and entry at next 5m open.",
        "- Maker scenarios are proxies: zero explicit fee/slippage plus a penalty on positive PnL to represent missed or delayed fills.",
        "- This does not validate venue availability, leverage, borrow, funding, or API execution risk.",
    ]
    (output_dir / "iteration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output_dir: Path, summary: pd.DataFrame, equity_outputs: dict[str, pd.DataFrame]) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    if not combined.empty:
        ordered = combined.sort_values("monthly_return", ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(ordered["scenario"], ordered["monthly_return"].astype(float) * 100)
        plt.xticks(rotation=75, ha="right", fontsize=8)
        plt.ylabel("Monthly return (%)")
        plt.title("Zero-Fee Venue Sensitivity - Combined OOS")
        plt.tight_layout()
        plt.savefig(figures / "combined_monthly_return_by_scenario.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.bar(ordered["scenario"], ordered["monthly_p10"].astype(float) * 100)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=75, ha="right", fontsize=8)
        plt.ylabel("P10 monthly return (%)")
        plt.title("Zero-Fee Venue Sensitivity - P10 Monthly")
        plt.tight_layout()
        plt.savefig(figures / "p10_monthly_return_by_scenario.png", dpi=150)
        plt.close()

    plt.figure(figsize=(12, 6))
    for key, frame in equity_outputs.items():
        if "combined_oos" not in key:
            continue
        plt.plot(pd.to_datetime(frame["timestamp"], utc=True), frame["equity"], label=key.replace("__combined_oos", ""))
    plt.title("Zero-Fee Venue Sensitivity - Combined OOS Equity")
    plt.ylabel("Equity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figures / "combined_oos_equity.png", dpi=150)
    plt.close()


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
        config["walk_forward"]["train_days"] = min(float(config["walk_forward"]["train_days"]), 30.0)
        config["walk_forward"]["test_days"] = min(float(config["walk_forward"]["test_days"]), 5.0)
        config["walk_forward"]["step_days"] = min(float(config["walk_forward"]["step_days"]), 5.0)
        config["walk_forward"]["holdout_days"] = min(float(config["walk_forward"]["holdout_days"]), 15.0)
        config["scenarios"] = config["scenarios"][:4]

    output_root = resolve_path(config["iteration"]["output_root"])
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"zero_fee_venue_sensitivity_034_{stamp}"
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
    scenarios = load_venue_scenarios(config)
    base_risk = validate_strategy_config(load_strategy_config())
    account_equity = float(config["reporting"]["account_equity_usdt"])

    _events, _event_windows, blackout_masks = build_blackout_bundle(market.index, config)
    trend_components = build_trend_escape_components(market, config)
    entry_mask = fundamental_trend_mask(trend_components["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)
    context = EvaluationContext(entry_mask, None, None)

    windows_plus_holdout = windows + [WalkForwardWindow(fold_id=999999, train=holdout_train_index, test=holdout_index)]
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    equity_outputs: dict[str, pd.DataFrame] = {}

    for scenario in scenarios:
        LOGGER.info("Evaluating venue scenario %s", scenario.name)
        wf_fold_rows: list[dict[str, Any]] = []
        wf_trades: list[pd.DataFrame] = []
        wf_equities: list[tuple[int, pd.Series]] = []
        all_fold_rows: list[dict[str, Any]] = []
        all_trades: list[pd.DataFrame] = []
        all_equities: list[tuple[int, pd.Series]] = []
        holdout_fold_rows: list[dict[str, Any]] = []
        holdout_trades = pd.DataFrame()
        holdout_equity = pd.Series(dtype=float)
        scenario_spec = scenario.to_scenario_spec()
        for window in windows_plus_holdout:
            is_holdout = window.fold_id == 999999
            fold_label: int | str = "holdout_90d" if is_holdout else int(window.fold_id)
            selected_row = locked_row_for_holdout(policy, selected, holdout_start) if is_holdout else locked_row_for_fold(selected, int(window.fold_id))
            metrics, trades, equity = run_candidate_on_split(
                market,
                signal_frame,
                base_risk,
                policy,
                selected_row,
                window.test,
                "holdout" if is_holdout else "test",
                scenario_spec,
                context,
            )
            if scenario.maker_missed_fill_penalty > 0:
                split_frame = market.loc[window.test]
                trades = apply_maker_proxy(trades, scenario.maker_missed_fill_penalty)
                candidate = candidate_for_scenario(selected_row, scenario_spec)
                metrics, equity = recompute_metrics(split_frame.index, trades, "holdout" if is_holdout else "test", candidate)
                metrics.update({"policy": policy.name, "scenario": scenario.name})
            metrics = enrich_cost_metrics(metrics, trades, window.test, account_equity)
            row = fold_metrics_row(scenario, fold_label, window.train, window.test, str(selected_row["name"]), metrics)
            fold_rows.append(row)
            all_fold_rows.append(row)
            scoped_trades = trades.copy()
            scoped_trades.insert(0, "fold_id", fold_label)
            scoped_trades.insert(0, "scenario", scenario.name)
            all_trades.append(scoped_trades)
            all_equities.append((999999 if is_holdout else int(window.fold_id), equity))
            if is_holdout:
                holdout_fold_rows.append(row)
                holdout_trades = scoped_trades
                holdout_equity = equity
            else:
                wf_fold_rows.append(row)
                wf_trades.append(scoped_trades)
                wf_equities.append((int(window.fold_id), equity))

        wf_frame = pd.DataFrame(wf_fold_rows)
        holdout_frame = pd.DataFrame(holdout_fold_rows)
        all_frame = pd.DataFrame(all_fold_rows)
        wf_trades_frame = pd.concat(wf_trades, ignore_index=True) if wf_trades else pd.DataFrame()
        all_trades_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        wf_equity = stitch_oos_equity(wf_equities)["equity"] if wf_equities else pd.Series(dtype=float)
        all_equity = stitch_oos_equity(all_equities)["equity"] if all_equities else pd.Series(dtype=float)
        for scope, frame, equity, trades in [
            ("wf_pre_holdout", wf_frame, wf_equity, wf_trades_frame),
            ("holdout_90d", holdout_frame, holdout_equity, holdout_trades),
            ("combined_oos", all_frame, all_equity, all_trades_frame),
        ]:
            summary_rows.append(summarize_scope(scenario, frame, equity, trades, scope))
            if not equity.empty:
                key = f"{scenario.name}__{scope}"
                equity_frame = pd.DataFrame({"timestamp": equity.index.astype(str), "equity": equity.to_numpy()})
                equity_frame.to_csv(output_dir / "oos_equity" / f"{key}.csv", index=False)
                equity_outputs[key] = equity_frame

    fold_metrics = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    summary["verdict"] = [scenario_verdict(summary, str(row["scenario"]), config) for _, row in summary.iterrows()]
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    holdout_summary = summary[summary["scope"].eq("holdout_90d")].copy()
    ranking = combined.sort_values(["monthly_return", "monthly_p10", "max_drawdown"], ascending=[False, False, False])
    cost_boundary = combined[["scenario", "fee_rate", "slippage_bps", "maker_missed_fill_penalty", "monthly_return", "monthly_p10", "verdict"]].copy()

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    holdout_summary.to_csv(output_dir / "holdout_summary.csv", index=False)
    ranking.to_csv(output_dir / "ranking_zero_fee_scenarios.csv", index=False)
    cost_boundary.to_csv(output_dir / "zero_fee_boundary.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "max_folds": max_folds,
        "source_policy": config["source_policy"],
        "scenario_count": len(scenarios),
        "holdout_start_utc": str(holdout_start),
        "holdout_end_utc": str(holdout_index.max()),
        "fold_count_pre_holdout": len(windows),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    payload = {
        "manifest": manifest,
        "data_audit": [data_audit],
        "scenario_summary": summary.to_dict("records"),
        "ranking_zero_fee_scenarios": ranking.to_dict("records"),
    }
    write_report(output_dir, payload)
    write_figures(output_dir, summary, equity_outputs)
    return {
        "output_dir": str(output_dir),
        "top_scenarios": ranking.head(5).to_dict("records"),
        "verdict_counts": combined["verdict"].value_counts().to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteration 034 zero-fee venue sensitivity.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(args.config, smoke=args.smoke, max_folds=args.max_folds)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
