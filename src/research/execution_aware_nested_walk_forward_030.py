from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.fundamental_blackout_martingale_research import markdown_table
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/research_iteration_execution_aware_nested_walk_forward_030.yaml"


def load_finalists(campaign_dir: Path, allowed_families: list[str]) -> pd.DataFrame:
    path = campaign_dir / "walk_forward_finalist_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Validation campaign summary not found: {path}")
    frame = pd.read_csv(path)
    missing = {"family", "aggregate_monthly_return", "mc_fold_block_monthly_p05", "aggregate_max_drawdown"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame[frame["family"].astype(str).isin(allowed_families)].copy()


def load_conditional_costs(execution_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    costs_path = execution_dir / "conditional_execution_costs.csv"
    coverage_path = execution_dir / "coverage_report.csv"
    costs = pd.read_csv(costs_path) if costs_path.exists() else pd.DataFrame()
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    return costs, coverage


def coverage_is_usable(coverage: pd.DataFrame, threshold: float) -> bool:
    if coverage.empty or "exact_signal_fraction" not in coverage.columns:
        return False
    return float(coverage.iloc[0]["exact_signal_fraction"]) >= float(threshold)


def apply_execution_adjustment(finalist: pd.Series, cost: pd.Series | None) -> dict[str, Any]:
    gross_median = float(finalist["aggregate_monthly_return"])
    gross_p10 = float(finalist.get("mc_fold_block_monthly_p05", gross_median))
    drawdown = float(finalist.get("aggregate_max_drawdown", np.nan))
    ruin_rate = max(
        float(finalist.get("mc_fold_block_ruin_probability", 0.0) or 0.0),
        float(finalist.get("cpcv_ruin_rate", 0.0) or 0.0),
    )
    if cost is None:
        return {
            "family": str(finalist["family"]),
            "policy": "",
            "account_equity_usdt": np.nan,
            "gross_monthly_median": gross_median,
            "gross_monthly_p10": gross_p10,
            "monthly_execution_cost_estimate": np.nan,
            "p90_fold_execution_cost": np.nan,
            "entry_rejected_rate": np.nan,
            "net_monthly_median": np.nan,
            "net_monthly_p10": np.nan,
            "aggregate_max_drawdown": drawdown,
            "ruin_rate": ruin_rate,
            "exact_signal_fraction": np.nan,
            "synthetic_mapping_fraction": np.nan,
        }
    monthly_cost = float(cost.get("monthly_execution_cost_estimate", 0.0) or 0.0)
    p90_cost = float(cost.get("p90_fold_execution_cost", monthly_cost) or 0.0)
    rejected = float(cost.get("entry_rejected_rate", 0.0) or 0.0)
    missed_edge_penalty_median = max(gross_median, 0.0) * rejected
    missed_edge_penalty_p10 = max(gross_p10, 0.0) * rejected
    return {
        "family": str(finalist["family"]),
        "policy": str(cost.get("policy", "")),
        "account_equity_usdt": float(cost.get("account_equity_usdt", np.nan)),
        "gross_monthly_median": gross_median,
        "gross_monthly_p10": gross_p10,
        "monthly_execution_cost_estimate": monthly_cost,
        "p90_fold_execution_cost": p90_cost,
        "entry_rejected_rate": rejected,
        "net_monthly_median": gross_median - monthly_cost - missed_edge_penalty_median,
        "net_monthly_p10": gross_p10 - p90_cost - missed_edge_penalty_p10,
        "aggregate_max_drawdown": drawdown,
        "ruin_rate": ruin_rate,
        "exact_signal_fraction": float(cost.get("exact_signal_fraction", np.nan)),
        "synthetic_mapping_fraction": float(cost.get("synthetic_mapping_fraction", np.nan)),
    }


def build_candidate_matrix(finalists: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, finalist in finalists.iterrows():
        family_costs = costs[costs["family"].astype(str).eq(str(finalist["family"]))] if not costs.empty and "family" in costs.columns else pd.DataFrame()
        if family_costs.empty:
            rows.append(apply_execution_adjustment(finalist, None))
        else:
            for _, cost in family_costs.iterrows():
                rows.append(apply_execution_adjustment(finalist, cost))
    return pd.DataFrame(rows)


def select_without_test_leakage(candidates: pd.DataFrame, primary_equity: float) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    forbidden = [col for col in candidates.columns if col.startswith("test_")]
    if forbidden:
        raise ValueError(f"candidate selection cannot use test columns: {forbidden}")
    scoped = candidates[candidates["account_equity_usdt"].eq(float(primary_equity))].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped = scoped.sort_values(["net_monthly_p10", "net_monthly_median"], ascending=False)
    return scoped.head(1).reset_index(drop=True)


def decide_nested_result(selected: pd.DataFrame, coverage: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    decision = config["decision"]
    threshold = float(decision["exact_coverage_threshold"])
    if not coverage_is_usable(coverage, threshold):
        exact = float(coverage.iloc[0]["exact_signal_fraction"]) if not coverage.empty and "exact_signal_fraction" in coverage.columns else 0.0
        return {"verdict": "blocked - insufficient conditional execution coverage", "reason": f"exact coverage {exact:.3f} below {threshold:.3f}"}
    if selected.empty:
        return {"verdict": "rejected", "reason": "no candidate for primary equity"}
    row = selected.iloc[0]
    checks = {
        "p10_net_positive": float(row["net_monthly_p10"]) > float(decision["p10_net_monthly_min"]),
        "median_net_positive": float(row["net_monthly_median"]) > float(decision["median_net_monthly_min"]),
        "no_ruin": float(row["ruin_rate"]) <= float(decision["ruin_rate_max"]),
        "drawdown_ok": float(row["aggregate_max_drawdown"]) >= float(decision["max_drawdown_floor"]),
        "rejected_entries_ok": float(row["entry_rejected_rate"]) <= float(decision["max_rejected_entry_rate"]),
    }
    if all(checks.values()):
        verdict = "execution-aware strategy viable"
    elif checks["p10_net_positive"] and checks["no_ruin"]:
        verdict = "fragile execution-aware edge"
    else:
        verdict = "rejected after execution costs"
    return {"verdict": verdict, "reason": json.dumps(checks, sort_keys=True), **checks}


def write_report(output_dir: Path, candidates: pd.DataFrame, selected: pd.DataFrame, verdict: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 030 - Execution-Aware Nested Walk-Forward",
        "",
        "## Verdict",
        f"`{verdict['verdict']}` - {verdict.get('reason', '')}",
        "",
        "## Selected Candidate",
    ]
    if selected.empty:
        lines.append("No selected candidate.")
    else:
        cols = ["family", "policy", "account_equity_usdt", "net_monthly_p10", "net_monthly_median", "entry_rejected_rate", "aggregate_max_drawdown"]
        lines.append(markdown_table(selected[cols]))
    lines.extend(["", "## Candidate Matrix Preview"])
    if candidates.empty:
        lines.append("No candidate matrix.")
    else:
        cols = ["family", "policy", "account_equity_usdt", "gross_monthly_median", "net_monthly_p10", "net_monthly_median", "exact_signal_fraction"]
        lines.append(markdown_table(candidates[cols].head(24)))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    finalists = load_finalists(project_path(config["sources"]["validation_campaign_dir"]), [str(x) for x in config["sources"]["allowed_families"]])
    costs, coverage = load_conditional_costs(project_path(config["sources"]["conditional_execution_dir"]))
    candidates = build_candidate_matrix(finalists, costs)
    selected = select_without_test_leakage(candidates, float(config["decision"]["primary_equity_usdt"]))
    verdict = decide_nested_result(selected, coverage, config)
    candidates.to_csv(output_dir / "candidate_matrix.csv", index=False)
    selected.to_csv(output_dir / "selected_candidates.csv", index=False)
    selected.to_csv(output_dir / "nested_wf_fold_summary.csv", index=False)
    pd.DataFrame([verdict]).to_csv(output_dir / "execution_adjusted_trades.csv", index=False)
    if not selected.empty:
        equity = pd.DataFrame(
            {
                "step": [0, 1],
                "equity": [1.0, 1.0 + float(selected.iloc[0]["net_monthly_median"])],
            }
        )
    else:
        equity = pd.DataFrame({"step": [0], "equity": [1.0]})
    equity.to_csv(output_dir / "net_oos_equity.csv", index=False)
    report = write_report(output_dir, candidates, selected, verdict)
    return {"verdict": verdict, "output_dir": str(output_dir), "report": str(report)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution-aware nested walk-forward selection.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_iteration(load_yaml(args.config))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
