from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research.execution_aware_nested_walk_forward_030 import apply_execution_adjustment
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/research_iteration_execution_policy_ablation_031.yaml"


def load_sources(nested_dir: Path, execution_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_path = nested_dir / "candidate_matrix.csv"
    costs_path = execution_dir / "conditional_execution_costs.csv"
    candidates = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()
    costs = pd.read_csv(costs_path) if costs_path.exists() else pd.DataFrame()
    return candidates, costs


def synthesize_missing_policies(costs: pd.DataFrame, policies: list[str], baseline_policy: str) -> pd.DataFrame:
    if costs.empty:
        return costs
    frames = [costs.copy()]
    existing = set(costs["policy"].astype(str))
    baseline = costs[costs["policy"].astype(str).eq(baseline_policy)].copy()
    if "taker_only" in policies and "taker_only" not in existing and not baseline.empty:
        taker = baseline.copy()
        taker["policy"] = "taker_only"
        taker["entry_rejected_rate"] = 0.0
        taker["synthetic_policy_proxy"] = True
        frames.append(taker)
    maker_base = costs[costs["policy"].astype(str).eq("maker_entry_add_taker_exit")].copy()
    if "maker_no_chase" in policies and "maker_no_chase" not in existing and not maker_base.empty:
        maker = maker_base.copy()
        maker["policy"] = "maker_no_chase"
        maker["synthetic_policy_proxy"] = True
        frames.append(maker)
    result = pd.concat(frames, ignore_index=True)
    if "synthetic_policy_proxy" not in result.columns:
        result["synthetic_policy_proxy"] = False
    result["synthetic_policy_proxy"] = result["synthetic_policy_proxy"].fillna(False).astype(bool)
    return result[result["policy"].astype(str).isin(policies)].copy()


def build_policy_matrix(candidates: pd.DataFrame, costs: pd.DataFrame, policies: list[str]) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    finalist_cols = ["family", "gross_monthly_median", "gross_monthly_p10", "aggregate_max_drawdown", "ruin_rate"]
    finalists = (
        candidates[finalist_cols]
        .drop_duplicates("family")
        .rename(columns={"gross_monthly_median": "aggregate_monthly_return", "gross_monthly_p10": "mc_fold_block_monthly_p05"})
    )
    rows: list[dict[str, Any]] = []
    for _, finalist in finalists.iterrows():
        family_costs = costs[costs["family"].astype(str).eq(str(finalist["family"]))] if not costs.empty else pd.DataFrame()
        for _, cost in family_costs[family_costs["policy"].astype(str).isin(policies)].iterrows():
            row = apply_execution_adjustment(finalist, cost)
            row["synthetic_policy_proxy"] = bool(cost.get("synthetic_policy_proxy", False))
            rows.append(row)
    return pd.DataFrame(rows)


def policy_trade_attribution(matrix: pd.DataFrame, baseline_policy: str, primary_equity: float) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    scoped = matrix[matrix["account_equity_usdt"].eq(float(primary_equity))].copy()
    rows: list[dict[str, Any]] = []
    for family, group in scoped.groupby("family", sort=True):
        baseline = group[group["policy"].astype(str).eq(baseline_policy)]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        for _, row in group.iterrows():
            edge_lost = max(float(row["gross_monthly_median"]), 0.0) * float(row["entry_rejected_rate"])
            base_edge_lost = max(float(base["gross_monthly_median"]), 0.0) * float(base["entry_rejected_rate"])
            rows.append(
                {
                    "family": family,
                    "policy": row["policy"],
                    "baseline_policy": baseline_policy,
                    "account_equity_usdt": float(row["account_equity_usdt"]),
                    "net_p10_delta": float(row["net_monthly_p10"] - base["net_monthly_p10"]),
                    "net_median_delta": float(row["net_monthly_median"] - base["net_monthly_median"]),
                    "execution_cost_delta": float(row["monthly_execution_cost_estimate"] - base["monthly_execution_cost_estimate"]),
                    "entry_rejected_rate_delta": float(row["entry_rejected_rate"] - base["entry_rejected_rate"]),
                    "edge_lost_to_rejections": edge_lost,
                    "edge_lost_delta": edge_lost - base_edge_lost,
                    "synthetic_policy_proxy": bool(row.get("synthetic_policy_proxy", False)),
                }
            )
    return pd.DataFrame(rows)


def summarize_policy_ablation(matrix: pd.DataFrame, attribution: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    ablation = config["ablation"]
    primary = float(ablation["primary_equity_usdt"])
    min_p10 = float(ablation["min_net_p10_improvement"])
    min_median = float(ablation["min_net_median_improvement"])
    max_rejected = float(ablation["max_entry_rejected_rate"])
    rows: list[dict[str, Any]] = []
    scoped = matrix[matrix["account_equity_usdt"].eq(primary)]
    for _, row in scoped.iterrows():
        attr = attribution[(attribution["family"].eq(row["family"])) & (attribution["policy"].eq(row["policy"]))]
        attr_row = attr.iloc[0] if not attr.empty else pd.Series(dtype=object)
        improves = (
            float(attr_row.get("net_p10_delta", 0.0)) > min_p10
            and float(attr_row.get("net_median_delta", 0.0)) > min_median
            and float(row["entry_rejected_rate"]) <= max_rejected
            and not bool(row.get("synthetic_policy_proxy", False))
        )
        rows.append(
            {
                "family": row["family"],
                "policy": row["policy"],
                "account_equity_usdt": float(row["account_equity_usdt"]),
                "net_monthly_p10": float(row["net_monthly_p10"]),
                "net_monthly_median": float(row["net_monthly_median"]),
                "entry_rejected_rate": float(row["entry_rejected_rate"]),
                "net_p10_delta_vs_baseline": float(attr_row.get("net_p10_delta", 0.0)),
                "net_median_delta_vs_baseline": float(attr_row.get("net_median_delta", 0.0)),
                "synthetic_policy_proxy": bool(row.get("synthetic_policy_proxy", False)),
                "policy_improves_net": bool(improves),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy_improves_net", "net_monthly_p10"], ascending=False)


def size_viability(matrix: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    max_rejected = float(config["ablation"]["max_entry_rejected_rate"])
    rows: list[dict[str, Any]] = []
    for key, group in matrix.groupby(["family", "policy", "account_equity_usdt"], sort=True):
        row = group.iloc[0]
        rows.append(
            {
                "family": key[0],
                "policy": key[1],
                "account_equity_usdt": float(key[2]),
                "viable": bool(float(row["net_monthly_p10"]) > 0 and float(row["entry_rejected_rate"]) <= max_rejected and float(row["ruin_rate"]) == 0.0),
                "net_monthly_p10": float(row["net_monthly_p10"]),
                "entry_rejected_rate": float(row["entry_rejected_rate"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(output_dir: Path, summary: pd.DataFrame) -> Path:
    report = output_dir / "iteration_report.md"
    if summary.empty:
        verdict = "blocked - missing execution-aware candidate matrix"
    elif summary["policy_improves_net"].astype(bool).any():
        verdict = "execution policy improves net edge"
    else:
        verdict = "no execution policy edge"
    lines = [
        "# Iteration 031 - Execution Policy Ablation",
        "",
        f"## Verdict\n`{verdict}`",
        "",
        "## Policy Summary",
    ]
    if summary.empty:
        lines.append("No policy summary available.")
    else:
        cols = ["family", "policy", "net_monthly_p10", "net_monthly_median", "entry_rejected_rate", "policy_improves_net", "synthetic_policy_proxy"]
        lines.append(markdown_table(summary[cols].head(30)))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, costs = load_sources(project_path(config["sources"]["nested_walk_forward_dir"]), project_path(config["sources"]["conditional_execution_dir"]))
    policies = [str(value) for value in config["ablation"]["policies"]]
    costs = synthesize_missing_policies(costs, policies, str(config["ablation"]["baseline_policy"]))
    matrix = build_policy_matrix(candidates, costs, policies)
    attribution = policy_trade_attribution(matrix, str(config["ablation"]["baseline_policy"]), float(config["ablation"]["primary_equity_usdt"]))
    summary = summarize_policy_ablation(matrix, attribution, config)
    viability = size_viability(matrix, config)
    matrix.to_csv(output_dir / "policy_matrix.csv", index=False)
    summary.to_csv(output_dir / "policy_ablation_summary.csv", index=False)
    attribution.to_csv(output_dir / "policy_trade_attribution.csv", index=False)
    viability.to_csv(output_dir / "size_viability.csv", index=False)
    if not costs.empty:
        costs[costs["policy"].astype(str).str.contains("maker", na=False)].to_csv(output_dir / "missed_fill_diagnostics.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "missed_fill_diagnostics.csv", index=False)
    report = write_report(output_dir, summary)
    verdict = "execution policy improves net edge" if not summary.empty and summary["policy_improves_net"].astype(bool).any() else "no execution policy edge"
    return {"verdict": verdict, "output_dir": str(output_dir), "report": str(report)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution policy ablation for execution-aware candidates.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_iteration(load_yaml(args.config))
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
