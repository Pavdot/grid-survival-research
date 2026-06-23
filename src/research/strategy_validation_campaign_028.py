from __future__ import annotations

import argparse
import itertools
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import drawdown_series
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.monte_carlo_oos_robustness import (
    MonteCarloConfig,
    max_drawdown_from_values,
    monthly_return_from_total,
    run_monte_carlo,
)
from src.research.monthly_target_martingale_research import monthly_return_from_equity
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_strategy_validation_campaign_028.yaml"


@dataclass(frozen=True)
class CpcvSplit:
    split_id: int
    train_blocks: tuple[int, ...]
    test_blocks: tuple[int, ...]
    purged_blocks: tuple[int, ...]
    train_fold_ids: tuple[int, ...]
    test_fold_ids: tuple[int, ...]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_path(value)


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def load_family_fold_summary(iteration_dir: Path, variant: str) -> pd.DataFrame:
    path = iteration_dir / f"walk_forward_fold_summary_{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"fold summary not found: {path}")
    frame = pd.read_csv(path)
    require_columns(
        frame,
        [
            "fold_id",
            "test_start",
            "test_end",
            "test_total_return",
            "test_monthly_return",
            "test_positive",
            "test_target_reached",
            "test_equity_ruined",
        ],
        path,
    )
    frame = frame.copy()
    frame["fold_id"] = frame["fold_id"].astype(int)
    frame["test_start"] = pd.to_datetime(frame["test_start"], utc=True)
    frame["test_end"] = pd.to_datetime(frame["test_end"], utc=True)
    if frame["fold_id"].duplicated().any():
        raise ValueError(f"duplicated fold_id values in {path}")
    if (frame["test_end"] <= frame["test_start"]).any():
        raise ValueError(f"non-positive test window in {path}")
    return frame.sort_values("fold_id").reset_index(drop=True)


def load_family_equity(iteration_dir: Path, variant: str) -> pd.Series:
    path = iteration_dir / f"walk_forward_oos_equity_{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(f"OOS equity not found: {path}")
    frame = pd.read_csv(path)
    require_columns(frame, ["timestamp", "equity"], path)
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    equity = pd.Series(frame["equity"].astype(float).to_numpy(), index=timestamp, name="equity")
    equity = equity[~equity.index.duplicated(keep="last")].sort_index()
    if equity.empty:
        raise ValueError(f"OOS equity is empty: {path}")
    return equity


def summarize_family_from_outputs(
    family: dict[str, Any],
    fold_summary: pd.DataFrame,
    equity: pd.Series,
    robust_monthly: float,
    aspirational_monthly: float,
) -> dict[str, Any]:
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if float(equity.iloc[0]) != 0 else -1.0
    monthly = monthly_return_from_equity(equity)
    return {
        "family": str(family["family"]),
        "source_iteration_dir": str(family["source_iteration_dir"]),
        "source_variant": str(family["source_variant"]),
        "wave": str(family.get("wave", "")),
        "family_type": str(family.get("family_type", "")),
        "fold_count": int(len(fold_summary)),
        "aggregate_total_return": total_return,
        "aggregate_monthly_return": float(monthly),
        "aggregate_max_drawdown": float(drawdown_series(equity).min()),
        "positive_fold_rate": float(fold_summary["test_positive"].astype(bool).mean()),
        "target_fold_rate_20pct": float(fold_summary["test_monthly_return"].astype(float).ge(aspirational_monthly).mean()),
        "robust_fold_rate_13pct": float(fold_summary["test_monthly_return"].astype(float).ge(robust_monthly).mean()),
        "test_grid_count": int(fold_summary.get("test_number_of_grids", pd.Series([0] * len(fold_summary))).sum()),
        "worst_fold_monthly_return": float(fold_summary["test_monthly_return"].astype(float).min()),
        "equity_ruined": bool((equity <= 0).any() or fold_summary["test_equity_ruined"].astype(bool).any()),
        "execution_proxy": str(family.get("execution_proxy", "none")),
        "surface_proxy": str(family.get("surface_proxy", "none")),
    }


def build_candidate_matrix(
    families: list[dict[str, Any]],
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series]]:
    rows: list[pd.DataFrame] = []
    fold_by_family: dict[str, pd.DataFrame] = {}
    equity_by_family: dict[str, pd.Series] = {}
    for family in families:
        family_name = str(family["family"])
        iteration_dir = resolve_path(family["source_iteration_dir"])
        variant = str(family["source_variant"])
        fold_summary = load_family_fold_summary(iteration_dir, variant)
        if max_folds is not None:
            fold_summary = fold_summary.head(int(max_folds)).copy()
        equity = load_family_equity(iteration_dir, variant)
        if max_folds is not None and not fold_summary.empty:
            start = fold_summary["test_start"].min()
            end = fold_summary["test_end"].max()
            sliced_equity = equity[equity.index.to_series().between(start, end)]
            if not sliced_equity.empty:
                equity = sliced_equity
        fold_by_family[family_name] = fold_summary
        equity_by_family[family_name] = equity
        frame = fold_summary.copy()
        frame.insert(0, "family", family_name)
        frame.insert(1, "source_iteration_dir", str(family["source_iteration_dir"]))
        frame.insert(2, "source_variant", variant)
        rows.append(frame)
    if not rows:
        raise ValueError("candidate family list is empty")
    return pd.concat(rows, ignore_index=True), fold_by_family, equity_by_family


def summarize_fold_subset(frame: pd.DataFrame, fold_ids: tuple[int, ...], robust_monthly: float) -> dict[str, Any]:
    subset = frame[frame["fold_id"].astype(int).isin(fold_ids)].copy()
    if subset.empty:
        raise ValueError("fold subset is empty")
    returns = subset["test_total_return"].astype(float).to_numpy()
    total_return = float(np.prod(1.0 + returns) - 1.0)
    days = float(((subset["test_end"] - subset["test_start"]) / pd.Timedelta(days=1)).sum())
    monthly = monthly_return_from_total(total_return, days)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    return {
        "fold_count": int(len(subset)),
        "total_return": total_return,
        "monthly_return": monthly,
        "max_drawdown": max_drawdown_from_values(equity),
        "positive_fold_rate": float(subset["test_positive"].astype(bool).mean()),
        "robust_fold_rate": float(subset["test_monthly_return"].astype(float).ge(robust_monthly).mean()),
        "equity_ruined": bool(np.any(equity <= 0) or subset["test_equity_ruined"].astype(bool).any()),
    }


def make_cpcv_blocks(fold_ids: list[int], block_count: int) -> dict[int, tuple[int, ...]]:
    if block_count <= 1:
        raise ValueError("CPCV block_count must be > 1")
    if not fold_ids:
        raise ValueError("fold_ids cannot be empty")
    sorted_folds = sorted(int(value) for value in fold_ids)
    effective_blocks = min(int(block_count), len(sorted_folds))
    chunks = np.array_split(np.asarray(sorted_folds, dtype=int), effective_blocks)
    blocks = {block_id: tuple(int(value) for value in chunk.tolist()) for block_id, chunk in enumerate(chunks)}
    if any(len(values) == 0 for values in blocks.values()):
        raise ValueError("CPCV produced an empty block")
    return blocks


def make_cpcv_splits(
    blocks: dict[int, tuple[int, ...]],
    test_block_count: int,
    purge_adjacent_blocks: int,
) -> list[CpcvSplit]:
    block_ids = sorted(blocks)
    if test_block_count <= 0 or test_block_count >= len(block_ids):
        raise ValueError("CPCV test_block_count must stay within [1, block_count)")
    effective_purge = int(purge_adjacent_blocks)
    if len(block_ids) <= test_block_count + 2 * effective_purge:
        effective_purge = 0
    splits: list[CpcvSplit] = []
    split_id = 0
    for test_blocks_raw in itertools.combinations(block_ids, int(test_block_count)):
        test_blocks = tuple(int(value) for value in test_blocks_raw)
        purged = set(test_blocks)
        for block_id in test_blocks:
            for offset in range(1, effective_purge + 1):
                if block_id - offset in blocks:
                    purged.add(block_id - offset)
                if block_id + offset in blocks:
                    purged.add(block_id + offset)
        train_blocks = tuple(block_id for block_id in block_ids if block_id not in purged)
        if not train_blocks:
            continue
        train_folds = tuple(fold for block_id in train_blocks for fold in blocks[block_id])
        test_folds = tuple(fold for block_id in test_blocks for fold in blocks[block_id])
        if not train_folds or not test_folds:
            continue
        splits.append(
            CpcvSplit(
                split_id=split_id,
                train_blocks=train_blocks,
                test_blocks=test_blocks,
                purged_blocks=tuple(sorted(purged - set(test_blocks))),
                train_fold_ids=train_folds,
                test_fold_ids=test_folds,
            )
        )
        split_id += 1
    if not splits:
        raise ValueError("CPCV produced no valid splits")
    return splits


def run_cpcv(
    candidate_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    robust = float(config["target"]["robust_monthly_return"])
    cpcv_cfg = config["cpcv"]
    fold_ids = sorted(candidate_matrix["fold_id"].astype(int).unique().tolist())
    blocks = make_cpcv_blocks(fold_ids, int(cpcv_cfg["block_count"]))
    test_count = min(int(cpcv_cfg["test_block_count"]), max(1, len(blocks) - 1))
    splits = make_cpcv_splits(blocks, test_count, int(cpcv_cfg.get("purge_adjacent_blocks", 0)))
    path_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    families = sorted(candidate_matrix["family"].astype(str).unique().tolist())

    for split in splits:
        train_scores: list[dict[str, Any]] = []
        for family in families:
            frame = candidate_matrix[candidate_matrix["family"].eq(family)]
            metrics = summarize_fold_subset(frame, split.train_fold_ids, robust)
            train_scores.append({"family": family, **{f"train_{key}": value for key, value in metrics.items()}})
        train_frame = pd.DataFrame(train_scores).sort_values(
            ["train_monthly_return", "train_positive_fold_rate", "train_robust_fold_rate"],
            ascending=[False, False, False],
        )
        selected_family = str(train_frame.iloc[0]["family"])
        selected_frame = candidate_matrix[candidate_matrix["family"].eq(selected_family)]
        test_metrics = summarize_fold_subset(selected_frame, split.test_fold_ids, robust)
        path_rows.append(
            {
                "split_id": split.split_id,
                "selected_family": selected_family,
                "train_blocks": ",".join(map(str, split.train_blocks)),
                "test_blocks": ",".join(map(str, split.test_blocks)),
                "purged_blocks": ",".join(map(str, split.purged_blocks)),
                "train_fold_ids": ",".join(map(str, split.train_fold_ids)),
                "test_fold_ids": ",".join(map(str, split.test_fold_ids)),
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

    for family in families:
        frame = candidate_matrix[candidate_matrix["family"].eq(family)]
        monthly_values: list[float] = []
        dd_values: list[float] = []
        ruin_values: list[bool] = []
        for split in splits:
            metrics = summarize_fold_subset(frame, split.test_fold_ids, robust)
            monthly_values.append(float(metrics["monthly_return"]))
            dd_values.append(float(metrics["max_drawdown"]))
            ruin_values.append(bool(metrics["equity_ruined"]))
        monthly = pd.Series(monthly_values, dtype=float)
        candidate_rows.append(
            {
                "family": family,
                "cpcv_path_count": int(len(monthly_values)),
                "cpcv_monthly_p25": float(monthly.quantile(0.25)),
                "cpcv_monthly_median": float(monthly.median()),
                "cpcv_monthly_p75": float(monthly.quantile(0.75)),
                "cpcv_max_drawdown_median": float(pd.Series(dd_values, dtype=float).median()),
                "cpcv_ruin_rate": float(pd.Series(ruin_values, dtype=bool).mean()),
                "cpcv_pass": bool(monthly.median() >= robust and monthly.quantile(0.25) > 0 and not any(ruin_values)),
            }
        )
    return pd.DataFrame(candidate_rows), pd.DataFrame(path_rows)


def run_monte_carlo_gate(
    families: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    iterations_override: int | None = None,
    skip_monte_carlo: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    robust = float(config["target"]["robust_monthly_return"])
    iterations = int(iterations_override or config["monte_carlo"]["iterations"])
    for family in families:
        family_name = str(family["family"])
        if skip_monte_carlo:
            rows.append({"family": family_name, "mc_available": False, "mc_pass": False, "mc_decision": "skipped"})
            continue
        try:
            payload = run_monte_carlo(
                MonteCarloConfig(
                    iteration_dir=resolve_path(family["source_iteration_dir"]),
                    variant=str(family["source_variant"]),
                    iterations=iterations,
                    seed=int(config["monte_carlo"].get("seed", 42)),
                    target_monthly_return=robust,
                    output_dir=output_dir / "monte_carlo" / family_name,
                )
            )
            by_method = {row["method"]: row for row in payload["monte_carlo_summary"]}
            block = by_method.get("fold_block_bootstrap") or by_method.get("fold_bootstrap")
            if block is None:
                raise ValueError("Monte Carlo missing fold-block summary")
            mc_pass = (
                float(block["monthly_return_p50"]) >= robust
                and float(block["monthly_return_p05"]) > 0
                and float(block["positive_probability"]) >= 0.95
                and float(block["ruin_probability"]) <= 0.01
            )
            rows.append(
                {
                    "family": family_name,
                    "mc_available": True,
                    "mc_pass": bool(mc_pass),
                    "mc_decision": payload["decision"],
                    "mc_fold_block_monthly_p05": float(block["monthly_return_p05"]),
                    "mc_fold_block_monthly_median": float(block["monthly_return_p50"]),
                    "mc_fold_block_monthly_p95": float(block["monthly_return_p95"]),
                    "mc_fold_block_positive_probability": float(block["positive_probability"]),
                    "mc_fold_block_ruin_probability": float(block["ruin_probability"]),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report unavailable gate explicitly.
            rows.append(
                {
                    "family": family_name,
                    "mc_available": False,
                    "mc_pass": False,
                    "mc_decision": f"error: {exc}",
                }
            )
    return pd.DataFrame(rows)


def worst_case_gate(families: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config["execution"]["worst_case_iteration_dir"]) / "scenario_comparison.csv"
    robust = float(config["target"]["robust_monthly_return"])
    rows: list[dict[str, Any]] = []
    scenarios = pd.read_csv(path) if path.exists() else pd.DataFrame()
    scenario_index = scenarios.set_index("scenario") if not scenarios.empty and "scenario" in scenarios.columns else pd.DataFrame()
    for family in families:
        family_name = str(family["family"])
        proxy = str(family.get("execution_proxy", "none"))
        if proxy != "locked_017" or scenario_index.empty:
            rows.append(
                {
                    "family": family_name,
                    "worst_case_available": False,
                    "worst_case_pass": False,
                    "worst_case_reason": "no execution proxy available",
                }
            )
            continue
        required = ["worst_case_primary_locked", "fee_0p0006_slip_5_locked", "next_bar_open_conservative_locked"]
        missing = [name for name in required if name not in scenario_index.index]
        if missing:
            rows.append(
                {
                    "family": family_name,
                    "worst_case_available": False,
                    "worst_case_pass": False,
                    "worst_case_reason": f"missing scenarios: {missing}",
                }
            )
            continue
        worst = scenario_index.loc["worst_case_primary_locked"]
        cost = scenario_index.loc["fee_0p0006_slip_5_locked"]
        next_bar = scenario_index.loc["next_bar_open_conservative_locked"]
        passed = (
            float(worst["aggregate_monthly_return"]) >= robust
            and float(cost["aggregate_monthly_return"]) >= robust
            and float(next_bar["aggregate_monthly_return"]) >= robust
            and not bool(worst["equity_ruined"])
            and not bool(cost["equity_ruined"])
        )
        rows.append(
            {
                "family": family_name,
                "worst_case_available": True,
                "worst_case_pass": bool(passed),
                "worst_case_reason": "passed" if passed else "primary worst-case or cost stress below robust threshold",
                "worst_case_primary_monthly": float(worst["aggregate_monthly_return"]),
                "cost_0p0006_slip5_monthly": float(cost["aggregate_monthly_return"]),
                "next_bar_conservative_monthly": float(next_bar["aggregate_monthly_return"]),
                "worst_case_primary_ruined": bool(worst["equity_ruined"]),
            }
        )
    return pd.DataFrame(rows)


def execution_surface_gate(families: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config["execution"]["surface_iteration_dir"]) / "execution_surface_summary.csv"
    robust = float(config["target"]["robust_monthly_return"])
    rows: list[dict[str, Any]] = []
    surface = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not surface.empty:
        zone = surface[
            surface["fee_rate"].astype(float).le(float(config["execution"]["protected_fee_max"]))
            & surface["slippage_bps"].astype(float).le(float(config["execution"]["protected_slippage_bps_max"]))
            & surface["fill_rate"].astype(float).ge(float(config["execution"]["protected_fill_rate_min"]))
        ].copy()
    else:
        zone = pd.DataFrame()
    for family in families:
        family_name = str(family["family"])
        proxy = str(family.get("surface_proxy", "none"))
        if proxy != "locked_017" or zone.empty:
            rows.append(
                {
                    "family": family_name,
                    "surface_available": False,
                    "surface_full_zone_pass": False,
                    "surface_any_zone_pass": False,
                    "surface_reason": "no surface proxy available",
                }
            )
            continue
        full_pass = bool(
            zone["monthly_return_median"].astype(float).ge(robust).all()
            and zone["positive_probability"].astype(float).ge(0.95).all()
            and zone["ruin_probability"].astype(float).le(0.01).all()
        )
        any_pass = bool(
            (
                zone["monthly_return_median"].astype(float).ge(robust)
                & zone["positive_probability"].astype(float).ge(0.95)
                & zone["ruin_probability"].astype(float).le(0.01)
            ).any()
        )
        rows.append(
            {
                "family": family_name,
                "surface_available": True,
                "surface_full_zone_pass": full_pass,
                "surface_any_zone_pass": any_pass,
                "surface_reason": "passed protected zone" if full_pass else "protected zone partially or fully below robust threshold",
                "surface_zone_cells": int(len(zone)),
                "surface_best_monthly_median": float(zone["monthly_return_median"].max()),
                "surface_worst_monthly_median": float(zone["monthly_return_median"].min()),
                "surface_best_monthly_p10": float(zone["monthly_return_p10"].max()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_finalists(
    universe: pd.DataFrame,
    cpcv: pd.DataFrame,
    mc: pd.DataFrame,
    worst: pd.DataFrame,
    surface: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    robust = float(config["target"]["robust_monthly_return"])
    min_improvement = float(config["target"].get("min_improvement_vs_017", 0.0))
    min_positive = float(config["target"].get("min_positive_fold_rate", 0.60))
    baseline_rows = universe[universe["family"].eq("baseline_017_locked")]
    baseline_monthly = float(baseline_rows.iloc[0]["aggregate_monthly_return"]) if not baseline_rows.empty else 0.0
    final = universe.merge(cpcv, on="family", how="left").merge(mc, on="family", how="left")
    final = final.merge(worst, on="family", how="left").merge(surface, on="family", how="left")
    final["improvement_vs_017_monthly"] = final["aggregate_monthly_return"].astype(float) - baseline_monthly
    final["wf_pass"] = (
        final["aggregate_monthly_return"].astype(float).ge(robust)
        & final["improvement_vs_017_monthly"].astype(float).ge(min_improvement)
        & final["positive_fold_rate"].astype(float).ge(min_positive)
        & ~final["equity_ruined"].astype(bool)
    )
    final["research_pass"] = final["wf_pass"].astype(bool) & final["mc_pass"].fillna(False).astype(bool) & final[
        "cpcv_pass"
    ].fillna(False).astype(bool)
    final["execution_pass"] = final["worst_case_pass"].fillna(False).astype(bool) & final[
        "surface_full_zone_pass"
    ].fillna(False).astype(bool)
    verdicts: list[str] = []
    for _, row in final.iterrows():
        if bool(row["research_pass"]) and bool(row["execution_pass"]):
            verdicts.append("execution robust")
        elif bool(row["research_pass"]):
            verdicts.append("fragile but promising")
        elif bool(row["wf_pass"]):
            verdicts.append("research rejected by robustness gates")
        else:
            verdicts.append("rejected")
    final["final_verdict"] = verdicts
    return final.sort_values(
        ["research_pass", "execution_pass", "aggregate_monthly_return"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def copy_finalist_trades(finalists: pd.DataFrame, families: list[dict[str, Any]], output_dir: Path) -> None:
    finalist_dir = output_dir / "finalist_trades"
    finalist_dir.mkdir(parents=True, exist_ok=True)
    family_by_name = {str(family["family"]): family for family in families}
    for _, row in finalists.iterrows():
        if str(row["final_verdict"]) == "rejected":
            continue
        family = family_by_name[str(row["family"])]
        source_dir = resolve_path(family["source_iteration_dir"]) / "selected_fold_trades"
        if not source_dir.exists():
            continue
        variant = str(family["source_variant"])
        for path in source_dir.glob(f"{variant}_fold_*_trades.csv"):
            target = finalist_dir / f"{row['family']}_{path.name}"
            shutil.copy2(path, target)


def global_decision(finalists: pd.DataFrame) -> str:
    if finalists["final_verdict"].eq("execution robust").any():
        return "execution robust"
    if finalists["final_verdict"].eq("fragile but promising").any():
        return "fragile but promising"
    if finalists["wf_pass"].astype(bool).any():
        return "rejected by robustness gates"
    return "rejected"


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    finalists = pd.DataFrame(payload["walk_forward_finalist_summary"])
    cpcv_paths = pd.DataFrame(payload["cpcv_selection_paths"])
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 028 - Strategy Validation Campaign 017+",
        "",
        "## Decision",
        f"`{payload['decision']}`",
        "",
        "## Finalist Summary",
    ]
    cols = [
        "family",
        "final_verdict",
        "aggregate_monthly_return",
        "improvement_vs_017_monthly",
        "positive_fold_rate",
        "wf_pass",
        "mc_pass",
        "cpcv_pass",
        "worst_case_pass",
        "surface_full_zone_pass",
    ]
    lines.append(markdown_table(finalists[cols]) if not finalists.empty else "No finalists.")
    lines.extend(["", "## CPCV Selection Paths"])
    path_cols = ["split_id", "selected_family", "train_blocks", "test_blocks", "test_monthly_return", "test_equity_ruined"]
    lines.append(markdown_table(cpcv_paths[path_cols].head(20)) if not cpcv_paths.empty else "No CPCV paths.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "Iteration 028 is a validation campaign, not a live-trading system. It accepts a candidate only after walk-forward, Monte Carlo, CPCV, worst-case execution and execution-surface checks. A `fragile but promising` result means the research gates passed but execution robustness did not.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_campaign(
    config_path: str | Path = DEFAULT_CONFIG,
    max_folds: int | None = None,
    monte_carlo_iterations: int | None = None,
    skip_monte_carlo: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    families = list(config.get("families", []))
    if not families:
        raise ValueError("Iteration 028 config requires at least one family")
    output_dir = resolve_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_matrix, fold_by_family, equity_by_family = build_candidate_matrix(families, max_folds=max_folds)
    universe_rows = []
    robust = float(config["target"]["robust_monthly_return"])
    aspirational = float(config["target"]["aspirational_monthly_return"])
    for family in families:
        name = str(family["family"])
        universe_rows.append(
            summarize_family_from_outputs(
                family,
                fold_by_family[name],
                equity_by_family[name],
                robust_monthly=robust,
                aspirational_monthly=aspirational,
            )
        )
    universe = pd.DataFrame(universe_rows)
    cpcv_summary, cpcv_paths = run_cpcv(candidate_matrix, config)
    mc = run_monte_carlo_gate(
        families,
        config,
        output_dir,
        iterations_override=monte_carlo_iterations,
        skip_monte_carlo=skip_monte_carlo,
    )
    worst = worst_case_gate(families, config)
    surface = execution_surface_gate(families, config)
    finalists = evaluate_finalists(universe, cpcv_summary, mc, worst, surface, config)
    decision = global_decision(finalists)
    copy_finalist_trades(finalists, families, output_dir)

    universe.to_csv(output_dir / "candidate_universe.csv", index=False)
    candidate_matrix.to_csv(output_dir / "walk_forward_candidate_matrix.csv", index=False)
    finalists.to_csv(output_dir / "walk_forward_finalist_summary.csv", index=False)
    cpcv_summary.to_csv(output_dir / "cpcv_summary.csv", index=False)
    cpcv_paths.to_csv(output_dir / "cpcv_selection_paths.csv", index=False)
    worst.to_csv(output_dir / "worst_case_finalists.csv", index=False)
    surface.to_csv(output_dir / "execution_surface_finalists.csv", index=False)

    payload = {
        "iteration_name": str(config["iteration"]["name"]),
        "decision": decision,
        "candidate_universe": universe.to_dict("records"),
        "walk_forward_finalist_summary": finalists.to_dict("records"),
        "cpcv_summary": cpcv_summary.to_dict("records"),
        "cpcv_selection_paths": cpcv_paths.to_dict("records"),
        "worst_case_finalists": worst.to_dict("records"),
        "execution_surface_finalists": surface.to_dict("records"),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = write_report(output_dir, payload)
    LOGGER.info("Wrote Iteration 028 outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Iteration 028 strategy validation campaign.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--monte-carlo-iterations", type=int, default=None)
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_folds = args.max_folds
    mc_iterations = args.monte_carlo_iterations
    if args.smoke:
        max_folds = 6 if max_folds is None else max_folds
        mc_iterations = 64 if mc_iterations is None else mc_iterations
    payload = run_campaign(
        args.config,
        max_folds=max_folds,
        monte_carlo_iterations=mc_iterations,
        skip_monte_carlo=bool(args.skip_monte_carlo),
    )
    compact = {
        "decision": payload["decision"],
        "finalists": [
            {
                "family": row["family"],
                "final_verdict": row["final_verdict"],
                "monthly": row["aggregate_monthly_return"],
                "wf_pass": row["wf_pass"],
                "mc_pass": row.get("mc_pass"),
                "cpcv_pass": row.get("cpcv_pass"),
                "worst_case_pass": row.get("worst_case_pass"),
                "surface_full_zone_pass": row.get("surface_full_zone_pass"),
            }
            for row in payload["walk_forward_finalist_summary"]
        ],
    }
    print(json.dumps(compact, indent=2, default=str))


if __name__ == "__main__":
    main()
