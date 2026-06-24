from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.infra.microstructure_quality_report import compute_quality_metrics, load_ws_depth_files
from src.research.microstructure_execution_filter_017 import gate_config_from_yaml, summarize_snapshots
from src.research.microstructure_order_policy_017 import (
    SnapshotCache,
    order_events_from_trades,
    order_policy_from_yaml,
    simulate_policy_order,
    summarize_missed_fills,
    summarize_signal_attribution,
    summarize_slippage_by_equity_policy,
)
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/research_iteration_conditional_execution_measurement_029b.yaml"


def _stable_index(key: str, length: int, seed: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    return int(zlib.crc32(f"{seed}:{key}".encode("utf-8")) % length)


def load_variant_trades(source_dir: Path, variants: dict[str, str], max_trades: int | None = None) -> pd.DataFrame:
    trade_dir = source_dir / "selected_fold_trades"
    if not trade_dir.exists():
        raise FileNotFoundError(f"Iteration 018 selected_fold_trades not found: {trade_dir}")
    frames: list[pd.DataFrame] = []
    for family, raw_variant in variants.items():
        for path in sorted(trade_dir.glob(f"{raw_variant}_fold_*_trades.csv")):
            match = re.search(r"_fold_(\d+)_trades\.csv$", path.name)
            fold_id = int(match.group(1)) if match else -1
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            frame.insert(0, "family", family)
            frame.insert(1, "variant", raw_variant)
            frame.insert(2, "fold_id", fold_id)
            frames.append(frame)
    if not frames:
        raise ValueError("No Iteration 018 trades found for requested variants")
    trades = pd.concat(frames, ignore_index=True)
    trades["start_timestamp"] = pd.to_datetime(trades["start_timestamp"], utc=True)
    trades["exit_timestamp"] = pd.to_datetime(trades["exit_timestamp"], utc=True)
    trades = trades.sort_values(["family", "fold_id", "start_timestamp"]).reset_index(drop=True)
    trades.insert(0, "trade_id", np.arange(len(trades), dtype=int))
    if max_trades is not None:
        trades = trades.head(int(max_trades)).copy()
    return trades


def events_for_variants(trades: pd.DataFrame, equities: list[float]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for family, group in trades.groupby("family", sort=True):
        events = order_events_from_trades(group.copy(), equities)
        if events.empty:
            continue
        events.insert(0, "family", family)
        events.insert(1, "variant", str(group["variant"].iloc[0]))
        parts.append(events)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["family", "event_timestamp", "event_id"]).reset_index(drop=True)


def choose_conditional_snapshot(
    event: pd.Series,
    cache: SnapshotCache,
    gate: Any,
    nearest_snapshot_max_age_ms: float,
    seed: int,
) -> tuple[pd.Series, int, str, float]:
    event_ts = pd.Timestamp(event["event_timestamp"])
    event_ns = int(event_ts.value)
    exact_tolerance_ns = int(pd.Timedelta(milliseconds=gate.max_snapshot_age_ms).value)
    pos = int(np.searchsorted(cache.times_ns, event_ns, side="right") - 1)
    if pos >= 0 and event_ns - int(cache.times_ns[pos]) <= exact_tolerance_ns:
        age_ms = (event_ns - int(cache.times_ns[pos])) / 1_000_000.0
        return cache.frame.iloc[pos], pos, "exact_signal_timestamp", float(age_ms)

    nearest_tolerance_ns = int(pd.Timedelta(milliseconds=float(nearest_snapshot_max_age_ms)).value)
    candidates = []
    right = int(np.searchsorted(cache.times_ns, event_ns, side="left"))
    for candidate in [right - 1, right]:
        if 0 <= candidate < len(cache.frame):
            age_ns = abs(event_ns - int(cache.times_ns[candidate]))
            candidates.append((age_ns, candidate))
    if candidates:
        age_ns, nearest = min(candidates, key=lambda item: item[0])
        if age_ns <= nearest_tolerance_ns:
            return cache.frame.iloc[int(nearest)], int(nearest), "nearest_snapshot", float(age_ns / 1_000_000.0)

    positions = cache.hour_positions.get(int(event_ts.hour), np.array([], dtype=int))
    key = f"{event.get('event_id', '')}:{event_ts.isoformat()}"
    selected = _stable_index(key, len(cache.frame), seed) if len(positions) == 0 else int(positions[_stable_index(key, len(positions), seed)])
    synthetic_age = float(cache.frame.iloc[selected].get("source_latency_ms", np.nan))
    return cache.frame.iloc[selected], selected, "hourly_synthetic_mapping", synthetic_age


def simulate_conditional_orders(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    policy_config: dict[str, Any],
    nearest_snapshot_max_age_ms: float,
) -> pd.DataFrame:
    if snapshots.empty:
        raise ValueError("snapshots cannot be empty")
    if events.empty:
        raise ValueError("events cannot be empty")
    gate = gate_config_from_yaml(policy_config)
    policy = order_policy_from_yaml(policy_config)
    cache = SnapshotCache.from_frame(snapshots)
    gate_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        snapshot, snapshot_pos, mapping_mode, age_ms = choose_conditional_snapshot(
            event,
            cache,
            gate,
            nearest_snapshot_max_age_ms,
            policy.synthetic_mapping_seed,
        )
        for policy_name in policy.policies:
            result = simulate_policy_order(
                policy_name,
                event,
                cache.frame,
                snapshot,
                snapshot_pos,
                gate,
                policy,
                age_ms,
                snapshot_cache=cache,
                gate_cache=gate_cache,
            )
            rows.append(
                {
                    "family": event.get("family", ""),
                    "variant": event.get("variant", ""),
                    "policy": policy_name,
                    "mapping_mode": mapping_mode,
                    "snapshot_time_utc": snapshot["snapshot_time_utc"],
                    "snapshot_age_ms": age_ms,
                    "snapshot_pos": int(snapshot_pos),
                    **event.to_dict(),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def summarize_conditional_costs(simulation: pd.DataFrame) -> pd.DataFrame:
    if simulation.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["family", "policy", "account_equity_usdt"]
    for key, group in simulation.groupby(group_cols, sort=True):
        executed = group[group["executed"].astype(bool)]
        entries = group[group["event_type"].eq("entry")]
        forced = group[group["event_type"].eq("forced_exit")]
        fold_cost = (
            executed.groupby("fold_id")["execution_cost_pct_equity"].sum()
            if not executed.empty and "fold_id" in executed.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "family": key[0],
                "policy": key[1],
                "account_equity_usdt": float(key[2]),
                "order_count": int(len(group)),
                "exact_signal_fraction": float(group["mapping_mode"].eq("exact_signal_timestamp").mean()),
                "nearest_snapshot_fraction": float(group["mapping_mode"].eq("nearest_snapshot").mean()),
                "synthetic_mapping_fraction": float(group["mapping_mode"].eq("hourly_synthetic_mapping").mean()),
                "executed_rate": float(group["executed"].astype(bool).mean()),
                "entry_rejected_rate": float(entries["skipped"].astype(bool).mean()) if not entries.empty else 0.0,
                "forced_exit_executed_rate": float(forced["executed"].astype(bool).mean()) if not forced.empty else 1.0,
                "mean_execution_cost_pct_equity": float(executed["execution_cost_pct_equity"].mean()) if not executed.empty else 0.0,
                "p90_execution_cost_pct_equity": float(executed["execution_cost_pct_equity"].quantile(0.90)) if not executed.empty else 0.0,
                "monthly_execution_cost_estimate": float(fold_cost.mean()) if not fold_cost.empty else 0.0,
                "p90_fold_execution_cost": float(fold_cost.quantile(0.90)) if not fold_cost.empty else 0.0,
                "median_spread_bps": float(group["spread_bps"].median()),
                "p90_slippage_bps": float(group["market_impact_bps"].replace([np.inf, -np.inf], np.nan).quantile(0.90)),
                "mean_order_book_share": float(group["order_book_share"].replace([np.inf, -np.inf], np.nan).mean()),
                "p90_abs_imbalance": float(group["depth_imbalance"].abs().quantile(0.90)) if "depth_imbalance" in group else np.nan,
            }
        )
    return pd.DataFrame(rows)


def coverage_report(simulation: pd.DataFrame, config: dict[str, Any], snapshots: pd.DataFrame) -> pd.DataFrame:
    threshold = float(config["microstructure"]["exact_coverage_threshold"])
    required_days = float(config["microstructure"]["collection_days_required"])
    if snapshots.empty or simulation.empty:
        return pd.DataFrame(
            [
                {
                    "exact_signal_fraction": 0.0,
                    "nearest_snapshot_fraction": 0.0,
                    "synthetic_mapping_fraction": 1.0,
                    "collection_span_days": 0.0,
                    "verdict": "needs more collection",
                    "reason": "empty snapshots or simulation",
                }
            ]
        )
    timestamps = pd.to_datetime(snapshots["snapshot_time_utc"], utc=True)
    span_days = float((timestamps.max() - timestamps.min()) / pd.Timedelta(days=1))
    exact = float(simulation["mapping_mode"].eq("exact_signal_timestamp").mean())
    nearest = float(simulation["mapping_mode"].eq("nearest_snapshot").mean())
    synthetic = float(simulation["mapping_mode"].eq("hourly_synthetic_mapping").mean())
    if span_days < required_days:
        verdict = "needs more collection"
        reason = f"collection span {span_days:.2f}d below required {required_days:.0f}d"
    elif exact < threshold:
        verdict = "no verdict - insufficient exact coverage"
        reason = f"exact coverage {exact:.3f} below threshold {threshold:.3f}"
    else:
        verdict = "conditional execution measurement usable"
        reason = "exact coverage and collection span pass"
    return pd.DataFrame(
        [
            {
                "exact_signal_fraction": exact,
                "nearest_snapshot_fraction": nearest,
                "synthetic_mapping_fraction": synthetic,
                "collection_span_days": span_days,
                "required_collection_days": required_days,
                "exact_coverage_threshold": threshold,
                "verdict": verdict,
                "reason": reason,
            }
        ]
    )


def daily_quality(snapshots: pd.DataFrame, infra_like_config: dict[str, Any]) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    data = snapshots.copy()
    data["snapshot_time_utc"] = pd.to_datetime(data["snapshot_time_utc"], utc=True)
    for day, group in data.groupby(data["snapshot_time_utc"].dt.strftime("%Y-%m-%d"), sort=True):
        metrics = compute_quality_metrics(group, infra_like_config)
        metrics["day_utc"] = day
        rows.append(metrics)
    return pd.DataFrame(rows)


def write_report(output_dir: Path, coverage: pd.DataFrame, costs: pd.DataFrame) -> Path:
    report = output_dir / "iteration_report.md"
    verdict = coverage.iloc[0].to_dict() if not coverage.empty else {"verdict": "needs more collection", "reason": "empty coverage"}
    lines = [
        "# Iteration 029B - Conditional Execution Measurement",
        "",
        f"- verdict: `{verdict.get('verdict')}`",
        f"- reason: {verdict.get('reason')}",
        f"- exact_signal_fraction: `{float(verdict.get('exact_signal_fraction', 0.0)):.3f}`",
        f"- synthetic_mapping_fraction: `{float(verdict.get('synthetic_mapping_fraction', 1.0)):.3f}`",
        "",
        "This run measures execution at real signal timestamps when collected snapshots overlap. It must not be used for a final verdict when exact coverage is below the configured threshold.",
        "",
    ]
    if not costs.empty:
        cols = [
            "family",
            "policy",
            "account_equity_usdt",
            "exact_signal_fraction",
            "entry_rejected_rate",
            "mean_execution_cost_pct_equity",
            "p90_execution_cost_pct_equity",
            "p90_slippage_bps",
        ]
        lines.append(markdown_table(costs[cols].head(24)))
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(
    config: dict[str, Any],
    max_snapshots: int | None = None,
    max_trades: int | None = None,
) -> dict[str, Any]:
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_config = load_yaml(config["microstructure"]["policy_config"])
    equities = [float(value) for value in config["microstructure"].get("account_equity_usdt_grid", policy_config["microstructure_gate"]["account_equity_usdt_grid"])]
    snapshots = pd.DataFrame()
    try:
        snapshots = load_ws_depth_files(
            project_path(config["microstructure"]["ws_depth_dir"]),
            str(config["microstructure"].get("symbol", "BTCUSDT")).upper(),
            days=int(config["microstructure"].get("collection_days_required", 30)),
            max_rows=max_snapshots,
        )
    except (FileNotFoundError, ValueError):
        pass

    trades = load_variant_trades(
        project_path(config["source"]["iteration_018_dir"]),
        {str(k): str(v) for k, v in config["source"]["variants"].items()},
        max_trades=max_trades,
    )
    events = events_for_variants(trades, equities)
    if snapshots.empty:
        simulation = pd.DataFrame()
        costs = pd.DataFrame()
        coverage = coverage_report(simulation, config, snapshots)
        quality = pd.DataFrame()
    else:
        simulation = simulate_conditional_orders(
            snapshots,
            events,
            policy_config,
            float(config["microstructure"].get("nearest_snapshot_max_age_ms", 60000)),
        )
        costs = summarize_conditional_costs(simulation)
        coverage = coverage_report(simulation, config, snapshots)
        infra_like = {
            "quality": {
                "expected_interval_seconds": 1,
                "gap_warn_seconds": 5,
                "stale_bad_seconds": 60,
                "min_coverage_ratio_healthy": 0.95,
                "min_coverage_ratio_degraded": 0.80,
                "max_invalid_fraction_healthy": 0.01,
                "max_invalid_fraction_degraded": 0.05,
            }
        }
        quality = daily_quality(snapshots, infra_like)

    simulation.to_csv(output_dir / "signal_order_costs.csv", index=False)
    costs.to_csv(output_dir / "conditional_execution_costs.csv", index=False)
    coverage.to_csv(output_dir / "coverage_report.csv", index=False)
    quality.to_csv(output_dir / "daily_microstructure_quality.csv", index=False)
    if not simulation.empty:
        summarize_signal_attribution(simulation).to_csv(output_dir / "signal_execution_attribution.csv", index=False)
        summarize_slippage_by_equity_policy(simulation).to_csv(output_dir / "slippage_by_equity_policy.csv", index=False)
        summarize_missed_fills(simulation).to_csv(output_dir / "missed_fill_diagnostics.csv", index=False)
        summarize_snapshots(snapshots, gate_config_from_yaml(policy_config)).to_csv(output_dir / "microstructure_snapshots_summary.csv", index=False)
    report = write_report(output_dir, coverage, costs)
    return {
        "coverage": coverage.iloc[0].to_dict(),
        "output_dir": str(output_dir),
        "report": str(report),
        "order_count": int(len(simulation)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure conditional execution costs at strategy signal timestamps.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--max-trades", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_iteration(load_yaml(args.config), max_snapshots=args.max_snapshots, max_trades=args.max_trades)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
