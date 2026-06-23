from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.infra.microstructure_quality_report import load_ws_depth_files
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.microstructure_execution_filter_017 import gate_config_from_yaml, load_locked_017_candidates, summarize_snapshots
from src.research.microstructure_order_policy_017 import (
    load_locked_017_trades,
    order_events_from_trades,
    simulate_order_policies,
    summarize_missed_fills,
    summarize_policy_comparison,
    summarize_signal_attribution,
    summarize_slippage_by_equity_policy,
)
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_rolling_execution_evaluation.yaml"


def load_rolling_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    if "source_policy_config" not in config:
        raise ValueError("rolling execution config requires source_policy_config")
    if "data" not in config or "ws_depth_dir" not in config["data"]:
        raise ValueError("rolling execution config requires data.ws_depth_dir")
    return config


def load_policy_config(config: dict[str, Any]) -> dict[str, Any]:
    return load_yaml(config["source_policy_config"])


def normalize_depth_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("snapshot frame cannot be empty")
    normalized = frame.copy()
    normalized["snapshot_time_utc"] = pd.to_datetime(normalized["snapshot_time_utc"], utc=True)
    return normalized.sort_values("snapshot_time_utc").reset_index(drop=True)


def evaluate_policy_events(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    policy_config: dict[str, Any],
    window_label: str,
) -> dict[str, pd.DataFrame]:
    snapshots = normalize_depth_frame(snapshots)
    if events.empty:
        raise ValueError("order events cannot be empty")
    gate = gate_config_from_yaml(policy_config)
    snapshot_summary = summarize_snapshots(snapshots, gate)
    simulation = simulate_order_policies(snapshots, events, policy_config)
    comparison = summarize_policy_comparison(simulation, policy_config, snapshot_summary)
    attribution = summarize_signal_attribution(simulation)
    slippage = summarize_slippage_by_equity_policy(simulation)
    missed = summarize_missed_fills(simulation)
    for frame in [snapshot_summary, simulation, comparison, attribution, slippage, missed]:
        if not frame.empty:
            frame.insert(0, "window_label", window_label)
    return {
        "snapshot_summary": snapshot_summary,
        "simulation": simulation,
        "comparison": comparison,
        "attribution": attribution,
        "slippage": slippage,
        "missed": missed,
    }


def make_window_slices(snapshots: pd.DataFrame, rolling_hours: list[float]) -> list[tuple[str, pd.DataFrame]]:
    data = normalize_depth_frame(snapshots)
    end = data["snapshot_time_utc"].max()
    windows: list[tuple[str, pd.DataFrame]] = []
    for hours in rolling_hours:
        start = end - pd.Timedelta(hours=float(hours))
        label = f"rolling_{int(hours)}h"
        windows.append((label, data[data["snapshot_time_utc"].ge(start)].copy()))
    return windows


def make_daily_slices(snapshots: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    data = normalize_depth_frame(snapshots)
    slices: list[tuple[str, pd.DataFrame]] = []
    for day, group in data.groupby(data["snapshot_time_utc"].dt.strftime("%Y-%m-%d"), sort=True):
        slices.append((f"day_{day}", group.copy()))
    return slices


def decide_execution_viability(
    rolling_comparison: pd.DataFrame,
    rolling_quality: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    verdict_cfg = config["verdict"]
    if rolling_comparison.empty or rolling_quality.empty:
        return {"verdict": "needs more collection", "reason": "empty comparison or quality"}
    latest_label = "rolling_24h"
    quality = rolling_quality[rolling_quality["window_label"].eq(latest_label)]
    comparison = rolling_comparison[rolling_comparison["window_label"].eq(latest_label)]
    if quality.empty:
        quality = rolling_quality.tail(1)
    if comparison.empty:
        comparison = rolling_comparison.tail(0)
    quality_row = quality.iloc[0]
    if float(quality_row["collection_span_hours"]) < float(verdict_cfg["min_collection_hours"]):
        return {"verdict": "needs more collection", "reason": "less than 24h of usable snapshots"}
    if float(quality_row["invalid_snapshot_fraction"]) > float(verdict_cfg["max_invalid_snapshot_fraction"]):
        return {"verdict": "needs more collection", "reason": "invalid snapshot fraction too high"}
    required_ceiling = float(verdict_cfg["required_equity_ceiling"])
    max_slippage = float(verdict_cfg["max_p90_slippage_bps"])
    max_skipped = float(verdict_cfg["max_entry_skipped_rate"])
    monthly_floor = float(verdict_cfg["adjusted_monthly_floor"])
    viable = comparison[
        comparison["account_equity_usdt"].le(required_ceiling)
        & comparison["p90_slippage_bps"].le(max_slippage)
        & comparison["entry_skipped_rate"].le(max_skipped)
        & comparison["estimated_monthly_after_execution"].ge(monthly_floor)
    ]
    viable_equities = sorted(float(value) for value in viable["account_equity_usdt"].dropna().unique())
    required_equities = sorted(float(value) for value in comparison["account_equity_usdt"].dropna().unique() if value <= required_ceiling)
    passes_required = bool(required_equities) and set(required_equities).issubset(set(viable_equities))
    if not passes_required:
        return {
            "verdict": "execution not viable",
            "reason": f"not all <= {required_ceiling:g} USDT equities pass",
            "viable_equities": viable_equities,
        }
    size_cap = float(verdict_cfg.get("size_cap_probe", 25000))
    probe = comparison[comparison["account_equity_usdt"].eq(size_cap)]
    probe_viable = probe[
        probe["p90_slippage_bps"].le(max_slippage)
        & probe["entry_skipped_rate"].le(max_skipped)
        & probe["estimated_monthly_after_execution"].ge(monthly_floor)
    ]
    if not probe.empty and probe_viable.empty:
        return {
            "verdict": "size capped",
            "reason": f"{size_cap:g} USDT fails while smaller equities pass",
            "viable_equities": viable_equities,
        }
    return {"verdict": "execution viable by size", "reason": "required equities pass", "viable_equities": viable_equities}


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    rolling = pd.DataFrame(payload["rolling_policy_comparison"])
    daily = pd.DataFrame(payload["daily_policy_comparison"])
    verdict = payload["verdict"]
    lines = [
        "# Iteration 026 - Rolling Execution Evaluation",
        "",
        "## Verdict",
        f"`{verdict['verdict']}` - {verdict.get('reason', '')}",
        "",
        "## Rolling Windows",
    ]
    if rolling.empty:
        lines.append("No rolling comparison available.")
    else:
        cols = [
            "window_label",
            "policy",
            "account_equity_usdt",
            "entry_skipped_rate",
            "p90_slippage_bps",
            "estimated_monthly_after_execution",
            "synthetic_mapping_fraction",
        ]
        lines.append(markdown_table(rolling[cols].head(48)))
    lines.extend(["", "## Daily Sample"])
    if daily.empty:
        lines.append("No daily comparison available.")
    else:
        cols = [
            "window_label",
            "policy",
            "account_equity_usdt",
            "entry_skipped_rate",
            "p90_slippage_bps",
            "estimated_monthly_after_execution",
        ]
        lines.append(markdown_table(daily[cols].head(48)))
    lines.extend(
        [
            "",
            "## Guardrails",
            "This run reuses locked Iteration 017 trades/candidates through the Iteration 022 execution simulator. It does not reselect strategy parameters.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_rolling_evaluation(
    config: dict[str, Any],
    days: int | None = None,
    max_snapshots: int | None = None,
    max_trades: int | None = None,
) -> dict[str, Any]:
    policy_config = load_policy_config(config)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ws_dir = project_path(config["data"]["ws_depth_dir"])
    symbol = str(config["data"].get("symbol", "BTCUSDT")).upper()
    days = int(days or config["data"].get("default_days", 7))
    snapshots = load_ws_depth_files(ws_dir, symbol, days=days, max_rows=max_snapshots)
    locked = load_locked_017_candidates(policy_config)
    trades = load_locked_017_trades(policy_config, max_trades=max_trades)
    equities = [float(value) for value in policy_config["microstructure_gate"]["account_equity_usdt_grid"]]
    events = order_events_from_trades(trades, equities)

    rolling_frames: dict[str, list[pd.DataFrame]] = {
        "snapshot_summary": [],
        "comparison": [],
        "simulation": [],
        "attribution": [],
        "slippage": [],
        "missed": [],
    }
    rolling_hours = [float(value) for value in config.get("windows", {}).get("rolling_hours", [24, 168, 720])]
    for label, window_snapshots in make_window_slices(snapshots, rolling_hours):
        if window_snapshots.empty:
            continue
        result = evaluate_policy_events(window_snapshots, events, policy_config, label)
        for key, frame in result.items():
            rolling_frames[key].append(frame)

    daily_frames: dict[str, list[pd.DataFrame]] = {"snapshot_summary": [], "comparison": []}
    for label, day_snapshots in make_daily_slices(snapshots):
        result = evaluate_policy_events(day_snapshots, events, policy_config, label)
        daily_frames["snapshot_summary"].append(result["snapshot_summary"])
        daily_frames["comparison"].append(result["comparison"])

    rolling_quality = pd.concat(rolling_frames["snapshot_summary"], ignore_index=True) if rolling_frames["snapshot_summary"] else pd.DataFrame()
    rolling_comparison = pd.concat(rolling_frames["comparison"], ignore_index=True) if rolling_frames["comparison"] else pd.DataFrame()
    rolling_simulation = pd.concat(rolling_frames["simulation"], ignore_index=True) if rolling_frames["simulation"] else pd.DataFrame()
    rolling_attribution = pd.concat(rolling_frames["attribution"], ignore_index=True) if rolling_frames["attribution"] else pd.DataFrame()
    rolling_slippage = pd.concat(rolling_frames["slippage"], ignore_index=True) if rolling_frames["slippage"] else pd.DataFrame()
    rolling_missed = pd.concat(rolling_frames["missed"], ignore_index=True) if rolling_frames["missed"] else pd.DataFrame()
    daily_quality = pd.concat(daily_frames["snapshot_summary"], ignore_index=True) if daily_frames["snapshot_summary"] else pd.DataFrame()
    daily_comparison = pd.concat(daily_frames["comparison"], ignore_index=True) if daily_frames["comparison"] else pd.DataFrame()
    verdict = decide_execution_viability(rolling_comparison, rolling_quality, config)

    rolling_quality.to_csv(output_dir / "rolling_collection_quality.csv", index=False)
    rolling_comparison.to_csv(output_dir / "rolling_policy_comparison.csv", index=False)
    rolling_simulation.to_csv(output_dir / "rolling_order_execution_simulation.csv", index=False)
    rolling_attribution.to_csv(output_dir / "rolling_signal_execution_attribution.csv", index=False)
    rolling_slippage.to_csv(output_dir / "rolling_slippage_by_equity_policy.csv", index=False)
    rolling_missed.to_csv(output_dir / "rolling_missed_fill_diagnostics.csv", index=False)
    daily_quality.to_csv(output_dir / "daily_collection_quality.csv", index=False)
    daily_comparison.to_csv(output_dir / "daily_policy_comparison.csv", index=False)

    payload = {
        "iteration_name": config["iteration"]["name"],
        "verdict": verdict,
        "snapshot_count": int(len(snapshots)),
        "locked_candidate_count": int(len(locked)),
        "trade_count": int(len(trades)),
        "order_event_count": int(len(events)),
        "rolling_policy_comparison": rolling_comparison.to_dict("records"),
        "daily_policy_comparison": daily_comparison.to_dict("records"),
    }
    (output_dir / "rolling_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = write_report(output_dir, payload)
    LOGGER.info("Wrote Iteration 026 outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling execution evaluation from WS depth parquets.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--max-trades", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_rolling_config(args.config)
    max_snapshots = args.max_snapshots
    max_trades = args.max_trades
    if args.smoke:
        max_snapshots = 500 if max_snapshots is None else max_snapshots
        max_trades = 50 if max_trades is None else max_trades
    payload = run_rolling_evaluation(
        config,
        days=args.days,
        max_snapshots=max_snapshots,
        max_trades=max_trades,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
