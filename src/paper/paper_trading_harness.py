from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.infra.binance_microstructure_collector import collector_config_from_yaml, healthcheck as collector_healthcheck
from src.infra.microstructure_quality_report import compute_quality_metrics, load_ws_depth_files
from src.research.microstructure_execution_filter_017 import load_locked_017_candidates
from src.research.microstructure_order_policy_017 import (
    evaluate_order_gate,
    order_policy_from_yaml,
    side_depth_column_for_action,
)
from src.research.microstructure_execution_filter_017 import gate_config_from_yaml
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/paper_trading_017.yaml"
TRUTHY = {"1", "true", "yes", "y", "on"}


def load_paper_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    if "paper" not in config:
        raise ValueError("paper config requires a paper section")
    return config


def assert_paper_only_environment(config: dict[str, Any]) -> None:
    paper = config["paper"]
    live_env = str(paper.get("live_trading_env_var", "LIVE_TRADING_ENABLED"))
    if os.getenv(live_env, "").strip().lower() in TRUTHY:
        raise RuntimeError(f"{live_env}=true is forbidden in paper-only mode")
    for env_name in paper.get("private_env_vars", []):
        if os.getenv(str(env_name)):
            raise RuntimeError(f"{env_name} is present; paper-only harness refuses private API keys")


def load_latest_ws_snapshot(infra_config: dict[str, Any], max_rows: int = 5000) -> tuple[pd.Series, pd.DataFrame]:
    collector = collector_config_from_yaml(infra_config)
    frame = load_ws_depth_files(collector.output_dir, collector.symbol, max_rows=max_rows)
    if frame.empty:
        raise ValueError("no microstructure snapshots available for paper harness")
    return frame.iloc[-1], frame


def is_event_blackout_active(path: Path, now: pd.Timestamp) -> tuple[bool, str]:
    if not path.exists():
        return False, "blackout_window_file_missing"
    frame = pd.read_csv(path)
    if frame.empty:
        return False, "blackout_window_file_empty"
    start_col = "start_time_utc" if "start_time_utc" in frame.columns else "start_utc"
    end_col = "end_time_utc" if "end_time_utc" in frame.columns else "end_utc"
    if start_col not in frame.columns or end_col not in frame.columns:
        return False, "blackout_window_columns_missing"
    starts = pd.to_datetime(frame[start_col], utc=True)
    ends = pd.to_datetime(frame[end_col], utc=True)
    active = frame[starts.le(now) & ends.ge(now)]
    if active.empty:
        return False, "no_active_blackout"
    first = active.iloc[0]
    return True, str(first.get("category", first.get("title", "event_blackout")))


def unique_base_multipliers(locked: pd.DataFrame) -> list[float]:
    values = sorted(float(value) for value in locked["base_position_size_pct"].dropna().unique())
    if not values:
        raise ValueError("locked Iteration 017 candidates do not contain base_position_size_pct")
    return values


def evaluate_current_entry_gates(
    snapshot: pd.Series,
    policy_config: dict[str, Any],
    equities: list[float],
    multipliers: list[float],
) -> pd.DataFrame:
    gate = gate_config_from_yaml(policy_config)
    policy = order_policy_from_yaml(policy_config)
    rows: list[dict[str, Any]] = []
    for equity in equities:
        for multiplier in multipliers:
            for side, action in [("long", "buy"), ("short", "sell")]:
                notional = float(equity) * float(multiplier)
                side_depth = float(snapshot.get(side_depth_column_for_action(action, gate.depth_band_bps), np.nan))
                result = evaluate_order_gate(
                    snapshot,
                    action,
                    float(equity),
                    notional,
                    gate,
                    policy.max_total_book_share,
                    snapshot_age_ms=float(snapshot.get("source_latency_ms", np.nan)),
                )
                rows.append(
                    {
                        "side": side,
                        "action": action,
                        "account_equity_usdt": float(equity),
                        "base_position_size_pct": float(multiplier),
                        "order_notional_usdt": notional,
                        "side_depth_usdt": side_depth,
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def write_paper_outputs(output_dir: Path, status: dict[str, Any], gates: pd.DataFrame) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    events_path = output_dir / f"paper_events_{day}.csv"
    gate_path = output_dir / f"paper_gate_decisions_{day}.csv"
    append_csv(events_path, pd.DataFrame([status]))
    append_csv(gate_path, gates)
    status_path = output_dir / "paper_status.json"
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    report_path = output_dir / "paper_report.md"
    lines = [
        "# Paper-Only Status",
        "",
        f"- status: `{status['status']}`",
        f"- reason: `{status['reason']}`",
        f"- checked_at_utc: `{status['checked_at_utc']}`",
        f"- authorized_gate_count: `{status['authorized_gate_count']}`",
        f"- live_trading_enabled: `{status['live_trading_enabled']}`",
        "",
        "No private endpoints, no API keys and no real orders are used by this harness.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "events": str(events_path),
        "gate_decisions": str(gate_path),
        "status": str(status_path),
        "report": str(report_path),
    }


def run_paper_once(config: dict[str, Any]) -> dict[str, Any]:
    assert_paper_only_environment(config)
    paper = config["paper"]
    infra_config = load_yaml(paper["infrastructure_config"])
    policy_config = load_yaml(paper["policy_config"])
    collector = collector_config_from_yaml(infra_config)
    max_age = float(paper.get("max_collector_age_seconds", 20))
    health_ok, health_message = collector_healthcheck(collector, max_age_seconds=max_age)
    now = pd.Timestamp.now(tz="UTC")
    status = {
        "checked_at_utc": now.isoformat(),
        "status": "paper_ready",
        "reason": "ready",
        "collector_health_ok": bool(health_ok),
        "collector_health_message": health_message,
        "live_trading_enabled": False,
        "blackout_active": False,
        "blackout_reason": "",
        "quality_score": "unknown",
        "authorized_gate_count": 0,
        "blocked_gate_count": 0,
        "latest_snapshot_time_utc": "",
    }
    gates = pd.DataFrame()
    if not health_ok:
        status["status"] = "paper_kill_switch"
        status["reason"] = "stale_or_unhealthy_collector"
    else:
        snapshot, recent = load_latest_ws_snapshot(infra_config)
        status["latest_snapshot_time_utc"] = pd.Timestamp(snapshot["snapshot_time_utc"]).isoformat()
        quality = compute_quality_metrics(recent.tail(min(len(recent), 3600)), infra_config)
        status["quality_score"] = quality["quality_score"]
        blackout_path = project_path(paper.get("blackout_windows_path", ""))
        blackout_active, blackout_reason = is_event_blackout_active(blackout_path, now)
        status["blackout_active"] = bool(blackout_active)
        status["blackout_reason"] = blackout_reason
        locked = load_locked_017_candidates(policy_config)
        equities = [float(value) for value in paper.get("account_equity_usdt_grid", policy_config["microstructure_gate"]["account_equity_usdt_grid"])]
        gates = evaluate_current_entry_gates(snapshot, policy_config, equities, unique_base_multipliers(locked))
        authorized = int(gates["authorized"].astype(bool).sum()) if not gates.empty else 0
        status["authorized_gate_count"] = authorized
        status["blocked_gate_count"] = int(len(gates) - authorized)
        if quality["quality_score"] == "bad":
            status["status"] = "paper_kill_switch"
            status["reason"] = "bad_microstructure_quality"
        elif blackout_active:
            status["status"] = "paper_kill_switch"
            status["reason"] = "fundamental_blackout"
        elif authorized == 0:
            status["status"] = "entries_blocked"
            status["reason"] = "microstructure_gate_blocked_all_entries"
    output_paths = write_paper_outputs(project_path(paper.get("output_dir", "reports/paper_trading")), status, gates)
    return {"status": status, "output_paths": output_paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked 017 paper-only harness.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--healthcheck-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paper_config(args.config)
    assert_paper_only_environment(config)
    if args.healthcheck_only:
        infra_config = load_yaml(config["paper"]["infrastructure_config"])
        collector = collector_config_from_yaml(infra_config)
        ok, message = collector_healthcheck(collector, max_age_seconds=float(config["paper"].get("max_collector_age_seconds", 20)))
        print(json.dumps({"ok": ok, "message": message}, indent=2))
        raise SystemExit(0 if ok else 1)
    if not args.run_once:
        raise SystemExit("Choose --run-once or --healthcheck-only")
    payload = run_paper_once(config)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
