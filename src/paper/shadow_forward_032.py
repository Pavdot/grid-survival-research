from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.infra.binance_microstructure_collector import collector_config_from_yaml, healthcheck as collector_healthcheck
from src.infra.microstructure_quality_report import compute_quality_metrics
from src.paper.paper_trading_harness import (
    assert_paper_only_environment,
    evaluate_current_entry_gates,
    load_latest_ws_snapshot,
    unique_base_multipliers,
)
from src.research.microstructure_execution_filter_017 import load_locked_017_candidates
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/shadow_forward_032.yaml"


def load_shadow_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    if "shadow" not in config:
        raise ValueError("shadow forward config requires a shadow section")
    return config


def _as_paper_config(config: dict[str, Any]) -> dict[str, Any]:
    shadow = config["shadow"]
    return {
        "paper": {
            "live_trading_env_var": shadow.get("live_trading_env_var", "LIVE_TRADING_ENABLED"),
            "private_env_vars": shadow.get("private_env_vars", []),
        }
    }


def choose_shadow_policy(config: dict[str, Any]) -> str:
    shadow = config["shadow"]
    default = str(shadow.get("default_policy", "maker_entry_add_taker_exit"))
    path = project_path(shadow.get("policy_ablation_dir", "")) / "policy_ablation_summary.csv"
    if not path.exists():
        return default
    frame = pd.read_csv(path)
    if frame.empty or "policy_improves_net" not in frame.columns:
        return default
    winners = frame[frame["policy_improves_net"].astype(bool)]
    if winners.empty:
        return default
    return str(winners.iloc[0]["policy"])


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def completed_shadow_days(output_dir: Path) -> int:
    path = output_dir / "shadow_orders.csv"
    if not path.exists():
        return 0
    frame = pd.read_csv(path)
    if frame.empty or "checked_at_utc" not in frame.columns:
        return 0
    return int(pd.to_datetime(frame["checked_at_utc"], utc=True).dt.date.nunique())


def cost_vs_backtest(config: dict[str, Any], gates: pd.DataFrame) -> pd.DataFrame:
    shadow = config["shadow"]
    conditional_path = project_path(shadow.get("conditional_execution_dir", "")) / "conditional_execution_costs.csv"
    surface_path = project_path(shadow.get("execution_surface_dir", "")) / "execution_surface_summary.csv"
    measured_cost = float(gates["theoretical_slippage_bps"].replace([float("inf"), -float("inf")], pd.NA).dropna().median()) if not gates.empty else float("nan")
    rows: list[dict[str, Any]] = [
        {
            "metric": "shadow_median_theoretical_slippage_bps",
            "value": measured_cost,
            "source": "shadow_current_snapshot",
        }
    ]
    if conditional_path.exists():
        conditional = pd.read_csv(conditional_path)
        if not conditional.empty and "p90_slippage_bps" in conditional.columns:
            rows.append(
                {
                    "metric": "conditional_p90_slippage_bps",
                    "value": float(conditional["p90_slippage_bps"].dropna().median()),
                    "source": str(conditional_path),
                }
            )
    if surface_path.exists():
        surface = pd.read_csv(surface_path)
        zone = surface[(surface["fee_rate"].le(0.0001)) & (surface["slippage_bps"].le(0.5)) & (surface["fill_rate"].ge(0.99))]
        if not zone.empty:
            rows.append(
                {
                    "metric": "surface_protected_zone_worst_monthly",
                    "value": float(zone["monthly_return_median"].min()),
                    "source": str(surface_path),
                }
            )
    return pd.DataFrame(rows)


def shadow_verdict(status: dict[str, Any], output_dir: Path, min_days: int) -> tuple[str, str]:
    if status["status"] != "shadow_ready":
        return "collecting - blocked this check", str(status["reason"])
    days = completed_shadow_days(output_dir)
    if days < min_days:
        return "collecting", f"{days} completed shadow days below required {min_days}"
    return "shadow forward ready for review", "minimum shadow duration reached"


def write_outputs(output_dir: Path, status: dict[str, Any], gates: pd.DataFrame, fills: pd.DataFrame, cost_compare: pd.DataFrame) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    append_csv(output_dir / "shadow_orders.csv", gates.assign(**status))
    append_csv(output_dir / "shadow_fills.csv", fills.assign(**status))
    daily = pd.DataFrame(
        [
            {
                "day_utc": pd.Timestamp(status["checked_at_utc"]).strftime("%Y-%m-%d"),
                "paper_pnl": 0.0,
                "authorized_gate_count": status["authorized_gate_count"],
                "blocked_gate_count": status["blocked_gate_count"],
                "status": status["status"],
            }
        ]
    )
    append_csv(output_dir / "shadow_daily_pnl.csv", daily)
    cost_compare.to_csv(output_dir / "shadow_cost_vs_backtest.csv", index=False)
    report = output_dir / "shadow_forward_report.md"
    lines = [
        "# Iteration 032 - 30-Day Shadow Forward",
        "",
        f"- verdict: `{status['verdict']}`",
        f"- reason: {status['verdict_reason']}",
        f"- status: `{status['status']}`",
        f"- selected_policy: `{status['selected_policy']}`",
        f"- checked_at_utc: `{status['checked_at_utc']}`",
        f"- authorized_gate_count: `{status['authorized_gate_count']}`",
        f"- blocked_gate_count: `{status['blocked_gate_count']}`",
        "",
        "Paper-only: no private endpoints, no API keys, and no real orders.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    status_path = output_dir / "shadow_status.json"
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return {
        "orders": str(output_dir / "shadow_orders.csv"),
        "fills": str(output_dir / "shadow_fills.csv"),
        "daily_pnl": str(output_dir / "shadow_daily_pnl.csv"),
        "cost_vs_backtest": str(output_dir / "shadow_cost_vs_backtest.csv"),
        "report": str(report),
        "status": str(status_path),
    }


def run_shadow_once(config: dict[str, Any], duration_days: int | None = None) -> dict[str, Any]:
    assert_paper_only_environment(_as_paper_config(config))
    shadow = config["shadow"]
    output_dir = project_path(shadow.get("output_dir", "reports/research_iterations/iteration_032_shadow_forward"))
    infra_config = load_yaml(shadow["infrastructure_config"])
    policy_config = load_yaml(shadow["policy_config"])
    collector = collector_config_from_yaml(infra_config)
    now = pd.Timestamp.now(tz="UTC")
    status: dict[str, Any] = {
        "checked_at_utc": now.isoformat(),
        "status": "shadow_ready",
        "reason": "ready",
        "selected_policy": choose_shadow_policy(config),
        "live_trading_enabled": os.getenv(str(shadow.get("live_trading_env_var", "LIVE_TRADING_ENABLED")), "").lower() in {"1", "true", "yes", "on"},
        "quality_score": "unknown",
        "collector_health_ok": False,
        "collector_health_message": "",
        "authorized_gate_count": 0,
        "blocked_gate_count": 0,
    }
    health_ok, health_message = collector_healthcheck(collector, max_age_seconds=float(shadow.get("max_collector_age_seconds", 20)))
    status["collector_health_ok"] = bool(health_ok)
    status["collector_health_message"] = health_message
    gates = pd.DataFrame()
    fills = pd.DataFrame()
    if not health_ok:
        status["status"] = "shadow_kill_switch"
        status["reason"] = "stale_or_unhealthy_collector"
    else:
        snapshot, recent = load_latest_ws_snapshot(infra_config)
        quality = compute_quality_metrics(recent.tail(min(len(recent), 3600)), infra_config)
        status["quality_score"] = quality["quality_score"]
        if quality["quality_score"] == "bad":
            status["status"] = "shadow_kill_switch"
            status["reason"] = "bad_microstructure_quality"
        locked = load_locked_017_candidates(policy_config)
        gates = evaluate_current_entry_gates(
            snapshot,
            policy_config,
            [float(shadow.get("primary_equity_usdt", 10000))],
            unique_base_multipliers(locked),
        )
        gates.insert(0, "selected_policy", status["selected_policy"])
        gates.insert(0, "checked_at_utc", status["checked_at_utc"])
        status["authorized_gate_count"] = int(gates["authorized"].astype(bool).sum()) if not gates.empty else 0
        status["blocked_gate_count"] = int(len(gates) - status["authorized_gate_count"])
        fills = gates[gates["authorized"].astype(bool)].copy()
        fills["simulated_fill"] = True
        fills["real_order_sent"] = False
    min_days = int(duration_days or shadow.get("min_completed_days_for_verdict", 30))
    verdict, reason = shadow_verdict(status, output_dir, min_days)
    status["verdict"] = verdict
    status["verdict_reason"] = reason
    cost_compare = cost_vs_backtest(config, gates)
    paths = write_outputs(output_dir, status, gates, fills, cost_compare)
    return {"status": status, "output_paths": paths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Iteration 032 paper-only shadow forward harness.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--duration-days", type=int)
    parser.add_argument("--run-once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_shadow_config(args.config)
    if not args.run_once:
        raise SystemExit("Use --run-once; schedule it repeatedly for the 30-day shadow forward.")
    payload = run_shadow_once(config, duration_days=args.duration_days)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
