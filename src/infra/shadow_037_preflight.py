from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.infra.binance_kline_collector import kline_config_from_yaml, validate_kline_config
from src.infra.binance_market_connectivity import run_connectivity_checks, summarize_statuses
from src.infra.binance_microstructure_collector import collector_config_from_yaml, validate_collector_config
from src.paper.shadow_live_037 import assert_shadow_safe, load_shadow_live_config, validate_fundamental_schedule
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/shadow_live_037.yaml"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")


def _capture(name: str, callback: Callable[[], dict[str, Any] | None]) -> PreflightCheck:
    try:
        details = callback() or {}
        return PreflightCheck(name=name, status="ok", message="ok", details=details)
    except Exception as exc:  # noqa: BLE001 - preflight must report all blockers.
        return PreflightCheck(name=name, status="bad", message=str(exc), details={})


def _check_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    source_dir = project_path(config["shadow_live"]["source_037_dir"])
    if not source_dir.is_dir():
        raise FileNotFoundError(f"037 source directory missing: {source_dir}")
    required = ["manifest.json", "scenario_summary.csv", "holdout_summary.csv"]
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"037 source artifacts missing: {missing}")
    pointers = [str(path) for path in source_dir.rglob("*") if path.is_file() and is_lfs_pointer(path)]
    if pointers:
        raise RuntimeError(f"Git LFS objects are not materialized; run git lfs pull: {pointers[:3]}")
    return {"source_dir": str(source_dir), "required_artifacts": required}


def _check_configs(config: dict[str, Any]) -> dict[str, Any]:
    signal = kline_config_from_yaml(config, section="kline_collector")
    execution = kline_config_from_yaml(config, section="execution_kline_collector")
    validate_kline_config(signal)
    validate_kline_config(execution)
    infra_path = config["shadow_live"]["infrastructure_config"]
    infra = load_yaml(infra_path)
    depth = collector_config_from_yaml(infra)
    validate_collector_config(depth)
    if signal.market != "spot" or execution.market != "futures_usdm":
        raise ValueError("037 requires Spot signal klines and USD-M Futures execution klines")
    if "/fapi/" not in depth.rest_depth_endpoint or "fstream.binance.com" not in depth.websocket_url:
        raise ValueError("037 execution depth collector must use Binance USD-M Futures public endpoints")
    return {
        "signal_market": signal.market,
        "execution_market": execution.market,
        "depth_endpoint": depth.rest_depth_endpoint,
        "depth_stream": depth.websocket_url,
    }


def _check_writable_paths(config: dict[str, Any]) -> dict[str, Any]:
    paths = [
        project_path(config["shadow_live"]["output_dir"]),
        project_path(config["kline_collector"]["output_path"]).parent,
        project_path(config["execution_kline_collector"]["output_path"]).parent,
    ]
    infra = load_yaml(config["shadow_live"]["infrastructure_config"])
    paths.append(collector_config_from_yaml(infra).output_dir)
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="preflight_", dir=path, delete=True):
            pass
    return {"writable_paths": [str(path) for path in paths]}


def run_preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    online: bool = False,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    config = load_shadow_live_config(config_path)
    checks = [
        _capture("paper_only_safety", lambda: (assert_shadow_safe(config), {"paper_only": True})[1]),
        _capture("locked_037_artifacts", lambda: _check_artifacts(config)),
        _capture("public_market_configs", lambda: _check_configs(config)),
        _capture("fundamental_schedule", lambda: validate_fundamental_schedule(config, now=now)),
        _capture("persistent_paths", lambda: _check_writable_paths(config)),
    ]
    if online:
        connectivity, output_dir = run_connectivity_checks("config/binance_market_connectivity.yaml")
        summary = summarize_statuses(connectivity)
        checks.append(
            PreflightCheck(
                name="public_api_connectivity",
                status="ok" if summary["failed"] == 0 else "bad",
                message="ok" if summary["failed"] == 0 else f"{summary['failed']} public API checks failed",
                details={"output_dir": str(output_dir), **summary},
            )
        )
    bad = [check for check in checks if check.status == "bad"]
    payload = {
        "ready": not bad,
        "mode": "paper_only",
        "checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": str(config_path),
        "checks": [asdict(check) for check in checks],
        "blocker_count": len(bad),
    }
    output = project_path("reports/infra/runtime/shadow_037_preflight.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    payload["report_path"] = str(output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the complete paper-only 037 infrastructure.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--online", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_preflight(args.config, online=args.online)
    print(json.dumps(payload, indent=2, default=str))
    raise SystemExit(0 if payload["ready"] else 1)


if __name__ == "__main__":
    main()
