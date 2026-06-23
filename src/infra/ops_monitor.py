from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.infra.binance_microstructure_collector import (
    CollectorConfig,
    collector_config_from_yaml,
    healthcheck as collector_healthcheck,
    validate_collector_config,
)
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/infrastructure_microstructure.yaml"
TRUTHY = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class OpsCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]


def load_infra_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    collector = collector_config_from_yaml(config)
    validate_collector_config(collector)
    return config


def collector_from_config(config: dict[str, Any]) -> CollectorConfig:
    collector = collector_config_from_yaml(config)
    validate_collector_config(collector)
    return collector


def latest_parquet(output_dir: Path, symbol: str) -> Path | None:
    paths = sorted(output_dir.glob(f"{symbol.lower()}_depth_*.parquet"))
    if not paths:
        return None
    return max(paths, key=lambda path: (path.stat().st_mtime, path.name))


def read_last_snapshot_time(path: Path) -> pd.Timestamp:
    frame = pd.read_parquet(path, columns=["snapshot_time_utc"])
    if frame.empty:
        raise ValueError(f"parquet file is empty: {path}")
    return pd.to_datetime(frame["snapshot_time_utc"], utc=True).max()


def evaluate_ops_status(
    config: dict[str, Any],
    now: pd.Timestamp | None = None,
    disk_usage: Callable[[Path], shutil._ntuple_diskusage] | None = None,
) -> dict[str, Any]:
    collector = collector_from_config(config)
    ops = config.get("ops", {})
    now = now or pd.Timestamp.now(tz="UTC")
    disk_usage = disk_usage or shutil.disk_usage
    checks: list[OpsCheck] = []

    max_age = float(ops.get("max_parquet_age_seconds", collector.max_snapshot_age_seconds_for_health))
    health_ok, health_message = collector_healthcheck(collector, max_age_seconds=collector.max_snapshot_age_seconds_for_health)
    checks.append(
        OpsCheck(
            name="collector_health",
            status="ok" if health_ok else "bad",
            message=health_message,
            details={"health_path": str(collector.health_path)},
        )
    )

    latest = latest_parquet(collector.output_dir, collector.symbol)
    if latest is None:
        checks.append(
            OpsCheck(
                name="latest_parquet",
                status="bad",
                message=f"no daily parquet found in {collector.output_dir}",
                details={},
            )
        )
    else:
        try:
            last_snapshot = read_last_snapshot_time(latest)
            age_seconds = (now - last_snapshot) / pd.Timedelta(seconds=1)
            checks.append(
                OpsCheck(
                    name="parquet_freshness",
                    status="ok" if age_seconds <= max_age else "bad",
                    message=f"latest parquet snapshot age is {age_seconds:.1f}s",
                    details={
                        "path": str(latest),
                        "last_snapshot_time_utc": last_snapshot.isoformat(),
                        "age_seconds": float(age_seconds),
                        "max_age_seconds": max_age,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should not mask status.
            checks.append(
                OpsCheck(
                    name="parquet_freshness",
                    status="bad",
                    message=f"cannot read latest parquet: {exc}",
                    details={"path": str(latest)},
                )
            )

    disk_path = project_path(ops.get("disk_path", "."))
    usage = disk_usage(disk_path)
    used_fraction = usage.used / usage.total if usage.total else 1.0
    warn_fraction = float(ops.get("disk_warn_fraction", 0.80))
    checks.append(
        OpsCheck(
            name="disk_usage",
            status="ok" if used_fraction <= warn_fraction else "bad",
            message=f"disk usage is {used_fraction:.1%}",
            details={
                "path": str(disk_path),
                "used_fraction": float(used_fraction),
                "warn_fraction": warn_fraction,
                "free_bytes": int(usage.free),
            },
        )
    )

    bad_checks = [check for check in checks if check.status == "bad"]
    overall = "bad" if bad_checks else "ok"
    return {
        "overall_status": overall,
        "checked_at_utc": now.isoformat(),
        "checks": [asdict(check) for check in checks],
        "bad_check_count": len(bad_checks),
    }


def format_ops_message(status: dict[str, Any]) -> str:
    lines = [
        f"grid-survival-research ops: {status['overall_status']}",
        f"checked_at_utc: {status['checked_at_utc']}",
    ]
    for check in status["checks"]:
        prefix = "OK" if check["status"] == "ok" else "BAD"
        lines.append(f"{prefix} {check['name']}: {check['message']}")
    return "\n".join(lines)


def send_telegram_alert(
    text: str,
    token: str | None = None,
    chat_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required")
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if dry_run:
        return {"dry_run": True, "payload": payload}
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
        headers={"User-Agent": "grid-survival-research/ops"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def maybe_send_ops_alert(config: dict[str, Any], status: dict[str, Any], force: bool = False) -> dict[str, Any]:
    telegram = config.get("ops", {}).get("telegram", {})
    if not bool(telegram.get("enabled", True)):
        return {"sent": False, "reason": "telegram_disabled"}
    if status.get("overall_status") == "ok" and not force:
        return {"sent": False, "reason": "status_ok"}
    token = os.getenv(str(telegram.get("token_env", "TELEGRAM_BOT_TOKEN")))
    chat_id = os.getenv(str(telegram.get("chat_id_env", "TELEGRAM_CHAT_ID")))
    if not token or not chat_id:
        return {"sent": False, "reason": "telegram_env_missing"}
    result = send_telegram_alert(format_ops_message(status), token=token, chat_id=chat_id)
    return {"sent": True, "result": result}


def iter_backup_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(child for child in path.rglob("*") if child.is_file()))
    return sorted(files)


def metadata_checksum(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    return digest.hexdigest()


def build_rclone_commands(include_paths: list[Path], remote: str, dry_run: bool) -> list[list[str]]:
    if not remote:
        raise ValueError("rclone remote cannot be empty")
    commands: list[list[str]] = []
    for path in include_paths:
        target = f"{remote.rstrip('/')}/{path.name}"
        command = ["rclone", "copy", str(path), target, "--checksum", "--create-empty-src-dirs"]
        if dry_run:
            command.append("--dry-run")
        commands.append(command)
    return commands


def backup_manifest(config: dict[str, Any], dry_run: bool, status: str, message: str) -> dict[str, Any]:
    backup = config.get("ops", {}).get("backup", {})
    include_paths = [project_path(path) for path in backup.get("include_paths", [])]
    files = iter_backup_files(include_paths)
    return {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": status,
        "message": message,
        "dry_run": bool(dry_run),
        "remote_env": str(backup.get("rclone_remote_env", "RCLONE_REMOTE")),
        "include_paths": [str(path) for path in include_paths],
        "file_count": int(len(files)),
        "total_bytes": int(sum(path.stat().st_size for path in files)),
        "metadata_checksum_sha256": metadata_checksum(files),
    }


def write_backup_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> Path:
    manifest_dir = project_path(config.get("ops", {}).get("backup", {}).get("manifest_dir", "reports/infra/backups"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    path = manifest_dir / f"backup_manifest_{stamp}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def run_backup(
    config: dict[str, Any],
    dry_run: bool,
    execute: bool = True,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[dict[str, Any], Path]:
    backup = config.get("ops", {}).get("backup", {})
    remote = os.getenv(str(backup.get("rclone_remote_env", "RCLONE_REMOTE")), "")
    include_paths = [project_path(path) for path in backup.get("include_paths", [])]
    command_runner = command_runner or subprocess.run
    commands = build_rclone_commands(include_paths, remote, dry_run)
    results: list[dict[str, Any]] = []
    status = "ok"
    message = "backup completed"
    for command in commands:
        if not execute:
            results.append({"command": command, "returncode": 0, "skipped_execute": True})
            continue
        completed = command_runner(command, capture_output=True, text=True, check=False)
        results.append(
            {
                "command": command,
                "returncode": int(completed.returncode),
                "stdout_tail": (completed.stdout or "")[-2000:],
                "stderr_tail": (completed.stderr or "")[-2000:],
            }
        )
        if completed.returncode != 0:
            status = "failed"
            message = f"rclone failed with return code {completed.returncode}"
            break
    manifest = backup_manifest(config, dry_run=dry_run, status=status, message=message)
    manifest["commands"] = results
    path = write_backup_manifest(config, manifest)
    return manifest, path


def write_daily_report(config: dict[str, Any], status: dict[str, Any]) -> Path:
    reports_dir = project_path(config.get("ops", {}).get("reports_dir", "reports/infra"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    day = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    path = reports_dir / f"ops_daily_{day}.md"
    lines = [
        "# Ops Daily Status",
        "",
        f"- overall_status: `{status['overall_status']}`",
        f"- checked_at_utc: `{status['checked_at_utc']}`",
        "",
        "## Checks",
        "",
    ]
    for check in status["checks"]:
        lines.append(f"- `{check['status']}` {check['name']}: {check['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor collector health, alerts and backups.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--send-test-alert", action="store_true")
    parser.add_argument("--backup-dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--write-daily-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_infra_config(args.config)
    if args.send_test_alert:
        result = send_telegram_alert("grid-survival-research Telegram alert test")
        print(json.dumps(result, indent=2, default=str))
        return
    if args.backup_dry_run or args.backup:
        manifest, path = run_backup(config, dry_run=args.backup_dry_run)
        print(json.dumps({"manifest_path": str(path), **manifest}, indent=2, default=str))
        return
    status = evaluate_ops_status(config)
    if args.write_daily_report:
        path = write_daily_report(config, status)
        alert = maybe_send_ops_alert(config, status)
        print(json.dumps({"daily_report": str(path), "alert": alert, **status}, indent=2, default=str))
        return
    if args.healthcheck:
        alert = maybe_send_ops_alert(config, status)
        print(json.dumps({"alert": alert, **status}, indent=2, default=str))
        raise SystemExit(0 if status["overall_status"] == "ok" else 1)
    print(format_ops_message(status))


if __name__ == "__main__":
    main()
