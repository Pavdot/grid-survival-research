from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.infra.binance_microstructure_collector import collector_config_from_yaml, write_health
from src.infra.ops_monitor import (
    build_rclone_commands,
    evaluate_ops_status,
    maybe_send_ops_alert,
    run_backup,
    send_telegram_alert,
)


def infra_config(tmp: Path) -> dict:
    return {
        "collector": {
            "name": "test",
            "symbol": "BTCUSDT",
            "websocket_url": "wss://data-stream.binance.vision/ws/btcusdt@depth@1000ms",
            "rest_base_urls": ["https://api.binance.com"],
            "rest_depth_endpoint": "/api/v3/depth",
            "rest_depth_limit": 1000,
            "timeout_seconds": 10,
            "snapshot_interval_seconds": 1,
            "flush_every_rows": 10,
            "max_buffered_events": 100,
            "depth_bands_bps": [1, 2, 5, 10],
            "top_levels_to_store": 100,
            "output_dir": str(tmp / "ws_depth"),
            "health_path": str(tmp / "ws_depth" / "health.json"),
            "reconnect_backoff_seconds": 1,
            "max_snapshot_age_seconds_for_health": 15,
        },
        "ops": {
            "disk_path": str(tmp),
            "disk_warn_fraction": 0.99,
            "max_parquet_age_seconds": 20,
            "backup": {
                "rclone_remote_env": "RCLONE_REMOTE",
                "manifest_dir": str(tmp / "manifests"),
                "include_paths": [str(tmp / "ws_depth")],
            },
        },
    }


def write_snapshot(path: Path, timestamp: pd.Timestamp) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "snapshot_time_utc": timestamp,
                "last_update_id": 1,
                "spread_bps": 0.1,
                "bid_depth_5bps_usdt": 1_000_000.0,
                "ask_depth_5bps_usdt": 1_000_000.0,
                "source_latency_ms": 10.0,
            }
        ]
    ).to_parquet(path, index=False)


class OpsMonitorTests(unittest.TestCase):
    def test_stale_snapshot_marks_ops_bad(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = infra_config(tmp)
            collector = collector_config_from_yaml(config)
            writeHealthTime = pd.Timestamp.now(tz="UTC")
            write_health(collector, "running", writeHealthTime, 1, buffered_rows=0)
            write_snapshot(tmp / "ws_depth" / "btcusdt_depth_2026-01-01.parquet", pd.Timestamp("2026-01-01T00:00:00Z"))
            status = evaluate_ops_status(config, now=pd.Timestamp("2026-01-01T00:01:00Z"))
            self.assertEqual(status["overall_status"], "bad")
            messages = [check["message"] for check in status["checks"]]
            self.assertTrue(any("stale" in message or "age" in message for message in messages))

    def test_telegram_dry_run_builds_payload(self) -> None:
        result = send_telegram_alert("hello", token="token", chat_id="chat", dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["payload"]["chat_id"], "chat")
        self.assertEqual(result["payload"]["text"], "hello")

    def test_rclone_command_includes_dry_run(self) -> None:
        commands = build_rclone_commands([Path("data/microstructure/ws_depth")], "remote:bucket", dry_run=True)
        self.assertEqual(commands[0][0], "rclone")
        self.assertIn("--dry-run", commands[0])

    def test_backup_manifest_written_without_executing_rclone(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = infra_config(tmp)
            (tmp / "ws_depth").mkdir(parents=True)
            (tmp / "ws_depth" / "sample.txt").write_text("x", encoding="utf-8")
            with patch.dict(os.environ, {"RCLONE_REMOTE": "remote:bucket"}):
                manifest, path = run_backup(config, dry_run=True, execute=False)
            self.assertTrue(path.exists())
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(loaded["file_count"], 1)

    def test_ops_alert_skips_cleanly_without_telegram_env(self) -> None:
        config = infra_config(Path("tmp"))
        config["ops"]["telegram"] = {"enabled": True, "token_env": "MISSING_TOKEN", "chat_id_env": "MISSING_CHAT"}
        status = {"overall_status": "bad", "checked_at_utc": "2026-01-01T00:00:00Z", "checks": []}
        with patch.dict(os.environ, {}, clear=True):
            result = maybe_send_ops_alert(config, status)
        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "telegram_env_missing")


if __name__ == "__main__":
    unittest.main()
