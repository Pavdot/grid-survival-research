from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.paper.paper_trading_harness import (
    assert_paper_only_environment,
    evaluate_current_entry_gates,
    is_event_blackout_active,
    run_paper_once,
)
from src.research.microstructure_execution_filter_017 import normalize_depth_snapshot


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
        "quality": {
            "expected_interval_seconds": 1,
            "gap_warn_seconds": 5,
            "stale_bad_seconds": 60,
            "min_coverage_ratio_healthy": 0.95,
            "min_coverage_ratio_degraded": 0.80,
            "max_invalid_fraction_healthy": 0.01,
            "max_invalid_fraction_degraded": 0.05,
        },
    }


def policy_config() -> dict:
    return {
        "source_iteration": {
            "output_dir": "reports/research_iterations/iteration_017_fundamental_trend_escape_best_policy_expanded",
            "variant": "fundamental_trend_escape_entry_only",
            "reference_monthly_return": 0.13433973204507899,
        },
        "collection": {
            "base_urls": ["https://api.binance.com"],
            "depth_endpoint": "/api/v3/depth",
            "book_ticker_endpoint": "/api/v3/ticker/bookTicker",
            "depth_limit": 500,
            "timeout_seconds": 10,
            "snapshot_path": "data/microstructure/btcusdt_order_book_snapshots.parquet",
        },
        "microstructure_gate": {
            "account_equity_usdt_grid": [1000],
            "depth_band_bps": 5,
            "depth_bands_bps": [1, 2, 5, 10],
            "max_spread_bps": 20,
            "max_order_book_share": 0.05,
            "min_side_depth_usdt_floor": 100,
            "max_abs_depth_imbalance": 0.95,
            "max_snapshot_age_ms": 2000,
            "max_theoretical_slippage_bps": 50,
        },
        "order_policy": {
            "policies": ["gate_only_taker"],
            "max_clip_book_share": 0.01,
            "max_total_book_share": 0.05,
            "maker_ttl_seconds": 10,
            "max_reprices": 1,
            "synthetic_mapping_seed": 7,
        },
    }


def paper_config(tmp: Path, infra_path: Path, policy_path: Path) -> dict:
    return {
        "paper": {
            "output_dir": str(tmp / "paper"),
            "infrastructure_config": str(infra_path),
            "policy_config": str(policy_path),
            "max_collector_age_seconds": 20,
            "account_equity_usdt_grid": [1000],
            "blackout_windows_path": str(tmp / "blackout_windows.csv"),
            "private_env_vars": ["BINANCE_API_KEY", "BINANCE_SECRET_KEY"],
            "live_trading_env_var": "LIVE_TRADING_ENABLED",
        }
    }


def snapshot_row(ts: str) -> pd.Series:
    return pd.Series(
        normalize_depth_snapshot(
            "BTCUSDT",
            {
                "lastUpdateId": int(pd.Timestamp(ts).timestamp()),
                "bids": [["100.00", "1000.0"], ["99.99", "1000.0"]],
                "asks": [["100.01", "1000.0"], ["100.02", "1000.0"]],
            },
            {"symbol": "BTCUSDT", "bidPrice": "100.00", "bidQty": "1000.0", "askPrice": "100.01", "askQty": "1000.0"},
            [1, 2, 5, 10],
            source_latency_ms=20.0,
            depth_source_url="depth",
            book_ticker_source_url="ticker",
            snapshot_time=pd.Timestamp(ts),
        )
    )


class PaperTradingHarnessTests(unittest.TestCase):
    def test_live_trading_env_is_refused(self) -> None:
        config = {"paper": {"live_trading_env_var": "LIVE_TRADING_ENABLED", "private_env_vars": []}}
        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "true"}, clear=True):
            with self.assertRaises(RuntimeError):
                assert_paper_only_environment(config)

    def test_private_key_env_is_refused(self) -> None:
        config = {"paper": {"live_trading_env_var": "LIVE_TRADING_ENABLED", "private_env_vars": ["BINANCE_API_KEY"]}}
        with patch.dict(os.environ, {"BINANCE_API_KEY": "secret"}, clear=True):
            with self.assertRaises(RuntimeError):
                assert_paper_only_environment(config)

    def test_blackout_window_detects_active_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "blackout.csv"
            pd.DataFrame(
                [
                    {
                        "start_time_utc": "2026-01-01T00:00:00Z",
                        "end_time_utc": "2026-01-01T01:00:00Z",
                        "category": "macro_cpi",
                    }
                ]
            ).to_csv(path, index=False)
            active, reason = is_event_blackout_active(path, pd.Timestamp("2026-01-01T00:30:00Z"))
        self.assertTrue(active)
        self.assertEqual(reason, "macro_cpi")

    def test_current_entry_gates_authorize_clean_snapshot(self) -> None:
        gates = evaluate_current_entry_gates(snapshot_row("2026-01-01T00:00:00Z"), policy_config(), [1000.0], [1.0])
        self.assertEqual(len(gates), 2)
        self.assertTrue(gates["authorized"].astype(bool).all())

    def test_stale_collector_blocks_paper_cycle_before_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            infra_path = tmp / "infra.yaml"
            policy_path = tmp / "policy.yaml"
            infra_path.write_text(yaml.safe_dump(infra_config(tmp)), encoding="utf-8")
            policy_path.write_text(yaml.safe_dump(policy_config()), encoding="utf-8")
            config = paper_config(tmp, infra_path, policy_path)
            with patch.dict(os.environ, {}, clear=True):
                payload = run_paper_once(config)
        self.assertEqual(payload["status"]["status"], "paper_kill_switch")
        self.assertEqual(payload["status"]["reason"], "stale_or_unhealthy_collector")


if __name__ == "__main__":
    unittest.main()
