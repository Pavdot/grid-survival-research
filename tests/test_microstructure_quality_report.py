from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.infra.microstructure_quality_report import (
    compute_quality_metrics,
    generate_quality_report,
    hourly_microstructure,
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
        "quality": {
            "expected_interval_seconds": 1,
            "gap_warn_seconds": 5,
            "stale_bad_seconds": 60,
            "min_coverage_ratio_healthy": 0.95,
            "min_coverage_ratio_degraded": 0.80,
            "max_invalid_fraction_healthy": 0.01,
            "max_invalid_fraction_degraded": 0.05,
            "output_dir": str(tmp / "reports"),
        },
    }


def quality_frame(start: str = "2026-01-01T00:00:00Z", rows: int = 20, crossed: bool = False) -> pd.DataFrame:
    times = pd.date_range(start, periods=rows, freq="1s", tz="UTC")
    best_bid = 100.0
    best_ask = 99.99 if crossed else 100.01
    return pd.DataFrame(
        {
            "snapshot_time_utc": times,
            "symbol": "BTCUSDT",
            "last_update_id": range(rows),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": 0.1,
            "bid_depth_5bps_usdt": 1_000_000.0,
            "ask_depth_5bps_usdt": 1_100_000.0,
            "depth_imbalance_5bps": -0.05,
            "source_latency_ms": 10.0,
        }
    )


class MicrostructureQualityReportTests(unittest.TestCase):
    def test_healthy_quality_score_for_regular_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = compute_quality_metrics(quality_frame(), infra_config(Path(tmpdir)))
        self.assertEqual(metrics["quality_score"], "healthy")
        self.assertEqual(metrics["gap_count"], 0)
        self.assertGreater(metrics["coverage_ratio"], 0.95)

    def test_crossed_book_marks_quality_bad(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = compute_quality_metrics(quality_frame(crossed=True), infra_config(Path(tmpdir)))
        self.assertEqual(metrics["quality_score"], "bad")
        self.assertGreater(metrics["crossed_book_count"], 0)

    def test_hourly_microstructure_has_24_rows(self) -> None:
        hourly = hourly_microstructure(quality_frame(rows=5))
        self.assertEqual(len(hourly), 24)
        self.assertIn("spread_bps_p90", hourly.columns)

    def test_generate_quality_report_writes_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = infra_config(tmp)
            out_dir = tmp / "ws_depth"
            out_dir.mkdir(parents=True)
            quality_frame(rows=5).to_parquet(out_dir / "btcusdt_depth_2026-01-01.parquet", index=False)
            payload = generate_quality_report(config, output_dir=tmp / "dashboard")
            self.assertTrue(Path(payload["html_dashboard"]).exists())
            self.assertTrue(Path(payload["heatmap"]).exists())


if __name__ == "__main__":
    unittest.main()
