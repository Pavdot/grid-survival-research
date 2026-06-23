from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.infra.binance_microstructure_collector import (
    CollectorConfig,
    LocalOrderBook,
    append_daily_rows,
    daily_output_path,
    healthcheck,
    normalize_book_row,
    validate_collector_config,
    write_health,
)


def collector_config(tmp: Path | None = None) -> CollectorConfig:
    root = tmp or Path("data/microstructure/ws_depth")
    return CollectorConfig(
        name="test_collector",
        symbol="BTCUSDT",
        websocket_url="wss://data-stream.binance.vision/ws/btcusdt@depth@1000ms",
        rest_base_urls=("https://api.binance.com",),
        rest_depth_endpoint="/api/v3/depth",
        rest_depth_limit=1000,
        timeout_seconds=10.0,
        snapshot_interval_seconds=1.0,
        flush_every_rows=10,
        max_buffered_events=100,
        depth_bands_bps=(1.0, 2.0, 5.0, 10.0),
        top_levels_to_store=100,
        output_dir=root,
        health_path=root / "health.json",
        reconnect_backoff_seconds=1.0,
        max_snapshot_age_seconds_for_health=15.0,
    )


def snapshot_payload() -> dict:
    return {
        "lastUpdateId": 100,
        "bids": [["100.00", "1.0"], ["99.99", "2.0"], ["99.95", "3.0"]],
        "asks": [["100.01", "1.5"], ["100.02", "2.0"], ["100.05", "3.0"]],
    }


class BinanceMicrostructureCollectorTests(unittest.TestCase):
    def test_local_order_book_applies_updates_and_removes_zero_qty(self) -> None:
        book = LocalOrderBook.from_snapshot(snapshot_payload())
        applied = book.apply_update(
            {
                "U": 101,
                "u": 102,
                "b": [["99.99", "0"], ["99.98", "4.0"]],
                "a": [["100.02", "1.0"]],
            }
        )
        self.assertTrue(applied)
        self.assertEqual(book.last_update_id, 102)
        self.assertNotIn(99.99, book.bids)
        self.assertEqual(book.bids[99.98], 4.0)
        self.assertEqual(book.asks[100.02], 1.0)

    def test_sequence_gap_is_rejected(self) -> None:
        book = LocalOrderBook.from_snapshot(snapshot_payload())
        with self.assertRaises(ValueError):
            book.apply_update({"U": 105, "u": 106, "b": [], "a": []})

    def test_old_update_is_ignored(self) -> None:
        book = LocalOrderBook.from_snapshot(snapshot_payload())
        self.assertFalse(book.apply_update({"U": 90, "u": 95, "b": [["100", "9"]], "a": []}))
        self.assertEqual(book.bids[100.0], 1.0)

    def test_normalize_book_row_outputs_depth_metrics(self) -> None:
        config = collector_config()
        book = LocalOrderBook.from_snapshot(snapshot_payload())
        row = normalize_book_row(
            book,
            config,
            source_latency_ms=12.0,
            stream_event_time_ms=1767225600000,
            event_receive_time=pd.Timestamp("2026-01-01T00:00:00Z"),
        )
        self.assertAlmostEqual(row["spread_bps"], (100.01 - 100.0) / 100.005 * 10000.0)
        self.assertGreater(row["bid_depth_5bps_usdt"], 0)
        self.assertGreater(row["ask_depth_5bps_usdt"], 0)
        self.assertIn("bids_json", row)
        self.assertIn("asks_json", row)

    def test_daily_append_writes_partitioned_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = collector_config(Path(tmpdir))
            book = LocalOrderBook.from_snapshot(snapshot_payload())
            rows = [
                normalize_book_row(
                    book,
                    config,
                    1.0,
                    event_receive_time=pd.Timestamp("2026-01-01T00:00:00Z"),
                ),
                normalize_book_row(
                    book,
                    config,
                    1.0,
                    event_receive_time=pd.Timestamp("2026-01-02T00:00:00Z"),
                ),
            ]
            paths = append_daily_rows(rows, config)
            self.assertEqual(len(paths), 2)
            self.assertTrue((Path(tmpdir) / "btcusdt_depth_2026-01-01.parquet").exists())
            self.assertTrue((Path(tmpdir) / "btcusdt_depth_2026-01-02.parquet").exists())

    def test_daily_output_path_uses_utc_day(self) -> None:
        config = collector_config(Path("tmp"))
        path = daily_output_path(config, pd.Timestamp("2026-01-01T23:30:00Z"))
        self.assertEqual(path.name, "btcusdt_depth_2026-01-01.parquet")

    def test_healthcheck_detects_running_and_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = collector_config(Path(tmpdir))
            write_health(config, "running", pd.Timestamp.now(tz="UTC"), 123, buffered_rows=0)
            ok, message = healthcheck(config, max_age_seconds=15.0)
            self.assertTrue(ok, message)
            write_health(config, "running", pd.Timestamp("2020-01-01T00:00:00Z"), 123, buffered_rows=0)
            ok, message = healthcheck(config, max_age_seconds=15.0)
            self.assertFalse(ok)
            self.assertIn("stale", message)

    def test_config_rejects_private_endpoint(self) -> None:
        config = collector_config()
        bad = CollectorConfig(**{**config.__dict__, "rest_depth_endpoint": "/api/v3/order"})
        with self.assertRaises(ValueError):
            validate_collector_config(bad)


if __name__ == "__main__":
    unittest.main()
