from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.infra.binance_kline_collector import (
    KlineCollectorConfig,
    audit_kline_frame,
    normalize_klines,
    resample_closed_1h_from_5m,
    validate_kline_config,
    write_kline_frame,
    load_kline_frame,
    seed_klines,
    healthcheck,
    write_health,
)


def kline(open_time: str, open_: float = 100.0) -> list[object]:
    start = pd.Timestamp(open_time, tz="UTC")
    close = start + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)
    return [
        int(start.timestamp() * 1000),
        str(open_),
        str(open_ + 2),
        str(open_ - 2),
        str(open_ + 1),
        "10.0",
        int(close.timestamp() * 1000),
    ]


def config(tmp: Path) -> KlineCollectorConfig:
    return KlineCollectorConfig(
        name="test",
        symbol="BTCUSDT",
        interval="5m",
        rest_base_urls=("https://api.binance.com",),
        rest_klines_endpoint="/api/v3/klines",
        rest_limit=1000,
        timeout_seconds=10,
        poll_interval_seconds=60,
        lookback_days=1,
        output_path=tmp / "btcusdt_5m.parquet",
        resampled_1h_path=tmp / "btcusdt_1h.parquet",
        health_path=tmp / "health.json",
        audit_path=tmp / "audit.json",
    )


class BinanceKlineCollectorTests(unittest.TestCase):
    def test_normalize_klines_drops_unclosed_current_bar(self) -> None:
        frame = normalize_klines(
            [kline("2026-01-01T00:00:00Z"), kline("2026-01-01T00:05:00Z")],
            now=pd.Timestamp("2026-01-01T00:07:00Z"),
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.index[0], pd.Timestamp("2026-01-01T00:00:00Z"))

    def test_resample_1h_requires_all_twelve_closed_5m_bars(self) -> None:
        incomplete = normalize_klines(
            [kline(f"2026-01-01T00:{minute:02d}:00Z") for minute in range(0, 55, 5)],
            now=pd.Timestamp("2026-01-01T01:00:00Z"),
        )
        self.assertTrue(resample_closed_1h_from_5m(incomplete).empty)
        complete = normalize_klines(
            [kline(f"2026-01-01T00:{minute:02d}:00Z", 100 + minute) for minute in range(0, 60, 5)],
            now=pd.Timestamp("2026-01-01T01:00:00Z"),
        )
        hourly = resample_closed_1h_from_5m(complete)
        self.assertEqual(len(hourly), 1)
        self.assertEqual(hourly.iloc[0]["open"], 100.0)
        self.assertEqual(hourly.iloc[0]["close"], 156.0)

    def test_audit_reports_coverage_and_gaps(self) -> None:
        frame = normalize_klines(
            [kline("2026-01-01T00:00:00Z"), kline("2026-01-01T00:10:00Z")],
            now=pd.Timestamp("2026-01-01T00:20:00Z"),
        )
        audit = audit_kline_frame(frame, min_coverage=0.99)
        self.assertEqual(audit["status"], "bad")
        self.assertEqual(audit["gap_count"], 1)

    def test_write_and_load_preserves_timezone_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "klines.parquet"
            frame = normalize_klines([kline("2026-01-01T00:00:00Z")], now=pd.Timestamp("2026-01-01T00:10:00Z"))
            write_kline_frame(frame, path)
            loaded = load_kline_frame(path)
        self.assertEqual(loaded.index[0], pd.Timestamp("2026-01-01T00:00:00Z"))
        self.assertIsNotNone(loaded.index.tz)

    def test_config_rejects_private_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            good = config(Path(tmpdir))
            bad = KlineCollectorConfig(**{**good.__dict__, "rest_klines_endpoint": "/api/v3/order"})
            with self.assertRaises(ValueError):
                validate_kline_config(bad)

    def test_futures_public_kline_config_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spot = config(Path(tmpdir))
            futures = KlineCollectorConfig(
                **{
                    **spot.__dict__,
                    "market": "futures_usdm",
                    "rest_base_urls": ("https://fapi.binance.com",),
                    "rest_klines_endpoint": "/fapi/v1/klines",
                }
            )
            validate_kline_config(futures)

    def test_market_endpoint_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spot = config(Path(tmpdir))
            bad = KlineCollectorConfig(
                **{
                    **spot.__dict__,
                    "market": "futures_usdm",
                    "rest_base_urls": ("https://fapi.binance.com",),
                }
            )
            with self.assertRaises(ValueError):
                validate_kline_config(bad)

    def test_unapproved_kline_host_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spot = config(Path(tmpdir))
            bad = KlineCollectorConfig(**{**spot.__dict__, "rest_base_urls": ("https://example.com",)})
            with self.assertRaises(ValueError):
                validate_kline_config(bad)

    def test_healthcheck_detects_fresh_and_stale_closed_klines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = config(Path(tmpdir))
            write_health(cfg, "running", pd.Timestamp.now(tz="UTC"))
            ok, message = healthcheck(cfg, max_age_minutes=15)
            self.assertTrue(ok, message)
            write_health(cfg, "running", pd.Timestamp("2020-01-01T00:00:00Z"))
            ok, message = healthcheck(cfg, max_age_minutes=15)
        self.assertFalse(ok)
        self.assertIn("stale", message)

    def test_seed_paginates_until_final_partial_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = KlineCollectorConfig(**{**config(Path(tmpdir)).__dict__, "rest_limit": 2})

            def fake_fetch(_cfg: KlineCollectorConfig, start_time_ms: int | None = None, end_time_ms: int | None = None):
                start = pd.to_datetime(start_time_ms, unit="ms", utc=True)
                if fake_fetch.calls == 0:
                    payload = [kline(start.isoformat()), kline((start + pd.Timedelta(minutes=5)).isoformat())]
                else:
                    payload = [kline(start.isoformat())]
                fake_fetch.calls += 1
                return payload, 1.0, f"page{fake_fetch.calls}"

            fake_fetch.calls = 0
            with patch("src.infra.binance_kline_collector.fetch_klines", side_effect=fake_fetch) as mocked:
                payload = seed_klines(cfg)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(payload["rows_fetched"], 3)
        self.assertEqual(payload["request_count"], 2)


if __name__ == "__main__":
    unittest.main()
