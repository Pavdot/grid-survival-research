from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.infra.binance_market_connectivity import (
    RestCheck,
    WebSocketCheck,
    _unwrap_stream_payload,
    load_checks,
    run_rest_check,
    summarize_depth_payload,
    validate_public_rest_endpoint,
    validate_public_websocket_url,
    websocket_payload_summary,
)


class BinanceMarketConnectivityTests(unittest.TestCase):
    def test_rejects_private_or_trading_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            validate_public_rest_endpoint("/fapi/v1/order")
        with self.assertRaises(ValueError):
            validate_public_rest_endpoint("/fapi/v2/account")
        with self.assertRaises(ValueError):
            validate_public_websocket_url("wss://fstream.binance.com/private/ws?listenKey=abc")

    def test_depth_summary_reports_spread_and_band_depth(self) -> None:
        summary = summarize_depth_payload(
            {
                "lastUpdateId": 123,
                "bids": [["100.00", "1"], ["99.98", "2"]],
                "asks": [["100.02", "1.5"], ["100.04", "3"]],
            }
        )
        self.assertEqual(summary["last_update_id"], 123)
        self.assertGreater(summary["spread_bps"], 0)
        self.assertGreater(summary["bid_depth_5bps_usdt"], 0)
        self.assertGreater(summary["ask_depth_5bps_usdt"], 0)

    def test_unwraps_combined_stream_payload(self) -> None:
        payload = _unwrap_stream_payload({"stream": "btcusdt@depth", "data": {"e": "depthUpdate", "s": "BTCUSDT"}})
        self.assertEqual(payload["e"], "depthUpdate")

    def test_websocket_summary_handles_book_ticker_and_kline(self) -> None:
        ticker = websocket_payload_summary(
            {
                "e": "bookTicker",
                "E": 1767225600000,
                "s": "BTCUSDT",
                "u": 5,
                "b": "100.0",
                "B": "1.0",
                "a": "100.1",
                "A": "2.0",
            }
        )
        self.assertEqual(ticker["event_time_utc"], pd.Timestamp("2026-01-01T00:00:00Z").isoformat())
        self.assertGreater(ticker["spread_bps"], 0)
        kline = websocket_payload_summary(
            {
                "e": "kline",
                "E": 1767225600000,
                "s": "BTCUSDT",
                "k": {"i": "5m", "x": False, "t": 1767225600000},
            }
        )
        self.assertEqual(kline["kline_interval"], "5m")
        self.assertFalse(kline["kline_closed"])

    def test_load_checks_reads_public_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(
                """
output:
  dir: reports/tmp
venues:
  futures_usdm:
    enabled: true
    rest:
      - name: depth
        base_url: https://fapi.binance.com
        endpoint: /fapi/v1/depth
        params: {symbol: BTCUSDT, limit: 100}
        kind: depth
    websocket:
      - name: depth_ws
        url: wss://fstream.binance.com/public/ws/btcusdt@depth@100ms
        required_fields: [e, U, u]
        expected_event: depthUpdate
""",
                encoding="utf-8",
            )
            rest_checks, ws_checks, _output_dir = load_checks(path)
        self.assertEqual(len(rest_checks), 1)
        self.assertEqual(len(ws_checks), 1)
        self.assertIsInstance(rest_checks[0], RestCheck)
        self.assertIsInstance(ws_checks[0], WebSocketCheck)

    def test_rest_check_captures_failure_without_raising(self) -> None:
        check = RestCheck(
            venue="test",
            name="depth",
            base_url="https://example.invalid",
            endpoint="/fapi/v1/depth",
            params={"symbol": "BTCUSDT"},
            kind="depth",
            timeout_seconds=1.0,
        )
        with patch("src.infra.binance_market_connectivity._json_get", side_effect=RuntimeError("boom")):
            row = run_rest_check(check)
        self.assertEqual(row["status"], "failed")
        self.assertIn("boom", row["message"])


if __name__ == "__main__":
    unittest.main()
