from __future__ import annotations

import unittest

import pandas as pd

from src.research.microstructure_execution_filter_017 import (
    MicrostructureGateConfig,
    evaluate_gate,
    load_locked_017_candidates,
    normalize_depth_snapshot,
    quote_depth_within_band,
    validate_microstructure_config,
    walk_book_slippage_bps,
)


def depth_payload() -> dict:
    return {
        "lastUpdateId": 123,
        "bids": [["100.00", "1.0"], ["99.98", "2.0"], ["99.90", "5.0"]],
        "asks": [["100.10", "1.5"], ["100.12", "2.0"], ["100.30", "5.0"]],
    }


def book_ticker_payload() -> dict:
    return {
        "symbol": "BTCUSDT",
        "bidPrice": "100.00",
        "bidQty": "1.0",
        "askPrice": "100.10",
        "askQty": "1.5",
    }


def gate_config(**overrides) -> MicrostructureGateConfig:
    values = {
        "depth_band_bps": 5.0,
        "max_spread_bps": 20.0,
        "max_order_book_share": 0.05,
        "min_side_depth_usdt_floor": 100.0,
        "max_abs_depth_imbalance": 0.35,
        "max_snapshot_age_ms": 2000.0,
        "max_theoretical_slippage_bps": 10.0,
    }
    values.update(overrides)
    return MicrostructureGateConfig(**values)


def normalized_snapshot() -> pd.Series:
    row = normalize_depth_snapshot(
        "BTCUSDT",
        depth_payload(),
        book_ticker_payload(),
        [1, 2, 5, 10],
        source_latency_ms=50.0,
        depth_source_url="https://example/depth",
        book_ticker_source_url="https://example/ticker",
        snapshot_time=pd.Timestamp("2026-01-01T00:00:00Z"),
    )
    return pd.Series(row)


class MicrostructureExecutionFilter017Tests(unittest.TestCase):
    def test_spread_bps_from_best_bid_ask(self) -> None:
        snapshot = normalized_snapshot()
        expected = (100.10 - 100.00) / ((100.10 + 100.00) / 2.0) * 10000.0
        self.assertAlmostEqual(float(snapshot["spread_bps"]), expected)

    def test_quote_depth_within_band_for_bid_and_ask(self) -> None:
        bids = [(100.00, 1.0), (99.98, 2.0), (99.90, 5.0)]
        asks = [(100.10, 1.5), (100.12, 2.0), (100.30, 5.0)]
        self.assertAlmostEqual(quote_depth_within_band(bids, 100.00, 5.0, "bid"), 100.00 * 1.0 + 99.98 * 2.0)
        self.assertAlmostEqual(quote_depth_within_band(asks, 100.10, 5.0, "ask"), 100.10 * 1.5 + 100.12 * 2.0)

    def test_order_book_share_rejects_large_order(self) -> None:
        snapshot = normalized_snapshot()
        decision = evaluate_gate(
            snapshot,
            "long",
            account_equity_usdt=1000.0,
            initial_notional_multiplier=50.0,
            gate=gate_config(),
        )
        self.assertFalse(decision["authorized"])
        self.assertIn("order_too_large_for_book", decision["reject_reasons"])

    def test_imbalance_rejects_violent_book(self) -> None:
        snapshot = normalized_snapshot().copy()
        snapshot["depth_imbalance_5bps"] = 0.90
        decision = evaluate_gate(
            snapshot,
            "short",
            account_equity_usdt=10.0,
            initial_notional_multiplier=1.0,
            gate=gate_config(),
        )
        self.assertFalse(decision["authorized"])
        self.assertIn("violent_imbalance", decision["reject_reasons"])

    def test_stale_snapshot_is_rejected_explicitly(self) -> None:
        snapshot = normalized_snapshot()
        decision = evaluate_gate(
            snapshot,
            "long",
            account_equity_usdt=10.0,
            initial_notional_multiplier=1.0,
            gate=gate_config(),
            snapshot_age_ms=5000.0,
        )
        self.assertFalse(decision["authorized"])
        self.assertIn("stale_snapshot", decision["reject_reasons"])

    def test_walk_book_slippage_uses_multiple_levels(self) -> None:
        levels = [(100.00, 1.0), (101.00, 1.0)]
        slippage = walk_book_slippage_bps(levels, 150.0, "buy")
        self.assertGreater(slippage, 0.0)
        self.assertLess(slippage, 100.0)

    def test_empty_book_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_depth_snapshot(
                "BTCUSDT",
                {"lastUpdateId": 1, "bids": [], "asks": []},
                book_ticker_payload(),
                [5],
                1.0,
                "depth",
                "ticker",
            )

    def test_config_validation_rejects_bad_thresholds(self) -> None:
        config = {
            "microstructure_gate": {
                "depth_band_bps": 5,
                "depth_bands_bps": [1, 2, 5],
                "max_spread_bps": 0.5,
                "max_order_book_share": 2.0,
                "min_side_depth_usdt_floor": 250000,
                "max_abs_depth_imbalance": 0.35,
                "max_snapshot_age_ms": 2000,
                "max_theoretical_slippage_bps": 1.0,
                "account_equity_usdt_grid": [1000],
            }
        }
        with self.assertRaises(ValueError):
            validate_microstructure_config(config)

    def test_locked_017_loader_filters_variant_without_reselection(self) -> None:
        config = {
            "source_iteration": {
                "output_dir": "reports/research_iterations/iteration_017_fundamental_trend_escape_best_policy_expanded",
                "variant": "fundamental_trend_escape_entry_only",
            }
        }
        locked = load_locked_017_candidates(config)
        self.assertFalse(locked.empty)
        self.assertEqual(set(locked["variant"].unique()), {"fundamental_trend_escape_entry_only"})
        self.assertIn("fold_id", locked.columns)


if __name__ == "__main__":
    unittest.main()
