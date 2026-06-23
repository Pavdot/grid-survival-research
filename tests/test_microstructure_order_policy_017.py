from __future__ import annotations

import unittest

import pandas as pd

from src.research.microstructure_execution_filter_017 import (
    MicrostructureGateConfig,
    load_locked_017_candidates,
    normalize_depth_snapshot,
)
from src.research.microstructure_order_policy_017 import (
    choose_event_snapshot,
    find_maker_fill,
    order_policy_from_yaml,
    simulate_policy_order,
    split_order_notional,
    validate_order_policy_config,
)


def gate_config() -> MicrostructureGateConfig:
    return MicrostructureGateConfig(
        depth_band_bps=5.0,
        max_spread_bps=20.0,
        max_order_book_share=0.05,
        min_side_depth_usdt_floor=100.0,
        max_abs_depth_imbalance=0.95,
        max_snapshot_age_ms=2000.0,
        max_theoretical_slippage_bps=50.0,
    )


def base_config() -> dict:
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
            "policies": ["gate_only_taker", "sliced_taker", "maker_entry_add_taker_exit", "maker_all_non_forced"],
            "max_clip_book_share": 0.01,
            "max_total_book_share": 0.05,
            "maker_ttl_seconds": 10,
            "max_reprices": 1,
            "synthetic_mapping_seed": 7,
        },
    }


def snapshot_row(ts: str, bid: float, ask: float) -> dict:
    return normalize_depth_snapshot(
        "BTCUSDT",
        {
            "lastUpdateId": int(pd.Timestamp(ts).timestamp()),
            "bids": [[f"{bid:.2f}", "20.0"], [f"{bid - 0.02:.2f}", "20.0"]],
            "asks": [[f"{ask:.2f}", "20.0"], [f"{ask + 0.02:.2f}", "20.0"]],
        },
        {
            "symbol": "BTCUSDT",
            "bidPrice": f"{bid:.2f}",
            "bidQty": "20.0",
            "askPrice": f"{ask:.2f}",
            "askQty": "20.0",
        },
        [1, 2, 5, 10],
        source_latency_ms=20.0,
        depth_source_url="depth",
        book_ticker_source_url="ticker",
        snapshot_time=pd.Timestamp(ts),
    )


def snapshots() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            snapshot_row("2026-01-01T00:00:00Z", 100.00, 100.10),
            snapshot_row("2026-01-01T00:00:05Z", 99.90, 100.00),
            snapshot_row("2026-01-01T00:00:20Z", 99.80, 99.90),
        ]
    )
    frame["snapshot_time_utc"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    return frame


def order_event(event_type: str = "entry", action: str = "buy", notional: float = 1000.0) -> pd.Series:
    return pd.Series(
        {
            "event_id": "event_1",
            "event_timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
            "event_type": event_type,
            "action": action,
            "account_equity_usdt": 1000.0,
            "order_notional_usdt": notional,
        }
    )


class MicrostructureOrderPolicy017Tests(unittest.TestCase):
    def test_slicing_never_exceeds_max_clip_book_share(self) -> None:
        clips = split_order_notional(2500.0, side_depth=100000.0, max_clip_book_share=0.01)
        self.assertEqual(clips, [1000.0, 1000.0, 500.0])
        self.assertLessEqual(max(clips), 100000.0 * 0.01)

    def test_maker_fill_uses_future_snapshot_only(self) -> None:
        frame = snapshots()
        filled, fill_pos, attempts, waited = find_maker_fill(frame, 0, "buy", ttl_seconds=10.0, max_reprices=0)
        self.assertTrue(filled)
        self.assertEqual(fill_pos, 1)
        self.assertEqual(attempts, 1)
        self.assertGreater(waited, 0.0)

    def test_maker_does_not_fill_on_same_snapshot(self) -> None:
        frame = snapshots().head(1).copy()
        filled, fill_pos, attempts, _waited = find_maker_fill(frame, 0, "buy", ttl_seconds=10.0, max_reprices=0)
        self.assertFalse(filled)
        self.assertIsNone(fill_pos)
        self.assertEqual(attempts, 1)

    def test_ttl_reprice_cancels_unfilled_order(self) -> None:
        frame = snapshots()
        filled, fill_pos, _attempts, _waited = find_maker_fill(frame, 0, "buy", ttl_seconds=1.0, max_reprices=0)
        self.assertFalse(filled)
        self.assertIsNone(fill_pos)

    def test_forced_exit_is_taker_and_cannot_be_skipped(self) -> None:
        config = base_config()
        policy = order_policy_from_yaml(config)
        event = order_event(event_type="forced_exit", action="sell", notional=1_000_000.0)
        result = simulate_policy_order(
            "maker_all_non_forced",
            event,
            snapshots(),
            snapshots().iloc[0],
            0,
            gate_config(),
            policy,
            snapshot_age_ms=20.0,
        )
        self.assertEqual(result["execution_style"], "taker")
        self.assertTrue(result["executed"])
        self.assertFalse(result["skipped"])

    def test_locked_017_candidates_are_loaded_without_reselection(self) -> None:
        locked = load_locked_017_candidates(base_config())
        self.assertFalse(locked.empty)
        self.assertEqual(set(locked["variant"].unique()), {"fundamental_trend_escape_entry_only"})
        self.assertTrue(locked["selected_from_validation_only"].astype(bool).all())

    def test_synthetic_mapping_is_explicitly_tagged(self) -> None:
        event = order_event()
        event["event_timestamp"] = pd.Timestamp("2020-01-01T12:00:00Z")
        snapshot, _pos, mapping_mode, _age = choose_event_snapshot(event, snapshots(), gate_config(), seed=1)
        self.assertEqual(mapping_mode, "synthetic_microstructure_mapping")
        self.assertIn("snapshot_time_utc", snapshot)

    def test_private_endpoint_is_rejected(self) -> None:
        config = base_config()
        config["collection"]["depth_endpoint"] = "/api/v3/order"
        with self.assertRaises(ValueError):
            validate_order_policy_config(config)


if __name__ == "__main__":
    unittest.main()
