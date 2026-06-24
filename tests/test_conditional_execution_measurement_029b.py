from __future__ import annotations

import unittest

import pandas as pd

from src.research.conditional_execution_measurement_029b import (
    choose_conditional_snapshot,
    coverage_report,
    events_for_variants,
    simulate_conditional_orders,
    summarize_conditional_costs,
)
from src.research.microstructure_execution_filter_017 import MicrostructureGateConfig, normalize_depth_snapshot
from src.research.microstructure_order_policy_017 import SnapshotCache


def policy_config() -> dict:
    return {
        "source_iteration": {"reference_monthly_return": 0.13},
        "collection": {
            "base_urls": ["https://api.binance.com"],
            "depth_endpoint": "/api/v3/depth",
            "book_ticker_endpoint": "/api/v3/ticker/bookTicker",
            "depth_limit": 500,
            "timeout_seconds": 10,
            "snapshot_path": "data/microstructure/test.parquet",
        },
        "microstructure_gate": {
            "account_equity_usdt_grid": [1000, 10000],
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
            "policies": ["gate_only_taker", "maker_entry_add_taker_exit"],
            "max_clip_book_share": 0.01,
            "max_total_book_share": 0.05,
            "maker_ttl_seconds": 10,
            "max_reprices": 1,
            "synthetic_mapping_seed": 7,
        },
    }


def gate() -> MicrostructureGateConfig:
    return MicrostructureGateConfig(5.0, 20.0, 0.05, 100.0, 0.95, 2000.0, 50.0)


def snapshot_row(ts: str, bid: float = 100.0, ask: float = 100.02) -> dict:
    return normalize_depth_snapshot(
        "BTCUSDT",
        {
            "lastUpdateId": int(pd.Timestamp(ts).timestamp()),
            "bids": [[f"{bid:.2f}", "1000.0"], [f"{bid - 0.01:.2f}", "1000.0"]],
            "asks": [[f"{ask:.2f}", "1000.0"], [f"{ask + 0.01:.2f}", "1000.0"]],
        },
        {"bidPrice": f"{bid:.2f}", "bidQty": "1000.0", "askPrice": f"{ask:.2f}", "askQty": "1000.0"},
        [1, 2, 5, 10],
        source_latency_ms=10.0,
        depth_source_url="depth",
        book_ticker_source_url="ticker",
        snapshot_time=pd.Timestamp(ts),
    )


def snapshots() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            snapshot_row("2026-01-01T00:00:00Z"),
            snapshot_row("2026-01-01T00:00:01Z"),
            snapshot_row("2026-01-01T00:00:05Z", bid=100.01, ask=100.03),
            snapshot_row("2026-01-01T01:00:00Z", bid=101.0, ask=101.02),
        ]
    )
    frame["snapshot_time_utc"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    return frame


def trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": 1,
                "family": "fundamental_trend_escape_v2",
                "variant": "fundamental_trend_escape_entry_only",
                "fold_id": 1,
                "start_timestamp": pd.Timestamp("2026-01-01T00:00:01.500Z"),
                "exit_timestamp": pd.Timestamp("2026-01-01T00:00:05Z"),
                "side": "long",
                "number_of_levels_filled": 1,
                "base_position_size_pct": 2.0,
                "progression_multiplier": 1.5,
                "max_total_exposure_pct": 30.0,
                "exit_reason": "take_profit",
                "name": "candidate",
            }
        ]
    )


class ConditionalExecutionMeasurement029BTests(unittest.TestCase):
    def test_snapshot_mapping_modes_are_explicit(self) -> None:
        cache = SnapshotCache.from_frame(snapshots())
        exact_event = pd.Series({"event_id": "a", "event_timestamp": pd.Timestamp("2026-01-01T00:00:01.500Z")})
        _snap, _pos, mode, _age = choose_conditional_snapshot(exact_event, cache, gate(), 60000, seed=1)
        self.assertEqual(mode, "exact_signal_timestamp")
        near_event = pd.Series({"event_id": "b", "event_timestamp": pd.Timestamp("2026-01-01T00:00:30Z")})
        _snap, _pos, mode, _age = choose_conditional_snapshot(near_event, cache, gate(), 60000, seed=1)
        self.assertEqual(mode, "nearest_snapshot")
        far_event = pd.Series({"event_id": "c", "event_timestamp": pd.Timestamp("2026-02-01T12:00:00Z")})
        _snap, _pos, mode, _age = choose_conditional_snapshot(far_event, cache, gate(), 1000, seed=1)
        self.assertEqual(mode, "hourly_synthetic_mapping")

    def test_events_preserve_family_and_variant(self) -> None:
        events = events_for_variants(trades(), [1000.0])
        self.assertFalse(events.empty)
        self.assertEqual(set(events["family"]), {"fundamental_trend_escape_v2"})
        self.assertIn("variant", events.columns)

    def test_conditional_cost_summary_contains_exact_fraction(self) -> None:
        events = events_for_variants(trades(), [1000.0])
        simulation = simulate_conditional_orders(snapshots(), events, policy_config(), nearest_snapshot_max_age_ms=60000)
        summary = summarize_conditional_costs(simulation)
        self.assertFalse(summary.empty)
        self.assertIn("exact_signal_fraction", summary.columns)
        self.assertIn("mean_execution_cost_pct_equity", summary.columns)

    def test_coverage_blocks_verdict_when_exact_fraction_is_low(self) -> None:
        simulation = pd.DataFrame({"mapping_mode": ["hourly_synthetic_mapping"] * 10})
        report = coverage_report(
            simulation,
            {"microstructure": {"exact_coverage_threshold": 0.8, "collection_days_required": 0}},
            snapshots(),
        )
        self.assertIn("insufficient exact coverage", str(report.iloc[0]["verdict"]))


if __name__ == "__main__":
    unittest.main()
