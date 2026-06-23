from __future__ import annotations

import unittest

import pandas as pd

from src.research.microstructure_execution_filter_017 import normalize_depth_snapshot
from src.research.rolling_execution_evaluation import (
    decide_execution_viability,
    evaluate_policy_events,
    make_daily_slices,
)


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


def rolling_config() -> dict:
    return {
        "verdict": {
            "min_collection_hours": 24,
            "max_invalid_snapshot_fraction": 0.05,
            "max_p90_slippage_bps": 1.0,
            "max_entry_skipped_rate": 0.15,
            "adjusted_monthly_floor": 0.13,
            "required_equity_ceiling": 10000,
            "size_cap_probe": 25000,
        }
    }


def snapshot_row(ts: str) -> dict:
    return normalize_depth_snapshot(
        "BTCUSDT",
        {
            "lastUpdateId": int(pd.Timestamp(ts).timestamp()),
            "bids": [["100.00", "20.0"], ["99.99", "20.0"]],
            "asks": [["100.01", "20.0"], ["100.02", "20.0"]],
        },
        {"symbol": "BTCUSDT", "bidPrice": "100.00", "bidQty": "20.0", "askPrice": "100.01", "askQty": "20.0"},
        [1, 2, 5, 10],
        source_latency_ms=20.0,
        depth_source_url="depth",
        book_ticker_source_url="ticker",
        snapshot_time=pd.Timestamp(ts),
    )


def snapshots() -> pd.DataFrame:
    frame = pd.DataFrame([snapshot_row("2026-01-01T00:00:00Z"), snapshot_row("2026-01-01T00:00:01Z")])
    frame["snapshot_time_utc"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    return frame


def events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": 1,
                "fold_id": 1,
                "event_id": "e1",
                "event_timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "event_type": "entry",
                "level_index": 0,
                "position_side": "long",
                "action": "buy",
                "account_equity_usdt": 1000.0,
                "order_notional_usdt": 1000.0,
                "order_notional_multiplier": 1.0,
                "candidate_name": "test",
                "exit_reason": "take_profit",
            }
        ]
    )


class RollingExecutionEvaluationTests(unittest.TestCase):
    def test_evaluate_policy_events_tags_window_and_does_not_need_reselection(self) -> None:
        result = evaluate_policy_events(snapshots(), events(), policy_config(), "rolling_24h")
        self.assertFalse(result["comparison"].empty)
        self.assertEqual(set(result["comparison"]["window_label"]), {"rolling_24h"})
        self.assertEqual(set(result["simulation"]["mapping_mode"]), {"exact_signal_timestamp"})

    def test_daily_slices_group_by_utc_day(self) -> None:
        frame = pd.concat(
            [
                snapshots(),
                pd.DataFrame([snapshot_row("2026-01-02T00:00:00Z")]),
            ],
            ignore_index=True,
        )
        slices = make_daily_slices(frame)
        self.assertEqual([label for label, _group in slices], ["day_2026-01-01", "day_2026-01-02"])

    def test_decision_rejects_when_10k_fails(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "window_label": "rolling_24h",
                    "policy": "gate_only_taker",
                    "account_equity_usdt": 1000.0,
                    "p90_slippage_bps": 0.5,
                    "entry_skipped_rate": 0.0,
                    "estimated_monthly_after_execution": 0.14,
                },
                {
                    "window_label": "rolling_24h",
                    "policy": "gate_only_taker",
                    "account_equity_usdt": 10000.0,
                    "p90_slippage_bps": 5.0,
                    "entry_skipped_rate": 0.0,
                    "estimated_monthly_after_execution": 0.14,
                },
            ]
        )
        quality = pd.DataFrame(
            [
                {
                    "window_label": "rolling_24h",
                    "collection_span_hours": 25.0,
                    "invalid_snapshot_fraction": 0.0,
                }
            ]
        )
        decision = decide_execution_viability(comparison, quality, rolling_config())
        self.assertEqual(decision["verdict"], "execution not viable")


if __name__ == "__main__":
    unittest.main()
