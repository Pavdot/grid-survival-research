from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.paper.shadow_live_037 import (
    assert_shadow_safe,
    latest_actionable_signal,
    locked_candidate,
    open_position_from_signal,
    update_position_on_closed_bars,
)


def base_config(tmp: Path | None = None) -> dict:
    output = str((tmp or Path("tmp")) / "shadow")
    return {
        "shadow_live": {
            "output_dir": output,
            "infrastructure_config": "config/infrastructure_microstructure.yaml",
            "policy_config": "config/research_iteration_microstructure_order_policy_017.yaml",
            "primary_equity_usdt": 10000,
            "default_policy": "maker_entry_add_taker_exit",
            "signal_lag_bars": 1,
            "mask_lag_bars": 1,
            "private_env_vars": ["BINANCE_API_KEY", "BINANCE_SECRET_KEY"],
            "live_trading_env_var": "LIVE_TRADING_ENABLED",
        },
        "strategy_037": {
            "candidate": {
                "name": "test_037",
                "side_mode": "mean_reversion_dual",
                "entry_mode": "hourly_extreme",
                "entry_cooldown_hours": 3.0,
                "rsi_window": 24,
                "rsi_low": 40.0,
                "rsi_high": 60.0,
                "spacing_atr_multiplier": 4.5,
                "take_profit_spacing_multiplier": 2.0,
                "max_levels": 5,
                "base_position_size_pct": 2.5,
                "progression_multiplier": 1.55,
                "max_total_exposure_pct": 50.0,
                "fee_rate": 0.0,
                "slippage_bps": 0.25,
                "max_grid_loss_pct": 0.35,
                "max_holding_hours": 6.0,
                "stop_on_regime_break": False,
                "stop_on_volatility_shock": True,
            }
        },
    }


def market_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="5min")
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100],
            "high": [101, 100.5, 103, 103, 103],
            "low": [99.5, 100.0, 99.0, 100, 100],
            "close": [100, 100, 100, 100, 100],
            "volume": [1, 1, 1, 1, 1],
            "atr_5m": [1, 1, 1, 1, 1],
            "volatility_shock": [0, 0, 0, 0, 0],
        },
        index=index,
    )


def snapshot() -> pd.Series:
    return pd.Series(
        {
            "snapshot_time_utc": pd.Timestamp("2026-01-01T00:05:00Z"),
            "best_bid": 99.99,
            "best_ask": 100.01,
        }
    )


class ShadowLive037Tests(unittest.TestCase):
    def test_private_keys_and_live_flag_are_refused(self) -> None:
        config = base_config()
        with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "true"}, clear=True):
            with self.assertRaises(RuntimeError):
                assert_shadow_safe(config)
        with patch.dict(os.environ, {"BINANCE_API_KEY": "secret"}, clear=True):
            with self.assertRaises(RuntimeError):
                assert_shadow_safe(config)

    def test_latest_actionable_signal_uses_next_bar_not_same_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            market = market_frame()
            shifted_signal = pd.Series(pd.NA, index=market.index, dtype="object")
            shifted_signal.iloc[0] = "long"
            shifted_mask = pd.Series(False, index=market.index)
            early = latest_actionable_signal(
                market,
                shifted_signal,
                shifted_mask,
                pd.Timestamp("2026-01-01T00:04:59Z"),
                output_dir,
                lookback_bars=5,
            )
            ready = latest_actionable_signal(
                market,
                shifted_signal,
                shifted_mask,
                pd.Timestamp("2026-01-01T00:05:00Z"),
                output_dir,
                lookback_bars=5,
            )
        self.assertIsNone(early)
        self.assertIsNotNone(ready)
        assert ready is not None
        self.assertEqual(ready["decision_timestamp_utc"], pd.Timestamp("2026-01-01T00:00:00Z").isoformat())
        self.assertEqual(ready["entry_timestamp_utc"], pd.Timestamp("2026-01-01T00:05:00Z").isoformat())

    def test_open_position_records_no_real_order_and_locked_candidate(self) -> None:
        config = base_config()
        candidate = locked_candidate(config)
        self.assertEqual(candidate.base_position_size_pct, 2.5)
        signal = {
            "entry_timestamp_utc": pd.Timestamp("2026-01-01T00:05:00Z").isoformat(),
            "side": "long",
        }
        gate = {"authorized": True, "reasons": "pass", "theoretical_slippage_bps": 0.2}
        position, orders, fills = open_position_from_signal(signal, market_frame(), snapshot(), config, gate)
        self.assertFalse(bool(position["real_order_sent"]))
        self.assertEqual(position["status"], "open")
        self.assertEqual(len(orders), 1)
        self.assertEqual(len(fills), 1)

    def test_grid_state_add_then_take_profit_closes_position(self) -> None:
        config = base_config()
        signal = {
            "entry_timestamp_utc": pd.Timestamp("2026-01-01T00:05:00Z").isoformat(),
            "side": "long",
        }
        position, _orders, _fills = open_position_from_signal(signal, market_frame(), snapshot(), config, {"authorized": True})
        position["spacing"] = 1.0
        updated, orders, fills = update_position_on_closed_bars(
            pd.Series(position),
            market_frame(),
            config,
            pd.Series(False, index=market_frame().index),
        )
        self.assertEqual(updated["status"], "closed")
        self.assertEqual(updated["exit_reason"], "take_profit")
        self.assertGreaterEqual(updated["levels_filled"], 2)
        self.assertTrue(any(order["order_type"] == "add" for order in orders))
        self.assertTrue(any(order["order_type"] == "exit" for order in orders))
        self.assertTrue(all(not fill["real_order_sent"] for fill in fills))

    def test_grid_state_forced_exit_on_volatility_shock(self) -> None:
        config = base_config()
        signal = {
            "entry_timestamp_utc": pd.Timestamp("2026-01-01T00:05:00Z").isoformat(),
            "side": "long",
        }
        position, _orders, _fills = open_position_from_signal(signal, market_frame(), snapshot(), config, {"authorized": True})
        frame = market_frame()
        frame.loc[frame.index[2], "volatility_shock"] = 1
        updated, _orders, _fills = update_position_on_closed_bars(
            pd.Series(position),
            frame,
            config,
            pd.Series(False, index=frame.index),
        )
        self.assertEqual(updated["status"], "closed")
        self.assertEqual(updated["exit_reason"], "volatility_shock")


if __name__ == "__main__":
    unittest.main()
