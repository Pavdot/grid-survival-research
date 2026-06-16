from __future__ import annotations

import unittest

import pandas as pd

from src.data.load_dukascopy_ohlcv import normalize_dukascopy_ohlcv
from src.data.resample_timeframes import resample_closed_ohlcv
from src.fundamentals.event_blackout import default_fundamental_events
from src.labeling.grid_engine import simulate_grid_from_index
from src.labeling.grid_risk import validate_strategy_config


def dukascopy_rows(timestamps: list[str], spread: float = 1.0) -> pd.DataFrame:
    rows = []
    for i, ts in enumerate(timestamps):
        price = 100.0 + i
        rows.append(
            {
                "timestamp": i,
                "datetime_utc": ts,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + 0.1,
                "bid_open": price - spread / 2,
                "bid_high": price + 0.5 - spread / 2,
                "bid_low": price - 0.5 - spread / 2,
                "bid_close": price + 0.1 - spread / 2,
                "bid_volume": 1.0,
                "ask_open": price + spread / 2,
                "ask_high": price + 0.5 + spread / 2,
                "ask_low": price - 0.5 + spread / 2,
                "ask_close": price + 0.1 + spread / 2,
                "ask_volume": 1.0,
                "volume": 2.0,
                "spread_open": spread,
                "spread_high": spread,
                "spread_low": spread,
                "spread_close": spread,
                "spread_avg": spread,
                "spread_bps_close": spread,
                "tick_count": 10,
            }
        )
    return pd.DataFrame(rows)


def spread_risk() -> dict:
    return {
        "fees": {"cost_model": "bid_ask_spread", "taker_fee": 0.0, "maker_fee": 0.0, "slippage_bps": 0},
        "grid": {
            "spacing_atr_multiplier": 1.0,
            "max_levels": 1,
            "base_position_size_pct": 0.10,
            "sizing_mode": "linear",
            "sizing_sequence": [1.0],
            "allow_exponential_martingale": False,
        },
        "risk": {
            "max_grid_loss_pct": 1.0,
            "max_daily_loss_pct": 1.0,
            "max_total_exposure_pct": 0.50,
            "max_holding_hours": 1,
            "stop_on_regime_break": False,
            "stop_on_volatility_shock": False,
        },
    }


class XauusdPortTests(unittest.TestCase):
    def test_dukascopy_loader_requires_timezone_and_accepts_market_gaps(self) -> None:
        frame = dukascopy_rows(
            [
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:05:00Z",
                "2024-01-01T00:10:00Z",
                "2024-01-01T03:00:00Z",
            ]
        )
        normalized = normalize_dukascopy_ohlcv(frame, timeframe="5m")
        self.assertEqual(str(normalized.index.tz), "UTC")
        self.assertIn("spread_close", normalized.columns)
        self.assertIn("open_time", normalized.columns)
        self.assertIn("close_time", normalized.columns)
        self.assertEqual(normalized.index[0], pd.Timestamp("2024-01-01T00:05:00Z"))

    def test_dukascopy_loader_rejects_naive_timestamps(self) -> None:
        frame = dukascopy_rows(["2024-01-01 00:00:00"])
        with self.assertRaises(ValueError):
            normalize_dukascopy_ohlcv(frame, timeframe="5m")

    def test_dukascopy_loader_rejects_non_positive_spread(self) -> None:
        frame = dukascopy_rows(["2024-01-01T00:00:00Z"], spread=0.0)
        with self.assertRaises(ValueError):
            normalize_dukascopy_ohlcv(frame, timeframe="5m")

    def test_resampling_preserves_bid_ask_spread_columns(self) -> None:
        frame = normalize_dukascopy_ohlcv(
            dukascopy_rows(["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z", "2024-01-01T00:10:00Z"]),
            timeframe="5m",
        )
        resampled = resample_closed_ohlcv(frame, "15m")
        self.assertEqual(len(resampled), 1)
        self.assertEqual(float(resampled.iloc[0]["tick_count"]), 30.0)
        self.assertAlmostEqual(float(resampled.iloc[0]["spread_avg"]), 1.0)
        self.assertAlmostEqual(float(resampled.iloc[0]["ask_high"]), float(frame["ask_high"].max()))

    def test_bid_ask_cost_model_charges_spread_on_long_execution(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00Z", periods=3, freq="5min")
        market = pd.DataFrame(
            {
                "open": [100.0, 101.0, 101.0],
                "high": [100.0, 102.0, 102.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 101.0, 101.0],
                "volume": 1.0,
                "spread_close": 2.0,
                "spread_avg": 2.0,
                "atr_5m": 1.0,
                "breakout_risk": 0,
                "volatility_shock": 0,
            },
            index=index,
        )
        result = simulate_grid_from_index(
            market,
            0,
            validate_strategy_config(spread_risk()),
            take_profit_spacing_multiplier=1.0,
            survival_min_realized_pnl=0.0,
            side="long",
        )
        self.assertEqual(result.exit_reason, "take_profit")
        self.assertAlmostEqual(result.realized_pnl, 0.0)
        self.assertGreater(result.slippage_paid, 0.0)

    def test_bid_ask_cost_model_requires_spread_columns(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00Z", periods=3, freq="5min")
        market = pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
                "atr_5m": 1.0,
                "breakout_risk": 0,
                "volatility_shock": 0,
            },
            index=index,
        )
        with self.assertRaises(ValueError):
            simulate_grid_from_index(market, 0, validate_strategy_config(spread_risk()))

    def test_xau_fundamental_seed_excludes_crypto_and_includes_ppi(self) -> None:
        events = default_fundamental_events("xau_macro")
        categories = set(events["category"])
        self.assertIn("macro_ppi", categories)
        self.assertNotIn("halving", categories)
        self.assertNotIn("exchange_hack", categories)


if __name__ == "__main__":
    unittest.main()
