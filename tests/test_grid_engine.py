from __future__ import annotations

import unittest

import pandas as pd

from src.labeling.grid_engine import simulate_grid_from_index
from src.labeling.grid_risk import validate_strategy_config


def base_config() -> dict:
    return {
        "fees": {"taker_fee": 0.0004, "maker_fee": 0.0002, "slippage_bps": 2},
        "grid": {
            "spacing_atr_multiplier": 0.5,
            "max_levels": 5,
            "base_position_size_pct": 0.10,
            "sizing_mode": "linear",
            "sizing_sequence": [1.0, 1.15, 1.30, 1.45, 1.60],
            "allow_exponential_martingale": False,
        },
        "risk": {
            "max_grid_loss_pct": 0.01,
            "max_daily_loss_pct": 0.025,
            "max_total_exposure_pct": 0.50,
            "max_holding_hours": 12,
            "stop_on_regime_break": True,
            "stop_on_volatility_shock": True,
        },
    }


class GridEngineTests(unittest.TestCase):
    def test_rejects_exponential_sizing(self) -> None:
        config = base_config()
        config["grid"]["sizing_sequence"] = [1, 2, 4, 8, 16]
        with self.assertRaises(ValueError):
            validate_strategy_config(config)

    def test_max_loss_stop_and_costs_are_applied(self) -> None:
        config = base_config()
        config["risk"]["max_grid_loss_pct"] = 0.001
        config["grid"]["max_levels"] = 1
        config["grid"]["base_position_size_pct"] = 0.50
        config["grid"]["sizing_sequence"] = [1.0]
        risk = validate_strategy_config(config)
        index = pd.date_range("2024-01-01 00:05:00Z", periods=80, freq="5min")
        close = pd.Series([100 - i * 0.2 for i in range(80)], index=index)
        market = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.05,
                "low": close - 0.40,
                "close": close,
                "volume": 1,
                "atr_5m": 0.5,
                "breakout_risk": 0,
                "volatility_shock": 0,
            },
            index=index,
        )
        result = simulate_grid_from_index(market, 0, risk)
        self.assertEqual(result.grid_survived, 0)
        self.assertEqual(result.stopped_by_max_loss, 1)
        self.assertGreater(result.fees_paid, 0)
        self.assertGreaterEqual(result.number_of_levels_filled, 1)
        self.assertLessEqual(result.max_exposure_pct, risk.max_total_exposure_pct)


if __name__ == "__main__":
    unittest.main()
