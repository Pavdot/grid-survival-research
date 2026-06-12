from __future__ import annotations

import unittest

import pandas as pd

from src.labeling.grid_engine import simulate_grid_from_index
from src.labeling.grid_risk import validate_strategy_config
from tests.test_grid_engine import base_config


class LabelTests(unittest.TestCase):
    def test_grid_result_contains_required_label_fields(self) -> None:
        risk = validate_strategy_config(base_config())
        index = pd.date_range("2024-01-01 00:05:00Z", periods=30, freq="5min")
        close = pd.Series([100, 99.8, 99.6, 100.5] + [100.6] * 26, index=index)
        market = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 0.7,
                "close": close,
                "volume": 1,
                "atr_5m": 0.5,
                "breakout_risk": 0,
                "volatility_shock": 0,
            },
            index=index,
        )
        result = simulate_grid_from_index(market, 0, risk).to_dict()
        required = {
            "grid_survived",
            "max_adverse_excursion",
            "max_favorable_excursion",
            "realized_pnl",
            "unrealized_drawdown_max",
            "time_to_exit",
            "number_of_levels_filled",
            "stopped_by_regime_break",
            "stopped_by_max_loss",
            "stopped_by_max_holding",
        }
        self.assertTrue(required.issubset(result))


if __name__ == "__main__":
    unittest.main()

