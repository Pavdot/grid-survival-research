from __future__ import annotations

import unittest

import pandas as pd

from src.backtesting.metrics import calculate_metrics, drawdown_series


class MetricsTests(unittest.TestCase):
    def test_metrics_include_requested_core_values(self) -> None:
        index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        equity = pd.Series([1.0, 1.1, 1.05, 1.2, 1.15], index=index)
        trades = pd.DataFrame(
            {
                "realized_pnl": [0.10, -0.05, 0.15],
                "number_of_levels_filled": [1, 2, 1],
                "time_to_exit": [1, 2, 1],
                "fees_paid": [0.001, 0.002, 0.001],
                "slippage_paid": [0.001, 0.001, 0.001],
                "unrealized_drawdown_max": [0.01, 0.03, 0.01],
                "stopped_by_max_loss": [0, 1, 0],
                "stopped_by_max_holding": [0, 0, 0],
                "stopped_by_volatility_shock": [0, 0, 0],
                "stopped_by_exposure": [0, 0, 0],
                "stopped_by_kill_switch": [0, 0, 0],
                "stopped_by_regime_break": [0, 0, 0],
            }
        )
        metrics = calculate_metrics(equity, trades)
        self.assertIn("total_return", metrics)
        self.assertIn("profit_factor", metrics)
        self.assertIn("expectancy", metrics)
        self.assertEqual(metrics["number_of_grids_stopped_by_max_loss"], 1)
        self.assertLess(drawdown_series(equity).min(), 0)


if __name__ == "__main__":
    unittest.main()

