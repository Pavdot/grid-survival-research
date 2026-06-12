from __future__ import annotations

import unittest

import pandas as pd

from src.labeling.grid_engine import simulate_grid_from_index
from src.labeling.grid_risk import validate_strategy_config
from src.research.economy_first_research import (
    make_economy_candidates,
    select_best_candidate,
    summarize_simulations,
)
from tests.test_grid_engine import base_config


def economy_config() -> dict:
    return {
        "search": {
            "take_profit_spacing_multipliers": [0.5, 1.0],
            "survival_min_net_profit_pcts": [0.0, 0.0001],
            "min_grid_fraction_baseline": 0.05,
            "min_grid_absolute_cap": 500,
        }
    }


def one_level_risk():
    config = base_config()
    config["grid"]["max_levels"] = 1
    config["grid"]["base_position_size_pct"] = 0.10
    config["grid"]["sizing_sequence"] = [1.0]
    config["risk"]["max_holding_hours"] = 0.25
    return validate_strategy_config(config)


def simple_market() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:05:00Z", periods=8, freq="5min")
    close = pd.Series([100.0, 100.1, 100.1, 100.1, 100.1, 100.1, 100.1, 100.1], index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": [100.0, 100.3, 100.3, 100.3, 100.3, 100.3, 100.3, 100.3],
            "low": [99.9] * 8,
            "close": close,
            "volume": 1,
            "atr_5m": 1.0,
            "breakout_risk": 0,
            "volatility_shock": 0,
        },
        index=index,
    )


class EconomyFirstResearchTests(unittest.TestCase):
    def test_net_profit_label_can_mark_take_profit_as_not_survived(self) -> None:
        risk = one_level_risk()
        result = simulate_grid_from_index(
            simple_market(),
            0,
            risk,
            take_profit_spacing_multiplier=0.5,
            survival_min_realized_pnl=1.0,
        )
        self.assertEqual(result.exit_reason, "take_profit")
        self.assertEqual(result.grid_survived, 0)

    def test_take_profit_multiplier_changes_exit_behavior(self) -> None:
        risk = one_level_risk()
        low_tp = simulate_grid_from_index(
            simple_market(),
            0,
            risk,
            take_profit_spacing_multiplier=0.5,
            survival_min_realized_pnl=0.0,
        )
        high_tp = simulate_grid_from_index(
            simple_market(),
            0,
            risk,
            take_profit_spacing_multiplier=2.0,
            survival_min_realized_pnl=0.0,
        )
        self.assertEqual(low_tp.exit_reason, "take_profit")
        self.assertNotEqual(high_tp.exit_reason, "take_profit")

    def test_candidate_generation_rejects_invalid_values(self) -> None:
        config = economy_config()
        config["search"]["take_profit_spacing_multipliers"] = [0]
        with self.assertRaises(ValueError):
            make_economy_candidates(config)

    def test_selection_refuses_test_rows(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "split": "test",
                    "take_profit_spacing_multiplier": 1.0,
                    "survival_min_net_profit_pct": 0.0,
                    "baseline_grids": 100,
                    "number_of_grids": 100,
                    "expectancy": 1.0,
                    "profit_factor": 2.0,
                    "max_drawdown": 0.0,
                    "number_of_forced_exits": 0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            select_best_candidate(summary, economy_config())

    def test_summary_reports_positive_take_profit_rate(self) -> None:
        frame = pd.DataFrame(
            {
                "realized_pnl": [0.1, -0.2],
                "exit_reason": ["take_profit", "regime_break"],
                "grid_survived": [1, 0],
                "stopped_by_regime_break": [0, 1],
                "stopped_by_max_loss": [0, 0],
                "stopped_by_max_holding": [0, 0],
                "stopped_by_volatility_shock": [0, 0],
                "stopped_by_exposure": [0, 0],
                "stopped_by_kill_switch": [0, 0],
                "fees_paid": [0.01, 0.01],
                "slippage_paid": [0.01, 0.01],
                "time_to_exit": [1, 2],
                "unrealized_drawdown_max": [0.01, 0.02],
                "number_of_levels_filled": [1, 2],
            }
        )
        summary = summarize_simulations(frame)
        self.assertEqual(summary["positive_take_profit_rate"], 1.0)
        self.assertEqual(summary["number_of_regime_exits"], 1)


if __name__ == "__main__":
    unittest.main()

