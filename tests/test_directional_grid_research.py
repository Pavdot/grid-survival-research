from __future__ import annotations

import unittest

import pandas as pd

from src.research.directional_grid_research import make_candidates, select_best


def directional_config() -> dict:
    return {
        "search": {
            "sides": ["long", "short"],
            "take_profit_spacing_multipliers": [1.0],
            "spacing_atr_multipliers": [2.0],
            "max_levels": [1],
            "stop_on_regime_break": [True, False],
            "stop_on_volatility_shock": [True],
            "survival_min_net_profit_pcts": [0.0],
            "min_grid_fraction_baseline": 0.05,
            "min_grid_absolute_cap": 50,
        }
    }


class DirectionalGridResearchTests(unittest.TestCase):
    def test_make_candidates_includes_long_and_short(self) -> None:
        candidates = make_candidates(directional_config())
        self.assertEqual({candidate.side for candidate in candidates}, {"long", "short"})

    def test_selection_refuses_test_rows(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "split": "test",
                    "side": "short",
                    "spacing_atr_multiplier": 2.0,
                    "max_levels": 1,
                    "take_profit_spacing_multiplier": 1.0,
                    "stop_on_regime_break": False,
                    "stop_on_volatility_shock": True,
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
            select_best(summary, directional_config())


if __name__ == "__main__":
    unittest.main()
