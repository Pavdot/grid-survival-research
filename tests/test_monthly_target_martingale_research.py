from __future__ import annotations

import unittest

import pandas as pd

from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    build_side_signal,
    make_candidates,
    monthly_return_from_equity,
    planned_exposure,
    select_best_exact,
)


def martingale_config() -> dict:
    return {
        "target": {"monthly_return": 0.20, "max_drawdown": -0.35, "min_exact_grids": 1},
        "search": {
            "side_modes": ["mean_reversion_dual"],
            "rsi_windows": [14],
            "rsi_threshold_pairs": [[35, 65]],
            "entry_modes": ["hourly_extreme"],
            "entry_cooldown_hours": [3],
            "spacing_atr_multipliers": [0.5],
            "take_profit_spacing_multipliers": [0.5],
            "max_levels": [3],
            "base_position_size_pcts": [0.02],
            "progression_multipliers": [1.35],
            "max_total_exposure_pcts": [0.5],
            "fee_rates": [0.0004],
            "slippage_bps_values": [2],
            "max_grid_loss_pcts": [0.02],
            "max_holding_hours": [6],
            "stop_on_regime_break": [True],
            "stop_on_volatility_shock": [True],
            "max_progression_multiplier": 1.75,
            "exact_top_n": 5,
        },
    }


class MonthlyTargetMartingaleResearchTests(unittest.TestCase):
    def test_candidate_generation_keeps_planned_exposure_bounded(self) -> None:
        candidate = make_candidates(martingale_config())[0]
        self.assertLessEqual(planned_exposure(candidate), candidate.max_total_exposure_pct)
        self.assertGreater(candidate.progression_multiplier, 1.0)

    def test_rejects_exponential_like_progression_cap(self) -> None:
        config = martingale_config()
        config["search"]["max_progression_multiplier"] = 2.0
        with self.assertRaises(ValueError):
            make_candidates(config)

    def test_selection_refuses_test_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "split": "test",
                    "name": "bad",
                    "number_of_grids": 10,
                    "monthly_return": 1.0,
                    "max_drawdown": -0.01,
                    "profit_factor": 2.0,
                    "number_of_forced_exits": 0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            select_best_exact(frame, martingale_config())

    def test_monthly_return_normalizes_to_month(self) -> None:
        index = pd.date_range("2024-01-01", periods=31, freq="D", tz="UTC")
        equity = pd.Series(1.20, index=index)
        equity.iloc[0] = 1.0
        self.assertAlmostEqual(monthly_return_from_equity(equity), 0.20, places=2)

    def test_side_signal_uses_rsi_extremes_without_lookahead_shape_change(self) -> None:
        index = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
        close = pd.Series(list(range(20)) + list(range(20, 0, -1)), index=index, dtype=float)
        market = pd.DataFrame({"close": close}, index=index)
        candidate = MonthlyMartingaleCandidate(
            name="test",
            side_mode="mean_reversion_dual",
            entry_mode="hourly_extreme",
            entry_cooldown_hours=3.0,
            rsi_window=3,
            rsi_low=35.0,
            rsi_high=65.0,
            spacing_atr_multiplier=0.5,
            take_profit_spacing_multiplier=0.5,
            max_levels=3,
            base_position_size_pct=0.02,
            progression_multiplier=1.35,
            max_total_exposure_pct=0.5,
            fee_rate=0.0004,
            slippage_bps=2.0,
            max_grid_loss_pct=0.02,
            max_holding_hours=6,
            stop_on_regime_break=True,
            stop_on_volatility_shock=True,
        )
        signal = build_side_signal(market, market, candidate)
        self.assertEqual(len(signal), len(market))
        self.assertIn("short", set(signal.dropna()))
        self.assertIn("long", set(signal.dropna()))


if __name__ == "__main__":
    unittest.main()
