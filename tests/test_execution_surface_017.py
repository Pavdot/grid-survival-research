from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.labeling.grid_engine import simulate_grid_from_index
from src.labeling.grid_risk import validate_strategy_config
from src.research.execution_surface_017 import (
    ExecutionSurfaceCell,
    aggregate_surface_by_seed,
    locked_candidate_for_fold,
    make_surface_cells,
    simulate_grid_with_fill_rate,
    validate_surface_config,
)
from src.research.monthly_target_martingale_research import MonthlyMartingaleCandidate


def risk_config(max_levels: int = 2) -> dict:
    return {
        "fees": {"taker_fee": 0.0, "maker_fee": 0.0, "slippage_bps": 0},
        "grid": {
            "spacing_atr_multiplier": 1.0,
            "max_levels": max_levels,
            "base_position_size_pct": 0.10,
            "sizing_mode": "linear",
            "sizing_sequence": [1.0, 1.0],
            "allow_exponential_martingale": False,
        },
        "risk": {
            "max_grid_loss_pct": 1.0,
            "max_daily_loss_pct": 1.0,
            "max_total_exposure_pct": 1.0,
            "max_holding_hours": 0.25,
            "stop_on_regime_break": False,
            "stop_on_volatility_shock": False,
        },
    }


def make_market(rows: int = 8) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00:00Z", periods=rows, freq="5min")
    close = pd.Series(100.0, index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1.0,
            "atr_5m": 1.0,
            "breakout_risk": 0,
            "volatility_shock": 0,
        },
        index=index,
    )


def make_candidate() -> MonthlyMartingaleCandidate:
    return MonthlyMartingaleCandidate(
        name="locked",
        side_mode="mean_reversion_dual",
        entry_mode="hourly_extreme",
        entry_cooldown_hours=3.0,
        rsi_window=24,
        rsi_low=40.0,
        rsi_high=60.0,
        spacing_atr_multiplier=3.0,
        take_profit_spacing_multiplier=2.0,
        max_levels=5,
        base_position_size_pct=2.0,
        progression_multiplier=1.55,
        max_total_exposure_pct=50.0,
        fee_rate=0.0001,
        slippage_bps=0.0,
        max_grid_loss_pct=0.35,
        max_holding_hours=6.0,
        stop_on_regime_break=False,
        stop_on_volatility_shock=True,
    )


def surface_config() -> dict:
    return {
        "target": {"monthly_return": 0.20},
        "surface": {
            "fee_rates": [0.0001],
            "slippage_bps": [0.0],
            "fill_rates": [1.0],
            "seeds": 2,
        },
    }


class ExecutionSurface017Tests(unittest.TestCase):
    def test_fill_rate_one_reproduces_no_miss_grid_engine(self) -> None:
        market = make_market()
        market.iloc[1, market.columns.get_loc("high")] = 101.5
        risk = validate_strategy_config(risk_config(max_levels=1))
        expected = simulate_grid_from_index(
            market,
            0,
            risk,
            take_profit_spacing_multiplier=1.0,
            side="long",
            survival_min_realized_pnl=0.0,
        )
        actual, stats = simulate_grid_with_fill_rate(
            market,
            0,
            risk,
            "long",
            take_profit_spacing_multiplier=1.0,
            fill_rate=1.0,
            rng=np.random.default_rng(123),
            survival_min_realized_pnl=0.0,
        )
        self.assertEqual(actual.to_dict(), expected.to_dict())
        self.assertEqual(stats.tp_fill_misses, 0)
        self.assertEqual(stats.add_fill_misses, 0)

    def test_fill_rate_zero_blocks_adds_and_tp_but_not_initial_or_forced_exit(self) -> None:
        market = make_market()
        market.iloc[1, market.columns.get_loc("low")] = 99.0
        market.iloc[1, market.columns.get_loc("high")] = 101.5
        market.iloc[2, market.columns.get_loc("high")] = 101.5
        risk = validate_strategy_config(risk_config(max_levels=2))
        result, stats = simulate_grid_with_fill_rate(
            market,
            0,
            risk,
            "long",
            take_profit_spacing_multiplier=1.0,
            fill_rate=0.0,
            rng=np.random.default_rng(123),
            survival_min_realized_pnl=0.0,
        )
        self.assertEqual(stats.initial_fill_count, 1)
        self.assertGreater(stats.add_fill_misses, 0)
        self.assertGreater(stats.tp_fill_misses, 0)
        self.assertEqual(result.number_of_levels_filled, 1)
        self.assertEqual(result.exit_reason, "max_holding")
        self.assertEqual(stats.forced_exit_count, 1)

    def test_fixed_seed_is_reproducible(self) -> None:
        market = make_market(12)
        market.iloc[1:5, market.columns.get_loc("low")] = 99.0
        market.iloc[1:5, market.columns.get_loc("high")] = 101.5
        risk = validate_strategy_config(risk_config(max_levels=2))
        first = simulate_grid_with_fill_rate(
            market,
            0,
            risk,
            "long",
            1.0,
            0.5,
            np.random.default_rng(77),
        )
        second = simulate_grid_with_fill_rate(
            market,
            0,
            risk,
            "long",
            1.0,
            0.5,
            np.random.default_rng(77),
        )
        self.assertEqual(first[0].to_dict(), second[0].to_dict())
        self.assertEqual(first[1].to_dict(), second[1].to_dict())

    def test_invalid_surface_config_is_rejected(self) -> None:
        config = surface_config()
        config["surface"]["fill_rates"] = [1.2]
        with self.assertRaises(ValueError):
            validate_surface_config(config)
        config = surface_config()
        config["surface"]["fee_rates"] = [-0.1]
        with self.assertRaises(ValueError):
            make_surface_cells(config)

    def test_locked_candidate_uses_017_row_without_reselection(self) -> None:
        row = {"variant": "fundamental_trend_escape_entry_only", "fold_id": 3, **make_candidate().__dict__}
        frame = pd.DataFrame([row])
        candidate = locked_candidate_for_fold(frame, 3, ExecutionSurfaceCell(0.0004, 2.0, 0.99))
        self.assertAlmostEqual(candidate.fee_rate, 0.0004)
        self.assertAlmostEqual(candidate.slippage_bps, 2.0)
        self.assertEqual(candidate.rsi_window, make_candidate().rsi_window)
        with self.assertRaises(ValueError):
            locked_candidate_for_fold(frame, 4, ExecutionSurfaceCell(0.0004, 2.0, 0.99))

    def test_cell_aggregation_computes_probabilities_and_quantiles(self) -> None:
        config = surface_config()
        rows = pd.DataFrame(
            [
                {
                    "fee_rate": 0.0001,
                    "slippage_bps": 0.0,
                    "fill_rate": 0.99,
                    "seed": 1,
                    "aggregate_monthly_return": 0.10,
                    "aggregate_total_return": 1.0,
                    "aggregate_max_drawdown": -0.20,
                    "positive_fold_rate": 0.7,
                    "target_fold_rate": 0.2,
                    "equity_ruined": False,
                    "test_grid_count": 10,
                    "fees_paid": 1.0,
                    "slippage_paid": 0.0,
                    "add_miss_rate": 0.1,
                    "tp_miss_rate": 0.2,
                },
                {
                    "fee_rate": 0.0001,
                    "slippage_bps": 0.0,
                    "fill_rate": 0.99,
                    "seed": 2,
                    "aggregate_monthly_return": 0.30,
                    "aggregate_total_return": 2.0,
                    "aggregate_max_drawdown": -0.40,
                    "positive_fold_rate": 0.9,
                    "target_fold_rate": 0.6,
                    "equity_ruined": True,
                    "test_grid_count": 20,
                    "fees_paid": 2.0,
                    "slippage_paid": 0.1,
                    "add_miss_rate": 0.3,
                    "tp_miss_rate": 0.4,
                },
            ]
        )
        summary = aggregate_surface_by_seed(rows, config)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertAlmostEqual(float(row["monthly_return_median"]), 0.20)
        self.assertAlmostEqual(float(row["target_probability"]), 0.5)
        self.assertAlmostEqual(float(row["positive_probability"]), 1.0)
        self.assertAlmostEqual(float(row["ruin_probability"]), 0.5)


if __name__ == "__main__":
    unittest.main()
