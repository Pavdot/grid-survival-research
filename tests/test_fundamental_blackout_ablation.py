from __future__ import annotations

import unittest

import pandas as pd

from src.labeling.grid_risk import validate_strategy_config
from src.research.fundamental_blackout_ablation_research import (
    build_regime_danger_mask,
    decide_ablation,
    trade_attribution_by_fold,
)
from src.research.fundamental_blackout_martingale_research import select_best_exact_no_drawdown
from src.research.monthly_target_martingale_research import MonthlyMartingaleCandidate, run_signal_grid_backtest


def risk_config() -> dict:
    return {
        "fees": {"taker_fee": 0.0, "maker_fee": 0.0, "slippage_bps": 0},
        "grid": {
            "spacing_atr_multiplier": 5.0,
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


def make_test_candidate() -> MonthlyMartingaleCandidate:
    return MonthlyMartingaleCandidate(
        name="test",
        side_mode="mean_reversion_dual",
        entry_mode="hourly_extreme",
        entry_cooldown_hours=0.0,
        rsi_window=3,
        rsi_low=40.0,
        rsi_high=60.0,
        spacing_atr_multiplier=5.0,
        take_profit_spacing_multiplier=1.0,
        max_levels=1,
        base_position_size_pct=0.10,
        progression_multiplier=1.2,
        max_total_exposure_pct=0.50,
        fee_rate=0.0,
        slippage_bps=0.0,
        max_grid_loss_pct=1.0,
        max_holding_hours=1.0,
        stop_on_regime_break=False,
        stop_on_volatility_shock=False,
    )


def flat_market(periods: int = 40) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00:00Z", periods=periods, freq="5min")
    close = pd.Series(100.0, index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 1.0,
            "atr_5m": 10.0,
            "breakout_risk": 0,
            "volatility_shock": 0,
            "realized_volatility_ratio": 0.0,
            "range_expansion_ratio": 0.0,
        },
        index=index,
    )


class FundamentalBlackoutAblationTests(unittest.TestCase):
    def test_entry_only_blocks_new_grid_without_closing_open_grid(self) -> None:
        market = flat_market()
        risk = validate_strategy_config(risk_config())
        side_signal = pd.Series("long", index=market.index, dtype="object")
        entry_blackout = pd.Series(False, index=market.index)
        exit_blackout = pd.Series(False, index=market.index)
        entry_blackout.iloc[2:8] = True
        result = run_signal_grid_backtest(
            market,
            risk,
            side_signal,
            make_test_candidate(),
            entry_blackout_series=entry_blackout,
            exit_blackout_series=exit_blackout,
        )
        self.assertFalse(result.trades.empty)
        self.assertNotEqual(result.trades.iloc[0]["exit_reason"], "fundamental_blackout")

    def test_close_on_blackout_closes_open_grid(self) -> None:
        market = flat_market()
        risk = validate_strategy_config(risk_config())
        side_signal = pd.Series("long", index=market.index, dtype="object")
        blackout = pd.Series(False, index=market.index)
        blackout.iloc[2:8] = True
        result = run_signal_grid_backtest(
            market,
            risk,
            side_signal,
            make_test_candidate(),
            entry_blackout_series=blackout,
            exit_blackout_series=blackout,
        )
        self.assertFalse(result.trades.empty)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "fundamental_blackout")

    def test_regime_entry_mask_uses_delayed_closed_signals(self) -> None:
        market = flat_market(periods=20)
        market.loc[market.index[5], "volatility_shock"] = 1
        mask = build_regime_danger_mask(market, {"regime_danger": {"lookback_bars": 3}})
        self.assertFalse(bool(mask.iloc[5]))
        self.assertTrue(bool(mask.iloc[6]))
        self.assertTrue(bool(mask.iloc[8]))
        self.assertFalse(bool(mask.iloc[9]))

    def test_ablation_selection_refuses_test_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "split": "test",
                    "name": "bad",
                    "number_of_grids": 10,
                    "monthly_return": 1.0,
                    "max_drawdown": -0.99,
                    "profit_factor": 2.0,
                    "number_of_forced_exits": 0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            select_best_exact_no_drawdown(frame, {"target": {"monthly_return": 0.20, "min_exact_grids": 1}})

    def test_trade_attribution_classifies_trade_sets(self) -> None:
        baseline = pd.DataFrame(
            [
                {"start_timestamp": "2024-01-01 00:00:00+00:00", "side": "long", "realized_pnl": 1.0},
                {"start_timestamp": "2024-01-01 01:00:00+00:00", "side": "long", "realized_pnl": 2.0},
            ]
        )
        variant = pd.DataFrame(
            [
                {"start_timestamp": "2024-01-01 01:00:00+00:00", "side": "long", "realized_pnl": 3.0},
                {"start_timestamp": "2024-01-01 02:00:00+00:00", "side": "short", "realized_pnl": 4.0},
            ]
        )
        row = trade_attribution_by_fold(baseline, variant, "variant", 1, same_selected_candidate=True)
        self.assertEqual(row["baseline_only_count"], 1)
        self.assertEqual(row["variant_only_count"], 1)
        self.assertEqual(row["common_count"], 1)
        self.assertEqual(row["baseline_only_pnl"], 1.0)
        self.assertEqual(row["variant_only_pnl"], 4.0)
        self.assertEqual(row["common_pnl_delta"], 1.0)
        self.assertEqual(row["effect_scope"], "policy_only")

    def test_decision_uses_best_realistic_improvement(self) -> None:
        comparison = pd.DataFrame(
            [
                {"variant": "baseline", "aggregate_monthly_return": -0.13},
                {"variant": "realistic_entry_only", "aggregate_monthly_return": -0.09},
                {"variant": "realistic_close_on_blackout", "aggregate_monthly_return": -0.10},
                {"variant": "realistic_regime_entry_only", "aggregate_monthly_return": -0.11},
                {"variant": "oracle_entry_only", "aggregate_monthly_return": -0.09},
            ]
        )
        self.assertEqual(decide_ablation(comparison), "entry-only helps")


if __name__ == "__main__":
    unittest.main()
