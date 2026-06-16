from __future__ import annotations

import unittest

import pandas as pd

from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.fundamental_trend_escape_martingale_research import (
    build_variant_masks,
    decide_trend_escape,
    validate_variants,
)
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


def market_with_breakout() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00:00Z", periods=12, freq="5min")
    close = pd.Series([100, 100, 100, 100, 100, 100, 103, 104, 104, 104, 104, 104], index=index, dtype=float)
    high = close + 0.1
    low = close - 0.1
    high.iloc[:6] = 101.0
    low.iloc[:6] = 99.0
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
            "atr_5m": 1.0,
            "trend_alignment_score": 1.0,
            "range_expansion_ratio": 1.0,
            "realized_volatility_ratio": 1.0,
        },
        index=index,
    )


def flat_market(periods: int = 20) -> pd.DataFrame:
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


class FundamentalTrendEscapeMartingaleTests(unittest.TestCase):
    def test_trend_escape_mask_is_delayed_after_closed_breakout(self) -> None:
        market = market_with_breakout()
        components = build_trend_escape_components(
            market,
            {
                "trend_escape": {
                    "range_lookback_bars": 4,
                    "breakout_atr_buffer": 0.25,
                    "confirmation_bars": 1,
                    "min_confirmations": 1,
                    "delay_bars": 1,
                    "propagation_bars": 2,
                    "min_abs_trend_alignment": 0.5,
                }
            },
        )
        self.assertEqual(int(components["trend_escape_raw"].iloc[6]), 1)
        self.assertEqual(int(components["trend_escape"].iloc[6]), 0)
        self.assertEqual(int(components["trend_escape"].iloc[7]), 1)

    def test_fundamental_trend_variant_requires_event_and_trend(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00Z", periods=5, freq="5min")
        trend = pd.Series([False, False, True, True, False], index=index)
        realistic = pd.Series([False, False, False, True, True], index=index)
        oracle = pd.Series([False, True, True, True, False], index=index)
        entry, exit_mask, reason = build_variant_masks(
            "fundamental_trend_escape_close",
            trend,
            {"realistic": realistic, "oracle": oracle},
        )
        self.assertEqual(reason, "fundamental_trend_escape")
        self.assertTrue(bool(entry.iloc[3]))
        self.assertFalse(bool(entry.iloc[2]))
        self.assertTrue(bool(exit_mask.iloc[3]))

    def test_trend_escape_close_uses_custom_exit_reason(self) -> None:
        market = flat_market()
        risk = validate_strategy_config(risk_config())
        side_signal = pd.Series("long", index=market.index, dtype="object")
        exit_mask = pd.Series(False, index=market.index)
        exit_mask.iloc[2:5] = True
        result = run_signal_grid_backtest(
            market,
            risk,
            side_signal,
            make_test_candidate(),
            exit_blackout_series=exit_mask,
            exit_blackout_reason="trend_escape",
        )
        self.assertFalse(result.trades.empty)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "trend_escape")

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_variants({"trend_escape_ablation": {"variants": ["baseline", "unknown"]}})

    def test_decision_identifies_realistic_trend_improvement(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "variant": "baseline",
                    "aggregate_monthly_return": 0.05,
                    "positive_fold_rate": 0.5,
                    "target_fold_rate": 0.2,
                    "worst_fold_total_return": -0.5,
                },
                {
                    "variant": "fundamental_trend_escape_close",
                    "aggregate_monthly_return": 0.08,
                    "positive_fold_rate": 0.5,
                    "target_fold_rate": 0.2,
                    "worst_fold_total_return": -0.4,
                },
            ]
        )
        decision = decide_trend_escape(
            comparison,
            {"target": {"monthly_return": 0.20, "min_positive_fold_rate": 0.60, "min_target_fold_rate": 0.60}},
        )
        self.assertEqual(decision, "fundamental trend filter helps but below target")


if __name__ == "__main__":
    unittest.main()
