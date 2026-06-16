from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.labeling.grid_engine import simulate_grid_from_index
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.range_break_labels import build_range_break_labels
from src.research.monthly_target_martingale_research import MonthlyMartingaleCandidate, run_signal_grid_backtest
from src.research.range_break_classifier_martingale_research import split_internal_train, threshold_grid
from src.research.walk_forward_martingale_research import WalkForwardWindow


def risk_config() -> dict:
    return {
        "fees": {"taker_fee": 0.0, "maker_fee": 0.0, "slippage_bps": 0},
        "grid": {
            "spacing_atr_multiplier": 1.0,
            "max_levels": 3,
            "base_position_size_pct": 0.10,
            "sizing_mode": "linear",
            "sizing_sequence": [1.0, 1.0, 1.0],
            "allow_exponential_martingale": False,
        },
        "risk": {
            "max_grid_loss_pct": 1.0,
            "max_daily_loss_pct": 1.0,
            "max_total_exposure_pct": 1.0,
            "max_holding_hours": 1,
            "stop_on_regime_break": False,
            "stop_on_volatility_shock": False,
        },
    }


def candidate() -> MonthlyMartingaleCandidate:
    return MonthlyMartingaleCandidate(
        name="test",
        side_mode="mean_reversion_dual",
        entry_mode="hourly_extreme",
        entry_cooldown_hours=0.0,
        rsi_window=3,
        rsi_low=40.0,
        rsi_high=60.0,
        spacing_atr_multiplier=1.0,
        take_profit_spacing_multiplier=4.0,
        max_levels=3,
        base_position_size_pct=0.10,
        progression_multiplier=1.2,
        max_total_exposure_pct=1.0,
        fee_rate=0.0,
        slippage_bps=0.0,
        max_grid_loss_pct=1.0,
        max_holding_hours=1.0,
        stop_on_regime_break=False,
        stop_on_volatility_shock=False,
    )


def market_for_break(direction: str) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00:00Z", periods=80, freq="5min")
    close = pd.Series(100.0, index=index)
    high = pd.Series(100.2, index=index)
    low = pd.Series(99.8, index=index)
    if direction == "up":
        close.iloc[30:36] = [101.0, 101.4, 101.8, 102.0, 102.2, 102.4]
        high.iloc[30:36] = close.iloc[30:36] + 0.2
        low.iloc[30:36] = close.iloc[30:36] - 0.2
        trend = 1.0
    else:
        close.iloc[30:36] = [99.0, 98.6, 98.2, 98.0, 97.8, 97.6]
        high.iloc[30:36] = close.iloc[30:36] + 0.2
        low.iloc[30:36] = close.iloc[30:36] - 0.2
        trend = -1.0
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
            "atr_5m": 0.2,
            "trend_alignment_score": trend,
            "range_expansion_ratio": 2.0,
            "realized_volatility_ratio": 2.0,
        },
        index=index,
    )


def flat_market(periods: int = 24) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00:00Z", periods=periods, freq="5min")
    close = pd.Series(100.0, index=index)
    low = close.copy()
    low.iloc[2:6] = 99.0
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": low,
            "close": close,
            "volume": 1.0,
            "atr_5m": 0.5,
            "breakout_risk": 0,
            "volatility_shock": 0,
        },
        index=index,
    )


class RangeBreakClassifierMartingaleTests(unittest.TestCase):
    def test_up_breakout_is_danger_for_short_not_long(self) -> None:
        labels = build_range_break_labels(
            market_for_break("up"),
            {
                "range_break_label": {
                    "primary_horizon_hours": 1,
                    "diagnostic_horizons_hours": [1],
                    "range_lookback_bars": 12,
                    "breakout_atr_buffer": 0.25,
                    "extension_atr": 0.25,
                    "min_persistent_closes": 2,
                }
            },
        )
        self.assertTrue(np.isnan(labels["range_break_danger_short_1h"].iloc[0]))
        self.assertEqual(labels["range_break_danger_short_1h"].iloc[24], 1)
        self.assertEqual(labels["range_break_danger_long_1h"].iloc[24], 0)

    def test_down_breakout_is_danger_for_long_not_short(self) -> None:
        labels = build_range_break_labels(
            market_for_break("down"),
            {
                "range_break_label": {
                    "primary_horizon_hours": 1,
                    "diagnostic_horizons_hours": [1],
                    "range_lookback_bars": 12,
                    "breakout_atr_buffer": 0.25,
                    "extension_atr": 0.25,
                    "min_persistent_closes": 2,
                }
            },
        )
        self.assertEqual(labels["range_break_danger_long_1h"].iloc[24], 1)
        self.assertEqual(labels["range_break_danger_short_1h"].iloc[24], 0)

    def test_labels_refuse_naive_timestamps(self) -> None:
        market = market_for_break("up").copy()
        market.index = market.index.tz_convert(None)
        with self.assertRaises(ValueError):
            build_range_break_labels(market, {"range_break_label": {"range_lookback_bars": 12}})

    def test_internal_split_is_chronological_and_before_test(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00Z", periods=220 * 24, freq="h")
        fold = WalkForwardWindow(fold_id=1, train=index[: 180 * 24], test=index[180 * 24 : 210 * 24])
        split = split_internal_train(
            fold,
            {"range_break_classifier": {"internal_split": {"model_train_days": 120, "selection_days": 60}}},
        )
        self.assertLess(split.model_train.max(), split.selection.min())
        self.assertLess(split.selection.max(), fold.test.min())

    def test_threshold_grid_refuses_incoherent_config(self) -> None:
        with self.assertRaises(ValueError):
            threshold_grid({"range_break_classifier": {"threshold_search": {"start": 0.95, "stop": 0.50, "step": 0.05}}})

    def test_side_specific_entry_mask_blocks_only_matching_side(self) -> None:
        market = flat_market()
        side_signal = pd.Series("long", index=market.index, dtype="object")
        entry_masks = {
            "long": pd.Series(True, index=market.index),
            "short": pd.Series(False, index=market.index),
        }
        result = run_signal_grid_backtest(
            market,
            validate_strategy_config(risk_config()),
            side_signal,
            candidate(),
            entry_blackout_series=entry_masks,
        )
        self.assertTrue(result.trades.empty)

    def test_add_block_prevents_new_level_without_forced_exit(self) -> None:
        market = flat_market()
        add_block = pd.Series(False, index=market.index)
        add_block.iloc[2:8] = True
        result = simulate_grid_from_index(
            market,
            0,
            validate_strategy_config(risk_config()),
            side="long",
            take_profit_spacing_multiplier=4.0,
            add_block_series=add_block,
        )
        self.assertEqual(result.number_of_levels_filled, 1)
        self.assertNotEqual(result.exit_reason, "range_break_emergency")

    def test_emergency_exit_uses_dedicated_reason(self) -> None:
        market = flat_market()
        emergency = pd.Series(False, index=market.index)
        emergency.iloc[2:5] = True
        result = simulate_grid_from_index(
            market,
            0,
            validate_strategy_config(risk_config()),
            side="long",
            take_profit_spacing_multiplier=4.0,
            emergency_exit_series=emergency,
            emergency_exit_reason="range_break_emergency",
        )
        self.assertEqual(result.exit_reason, "range_break_emergency")
        self.assertEqual(result.stopped_by_kill_switch, 1)


if __name__ == "__main__":
    unittest.main()
