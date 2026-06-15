from __future__ import annotations

import unittest

import pandas as pd

from src.fundamentals.event_blackout import build_blackout_windows, blackout_mask, normalize_fundamental_events
from src.labeling.grid_risk import validate_strategy_config
from src.research.fundamental_blackout_martingale_research import select_best_exact_no_drawdown
from src.research.monthly_target_martingale_research import MonthlyMartingaleCandidate, run_signal_grid_backtest


def event_frame(
    event_time: str,
    known_time: str | None = None,
    is_scheduled: bool = True,
    is_surprise: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_time_utc": event_time,
                "known_time_utc": known_time or event_time,
                "category": "macro_fomc",
                "severity": 5,
                "source": "test",
                "title": "test event",
                "is_scheduled": is_scheduled,
                "is_surprise": is_surprise,
            }
        ]
    )


def blackout_config() -> dict:
    return {
        "pre_event_hours": 2,
        "post_event_hours": 1,
        "surprise_reaction_hours": 4,
        "oracle_pre_event_hours": 3,
        "categories": ["macro_fomc"],
        "min_severity": 1,
    }


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
        },
        index=index,
    )


class FundamentalBlackoutTests(unittest.TestCase):
    def test_normalization_rejects_naive_timestamps(self) -> None:
        with self.assertRaises(ValueError):
            normalize_fundamental_events(event_frame("2024-01-01 10:00:00"))

    def test_scheduled_blackout_cuts_before_and_after_event(self) -> None:
        events = normalize_fundamental_events(event_frame("2024-01-01T10:00:00Z"))
        windows = build_blackout_windows(events, blackout_config(), "realistic")
        index = pd.date_range("2024-01-01T07:00:00Z", periods=6, freq="h")
        mask = blackout_mask(index, windows)
        self.assertFalse(bool(mask.loc[pd.Timestamp("2024-01-01T07:00:00Z")]))
        self.assertTrue(bool(mask.loc[pd.Timestamp("2024-01-01T08:00:00Z")]))
        self.assertTrue(bool(mask.loc[pd.Timestamp("2024-01-01T11:00:00Z")]))
        self.assertFalse(bool(mask.loc[pd.Timestamp("2024-01-01T12:00:00Z")]))

    def test_realistic_surprise_does_not_cut_before_known_time(self) -> None:
        events = normalize_fundamental_events(
            event_frame("2024-01-01T10:00:00Z", "2024-01-01T12:00:00Z", False, True)
        )
        windows = build_blackout_windows(events, blackout_config(), "realistic")
        index = pd.date_range("2024-01-01T09:00:00Z", periods=5, freq="h")
        mask = blackout_mask(index, windows)
        self.assertFalse(bool(mask.loc[pd.Timestamp("2024-01-01T11:00:00Z")]))
        self.assertTrue(bool(mask.loc[pd.Timestamp("2024-01-01T12:00:00Z")]))

    def test_oracle_surprise_cuts_before_event_time(self) -> None:
        events = normalize_fundamental_events(
            event_frame("2024-01-01T10:00:00Z", "2024-01-01T12:00:00Z", False, True)
        )
        windows = build_blackout_windows(events, blackout_config(), "oracle")
        index = pd.date_range("2024-01-01T06:00:00Z", periods=6, freq="h")
        mask = blackout_mask(index, windows)
        self.assertFalse(bool(mask.loc[pd.Timestamp("2024-01-01T06:00:00Z")]))
        self.assertTrue(bool(mask.loc[pd.Timestamp("2024-01-01T07:00:00Z")]))
        self.assertTrue(bool(mask.loc[pd.Timestamp("2024-01-01T10:00:00Z")]))

    def test_grid_is_closed_at_blackout_timestamp(self) -> None:
        market = flat_market()
        risk = validate_strategy_config(risk_config())
        side_signal = pd.Series("long", index=market.index, dtype="object")
        blackout = pd.Series(False, index=market.index)
        blackout.iloc[2:5] = True
        result = run_signal_grid_backtest(market, risk, side_signal, make_test_candidate(), blackout_series=blackout)
        self.assertFalse(result.trades.empty)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "fundamental_blackout")

    def test_grid_opening_is_blocked_during_blackout(self) -> None:
        market = flat_market()
        risk = validate_strategy_config(risk_config())
        side_signal = pd.Series("long", index=market.index, dtype="object")
        blackout = pd.Series(False, index=market.index)
        blackout.iloc[:5] = True
        result = run_signal_grid_backtest(market, risk, side_signal, make_test_candidate(), blackout_series=blackout)
        self.assertFalse(result.trades.empty)
        self.assertGreaterEqual(result.trades.iloc[0]["start_timestamp"], market.index[5])

    def test_blackout_selection_refuses_test_rows(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
