from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from src.fundamentals.event_blackout import build_blackout_windows
from src.labeling.grid_risk import validate_strategy_config
from src.research.execution_worst_case_audit import (
    AuditScenario,
    _strict_after_mask,
    future_invariance_check,
    simulate_grid_from_signal_pos,
    transform_candidate,
)
from src.research.monthly_target_martingale_research import MonthlyMartingaleCandidate, planned_exposure


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
            "trend_alignment_score": 1.0,
            "range_expansion_ratio": 1.0,
            "realized_volatility_ratio": 1.0,
        },
        index=index,
    )


def make_candidate() -> MonthlyMartingaleCandidate:
    return MonthlyMartingaleCandidate(
        name="candidate",
        side_mode="mean_reversion_dual",
        entry_mode="hourly_extreme",
        entry_cooldown_hours=0.0,
        rsi_window=3,
        rsi_low=40.0,
        rsi_high=60.0,
        spacing_atr_multiplier=1.0,
        take_profit_spacing_multiplier=1.0,
        max_levels=2,
        base_position_size_pct=2.5,
        progression_multiplier=1.55,
        max_total_exposure_pct=50.0,
        fee_rate=0.0,
        slippage_bps=0.0,
        max_grid_loss_pct=1.0,
        max_holding_hours=0.25,
        stop_on_regime_break=False,
        stop_on_volatility_shock=False,
    )


class ExecutionWorstCaseAuditTests(unittest.TestCase):
    def test_next_bar_entry_executes_after_signal_timestamp(self) -> None:
        market = make_market()
        scenario = AuditScenario(
            name="next",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            entry_execution_mode="next_bar_open",
        )
        result, extras = simulate_grid_from_signal_pos(
            market,
            0,
            validate_strategy_config(risk_config(max_levels=1)),
            "long",
            scenario,
            take_profit_spacing_multiplier=1.0,
        )
        self.assertEqual(extras["signal_timestamp"], market.index[0])
        self.assertEqual(extras["entry_timestamp"], market.index[1])
        self.assertGreater(result.start_timestamp, extras["signal_timestamp"])

    def test_conservative_intrabar_does_not_credit_tp_after_same_bar_add(self) -> None:
        market = make_market()
        market.iloc[1, market.columns.get_loc("low")] = 99.0
        market.iloc[1, market.columns.get_loc("high")] = 102.0
        market.iloc[2:, market.columns.get_loc("high")] = 100.1
        scenario = AuditScenario(
            name="conservative",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            conservative_intrabar=True,
        )
        result, extras = simulate_grid_from_signal_pos(
            market,
            0,
            validate_strategy_config(risk_config(max_levels=2)),
            "long",
            scenario,
            take_profit_spacing_multiplier=1.0,
        )
        self.assertGreater(extras["intrabar_ambiguous_tp_after_add"], 0)
        self.assertNotEqual(result.exit_reason, "take_profit")

    def test_sizing_percent_unit_correction_scales_candidate_exposure(self) -> None:
        candidate = make_candidate()
        corrected = transform_candidate(
            candidate,
            AuditScenario(
                name="sizing",
                variant="fundamental_trend_escape_entry_only",
                selection_policy="locked_017",
                sizing_scale=0.01,
            ),
        )
        self.assertAlmostEqual(candidate.base_position_size_pct, 2.5)
        self.assertAlmostEqual(corrected.base_position_size_pct, 0.025)
        self.assertAlmostEqual(planned_exposure(corrected), planned_exposure(candidate) / 100.0)

    def test_higher_fees_and_slippage_reduce_realized_pnl(self) -> None:
        market = make_market()
        market.iloc[1, market.columns.get_loc("high")] = 101.5
        scenario = AuditScenario("cost", "baseline", "locked_017")
        base_risk = validate_strategy_config(risk_config(max_levels=1))
        clean, _ = simulate_grid_from_signal_pos(market, 0, base_risk, "long", scenario, 1.0)
        costly_risk = replace(base_risk, taker_fee=0.001, maker_fee=0.001, slippage_bps=10.0)
        costly, _ = simulate_grid_from_signal_pos(market, 0, costly_risk, "long", scenario, 1.0)
        self.assertLess(costly.realized_pnl, clean.realized_pnl)
        self.assertGreater(costly.fees_paid + costly.slippage_paid, clean.fees_paid + clean.slippage_paid)

    def test_future_invariance_checks_pass_on_synthetic_closed_features(self) -> None:
        market = make_market(240)
        market["close"] = pd.Series(range(100, 340), index=market.index, dtype=float)
        market["open"] = market["close"].shift(1).fillna(market["close"].iloc[0])
        market["high"] = market["close"] + 1.0
        market["low"] = market["close"] - 1.0
        signal_frame = market.resample("1h", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        config = {
            "trend_escape": {
                "range_lookback_bars": 12,
                "breakout_atr_buffer": 0.25,
                "confirmation_bars": 1,
                "min_confirmations": 1,
                "delay_bars": 1,
                "propagation_bars": 2,
                "min_abs_trend_alignment": 0.5,
            }
        }
        rows = future_invariance_check(market, signal_frame, config, make_candidate())
        self.assertTrue(all(row["passed"] for row in rows))

    def test_strict_surprise_blackout_starts_after_known_time(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00Z", periods=12, freq="5min")
        events = pd.DataFrame(
            [
                {
                    "event_time_utc": pd.Timestamp("2024-01-01 00:10:00Z"),
                    "known_time_utc": pd.Timestamp("2024-01-01 00:10:00Z"),
                    "category": "exchange_hack",
                    "severity": 5,
                    "source": "test",
                    "title": "surprise",
                    "is_scheduled": False,
                    "is_surprise": True,
                }
            ]
        )
        windows = build_blackout_windows(
            events,
            {"categories": ["exchange_hack"], "min_severity": 1, "surprise_reaction_hours": 1},
            "realistic",
        )
        mask = _strict_after_mask(index, windows)
        self.assertFalse(bool(mask.loc[pd.Timestamp("2024-01-01 00:10:00Z")]))
        self.assertTrue(bool(mask.loc[pd.Timestamp("2024-01-01 00:15:00Z")]))


if __name__ == "__main__":
    unittest.main()
