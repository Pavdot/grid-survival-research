from __future__ import annotations

import unittest

import pandas as pd

from src.research.momentum_switch_research import (
    MomentumCandidate,
    backtest_signal,
    build_signal,
    make_candidates,
    select_best,
)


def momentum_config() -> dict:
    return {
        "execution": {
            "max_position_pct": 0.5,
            "fee_rate": 0.0004,
            "slippage_bps": 2,
            "max_total_loss_pct": 0.2,
            "max_drawdown_constraint": -0.2,
        },
        "search": {
            "rsi_windows": [14],
            "rsi_threshold_pairs": [[35, 65]],
            "ema_fast_windows": [10],
            "ema_slow_windows": [100],
            "include_long_short": True,
            "include_long_only": True,
        },
    }


class MomentumSwitchResearchTests(unittest.TestCase):
    def test_candidate_generation_includes_rsi_and_ema(self) -> None:
        candidates = make_candidates(momentum_config())
        self.assertTrue(any(candidate.signal_type == "rsi_momentum" for candidate in candidates))
        self.assertTrue(any(candidate.signal_type == "ema_long_short" for candidate in candidates))

    def test_rsi_signal_uses_bounded_positions(self) -> None:
        index = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
        frame = pd.DataFrame({"close": list(range(20)) + list(range(20, 0, -1))}, index=index)
        candidate = MomentumCandidate("rsi_momentum_14_35_65", "rsi_momentum", {"window": 14, "low": 35, "high": 65})
        signal = build_signal(frame, candidate)
        self.assertLessEqual(signal.abs().max(), 1.0)

    def test_backtest_applies_costs_on_position_changes(self) -> None:
        index = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
        base = pd.DataFrame({"close": [100, 101, 102, 103]}, index=index)
        signal = pd.Series([0, 1, 1, 0], index=index)
        _, trades = backtest_signal(base, signal, momentum_config())
        self.assertGreater(trades["cost"].sum(), 0)
        self.assertLessEqual(trades["position"].abs().max(), 0.5)

    def test_backtest_has_total_loss_kill_switch(self) -> None:
        index = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
        base = pd.DataFrame({"close": [100, 50, 50, 50, 50]}, index=index)
        signal = pd.Series([1, 1, 1, 1, 1], index=index)
        _, trades = backtest_signal(base, signal, momentum_config())
        self.assertEqual(trades["risk_killed"].max(), 1)

    def test_selection_refuses_test_rows(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "name": "x",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "test",
                    "total_return": 1.0,
                    "max_drawdown": 0.0,
                    "trades": 1,
                }
            ]
        )
        with self.assertRaises(ValueError):
            select_best(summary, momentum_config())


if __name__ == "__main__":
    unittest.main()
