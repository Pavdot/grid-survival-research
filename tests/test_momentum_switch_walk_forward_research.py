from __future__ import annotations

import unittest

import pandas as pd

from src.research.momentum_switch_research import MomentumCandidate, make_candidates
from src.research.momentum_switch_walk_forward_research import (
    _validation_subwindows,
    choose_candidate_subset,
    same_candidate,
    select_best_validation,
    stitch_oos_equity,
)


def wf_momentum_config() -> dict:
    return {
        "target": {
            "monthly_return": 0.05,
            "stretch_monthly_return": 0.20,
            "max_drawdown": -0.20,
            "min_positive_fold_rate": 0.60,
            "min_target_fold_rate": 0.40,
            "min_trades": 1,
        },
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
            "donchian_windows": [24],
            "rsi_ema_windows": [14],
            "rsi_ema_threshold_pairs": [[35, 65]],
            "rsi_ema_fast_windows": [10],
            "rsi_ema_slow_windows": [100],
            "max_position_pcts": [0.25, 0.50],
            "include_long_short": True,
            "include_long_only": True,
        },
    }


class MomentumSwitchWalkForwardResearchTests(unittest.TestCase):
    def test_extended_candidate_generation_includes_position_variants(self) -> None:
        candidates = make_candidates(wf_momentum_config())
        names = {candidate.name for candidate in candidates}
        self.assertTrue(any(name.startswith("donchian_long_short") for name in names))
        self.assertTrue(any(name.startswith("rsi_ema_momentum") for name in names))
        self.assertTrue(all("max_position_pct" in candidate.params for candidate in candidates))

    def test_candidate_subset_is_reproducible(self) -> None:
        candidates = [MomentumCandidate(f"c{i}", "ema_long_only", {"fast": 1, "slow": 2}) for i in range(10)]
        left = choose_candidate_subset(candidates, 4, 42)
        right = choose_candidate_subset(candidates, 4, 42)
        self.assertEqual([candidate.name for candidate in left], [candidate.name for candidate in right])

    def test_selection_uses_validation_only(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "name": "bad",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "test",
                    "monthly_return": 1.0,
                    "total_return": 1.0,
                    "max_drawdown": 0.0,
                    "trades": 2,
                    "risk_kill_triggered": 0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            select_best_validation(frame, wf_momentum_config())

    def test_selection_prefers_monthly_return_under_constraints(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "name": "a",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "validation",
                    "monthly_return": 0.05,
                    "total_return": 0.04,
                    "max_drawdown": -0.10,
                    "trades": 2,
                    "risk_kill_triggered": 0,
                },
                {
                    "name": "b",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "validation",
                    "monthly_return": 0.10,
                    "total_return": 0.03,
                    "max_drawdown": -0.12,
                    "trades": 2,
                    "risk_kill_triggered": 0,
                },
            ]
        )
        selected = select_best_validation(frame, wf_momentum_config())
        self.assertEqual(selected["name"], "b")

    def test_risk_adjusted_selection_penalizes_drawdown(self) -> None:
        config = wf_momentum_config()
        config["selection"] = {"primary_metric": "risk_adjusted_monthly", "drawdown_penalty": 0.75}
        frame = pd.DataFrame(
            [
                {
                    "name": "high_return_deep_dd",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "validation",
                    "monthly_return": 0.12,
                    "total_return": 0.10,
                    "max_drawdown": -0.14,
                    "trades": 2,
                    "risk_kill_triggered": 0,
                },
                {
                    "name": "lower_return_stable",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "validation",
                    "monthly_return": 0.06,
                    "total_return": 0.05,
                    "max_drawdown": -0.02,
                    "trades": 2,
                    "risk_kill_triggered": 0,
                },
            ]
        )
        selected = select_best_validation(frame, config)
        self.assertEqual(selected["name"], "lower_return_stable")

    def test_stability_selection_prefers_consistent_validation(self) -> None:
        config = wf_momentum_config()
        config["selection"] = {
            "primary_metric": "stability_adjusted_monthly",
            "stability_windows": 3,
            "min_stability_positive_rate": 0.67,
            "drawdown_penalty": 0.5,
        }
        frame = pd.DataFrame(
            [
                {
                    "name": "spiky",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "validation",
                    "monthly_return": 0.12,
                    "total_return": 0.10,
                    "max_drawdown": -0.08,
                    "trades": 2,
                    "risk_kill_triggered": 0,
                    "stability_median_monthly_return": -0.01,
                    "stability_min_monthly_return": -0.04,
                    "stability_positive_rate": 0.33,
                    "stability_worst_drawdown": -0.08,
                },
                {
                    "name": "steady",
                    "signal_type": "rsi_momentum",
                    "params": "{}",
                    "split": "validation",
                    "monthly_return": 0.04,
                    "total_return": 0.03,
                    "max_drawdown": -0.03,
                    "trades": 2,
                    "risk_kill_triggered": 0,
                    "stability_median_monthly_return": 0.03,
                    "stability_min_monthly_return": 0.01,
                    "stability_positive_rate": 1.0,
                    "stability_worst_drawdown": -0.03,
                },
            ]
        )
        selected = select_best_validation(frame, config)
        self.assertEqual(selected["name"], "steady")

    def test_validation_subwindows_are_ordered_and_complete(self) -> None:
        index = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        windows = _validation_subwindows(index, 3)
        self.assertEqual(sum(len(window) for window in windows), len(index))
        self.assertLess(windows[0].max(), windows[1].min())
        self.assertLess(windows[1].max(), windows[2].min())

    def test_same_candidate_compares_name_type_and_params(self) -> None:
        left = {"name": "a", "signal_type": "rsi_momentum", "params": '{"window": 14}'}
        right = {"name": "a", "signal_type": "rsi_momentum", "params": '{"window": 14}'}
        changed = {"name": "a", "signal_type": "rsi_momentum", "params": '{"window": 24}'}
        self.assertTrue(same_candidate(left, right))
        self.assertFalse(same_candidate(left, changed))

    def test_stitched_equity_normalizes_each_fold(self) -> None:
        index_1 = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        index_2 = pd.date_range("2024-02-01", periods=2, freq="D", tz="UTC")
        stitched = stitch_oos_equity(
            [
                (1, pd.Series([1.01, 1.111], index=index_1)),
                (2, pd.Series([0.99, 1.089], index=index_2)),
            ]
        )
        self.assertAlmostEqual(float(stitched["equity"].iloc[-1]), 1.21)


if __name__ == "__main__":
    unittest.main()
