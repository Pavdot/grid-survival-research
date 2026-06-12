from __future__ import annotations

import unittest

import pandas as pd

from src.research.walk_forward_martingale_research import (
    CANDIDATE_COLUMNS,
    decide_walk_forward,
    make_walk_forward_windows,
    same_candidate,
    stitch_oos_equity,
    summarize_walk_forward,
)


def wf_config() -> dict:
    return {
        "target": {
            "monthly_return": 0.20,
            "max_drawdown": -0.35,
            "min_positive_fold_rate": 0.60,
            "min_target_fold_rate": 0.60,
        }
    }


class WalkForwardMartingaleResearchTests(unittest.TestCase):
    def test_windows_are_chronological_with_embargo(self) -> None:
        index = pd.date_range("2024-01-01", periods=24 * 60, freq="h", tz="UTC")
        folds = make_walk_forward_windows(index, train_days=10, test_days=5, step_days=5, embargo_bars=2)
        self.assertTrue(folds)
        for fold in folds:
            self.assertLess(fold.train.max(), fold.test.min())
            self.assertGreaterEqual(index.get_loc(fold.test.min()) - index.get_loc(fold.train.max()), 3)
            self.assertTrue(set(fold.train).isdisjoint(set(fold.test)))

    def test_windows_refuse_empty_configuration(self) -> None:
        index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        with self.assertRaises(ValueError):
            make_walk_forward_windows(index, train_days=10, test_days=10, step_days=5, embargo_bars=0)

    def test_same_candidate_requires_test_candidate_to_match_selection(self) -> None:
        selected = {column: column for column in CANDIDATE_COLUMNS}
        tested = selected.copy()
        self.assertTrue(same_candidate(selected, tested))
        tested["fee_rate"] = "changed"
        self.assertFalse(same_candidate(selected, tested))

    def test_decision_rejects_target_return_with_bad_drawdown(self) -> None:
        summary = {
            "equity_ruined": False,
            "aggregate_max_drawdown": -0.50,
            "max_drawdown_constraint": -0.35,
            "aggregate_monthly_return": 0.30,
            "target_monthly_return": 0.20,
            "positive_fold_rate": 1.0,
            "target_fold_rate": 1.0,
            "validation_target_rate": 1.0,
            "min_positive_fold_rate": 0.60,
            "min_target_fold_rate": 0.60,
        }
        self.assertEqual(decide_walk_forward(summary), "rejected by drawdown")

    def test_summary_detects_validation_overfit_shape(self) -> None:
        fold_summary = pd.DataFrame(
            {
                "test_positive": [True, False, False],
                "test_target_reached": [False, False, False],
                "validation_target_reached": [True, True, True],
                "test_equity_ruined": [False, False, False],
            }
        )
        index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        equity = pd.Series([1.0, 1.01, 1.02], index=index)
        summary = summarize_walk_forward(fold_summary, equity, wf_config())
        self.assertEqual(decide_walk_forward(summary), "validation overfit")

    def test_stitched_equity_compounds_fold_results(self) -> None:
        index_1 = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        index_2 = pd.date_range("2024-01-03", periods=2, freq="D", tz="UTC")
        stitched = stitch_oos_equity(
            [
                (1, pd.Series([1.0, 1.10], index=index_1)),
                (2, pd.Series([1.0, 1.10], index=index_2)),
            ]
        )
        self.assertAlmostEqual(float(stitched["equity"].iloc[-1]), 1.21)


if __name__ == "__main__":
    unittest.main()
