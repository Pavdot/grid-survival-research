from __future__ import annotations

import unittest

import pandas as pd

from src.research.ml_threshold_research import (
    ThresholdCandidate,
    add_intratrade_score_diagnostics,
    evaluate_candidate_on_split,
    make_threshold_candidates,
    select_best_candidate,
    validate_prediction_frame,
)


def research_config() -> dict:
    return {
        "threshold_search": {
            "open_threshold_start": 0.50,
            "open_threshold_stop": 0.60,
            "threshold_step": 0.05,
            "minimum_add_threshold_floor": 0.60,
            "kill_switch_thresholds": [None, 0.20],
            "min_grid_fraction_baseline": 0.05,
            "min_grid_absolute_cap": 500,
        },
        "diagnostics": {"worst_pnl_quantile": 0.10, "score_deciles": 5},
    }


def scored_fixture() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "exit_timestamp": index,
            "grid_survived": [1, 0, 1, 0, 1, 1, 0, 1],
            "realized_pnl": [0.01, -0.02, 0.02, -0.01, 0.03, 0.02, -0.03, 0.01],
            "max_adverse_excursion": [0.0] * 8,
            "number_of_levels_filled": [1, 2, 1, 3, 1, 2, 1, 1],
            "stopped_by_regime_break": [0, 1, 0, 0, 0, 0, 1, 0],
            "stopped_by_max_loss": [0] * 8,
            "stopped_by_max_holding": [0] * 8,
            "stopped_by_volatility_shock": [0] * 8,
            "stopped_by_exposure": [0] * 8,
            "stopped_by_kill_switch": [0] * 8,
            "fees_paid": [0.001] * 8,
            "slippage_paid": [0.001] * 8,
            "time_to_exit": [1.0] * 8,
            "grid_survival_score": [0.55, 0.58, 0.75, 0.52, 0.90, 0.85, 0.40, 0.95],
            "dataset_split": ["validation"] * 4 + ["test"] * 4,
        },
        index=index,
    )
    predictions = frame[["grid_survival_score", "dataset_split"]].copy()
    return add_intratrade_score_diagnostics(frame, predictions)


class MLThresholdResearchTests(unittest.TestCase):
    def test_candidates_keep_add_threshold_at_or_above_open_and_floor(self) -> None:
        candidates = make_threshold_candidates(research_config())
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertGreaterEqual(candidate.add_threshold, candidate.open_threshold)
            self.assertGreaterEqual(candidate.add_threshold, 0.60)
            if candidate.kill_switch_threshold is not None:
                self.assertLess(candidate.kill_switch_threshold, candidate.open_threshold)

    def test_selection_refuses_test_rows(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "split": "test",
                    "open_threshold": 0.5,
                    "add_threshold": 0.6,
                    "kill_switch_threshold": None,
                    "baseline_grids": 10,
                    "number_of_grids": 10,
                    "expectancy": 1.0,
                    "max_drawdown": 0.0,
                    "number_of_forced_exits": 0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            select_best_candidate(summary, research_config())

    def test_selected_threshold_can_be_applied_to_test_without_reselection(self) -> None:
        scored = scored_fixture()
        candidate = ThresholdCandidate(open_threshold=0.80, add_threshold=0.80, kill_switch_threshold=None)
        validation = evaluate_candidate_on_split(scored, candidate, "validation", 0.10)
        test = evaluate_candidate_on_split(scored, candidate, "test", 0.10)
        self.assertEqual(validation["open_threshold"], test["open_threshold"])
        self.assertEqual(validation["add_threshold"], test["add_threshold"])
        self.assertEqual(test["number_of_grids"], 3)

    def test_prediction_validation_rejects_missing_scores(self) -> None:
        predictions = pd.DataFrame(
            {
                "grid_survival_score": [0.7, None],
                "dataset_split": ["validation", "test"],
            }
        )
        with self.assertRaises(ValueError):
            validate_prediction_frame(predictions)


if __name__ == "__main__":
    unittest.main()

