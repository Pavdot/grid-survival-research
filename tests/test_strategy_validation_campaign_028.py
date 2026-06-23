from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.research.strategy_validation_campaign_028 import (
    evaluate_finalists,
    global_decision,
    make_cpcv_blocks,
    make_cpcv_splits,
    run_campaign,
    run_cpcv,
    summarize_fold_subset,
    write_report,
)


def fold_rows(family: str, returns: list[float]) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for idx, value in enumerate(returns, start=1):
        test_start = start + pd.Timedelta(days=30 * (idx - 1))
        rows.append(
            {
                "family": family,
                "fold_id": idx,
                "test_start": test_start,
                "test_end": test_start + pd.Timedelta(days=30),
                "test_total_return": value,
                "test_monthly_return": value,
                "test_positive": value > 0,
                "test_target_reached": value >= 0.20,
                "test_equity_ruined": False,
            }
        )
    return pd.DataFrame(rows)


def cpcv_config() -> dict:
    return {
        "target": {
            "robust_monthly_return": 0.13,
            "aspirational_monthly_return": 0.20,
            "min_positive_fold_rate": 0.60,
            "min_improvement_vs_017": 0.001,
        },
        "cpcv": {"block_count": 3, "test_block_count": 1, "purge_adjacent_blocks": 0},
    }


class StrategyValidationCampaign028Tests(unittest.TestCase):
    def test_cpcv_blocks_are_chronological_and_non_overlapping(self) -> None:
        blocks = make_cpcv_blocks([1, 2, 3, 4, 5, 6], block_count=3)
        self.assertEqual(blocks, {0: (1, 2), 1: (3, 4), 2: (5, 6)})
        all_folds = [fold for values in blocks.values() for fold in values]
        self.assertEqual(sorted(all_folds), [1, 2, 3, 4, 5, 6])

    def test_cpcv_purges_adjacent_blocks(self) -> None:
        blocks = make_cpcv_blocks([1, 2, 3, 4, 5, 6], block_count=6)
        splits = make_cpcv_splits(blocks, test_block_count=2, purge_adjacent_blocks=1)
        for split in splits:
            self.assertTrue(set(split.train_blocks).isdisjoint(split.test_blocks))
            self.assertTrue(set(split.train_blocks).isdisjoint(split.purged_blocks))
            for test_block in split.test_blocks:
                self.assertNotIn(test_block - 1, split.train_blocks)
                self.assertNotIn(test_block + 1, split.train_blocks)

    def test_cpcv_selection_uses_train_folds_only(self) -> None:
        # Candidate A is best on the first train splits; candidate B has a huge
        # return in one possible test fold, which must not influence selection.
        matrix = pd.concat(
            [
                fold_rows("candidate_a", [0.20, 0.20, -0.10, -0.10, -0.10, -0.10]),
                fold_rows("candidate_b", [-0.05, -0.05, 0.50, 0.50, 0.50, 0.50]),
            ],
            ignore_index=True,
        )
        config = cpcv_config()
        config["cpcv"] = {"block_count": 3, "test_block_count": 1, "purge_adjacent_blocks": 0}
        _summary, paths = run_cpcv(matrix, config)
        split_zero = paths[paths["split_id"].eq(0)].iloc[0]
        self.assertEqual(split_zero["test_blocks"], "0")
        self.assertEqual(split_zero["selected_family"], "candidate_b")
        self.assertEqual(split_zero["train_fold_ids"], "3,4,5,6")

    def test_summarize_fold_subset_reports_ruin_and_monthly(self) -> None:
        frame = fold_rows("candidate", [0.10, -1.10])
        frame.loc[1, "test_equity_ruined"] = True
        summary = summarize_fold_subset(frame, (1, 2), robust_monthly=0.13)
        self.assertTrue(summary["equity_ruined"])
        self.assertLessEqual(summary["total_return"], -1.0)

    def test_evaluate_finalists_requires_all_research_gates(self) -> None:
        universe = pd.DataFrame(
            [
                {
                    "family": "baseline_017_locked",
                    "aggregate_monthly_return": 0.13,
                    "positive_fold_rate": 0.7,
                    "equity_ruined": False,
                },
                {
                    "family": "candidate",
                    "aggregate_monthly_return": 0.16,
                    "positive_fold_rate": 0.8,
                    "equity_ruined": False,
                },
            ]
        )
        cpcv = pd.DataFrame([{"family": "candidate", "cpcv_pass": True}])
        mc = pd.DataFrame([{"family": "candidate", "mc_pass": False}])
        worst = pd.DataFrame([{"family": "candidate", "worst_case_pass": True}])
        surface = pd.DataFrame([{"family": "candidate", "surface_full_zone_pass": True}])
        final = evaluate_finalists(universe, cpcv, mc, worst, surface, cpcv_config())
        row = final[final["family"].eq("candidate")].iloc[0]
        self.assertTrue(row["wf_pass"])
        self.assertFalse(row["research_pass"])
        self.assertEqual(row["final_verdict"], "research rejected by robustness gates")

    def test_report_is_written_even_without_accepted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            payload = {
                "decision": "rejected",
                "walk_forward_finalist_summary": [
                    {
                        "family": "candidate",
                        "final_verdict": "rejected",
                        "aggregate_monthly_return": 0.01,
                        "improvement_vs_017_monthly": -0.01,
                        "positive_fold_rate": 0.5,
                        "wf_pass": False,
                        "mc_pass": False,
                        "cpcv_pass": False,
                        "worst_case_pass": False,
                        "surface_full_zone_pass": False,
                    }
                ],
                "cpcv_selection_paths": [
                    {
                        "split_id": 0,
                        "selected_family": "candidate",
                        "train_blocks": "1",
                        "test_blocks": "2",
                        "test_monthly_return": 0.01,
                        "test_equity_ruined": False,
                    }
                ],
            }
            report = write_report(output, payload)
            self.assertTrue(report.exists())
            self.assertIn("rejected", report.read_text(encoding="utf-8"))

    def test_global_decision_prioritizes_fragile_before_rejected(self) -> None:
        finalists = pd.DataFrame(
            [
                {"final_verdict": "rejected", "wf_pass": False},
                {"final_verdict": "fragile but promising", "wf_pass": True},
            ]
        )
        self.assertEqual(global_decision(finalists), "fragile but promising")


if __name__ == "__main__":
    unittest.main()
