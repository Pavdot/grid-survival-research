from __future__ import annotations

import unittest

import pandas as pd

from src.research.execution_aware_nested_walk_forward_030 import (
    apply_execution_adjustment,
    build_candidate_matrix,
    coverage_is_usable,
    decide_nested_result,
    select_without_test_leakage,
)


def finalist() -> pd.Series:
    return pd.Series(
        {
            "family": "fundamental_trend_escape_v2",
            "aggregate_monthly_return": 0.16,
            "mc_fold_block_monthly_p05": 0.08,
            "aggregate_max_drawdown": -0.25,
            "mc_fold_block_ruin_probability": 0.0,
            "cpcv_ruin_rate": 0.0,
        }
    )


def cost(policy: str = "gate_only_taker", rejected: float = 0.1) -> pd.Series:
    return pd.Series(
        {
            "family": "fundamental_trend_escape_v2",
            "policy": policy,
            "account_equity_usdt": 10000.0,
            "monthly_execution_cost_estimate": 0.01,
            "p90_fold_execution_cost": 0.02,
            "entry_rejected_rate": rejected,
            "exact_signal_fraction": 0.9,
            "synthetic_mapping_fraction": 0.1,
        }
    )


class ExecutionAwareNestedWalkForward030Tests(unittest.TestCase):
    def test_execution_adjustment_penalizes_costs_and_rejected_entries(self) -> None:
        row = apply_execution_adjustment(finalist(), cost())
        self.assertAlmostEqual(row["net_monthly_median"], 0.16 - 0.01 - 0.016)
        self.assertAlmostEqual(row["net_monthly_p10"], 0.08 - 0.02 - 0.008)

    def test_selection_refuses_test_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "family": "a",
                    "account_equity_usdt": 10000.0,
                    "net_monthly_p10": 0.1,
                    "net_monthly_median": 0.2,
                    "test_monthly_return": -1.0,
                }
            ]
        )
        with self.assertRaises(ValueError):
            select_without_test_leakage(frame, 10000.0)

    def test_selection_uses_primary_equity_and_p10(self) -> None:
        frame = pd.DataFrame(
            [
                {"family": "a", "account_equity_usdt": 10000.0, "net_monthly_p10": 0.02, "net_monthly_median": 0.20},
                {"family": "b", "account_equity_usdt": 10000.0, "net_monthly_p10": 0.05, "net_monthly_median": 0.10},
                {"family": "c", "account_equity_usdt": 5000.0, "net_monthly_p10": 0.50, "net_monthly_median": 0.60},
            ]
        )
        selected = select_without_test_leakage(frame, 10000.0)
        self.assertEqual(selected.iloc[0]["family"], "b")

    def test_coverage_gate_blocks_low_exact_fraction(self) -> None:
        self.assertFalse(coverage_is_usable(pd.DataFrame([{"exact_signal_fraction": 0.79}]), 0.8))
        self.assertTrue(coverage_is_usable(pd.DataFrame([{"exact_signal_fraction": 0.80}]), 0.8))

    def test_verdict_accepts_positive_p10_no_ruin(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "net_monthly_p10": 0.01,
                    "net_monthly_median": 0.04,
                    "ruin_rate": 0.0,
                    "aggregate_max_drawdown": -0.2,
                    "entry_rejected_rate": 0.2,
                }
            ]
        )
        config = {
            "decision": {
                "exact_coverage_threshold": 0.8,
                "p10_net_monthly_min": 0.0,
                "median_net_monthly_min": 0.0,
                "ruin_rate_max": 0.0,
                "max_drawdown_floor": -0.35,
                "max_rejected_entry_rate": 0.35,
            }
        }
        verdict = decide_nested_result(selected, pd.DataFrame([{"exact_signal_fraction": 0.9}]), config)
        self.assertEqual(verdict["verdict"], "execution-aware strategy viable")

    def test_candidate_matrix_combines_finalists_and_costs(self) -> None:
        matrix = build_candidate_matrix(pd.DataFrame([finalist()]), pd.DataFrame([cost("a"), cost("b")]))
        self.assertEqual(len(matrix), 2)
        self.assertEqual(set(matrix["policy"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
