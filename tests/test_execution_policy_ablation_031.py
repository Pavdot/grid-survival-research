from __future__ import annotations

import unittest

import pandas as pd

from src.research.execution_policy_ablation_031 import (
    build_policy_matrix,
    policy_trade_attribution,
    size_viability,
    summarize_policy_ablation,
    synthesize_missing_policies,
)


def candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": "fundamental_trend_escape_v2",
                "gross_monthly_median": 0.16,
                "gross_monthly_p10": 0.08,
                "aggregate_max_drawdown": -0.2,
                "ruin_rate": 0.0,
            }
        ]
    )


def costs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "family": "fundamental_trend_escape_v2",
                "policy": "gate_only_taker",
                "account_equity_usdt": 10000.0,
                "monthly_execution_cost_estimate": 0.02,
                "p90_fold_execution_cost": 0.03,
                "entry_rejected_rate": 0.20,
                "exact_signal_fraction": 0.9,
                "synthetic_mapping_fraction": 0.1,
            },
            {
                "family": "fundamental_trend_escape_v2",
                "policy": "sliced_taker",
                "account_equity_usdt": 10000.0,
                "monthly_execution_cost_estimate": 0.01,
                "p90_fold_execution_cost": 0.015,
                "entry_rejected_rate": 0.10,
                "exact_signal_fraction": 0.9,
                "synthetic_mapping_fraction": 0.1,
            },
        ]
    )


def config() -> dict:
    return {
        "ablation": {
            "baseline_policy": "gate_only_taker",
            "primary_equity_usdt": 10000.0,
            "policies": ["taker_only", "gate_only_taker", "sliced_taker"],
            "min_net_p10_improvement": 0.0,
            "min_net_median_improvement": 0.0,
            "max_entry_rejected_rate": 0.35,
        }
    }


class ExecutionPolicyAblation031Tests(unittest.TestCase):
    def test_synthesize_taker_only_marks_proxy(self) -> None:
        result = synthesize_missing_policies(costs(), ["taker_only", "gate_only_taker"], "gate_only_taker")
        taker = result[result["policy"].eq("taker_only")].iloc[0]
        self.assertTrue(bool(taker["synthetic_policy_proxy"]))
        self.assertEqual(float(taker["entry_rejected_rate"]), 0.0)

    def test_policy_attribution_compares_against_baseline(self) -> None:
        matrix = build_policy_matrix(candidates(), costs(), ["gate_only_taker", "sliced_taker"])
        attribution = policy_trade_attribution(matrix, "gate_only_taker", 10000.0)
        sliced = attribution[attribution["policy"].eq("sliced_taker")].iloc[0]
        self.assertGreater(float(sliced["net_p10_delta"]), 0.0)
        self.assertLess(float(sliced["entry_rejected_rate_delta"]), 0.0)

    def test_summary_flags_real_policy_improvement(self) -> None:
        matrix = build_policy_matrix(candidates(), costs(), ["gate_only_taker", "sliced_taker"])
        attribution = policy_trade_attribution(matrix, "gate_only_taker", 10000.0)
        summary = summarize_policy_ablation(matrix, attribution, config())
        sliced = summary[summary["policy"].eq("sliced_taker")].iloc[0]
        self.assertTrue(bool(sliced["policy_improves_net"]))

    def test_synthetic_policy_proxy_is_not_accepted_as_improvement(self) -> None:
        synthesized = synthesize_missing_policies(costs(), ["taker_only", "gate_only_taker"], "gate_only_taker")
        matrix = build_policy_matrix(candidates(), synthesized, ["taker_only", "gate_only_taker"])
        attribution = policy_trade_attribution(matrix, "gate_only_taker", 10000.0)
        summary = summarize_policy_ablation(matrix, attribution, config())
        taker = summary[summary["policy"].eq("taker_only")].iloc[0]
        self.assertTrue(bool(taker["synthetic_policy_proxy"]))
        self.assertFalse(bool(taker["policy_improves_net"]))

    def test_size_viability_requires_positive_p10_and_rejection_limit(self) -> None:
        matrix = build_policy_matrix(candidates(), costs(), ["gate_only_taker", "sliced_taker"])
        viability = size_viability(matrix, config())
        self.assertIn("viable", viability.columns)
        self.assertTrue(viability["viable"].astype(bool).any())


if __name__ == "__main__":
    unittest.main()
