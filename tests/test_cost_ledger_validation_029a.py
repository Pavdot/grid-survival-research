from __future__ import annotations

import unittest

from src.research.cost_ledger_validation_029a import (
    LedgerCase,
    LedgerOrder,
    default_cases,
    execution_price,
    reconcile_case,
    run_validation,
)


class CostLedgerValidation029ATests(unittest.TestCase):
    def test_long_and_short_cases_reconcile_exactly(self) -> None:
        cases = [case for case in default_cases({"ledger": {}}) if case.case_id in {"long_taker_tp", "short_taker_tp"}]
        self.assertEqual(len(cases), 2)
        for case in cases:
            _ledger, summary = reconcile_case(case)
            self.assertEqual(summary["status"], "ok")
            self.assertLessEqual(summary["abs_diff"], 1e-12)
            self.assertAlmostEqual(summary["net_pnl"], summary["engine_pnl"], places=12)

    def test_missed_fill_has_zero_quantity_and_zero_cost(self) -> None:
        case = [case for case in default_cases({"ledger": {}}) if case.case_id == "missed_add_then_tp"][0]
        ledger, summary = reconcile_case(case)
        missed = ledger[ledger["event_type"].eq("add")].iloc[0]
        self.assertEqual(float(missed["filled_notional_pct"]), 0.0)
        self.assertEqual(float(missed["quantity"]), 0.0)
        self.assertEqual(float(missed["fees"]), 0.0)
        self.assertEqual(summary["status"], "ok")

    def test_partial_fill_is_rejected_when_not_supported(self) -> None:
        case = LedgerCase(
            "partial",
            "unsupported partial",
            (
                LedgerOrder("entry", "long", "entry", 100.0, 1.0, "maker", 0.0, 0.0, 0.0, fill_fraction=0.5),
                LedgerOrder("take_profit", "long", "exit", 101.0, 0.5, "taker", 0.0, 0.0, 0.0),
            ),
            expected_status="rejected",
        )
        _ledger, summary = reconcile_case(case)
        self.assertEqual(summary["status"], "expected_rejected")
        self.assertIn("partial fills", summary["error"])

    def test_maker_buy_price_is_better_than_taker_buy(self) -> None:
        maker = LedgerOrder("entry", "long", "entry", 100.0, 1.0, "maker", 0.0, 0.0, 2.0)
        taker = LedgerOrder("entry", "long", "entry", 100.0, 1.0, "taker", 0.0, 0.0, 2.0)
        self.assertLess(execution_price(maker), execution_price(taker))

    def test_runner_passes_default_cases(self) -> None:
        payload = run_validation(
            {
                "iteration": {"output_dir": "reports/research_iterations/test_029a_cost_ledger"},
                "ledger": {"scenario_tolerance": 1e-10, "aggregate_tolerance": 1e-8},
            }
        )
        self.assertTrue(payload["passed"])


if __name__ == "__main__":
    unittest.main()
