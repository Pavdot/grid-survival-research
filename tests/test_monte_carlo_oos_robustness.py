from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.monte_carlo_oos_robustness import (
    MonteCarloConfig,
    decide_monte_carlo,
    equity_from_trade_pnl,
    fold_block_bootstrap,
    fold_bootstrap,
    load_variant_outputs,
    max_drawdown_from_values,
    monthly_return_from_total,
    run_monte_carlo,
)


def make_fold_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": ["test", "test", "test"],
            "fold_id": [1, 2, 3],
            "test_start": [
                "2024-01-01 00:00:00+00:00",
                "2024-02-01 00:00:00+00:00",
                "2024-03-01 00:00:00+00:00",
            ],
            "test_end": [
                "2024-01-31 00:00:00+00:00",
                "2024-03-02 00:00:00+00:00",
                "2024-03-31 00:00:00+00:00",
            ],
            "test_total_return": [0.20, -0.05, 0.10],
            "test_monthly_return": [0.20, -0.05, 0.10],
            "test_positive": [True, False, True],
            "test_target_reached": [True, False, False],
            "test_equity_ruined": [False, False, False],
        }
    )


def write_iteration_fixture(root: Path, variant: str = "test") -> None:
    make_fold_summary().to_csv(root / f"walk_forward_fold_summary_{variant}.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "fold_id": [1, 1, 2, 3],
            "equity": [1.0, 1.20, 1.14, 1.254],
        }
    ).to_csv(root / f"walk_forward_oos_equity_{variant}.csv", index=False)
    trade_dir = root / "selected_fold_trades"
    trade_dir.mkdir()
    for fold_id, pnl in {1: [0.08, 0.12], 2: [-0.05], 3: [0.04, 0.06]}.items():
        pd.DataFrame(
            {
                "exit_timestamp": pd.date_range("2024-01-01", periods=len(pnl), freq="h", tz="UTC"),
                "realized_pnl": pnl,
            }
        ).to_csv(trade_dir / f"{variant}_fold_{fold_id:03d}_trades.csv", index=False)


class MonteCarloOosRobustnessTests(unittest.TestCase):
    def test_monthly_return_from_total_rejects_non_positive_days(self) -> None:
        with self.assertRaises(ValueError):
            monthly_return_from_total(0.10, 0)

    def test_equity_and_drawdown_from_trade_pnl(self) -> None:
        equity = equity_from_trade_pnl(np.array([0.10, -0.20, 0.05]))
        self.assertEqual(equity.tolist(), [1.0, 1.10, 0.90, 0.95])
        self.assertAlmostEqual(max_drawdown_from_values(equity), 0.90 / 1.10 - 1.0)

    def test_load_variant_outputs_reads_oos_folds_and_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_iteration_fixture(root)
            outputs = load_variant_outputs(root, "test")
            self.assertEqual(len(outputs.fold_summary), 3)
            self.assertEqual(len(outputs.trades), 5)
            self.assertEqual(len(outputs.oos_equity), 4)

    def test_load_variant_outputs_refuses_missing_predictions_or_trades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_iteration_fixture(root)
            (root / "selected_fold_trades" / "test_fold_002_trades.csv").unlink()
            with self.assertRaises(FileNotFoundError):
                load_variant_outputs(root, "test")

    def test_fold_bootstrap_is_seed_deterministic(self) -> None:
        fold_summary = make_fold_summary()
        first = fold_bootstrap(fold_summary, 5, np.random.default_rng(7), 0.20)
        second = fold_bootstrap(fold_summary, 5, np.random.default_rng(7), 0.20)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["method"]), {"fold_bootstrap"})

    def test_fold_block_bootstrap_uses_only_existing_oos_blocks(self) -> None:
        fold_summary = make_fold_summary()
        trades = pd.DataFrame(
            {
                "fold_id": [1, 1, 2, 3],
                "realized_pnl": [0.10, 0.10, -0.05, 0.10],
            }
        )
        samples = fold_block_bootstrap(fold_summary, trades, 10, np.random.default_rng(11), 0.20)
        self.assertEqual(len(samples), 10)
        self.assertEqual(set(samples["method"]), {"fold_block_bootstrap"})
        self.assertTrue(samples["target_fold_rate"].between(0.0, 1.0).all())

    def test_run_monte_carlo_writes_outputs_without_reselecting_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "mc"
            write_iteration_fixture(root)
            payload = run_monte_carlo(
                MonteCarloConfig(
                    iteration_dir=root,
                    variant="test",
                    iterations=8,
                    seed=42,
                    target_monthly_return=0.20,
                    output_dir=output,
                )
            )
            self.assertIn(
                payload["decision"],
                {
                    "robust",
                    "fragile positive edge",
                    "positive but target not robust",
                    "target likely overfit",
                    "not robust",
                },
            )
            self.assertTrue((output / "monte_carlo_samples_test.csv").exists())
            self.assertTrue((output / "monte_carlo_report_test.md").exists())

    def test_decision_marks_target_as_not_robust_when_target_probability_is_low(self) -> None:
        decision = decide_monte_carlo(
            {"monthly_return": 0.13, "target_monthly_return": 0.20},
            [
                {
                    "method": "fold_block_bootstrap",
                    "monthly_return_p05": 0.05,
                    "monthly_return_p50": 0.13,
                    "target_probability": 0.10,
                    "ruin_probability": 0.0,
                }
            ],
        )
        self.assertEqual(decision, "positive but target not robust")


if __name__ == "__main__":
    unittest.main()
