from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.research.revalidate_top3_distinct_policies import (
    ScenarioSpec,
    audit_data_files,
    prefilter_side_signal_for_audit,
    rank_outputs,
    select_distinct_policies,
    split_pre_holdout,
)


def make_ohlcv(index: pd.DatetimeIndex, minutes: int) -> pd.DataFrame:
    open_dt = index - pd.Timedelta(minutes=minutes)
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1.0,
            "open_time": (open_dt.view("int64") // 1_000_000).astype("int64"),
            "close_time": ((index - pd.Timedelta(milliseconds=1)).view("int64") // 1_000_000).astype("int64"),
            "open_datetime": open_dt,
            "close_datetime": index - pd.Timedelta(milliseconds=1),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


class RevalidateTop3PolicyTests(unittest.TestCase):
    def test_data_audit_excludes_incomplete_1h_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_5m = pd.date_range("2024-01-01 00:05", periods=25, freq="5min", tz="UTC")
            one_h = pd.date_range("2024-01-01 01:00", periods=3, freq="1h", tz="UTC")
            market_path = root / "m5.parquet"
            signal_path = root / "h1.parquet"
            make_ohlcv(full_5m, 5).to_parquet(market_path)
            make_ohlcv(one_h, 60).to_parquet(signal_path)

            market, signal, audit = audit_data_files(market_path, signal_path, min_coverage_rate=0.99)

            self.assertEqual(audit["observed_5m_bars"], 25)
            self.assertEqual(audit["filtered_5m_bars"], 24)
            self.assertEqual(audit["filtered_1h_bars"], 2)
            self.assertEqual(audit["excluded_incomplete_1h_bars"], 1)
            self.assertEqual(len(market), 24)
            self.assertEqual(len(signal), 2)

    def test_data_audit_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            idx = pd.date_range("2024-01-01 00:05", periods=12, freq="5min", tz="UTC")
            dup_idx = idx.insert(3, idx[3])
            one_h = pd.date_range("2024-01-01 01:00", periods=2, freq="1h", tz="UTC")
            market_path = root / "m5.parquet"
            signal_path = root / "h1.parquet"
            make_ohlcv(dup_idx, 5).to_parquet(market_path)
            make_ohlcv(one_h, 60).to_parquet(signal_path)

            with self.assertRaises(ValueError):
                audit_data_files(market_path, signal_path, min_coverage_rate=0.99)

    def test_split_pre_holdout_keeps_holdout_out_of_walk_forward(self) -> None:
        index = pd.date_range("2024-01-01 00:00", periods=420, freq="1D", tz="UTC")
        frame = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1}, index=index)
        windows, holdout, holdout_start = split_pre_holdout(
            frame,
            train_days=180,
            test_days=30,
            step_days=30,
            embargo_bars=0,
            holdout_days=90,
        )
        self.assertTrue(windows)
        self.assertLess(windows[-1].test.max(), holdout_start)
        self.assertGreaterEqual(holdout.min(), holdout_start)

    def test_prefilter_blocks_next_bar_entry_not_signal_bar(self) -> None:
        index = pd.date_range("2024-01-01 00:00", periods=5, freq="5min", tz="UTC")
        signal = pd.Series([pd.NA, "long", pd.NA, pd.NA, pd.NA], index=index, dtype="object")
        mask = pd.Series(False, index=index)
        mask.iloc[2] = True
        scenario = ScenarioSpec("next_bar_lagged", "audit", "next_bar_open", signal_lag_bars=0)

        filtered = prefilter_side_signal_for_audit(signal, index, mask, scenario)

        self.assertTrue(pd.isna(filtered.iloc[1]))
        self.assertFalse(bool(mask.iloc[1]))

    def test_distinct_selection_skips_duplicate_fingerprints(self) -> None:
        selected, distinctness = select_distinct_policies(
            [
                {
                    "policy": "a",
                    "source_variant": "v1",
                    "trades_fingerprint": "same",
                    "equity_fingerprint": "same",
                    "legacy_monthly_return": 0.2,
                },
                {
                    "policy": "b",
                    "source_variant": "v2",
                    "trades_fingerprint": "same",
                    "equity_fingerprint": "same",
                    "legacy_monthly_return": 0.1,
                },
                {
                    "policy": "c",
                    "source_variant": "v3",
                    "trades_fingerprint": "other",
                    "equity_fingerprint": "other",
                    "legacy_monthly_return": 0.05,
                },
            ],
            desired_count=2,
        )
        self.assertEqual(selected, ["a", "c"])
        duplicate = distinctness[distinctness["policy"].eq("b")].iloc[0]
        self.assertEqual(duplicate["duplicate_of"], "a")
        self.assertFalse(bool(duplicate["selected"]))

    def test_rank_outputs_uses_requested_sort_keys(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "scope": "combined_oos",
                    "selection_mode": "locked",
                    "policy": "a",
                    "scenario": "legacy",
                    "monthly_return": 0.10,
                    "total_return": 1.0,
                    "positive_fold_rate": 1.0,
                    "net_pnl": 1.0,
                    "profit_factor": 2.0,
                    "fees_paid": 0.1,
                    "slippage_paid": 0.1,
                    "equity_ruined": False,
                    "max_drawdown": -0.20,
                    "fold_rate_above_8pct": 0.5,
                    "worst_fold_monthly": -0.01,
                },
                {
                    "scope": "combined_oos",
                    "selection_mode": "locked",
                    "policy": "b",
                    "scenario": "legacy",
                    "monthly_return": 0.15,
                    "total_return": 0.8,
                    "positive_fold_rate": 0.8,
                    "net_pnl": 0.8,
                    "profit_factor": 1.5,
                    "fees_paid": 0.1,
                    "slippage_paid": 0.1,
                    "equity_ruined": False,
                    "max_drawdown": -0.40,
                    "fold_rate_above_8pct": 0.4,
                    "worst_fold_monthly": -0.2,
                },
                {
                    "scope": "combined_oos",
                    "selection_mode": "locked",
                    "policy": "c",
                    "scenario": "realistic_timing_costs",
                    "monthly_return": 0.03,
                    "total_return": 0.3,
                    "positive_fold_rate": 0.9,
                    "net_pnl": 0.3,
                    "profit_factor": 1.2,
                    "fees_paid": 0.2,
                    "slippage_paid": 0.1,
                    "equity_ruined": False,
                    "max_drawdown": -0.10,
                    "fold_rate_above_8pct": 0.2,
                    "worst_fold_monthly": -0.01,
                },
            ]
        )
        legacy, realistic, robust = rank_outputs(summary)
        self.assertEqual(legacy.iloc[0]["policy"], "b")
        self.assertEqual(realistic.iloc[0]["policy"], "c")
        self.assertNotIn("b", set(robust["policy"]))


if __name__ == "__main__":
    unittest.main()
