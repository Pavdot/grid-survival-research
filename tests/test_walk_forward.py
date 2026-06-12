from __future__ import annotations

import unittest

import pandas as pd

from src.backtesting.walk_forward import temporal_train_validation_test_split, walk_forward_splits


class WalkForwardTests(unittest.TestCase):
    def test_temporal_split_orders_partitions_with_embargo(self) -> None:
        index = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
        split = temporal_train_validation_test_split(index, 0.6, 0.2, embargo_bars=3)
        self.assertLess(split.train.max(), split.validation.min())
        self.assertLess(split.validation.max(), split.test.min())
        self.assertEqual(len(split.train), 57)

    def test_walk_forward_has_no_train_test_overlap(self) -> None:
        index = pd.date_range("2024-01-01", periods=80, freq="5min", tz="UTC")
        folds = walk_forward_splits(index, train_bars=30, test_bars=10, step_bars=10, embargo_bars=2)
        self.assertTrue(folds)
        for fold in folds:
            self.assertLess(fold.train.max(), fold.test.min())
            self.assertTrue(set(fold.train).isdisjoint(set(fold.test)))


if __name__ == "__main__":
    unittest.main()

