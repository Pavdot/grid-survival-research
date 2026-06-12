from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.Index
    validation: pd.Index
    test: pd.Index


@dataclass(frozen=True)
class WalkForwardFold:
    train: pd.Index
    test: pd.Index


def temporal_train_validation_test_split(
    index: pd.Index,
    train_fraction: float,
    validation_fraction: float,
    embargo_bars: int,
) -> TemporalSplit:
    if not index.is_monotonic_increasing:
        raise ValueError("Temporal split requires a sorted index")
    n = len(index)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    train = index[: max(0, train_end - embargo_bars)]
    validation = index[train_end: max(train_end, validation_end - embargo_bars)]
    test = index[validation_end:]
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError("Temporal split produced an empty partition")
    if train.max() >= validation.min() or validation.max() >= test.min():
        raise ValueError("Temporal split order is invalid")
    return TemporalSplit(train=train, validation=validation, test=test)


def walk_forward_splits(
    index: pd.Index,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    embargo_bars: int = 0,
) -> list[WalkForwardFold]:
    if not index.is_monotonic_increasing:
        raise ValueError("Walk-forward split requires a sorted index")
    folds: list[WalkForwardFold] = []
    start = 0
    while start + train_bars + embargo_bars + test_bars <= len(index):
        train = index[start : start + train_bars]
        test_start = start + train_bars + embargo_bars
        test = index[test_start : test_start + test_bars]
        if train.max() >= test.min():
            raise ValueError("Embargo failed to separate train and test")
        folds.append(WalkForwardFold(train=train, test=test))
        start += step_bars
    return folds

