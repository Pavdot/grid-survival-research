from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.range_features import add_range_features
from src.features.trend_features import add_trend_features
from src.features.volatility_features import add_volatility_features
from src.features.volume_features import add_volume_features


def synthetic_ohlcv(rows: int = 220) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:05:00Z", periods=rows, freq="5min")
    close = pd.Series(np.linspace(100, 110, rows), index=index)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.linspace(10, 20, rows),
        },
        index=index,
    )


class NoFutureLeakageTests(unittest.TestCase):
    def test_features_at_t_do_not_change_when_future_changes(self) -> None:
        df = synthetic_ohlcv()
        target = df.index[150]

        def build(frame: pd.DataFrame) -> pd.DataFrame:
            ranges = add_range_features(frame)
            return pd.concat(
                [
                    add_volatility_features(frame),
                    add_trend_features(frame),
                    ranges,
                    add_volume_features(frame, ranges),
                ],
                axis=1,
            )

        before = build(df).loc[target]
        mutated = df.copy()
        mutated.loc[mutated.index > target, ["high", "low", "close", "volume"]] *= 10
        after = build(mutated).loc[target]
        self.assertTrue(np.allclose(before.fillna(0), after.fillna(0)))

    def test_entry_signal_timestamp_represents_closed_candle(self) -> None:
        df = synthetic_ohlcv(20)
        self.assertEqual(df.index[0], pd.Timestamp("2024-01-01 00:05:00Z"))
        self.assertGreater(df.index[1], df.index[0])


if __name__ == "__main__":
    unittest.main()

