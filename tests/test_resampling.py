from __future__ import annotations

import unittest

import pandas as pd

from src.data.resample_timeframes import resample_closed_ohlcv


class ResamplingTests(unittest.TestCase):
    def test_resample_uses_only_closed_source_candles(self) -> None:
        index = pd.date_range("2024-01-01 00:05:00Z", periods=6, freq="5min")
        df = pd.DataFrame(
            {
                "open": [1, 2, 3, 4, 5, 6],
                "high": [2, 3, 4, 100, 6, 7],
                "low": [0, 1, 2, 3, 4, 5],
                "close": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                "volume": [10, 10, 10, 10, 10, 10],
                "open_time": range(6),
                "close_time": range(6),
            },
            index=index,
        )
        out = resample_closed_ohlcv(df, "15m")
        first = out.loc[pd.Timestamp("2024-01-01 00:15:00Z")]
        self.assertEqual(first["open"], 1)
        self.assertEqual(first["high"], 4)
        self.assertEqual(first["close"], 3.5)
        self.assertEqual(first["volume"], 30)
        self.assertNotEqual(first["high"], 100)


if __name__ == "__main__":
    unittest.main()

