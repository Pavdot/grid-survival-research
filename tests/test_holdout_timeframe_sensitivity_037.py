from __future__ import annotations

import pandas as pd

from src.research.holdout_timeframe_sensitivity_037 import complete_resample_from_5m, normalize_binance_klines_1m


def _frame(rows: int) -> pd.DataFrame:
    index = pd.date_range("2026-01-01 00:05", periods=rows, freq="5min", tz="UTC", name="timestamp")
    values = [float(i + 1) for i in range(rows)]
    open_times = (index - pd.Timedelta(minutes=5)).view("int64") // 1_000_000
    close_times = index.view("int64") // 1_000_000 - 1
    return pd.DataFrame(
        {
            "open": values,
            "high": [value + 1.0 for value in values],
            "low": [value - 1.0 for value in values],
            "close": values,
            "volume": [1.0] * rows,
            "open_time": open_times,
            "close_time": close_times,
        },
        index=index,
    )


def test_complete_resample_drops_incomplete_signal_bar() -> None:
    signal, audit = complete_resample_from_5m(_frame(5), "15m")

    assert len(signal) == 1
    assert signal.index[0] == pd.Timestamp("2026-01-01 00:15", tz="UTC")
    assert audit["expected_5m_per_signal_bar"] == 3
    assert audit["dropped_incomplete_signal_bars"] == 1


def test_complete_resample_rejects_non_5m_multiple() -> None:
    try:
        complete_resample_from_5m(_frame(3), "7m")
    except ValueError as exc:
        assert "multiple of 5m" in str(exc)
    else:
        raise AssertionError("Expected non-5m multiple timeframe to fail")


def test_normalize_binance_1m_uses_closed_boundary_index() -> None:
    rows = []
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    for i in range(3):
        open_time = int((start + pd.Timedelta(minutes=i)).timestamp() * 1000)
        rows.append(
            [
                open_time,
                "100.0",
                "101.0",
                "99.0",
                "100.5",
                "1.0",
                open_time + 60_000 - 1,
                "100.5",
                10,
                "0.5",
                "50.0",
                "0",
            ]
        )

    frame = normalize_binance_klines_1m(rows)

    assert list(frame.index) == [
        pd.Timestamp("2026-01-01 00:01", tz="UTC"),
        pd.Timestamp("2026-01-01 00:02", tz="UTC"),
        pd.Timestamp("2026-01-01 00:03", tz="UTC"),
    ]
    assert frame.index.name == "timestamp"
