from __future__ import annotations

import pandas as pd


def add_session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.to_datetime(index, utc=True)
    features = pd.DataFrame(index=index)
    features["hour_utc"] = idx.hour
    features["day_of_week"] = idx.dayofweek
    features["is_asia_session"] = ((idx.hour >= 0) & (idx.hour < 8)).astype(int)
    features["is_london_session"] = ((idx.hour >= 7) & (idx.hour < 16)).astype(int)
    features["is_new_york_session"] = ((idx.hour >= 13) & (idx.hour < 22)).astype(int)
    features["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return features

