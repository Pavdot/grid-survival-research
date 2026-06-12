from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_grid_returns(
    grid_returns: pd.Series,
    n_paths: int = 1000,
    path_length: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    values = grid_returns.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame()
    if path_length is None:
        path_length = len(values)
    paths = rng.choice(values, size=(n_paths, path_length), replace=True)
    equity = 1 + paths.cumsum(axis=1)
    return pd.DataFrame(equity)

