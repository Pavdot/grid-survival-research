from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.labeling.grid_engine import GridSimulationResult, simulate_grid_from_index
from src.labeling.grid_risk import GridRiskConfig


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series


def _make_equity_curve(index: pd.Index, trades: pd.DataFrame) -> pd.Series:
    equity = pd.Series(1.0, index=index, dtype=float)
    current_equity = 1.0
    for _, trade in trades.iterrows():
        current_equity += float(trade["realized_pnl"])
        exit_ts = trade["exit_timestamp"]
        equity.loc[equity.index >= exit_ts] = current_equity
    equity.name = "equity"
    return equity


def run_grid_backtest(
    market: pd.DataFrame,
    risk: GridRiskConfig,
    allow_open: pd.Series | None = None,
    scores: pd.Series | None = None,
    min_open_score: float | None = None,
    add_level_min_score: float | None = None,
    kill_switch_threshold: float | None = None,
    constant_size: bool = False,
) -> BacktestResult:
    if allow_open is None:
        allow_open = pd.Series(True, index=market.index)
    else:
        allow_open = allow_open.reindex(market.index).fillna(False).astype(bool)

    if scores is not None:
        scores = scores.reindex(market.index)
        score_positions = scores.reset_index(drop=True)
    else:
        score_positions = None

    rows: list[dict[str, object]] = []
    i = 0
    horizon_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    while i < len(market) - horizon_bars:
        ts = market.index[i]
        score_ok = True
        if scores is not None and min_open_score is not None:
            score_ok = pd.notna(scores.loc[ts]) and float(scores.loc[ts]) >= min_open_score
        if not bool(allow_open.loc[ts]) or not score_ok:
            i += 1
            continue

        result: GridSimulationResult = simulate_grid_from_index(
            market,
            i,
            risk,
            constant_size=constant_size,
            score_series=score_positions,
            kill_switch_threshold=kill_switch_threshold,
            add_level_min_score=add_level_min_score,
        )
        rows.append(result.to_dict())
        i = market.index.get_loc(result.exit_timestamp) + 1

    trades = pd.DataFrame(rows)
    if trades.empty:
        trades = pd.DataFrame(columns=list(GridSimulationResult.__dataclass_fields__.keys()))
    equity = _make_equity_curve(market.index, trades)
    return BacktestResult(trades=trades, equity_curve=equity)

