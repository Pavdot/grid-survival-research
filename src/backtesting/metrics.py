from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    return equity / peak - 1.0


def calculate_metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, float]:
    equity = equity.dropna()
    if equity.empty:
        return {}

    returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if equity.iloc[0] else 0.0
    days = max((equity.index[-1] - equity.index[0]) / pd.Timedelta(days=1), 1 / 24)
    annualized_return = (1 + total_return) ** (365.25 / days) - 1 if total_return > -1 else -1.0
    ann_vol = returns.std() * np.sqrt(365.25 * 24 * 12) if len(returns) else 0.0
    downside = returns[returns < 0].std() * np.sqrt(365.25 * 24 * 12) if len(returns) else 0.0
    max_dd = drawdown_series(equity).min()

    metrics = {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(max_dd),
        "sharpe": float(annualized_return / ann_vol) if ann_vol and np.isfinite(ann_vol) else 0.0,
        "sortino": float(annualized_return / downside) if downside and np.isfinite(downside) else 0.0,
        "calmar": float(annualized_return / abs(max_dd)) if max_dd < 0 else 0.0,
    }

    if trades is not None and not trades.empty:
        pnl = trades["realized_pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        metrics.update(
            {
                "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf"),
                "expectancy": float(pnl.mean()),
                "number_of_grids": int(len(trades)),
                "number_of_trades": int(trades["number_of_levels_filled"].sum()),
                "winrate": float((pnl > 0).mean()),
                "average_win": float(wins.mean()) if not wins.empty else 0.0,
                "average_loss": float(losses.mean()) if not losses.empty else 0.0,
                "average_holding_time": float(trades["time_to_exit"].mean()),
                "worst_trade": float(pnl.min()),
                "worst_grid": float(pnl.min()),
                "max_levels_reached": int(trades["number_of_levels_filled"].max()),
                "average_levels_reached": float(trades["number_of_levels_filled"].mean()),
                "fees_paid": float(trades["fees_paid"].sum()),
                "slippage_paid": float(trades["slippage_paid"].sum()),
                "exposure_time": float(trades["time_to_exit"].sum()),
                "realized_pnl": float(pnl.sum()),
                "max_unrealized_drawdown": float(trades["unrealized_drawdown_max"].max()),
                "ratio_realized_unrealized_risk": float(pnl.sum() / trades["unrealized_drawdown_max"].sum())
                if trades["unrealized_drawdown_max"].sum() != 0
                else 0.0,
                "number_of_forced_exits": int(
                    trades[
                        [
                            "stopped_by_max_loss",
                            "stopped_by_max_holding",
                            "stopped_by_volatility_shock",
                            "stopped_by_exposure",
                            "stopped_by_kill_switch",
                        ]
                    ].max(axis=1).sum()
                ),
                "number_of_regime_exits": int(trades["stopped_by_regime_break"].sum()),
                "number_of_grids_stopped_by_max_loss": int(trades["stopped_by_max_loss"].sum()),
            }
        )
    return metrics

