from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.labeling.grid_risk import GridRiskConfig, default_level_sizes


@dataclass
class GridSimulationResult:
    start_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    side: str
    grid_survived: int
    max_adverse_excursion: float
    max_favorable_excursion: float
    realized_pnl: float
    unrealized_drawdown_max: float
    time_to_exit: float
    number_of_levels_filled: int
    stopped_by_regime_break: int
    stopped_by_max_loss: int
    stopped_by_max_holding: int
    stopped_by_volatility_shock: int
    stopped_by_exposure: int
    stopped_by_kill_switch: int
    exit_reason: str
    fees_paid: float
    slippage_paid: float
    max_exposure_pct: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _safe_spacing(entry_price: float, atr_value: float, risk: GridRiskConfig) -> float:
    if np.isfinite(atr_value) and atr_value > 0:
        return float(atr_value) * risk.spacing_atr_multiplier
    return entry_price * 0.001


def _spread_from_row(row: pd.Series, risk: GridRiskConfig) -> float:
    if risk.cost_model != "bid_ask_spread":
        return 0.0
    candidates = [
        row.get("spread_close", np.nan),
        row.get("spread_avg", np.nan),
    ]
    if pd.notna(row.get("ask_close", np.nan)) and pd.notna(row.get("bid_close", np.nan)):
        candidates.append(float(row["ask_close"]) - float(row["bid_close"]))
    for value in candidates:
        spread = float(value)
        if np.isfinite(spread) and spread > 0:
            return spread
    raise ValueError("bid_ask_spread cost_model requires positive spread columns")


def _execution_price(mid_price: float, row: pd.Series, risk: GridRiskConfig, side: str, action: str) -> float:
    if action not in {"entry", "exit"}:
        raise ValueError("action must be entry or exit")
    spread = _spread_from_row(row, risk)
    is_buy = (side == "long" and action == "entry") or (side == "short" and action == "exit")
    price = mid_price + spread / 2 if is_buy else mid_price - spread / 2
    if is_buy:
        return price * (1 + risk.slippage_pct)
    return price * (1 - risk.slippage_pct)


def _unrealized_pnl_pct(
    fills: list[tuple[float, float]],
    mark_price: float,
    exit_fee: float,
    side: str = "long",
) -> float:
    if side == "short":
        gross = sum(qty * (fill_price - mark_price) for fill_price, qty in fills)
    else:
        gross = sum(qty * (mark_price - fill_price) for fill_price, qty in fills)
    exit_notional = sum(qty * mark_price for _, qty in fills)
    return gross - exit_notional * exit_fee


def _realize(
    fills: list[tuple[float, float]],
    exit_price: float,
    taker_fee: float,
    entry_fees: float,
    side: str = "long",
) -> tuple[float, float]:
    if side == "short":
        gross = sum(qty * (fill_price - exit_price) for fill_price, qty in fills)
    else:
        gross = sum(qty * (exit_price - fill_price) for fill_price, qty in fills)
    exit_fee_paid = sum(qty * exit_price for _, qty in fills) * taker_fee
    return gross - entry_fees - exit_fee_paid, entry_fees + exit_fee_paid


def simulate_grid_from_index(
    market: pd.DataFrame,
    start_pos: int,
    risk: GridRiskConfig,
    constant_size: bool = False,
    score_series: pd.Series | None = None,
    kill_switch_threshold: float | None = None,
    add_level_min_score: float | None = None,
    take_profit_spacing_multiplier: float = 0.5,
    survival_min_realized_pnl: float | None = None,
    side: str = "long",
    blackout_series: pd.Series | None = None,
) -> GridSimulationResult:
    if start_pos < 0 or start_pos >= len(market) - 1:
        raise IndexError("start_pos must leave at least one future candle")
    if take_profit_spacing_multiplier <= 0:
        raise ValueError("take_profit_spacing_multiplier must be positive")
    if side not in {"long", "short"}:
        raise ValueError("side must be either 'long' or 'short'")
    if blackout_series is not None and len(blackout_series) != len(market):
        raise ValueError("blackout_series must be aligned to market length")

    sizes = default_level_sizes(risk, constant=constant_size)
    start_row = market.iloc[start_pos]
    start_ts = market.index[start_pos]
    entry_reference = float(start_row["close"])
    spacing = _safe_spacing(entry_reference, float(start_row.get("atr_5m", np.nan)), risk)
    max_bars = max(1, int(risk.max_holding_hours * 60 / 5))

    fills: list[tuple[float, float]] = []
    entry_fees = 0.0
    slippage_paid = 0.0
    exposure_pct = 0.0
    max_exposure = 0.0
    next_level = 0
    max_adverse = 0.0
    max_favorable = 0.0
    exit_reason = "end_of_data"
    stopped_by_regime = 0
    stopped_by_max_loss = 0
    stopped_by_holding = 0
    stopped_by_vol = 0
    stopped_by_exposure = 0
    stopped_by_kill = 0

    def add_fill(level_price: float, fill_row: pd.Series, size_pct: float) -> None:
        nonlocal exposure_pct, entry_fees, slippage_paid
        fill_price = _execution_price(level_price, fill_row, risk, side, "entry")
        qty = size_pct / fill_price
        fills.append((fill_price, qty))
        exposure_pct += size_pct
        entry_fees += size_pct * risk.taker_fee
        slippage_paid += abs(fill_price - level_price) * qty

    add_fill(entry_reference, start_row, sizes[0])
    next_level = 1
    max_exposure = exposure_pct

    exit_price = _execution_price(entry_reference, start_row, risk, side, "exit")
    exit_pos = start_pos

    for pos in range(start_pos + 1, min(len(market), start_pos + max_bars + 1)):
        row = market.iloc[pos]
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        exit_pos = pos
        mark_price = _execution_price(close, row, risk, side, "exit")

        if blackout_series is not None and bool(blackout_series.iloc[pos]):
            exit_reason = "fundamental_blackout"
            exit_price = mark_price
            break

        while next_level < risk.max_levels:
            level_price = entry_reference - spacing * next_level if side == "long" else entry_reference + spacing * next_level
            level_reached = low <= level_price if side == "long" else high >= level_price
            if not level_reached:
                break
            if (
                score_series is not None
                and add_level_min_score is not None
                and pos < len(score_series)
                and pd.notna(score_series.iloc[pos])
                and float(score_series.iloc[pos]) < add_level_min_score
            ):
                break
            proposed_exposure = exposure_pct + sizes[next_level]
            if proposed_exposure > risk.max_total_exposure_pct + 1e-12:
                stopped_by_exposure = 1
                exit_reason = "max_exposure"
                exit_price = mark_price
                break
            add_fill(level_price, row, sizes[next_level])
            next_level += 1
            max_exposure = max(max_exposure, exposure_pct)
        if stopped_by_exposure:
            break

        unrealized = _unrealized_pnl_pct(fills, mark_price, risk.taker_fee, side=side) - entry_fees
        max_adverse = max(max_adverse, max(0.0, -unrealized))
        max_favorable = max(max_favorable, max(0.0, unrealized))

        if unrealized <= -risk.max_grid_loss_pct:
            stopped_by_max_loss = 1
            exit_reason = "max_loss"
            exit_price = mark_price
            break

        avg_entry = sum(fill_price * qty for fill_price, qty in fills) / sum(qty for _, qty in fills)
        take_profit = (
            avg_entry + spacing * take_profit_spacing_multiplier
            if side == "long"
            else avg_entry - spacing * take_profit_spacing_multiplier
        )
        take_profit_reached = high >= take_profit if side == "long" else low <= take_profit
        if take_profit_reached:
            exit_reason = "take_profit"
            exit_price = _execution_price(take_profit, row, risk, side, "exit")
            break

        if risk.stop_on_regime_break and bool(row.get("breakout_risk", 0)):
            stopped_by_regime = 1
            exit_reason = "regime_break"
            exit_price = mark_price
            break

        if risk.stop_on_volatility_shock and bool(row.get("volatility_shock", 0)):
            stopped_by_vol = 1
            exit_reason = "volatility_shock"
            exit_price = mark_price
            break

        if (
            score_series is not None
            and kill_switch_threshold is not None
            and pos < len(score_series)
            and pd.notna(score_series.iloc[pos])
            and float(score_series.iloc[pos]) < kill_switch_threshold
        ):
            stopped_by_kill = 1
            exit_reason = "kill_switch"
            exit_price = mark_price
            break
    else:
        stopped_by_holding = 1
        exit_reason = "max_holding"
        exit_row = market.iloc[exit_pos]
        exit_price = _execution_price(float(exit_row["close"]), exit_row, risk, side, "exit")

    realized_pnl, fees_paid = _realize(fills, exit_price, risk.taker_fee, entry_fees, side=side)
    if exit_reason == "take_profit":
        min_success_pnl = -risk.max_grid_loss_pct if survival_min_realized_pnl is None else survival_min_realized_pnl
        survived = int(realized_pnl >= min_success_pnl)
    else:
        survived = 0

    exit_ts = market.index[exit_pos]
    hours = (exit_ts - start_ts) / pd.Timedelta(hours=1)
    return GridSimulationResult(
        start_timestamp=start_ts,
        exit_timestamp=exit_ts,
        side=side,
        grid_survived=survived,
        max_adverse_excursion=float(max_adverse),
        max_favorable_excursion=float(max_favorable),
        realized_pnl=float(realized_pnl),
        unrealized_drawdown_max=float(max_adverse),
        time_to_exit=float(hours),
        number_of_levels_filled=len(fills),
        stopped_by_regime_break=stopped_by_regime,
        stopped_by_max_loss=stopped_by_max_loss,
        stopped_by_max_holding=stopped_by_holding,
        stopped_by_volatility_shock=stopped_by_vol,
        stopped_by_exposure=stopped_by_exposure,
        stopped_by_kill_switch=stopped_by_kill,
        exit_reason=exit_reason,
        fees_paid=float(fees_paid),
        slippage_paid=float(slippage_paid),
        max_exposure_pct=float(max_exposure),
    )
