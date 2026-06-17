from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.backtesting.metrics import calculate_metrics, drawdown_series
from src.data.validate_data import load_processed
from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_engine import (
    GridSimulationResult,
    _execution_price,
    _realize,
    _safe_spacing,
    _unrealized_pnl_pct,
)
from src.labeling.grid_risk import GridRiskConfig, default_level_sizes, validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market, summarize_simulations
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.fundamental_trend_escape_martingale_research import build_variant_masks
from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    build_side_signal,
    candidate_from_row,
    monthly_return_from_equity,
    planned_exposure,
    risk_for_candidate,
)
from src.research.walk_forward_martingale_research import make_walk_forward_windows, split_frame_from_index, stitch_oos_equity
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_execution_surface_017.yaml"
LOCKED_VARIANT = "fundamental_trend_escape_entry_only"
FORCED_EXIT_REASONS = {
    "fundamental_blackout",
    "max_exposure",
    "max_loss",
    "max_holding",
    "regime_break",
    "volatility_shock",
    "kill_switch",
    "range_break_emergency",
}


@dataclass(frozen=True)
class ExecutionSurfaceCell:
    fee_rate: float
    slippage_bps: float
    fill_rate: float

    @property
    def key(self) -> tuple[float, float, float]:
        return (float(self.fee_rate), float(self.slippage_bps), float(self.fill_rate))


@dataclass
class FillStats:
    initial_fill_count: int = 0
    add_fill_attempts: int = 0
    add_fill_misses: int = 0
    tp_fill_attempts: int = 0
    tp_fill_misses: int = 0
    forced_exit_count: int = 0

    @property
    def add_miss_rate(self) -> float:
        return self.add_fill_misses / self.add_fill_attempts if self.add_fill_attempts else 0.0

    @property
    def tp_miss_rate(self) -> float:
        return self.tp_fill_misses / self.tp_fill_attempts if self.tp_fill_attempts else 0.0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "initial_fill_count": int(self.initial_fill_count),
            "add_fill_attempts": int(self.add_fill_attempts),
            "add_fill_misses": int(self.add_fill_misses),
            "tp_fill_attempts": int(self.tp_fill_attempts),
            "tp_fill_misses": int(self.tp_fill_misses),
            "forced_exit_count": int(self.forced_exit_count),
            "add_miss_rate": float(self.add_miss_rate),
            "tp_miss_rate": float(self.tp_miss_rate),
        }

    def add(self, other: "FillStats") -> None:
        self.initial_fill_count += other.initial_fill_count
        self.add_fill_attempts += other.add_fill_attempts
        self.add_fill_misses += other.add_fill_misses
        self.tp_fill_attempts += other.tp_fill_attempts
        self.tp_fill_misses += other.tp_fill_misses
        self.forced_exit_count += other.forced_exit_count


@dataclass
class FillRateBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    fill_stats: FillStats


@dataclass(frozen=True)
class FastMarket:
    index: pd.Index
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr_5m: np.ndarray
    breakout_risk: np.ndarray
    volatility_shock: np.ndarray


def validate_surface_config(config: dict[str, Any]) -> None:
    surface = config["surface"]
    fee_rates = [float(value) for value in surface["fee_rates"]]
    slippage_values = [float(value) for value in surface["slippage_bps"]]
    fill_rates = [float(value) for value in surface["fill_rates"]]
    seeds = int(surface["seeds"])
    if not fee_rates or any(value < 0 for value in fee_rates):
        raise ValueError("surface.fee_rates must contain non-negative values")
    if not slippage_values or any(value < 0 for value in slippage_values):
        raise ValueError("surface.slippage_bps must contain non-negative values")
    if not fill_rates or any(value < 0 or value > 1 for value in fill_rates):
        raise ValueError("surface.fill_rates must stay within [0, 1]")
    if seeds <= 0:
        raise ValueError("surface.seeds must be positive")


def make_surface_cells(config: dict[str, Any], max_cells: int | None = None) -> list[ExecutionSurfaceCell]:
    validate_surface_config(config)
    cells = [
        ExecutionSurfaceCell(float(fee), float(slippage), float(fill))
        for fee in config["surface"]["fee_rates"]
        for slippage in config["surface"]["slippage_bps"]
        for fill in config["surface"]["fill_rates"]
    ]
    if max_cells is not None:
        if max_cells <= 0:
            raise ValueError("max_cells must be positive")
        cells = cells[: int(max_cells)]
    return cells


def seed_values(config: dict[str, Any], seeds_override: int | None = None) -> list[int]:
    count = int(seeds_override if seeds_override is not None else config["surface"]["seeds"])
    if count <= 0:
        raise ValueError("seeds must be positive")
    base = int(config["surface"].get("seed_base", 0))
    return [base + offset for offset in range(count)]


def load_selected_017(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "walk_forward_selected_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Iteration 017 selected candidates not found: {path}")
    frame = pd.read_csv(path)
    missing = {"variant", "fold_id", "name"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def locked_candidate_for_fold(
    selected: pd.DataFrame,
    fold_id: int,
    cell: ExecutionSurfaceCell,
    variant: str = LOCKED_VARIANT,
) -> MonthlyMartingaleCandidate:
    rows = selected[selected["variant"].astype(str).eq(variant) & selected["fold_id"].astype(int).eq(int(fold_id))]
    if rows.empty:
        raise ValueError(f"No locked iteration 017 candidate for {variant} fold {fold_id}")
    candidate = candidate_from_row(rows.iloc[0].to_dict())
    return replace(
        candidate,
        name=f"{candidate.name}__fee{cell.fee_rate:g}_slip{cell.slippage_bps:g}_fill{cell.fill_rate:g}",
        fee_rate=float(cell.fee_rate),
        slippage_bps=float(cell.slippage_bps),
    )


def _bernoulli_pass(fill_rate: float, rng: np.random.Generator) -> bool:
    if fill_rate >= 1.0:
        return True
    if fill_rate <= 0.0:
        return False
    return bool(rng.random() < fill_rate)


def _to_fast_market(market: pd.DataFrame) -> FastMarket:
    return FastMarket(
        index=market.index,
        high=market["high"].to_numpy(dtype=float),
        low=market["low"].to_numpy(dtype=float),
        close=market["close"].to_numpy(dtype=float),
        atr_5m=market.get("atr_5m", pd.Series(np.nan, index=market.index)).to_numpy(dtype=float),
        breakout_risk=market.get("breakout_risk", pd.Series(0, index=market.index)).to_numpy(dtype=bool),
        volatility_shock=market.get("volatility_shock", pd.Series(0, index=market.index)).to_numpy(dtype=bool),
    )


def _execution_price_fast(mid_price: float, risk: GridRiskConfig, side: str, action: str) -> float:
    is_buy = (side == "long" and action == "entry") or (side == "short" and action == "exit")
    if is_buy:
        return mid_price * (1 + risk.slippage_pct)
    return mid_price * (1 - risk.slippage_pct)


def simulate_grid_arrays_with_fill_rate(
    fast: FastMarket,
    start_pos: int,
    risk: GridRiskConfig,
    side: str,
    take_profit_spacing_multiplier: float,
    fill_rate: float,
    rng: np.random.Generator,
    survival_min_realized_pnl: float | None = 0.0,
) -> tuple[GridSimulationResult, FillStats]:
    if risk.cost_model != "bps":
        raise ValueError("fast execution surface replay only supports bps cost_model")
    if start_pos < 0 or start_pos >= len(fast.index) - 1:
        raise IndexError("start_pos must leave at least one future candle")
    if fill_rate < 0 or fill_rate > 1:
        raise ValueError("fill_rate must stay within [0, 1]")
    if side not in {"long", "short"}:
        raise ValueError("side must be either 'long' or 'short'")

    sizes = default_level_sizes(risk)
    start_ts = fast.index[start_pos]
    entry_reference = float(fast.close[start_pos])
    atr_value = float(fast.atr_5m[start_pos])
    spacing = float(atr_value) * risk.spacing_atr_multiplier if np.isfinite(atr_value) and atr_value > 0 else entry_reference * 0.001
    max_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    stats = FillStats(initial_fill_count=1)

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

    def add_fill(level_price: float, size_pct: float) -> None:
        nonlocal exposure_pct, entry_fees, slippage_paid
        fill_price = _execution_price_fast(level_price, risk, side, "entry")
        qty = size_pct / fill_price
        fills.append((fill_price, qty))
        exposure_pct += size_pct
        entry_fees += size_pct * risk.taker_fee
        slippage_paid += abs(fill_price - level_price) * qty

    add_fill(entry_reference, sizes[0])
    next_level = 1
    max_exposure = exposure_pct
    exit_price = _execution_price_fast(entry_reference, risk, side, "exit")
    exit_pos = start_pos

    for pos in range(start_pos + 1, min(len(fast.index), start_pos + max_bars + 1)):
        low = float(fast.low[pos])
        high = float(fast.high[pos])
        close = float(fast.close[pos])
        exit_pos = pos
        mark_price = _execution_price_fast(close, risk, side, "exit")

        while next_level < risk.max_levels:
            level_price = entry_reference - spacing * next_level if side == "long" else entry_reference + spacing * next_level
            level_reached = low <= level_price if side == "long" else high >= level_price
            if not level_reached:
                break
            proposed_exposure = exposure_pct + sizes[next_level]
            if proposed_exposure > risk.max_total_exposure_pct + 1e-12:
                stopped_by_exposure = 1
                exit_reason = "max_exposure"
                exit_price = mark_price
                break
            stats.add_fill_attempts += 1
            if not _bernoulli_pass(fill_rate, rng):
                stats.add_fill_misses += 1
                break
            add_fill(level_price, sizes[next_level])
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
            stats.tp_fill_attempts += 1
            if _bernoulli_pass(fill_rate, rng):
                exit_reason = "take_profit"
                exit_price = _execution_price_fast(take_profit, risk, side, "exit")
                break
            stats.tp_fill_misses += 1

        if risk.stop_on_regime_break and bool(fast.breakout_risk[pos]):
            stopped_by_regime = 1
            exit_reason = "regime_break"
            exit_price = mark_price
            break

        if risk.stop_on_volatility_shock and bool(fast.volatility_shock[pos]):
            stopped_by_vol = 1
            exit_reason = "volatility_shock"
            exit_price = mark_price
            break
    else:
        stopped_by_holding = 1
        exit_reason = "max_holding"
        exit_price = _execution_price_fast(float(fast.close[exit_pos]), risk, side, "exit")

    realized_pnl, fees_paid = _realize(fills, exit_price, risk.taker_fee, entry_fees, side=side)
    if exit_reason == "take_profit":
        min_success_pnl = -risk.max_grid_loss_pct if survival_min_realized_pnl is None else survival_min_realized_pnl
        survived = int(realized_pnl >= min_success_pnl)
    else:
        survived = 0
    if exit_reason in FORCED_EXIT_REASONS:
        stats.forced_exit_count += 1

    exit_ts = fast.index[exit_pos]
    hours = (exit_ts - start_ts) / pd.Timedelta(hours=1)
    return (
        GridSimulationResult(
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
        ),
        stats,
    )


def simulate_grid_with_fill_rate(
    market: pd.DataFrame,
    start_pos: int,
    risk: GridRiskConfig,
    side: str,
    take_profit_spacing_multiplier: float,
    fill_rate: float,
    rng: np.random.Generator,
    survival_min_realized_pnl: float | None = 0.0,
    blackout_series: pd.Series | None = None,
    blackout_exit_reason: str = "fundamental_blackout",
) -> tuple[GridSimulationResult, FillStats]:
    if start_pos < 0 or start_pos >= len(market) - 1:
        raise IndexError("start_pos must leave at least one future candle")
    if take_profit_spacing_multiplier <= 0:
        raise ValueError("take_profit_spacing_multiplier must be positive")
    if fill_rate < 0 or fill_rate > 1:
        raise ValueError("fill_rate must stay within [0, 1]")
    if side not in {"long", "short"}:
        raise ValueError("side must be either 'long' or 'short'")
    if blackout_series is not None and len(blackout_series) != len(market):
        raise ValueError("blackout_series must be aligned to market length")

    sizes = default_level_sizes(risk)
    start_row = market.iloc[start_pos]
    start_ts = market.index[start_pos]
    entry_reference = float(start_row["close"])
    spacing = _safe_spacing(entry_reference, float(start_row.get("atr_5m", np.nan)), risk)
    max_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    stats = FillStats(initial_fill_count=1)

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
            exit_reason = str(blackout_exit_reason)
            exit_price = mark_price
            break

        while next_level < risk.max_levels:
            level_price = entry_reference - spacing * next_level if side == "long" else entry_reference + spacing * next_level
            level_reached = low <= level_price if side == "long" else high >= level_price
            if not level_reached:
                break
            proposed_exposure = exposure_pct + sizes[next_level]
            if proposed_exposure > risk.max_total_exposure_pct + 1e-12:
                stopped_by_exposure = 1
                exit_reason = "max_exposure"
                exit_price = mark_price
                break
            stats.add_fill_attempts += 1
            if not _bernoulli_pass(fill_rate, rng):
                stats.add_fill_misses += 1
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
            stats.tp_fill_attempts += 1
            if _bernoulli_pass(fill_rate, rng):
                exit_reason = "take_profit"
                exit_price = _execution_price(take_profit, row, risk, side, "exit")
                break
            stats.tp_fill_misses += 1

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
    if exit_reason in FORCED_EXIT_REASONS:
        stats.forced_exit_count += 1

    exit_ts = market.index[exit_pos]
    hours = (exit_ts - start_ts) / pd.Timedelta(hours=1)
    return (
        GridSimulationResult(
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
        ),
        stats,
    )


def make_equity_curve(index: pd.Index, trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        equity = pd.Series(1.0, index=index, dtype=float)
        equity.name = "equity"
        return equity
    increments = pd.Series(0.0, index=index, dtype=float)
    exits = pd.to_datetime(trades["exit_timestamp"], utc=True)
    pnls = trades["realized_pnl"].astype(float).to_numpy()
    positions = index.get_indexer(exits)
    for pos, pnl in zip(positions, pnls):
        if pos >= 0:
            increments.iloc[int(pos)] += float(pnl)
    equity = 1.0 + increments.cumsum()
    equity.name = "equity"
    return equity


def _signal_cache_key(candidate: MonthlyMartingaleCandidate) -> tuple[Any, ...]:
    return (
        candidate.side_mode,
        candidate.entry_mode,
        candidate.rsi_window,
        candidate.rsi_low,
        candidate.rsi_high,
    )


def _side_codes(side_signal: pd.Series, index: pd.Index) -> np.ndarray:
    values = side_signal.reindex(index)
    codes = np.zeros(len(index), dtype=np.int8)
    codes[values.eq("long").to_numpy()] = 1
    codes[values.eq("short").to_numpy()] = -1
    return codes


def run_signal_grid_backtest_fill_rate_fast(
    market: pd.DataFrame,
    risk: GridRiskConfig,
    side_signal: pd.Series,
    candidate: MonthlyMartingaleCandidate,
    fill_rate: float,
    seed: int,
    entry_blackout_series: pd.Series | None = None,
) -> FillRateBacktestResult:
    fast = _to_fast_market(market)
    rng = np.random.default_rng(int(seed))
    codes = _side_codes(side_signal, market.index)
    entry_mask = (
        np.zeros(len(market), dtype=bool)
        if entry_blackout_series is None
        else entry_blackout_series.reindex(market.index).fillna(False).to_numpy(dtype=bool)
    )
    rows: list[dict[str, Any]] = []
    total_stats = FillStats()
    i = 0
    horizon_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    while i < len(market) - horizon_bars:
        code = int(codes[i])
        if code == 0:
            i += 1
            continue
        if bool(entry_mask[i]):
            i += 1
            continue
        side = "long" if code == 1 else "short"
        result, stats = simulate_grid_arrays_with_fill_rate(
            fast,
            i,
            risk,
            side,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            fill_rate=fill_rate,
            rng=rng,
            survival_min_realized_pnl=0.0,
        )
        total_stats.add(stats)
        row = result.to_dict()
        row.update(asdict(candidate))
        row.update(stats.to_dict())
        row["fill_rate"] = float(fill_rate)
        row["fill_seed"] = int(seed)
        rows.append(row)
        exit_pos = int(market.index.get_loc(result.exit_timestamp))
        cooldown_until = result.exit_timestamp + pd.Timedelta(hours=candidate.entry_cooldown_hours)
        i = max(exit_pos + 1, int(market.index.searchsorted(cooldown_until)))
    trades = pd.DataFrame(rows)
    if trades.empty:
        trades = pd.DataFrame(columns=list(GridSimulationResult.__dataclass_fields__.keys()))
    equity = make_equity_curve(market.index, trades)
    return FillRateBacktestResult(trades=trades, equity_curve=equity, fill_stats=total_stats)


def run_signal_grid_backtest_fill_rate(
    market: pd.DataFrame,
    risk: GridRiskConfig,
    side_signal: pd.Series,
    candidate: MonthlyMartingaleCandidate,
    fill_rate: float,
    seed: int,
    entry_blackout_series: pd.Series | None = None,
    exit_blackout_series: pd.Series | None = None,
) -> FillRateBacktestResult:
    if risk.cost_model == "bps" and exit_blackout_series is None:
        return run_signal_grid_backtest_fill_rate_fast(
            market,
            risk,
            side_signal,
            candidate,
            fill_rate,
            seed,
            entry_blackout_series=entry_blackout_series,
        )
    rng = np.random.default_rng(int(seed))
    side_signal = side_signal.reindex(market.index)
    entry_blackout_series = (
        None if entry_blackout_series is None else entry_blackout_series.reindex(market.index).fillna(False).astype(bool)
    )
    exit_blackout_series = (
        None if exit_blackout_series is None else exit_blackout_series.reindex(market.index).fillna(False).astype(bool)
    )
    rows: list[dict[str, Any]] = []
    total_stats = FillStats()
    i = 0
    horizon_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    while i < len(market) - horizon_bars:
        side = side_signal.iloc[i]
        if side not in {"long", "short"}:
            i += 1
            continue
        if entry_blackout_series is not None and bool(entry_blackout_series.iloc[i]):
            i += 1
            continue
        result, stats = simulate_grid_with_fill_rate(
            market,
            i,
            risk,
            str(side),
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            fill_rate=fill_rate,
            rng=rng,
            survival_min_realized_pnl=0.0,
            blackout_series=exit_blackout_series,
        )
        total_stats.add(stats)
        row = result.to_dict()
        row.update(asdict(candidate))
        row.update(stats.to_dict())
        row["fill_rate"] = float(fill_rate)
        row["fill_seed"] = int(seed)
        rows.append(row)
        cooldown_until = result.exit_timestamp + pd.Timedelta(hours=candidate.entry_cooldown_hours)
        i = max(market.index.get_loc(result.exit_timestamp) + 1, int(market.index.searchsorted(cooldown_until)))
    trades = pd.DataFrame(rows)
    if trades.empty:
        trades = pd.DataFrame(columns=list(GridSimulationResult.__dataclass_fields__.keys()))
    equity = make_equity_curve(market.index, trades)
    return FillRateBacktestResult(trades=trades, equity_curve=equity, fill_stats=total_stats)


def summarize_fill_rate_result(
    equity: pd.Series,
    trades: pd.DataFrame,
    candidate: MonthlyMartingaleCandidate,
    split: str,
    fill_stats: FillStats,
) -> dict[str, Any]:
    metrics = calculate_metrics(equity, trades if not trades.empty else None)
    if not trades.empty:
        metrics.update(summarize_simulations(trades, baseline_grids=len(trades)))
    else:
        metrics.update(
            {
                "number_of_grids": 0,
                "expectancy": 0.0,
                "realized_pnl": 0.0,
                "profit_factor": 0.0,
                "number_of_forced_exits": 0,
                "fees_paid": 0.0,
                "slippage_paid": 0.0,
                "max_unrealized_drawdown": 0.0,
            }
        )
    metrics["monthly_return"] = monthly_return_from_equity(equity)
    metrics["min_equity"] = float(equity.min()) if not equity.empty else 1.0
    metrics.update({**asdict(candidate), "split": split, "planned_exposure": planned_exposure(candidate)})
    metrics.update(fill_stats.to_dict())
    return metrics


def run_locked_fold_for_cell(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: GridRiskConfig,
    selected_017: pd.DataFrame,
    fold_id: int,
    test_index: pd.Index,
    cell: ExecutionSurfaceCell,
    seed: int,
    entry_mask: pd.Series | None,
    split_cache: dict[int, tuple[pd.DataFrame, pd.Series | None]] | None = None,
    side_signal_cache: dict[tuple[Any, ...], pd.Series] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series, FillStats]:
    candidate = locked_candidate_for_fold(selected_017, fold_id, cell)
    if split_cache is not None and fold_id in split_cache:
        split_frame, split_entry = split_cache[fold_id]
    else:
        split_frame = split_frame_from_index(market, test_index)
        split_entry = None if entry_mask is None else entry_mask.reindex(split_frame.index).fillna(False).astype(bool)
        if split_cache is not None:
            split_cache[fold_id] = (split_frame, split_entry)
    risk = risk_for_candidate(base_risk, candidate)
    signal_key = _signal_cache_key(candidate)
    if side_signal_cache is not None and signal_key in side_signal_cache:
        side_signal = side_signal_cache[signal_key]
    else:
        side_signal = build_side_signal(market, signal_frame, candidate)
        if side_signal_cache is not None:
            side_signal_cache[signal_key] = side_signal
    result = run_signal_grid_backtest_fill_rate(
        split_frame,
        risk,
        side_signal,
        candidate,
        fill_rate=cell.fill_rate,
        seed=seed,
        entry_blackout_series=split_entry,
    )
    metrics = summarize_fill_rate_result(result.equity_curve, result.trades, candidate, "test", result.fill_stats)
    fold_row = {
        "fold_id": int(fold_id),
        "selected_name": candidate.name,
        "test_monthly_return": float(metrics["monthly_return"]),
        "test_total_return": float(metrics["total_return"]),
        "test_max_drawdown": float(metrics["max_drawdown"]),
        "test_profit_factor": float(metrics["profit_factor"]),
        "test_number_of_grids": int(metrics["number_of_grids"]),
        "test_positive": bool(float(metrics["total_return"]) > 0),
        "test_target_reached": False,
        "test_equity_ruined": bool(result.equity_curve.min() <= 0),
        "fees_paid": float(metrics["fees_paid"]),
        "slippage_paid": float(metrics["slippage_paid"]),
        **result.fill_stats.to_dict(),
    }
    return fold_row, result.trades, result.equity_curve, result.fill_stats


def summarize_surface_path(
    cell: ExecutionSurfaceCell,
    seed: int,
    fold_rows: list[dict[str, Any]],
    oos_equity: pd.Series,
    config: dict[str, Any],
    fill_stats: FillStats,
) -> dict[str, Any]:
    fold_summary = pd.DataFrame(fold_rows)
    target = float(config["target"]["monthly_return"])
    monthly = monthly_return_from_equity(oos_equity)
    total_return = float(oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) if float(oos_equity.iloc[0]) != 0 else -1.0
    max_dd = float(drawdown_series(oos_equity).min())
    return {
        "fee_rate": float(cell.fee_rate),
        "slippage_bps": float(cell.slippage_bps),
        "fill_rate": float(cell.fill_rate),
        "seed": int(seed),
        "fold_count": int(len(fold_summary)),
        "aggregate_monthly_return": float(monthly),
        "aggregate_total_return": float(total_return),
        "aggregate_max_drawdown": float(max_dd),
        "positive_fold_rate": float(fold_summary["test_positive"].mean()) if not fold_summary.empty else 0.0,
        "target_fold_rate": float((fold_summary["test_monthly_return"].astype(float) >= target).mean())
        if not fold_summary.empty
        else 0.0,
        "equity_ruined": bool((oos_equity <= 0).any() or fold_summary["test_equity_ruined"].any()),
        "test_grid_count": int(fold_summary["test_number_of_grids"].sum()) if not fold_summary.empty else 0,
        "fees_paid": float(fold_summary["fees_paid"].sum()) if not fold_summary.empty else 0.0,
        "slippage_paid": float(fold_summary["slippage_paid"].sum()) if not fold_summary.empty else 0.0,
        **fill_stats.to_dict(),
    }


def run_surface_path(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: GridRiskConfig,
    selected_017: pd.DataFrame,
    windows: list[Any],
    cell: ExecutionSurfaceCell,
    seed: int,
    entry_mask: pd.Series | None,
    config: dict[str, Any],
    split_cache: dict[int, tuple[pd.DataFrame, pd.Series | None]] | None = None,
    side_signal_cache: dict[tuple[Any, ...], pd.Series] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    fold_equities: list[tuple[int, pd.Series]] = []
    total_stats = FillStats()
    for window in windows:
        fold_row, _trades, equity, stats = run_locked_fold_for_cell(
            market,
            signal_frame,
            base_risk,
            selected_017,
            int(window.fold_id),
            window.test,
            cell,
            seed + int(window.fold_id) * 1_000_003,
            entry_mask,
            split_cache=split_cache,
            side_signal_cache=side_signal_cache,
        )
        fold_rows.append(fold_row)
        fold_equities.append((int(window.fold_id), equity))
        total_stats.add(stats)
    equity_frame = stitch_oos_equity(fold_equities)
    row = summarize_surface_path(cell, seed, fold_rows, equity_frame["equity"], config, total_stats)
    return row, equity_frame


def aggregate_surface_by_seed(by_seed: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if by_seed.empty:
        raise ValueError("execution surface by-seed frame is empty")
    target = float(config["target"]["monthly_return"])
    group_cols = ["fee_rate", "slippage_bps", "fill_rate"]
    rows: list[dict[str, Any]] = []
    for key, group in by_seed.groupby(group_cols, sort=True):
        monthly = group["aggregate_monthly_return"].astype(float)
        drawdown = group["aggregate_max_drawdown"].astype(float)
        rows.append(
            {
                "fee_rate": float(key[0]),
                "slippage_bps": float(key[1]),
                "fill_rate": float(key[2]),
                "seed_count": int(len(group)),
                "monthly_return_p10": float(monthly.quantile(0.10)),
                "monthly_return_median": float(monthly.median()),
                "monthly_return_p90": float(monthly.quantile(0.90)),
                "monthly_return_mean": float(monthly.mean()),
                "target_probability": float((monthly >= target).mean()),
                "positive_probability": float((monthly > 0).mean()),
                "ruin_probability": float(group["equity_ruined"].astype(bool).mean()),
                "max_drawdown_p10": float(drawdown.quantile(0.10)),
                "max_drawdown_median": float(drawdown.median()),
                "total_return_median": float(group["aggregate_total_return"].astype(float).median()),
                "positive_fold_rate_median": float(group["positive_fold_rate"].astype(float).median()),
                "target_fold_rate_median": float(group["target_fold_rate"].astype(float).median()),
                "grid_count_median": float(group["test_grid_count"].astype(float).median()),
                "fees_paid_median": float(group["fees_paid"].astype(float).median()),
                "slippage_paid_median": float(group["slippage_paid"].astype(float).median()),
                "add_miss_rate_mean": float(group["add_miss_rate"].astype(float).mean()),
                "tp_miss_rate_mean": float(group["tp_miss_rate"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_profitability_boundary(summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    robust_floor = float(config["target"]["robust_monthly_floor"])
    target = float(config["target"]["monthly_return"])
    for (fee, slippage), group in summary.groupby(["fee_rate", "slippage_bps"], sort=True):
        profitable = group[group["monthly_return_median"].astype(float) > 0]
        robust = group[group["monthly_return_median"].astype(float) >= robust_floor]
        target_rows = group[group["monthly_return_median"].astype(float) >= target]
        rows.append(
            {
                "fee_rate": float(fee),
                "slippage_bps": float(slippage),
                "best_monthly_return_median": float(group["monthly_return_median"].astype(float).max()),
                "best_target_probability": float(group["target_probability"].astype(float).max()),
                "min_fill_rate_positive": float(profitable["fill_rate"].min()) if not profitable.empty else np.nan,
                "min_fill_rate_ge_13pct": float(robust["fill_rate"].min()) if not robust.empty else np.nan,
                "min_fill_rate_ge_20pct": float(target_rows["fill_rate"].min()) if not target_rows.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["fee_rate", "slippage_bps"]).reset_index(drop=True)


def build_fill_miss_diagnostics(by_seed: pd.DataFrame) -> pd.DataFrame:
    cols = ["fee_rate", "slippage_bps", "fill_rate"]
    rows: list[dict[str, Any]] = []
    for key, group in by_seed.groupby(cols, sort=True):
        add_attempts = float(group["add_fill_attempts"].sum())
        tp_attempts = float(group["tp_fill_attempts"].sum())
        rows.append(
            {
                "fee_rate": float(key[0]),
                "slippage_bps": float(key[1]),
                "fill_rate": float(key[2]),
                "add_fill_attempts": int(add_attempts),
                "add_fill_misses": int(group["add_fill_misses"].sum()),
                "tp_fill_attempts": int(tp_attempts),
                "tp_fill_misses": int(group["tp_fill_misses"].sum()),
                "add_miss_rate": float(group["add_fill_misses"].sum() / add_attempts) if add_attempts else 0.0,
                "tp_miss_rate": float(group["tp_fill_misses"].sum() / tp_attempts) if tp_attempts else 0.0,
                "forced_exit_count": int(group["forced_exit_count"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(cols).reset_index(drop=True)


def _safe_float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write_heatmaps(summary: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old_file in list(figures_dir.glob("monthly_return_heatmaps_*.png")) + list(
        figures_dir.glob("target_probability_heatmaps_*.png")
    ):
        old_file.unlink()
    written: list[str] = []
    for fill_rate in sorted(summary["fill_rate"].astype(float).unique(), reverse=True):
        subset = summary[summary["fill_rate"].astype(float).eq(fill_rate)]
        for metric, prefix, title in [
            ("monthly_return_median", "monthly_return_heatmaps", "Median monthly return"),
            ("target_probability", "target_probability_heatmaps", "Target probability"),
        ]:
            pivot = subset.pivot(index="slippage_bps", columns="fee_rate", values=metric).sort_index(ascending=True)
            fig, ax = plt.subplots(figsize=(9, 5))
            image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
            ax.set_title(f"{title} | fill_rate={fill_rate:g}")
            ax.set_xlabel("fee_rate")
            ax.set_ylabel("slippage_bps")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{value:g}" for value in pivot.index])
            for y in range(len(pivot.index)):
                for x in range(len(pivot.columns)):
                    value = float(pivot.iloc[y, x])
                    label = f"{value:.1%}" if metric == "monthly_return_median" else f"{value:.0%}"
                    ax.text(x, y, label, ha="center", va="center", color="white" if abs(value) > 0.2 else "black", fontsize=8)
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            filename = f"{prefix}_{_safe_float_label(fill_rate)}.png"
            path = figures_dir / filename
            fig.savefig(path, dpi=140)
            plt.close(fig)
            written.append(str(path))
    return written


def decide_execution_edge(summary: pd.DataFrame, config: dict[str, Any]) -> str:
    verdict_config = config["verdict"]
    robust_zone = summary[
        (summary["fee_rate"].astype(float) <= float(verdict_config["robust_fee_max"]))
        & (summary["slippage_bps"].astype(float) <= float(verdict_config["robust_slippage_max"]))
        & (summary["fill_rate"].astype(float) >= float(verdict_config["robust_fill_min"]))
    ]
    target = float(config["target"]["monthly_return"])
    robust_floor = float(config["target"]["robust_monthly_floor"])
    target_probability_floor = float(config["target"]["robust_target_probability_floor"])
    if not robust_zone.empty:
        robust_pass = (
            (robust_zone["monthly_return_median"].astype(float) >= target)
            | (
                (robust_zone["monthly_return_median"].astype(float) >= robust_floor)
                & (robust_zone["target_probability"].astype(float) >= target_probability_floor)
            )
        )
        if bool(robust_pass.any()):
            return "execution edge robust enough"

    stress_negative = summary[
        (summary["slippage_bps"].astype(float) >= float(verdict_config["robust_slippage_max"]))
        | (summary["fill_rate"].astype(float) <= float(verdict_config["robust_fill_min"]))
    ]
    if not stress_negative.empty and bool((stress_negative["monthly_return_median"].astype(float) < 0).any()):
        return "execution edge too fragile"

    positive = summary[summary["monthly_return_median"].astype(float) > 0]
    if positive.empty:
        return "execution edge too fragile"
    narrow = positive[
        (positive["fee_rate"].astype(float) <= float(verdict_config["narrow_fee_max"]))
        & (positive["slippage_bps"].astype(float) <= float(verdict_config["narrow_slippage_max"]))
        & (positive["fill_rate"].astype(float) >= float(verdict_config["narrow_fill_min"]))
    ]
    if len(narrow) == len(positive):
        return "execution edge narrow"
    return "execution edge narrow"


def write_top_equity_curves(
    summary: pd.DataFrame,
    by_seed: pd.DataFrame,
    output_dir: Path,
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: GridRiskConfig,
    selected_017: pd.DataFrame,
    windows: list[Any],
    entry_mask: pd.Series | None,
    config: dict[str, Any],
    split_cache: dict[int, tuple[pd.DataFrame, pd.Series | None]] | None = None,
    side_signal_cache: dict[tuple[Any, ...], pd.Series] | None = None,
) -> list[str]:
    save_dir = output_dir / "oos_equity_best_surviving_cells"
    save_dir.mkdir(parents=True, exist_ok=True)
    for old_file in save_dir.glob("*.csv"):
        old_file.unlink()
    top_n = int(config["surface"].get("save_top_equity_cells", 5))
    eligible = summary[~summary["ruin_probability"].astype(float).ge(1.0)].copy()
    if eligible.empty:
        eligible = summary.copy()
    top = eligible.sort_values("monthly_return_median", ascending=False).head(top_n)
    written: list[str] = []
    for _, cell_row in top.iterrows():
        cell = ExecutionSurfaceCell(
            float(cell_row["fee_rate"]),
            float(cell_row["slippage_bps"]),
            float(cell_row["fill_rate"]),
        )
        candidates = by_seed[
            by_seed["fee_rate"].astype(float).eq(cell.fee_rate)
            & by_seed["slippage_bps"].astype(float).eq(cell.slippage_bps)
            & by_seed["fill_rate"].astype(float).eq(cell.fill_rate)
        ].copy()
        median = float(cell_row["monthly_return_median"])
        candidates["median_distance"] = (candidates["aggregate_monthly_return"].astype(float) - median).abs()
        seed = int(candidates.sort_values("median_distance").iloc[0]["seed"])
        _row, equity = run_surface_path(
            market,
            signal_frame,
            base_risk,
            selected_017,
            windows,
            cell,
            seed,
            entry_mask,
            config,
            split_cache=split_cache,
            side_signal_cache=side_signal_cache,
        )
        filename = (
            f"equity_fee{_safe_float_label(cell.fee_rate)}_slip{_safe_float_label(cell.slippage_bps)}"
            f"_fill{_safe_float_label(cell.fill_rate)}_seed{seed}.csv"
        )
        path = save_dir / filename
        equity.to_csv(path)
        written.append(str(path))
    return written


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    comparison = pd.DataFrame(payload["summary"])
    top = comparison.sort_values("monthly_return_median", ascending=False).head(12)
    control = payload.get("control_replay", {})
    lines = [
        "# Iteration 020 - Execution Surface 017",
        "",
        "## Verdict",
        f"`{payload['verdict']}`",
        "",
        "## Control Replay",
        "",
        markdown_table(pd.DataFrame([control])),
        "",
        "## Best Surface Cells",
        "",
        markdown_table(
            top[
                [
                    "fee_rate",
                    "slippage_bps",
                    "fill_rate",
                    "monthly_return_median",
                    "monthly_return_p10",
                    "monthly_return_p90",
                    "target_probability",
                    "positive_probability",
                    "ruin_probability",
                    "max_drawdown_median",
                ]
            ]
        ),
        "",
        "## Notes",
        "",
        "This iteration keeps the Iteration 017 candidates and folds locked. It does not re-select parameters. The first entry of each grid is always filled; add levels and take-profit exits are stochastic Bernoulli fills; forced exits always execute. Drawdown, ruin and fold rates are reported, but they are not optimization criteria.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(
    config_path: str = DEFAULT_CONFIG,
    max_cells: int | None = None,
    seeds: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_surface_config(config)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    source_config = load_yaml(config["source_iteration"]["config_path"])
    source_dir = project_path(config["source_iteration"]["output_dir"])
    selected_017 = load_selected_017(source_dir)
    base_risk = validate_strategy_config(load_strategy_config())
    market = prepare_market()
    signal_frame = load_processed(project_path("data/processed/btcusdt_1h.parquet"))
    trend_components = build_trend_escape_components(market, source_config)
    _events, _windows_frame, blackout_masks = build_blackout_bundle(market.index, source_config)
    entry_mask, _exit_mask, _reason = build_variant_masks(
        LOCKED_VARIANT,
        trend_components["trend_escape"].astype(bool),
        blackout_masks,
    )
    wf = config["walk_forward"]
    windows = make_walk_forward_windows(
        market.index,
        train_days=float(wf["train_days"]),
        test_days=float(wf["test_days"]),
        step_days=float(wf["step_days"]),
        embargo_bars=int(wf.get("embargo_bars", 0)),
    )
    cells = make_surface_cells(config, max_cells=max_cells)
    seeds_list = seed_values(config, seeds_override=seeds)
    rows: list[dict[str, Any]] = []
    deterministic_cache: dict[tuple[float, float], dict[str, Any]] = {}
    split_cache: dict[int, tuple[pd.DataFrame, pd.Series | None]] = {}
    side_signal_cache: dict[tuple[Any, ...], pd.Series] = {}
    total_paths = len(cells) * len(seeds_list)
    path_number = 0
    for cell in cells:
        cache_key = (cell.fee_rate, cell.slippage_bps)
        for seed in seeds_list:
            path_number += 1
            if cell.fill_rate >= 1.0 and cache_key in deterministic_cache:
                cached = dict(deterministic_cache[cache_key])
                cached["seed"] = int(seed)
                rows.append(cached)
                continue
            LOGGER.info(
                "Running execution surface path %s/%s fee=%s slippage=%s fill=%s seed=%s",
                path_number,
                total_paths,
                cell.fee_rate,
                cell.slippage_bps,
                cell.fill_rate,
                seed,
            )
            row, _equity = run_surface_path(
                market,
                signal_frame,
                base_risk,
                selected_017,
                windows,
                cell,
                seed,
                entry_mask,
                config,
                split_cache=split_cache,
                side_signal_cache=side_signal_cache,
            )
            rows.append(row)
            if cell.fill_rate >= 1.0:
                deterministic_cache[cache_key] = dict(row)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(output_dir / "execution_surface_by_seed.csv", index=False)
    summary = aggregate_surface_by_seed(by_seed, config)
    summary.to_csv(output_dir / "execution_surface_summary.csv", index=False)
    boundary = build_profitability_boundary(summary, config)
    boundary.to_csv(output_dir / "profitability_boundary.csv", index=False)
    diagnostics = build_fill_miss_diagnostics(by_seed)
    diagnostics.to_csv(output_dir / "fill_miss_diagnostics.csv", index=False)
    figures = write_heatmaps(summary, figures_dir)
    equity_files = write_top_equity_curves(
        summary,
        by_seed,
        output_dir,
        market,
        signal_frame,
        base_risk,
        selected_017,
        windows,
        entry_mask,
        config,
        split_cache=split_cache,
        side_signal_cache=side_signal_cache,
    )

    control_rows = by_seed[
        by_seed["fee_rate"].astype(float).eq(float(config["source_iteration"]["control_audited_monthly_return"] * 0 + 0.0001))
        & by_seed["slippage_bps"].astype(float).eq(0.0)
        & by_seed["fill_rate"].astype(float).eq(1.0)
    ]
    control_monthly = float(control_rows.iloc[0]["aggregate_monthly_return"]) if not control_rows.empty else np.nan
    control_expected = float(config["source_iteration"]["control_audited_monthly_return"])
    control_tolerance = float(config["source_iteration"].get("control_tolerance", 1e-6))
    control_replay = {
        "expected_017_monthly_return": control_expected,
        "replayed_monthly_return": control_monthly,
        "absolute_error": float(abs(control_monthly - control_expected)) if np.isfinite(control_monthly) else np.nan,
        "control_replay_ok": bool(np.isfinite(control_monthly) and abs(control_monthly - control_expected) <= control_tolerance),
    }
    verdict = decide_execution_edge(summary, config)
    payload = {
        "iteration_name": config["iteration"]["name"],
        "verdict": verdict,
        "cell_count": int(len(cells)),
        "seed_count": int(len(seeds_list)),
        "path_count": int(len(by_seed)),
        "control_replay": control_replay,
        "summary": summary.to_dict("records"),
        "boundary": boundary.to_dict("records"),
        "figure_files": figures,
        "equity_files": equity_files,
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, payload)
    LOGGER.info("Wrote execution surface outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Iteration 020 fee/slippage/fill-rate surface for locked 017.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_iteration(args.config, max_cells=args.max_cells, seeds=args.seeds)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
