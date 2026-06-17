from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import calculate_metrics, drawdown_series
from src.data.validate_data import load_processed
from src.fundamentals.event_blackout import (
    blackout_mask,
    build_blackout_bundle,
    build_blackout_windows,
    load_fundamental_events,
)
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
from src.research.fundamental_blackout_martingale_research import markdown_table, select_best_exact_no_drawdown
from src.research.fundamental_trend_escape_martingale_research import build_variant_masks
from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    build_side_signal,
    candidate_from_row,
    choose_candidate_subset,
    make_candidates,
    monthly_return_from_equity,
    planned_exposure,
    risk_for_candidate,
    sizing_sequence,
    summarize_exact,
    top_sample_candidates,
)
from src.research.walk_forward_martingale_research import (
    WalkForwardWindow,
    make_walk_forward_windows,
    same_candidate,
    split_frame_from_index,
    stitch_oos_equity,
)
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_execution_worst_case_audit.yaml"
SEARCH_SPLIT = "validation"
ENTRY_MODES = {"current_close", "next_bar_open", "next_bar_close"}


@dataclass(frozen=True)
class AuditScenario:
    name: str
    variant: str
    selection_policy: str
    entry_execution_mode: str = "current_close"
    conservative_intrabar: bool = False
    signal_lag_bars: int = 0
    mask_lag_bars: int = 0
    scheduled_macro_only: bool = False
    strict_event_after_known: bool = False
    sizing_scale: float = 1.0
    fee_rate: float | None = None
    slippage_bps: float | None = None

    @property
    def source_variant(self) -> str:
        return "baseline" if self.variant == "baseline" else "fundamental_trend_escape_entry_only"


@dataclass
class AuditGridBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series


def scenario_catalog(config: dict[str, Any]) -> dict[str, AuditScenario]:
    costs = config["audit"]["cost_scenarios"]
    percent_scale = float(config["audit"].get("percent_unit_scale", 0.01))
    strict = bool(config["audit"].get("strict_event_reaction_after_known_time", True))
    return {
        "control_baseline_replay": AuditScenario(
            name="control_baseline_replay",
            variant="baseline",
            selection_policy="locked_017",
        ),
        "control_fundamental_replay": AuditScenario(
            name="control_fundamental_replay",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
        ),
        "sizing_percent_corrected_locked": AuditScenario(
            name="sizing_percent_corrected_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            sizing_scale=percent_scale,
        ),
        "next_bar_open_locked": AuditScenario(
            name="next_bar_open_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            entry_execution_mode="next_bar_open",
        ),
        "next_bar_open_conservative_locked": AuditScenario(
            name="next_bar_open_conservative_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            entry_execution_mode="next_bar_open",
            conservative_intrabar=True,
        ),
        "signals_lagged_locked": AuditScenario(
            name="signals_lagged_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            signal_lag_bars=1,
            mask_lag_bars=1,
            strict_event_after_known=strict,
        ),
        "scheduled_macro_only_locked": AuditScenario(
            name="scheduled_macro_only_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            scheduled_macro_only=True,
            strict_event_after_known=strict,
        ),
        "fee_0p0004_slip_2_locked": AuditScenario(
            name="fee_0p0004_slip_2_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            fee_rate=float(costs["realistic"]["fee_rate"]),
            slippage_bps=float(costs["realistic"]["slippage_bps"]),
        ),
        "fee_0p0006_slip_5_locked": AuditScenario(
            name="fee_0p0006_slip_5_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            fee_rate=float(costs["conservative"]["fee_rate"]),
            slippage_bps=float(costs["conservative"]["slippage_bps"]),
        ),
        "fee_0p0010_slip_10_locked": AuditScenario(
            name="fee_0p0010_slip_10_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            fee_rate=float(costs["extreme"]["fee_rate"]),
            slippage_bps=float(costs["extreme"]["slippage_bps"]),
        ),
        "fee_0p0004_slip_2_reselect": AuditScenario(
            name="fee_0p0004_slip_2_reselect",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="validation_reselect",
            fee_rate=float(costs["realistic"]["fee_rate"]),
            slippage_bps=float(costs["realistic"]["slippage_bps"]),
        ),
        "fee_0p0006_slip_5_reselect": AuditScenario(
            name="fee_0p0006_slip_5_reselect",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="validation_reselect",
            fee_rate=float(costs["conservative"]["fee_rate"]),
            slippage_bps=float(costs["conservative"]["slippage_bps"]),
        ),
        "fee_0p0010_slip_10_reselect": AuditScenario(
            name="fee_0p0010_slip_10_reselect",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="validation_reselect",
            fee_rate=float(costs["extreme"]["fee_rate"]),
            slippage_bps=float(costs["extreme"]["slippage_bps"]),
        ),
        "worst_case_primary_locked": AuditScenario(
            name="worst_case_primary_locked",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="locked_017",
            entry_execution_mode="next_bar_open",
            conservative_intrabar=True,
            signal_lag_bars=1,
            mask_lag_bars=1,
            scheduled_macro_only=True,
            strict_event_after_known=strict,
            fee_rate=float(costs["conservative"]["fee_rate"]),
            slippage_bps=float(costs["conservative"]["slippage_bps"]),
        ),
        "worst_case_primary_reselect": AuditScenario(
            name="worst_case_primary_reselect",
            variant="fundamental_trend_escape_entry_only",
            selection_policy="validation_reselect",
            entry_execution_mode="next_bar_open",
            conservative_intrabar=True,
            signal_lag_bars=1,
            mask_lag_bars=1,
            scheduled_macro_only=True,
            strict_event_after_known=strict,
            fee_rate=float(costs["conservative"]["fee_rate"]),
            slippage_bps=float(costs["conservative"]["slippage_bps"]),
        ),
    }


def validate_scenarios(config: dict[str, Any]) -> list[AuditScenario]:
    catalog = scenario_catalog(config)
    requested = list(config["audit"]["scenario_order"])
    missing = sorted(set(requested) - set(catalog))
    if missing:
        raise ValueError(f"Unknown execution audit scenarios: {missing}")
    scenarios = [catalog[name] for name in requested]
    for scenario in scenarios:
        if scenario.selection_policy not in {"locked_017", "validation_reselect"}:
            raise ValueError(f"Unsupported selection policy: {scenario.selection_policy}")
        if scenario.entry_execution_mode not in ENTRY_MODES:
            raise ValueError(f"Unsupported entry execution mode: {scenario.entry_execution_mode}")
    return scenarios


def _lag_bool(mask: pd.Series | None, bars: int) -> pd.Series | None:
    if mask is None:
        return None
    if bars <= 0:
        return mask.astype(bool)
    return mask.astype(bool).shift(bars).fillna(False).astype(bool)


def _strict_after_mask(index: pd.Index, windows: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=index, dtype=bool)
    if windows.empty:
        mask.name = "blackout"
        return mask
    for _, window in windows.iterrows():
        start = pd.Timestamp(window["window_start_utc"]).tz_convert("UTC")
        end = pd.Timestamp(window["window_end_utc"]).tz_convert("UTC")
        mask |= (index > start) & (index <= end)
    mask.name = "blackout"
    return mask


def build_event_masks_for_scenario(
    index: pd.Index,
    config: dict[str, Any],
    scenario: AuditScenario,
) -> dict[str, pd.Series]:
    if not scenario.scheduled_macro_only and not scenario.strict_event_after_known:
        _events, _windows, masks = build_blackout_bundle(index, config)
        return masks
    events = load_fundamental_events(config)
    if scenario.scheduled_macro_only:
        events = events[events["is_scheduled"].astype(bool)].copy()
    blackout_config = config.get("fundamental_blackout", {})
    realistic_windows = build_blackout_windows(events, blackout_config, "realistic")
    oracle_windows = build_blackout_windows(events, blackout_config, "oracle")
    if scenario.strict_event_after_known:
        realistic_mask = _strict_after_mask(index, realistic_windows)
        oracle_mask = _strict_after_mask(index, oracle_windows)
    else:
        realistic_mask = blackout_mask(index, realistic_windows)
        oracle_mask = blackout_mask(index, oracle_windows)
    return {"realistic": realistic_mask, "oracle": oracle_mask}


def build_entry_mask_for_scenario(
    market: pd.DataFrame,
    config: dict[str, Any],
    trend_escape: pd.Series,
    scenario: AuditScenario,
) -> pd.Series | None:
    if scenario.variant == "baseline":
        return None
    masks = build_event_masks_for_scenario(market.index, config, scenario)
    entry_mask, _exit_mask, _reason = build_variant_masks(
        "fundamental_trend_escape_entry_only",
        trend_escape.astype(bool),
        masks,
    )
    return _lag_bool(entry_mask, scenario.mask_lag_bars)


def transform_candidate(candidate: MonthlyMartingaleCandidate, scenario: AuditScenario) -> MonthlyMartingaleCandidate:
    fee_rate = candidate.fee_rate if scenario.fee_rate is None else float(scenario.fee_rate)
    slippage_bps = candidate.slippage_bps if scenario.slippage_bps is None else float(scenario.slippage_bps)
    return replace(
        candidate,
        name=f"{candidate.name}__{scenario.name}",
        base_position_size_pct=float(candidate.base_position_size_pct) * float(scenario.sizing_scale),
        max_total_exposure_pct=float(candidate.max_total_exposure_pct) * float(scenario.sizing_scale),
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
    )


def scenario_config(base_config: dict[str, Any], scenario: AuditScenario) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    if scenario.fee_rate is not None:
        config["search"]["fee_rates"] = [float(scenario.fee_rate)]
    if scenario.slippage_bps is not None:
        config["search"]["slippage_bps_values"] = [float(scenario.slippage_bps)]
    if scenario.sizing_scale != 1.0:
        config["search"]["base_position_size_pcts"] = [
            float(value) * float(scenario.sizing_scale) for value in config["search"]["base_position_size_pcts"]
        ]
        config["search"]["max_total_exposure_pcts"] = [
            float(value) * float(scenario.sizing_scale) for value in config["search"]["max_total_exposure_pcts"]
        ]
        config["search"]["max_notional_exposure_pct"] = (
            float(config["search"]["max_notional_exposure_pct"]) * float(scenario.sizing_scale)
        )
    return config


def load_selected_017(source_dir: Path) -> pd.DataFrame:
    path = source_dir / "walk_forward_selected_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Iteration 017 selected candidates not found: {path}")
    frame = pd.read_csv(path)
    required = {"variant", "fold_id", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def selected_candidate_for_fold(selected: pd.DataFrame, scenario: AuditScenario, fold_id: int) -> MonthlyMartingaleCandidate:
    rows = selected[
        selected["variant"].astype(str).eq(scenario.source_variant) & selected["fold_id"].astype(int).eq(int(fold_id))
    ]
    if rows.empty:
        raise ValueError(f"No selected iteration 017 candidate for {scenario.source_variant} fold {fold_id}")
    return transform_candidate(candidate_from_row(rows.iloc[0].to_dict()), scenario)


def _entry_plan(signal_pos: int, market: pd.DataFrame, mode: str) -> tuple[int, str, int]:
    if mode == "current_close":
        return signal_pos, "close", signal_pos + 1
    if mode == "next_bar_open":
        return signal_pos + 1, "open", signal_pos + 2
    if mode == "next_bar_close":
        return signal_pos + 1, "close", signal_pos + 2
    raise ValueError(f"Unsupported entry execution mode: {mode}")


def simulate_grid_from_signal_pos(
    market: pd.DataFrame,
    signal_pos: int,
    risk: GridRiskConfig,
    side: str,
    scenario: AuditScenario,
    take_profit_spacing_multiplier: float,
    survival_min_realized_pnl: float | None = 0.0,
    exit_blackout_series: pd.Series | None = None,
) -> tuple[GridSimulationResult, dict[str, Any]]:
    if signal_pos < 0 or signal_pos >= len(market) - 1:
        raise IndexError("signal_pos must leave at least one future candle")
    if side not in {"long", "short"}:
        raise ValueError("side must be either long or short")
    entry_pos, entry_column, monitor_start = _entry_plan(signal_pos, market, scenario.entry_execution_mode)
    if monitor_start >= len(market):
        raise IndexError("entry execution mode leaves no future monitoring candle")
    if exit_blackout_series is not None and len(exit_blackout_series) != len(market):
        raise ValueError("exit_blackout_series must be aligned to market length")

    sizes = default_level_sizes(risk)
    signal_row = market.iloc[signal_pos]
    entry_row = market.iloc[entry_pos]
    signal_ts = market.index[signal_pos]
    entry_ts = market.index[entry_pos]
    entry_reference = float(entry_row[entry_column])
    spacing = _safe_spacing(entry_reference, float(signal_row.get("atr_5m", np.nan)), risk)
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
    ambiguous_tp_after_add = 0
    ambiguous_stop_and_tp = 0

    def add_fill(level_price: float, fill_row: pd.Series, size_pct: float) -> None:
        nonlocal exposure_pct, entry_fees, slippage_paid
        fill_price = _execution_price(level_price, fill_row, risk, side, "entry")
        qty = size_pct / fill_price
        fills.append((fill_price, qty))
        exposure_pct += size_pct
        entry_fees += size_pct * risk.taker_fee
        slippage_paid += abs(fill_price - level_price) * qty

    add_fill(entry_reference, entry_row, sizes[0])
    next_level = 1
    max_exposure = exposure_pct
    exit_price = _execution_price(entry_reference, entry_row, risk, side, "exit")
    exit_pos = entry_pos
    last_pos = min(len(market), entry_pos + max_bars + 1)

    for pos in range(monitor_start, last_pos):
        row = market.iloc[pos]
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        exit_pos = pos
        mark_price = _execution_price(close, row, risk, side, "exit")

        if exit_blackout_series is not None and bool(exit_blackout_series.iloc[pos]):
            exit_reason = "fundamental_blackout"
            exit_price = mark_price
            break

        added_this_bar = False
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
            add_fill(level_price, row, sizes[next_level])
            added_this_bar = True
            next_level += 1
            max_exposure = max(max_exposure, exposure_pct)
        if stopped_by_exposure:
            break

        unrealized = _unrealized_pnl_pct(fills, mark_price, risk.taker_fee, side=side) - entry_fees
        max_adverse = max(max_adverse, max(0.0, -unrealized))
        max_favorable = max(max_favorable, max(0.0, unrealized))

        avg_entry = sum(fill_price * qty for fill_price, qty in fills) / sum(qty for _, qty in fills)
        take_profit = (
            avg_entry + spacing * take_profit_spacing_multiplier
            if side == "long"
            else avg_entry - spacing * take_profit_spacing_multiplier
        )
        take_profit_reached = high >= take_profit if side == "long" else low <= take_profit

        if unrealized <= -risk.max_grid_loss_pct:
            stopped_by_max_loss = 1
            exit_reason = "max_loss"
            exit_price = mark_price
            if take_profit_reached:
                ambiguous_stop_and_tp += 1
            break

        if take_profit_reached:
            if scenario.conservative_intrabar and added_this_bar:
                ambiguous_tp_after_add += 1
            else:
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
    hours = (exit_ts - entry_ts) / pd.Timedelta(hours=1)
    result = GridSimulationResult(
        start_timestamp=entry_ts,
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
    extras = {
        "signal_timestamp": signal_ts,
        "entry_timestamp": entry_ts,
        "entry_execution_mode": scenario.entry_execution_mode,
        "entry_mid_price": float(entry_reference),
        "intrabar_ambiguous_tp_after_add": int(ambiguous_tp_after_add),
        "intrabar_ambiguous_stop_and_tp": int(ambiguous_stop_and_tp),
    }
    return result, extras


def make_equity_curve(index: pd.Index, trades: pd.DataFrame) -> pd.Series:
    equity = pd.Series(1.0, index=index, dtype=float)
    current_equity = 1.0
    if trades.empty:
        equity.name = "equity"
        return equity
    for _, trade in trades.sort_values("exit_timestamp").iterrows():
        current_equity += float(trade["realized_pnl"])
        exit_ts = pd.Timestamp(trade["exit_timestamp"])
        equity.loc[equity.index >= exit_ts] = current_equity
    equity.name = "equity"
    return equity


def sample_positions_for_scenario(
    market: pd.DataFrame,
    split_index: pd.Index,
    side_signal: pd.Series,
    risk: GridRiskConfig,
    cooldown_hours: float,
    stride_bars: int,
    max_positions: int,
    entry_mask: pd.Series | None,
    scenario: AuditScenario,
) -> list[int]:
    if split_index.empty:
        raise ValueError("split index is empty")
    if stride_bars <= 0:
        raise ValueError("search_entry_stride_bars must be positive")
    if max_positions <= 0:
        raise ValueError("max_sample_positions_per_candidate must be positive")
    split_start = int(market.index.searchsorted(split_index.min(), side="left"))
    split_end = int(market.index.searchsorted(split_index.max(), side="right") - 1)
    max_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    side_signal = side_signal.reindex(market.index)
    entry_mask = None if entry_mask is None else entry_mask.reindex(market.index).fillna(False).astype(bool)
    eligible = np.flatnonzero(side_signal.iloc[split_start : max(split_start, split_end - max_bars)].isin(["long", "short"]).to_numpy()) + split_start
    positions: list[int] = []
    cooldown_until = market.index[split_start]
    last_position = split_start - stride_bars
    for pos in eligible:
        if pos - last_position < stride_bars:
            continue
        if market.index[pos] < cooldown_until:
            continue
        try:
            entry_pos, _entry_column, monitor_start = _entry_plan(pos, market, scenario.entry_execution_mode)
        except ValueError:
            raise
        if monitor_start + max_bars > len(market):
            continue
        if entry_pos > split_end:
            continue
        if entry_mask is not None and bool(entry_mask.iloc[entry_pos]):
            continue
        if np.isfinite(float(market.iloc[pos].get("atr_5m", np.nan))):
            positions.append(pos)
            last_position = pos
            cooldown_until = market.index[pos] + pd.Timedelta(hours=max(risk.max_holding_hours, cooldown_hours))
        if len(positions) >= max_positions:
            break
    return positions


def run_signal_grid_backtest_audit(
    market: pd.DataFrame,
    risk: GridRiskConfig,
    side_signal: pd.Series,
    candidate: MonthlyMartingaleCandidate,
    scenario: AuditScenario,
    entry_mask: pd.Series | None = None,
    exit_mask: pd.Series | None = None,
) -> AuditGridBacktestResult:
    side_signal = side_signal.reindex(market.index)
    if scenario.signal_lag_bars > 0:
        side_signal = side_signal.shift(scenario.signal_lag_bars)
    entry_mask = None if entry_mask is None else entry_mask.reindex(market.index).fillna(False).astype(bool)
    exit_mask = None if exit_mask is None else exit_mask.reindex(market.index).fillna(False).astype(bool)
    rows: list[dict[str, Any]] = []
    i = 0
    horizon_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    entry_offset = 0 if scenario.entry_execution_mode == "current_close" else 1
    while i < len(market) - horizon_bars - entry_offset:
        side = side_signal.iloc[i]
        if side not in {"long", "short"}:
            i += 1
            continue
        entry_pos, _entry_column, _monitor_start = _entry_plan(i, market, scenario.entry_execution_mode)
        if entry_pos >= len(market):
            break
        if entry_mask is not None and bool(entry_mask.iloc[entry_pos]):
            i += 1
            continue
        result, extras = simulate_grid_from_signal_pos(
            market,
            i,
            risk,
            str(side),
            scenario,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            survival_min_realized_pnl=0.0,
            exit_blackout_series=exit_mask,
        )
        row = result.to_dict()
        row.update(asdict(candidate))
        row.update(extras)
        rows.append(row)
        cooldown_until = result.exit_timestamp + pd.Timedelta(hours=candidate.entry_cooldown_hours)
        i = max(market.index.get_loc(result.exit_timestamp) + 1, int(market.index.searchsorted(cooldown_until)))
    trades = pd.DataFrame(rows)
    if trades.empty:
        trades = pd.DataFrame(columns=list(GridSimulationResult.__dataclass_fields__.keys()))
    equity = make_equity_curve(market.index, trades)
    return AuditGridBacktestResult(trades=trades, equity_curve=equity)


def summarize_audit_result(
    equity: pd.Series,
    trades: pd.DataFrame,
    candidate: MonthlyMartingaleCandidate,
    split: str,
    scenario: AuditScenario,
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
    metrics.update(
        {
            **asdict(candidate),
            "scenario": scenario.name,
            "split": split,
            "planned_exposure": planned_exposure(candidate),
            "entry_execution_mode": scenario.entry_execution_mode,
            "conservative_intrabar": bool(scenario.conservative_intrabar),
            "signal_lag_bars": int(scenario.signal_lag_bars),
            "mask_lag_bars": int(scenario.mask_lag_bars),
            "scheduled_macro_only": bool(scenario.scheduled_macro_only),
            "selection_policy": scenario.selection_policy,
            "intrabar_ambiguous_tp_after_add": int(trades.get("intrabar_ambiguous_tp_after_add", pd.Series(dtype=int)).sum())
            if not trades.empty
            else 0,
            "intrabar_ambiguous_stop_and_tp": int(trades.get("intrabar_ambiguous_stop_and_tp", pd.Series(dtype=int)).sum())
            if not trades.empty
            else 0,
        }
    )
    return metrics


def run_exact_candidate_for_scenario(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: GridRiskConfig,
    candidate: MonthlyMartingaleCandidate,
    split_index: pd.Index,
    split_name: str,
    scenario: AuditScenario,
    entry_mask: pd.Series | None,
) -> tuple[dict[str, Any], AuditGridBacktestResult]:
    split_frame = split_frame_from_index(market, split_index)
    split_entry = None if entry_mask is None else entry_mask.reindex(split_frame.index).fillna(False).astype(bool)
    risk = risk_for_candidate(base_risk, candidate)
    side_signal = build_side_signal(market, signal_frame, candidate)
    result = run_signal_grid_backtest_audit(
        split_frame,
        risk,
        side_signal,
        candidate,
        scenario,
        entry_mask=split_entry,
    )
    metrics = summarize_audit_result(result.equity_curve, result.trades, candidate, split_name, scenario)
    return metrics, result


def simulate_candidate_sample_for_scenario(
    market: pd.DataFrame,
    positions: list[int],
    risk: GridRiskConfig,
    side_signal: pd.Series,
    candidate: MonthlyMartingaleCandidate,
    split: str,
    scenario: AuditScenario,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pos in positions:
        side = str(side_signal.iloc[pos])
        result, extras = simulate_grid_from_signal_pos(
            market,
            pos,
            risk,
            side,
            scenario,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            survival_min_realized_pnl=0.0,
        )
        row = result.to_dict()
        row.update({**asdict(candidate), "split": split, **extras})
        rows.append(row)
    return pd.DataFrame(rows)


def search_validation_sample_for_scenario(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    window: WalkForwardWindow,
    base_risk: GridRiskConfig,
    candidates: list[MonthlyMartingaleCandidate],
    config: dict[str, Any],
    scenario: AuditScenario,
    entry_mask: pd.Series | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stride = int(config["search"]["search_entry_stride_bars"])
    max_positions = int(config["search"]["max_sample_positions_per_candidate"])
    signal_cache: dict[tuple[Any, ...], pd.Series] = {}
    for candidate in candidates:
        risk = risk_for_candidate(base_risk, candidate)
        key = (
            candidate.side_mode,
            candidate.entry_mode,
            candidate.rsi_window,
            candidate.rsi_low,
            candidate.rsi_high,
            scenario.signal_lag_bars,
        )
        if key not in signal_cache:
            signal = build_side_signal(market, signal_frame, candidate)
            if scenario.signal_lag_bars > 0:
                signal = signal.shift(scenario.signal_lag_bars)
            signal_cache[key] = signal
        side_signal = signal_cache[key]
        positions = sample_positions_for_scenario(
            market,
            window.train,
            side_signal,
            risk,
            candidate.entry_cooldown_hours,
            stride,
            max_positions,
            entry_mask,
            scenario,
        )
        if not positions:
            continue
        sample = simulate_candidate_sample_for_scenario(
            market,
            positions,
            risk,
            side_signal,
            candidate,
            SEARCH_SPLIT,
            scenario,
        )
        summary = summarize_simulations(sample, baseline_grids=len(positions))
        rows.append(
            {
                **asdict(candidate),
                "split": SEARCH_SPLIT,
                "sample_positions": len(positions),
                "planned_exposure": planned_exposure(candidate),
                **summary,
            }
        )
    if not rows:
        raise ValueError(f"No validation samples generated for scenario {scenario.name}")
    return pd.DataFrame(rows)


def run_fold_locked(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: GridRiskConfig,
    selected_017: pd.DataFrame,
    window: WalkForwardWindow,
    scenario: AuditScenario,
    entry_mask: pd.Series | None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.Series]:
    candidate = selected_candidate_for_fold(selected_017, scenario, window.fold_id)
    metrics, result = run_exact_candidate_for_scenario(
        market,
        signal_frame,
        base_risk,
        candidate,
        window.test,
        "test",
        scenario,
        entry_mask,
    )
    selected_row = {
        "scenario": scenario.name,
        "fold_id": window.fold_id,
        "selected_from": "iteration_017_locked",
        **asdict(candidate),
        "planned_exposure": planned_exposure(candidate),
    }
    fold_row = fold_row_from_metrics(window, scenario, candidate, metrics, validation_metrics=None)
    return fold_row, selected_row, result.trades, result.equity_curve


def run_fold_reselect(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    base_risk: GridRiskConfig,
    candidates: list[MonthlyMartingaleCandidate],
    window: WalkForwardWindow,
    config: dict[str, Any],
    scenario: AuditScenario,
    entry_mask: pd.Series | None,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.Series]:
    sample_summary = search_validation_sample_for_scenario(
        market,
        signal_frame,
        window,
        base_risk,
        candidates,
        config,
        scenario,
        entry_mask,
    )
    top_rows = top_sample_candidates(sample_summary, config)
    validation_rows: list[dict[str, Any]] = []
    validation_results: dict[str, AuditGridBacktestResult] = {}
    for row in top_rows:
        candidate = candidate_from_row(row)
        validation_metrics, validation_result = run_exact_candidate_for_scenario(
            market,
            signal_frame,
            base_risk,
            candidate,
            window.train,
            SEARCH_SPLIT,
            scenario,
            entry_mask,
        )
        validation_rows.append(validation_metrics)
        validation_results[candidate.name] = validation_result
    validation_frame = pd.DataFrame(validation_rows)
    selected = select_best_exact_no_drawdown(validation_frame, config)
    selected_candidate = candidate_from_row(selected)
    test_metrics, test_result = run_exact_candidate_for_scenario(
        market,
        signal_frame,
        base_risk,
        selected_candidate,
        window.test,
        "test",
        scenario,
        entry_mask,
    )
    if not same_candidate(selected, test_metrics):
        raise ValueError("selected candidate changed before scenario test evaluation")
    validation_metrics = validation_frame[validation_frame["name"] == selected_candidate.name].iloc[0].to_dict()
    fold_row = fold_row_from_metrics(window, scenario, selected_candidate, test_metrics, validation_metrics)
    selected_row = {
        "scenario": scenario.name,
        "fold_id": window.fold_id,
        "selected_from": "validation_reselect",
        **selected,
    }
    return fold_row, selected_row, test_result.trades, test_result.equity_curve


def fold_row_from_metrics(
    window: WalkForwardWindow,
    scenario: AuditScenario,
    candidate: MonthlyMartingaleCandidate,
    test_metrics: dict[str, Any],
    validation_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    validation_metrics = validation_metrics or {}
    return {
        "scenario": scenario.name,
        "variant": scenario.variant,
        "selection_policy": scenario.selection_policy,
        "fold_id": window.fold_id,
        "train_start": window.train.min(),
        "train_end": window.train.max(),
        "test_start": window.test.min(),
        "test_end": window.test.max(),
        "selected_name": candidate.name,
        "validation_monthly_return": float(validation_metrics.get("monthly_return", np.nan)),
        "validation_total_return": float(validation_metrics.get("total_return", np.nan)),
        "validation_max_drawdown": float(validation_metrics.get("max_drawdown", np.nan)),
        "validation_profit_factor": float(validation_metrics.get("profit_factor", np.nan)),
        "validation_target_reached": bool(float(validation_metrics.get("monthly_return", -np.inf)) >= 0.20),
        "test_monthly_return": float(test_metrics["monthly_return"]),
        "test_total_return": float(test_metrics["total_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_profit_factor": float(test_metrics["profit_factor"]),
        "test_number_of_grids": int(test_metrics["number_of_grids"]),
        "test_positive": bool(float(test_metrics["total_return"]) > 0),
        "test_target_reached": bool(float(test_metrics["monthly_return"]) >= 0.20),
        "test_equity_ruined": bool(float(test_metrics.get("min_equity", 1.0)) <= 0),
        "entry_execution_mode": scenario.entry_execution_mode,
        "conservative_intrabar": bool(scenario.conservative_intrabar),
        "signal_lag_bars": int(scenario.signal_lag_bars),
        "mask_lag_bars": int(scenario.mask_lag_bars),
        "scheduled_macro_only": bool(scenario.scheduled_macro_only),
        "fee_rate": float(candidate.fee_rate),
        "slippage_bps": float(candidate.slippage_bps),
        "base_position_size_pct": float(candidate.base_position_size_pct),
        "max_total_exposure_pct": float(candidate.max_total_exposure_pct),
        "planned_exposure": planned_exposure(candidate),
        "intrabar_ambiguous_tp_after_add": int(test_metrics["intrabar_ambiguous_tp_after_add"]),
        "intrabar_ambiguous_stop_and_tp": int(test_metrics["intrabar_ambiguous_stop_and_tp"]),
    }


def summarize_scenario(
    scenario: AuditScenario,
    fold_rows: list[dict[str, Any]],
    oos_equity: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    fold_summary = pd.DataFrame(fold_rows)
    total = float(oos_equity.iloc[-1] / oos_equity.iloc[0] - 1.0) if float(oos_equity.iloc[0]) != 0 else -1.0
    monthly = monthly_return_from_equity(oos_equity)
    max_dd = float(drawdown_series(oos_equity).min())
    target = float(config["target"]["monthly_return"])
    return {
        "scenario": scenario.name,
        "variant": scenario.variant,
        "selection_policy": scenario.selection_policy,
        "fold_count": int(len(fold_summary)),
        "aggregate_total_return": total,
        "aggregate_monthly_return": monthly,
        "aggregate_max_drawdown": max_dd,
        "positive_fold_rate": float(fold_summary["test_positive"].mean()),
        "target_fold_rate": float(fold_summary["test_target_reached"].mean()),
        "validation_target_rate": float(fold_summary["validation_target_reached"].mean()),
        "equity_ruined": bool((oos_equity <= 0).any() or fold_summary["test_equity_ruined"].any()),
        "test_grid_count": int(fold_summary["test_number_of_grids"].sum()),
        "worst_fold_total_return": float(fold_summary["test_total_return"].min()),
        "worst_fold_monthly_return": float(fold_summary["test_monthly_return"].min()),
        "target_monthly_return": target,
        "entry_execution_mode": scenario.entry_execution_mode,
        "conservative_intrabar": bool(scenario.conservative_intrabar),
        "signal_lag_bars": int(scenario.signal_lag_bars),
        "mask_lag_bars": int(scenario.mask_lag_bars),
        "scheduled_macro_only": bool(scenario.scheduled_macro_only),
        "fee_rate": scenario.fee_rate,
        "slippage_bps": scenario.slippage_bps,
        "sizing_scale": float(scenario.sizing_scale),
        "intrabar_ambiguous_tp_after_add": int(fold_summary["intrabar_ambiguous_tp_after_add"].sum()),
        "intrabar_ambiguous_stop_and_tp": int(fold_summary["intrabar_ambiguous_stop_and_tp"].sum()),
    }


def write_sizing_unit_audit(selected_017: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selected_017.iterrows():
        candidate = candidate_from_row(row.to_dict())
        sequence = sizing_sequence(candidate)
        current_planned = planned_exposure(candidate)
        rows.append(
            {
                "variant": row["variant"],
                "fold_id": int(row["fold_id"]),
                "selected_name": row["name"],
                "base_position_size_raw": float(candidate.base_position_size_pct),
                "max_total_exposure_raw": float(candidate.max_total_exposure_pct),
                "sizing_sequence": json.dumps(sequence),
                "planned_exposure_multiple": current_planned,
                "base_position_if_percent_unit": float(candidate.base_position_size_pct) / 100.0,
                "max_total_exposure_if_percent_unit": float(candidate.max_total_exposure_pct) / 100.0,
                "planned_exposure_if_percent_unit": current_planned / 100.0,
                "interpreted_as_leveraged_notional": bool(current_planned > 1.0),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "sizing_unit_audit.csv", index=False)
    return audit


def build_intrabar_ambiguity_report(output_dir: Path, trades_by_scenario: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for scenario, frames in trades_by_scenario.items():
        if not frames:
            continue
        trades = pd.concat(frames, ignore_index=True)
        if trades.empty or "intrabar_ambiguous_tp_after_add" not in trades:
            continue
        impacted = trades[
            trades["intrabar_ambiguous_tp_after_add"].fillna(0).astype(int).gt(0)
            | trades["intrabar_ambiguous_stop_and_tp"].fillna(0).astype(int).gt(0)
        ].copy()
        if impacted.empty:
            continue
        impacted.insert(0, "scenario", scenario)
        rows.append(
            impacted[
                [
                    "scenario",
                    "signal_timestamp",
                    "entry_timestamp",
                    "exit_timestamp",
                    "side",
                    "realized_pnl",
                    "exit_reason",
                    "number_of_levels_filled",
                    "intrabar_ambiguous_tp_after_add",
                    "intrabar_ambiguous_stop_and_tp",
                ]
            ]
        )
    report = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[
            "scenario",
            "signal_timestamp",
            "entry_timestamp",
            "exit_timestamp",
            "side",
            "realized_pnl",
            "exit_reason",
            "number_of_levels_filled",
            "intrabar_ambiguous_tp_after_add",
            "intrabar_ambiguous_stop_and_tp",
        ]
    )
    report.to_csv(output_dir / "intrabar_ambiguity_report.csv", index=False)
    return report


def future_invariance_check(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    config: dict[str, Any],
    candidate: MonthlyMartingaleCandidate,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = [market.index[int(len(market) * pct)] for pct in [0.35, 0.50, 0.65]]
    base_trend = build_trend_escape_components(market, config)
    base_signal = build_side_signal(market, signal_frame, candidate)
    for target in targets:
        mutated_market = market.copy()
        future_mask = mutated_market.index > target
        mutated_market.loc[future_mask, ["high", "low", "close"]] = mutated_market.loc[future_mask, ["high", "low", "close"]] * 1.25
        mutated_trend = build_trend_escape_components(mutated_market, config)
        rows.append(
            {
                "check": "trend_escape_future_invariance",
                "timestamp": target,
                "passed": bool(base_trend.loc[target].fillna(0).equals(mutated_trend.loc[target].fillna(0))),
                "detail": "trend_escape components at t unchanged after mutating future OHLC",
            }
        )
        mutated_signal_frame = signal_frame.copy()
        mutated_signal_frame.loc[mutated_signal_frame.index > target, ["high", "low", "close"]] = (
            mutated_signal_frame.loc[mutated_signal_frame.index > target, ["high", "low", "close"]] * 1.25
        )
        mutated_signal = build_side_signal(market, mutated_signal_frame, candidate)
        left_signal = base_signal.loc[target]
        right_signal = mutated_signal.loc[target]
        same_signal = (pd.isna(left_signal) and pd.isna(right_signal)) or left_signal == right_signal
        rows.append(
            {
                "check": "side_signal_future_invariance",
                "timestamp": target,
                "passed": bool(same_signal),
                "detail": "RSI side signal at t unchanged after mutating future 1h candles",
            }
        )
    return rows


def leakage_audit(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    config: dict[str, Any],
    selected_017: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    first_candidate = candidate_from_row(
        selected_017[selected_017["variant"].eq("fundamental_trend_escape_entry_only")].iloc[0].to_dict()
    )
    rows = future_invariance_check(market, signal_frame, config, first_candidate)
    if {"open_time", "close_time"}.issubset(market.columns):
        index_matches_closed_boundary = bool((pd.to_datetime(market["close_time"], unit="ms", utc=True) + pd.Timedelta(milliseconds=1)).iloc[:100].equals(pd.Series(market.index[:100], index=market.index[:100])))
    else:
        index_matches_closed_boundary = False
    rows.append(
        {
            "check": "closed_candle_timestamp",
            "timestamp": market.index[0],
            "passed": index_matches_closed_boundary,
            "detail": "5m index equals Binance close_time + 1ms closed-candle boundary",
        }
    )
    events = load_fundamental_events(config)
    surprise = events[events["is_surprise"].astype(bool)].copy()
    if surprise.empty:
        surprise_passed = True
    else:
        surprise_passed = True
        for _, event in surprise.iterrows():
            single_event = pd.DataFrame([event.to_dict()])
            realistic_windows = build_blackout_windows(single_event, config.get("fundamental_blackout", {}), "realistic")
            realistic = _strict_after_mask(market.index, realistic_windows)
            known = pd.Timestamp(event["known_time_utc"]).tz_convert("UTC")
            if bool(realistic.loc[realistic.index <= known].any()):
                surprise_passed = False
                break
    rows.append(
        {
            "check": "surprise_blackout_after_known_time",
            "timestamp": market.index[0],
            "passed": bool(surprise_passed),
            "detail": "strict realistic surprise mask does not activate before or exactly at known_time_utc",
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "leakage_audit.csv", index=False)
    return frame


def decide_audit(comparison: pd.DataFrame, leakage: pd.DataFrame, config: dict[str, Any]) -> dict[str, str]:
    by_scenario = comparison.set_index("scenario")
    control_baseline = by_scenario.loc["control_baseline_replay"]
    control_fundamental = by_scenario.loc["control_fundamental_replay"]
    expected_baseline = float(config["source_iteration"]["control_baseline_monthly_return"])
    expected_fundamental = float(config["source_iteration"]["control_audited_monthly_return"])
    tolerance = float(config["source_iteration"].get("control_tolerance", 1e-6))
    expected_full_folds = int(config["source_iteration"].get("expected_full_folds", 18))
    is_truncated = int(control_fundamental["fold_count"]) < expected_full_folds
    control_ok = (not is_truncated) and (
        abs(float(control_baseline["aggregate_monthly_return"]) - expected_baseline) <= tolerance
        and abs(float(control_fundamental["aggregate_monthly_return"]) - expected_fundamental) <= tolerance
    )
    sizing = by_scenario.loc["sizing_percent_corrected_locked"]
    cost = by_scenario.loc["fee_0p0006_slip_5_locked"]
    same_bar = by_scenario.loc["next_bar_open_conservative_locked"]
    worst = by_scenario.loc[str(config["audit"]["primary_worst_case_scenario"])]
    leakage_ok = bool(leakage["passed"].astype(bool).all())
    target = float(config["target"]["monthly_return"])
    out = {
        "control_reproduction": "control replay skipped on truncated run"
        if is_truncated
        else ("control replay ok" if control_ok else "control replay mismatch"),
        "sizing_unit_verdict": "sizing unit issue" if float(sizing["aggregate_monthly_return"]) < 0.02 else "sizing unit not primary issue",
        "same_bar_verdict": "same-bar optimism issue"
        if float(same_bar["aggregate_monthly_return"]) <= 0 or float(same_bar["aggregate_monthly_return"]) < float(control_fundamental["aggregate_monthly_return"]) * 0.5
        else "same-bar not primary issue",
        "cost_verdict": "cost sensitivity issue"
        if float(cost["aggregate_monthly_return"]) <= 0 or float(cost["aggregate_monthly_return"]) < float(control_fundamental["aggregate_monthly_return"]) * 0.5
        else "costs not primary issue",
        "leakage_verdict": "fundamental/feature leakage issue" if not leakage_ok else "no leakage detected by audit",
    }
    if bool(worst["equity_ruined"]) or float(worst["aggregate_monthly_return"]) <= 0:
        out["worst_case_verdict"] = "rejected under worst case"
    elif float(worst["aggregate_monthly_return"]) >= target:
        out["worst_case_verdict"] = "worst-case viable"
    else:
        out["worst_case_verdict"] = "fragile positive edge"
    return out


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    comparison = pd.DataFrame(payload["scenario_comparison"])
    verdicts = payload["verdicts"]
    rows = comparison[
        [
            "scenario",
            "aggregate_monthly_return",
            "aggregate_total_return",
            "aggregate_max_drawdown",
            "positive_fold_rate",
            "target_fold_rate",
            "test_grid_count",
            "equity_ruined",
        ]
    ]
    lines = [
        "# Iteration 019 - Execution Worst-Case Audit",
        "",
        "## Verdicts",
        "",
        markdown_table(pd.DataFrame([verdicts])),
        "",
        "## Scenario Comparison",
        "",
        markdown_table(rows),
        "",
        "## Interpretation",
        "",
        "This audit replays Iteration 017 with isolated stress scenarios for sizing units, same-bar execution, costs, and feature/fundamental actionability. The primary worst case keeps the leveraged sizing interpretation but adds next-bar entry, conservative intrabar priority, delayed signals/masks, scheduled macro only, and conservative costs.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(
    config_path: str = DEFAULT_CONFIG,
    max_folds: int | None = None,
    max_candidates: int | None = None,
    exact_top_n: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if exact_top_n is not None:
        config["search"]["exact_top_n"] = int(exact_top_n)
    scenarios = validate_scenarios(config)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = output_dir / "selected_fold_trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    for old_file in trades_dir.glob("*_fold_*_trades.csv"):
        old_file.unlink()

    source_dir = project_path(config["source_iteration"]["output_dir"])
    selected_017 = load_selected_017(source_dir)
    base_risk = validate_strategy_config(load_strategy_config())
    market = prepare_market()
    signal_frame = load_processed(project_path("data/processed/btcusdt_1h.parquet"))
    source_config = load_yaml(config["source_iteration"]["config_path"])
    trend_components = build_trend_escape_components(market, source_config)
    trend_escape = trend_components["trend_escape"].astype(bool)
    wf = config["walk_forward"]
    windows = make_walk_forward_windows(
        market.index,
        train_days=float(wf["train_days"]),
        test_days=float(wf["test_days"]),
        step_days=float(wf["step_days"]),
        embargo_bars=int(wf.get("embargo_bars", 0)),
        max_folds=max_folds,
    )

    selected_audit = write_sizing_unit_audit(selected_017, output_dir)
    leakage = leakage_audit(market, signal_frame, source_config, selected_017, output_dir)

    comparison_rows: list[dict[str, Any]] = []
    selected_rows_all: list[dict[str, Any]] = []
    trades_by_scenario: dict[str, list[pd.DataFrame]] = {}
    for scenario in scenarios:
        LOGGER.info("Running execution audit scenario: %s", scenario.name)
        scenario_output_config = scenario_config(config, scenario)
        candidate_config = scenario_config(source_config, scenario)
        if exact_top_n is not None:
            candidate_config["search"]["exact_top_n"] = int(exact_top_n)
            scenario_output_config["search"]["exact_top_n"] = int(exact_top_n)
        entry_mask = build_entry_mask_for_scenario(market, source_config, trend_escape, scenario)
        candidates = make_candidates(candidate_config)
        sample_size = max_candidates if max_candidates is not None else candidate_config["search"].get("candidate_sample_size")
        sample_seed = candidate_config["search"].get("candidate_sample_seed")
        candidates = choose_candidate_subset(
            candidates,
            None if sample_size is None else int(sample_size),
            None if sample_seed is None else int(sample_seed),
        )
        fold_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []
        fold_equities: list[tuple[int, pd.Series]] = []
        trades_by_scenario[scenario.name] = []
        for window in windows:
            LOGGER.info("Running %s fold %s/%s", scenario.name, window.fold_id, len(windows))
            if scenario.selection_policy == "locked_017":
                fold_row, selected_row, trades, equity = run_fold_locked(
                    market,
                    signal_frame,
                    base_risk,
                    selected_017,
                    window,
                    scenario,
                    entry_mask,
                )
            else:
                fold_row, selected_row, trades, equity = run_fold_reselect(
                    market,
                    signal_frame,
                    base_risk,
                    candidates,
                    window,
                    scenario_output_config,
                    scenario,
                    entry_mask,
                )
            fold_rows.append(fold_row)
            selected_rows.append(selected_row)
            selected_rows_all.append(selected_row)
            fold_equities.append((window.fold_id, equity))
            trades_by_scenario[scenario.name].append(trades)
            trades.to_csv(trades_dir / f"{scenario.name}_fold_{window.fold_id:03d}_trades.csv", index=False)
        fold_summary = pd.DataFrame(fold_rows)
        selected_frame = pd.DataFrame(selected_rows)
        equity_frame = stitch_oos_equity(fold_equities)
        fold_summary.to_csv(output_dir / f"fold_summary_{scenario.name}.csv", index=False)
        selected_frame.to_csv(output_dir / f"selected_candidates_{scenario.name}.csv", index=False)
        equity_frame.to_csv(output_dir / f"oos_equity_{scenario.name}.csv")
        comparison_rows.append(summarize_scenario(scenario, fold_rows, equity_frame["equity"], config))

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "scenario_comparison.csv", index=False)
    pd.DataFrame(selected_rows_all).to_csv(output_dir / "selected_candidates_all_scenarios.csv", index=False)
    ambiguity = build_intrabar_ambiguity_report(output_dir, trades_by_scenario)
    verdicts = decide_audit(comparison, leakage, config)
    payload = {
        "iteration_name": config["iteration"]["name"],
        "scenario_comparison": comparison.to_dict("records"),
        "verdicts": verdicts,
        "fold_count": len(windows),
        "sizing_audit_rows": int(len(selected_audit)),
        "leakage_checks": leakage.to_dict("records"),
        "intrabar_ambiguity_rows": int(len(ambiguity)),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report_path = write_report(output_dir, payload)
    LOGGER.info("Wrote execution worst-case audit outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Iteration 019 execution worst-case and leakage audit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--exact-top-n", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_iteration(
        args.config,
        max_folds=args.max_folds,
        max_candidates=args.max_candidates,
        exact_top_n=args.exact_top_n,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
