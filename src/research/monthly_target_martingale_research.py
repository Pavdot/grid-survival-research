from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.metrics import calculate_metrics
from src.backtesting.walk_forward import temporal_train_validation_test_split
from src.data.validate_data import load_processed
from src.labeling.grid_engine import GridSimulationResult, simulate_grid_from_index
from src.labeling.grid_risk import GridRiskConfig, validate_strategy_config
from src.research.economy_first_research import (
    FORCED_COLUMNS,
    prepare_market,
    summarize_simulations,
)
from src.research.momentum_switch_research import rsi
from src.utils.config_loader import load_settings, load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
SEARCH_SPLIT = "validation"
VALID_SIDE_MODES = {
    "mean_reversion_dual",
    "momentum_dual",
    "long_oversold",
    "short_overbought",
    "long_momentum",
    "short_momentum",
}


@dataclass(frozen=True)
class MonthlyMartingaleCandidate:
    name: str
    side_mode: str
    entry_mode: str
    entry_cooldown_hours: float
    rsi_window: int
    rsi_low: float
    rsi_high: float
    spacing_atr_multiplier: float
    take_profit_spacing_multiplier: float
    max_levels: int
    base_position_size_pct: float
    progression_multiplier: float
    max_total_exposure_pct: float
    fee_rate: float
    slippage_bps: float
    max_grid_loss_pct: float
    max_holding_hours: float
    stop_on_regime_break: bool
    stop_on_volatility_shock: bool


@dataclass
class SignalGridBacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series


def monthly_return_from_equity(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2 or float(equity.iloc[0]) <= 0:
        return 0.0
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    if total_return <= -1:
        return -1.0
    days = max((equity.index[-1] - equity.index[0]) / pd.Timedelta(days=1), 1 / 24)
    return float((1 + total_return) ** (30.4375 / days) - 1)


def _candidate_name(parts: dict[str, Any]) -> str:
    tokens = [
        parts["side_mode"],
        str(parts["entry_mode"]),
        f"cool{parts['entry_cooldown_hours']:g}",
        f"rsi{parts['rsi_window']}_{parts['rsi_low']:g}_{parts['rsi_high']:g}",
        f"sp{parts['spacing_atr_multiplier']:g}",
        f"tp{parts['take_profit_spacing_multiplier']:g}",
        f"lvl{parts['max_levels']}",
        f"base{parts['base_position_size_pct']:g}",
        f"prog{parts['progression_multiplier']:g}",
        f"cap{parts['max_total_exposure_pct']:g}",
        f"fee{parts['fee_rate']:g}",
        f"slip{parts['slippage_bps']:g}",
        f"loss{parts['max_grid_loss_pct']:g}",
        f"hold{parts['max_holding_hours']:g}",
        f"reg{int(parts['stop_on_regime_break'])}",
    ]
    return "_".join(tokens).replace(".", "p")


def sizing_sequence(candidate: MonthlyMartingaleCandidate) -> tuple[float, ...]:
    if candidate.progression_multiplier <= 1:
        raise ValueError("progression_multiplier must be > 1 for bounded martingale research")
    return tuple(float(candidate.progression_multiplier**level) for level in range(candidate.max_levels))


def planned_exposure(candidate: MonthlyMartingaleCandidate) -> float:
    return float(candidate.base_position_size_pct * sum(sizing_sequence(candidate)))


def make_candidates(config: dict[str, Any]) -> list[MonthlyMartingaleCandidate]:
    search = config["search"]
    max_progression = float(search.get("max_progression_multiplier", 1.75))
    if max_progression >= 1.8:
        raise ValueError("max_progression_multiplier must stay below exponential martingale territory")
    max_notional_exposure = float(search.get("max_notional_exposure_pct", 1.0))
    if max_notional_exposure <= 0:
        raise ValueError("max_notional_exposure_pct must be positive")

    candidates: list[MonthlyMartingaleCandidate] = []
    for side_mode in search["side_modes"]:
        if side_mode not in VALID_SIDE_MODES:
            raise ValueError(f"Unsupported side_mode: {side_mode}")
        for entry_mode in search["entry_modes"]:
            if entry_mode not in {"hourly_extreme", "hourly_cross"}:
                raise ValueError("entry_modes must contain only hourly_extreme or hourly_cross")
            for cooldown in search["entry_cooldown_hours"]:
                cooldown = float(cooldown)
                if cooldown < 0:
                    raise ValueError("entry_cooldown_hours must be non-negative")
                for window in search["rsi_windows"]:
                    if int(window) <= 1:
                        raise ValueError("rsi_windows must be > 1")
                    for low, high in search["rsi_threshold_pairs"]:
                        if float(low) >= float(high):
                            raise ValueError("RSI low threshold must be below high threshold")
                        for spacing in search["spacing_atr_multipliers"]:
                            if float(spacing) <= 0:
                                raise ValueError("spacing_atr_multipliers must be positive")
                            for tp in search["take_profit_spacing_multipliers"]:
                                if float(tp) <= 0:
                                    raise ValueError("take_profit_spacing_multipliers must be positive")
                                for max_levels in search["max_levels"]:
                                    if int(max_levels) <= 0:
                                        raise ValueError("max_levels must be positive")
                                    for base_size in search["base_position_size_pcts"]:
                                        if float(base_size) <= 0:
                                            raise ValueError("base_position_size_pcts must be positive")
                                        for progression in search["progression_multipliers"]:
                                            progression = float(progression)
                                            if progression <= 1 or progression > max_progression:
                                                raise ValueError(
                                                    "progression_multipliers must be within (1, max_progression]"
                                                )
                                            for max_exposure in search["max_total_exposure_pcts"]:
                                                max_exposure = float(max_exposure)
                                                if max_exposure <= 0 or max_exposure > max_notional_exposure:
                                                    raise ValueError(
                                                        "max_total_exposure_pcts must be within "
                                                        "(0, max_notional_exposure_pct]"
                                                    )
                                                for fee_rate in search.get("fee_rates", [0.0004]):
                                                    fee_rate = float(fee_rate)
                                                    if fee_rate < 0:
                                                        raise ValueError("fee_rates must be non-negative")
                                                    for slippage_bps in search.get("slippage_bps_values", [2]):
                                                        slippage_bps = float(slippage_bps)
                                                        if slippage_bps < 0:
                                                            raise ValueError("slippage_bps_values must be non-negative")
                                                        for max_loss in search["max_grid_loss_pcts"]:
                                                            max_loss = float(max_loss)
                                                            if max_loss <= 0:
                                                                raise ValueError("max_grid_loss_pcts must be positive")
                                                            for holding in search["max_holding_hours"]:
                                                                if float(holding) <= 0:
                                                                    raise ValueError("max_holding_hours must be positive")
                                                                for stop_regime in search["stop_on_regime_break"]:
                                                                    for stop_vol in search["stop_on_volatility_shock"]:
                                                                        fields = {
                                                                            "side_mode": str(side_mode),
                                                                            "entry_mode": str(entry_mode),
                                                                            "entry_cooldown_hours": cooldown,
                                                                            "rsi_window": int(window),
                                                                            "rsi_low": float(low),
                                                                            "rsi_high": float(high),
                                                                            "spacing_atr_multiplier": float(spacing),
                                                                            "take_profit_spacing_multiplier": float(tp),
                                                                            "max_levels": int(max_levels),
                                                                            "base_position_size_pct": float(base_size),
                                                                            "progression_multiplier": progression,
                                                                            "max_total_exposure_pct": max_exposure,
                                                                            "fee_rate": fee_rate,
                                                                            "slippage_bps": slippage_bps,
                                                                            "max_grid_loss_pct": max_loss,
                                                                            "max_holding_hours": float(holding),
                                                                            "stop_on_regime_break": bool(stop_regime),
                                                                            "stop_on_volatility_shock": bool(stop_vol),
                                                                        }
                                                                        candidate = MonthlyMartingaleCandidate(
                                                                            name=_candidate_name(fields), **fields
                                                                        )
                                                                        if planned_exposure(candidate) <= max_exposure + 1e-12:
                                                                            candidates.append(candidate)
    if not candidates:
        raise ValueError("No bounded martingale candidates generated")
    return candidates


def risk_for_candidate(base_risk: GridRiskConfig, candidate: MonthlyMartingaleCandidate) -> GridRiskConfig:
    return replace(
        base_risk,
        spacing_atr_multiplier=candidate.spacing_atr_multiplier,
        max_levels=candidate.max_levels,
        base_position_size_pct=candidate.base_position_size_pct,
        sizing_sequence=sizing_sequence(candidate),
        taker_fee=candidate.fee_rate,
        maker_fee=candidate.fee_rate,
        slippage_bps=candidate.slippage_bps,
        max_grid_loss_pct=candidate.max_grid_loss_pct,
        max_total_exposure_pct=candidate.max_total_exposure_pct,
        max_holding_hours=candidate.max_holding_hours,
        stop_on_regime_break=candidate.stop_on_regime_break,
        stop_on_volatility_shock=candidate.stop_on_volatility_shock,
    )


def signal_cache_key(candidate: MonthlyMartingaleCandidate) -> tuple[object, ...]:
    return (
        candidate.side_mode,
        candidate.entry_mode,
        candidate.rsi_window,
        candidate.rsi_low,
        candidate.rsi_high,
    )


def build_side_signal(market: pd.DataFrame, signal_frame: pd.DataFrame, candidate: MonthlyMartingaleCandidate) -> pd.Series:
    values = rsi(signal_frame["close"], candidate.rsi_window)
    signal_1h = pd.Series(pd.NA, index=signal_frame.index, dtype="object")
    low = candidate.rsi_low
    high = candidate.rsi_high
    if candidate.side_mode == "mean_reversion_dual":
        signal_1h[values <= low] = "long"
        signal_1h[values >= high] = "short"
    elif candidate.side_mode == "momentum_dual":
        signal_1h[values >= high] = "long"
        signal_1h[values <= low] = "short"
    elif candidate.side_mode == "long_oversold":
        signal_1h[values <= low] = "long"
    elif candidate.side_mode == "short_overbought":
        signal_1h[values >= high] = "short"
    elif candidate.side_mode == "long_momentum":
        signal_1h[values >= high] = "long"
    elif candidate.side_mode == "short_momentum":
        signal_1h[values <= low] = "short"
    else:
        raise ValueError(f"Unsupported side_mode: {candidate.side_mode}")
    if candidate.entry_mode == "hourly_cross":
        current = signal_1h.fillna("__none__")
        previous = signal_1h.shift(1).fillna("__none__")
        signal_1h = signal_1h.where(signal_1h.notna() & current.ne(previous))
    elif candidate.entry_mode != "hourly_extreme":
        raise ValueError("entry_mode must be hourly_extreme or hourly_cross")
    return signal_1h.reindex(market.index)


def _make_equity_curve(index: pd.Index, trades: pd.DataFrame) -> pd.Series:
    equity = pd.Series(1.0, index=index, dtype=float)
    current_equity = 1.0
    if trades.empty:
        equity.name = "equity"
        return equity
    for _, trade in trades.iterrows():
        current_equity += float(trade["realized_pnl"])
        exit_ts = trade["exit_timestamp"]
        equity.loc[equity.index >= exit_ts] = current_equity
    equity.name = "equity"
    return equity


def run_signal_grid_backtest(
    market: pd.DataFrame,
    risk: GridRiskConfig,
    side_signal: pd.Series,
    candidate: MonthlyMartingaleCandidate,
) -> SignalGridBacktestResult:
    side_signal = side_signal.reindex(market.index)
    rows: list[dict[str, object]] = []
    i = 0
    horizon_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    while i < len(market) - horizon_bars:
        side = side_signal.iloc[i]
        if side not in {"long", "short"}:
            i += 1
            continue
        result: GridSimulationResult = simulate_grid_from_index(
            market,
            i,
            risk,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            survival_min_realized_pnl=0.0,
            side=str(side),
        )
        row = result.to_dict()
        row.update(asdict(candidate))
        rows.append(row)
        cooldown_until = result.exit_timestamp + pd.Timedelta(hours=candidate.entry_cooldown_hours)
        i = max(market.index.get_loc(result.exit_timestamp) + 1, int(market.index.searchsorted(cooldown_until)))

    trades = pd.DataFrame(rows)
    if trades.empty:
        trades = pd.DataFrame(columns=list(GridSimulationResult.__dataclass_fields__.keys()))
    equity = _make_equity_curve(market.index, trades)
    return SignalGridBacktestResult(trades=trades, equity_curve=equity)


def split_indexes(index: pd.Index) -> dict[str, pd.Index]:
    settings = load_settings()
    split = temporal_train_validation_test_split(
        index,
        train_fraction=float(settings["validation"]["train_fraction"]),
        validation_fraction=float(settings["validation"]["validation_fraction"]),
        embargo_bars=int(settings["validation"]["embargo_bars"]),
    )
    return {"validation": split.validation, "test": split.test}


def split_market(market: pd.DataFrame, indexes: dict[str, pd.Index], split_name: str) -> pd.DataFrame:
    split_index = indexes[split_name]
    return market.loc[(market.index >= split_index.min()) & (market.index <= split_index.max())]


def sample_positions(
    market: pd.DataFrame,
    split_index: pd.Index,
    side_signal: pd.Series,
    risk: GridRiskConfig,
    cooldown_hours: float,
    stride_bars: int,
    max_positions: int,
) -> list[int]:
    if stride_bars <= 0:
        raise ValueError("search_entry_stride_bars must be positive")
    if max_positions <= 0:
        raise ValueError("max_sample_positions_per_candidate must be positive")
    max_bars = max(1, int(risk.max_holding_hours * 60 / 5))
    split_start = int(market.index.searchsorted(split_index.min(), side="left"))
    split_end = int(market.index.searchsorted(split_index.max(), side="right") - 1)
    positions: list[int] = []
    cooldown_until = market.index[split_start]
    eligible_signal = side_signal.iloc[split_start : max(split_start, split_end - max_bars + 1)]
    eligible_positions = np.flatnonzero(eligible_signal.isin(["long", "short"]).to_numpy()) + split_start
    last_position = split_start - stride_bars
    for pos in eligible_positions:
        if pos - last_position < stride_bars:
            continue
        if market.index[pos] < cooldown_until:
            continue
        if side_signal.iloc[pos] in {"long", "short"} and np.isfinite(float(market.iloc[pos].get("atr_5m", np.nan))):
            positions.append(pos)
            last_position = pos
            cooldown_until = market.index[pos] + pd.Timedelta(hours=max(risk.max_holding_hours, cooldown_hours))
        if len(positions) >= max_positions:
            break
    return positions


def simulate_candidate_sample(
    market: pd.DataFrame,
    positions: list[int],
    risk: GridRiskConfig,
    side_signal: pd.Series,
    candidate: MonthlyMartingaleCandidate,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pos in positions:
        side = str(side_signal.iloc[pos])
        result = simulate_grid_from_index(
            market,
            pos,
            risk,
            take_profit_spacing_multiplier=candidate.take_profit_spacing_multiplier,
            survival_min_realized_pnl=0.0,
            side=side,
        )
        row = result.to_dict()
        row.update({**asdict(candidate), "split": split})
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_exact(equity: pd.Series, trades: pd.DataFrame, candidate: MonthlyMartingaleCandidate, split: str) -> dict[str, Any]:
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
    metrics["target_monthly_return"] = np.nan
    metrics.update({**asdict(candidate), "split": split, "planned_exposure": planned_exposure(candidate)})
    return metrics


def search_validation_sample(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    indexes: dict[str, pd.Index],
    base_risk: GridRiskConfig,
    candidates: list[MonthlyMartingaleCandidate],
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stride = int(config["search"]["search_entry_stride_bars"])
    max_positions = int(config["search"]["max_sample_positions_per_candidate"])
    signal_cache: dict[tuple[object, ...], pd.Series] = {}
    for candidate in candidates:
        risk = risk_for_candidate(base_risk, candidate)
        key = signal_cache_key(candidate)
        if key not in signal_cache:
            signal_cache[key] = build_side_signal(market, signal_frame, candidate)
        side_signal = signal_cache[key]
        positions = sample_positions(
            market,
            indexes[SEARCH_SPLIT],
            side_signal,
            risk,
            candidate.entry_cooldown_hours,
            stride,
            max_positions,
        )
        if not positions:
            continue
        sample = simulate_candidate_sample(market, positions, risk, side_signal, candidate, SEARCH_SPLIT)
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
        raise ValueError("No validation samples generated")
    return pd.DataFrame(rows)


def top_sample_candidates(sample_summary: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    if sample_summary.empty:
        raise ValueError("Sample summary is empty")
    if set(sample_summary["split"].unique()) != {SEARCH_SPLIT}:
        raise ValueError("Sample ranking must use validation rows only")
    exact_top_n = int(config["search"]["exact_top_n"])
    ranked = sample_summary.sort_values(
        by=["expectancy", "realized_pnl", "profit_factor", "number_of_forced_exits"],
        ascending=[False, False, False, True],
    )
    return ranked.head(exact_top_n).to_dict("records")


def choose_candidate_subset(
    candidates: list[MonthlyMartingaleCandidate],
    max_candidates: int | None,
    seed: int | None,
) -> list[MonthlyMartingaleCandidate]:
    if max_candidates is None or max_candidates >= len(candidates):
        return candidates
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    rng = np.random.default_rng(seed)
    indexes = sorted(int(value) for value in rng.choice(len(candidates), size=max_candidates, replace=False))
    return [candidates[index] for index in indexes]


def candidate_from_row(row: dict[str, Any]) -> MonthlyMartingaleCandidate:
    return MonthlyMartingaleCandidate(
        name=str(row["name"]),
        side_mode=str(row["side_mode"]),
        entry_mode=str(row["entry_mode"]),
        entry_cooldown_hours=float(row["entry_cooldown_hours"]),
        rsi_window=int(row["rsi_window"]),
        rsi_low=float(row["rsi_low"]),
        rsi_high=float(row["rsi_high"]),
        spacing_atr_multiplier=float(row["spacing_atr_multiplier"]),
        take_profit_spacing_multiplier=float(row["take_profit_spacing_multiplier"]),
        max_levels=int(row["max_levels"]),
        base_position_size_pct=float(row["base_position_size_pct"]),
        progression_multiplier=float(row["progression_multiplier"]),
        max_total_exposure_pct=float(row["max_total_exposure_pct"]),
        fee_rate=float(row["fee_rate"]),
        slippage_bps=float(row["slippage_bps"]),
        max_grid_loss_pct=float(row["max_grid_loss_pct"]),
        max_holding_hours=float(row["max_holding_hours"]),
        stop_on_regime_break=bool(row["stop_on_regime_break"]),
        stop_on_volatility_shock=bool(row["stop_on_volatility_shock"]),
    )


def evaluate_exact_candidates(
    market: pd.DataFrame,
    signal_frame: pd.DataFrame,
    indexes: dict[str, pd.Index],
    base_risk: GridRiskConfig,
    candidate_rows: list[dict[str, Any]],
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, SignalGridBacktestResult]]:
    rows: list[dict[str, Any]] = []
    results: dict[str, SignalGridBacktestResult] = {}
    split_frame = split_market(market, indexes, split_name)
    signal_cache: dict[tuple[object, ...], pd.Series] = {}
    for row in candidate_rows:
        candidate = candidate_from_row(row)
        risk = risk_for_candidate(base_risk, candidate)
        key = signal_cache_key(candidate)
        if key not in signal_cache:
            signal_cache[key] = build_side_signal(market, signal_frame, candidate)
        side_signal = signal_cache[key]
        result = run_signal_grid_backtest(split_frame, risk, side_signal, candidate)
        rows.append(summarize_exact(result.equity_curve, result.trades, candidate, split_name))
        results[candidate.name] = result
    if not rows:
        raise ValueError(f"No exact rows evaluated for {split_name}")
    return pd.DataFrame(rows), results


def select_best_exact(validation_exact: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if validation_exact.empty:
        raise ValueError("Validation exact summary is empty")
    if set(validation_exact["split"].unique()) != {SEARCH_SPLIT}:
        raise ValueError("Monthly target selection must use validation rows only")
    target = float(config["target"]["monthly_return"])
    max_drawdown = float(config["target"]["max_drawdown"])
    min_grids = int(config["target"]["min_exact_grids"])
    eligible = validation_exact[
        (validation_exact["number_of_grids"].astype(int) >= min_grids)
        & (validation_exact["max_drawdown"].astype(float) >= max_drawdown)
    ].copy()
    if eligible.empty:
        eligible = validation_exact.copy()
    eligible["target_reached"] = eligible["monthly_return"].astype(float) >= target
    eligible = eligible.sort_values(
        by=["target_reached", "monthly_return", "max_drawdown", "profit_factor", "number_of_forced_exits"],
        ascending=[False, False, False, False, True],
    )
    selected = eligible.iloc[0].to_dict()
    selected["selected_from_validation_only"] = True
    selected["target_monthly_return"] = target
    selected["max_drawdown_constraint"] = max_drawdown
    selected["min_exact_grids"] = min_grids
    return selected


def decide(test_metrics: dict[str, Any], config: dict[str, Any]) -> str:
    target = float(config["target"]["monthly_return"])
    max_drawdown = float(config["target"]["max_drawdown"])
    min_grids = int(config["target"]["min_exact_grids"])
    if int(test_metrics.get("number_of_grids", 0)) < min_grids:
        return "no-trade / too few grid opportunities"
    if float(test_metrics.get("max_drawdown", -1.0)) < max_drawdown:
        return "monthly target candidate rejected by drawdown"
    if float(test_metrics.get("monthly_return", 0.0)) >= target:
        return "bounded martingale grid reached monthly target"
    if float(test_metrics.get("monthly_return", 0.0)) > 0:
        return "positive bounded martingale grid below monthly target"
    return "bounded martingale grid not viable"


def write_report(
    output_dir: Path,
    selected: dict[str, Any],
    validation_exact: dict[str, Any],
    test_exact: dict[str, Any],
    decision: str,
) -> Path:
    report = output_dir / "iteration_report.md"
    lines = [
        "# Iteration 005 - Monthly Target Bounded Martingale Grid",
        "",
        "## Decision",
        f"`{decision}`",
        "",
        "## Selected Validation Candidate",
        "```json",
        json.dumps(selected, indent=2, default=str),
        "```",
        "",
        "## Exact Sequential Backtest - Validation",
        "```json",
        json.dumps(validation_exact, indent=2, default=str),
        "```",
        "",
        "## Exact Sequential Backtest - Test",
        "```json",
        json.dumps(test_exact, indent=2, default=str),
        "```",
        "",
        "## Interpretation",
        "This iteration searches a bounded progressive martingale-style grid, not an unbounded exponential martingale. Candidate selection uses validation only. The test split is evaluated once for the selected candidate. Fees, slippage, max notional exposure, max grid loss, holding limits, and closed-candle RSI signals remain active. If max notional exposure is above 1.0, the result is leveraged research only and not a live-trading configuration.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_iteration(
    config_path: str,
    max_candidates: int | None = None,
    max_sample_positions: int | None = None,
    exact_top_n: int | None = None,
    candidate_sample_seed: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if max_sample_positions is not None:
        config["search"]["max_sample_positions_per_candidate"] = int(max_sample_positions)
    if exact_top_n is not None:
        config["search"]["exact_top_n"] = int(exact_top_n)

    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_risk = validate_strategy_config(load_strategy_config())
    market = prepare_market()
    signal_frame = load_processed(project_path("data/processed/btcusdt_1h.parquet"))
    indexes = split_indexes(market.index)
    candidates = make_candidates(config)
    configured_sample_size = config["search"].get("candidate_sample_size")
    configured_seed = config["search"].get("candidate_sample_seed")
    sample_size = max_candidates if max_candidates is not None else configured_sample_size
    sample_seed = candidate_sample_seed if candidate_sample_seed is not None else configured_seed
    sample_size = None if sample_size is None else int(sample_size)
    sample_seed = None if sample_seed is None else int(sample_seed)
    total_candidate_count = len(candidates)
    candidates = choose_candidate_subset(candidates, sample_size, sample_seed)

    sample_summary = search_validation_sample(market, signal_frame, indexes, base_risk, candidates, config)
    sample_summary.to_csv(output_dir / "martingale_sample_validation.csv", index=False)
    top_rows = top_sample_candidates(sample_summary, config)

    validation_exact_frame, validation_results = evaluate_exact_candidates(
        market,
        signal_frame,
        indexes,
        base_risk,
        top_rows,
        "validation",
    )
    validation_exact_frame.to_csv(output_dir / "martingale_exact_validation_top.csv", index=False)
    selected = select_best_exact(validation_exact_frame, config)
    selected_candidate = candidate_from_row(selected)
    selected_row = asdict(selected_candidate)

    test_exact_frame, test_results = evaluate_exact_candidates(
        market,
        signal_frame,
        indexes,
        base_risk,
        [selected_row],
        "test",
    )
    test_exact = test_exact_frame.iloc[0].to_dict()
    validation_exact = validation_exact_frame[validation_exact_frame["name"] == selected_candidate.name].iloc[0].to_dict()
    test_exact_frame.to_csv(output_dir / "selected_exact_test_metrics.csv", index=False)

    validation_result = validation_results[selected_candidate.name]
    test_result = test_results[selected_candidate.name]
    validation_result.trades.to_csv(output_dir / "selected_exact_validation_trades.csv", index=False)
    test_result.trades.to_csv(output_dir / "selected_exact_test_trades.csv", index=False)
    validation_result.equity_curve.to_frame().to_csv(output_dir / "selected_exact_validation_equity.csv")
    test_result.equity_curve.to_frame().to_csv(output_dir / "selected_exact_test_equity.csv")

    decision = decide(test_exact, config)
    payload = {
        "decision": decision,
        "selected_candidate": selected,
        "validation_exact": validation_exact,
        "test_exact": test_exact,
        "candidate_count": len(candidates),
        "total_candidate_count": total_candidate_count,
        "candidate_sample_seed": sample_seed,
        "sampled_candidate_count": int(sample_summary["name"].nunique()),
        "exact_top_n": int(config["search"]["exact_top_n"]),
    }
    (output_dir / "selected_monthly_martingale_candidate.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = write_report(output_dir, selected, validation_exact, test_exact, decision)
    LOGGER.info("Wrote monthly martingale research outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded martingale grid monthly-target research iteration.")
    parser.add_argument("--config", default="config/research_iteration_monthly_martingale.yaml")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-sample-positions", type=int, default=None)
    parser.add_argument("--exact-top-n", type=int, default=None)
    parser.add_argument("--candidate-sample-seed", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        max_candidates=args.max_candidates,
        max_sample_positions=args.max_sample_positions,
        exact_top_n=args.exact_top_n,
        candidate_sample_seed=args.candidate_sample_seed,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
