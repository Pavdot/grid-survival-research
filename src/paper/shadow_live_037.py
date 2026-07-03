from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.fundamentals.event_blackout import build_blackout_bundle, load_fundamental_events
from src.infra.binance_kline_collector import (
    audit_kline_frame,
    load_kline_frame,
    resample_closed_1h_from_5m,
    write_kline_frame,
)
from src.infra.binance_microstructure_collector import collector_config_from_yaml, healthcheck as collector_healthcheck
from src.infra.microstructure_quality_report import compute_quality_metrics
from src.paper.paper_trading_harness import assert_paper_only_environment, load_latest_ws_snapshot
from src.regimes.trend_escape import build_trend_escape_components
from src.research.microstructure_execution_filter_017 import gate_config_from_yaml
from src.research.microstructure_order_policy_017 import evaluate_order_gate, order_policy_from_yaml
from src.research.monthly_target_martingale_research import (
    MonthlyMartingaleCandidate,
    build_side_signal,
    candidate_from_row,
    sizing_sequence,
)
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/shadow_live_037.yaml"
TRUTHY = {"1", "true", "yes", "y", "on"}
ACTIONABLE_SIDES = {"long", "short"}


def load_shadow_live_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    if "shadow_live" not in config:
        raise ValueError("shadow live config requires a shadow_live section")
    return config


def as_paper_safety_config(config: dict[str, Any]) -> dict[str, Any]:
    shadow = config["shadow_live"]
    return {
        "paper": {
            "live_trading_env_var": shadow.get("live_trading_env_var", "LIVE_TRADING_ENABLED"),
            "private_env_vars": shadow.get("private_env_vars", []),
        }
    }


def assert_shadow_safe(config: dict[str, Any]) -> None:
    assert_paper_only_environment(as_paper_safety_config(config))
    forbidden = [
        "BINANCE_API_KEY",
        "BINANCE_SECRET_KEY",
        "BINANCE_API_SECRET",
        "BINANCE_PRIVATE_KEY",
    ]
    for env_name in forbidden:
        if os.getenv(env_name):
            raise RuntimeError(f"{env_name} is present; shadow live refuses private API keys")
    if os.getenv(str(config["shadow_live"].get("live_trading_env_var", "LIVE_TRADING_ENABLED")), "").lower() in TRUTHY:
        raise RuntimeError("LIVE_TRADING_ENABLED=true is forbidden in shadow live")


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame


def locked_candidate(config: dict[str, Any]) -> MonthlyMartingaleCandidate:
    raw = dict(config["strategy_037"]["candidate"])
    raw.setdefault("name", config["strategy_037"].get("candidate_name", "shadow_live_037_locked"))
    return candidate_from_row(raw)


def build_live_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    market = frame.copy()
    market.index = pd.to_datetime(market.index, utc=True)
    market = market.sort_index()
    tr = pd.concat(
        [
            market["high"] - market["low"],
            (market["high"] - market["close"].shift(1)).abs(),
            (market["low"] - market["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    market["atr_5m"] = tr.rolling(14, min_periods=14).mean()
    range_size = (market["high"] - market["low"]).replace(0, np.nan)
    range_baseline = range_size.rolling(288, min_periods=48).median().shift(1)
    market["range_expansion_ratio"] = (range_size / range_baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns = np.log(market["close"].astype(float)).diff()
    realized = returns.rolling(12, min_periods=12).std()
    realized_baseline = realized.rolling(288, min_periods=48).median().shift(1)
    market["realized_volatility_ratio"] = (realized / realized_baseline).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market["breakout_risk"] = 0
    market["regime_allows_grid"] = 1
    market["volatility_shock"] = (
        market["range_expansion_ratio"].ge(2.5) | market["realized_volatility_ratio"].ge(2.0)
    ).astype(int)
    return market


def load_live_market_section(
    config: dict[str, Any],
    section: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    shadow = config["shadow_live"]
    raw = config[section]
    kline_path = project_path(raw["output_path"])
    kline_1h_path = project_path(raw["resampled_1h_path"])
    frame = load_kline_frame(kline_path)
    min_coverage = float(shadow.get("min_kline_coverage_rate", 0.99))
    audit = audit_kline_frame(frame, kline_path, min_coverage=min_coverage)
    signal_1h = resample_closed_1h_from_5m(frame)
    write_kline_frame(signal_1h, kline_1h_path)
    return build_live_market_features(frame), signal_1h, audit


def load_live_market(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return load_live_market_section(config, "kline_collector")


def spot_futures_basis_bps(signal_market: pd.DataFrame, execution_market: pd.DataFrame) -> float:
    common = signal_market.index.intersection(execution_market.index)
    if common.empty:
        raise ValueError("spot and futures closed-kline feeds have no common timestamp")
    timestamp = common.max()
    spot = float(signal_market.loc[timestamp, "close"])
    futures = float(execution_market.loc[timestamp, "close"])
    if spot <= 0 or futures <= 0:
        raise ValueError("spot and futures closes must be positive")
    return float((futures / spot - 1.0) * 10000.0)


def load_live_markets(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], float]:
    signal_market, signal_1h, signal_audit = load_live_market_section(config, "kline_collector")
    execution_section = "execution_kline_collector" if "execution_kline_collector" in config else "kline_collector"
    execution_market, _execution_1h, execution_audit = load_live_market_section(config, execution_section)
    basis_bps = spot_futures_basis_bps(signal_market, execution_market)
    return signal_market, execution_market, signal_1h, signal_audit, execution_audit, basis_bps


def validate_fundamental_schedule(
    config: dict[str, Any],
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    blackout = config.get("fundamental_blackout", {})
    horizon_days = float(blackout.get("require_future_scheduled_event_within_days", 45))
    events = load_fundamental_events(config)
    categories = set(str(value) for value in blackout.get("categories", []))
    min_severity = int(blackout.get("min_severity", 1))
    eligible = events[
        events["is_scheduled"].astype(bool)
        & events["severity"].astype(int).ge(min_severity)
        & (events["event_time_utc"] >= now)
    ].copy()
    if categories:
        eligible = eligible[eligible["category"].astype(str).isin(categories)]
    deadline = now + pd.Timedelta(days=horizon_days)
    covered = eligible[eligible["event_time_utc"] <= deadline]
    if covered.empty:
        raise ValueError(f"no eligible scheduled fundamental event within {horizon_days:g} days")
    next_event = covered.sort_values("event_time_utc").iloc[0]
    return {
        "event_count": int(len(events)),
        "eligible_future_event_count": int(len(eligible)),
        "next_event_time_utc": pd.Timestamp(next_event["event_time_utc"]).isoformat(),
        "next_event_category": str(next_event["category"]),
        "next_event_title": str(next_event["title"]),
        "coverage_horizon_days": horizon_days,
    }


def build_entry_mask(market: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    _events, _windows, blackout_masks = build_blackout_bundle(market.index, config)
    trend = build_trend_escape_components(market, config)
    return fundamental_trend_mask(trend["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)


def shifted_signal_and_mask(
    market: pd.DataFrame,
    signal_1h: pd.DataFrame,
    candidate: MonthlyMartingaleCandidate,
    entry_mask: pd.Series,
    signal_lag_bars: int,
    mask_lag_bars: int,
) -> tuple[pd.Series, pd.Series]:
    raw_signal = build_side_signal(market, signal_1h, candidate)
    shifted_signal = raw_signal.reindex(market.index).shift(int(signal_lag_bars))
    shifted_mask = entry_mask.reindex(market.index).fillna(False).astype(bool).shift(int(mask_lag_bars)).fillna(False).astype(bool)
    return shifted_signal, shifted_mask


def processed_signal_keys(output_dir: Path) -> set[str]:
    signals = read_csv_or_empty(output_dir / "shadow_signals.csv")
    if signals.empty or "signal_key" not in signals:
        return set()
    return set(signals["signal_key"].astype(str))


def latest_actionable_signal(
    market: pd.DataFrame,
    shifted_signal: pd.Series,
    shifted_mask: pd.Series,
    now: pd.Timestamp,
    output_dir: Path,
    lookback_bars: int,
    max_entry_delay_seconds: float | None = None,
) -> dict[str, Any] | None:
    if market.empty:
        return None
    now = pd.Timestamp(now).tz_convert("UTC")
    keys = processed_signal_keys(output_dir)
    index = market.index[-int(lookback_bars) :]
    for decision_ts in reversed(index):
        side = shifted_signal.get(decision_ts)
        if side not in ACTIONABLE_SIDES:
            continue
        entry_pos = market.index.get_loc(decision_ts) + 1
        if entry_pos >= len(market.index):
            entry_ts = pd.Timestamp(decision_ts) + pd.Timedelta(minutes=5)
        else:
            entry_ts = market.index[entry_pos]
        if entry_ts <= decision_ts:
            raise ValueError("shadow signal cannot enter on the same bar")
        if now < entry_ts:
            continue
        entry_delay_seconds = float((now - entry_ts) / pd.Timedelta(seconds=1))
        if max_entry_delay_seconds is not None and entry_delay_seconds > float(max_entry_delay_seconds):
            continue
        signal_key = f"{pd.Timestamp(entry_ts).isoformat()}|{side}"
        if signal_key in keys:
            continue
        mask_value = bool(shifted_mask.get(entry_ts, False)) if entry_ts in shifted_mask.index else bool(shifted_mask.get(decision_ts, False))
        return {
            "signal_key": signal_key,
            "decision_timestamp_utc": pd.Timestamp(decision_ts).isoformat(),
            "entry_timestamp_utc": pd.Timestamp(entry_ts).isoformat(),
            "side": str(side),
            "entry_blocked_by_mask": mask_value,
            "signal_lag_bars": int((entry_ts - decision_ts) / pd.Timedelta(minutes=5)),
            "entry_delay_seconds": entry_delay_seconds,
        }
    return None


def order_action_for_side(side: str, intent: str) -> str:
    if intent == "entry":
        return "buy" if side == "long" else "sell"
    if intent == "exit":
        return "sell" if side == "long" else "buy"
    raise ValueError("intent must be entry or exit")


def snapshot_execution_price(snapshot: pd.Series, side: str, intent: str, slippage_bps: float) -> float:
    action = order_action_for_side(side, intent)
    if action == "buy":
        base = float(snapshot["best_ask"])
        return base * (1.0 + float(slippage_bps) / 10000.0)
    base = float(snapshot["best_bid"])
    return base * (1.0 - float(slippage_bps) / 10000.0)


def level_sizes(candidate: MonthlyMartingaleCandidate) -> list[float]:
    return [float(candidate.base_position_size_pct) * float(mult) for mult in sizing_sequence(candidate)]


def position_fills(position: pd.Series | dict[str, Any]) -> list[dict[str, float]]:
    raw = position.get("fills_json", "[]")
    if pd.isna(raw):
        return []
    return [dict(item) for item in json.loads(str(raw))]


def fills_to_json(fills: list[dict[str, float]]) -> str:
    return json.dumps(fills, separators=(",", ":"))


def average_entry(fills: list[dict[str, float]]) -> float:
    qty = sum(float(fill["qty"]) for fill in fills)
    if qty <= 0:
        raise ValueError("position fills must have positive quantity")
    return float(sum(float(fill["price"]) * float(fill["qty"]) for fill in fills) / qty)


def position_unrealized_pct(fills: list[dict[str, float]], mark_price: float, side: str) -> float:
    if side == "short":
        gross = sum(float(fill["qty"]) * (float(fill["price"]) - mark_price) for fill in fills)
    else:
        gross = sum(float(fill["qty"]) * (mark_price - float(fill["price"])) for fill in fills)
    return float(gross)


def realize_position_pct(fills: list[dict[str, float]], exit_price: float, side: str, fee_rate: float) -> tuple[float, float]:
    gross = position_unrealized_pct(fills, exit_price, side)
    entry_fees = sum(float(fill["size_mult"]) for fill in fills) * float(fee_rate)
    exit_fees = sum(float(fill["qty"]) * exit_price for fill in fills) * float(fee_rate)
    return float(gross - entry_fees - exit_fees), float(entry_fees + exit_fees)


def open_position_from_signal(
    signal: dict[str, Any],
    market: pd.DataFrame,
    snapshot: pd.Series,
    config: dict[str, Any],
    gate_result: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidate = locked_candidate(config)
    shadow = config["shadow_live"]
    equity = float(shadow.get("primary_equity_usdt", 10000))
    entry_ts = pd.Timestamp(signal["entry_timestamp_utc"])
    reference_ts = market.index[market.index.searchsorted(entry_ts, side="right") - 1]
    reference_row = market.loc[reference_ts]
    spacing = float(reference_row["atr_5m"]) * float(candidate.spacing_atr_multiplier)
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = float(reference_row["close"]) * 0.001
    side = str(signal["side"])
    entry_price = snapshot_execution_price(snapshot, side, "entry", candidate.slippage_bps)
    size_mult = float(candidate.base_position_size_pct)
    fills = [{"level": 0, "price": entry_price, "size_mult": size_mult, "qty": size_mult / entry_price}]
    position_id = f"pos_{entry_ts.strftime('%Y%m%dT%H%M%SZ')}_{side}"
    max_holding_deadline = entry_ts + pd.Timedelta(hours=float(candidate.max_holding_hours))
    position = {
        "position_id": position_id,
        "status": "open",
        "side": side,
        "entry_timestamp_utc": entry_ts.isoformat(),
        "entry_reference_timestamp_utc": pd.Timestamp(reference_ts).isoformat(),
        "entry_price": entry_price,
        "entry_reference_price": float(reference_row["close"]),
        "spacing": spacing,
        "take_profit_spacing_multiplier": float(candidate.take_profit_spacing_multiplier),
        "levels_filled": 1,
        "max_levels": int(candidate.max_levels),
        "exposure_mult": size_mult,
        "max_exposure_mult": float(candidate.max_total_exposure_pct),
        "account_equity_usdt": equity,
        "fills_json": fills_to_json(fills),
        "last_update_utc": entry_ts.isoformat(),
        "max_holding_deadline_utc": max_holding_deadline.isoformat(),
        "real_order_sent": False,
    }
    order = {
        "position_id": position_id,
        "order_type": "entry",
        "side": side,
        "action": order_action_for_side(side, "entry"),
        "order_timestamp_utc": entry_ts.isoformat(),
        "order_notional_usdt": equity * size_mult,
        "authorized": bool(gate_result["authorized"]),
        "gate_reasons": gate_result.get("reasons", ""),
        "execution_policy": shadow.get("default_policy", "maker_entry_add_taker_exit"),
        "real_order_sent": False,
        **{f"gate_{key}": value for key, value in gate_result.items() if key != "authorized"},
    }
    fill = {
        "position_id": position_id,
        "order_type": "entry",
        "fill_timestamp_utc": entry_ts.isoformat(),
        "fill_price": entry_price,
        "size_mult": size_mult,
        "qty": size_mult / entry_price,
        "fee_rate": float(candidate.fee_rate),
        "slippage_bps": float(candidate.slippage_bps),
        "real_order_sent": False,
    }
    return position, pd.DataFrame([order]), pd.DataFrame([fill])


def update_position_on_closed_bars(
    position: pd.Series,
    market: pd.DataFrame,
    config: dict[str, Any],
    blackout_mask: pd.Series,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate = locked_candidate(config)
    updated = position.to_dict()
    if str(updated.get("status", "open")) != "open":
        return updated, [], []
    side = str(updated["side"])
    fills = position_fills(updated)
    levels = level_sizes(candidate)
    next_level = int(updated.get("levels_filled", len(fills)))
    last_update = pd.Timestamp(updated.get("last_update_utc", updated["entry_timestamp_utc"])).tz_convert("UTC")
    bars = market[market.index > last_update].copy()
    orders: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    exit_reason = ""
    exit_price = np.nan
    exit_ts: pd.Timestamp | None = None
    for ts, row in bars.iterrows():
        add_filled_this_bar = False
        while next_level < int(candidate.max_levels):
            level_price = (
                float(updated["entry_reference_price"]) - float(updated["spacing"]) * next_level
                if side == "long"
                else float(updated["entry_reference_price"]) + float(updated["spacing"]) * next_level
            )
            touched = float(row["low"]) <= level_price if side == "long" else float(row["high"]) >= level_price
            if not touched:
                break
            proposed = float(updated["exposure_mult"]) + float(levels[next_level])
            if proposed > float(candidate.max_total_exposure_pct) + 1e-12:
                exit_reason = "max_exposure"
                exit_price = float(row["close"])
                exit_ts = pd.Timestamp(ts)
                break
            fill_price = level_price * (1.0 + float(candidate.slippage_bps) / 10000.0) if side == "long" else level_price * (1.0 - float(candidate.slippage_bps) / 10000.0)
            size_mult = float(levels[next_level])
            fill = {"level": next_level, "price": fill_price, "size_mult": size_mult, "qty": size_mult / fill_price}
            fills.append(fill)
            updated["exposure_mult"] = proposed
            updated["levels_filled"] = next_level + 1
            add_filled_this_bar = True
            orders.append(
                {
                    "position_id": updated["position_id"],
                    "order_type": "add",
                    "side": side,
                    "action": order_action_for_side(side, "entry"),
                    "order_timestamp_utc": pd.Timestamp(ts).isoformat(),
                    "order_notional_usdt": float(updated["account_equity_usdt"]) * size_mult,
                    "authorized": True,
                    "gate_reasons": "closed_bar_level_touch_proxy",
                    "execution_policy": config["shadow_live"].get("default_policy", "maker_entry_add_taker_exit"),
                    "real_order_sent": False,
                }
            )
            fill_rows.append(
                {
                    "position_id": updated["position_id"],
                    "order_type": "add",
                    "fill_timestamp_utc": pd.Timestamp(ts).isoformat(),
                    "fill_price": fill_price,
                    "size_mult": size_mult,
                    "qty": size_mult / fill_price,
                    "fee_rate": float(candidate.fee_rate),
                    "slippage_bps": float(candidate.slippage_bps),
                    "real_order_sent": False,
                }
            )
            next_level += 1
        if exit_reason:
            break

        mark = float(row["close"])
        unrealized = position_unrealized_pct(fills, mark, side)
        if unrealized <= -float(candidate.max_grid_loss_pct):
            exit_reason = "max_loss"
            exit_price = mark
            exit_ts = pd.Timestamp(ts)
            break
        avg_entry = average_entry(fills)
        take_profit = avg_entry + float(updated["spacing"]) * float(candidate.take_profit_spacing_multiplier) if side == "long" else avg_entry - float(updated["spacing"]) * float(candidate.take_profit_spacing_multiplier)
        take_profit_reached = float(row["high"]) >= take_profit if side == "long" else float(row["low"]) <= take_profit
        if take_profit_reached and not add_filled_this_bar:
            exit_reason = "take_profit"
            exit_price = take_profit
            exit_ts = pd.Timestamp(ts)
            break
        if bool(row.get("volatility_shock", 0)):
            exit_reason = "volatility_shock"
            exit_price = mark
            exit_ts = pd.Timestamp(ts)
            break
        if ts in blackout_mask.index and bool(blackout_mask.loc[ts]):
            exit_reason = "fundamental_blackout"
            exit_price = mark
            exit_ts = pd.Timestamp(ts)
            break
        if pd.Timestamp(ts) >= pd.Timestamp(updated["max_holding_deadline_utc"]):
            exit_reason = "max_holding"
            exit_price = mark
            exit_ts = pd.Timestamp(ts)
            break
        updated["last_update_utc"] = pd.Timestamp(ts).isoformat()
    updated["fills_json"] = fills_to_json(fills)
    if exit_reason and exit_ts is not None:
        realized_pct, fees_pct = realize_position_pct(fills, float(exit_price), side, float(candidate.fee_rate))
        updated.update(
            {
                "status": "closed",
                "exit_timestamp_utc": exit_ts.isoformat(),
                "exit_reason": exit_reason,
                "exit_price": float(exit_price),
                "realized_pnl_pct": realized_pct,
                "realized_pnl_usdt": realized_pct * float(updated["account_equity_usdt"]),
                "fees_pct": fees_pct,
                "last_update_utc": exit_ts.isoformat(),
            }
        )
        orders.append(
            {
                "position_id": updated["position_id"],
                "order_type": "exit",
                "side": side,
                "action": order_action_for_side(side, "exit"),
                "order_timestamp_utc": exit_ts.isoformat(),
                "order_notional_usdt": float(updated["account_equity_usdt"]) * float(updated["exposure_mult"]),
                "authorized": True,
                "gate_reasons": exit_reason,
                "execution_policy": "forced_or_tp_shadow_exit",
                "real_order_sent": False,
            }
        )
        fill_rows.append(
            {
                "position_id": updated["position_id"],
                "order_type": "exit",
                "fill_timestamp_utc": exit_ts.isoformat(),
                "fill_price": float(exit_price),
                "size_mult": float(updated["exposure_mult"]),
                "qty": sum(float(fill["qty"]) for fill in fills),
                "fee_rate": float(candidate.fee_rate),
                "slippage_bps": float(candidate.slippage_bps),
                "real_order_sent": False,
            }
        )
    return updated, orders, fill_rows


def merge_positions(output_dir: Path, updated_positions: list[dict[str, Any]]) -> pd.DataFrame:
    existing = read_csv_or_empty(output_dir / "shadow_positions.csv")
    if existing.empty:
        return pd.DataFrame(updated_positions)
    updated = pd.DataFrame(updated_positions)
    untouched = existing[~existing["position_id"].astype(str).isin(set(updated["position_id"].astype(str)))] if not updated.empty else existing
    return pd.concat([untouched, updated], ignore_index=True)


def update_open_positions(output_dir: Path, market: pd.DataFrame, config: dict[str, Any], blackout_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions = read_csv_or_empty(output_dir / "shadow_positions.csv")
    if positions.empty:
        return positions, pd.DataFrame(), pd.DataFrame()
    updated_positions: list[dict[str, Any]] = []
    all_orders: list[dict[str, Any]] = []
    all_fills: list[dict[str, Any]] = []
    for _, position in positions.iterrows():
        if str(position.get("status", "open")) != "open":
            updated_positions.append(position.to_dict())
            continue
        updated, orders, fills = update_position_on_closed_bars(position, market, config, blackout_mask)
        updated_positions.append(updated)
        all_orders.extend(orders)
        all_fills.extend(fills)
    merged = merge_positions(output_dir, updated_positions)
    return merged, pd.DataFrame(all_orders), pd.DataFrame(all_fills)


def current_open_count(positions: pd.DataFrame) -> int:
    if positions.empty or "status" not in positions:
        return 0
    return int(positions["status"].astype(str).eq("open").sum())


def write_report(output_dir: Path, status: dict[str, Any]) -> Path:
    report = output_dir / "shadow_live_report.md"
    lines = [
        "# Shadow Live 037",
        "",
        f"- status: `{status['status']}`",
        f"- reason: `{status['reason']}`",
        f"- checked_at_utc: `{status['checked_at_utc']}`",
        f"- selected_strategy: `{status['selected_strategy']}`",
        f"- collector_health_ok: `{status['collector_health_ok']}`",
        f"- kline_quality: `{status['kline_quality']}`",
        f"- open_positions: `{status['open_positions']}`",
        f"- latest_signal_side: `{status.get('latest_signal_side', '')}`",
        f"- entry_authorized: `{status.get('entry_authorized', False)}`",
        "",
        "Paper-only: no private endpoints, no API keys, and no real orders.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_daily_outputs(output_dir: Path) -> None:
    fills = read_csv_or_empty(output_dir / "shadow_fills.csv")
    positions = read_csv_or_empty(output_dir / "shadow_positions.csv")
    if not fills.empty and "fill_timestamp_utc" in fills:
        fills["day_utc"] = pd.to_datetime(fills["fill_timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
        costs = fills.groupby("day_utc", as_index=False).agg(
            fill_count=("position_id", "count"),
            median_slippage_bps=("slippage_bps", "median"),
            avg_fee_rate=("fee_rate", "mean"),
        )
    else:
        costs = pd.DataFrame(columns=["day_utc", "fill_count", "median_slippage_bps", "avg_fee_rate"])
    costs.to_csv(output_dir / "execution_costs_by_day.csv", index=False)
    if not positions.empty and "entry_timestamp_utc" in positions:
        positions["day_utc"] = pd.to_datetime(positions["entry_timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")
        daily = positions.groupby("day_utc", as_index=False).agg(
            opened_positions=("position_id", "count"),
            closed_positions=("status", lambda value: int(pd.Series(value).astype(str).eq("closed").sum())),
            paper_pnl_usdt=("realized_pnl_usdt", lambda value: float(pd.to_numeric(value, errors="coerce").fillna(0.0).sum())),
        )
    else:
        daily = pd.DataFrame(columns=["day_utc", "opened_positions", "closed_positions", "paper_pnl_usdt"])
    daily.to_csv(output_dir / "shadow_daily_pnl.csv", index=False)


def run_shadow_once(config: dict[str, Any]) -> dict[str, Any]:
    assert_shadow_safe(config)
    shadow = config["shadow_live"]
    output_dir = project_path(shadow.get("output_dir", "reports/shadow_live_037"))
    output_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    candidate = locked_candidate(config)
    status: dict[str, Any] = {
        "checked_at_utc": now.isoformat(),
        "status": "shadow_ready",
        "reason": "ready",
        "selected_strategy": "037_surgical_veto_zero_fee",
        "selected_candidate": candidate.name,
        "collector_health_ok": False,
        "collector_health_message": "",
        "kline_quality": "unknown",
        "open_positions": 0,
        "entry_authorized": False,
        "real_order_sent": False,
    }

    try:
        signal_market, execution_market, signal_1h, signal_audit, execution_audit, basis_bps = load_live_markets(config)
        status["kline_quality"] = signal_audit["status"]
        status["kline_coverage_rate"] = signal_audit.get("coverage_rate")
        status["execution_kline_quality"] = execution_audit["status"]
        status["execution_kline_coverage_rate"] = execution_audit.get("coverage_rate")
        status["spot_futures_basis_bps"] = basis_bps
        latest_kline_age_minutes = (
            (now - pd.Timestamp(signal_market.index.max())) / pd.Timedelta(minutes=1)
            if not signal_market.empty
            else float("inf")
        )
        latest_execution_kline_age_minutes = (
            (now - pd.Timestamp(execution_market.index.max())) / pd.Timedelta(minutes=1)
            if not execution_market.empty
            else float("inf")
        )
        status["latest_kline_age_minutes"] = float(latest_kline_age_minutes)
        status["latest_execution_kline_age_minutes"] = float(latest_execution_kline_age_minutes)
        if signal_audit["status"] == "bad" or execution_audit["status"] == "bad":
            status["status"] = "shadow_kill_switch"
            status["reason"] = "bad_kline_quality"
        elif max(latest_kline_age_minutes, latest_execution_kline_age_minutes) > float(shadow.get("max_kline_age_minutes", 15)):
            status["status"] = "shadow_kill_switch"
            status["reason"] = "stale_kline_data"
        elif abs(basis_bps) > float(shadow.get("max_abs_spot_futures_basis_bps", 10)):
            status["status"] = "shadow_kill_switch"
            status["reason"] = "spot_futures_basis_out_of_bounds"
    except Exception as exc:  # noqa: BLE001 - status file should explain failure.
        status["status"] = "shadow_kill_switch"
        status["reason"] = "kline_load_failed"
        status["error"] = str(exc)
        signal_market = pd.DataFrame()
        execution_market = pd.DataFrame()
        signal_1h = pd.DataFrame()

    try:
        schedule_status = validate_fundamental_schedule(config, now=now)
        status["fundamental_schedule"] = schedule_status
    except Exception as exc:  # noqa: BLE001 - fail closed when the calendar is not maintained.
        status["fundamental_schedule"] = {"error": str(exc)}
        if status["status"] == "shadow_ready":
            status["status"] = "shadow_kill_switch"
            status["reason"] = "fundamental_schedule_invalid"

    infra_config = load_yaml(shadow["infrastructure_config"])
    collector = collector_config_from_yaml(infra_config)
    health_ok, health_message = collector_healthcheck(collector, max_age_seconds=float(shadow.get("max_collector_age_seconds", 20)))
    status["collector_health_ok"] = bool(health_ok)
    status["collector_health_message"] = health_message
    if status["status"] == "shadow_ready" and not health_ok:
        status["status"] = "shadow_kill_switch"
        status["reason"] = "stale_or_unhealthy_collector"

    entry_mask = (
        pd.Series(False, index=signal_market.index, dtype=bool)
        if signal_market.empty
        else build_entry_mask(signal_market, config)
    )
    _events, _windows, blackout_masks = (
        build_blackout_bundle(execution_market.index, config)
        if not execution_market.empty
        else (pd.DataFrame(), pd.DataFrame(), {"realistic": pd.Series(dtype=bool)})
    )
    positions, position_orders, position_fills = update_open_positions(
        output_dir,
        execution_market,
        config,
        blackout_masks.get("realistic", pd.Series(False, index=execution_market.index)),
    )
    status["open_positions"] = current_open_count(positions)
    if not positions.empty:
        positions.to_csv(output_dir / "shadow_positions.csv", index=False)
    if not position_orders.empty:
        append_csv(output_dir / "shadow_orders.csv", position_orders)
    if not position_fills.empty:
        append_csv(output_dir / "shadow_fills.csv", position_fills)

    signal_rows: list[dict[str, Any]] = []
    order_rows = pd.DataFrame()
    fill_rows = pd.DataFrame()
    if status["status"] == "shadow_ready" and current_open_count(positions) == 0 and not signal_market.empty and not signal_1h.empty:
        shifted_signal, shifted_mask = shifted_signal_and_mask(
            signal_market,
            signal_1h,
            candidate,
            entry_mask,
            int(shadow.get("signal_lag_bars", 1)),
            int(shadow.get("mask_lag_bars", 1)),
        )
        signal = latest_actionable_signal(
            signal_market,
            shifted_signal,
            shifted_mask,
            now,
            output_dir,
            int(shadow.get("signal_lookback_bars", 36)),
            float(shadow.get("max_entry_delay_seconds", 120)),
        )
        if signal is not None:
            status["latest_signal_side"] = signal["side"]
            signal_rows.append({**signal, "checked_at_utc": now.isoformat(), "real_order_sent": False})
            if bool(signal["entry_blocked_by_mask"]):
                status["status"] = "entries_blocked"
                status["reason"] = "fundamental_or_trend_mask"
            else:
                snapshot, recent = load_latest_ws_snapshot(infra_config)
                quality = compute_quality_metrics(recent.tail(min(len(recent), 3600)), infra_config)
                status["microstructure_quality"] = quality["quality_score"]
                if quality["quality_score"] == "bad":
                    status["status"] = "shadow_kill_switch"
                    status["reason"] = "bad_microstructure_quality"
                else:
                    gate = gate_config_from_yaml(load_yaml(shadow["policy_config"]))
                    policy = order_policy_from_yaml(load_yaml(shadow["policy_config"]))
                    equity = float(shadow.get("primary_equity_usdt", 10000))
                    notional = equity * float(candidate.base_position_size_pct)
                    snapshot_time = pd.Timestamp(snapshot["snapshot_time_utc"])
                    snapshot_age_ms = abs((now - snapshot_time) / pd.Timedelta(milliseconds=1))
                    scheduled_entry_time = pd.Timestamp(signal["entry_timestamp_utc"])
                    if snapshot_time < scheduled_entry_time:
                        status["status"] = "entries_blocked"
                        status["reason"] = "snapshot_precedes_scheduled_entry"
                        gate_result = {"authorized": False, "reasons": "snapshot_precedes_scheduled_entry"}
                    else:
                        gate_result = evaluate_order_gate(
                            snapshot,
                            order_action_for_side(signal["side"], "entry"),
                            equity,
                            notional,
                            gate,
                            policy.max_total_book_share,
                            snapshot_age_ms=snapshot_age_ms,
                        )
                    status["entry_authorized"] = bool(gate_result["authorized"])
                    status["entry_gate_reasons"] = gate_result.get("reasons", "")
                    if bool(gate_result["authorized"]):
                        position, order_rows, fill_rows = open_position_from_signal(
                            signal,
                            execution_market,
                            snapshot,
                            config,
                            gate_result,
                        )
                        positions = merge_positions(output_dir, [position])
                        positions.to_csv(output_dir / "shadow_positions.csv", index=False)
                        append_csv(output_dir / "shadow_orders.csv", order_rows)
                        append_csv(output_dir / "shadow_fills.csv", fill_rows)
                        status["open_positions"] = current_open_count(positions)
                    else:
                        status["status"] = "entries_blocked"
                        status["reason"] = "microstructure_gate_blocked_entry"
                        blocked = pd.DataFrame(
                            [
                                {
                                    "position_id": "",
                                    "order_type": "entry",
                                    "side": signal["side"],
                                    "action": order_action_for_side(signal["side"], "entry"),
                                    "order_timestamp_utc": signal["entry_timestamp_utc"],
                                    "order_notional_usdt": notional,
                                    "authorized": False,
                                    "gate_reasons": gate_result.get("reasons", ""),
                                    "real_order_sent": False,
                                }
                            ]
                        )
                        append_csv(output_dir / "shadow_orders.csv", blocked)

    if signal_rows:
        append_csv(output_dir / "shadow_signals.csv", pd.DataFrame(signal_rows))
    if status["status"] == "shadow_ready" and not signal_rows:
        status["reason"] = "no_new_actionable_signal"
    status_path = output_dir / "shadow_status.json"
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    write_daily_outputs(output_dir)
    report = write_report(output_dir, status)
    return {
        "status": status,
        "output_paths": {
            "signals": str(output_dir / "shadow_signals.csv"),
            "orders": str(output_dir / "shadow_orders.csv"),
            "fills": str(output_dir / "shadow_fills.csv"),
            "positions": str(output_dir / "shadow_positions.csv"),
            "daily_pnl": str(output_dir / "shadow_daily_pnl.csv"),
            "execution_costs": str(output_dir / "execution_costs_by_day.csv"),
            "status": str(status_path),
            "report": str(report),
        },
    }


def shadow_healthcheck(
    config: dict[str, Any],
    max_age_seconds: float = 120,
    now: pd.Timestamp | None = None,
) -> tuple[bool, str]:
    output_dir = project_path(config["shadow_live"].get("output_dir", "reports/shadow_live_037"))
    status_path = output_dir / "shadow_status.json"
    if not status_path.exists():
        return False, f"shadow status file missing: {status_path}"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid shadow status json: {exc}"
    if bool(payload.get("real_order_sent", False)):
        return False, "shadow safety violation: real_order_sent is true"
    checked_at = payload.get("checked_at_utc")
    if not checked_at:
        return False, "shadow status has no checked_at_utc"
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    age_seconds = float((now - pd.Timestamp(checked_at)) / pd.Timedelta(seconds=1))
    if age_seconds > float(max_age_seconds):
        return False, f"shadow cycle is stale: {age_seconds:.1f}s > {max_age_seconds:.1f}s"
    return True, f"shadow runner alive; state={payload.get('status', 'unknown')}"


def run_shadow_loop(
    config: dict[str, Any],
    interval_seconds: float | None = None,
    collect_seconds: float | None = None,
) -> None:
    assert_shadow_safe(config)
    interval = float(interval_seconds or config["shadow_live"].get("cycle_interval_seconds", 30))
    if interval <= 0:
        raise ValueError("shadow cycle interval must be positive")
    deadline = time.monotonic() + float(collect_seconds) if collect_seconds is not None else None
    while True:
        payload = run_shadow_once(config)
        print(json.dumps(payload, default=str), flush=True)
        if deadline is not None and time.monotonic() >= deadline:
            return
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 037 shadow-live paper-only checks.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=None)
    parser.add_argument("--collect-seconds", type=float, default=None)
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--max-age-seconds", type=float, default=120)
    parser.add_argument("--daily-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_shadow_live_config(args.config)
    if args.healthcheck:
        ok, message = shadow_healthcheck(config, max_age_seconds=args.max_age_seconds)
        print(message)
        raise SystemExit(0 if ok else 1)
    if args.daily_report:
        output_dir = project_path(config["shadow_live"].get("output_dir", "reports/shadow_live_037"))
        write_daily_outputs(output_dir)
        print(json.dumps({"daily_report": str(output_dir / "shadow_daily_pnl.csv")}, indent=2))
        return
    if args.run:
        run_shadow_loop(config, interval_seconds=args.interval_seconds, collect_seconds=args.collect_seconds)
        return
    if not args.run_once:
        raise SystemExit("Choose --run, --run-once, --healthcheck, or --daily-report")
    payload = run_shadow_once(config)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
