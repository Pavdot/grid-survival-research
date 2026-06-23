from __future__ import annotations

import argparse
import json
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.microstructure_execution_filter_017 import (
    MicrostructureGateConfig,
    book_levels_for_side,
    collect_once,
    gate_config_from_yaml,
    load_locked_017_candidates,
    load_snapshots,
    summarize_snapshots,
    validate_microstructure_config,
    walk_book_slippage_bps,
)
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_microstructure_order_policy_017.yaml"
FORCED_EXIT_REASONS = {
    "max_holding",
    "max_loss",
    "max_exposure",
    "volatility_shock",
    "regime_break",
    "fundamental_blackout",
    "kill_switch",
    "end_of_data",
}


@dataclass(frozen=True)
class OrderPolicyConfig:
    policies: tuple[str, ...]
    max_clip_book_share: float
    max_total_book_share: float
    maker_ttl_seconds: float
    max_reprices: int
    synthetic_mapping_seed: int


@dataclass
class SnapshotCache:
    frame: pd.DataFrame
    times_ns: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    hour_positions: dict[int, np.ndarray]

    @classmethod
    def from_frame(cls, snapshots: pd.DataFrame) -> "SnapshotCache":
        frame = snapshots.sort_values("snapshot_time_utc").reset_index(drop=True).copy()
        frame["snapshot_time_utc"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
        times_ns = np.array([pd.Timestamp(value).value for value in frame["snapshot_time_utc"]], dtype=np.int64)
        hour_positions: dict[int, np.ndarray] = {}
        hours = frame["snapshot_time_utc"].dt.hour.to_numpy()
        for hour in range(24):
            hour_positions[hour] = np.flatnonzero(hours == hour)
        return cls(
            frame=frame,
            times_ns=times_ns,
            best_bid=frame["best_bid"].astype(float).to_numpy(),
            best_ask=frame["best_ask"].astype(float).to_numpy(),
            hour_positions=hour_positions,
        )

    def choose_event_snapshot(
        self,
        event: pd.Series,
        gate: MicrostructureGateConfig,
        seed: int,
    ) -> tuple[pd.Series, int, str, float]:
        if self.frame.empty:
            raise ValueError("snapshots cannot be empty")
        event_ts = pd.Timestamp(event["event_timestamp"])
        event_ns = int(event_ts.value)
        tolerance_ns = int(pd.Timedelta(milliseconds=gate.max_snapshot_age_ms).value)
        pos = int(np.searchsorted(self.times_ns, event_ns, side="right") - 1)
        if pos >= 0 and event_ns - int(self.times_ns[pos]) <= tolerance_ns:
            snapshot = self.frame.iloc[pos]
            age_ms = (event_ns - int(self.times_ns[pos])) / 1_000_000.0
            return snapshot, pos, "exact_signal_timestamp", float(age_ms)
        positions = self.hour_positions.get(int(event_ts.hour), np.array([], dtype=int))
        key = f"{event.get('event_id', '')}:{event_ts.isoformat()}"
        if len(positions) == 0:
            selected = _stable_index(key, len(self.frame), seed)
        else:
            selected = int(positions[_stable_index(key, len(positions), seed)])
        return self.frame.iloc[selected], selected, "synthetic_microstructure_mapping", float(
            self.frame.iloc[selected]["source_latency_ms"]
        )

    def find_maker_fill(
        self,
        start_pos: int,
        action: str,
        ttl_seconds: float,
        max_reprices: int,
    ) -> tuple[bool, int | None, int, float]:
        if start_pos < 0 or start_pos >= len(self.frame):
            raise IndexError("start_pos out of bounds")
        ttl_ns = int(pd.Timedelta(seconds=ttl_seconds).value)
        current_pos = int(start_pos)
        attempts = 0
        while attempts <= max_reprices and current_pos < len(self.frame):
            attempts += 1
            limit_price = self.best_bid[current_pos] if action == "buy" else self.best_ask[current_pos]
            deadline_ns = int(self.times_ns[current_pos]) + ttl_ns
            end_pos = int(np.searchsorted(self.times_ns, deadline_ns, side="right"))
            for future_pos in range(current_pos + 1, end_pos):
                if action == "buy" and self.best_ask[future_pos] <= limit_price:
                    waited = (int(self.times_ns[future_pos]) - int(self.times_ns[current_pos])) / 1_000_000_000.0
                    return True, int(future_pos), attempts, float(waited)
                if action == "sell" and self.best_bid[future_pos] >= limit_price:
                    waited = (int(self.times_ns[future_pos]) - int(self.times_ns[current_pos])) / 1_000_000_000.0
                    return True, int(future_pos), attempts, float(waited)
            next_pos = int(np.searchsorted(self.times_ns, deadline_ns, side="right"))
            if next_pos >= len(self.frame):
                break
            current_pos = next_pos
        return False, None, attempts, float("nan")


def order_policy_from_yaml(config: dict[str, Any]) -> OrderPolicyConfig:
    raw = config["order_policy"]
    return OrderPolicyConfig(
        policies=tuple(str(value) for value in raw["policies"]),
        max_clip_book_share=float(raw["max_clip_book_share"]),
        max_total_book_share=float(raw["max_total_book_share"]),
        maker_ttl_seconds=float(raw["maker_ttl_seconds"]),
        max_reprices=int(raw["max_reprices"]),
        synthetic_mapping_seed=int(raw.get("synthetic_mapping_seed", 22017)),
    )


def validate_order_policy_config(config: dict[str, Any]) -> None:
    validate_microstructure_config(config)
    policy = order_policy_from_yaml(config)
    allowed = {
        "gate_only_taker",
        "sliced_taker",
        "maker_entry_add_taker_exit",
        "maker_all_non_forced",
    }
    if not policy.policies or any(value not in allowed for value in policy.policies):
        raise ValueError("order_policy.policies contains an unsupported policy")
    if policy.max_clip_book_share <= 0 or policy.max_clip_book_share > 1:
        raise ValueError("order_policy.max_clip_book_share must stay within (0, 1]")
    if policy.max_total_book_share <= 0 or policy.max_total_book_share > 1:
        raise ValueError("order_policy.max_total_book_share must stay within (0, 1]")
    if policy.max_clip_book_share > policy.max_total_book_share:
        raise ValueError("order_policy.max_clip_book_share cannot exceed max_total_book_share")
    if policy.maker_ttl_seconds <= 0:
        raise ValueError("order_policy.maker_ttl_seconds must be positive")
    if policy.max_reprices < 0:
        raise ValueError("order_policy.max_reprices must be non-negative")
    collection = config["collection"]
    endpoints = [str(collection["depth_endpoint"]), str(collection["book_ticker_endpoint"])]
    if any("order" in endpoint.lower() for endpoint in endpoints):
        raise ValueError("Iteration 022 must use public market-data endpoints only")


def side_depth_column_for_action(action: str, band_bps: float) -> str:
    label = f"{band_bps:g}".replace(".", "p")
    if action == "buy":
        return f"ask_depth_{label}bps_usdt"
    if action == "sell":
        return f"bid_depth_{label}bps_usdt"
    raise ValueError("action must be buy or sell")


def levels_for_action(snapshot: pd.Series, action: str) -> list[tuple[float, float]]:
    if action == "buy":
        return book_levels_for_side(snapshot, "long")
    if action == "sell":
        return book_levels_for_side(snapshot, "short")
    raise ValueError("action must be buy or sell")


def best_limit_price(snapshot: pd.Series, action: str) -> float:
    if action == "buy":
        return float(snapshot["best_bid"])
    if action == "sell":
        return float(snapshot["best_ask"])
    raise ValueError("action must be buy or sell")


def maker_would_fill(snapshot: pd.Series, action: str, limit_price: float) -> bool:
    if action == "buy":
        return float(snapshot["best_ask"]) <= float(limit_price)
    if action == "sell":
        return float(snapshot["best_bid"]) >= float(limit_price)
    raise ValueError("action must be buy or sell")


def split_order_notional(order_notional: float, side_depth: float, max_clip_book_share: float) -> list[float]:
    if order_notional <= 0:
        raise ValueError("order_notional must be positive")
    if side_depth <= 0 or not np.isfinite(side_depth):
        return []
    if max_clip_book_share <= 0 or max_clip_book_share > 1:
        raise ValueError("max_clip_book_share must stay within (0, 1]")
    max_clip = float(side_depth) * float(max_clip_book_share)
    if max_clip <= 0:
        return []
    clips: list[float] = []
    remaining = float(order_notional)
    while remaining > 1e-9:
        clip = min(max_clip, remaining)
        clips.append(float(clip))
        remaining -= clip
    return clips


def evaluate_order_gate(
    snapshot: pd.Series,
    action: str,
    account_equity_usdt: float,
    order_notional_usdt: float,
    gate: MicrostructureGateConfig,
    max_total_book_share: float,
    snapshot_age_ms: float | None = None,
) -> dict[str, Any]:
    if account_equity_usdt <= 0 or order_notional_usdt <= 0:
        raise ValueError("account_equity_usdt and order_notional_usdt must be positive")
    side_depth = float(snapshot.get(side_depth_column_for_action(action, gate.depth_band_bps), np.nan))
    spread = float(snapshot.get("spread_bps", np.nan))
    imbalance = float(snapshot.get("depth_imbalance_5bps", np.nan))
    age_ms = float(snapshot_age_ms if snapshot_age_ms is not None else snapshot.get("source_latency_ms", np.nan))
    total_depth = float(snapshot.get("bid_depth_5bps_usdt", 0.0)) + float(snapshot.get("ask_depth_5bps_usdt", 0.0))
    min_required_depth = max(float(gate.min_side_depth_usdt_floor), order_notional_usdt / max_total_book_share)
    order_book_share = order_notional_usdt / side_depth if np.isfinite(side_depth) and side_depth > 0 else float("inf")
    slippage_bps = walk_book_slippage_bps(levels_for_action(snapshot, action), order_notional_usdt, action)
    reasons: list[str] = []
    if not np.isfinite(age_ms) or age_ms > gate.max_snapshot_age_ms:
        reasons.append("stale_snapshot")
    if not np.isfinite(spread) or spread > gate.max_spread_bps:
        reasons.append("wide_spread")
    if not np.isfinite(side_depth) or side_depth < min_required_depth:
        reasons.append("insufficient_depth")
    if not np.isfinite(order_book_share) or order_book_share > max_total_book_share:
        reasons.append("order_too_large_for_book")
    if not np.isfinite(imbalance) or abs(imbalance) > gate.max_abs_depth_imbalance:
        reasons.append("violent_imbalance")
    if not np.isfinite(slippage_bps) or slippage_bps > gate.max_theoretical_slippage_bps:
        reasons.append("theoretical_slippage_breach")
    if total_depth <= 0:
        reasons.append("empty_depth")
    return {
        "authorized": len(reasons) == 0,
        "reject_reasons": ",".join(reasons),
        "reject_reason_primary": reasons[0] if reasons else "",
        "side_depth_usdt": float(side_depth),
        "order_book_share": float(order_book_share),
        "theoretical_slippage_bps": float(slippage_bps),
        "spread_bps": spread,
        "depth_imbalance_5bps": imbalance,
        "snapshot_age_ms": age_ms,
    }


def action_for_order(position_side: str, event_type: str) -> str:
    if position_side not in {"long", "short"}:
        raise ValueError("position_side must be long or short")
    if event_type in {"entry", "add"}:
        return "buy" if position_side == "long" else "sell"
    if event_type in {"take_profit", "forced_exit"}:
        return "sell" if position_side == "long" else "buy"
    raise ValueError("Unsupported event_type")


def load_locked_017_trades(config: dict[str, Any], max_trades: int | None = None) -> pd.DataFrame:
    source_dir = project_path(config["source_iteration"]["output_dir"])
    variant = str(config["source_iteration"]["variant"])
    trades_dir = source_dir / "selected_fold_trades"
    paths = sorted(trades_dir.glob(f"{variant}_fold_*_trades.csv"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        match = re.search(r"_fold_(\d+)_trades\.csv$", path.name)
        fold_id = int(match.group(1)) if match else len(frames) + 1
        if "fold_id" not in frame.columns:
            frame.insert(0, "fold_id", fold_id)
        frames.append(frame)
    if not frames:
        from src.research.microstructure_execution_filter_017 import build_locked_017_trades

        frame = build_locked_017_trades(config)
    else:
        frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise ValueError("No locked Iteration 017 trades available")
    frame["start_timestamp"] = pd.to_datetime(frame["start_timestamp"], utc=True)
    frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True)
    frame = frame.sort_values(["fold_id", "start_timestamp"]).reset_index(drop=True)
    frame.insert(0, "trade_id", np.arange(len(frame), dtype=int))
    if max_trades is not None:
        frame = frame.head(int(max_trades)).copy()
    return frame


def order_events_from_trades(trades: pd.DataFrame, equities: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        levels_filled = max(1, int(float(trade.get("number_of_levels_filled", 1))))
        base_multiplier = float(trade["base_position_size_pct"])
        progression = float(trade["progression_multiplier"])
        cap_multiplier = float(trade["max_total_exposure_pct"])
        position_side = str(trade["side"])
        exit_reason = str(trade["exit_reason"])
        exit_event = "take_profit" if exit_reason == "take_profit" else "forced_exit"
        for equity in equities:
            cumulative_multiplier = 0.0
            level_multipliers: list[float] = []
            for level in range(levels_filled):
                level_multiplier = base_multiplier * float(progression**level)
                if cumulative_multiplier + level_multiplier > cap_multiplier + 1e-12:
                    break
                level_multipliers.append(level_multiplier)
                cumulative_multiplier += level_multiplier
            if not level_multipliers:
                continue
            for level, multiplier in enumerate(level_multipliers):
                event_type = "entry" if level == 0 else "add"
                rows.append(
                    {
                        "trade_id": int(trade["trade_id"]),
                        "fold_id": int(trade["fold_id"]),
                        "event_id": f"{int(trade['trade_id'])}_{event_type}_{level}_{equity:g}",
                        "event_timestamp": trade["start_timestamp"],
                        "event_type": event_type,
                        "level_index": int(level),
                        "position_side": position_side,
                        "action": action_for_order(position_side, event_type),
                        "account_equity_usdt": float(equity),
                        "order_notional_usdt": float(equity * multiplier),
                        "order_notional_multiplier": float(multiplier),
                        "candidate_name": trade.get("name", ""),
                        "exit_reason": exit_reason,
                    }
                )
            rows.append(
                {
                    "trade_id": int(trade["trade_id"]),
                    "fold_id": int(trade["fold_id"]),
                    "event_id": f"{int(trade['trade_id'])}_{exit_event}_{equity:g}",
                    "event_timestamp": trade["exit_timestamp"],
                    "event_type": exit_event,
                    "level_index": levels_filled,
                    "position_side": position_side,
                    "action": action_for_order(position_side, exit_event),
                    "account_equity_usdt": float(equity),
                    "order_notional_usdt": float(equity * cumulative_multiplier),
                    "order_notional_multiplier": float(cumulative_multiplier),
                    "candidate_name": trade.get("name", ""),
                    "exit_reason": exit_reason,
                }
            )
    events = pd.DataFrame(rows)
    if not events.empty:
        events["event_timestamp"] = pd.to_datetime(events["event_timestamp"], utc=True)
    return events


def _stable_index(key: str, length: int, seed: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    raw = zlib.crc32(f"{seed}:{key}".encode("utf-8"))
    return int(raw % length)


def choose_event_snapshot(
    event: pd.Series,
    snapshots: pd.DataFrame,
    gate: MicrostructureGateConfig,
    seed: int,
) -> tuple[pd.Series, int, str, float]:
    return SnapshotCache.from_frame(snapshots).choose_event_snapshot(event, gate, seed)


def simulate_taker_order(
    snapshot: pd.Series,
    action: str,
    order_notional: float,
    equity: float,
    gate: MicrostructureGateConfig,
    max_total_book_share: float,
    snapshot_age_ms: float,
    force_execute: bool = False,
    precomputed_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_result = dict(precomputed_gate) if precomputed_gate is not None else evaluate_order_gate(
        snapshot,
        action,
        equity,
        order_notional,
        gate,
        max_total_book_share,
        snapshot_age_ms=snapshot_age_ms,
    )
    executed = bool(gate_result["authorized"] or force_execute)
    spread_cost = float(order_notional / equity) * max(float(gate_result["spread_bps"]), 0.0) / 2.0 / 10000.0
    impact_cost = (
        float(order_notional / equity) * max(float(gate_result["theoretical_slippage_bps"]), 0.0) / 10000.0
        if np.isfinite(float(gate_result["theoretical_slippage_bps"]))
        else float("inf")
    )
    return {
        **gate_result,
        "execution_style": "taker",
        "executed": executed,
        "skipped": not executed,
        "fill_attempts": 1,
        "fill_price_type": "walk_the_book",
        "market_impact_bps": float(gate_result["theoretical_slippage_bps"]),
        "spread_cost_pct_equity": spread_cost if executed else 0.0,
        "market_impact_cost_pct_equity": impact_cost if executed else 0.0,
        "execution_cost_pct_equity": spread_cost + impact_cost if executed and np.isfinite(impact_cost) else 0.0,
    }


def simulate_sliced_taker_order(
    snapshot: pd.Series,
    action: str,
    order_notional: float,
    equity: float,
    gate: MicrostructureGateConfig,
    policy: OrderPolicyConfig,
    snapshot_age_ms: float,
    force_execute: bool = False,
    precomputed_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_result = dict(precomputed_gate) if precomputed_gate is not None else evaluate_order_gate(
        snapshot,
        action,
        equity,
        order_notional,
        gate,
        policy.max_total_book_share,
        snapshot_age_ms=snapshot_age_ms,
    )
    if not gate_result["authorized"] and not force_execute:
        return {
            **gate_result,
            "execution_style": "sliced_taker",
            "executed": False,
            "skipped": True,
            "fill_attempts": 0,
            "fill_price_type": "none",
            "market_impact_bps": float("inf"),
            "spread_cost_pct_equity": 0.0,
            "market_impact_cost_pct_equity": 0.0,
            "execution_cost_pct_equity": 0.0,
        }
    clips = split_order_notional(order_notional, float(gate_result["side_depth_usdt"]), policy.max_clip_book_share)
    if not clips:
        gate_result["reject_reasons"] = ",".join(filter(None, [gate_result["reject_reasons"], "empty_slices"]))
        gate_result["reject_reason_primary"] = gate_result["reject_reason_primary"] or "empty_slices"
        return {
            **gate_result,
            "execution_style": "sliced_taker",
            "executed": False,
            "skipped": True,
            "fill_attempts": 0,
            "fill_price_type": "none",
            "market_impact_bps": float("inf"),
            "spread_cost_pct_equity": 0.0,
            "market_impact_cost_pct_equity": 0.0,
            "execution_cost_pct_equity": 0.0,
        }
    levels = levels_for_action(snapshot, action)
    clip_slippages = [walk_book_slippage_bps(levels, clip, action) for clip in clips]
    weighted_slippage = float(np.average(clip_slippages, weights=clips))
    spread_cost = float(order_notional / equity) * max(float(gate_result["spread_bps"]), 0.0) / 2.0 / 10000.0
    impact_cost = float(order_notional / equity) * max(weighted_slippage, 0.0) / 10000.0
    return {
        **gate_result,
        "execution_style": "sliced_taker",
        "executed": True,
        "skipped": False,
        "fill_attempts": len(clips),
        "fill_price_type": "sliced_walk_the_book",
        "market_impact_bps": weighted_slippage,
        "spread_cost_pct_equity": spread_cost,
        "market_impact_cost_pct_equity": impact_cost,
        "execution_cost_pct_equity": spread_cost + impact_cost,
    }


def find_maker_fill(
    snapshots: pd.DataFrame,
    start_pos: int,
    action: str,
    ttl_seconds: float,
    max_reprices: int,
) -> tuple[bool, int | None, int, float]:
    return SnapshotCache.from_frame(snapshots).find_maker_fill(start_pos, action, ttl_seconds, max_reprices)


def simulate_maker_order(
    snapshot: pd.Series,
    snapshot_pos: int,
    snapshots: pd.DataFrame,
    action: str,
    order_notional: float,
    equity: float,
    gate: MicrostructureGateConfig,
    policy: OrderPolicyConfig,
    snapshot_age_ms: float,
    snapshot_cache: SnapshotCache | None = None,
    precomputed_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate_result = dict(precomputed_gate) if precomputed_gate is not None else evaluate_order_gate(
        snapshot,
        action,
        equity,
        order_notional,
        gate,
        policy.max_total_book_share,
        snapshot_age_ms=snapshot_age_ms,
    )
    if not gate_result["authorized"]:
        return {
            **gate_result,
            "execution_style": "maker",
            "executed": False,
            "skipped": True,
            "fill_attempts": 0,
            "fill_price_type": "none",
            "market_impact_bps": 0.0,
            "spread_cost_pct_equity": 0.0,
            "market_impact_cost_pct_equity": 0.0,
            "execution_cost_pct_equity": 0.0,
            "maker_wait_seconds": float("nan"),
        }
    if snapshot_cache is None:
        filled, fill_pos, attempts, waited = find_maker_fill(
            snapshots,
            snapshot_pos,
            action,
            policy.maker_ttl_seconds,
            policy.max_reprices,
        )
    else:
        filled, fill_pos, attempts, waited = snapshot_cache.find_maker_fill(
            snapshot_pos,
            action,
            policy.maker_ttl_seconds,
            policy.max_reprices,
        )
    return {
        **gate_result,
        "execution_style": "maker",
        "executed": bool(filled),
        "skipped": not bool(filled),
        "fill_attempts": int(attempts),
        "fill_price_type": "maker_best_bid_ask" if filled else "none",
        "market_impact_bps": 0.0,
        "spread_cost_pct_equity": 0.0,
        "market_impact_cost_pct_equity": 0.0,
        "execution_cost_pct_equity": 0.0,
        "maker_wait_seconds": waited,
        "fill_snapshot_pos": fill_pos if fill_pos is not None else -1,
    }


def policy_uses_maker(policy_name: str, event_type: str) -> bool:
    if policy_name == "maker_all_non_forced":
        return event_type != "forced_exit"
    if policy_name == "maker_entry_add_taker_exit":
        return event_type in {"entry", "add"}
    return False


def cached_order_gate(
    gate_cache: dict[tuple[Any, ...], dict[str, Any]] | None,
    snapshot_pos: int,
    snapshot: pd.Series,
    action: str,
    equity: float,
    order_notional: float,
    gate: MicrostructureGateConfig,
    policy: OrderPolicyConfig,
    snapshot_age_ms: float,
) -> dict[str, Any]:
    key = (
        int(snapshot_pos),
        str(action),
        round(float(equity), 8),
        round(float(order_notional), 8),
        round(float(snapshot_age_ms), 3),
    )
    if gate_cache is not None and key in gate_cache:
        return dict(gate_cache[key])
    result = evaluate_order_gate(
        snapshot,
        action,
        equity,
        order_notional,
        gate,
        policy.max_total_book_share,
        snapshot_age_ms=snapshot_age_ms,
    )
    if gate_cache is not None:
        gate_cache[key] = dict(result)
    return result


def simulate_policy_order(
    policy_name: str,
    event: pd.Series,
    snapshots: pd.DataFrame,
    snapshot: pd.Series,
    snapshot_pos: int,
    gate: MicrostructureGateConfig,
    policy: OrderPolicyConfig,
    snapshot_age_ms: float,
    snapshot_cache: SnapshotCache | None = None,
    gate_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    event_type = str(event["event_type"])
    action = str(event["action"])
    order_notional = float(event["order_notional_usdt"])
    equity = float(event["account_equity_usdt"])
    force_execute = event_type == "forced_exit"
    gate_result = cached_order_gate(
        gate_cache,
        snapshot_pos,
        snapshot,
        action,
        equity,
        order_notional,
        gate,
        policy,
        snapshot_age_ms,
    )
    if policy_uses_maker(policy_name, event_type):
        return simulate_maker_order(
            snapshot,
            snapshot_pos,
            snapshots,
            action,
            order_notional,
            equity,
            gate,
            policy,
            snapshot_age_ms,
            snapshot_cache=snapshot_cache,
            precomputed_gate=gate_result,
        )
    if policy_name == "sliced_taker" and event_type != "forced_exit":
        return simulate_sliced_taker_order(
            snapshot,
            action,
            order_notional,
            equity,
            gate,
            policy,
            snapshot_age_ms,
            force_execute,
            precomputed_gate=gate_result,
        )
    return simulate_taker_order(
        snapshot,
        action,
        order_notional,
        equity,
        gate,
        policy.max_total_book_share,
        snapshot_age_ms,
        force_execute,
        precomputed_gate=gate_result,
    )


def simulate_order_policies(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    validate_order_policy_config(config)
    gate = gate_config_from_yaml(config)
    policy = order_policy_from_yaml(config)
    cache = SnapshotCache.from_frame(snapshots)
    snaps = cache.frame
    gate_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        snapshot, snapshot_pos, mapping_mode, snapshot_age_ms = cache.choose_event_snapshot(
            event,
            gate,
            policy.synthetic_mapping_seed,
        )
        for policy_name in policy.policies:
            result = simulate_policy_order(
                policy_name,
                event,
                snaps,
                snapshot,
                snapshot_pos,
                gate,
                policy,
                snapshot_age_ms,
                snapshot_cache=cache,
                gate_cache=gate_cache,
            )
            rows.append(
                {
                    "policy": policy_name,
                    "mapping_mode": mapping_mode,
                    "snapshot_time_utc": snapshot["snapshot_time_utc"],
                    "snapshot_pos": int(snapshot_pos),
                    "last_update_id": int(snapshot.get("last_update_id", -1)),
                    **event.to_dict(),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def summarize_policy_comparison(simulation: pd.DataFrame, config: dict[str, Any], snapshot_summary: pd.DataFrame) -> pd.DataFrame:
    if simulation.empty:
        return pd.DataFrame()
    reference_monthly = float(config["source_iteration"]["reference_monthly_return"])
    invalid_fraction = float(snapshot_summary.iloc[0]["invalid_snapshot_fraction"]) if not snapshot_summary.empty else 1.0
    rows: list[dict[str, Any]] = []
    for (policy, equity), group in simulation.groupby(["policy", "account_equity_usdt"], sort=True):
        executed = group["executed"].astype(bool)
        entries = group[group["event_type"].eq("entry")]
        adds = group[group["event_type"].eq("add")]
        take_profits = group[group["event_type"].eq("take_profit")]
        forced = group[group["event_type"].eq("forced_exit")]
        fold_cost = (
            group[group["executed"].astype(bool)]
            .groupby("fold_id")["execution_cost_pct_equity"]
            .sum()
            .reindex(sorted(group["fold_id"].unique()), fill_value=0.0)
        )
        monthly_execution_cost = float(fold_cost.mean()) if not fold_cost.empty else 0.0
        entry_skipped_rate = float(entries["skipped"].astype(bool).mean()) if not entries.empty else 0.0
        missed_entry_penalty = max(reference_monthly, 0.0) * entry_skipped_rate
        adjusted = reference_monthly - monthly_execution_cost - missed_entry_penalty
        rows.append(
            {
                "policy": policy,
                "account_equity_usdt": float(equity),
                "order_count": int(len(group)),
                "executed_rate": float(executed.mean()),
                "entry_skipped_rate": entry_skipped_rate,
                "add_skipped_rate": float(adds["skipped"].astype(bool).mean()) if not adds.empty else 0.0,
                "tp_skipped_rate": float(take_profits["skipped"].astype(bool).mean()) if not take_profits.empty else 0.0,
                "forced_exit_executed_rate": float(forced["executed"].astype(bool).mean()) if not forced.empty else 1.0,
                "p90_slippage_bps": float(
                    group.loc[executed, "market_impact_bps"].replace([np.inf, -np.inf], np.nan).quantile(0.90)
                ),
                "median_spread_bps": float(group["spread_bps"].median()),
                "mean_order_book_share": float(
                    group["order_book_share"].replace([np.inf, -np.inf], np.nan).mean()
                ),
                "monthly_execution_cost_estimate": monthly_execution_cost,
                "estimated_monthly_after_execution": adjusted,
                "synthetic_mapping_fraction": float(group["mapping_mode"].eq("synthetic_microstructure_mapping").mean()),
                "invalid_snapshot_fraction": invalid_fraction,
            }
        )
    return pd.DataFrame(rows)


def summarize_signal_attribution(simulation: pd.DataFrame) -> pd.DataFrame:
    if simulation.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["policy", "account_equity_usdt", "trade_id", "fold_id"]
    for key, group in simulation.groupby(group_cols, sort=True):
        rows.append(
            {
                "policy": key[0],
                "account_equity_usdt": float(key[1]),
                "trade_id": int(key[2]),
                "fold_id": int(key[3]),
                "entry_executed": bool(group[group["event_type"].eq("entry")]["executed"].astype(bool).all()),
                "adds_attempted": int(group["event_type"].eq("add").sum()),
                "adds_skipped": int((group["event_type"].eq("add") & group["skipped"].astype(bool)).sum()),
                "tp_skipped": int((group["event_type"].eq("take_profit") & group["skipped"].astype(bool)).sum()),
                "forced_exit_count": int(group["event_type"].eq("forced_exit").sum()),
                "execution_cost_pct_equity": float(group["execution_cost_pct_equity"].sum()),
                "mapping_mode": "synthetic_microstructure_mapping"
                if group["mapping_mode"].eq("synthetic_microstructure_mapping").any()
                else "exact_signal_timestamp",
            }
        )
    return pd.DataFrame(rows)


def summarize_slippage_by_equity_policy(simulation: pd.DataFrame) -> pd.DataFrame:
    if simulation.empty:
        return pd.DataFrame()
    executed = simulation[simulation["executed"].astype(bool)].copy()
    if executed.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in executed.groupby(["policy", "account_equity_usdt", "event_type"], sort=True):
        rows.append(
            {
                "policy": key[0],
                "account_equity_usdt": float(key[1]),
                "event_type": key[2],
                "order_count": int(len(group)),
                "slippage_bps_median": float(group["market_impact_bps"].replace([np.inf, -np.inf], np.nan).median()),
                "slippage_bps_p90": float(group["market_impact_bps"].replace([np.inf, -np.inf], np.nan).quantile(0.90)),
                "spread_cost_pct_equity_sum": float(group["spread_cost_pct_equity"].sum()),
                "market_impact_cost_pct_equity_sum": float(group["market_impact_cost_pct_equity"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_missed_fills(simulation: pd.DataFrame) -> pd.DataFrame:
    if simulation.empty:
        return pd.DataFrame()
    missed = simulation[simulation["skipped"].astype(bool)].copy()
    if missed.empty:
        return pd.DataFrame(
            columns=["policy", "account_equity_usdt", "event_type", "missed_count", "primary_reason", "missed_rate"]
        )
    total = simulation.groupby(["policy", "account_equity_usdt", "event_type"]).size()
    rows: list[dict[str, Any]] = []
    for key, group in missed.groupby(["policy", "account_equity_usdt", "event_type"], sort=True):
        primary = group["reject_reason_primary"].replace("", "maker_not_filled").mode()
        rows.append(
            {
                "policy": key[0],
                "account_equity_usdt": float(key[1]),
                "event_type": key[2],
                "missed_count": int(len(group)),
                "primary_reason": str(primary.iloc[0]) if not primary.empty else "unknown",
                "missed_rate": float(len(group) / int(total.loc[key])),
            }
        )
    return pd.DataFrame(rows)


def decide_policy_verdict(comparison: pd.DataFrame, snapshot_summary: pd.DataFrame, config: dict[str, Any]) -> str:
    if comparison.empty or snapshot_summary.empty:
        return "needs more collection"
    verdict = config["verdict"]
    summary = snapshot_summary.iloc[0]
    if float(summary["collection_span_hours"]) < float(verdict["min_collection_hours_live_ready"]):
        return "needs more collection"
    if float(summary["invalid_snapshot_fraction"]) > float(verdict["max_invalid_snapshot_fraction"]):
        return "needs more collection"
    viable = comparison[
        comparison["account_equity_usdt"].le(float(verdict["promising_equity_ceiling"]))
        & comparison["p90_slippage_bps"].le(float(verdict["max_p90_slippage_bps_promising"]))
        & comparison["entry_skipped_rate"].le(float(verdict["max_entry_skipped_rate_promising"]))
        & comparison["estimated_monthly_after_execution"].ge(float(verdict["adjusted_monthly_floor"]))
    ]
    if not viable.empty:
        failing_25k = comparison[
            comparison["account_equity_usdt"].eq(25000.0)
            & (
                comparison["p90_slippage_bps"].gt(float(verdict["max_p90_slippage_bps_promising"]))
                | comparison["entry_skipped_rate"].gt(float(verdict["max_entry_skipped_rate_promising"]))
            )
        ]
        return "size capped" if not failing_25k.empty else "execution policy promising"
    return "execution edge still fragile"


def write_policy_heatmaps(snapshots: pd.DataFrame, simulation: pd.DataFrame, output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "microstructure_order_policy_heatmaps.png"
    if snapshots.empty:
        return path
    snapshot_frame = snapshots.copy()
    snapshot_frame["hour_utc"] = snapshot_frame["snapshot_time_utc"].dt.hour
    snap_metrics = snapshot_frame.groupby("hour_utc", sort=True)[
        ["spread_bps", "bid_depth_5bps_usdt", "ask_depth_5bps_usdt", "depth_imbalance_5bps"]
    ].median()
    fill_rate = pd.Series(dtype=float)
    if not simulation.empty:
        sim = simulation.copy()
        sim["hour_utc"] = pd.to_datetime(sim["snapshot_time_utc"], utc=True).dt.hour
        fill_rate = sim.groupby("hour_utc")["executed"].mean()
    hourly = pd.DataFrame(
        {
            "spread_bps": snap_metrics["spread_bps"],
            "bid_depth_5bps_usdt": snap_metrics["bid_depth_5bps_usdt"],
            "ask_depth_5bps_usdt": snap_metrics["ask_depth_5bps_usdt"],
            "depth_imbalance_5bps": snap_metrics["depth_imbalance_5bps"],
            "fill_rate": fill_rate,
        }
    ).reindex(range(24))
    data = hourly.T.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title("Microstructure and execution metrics by UTC hour")
    ax.set_xlabel("UTC hour")
    ax.set_yticks(range(len(hourly.columns)))
    ax.set_yticklabels(["Spread bps", "Bid depth 5bps", "Ask depth 5bps", "Imbalance 5bps", "Fill rate"])
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(hour) for hour in range(24)])
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "iteration_report.md"
    comparison = pd.DataFrame(payload["policy_comparison"])
    snapshot_summary = pd.DataFrame(payload["collection_quality_summary"])
    missed = pd.DataFrame(payload["missed_fill_diagnostics"])
    lines = [
        "# Iteration 022 - Microstructure Order Policy 017",
        "",
        "## Verdict",
        f"`{payload['verdict']}`",
        "",
        "## Collection Quality",
        markdown_table(snapshot_summary) if not snapshot_summary.empty else "No snapshots available.",
        "",
        "## Policy Comparison",
    ]
    if comparison.empty:
        lines.append("No policy comparison available.")
    else:
        cols = [
            "policy",
            "account_equity_usdt",
            "executed_rate",
            "entry_skipped_rate",
            "add_skipped_rate",
            "tp_skipped_rate",
            "p90_slippage_bps",
            "estimated_monthly_after_execution",
            "synthetic_mapping_fraction",
        ]
        lines.append(markdown_table(comparison[cols].head(32)))
    lines.extend(["", "## Missed Fill Diagnostics"])
    if missed.empty:
        lines.append("No missed fills.")
    else:
        lines.append(markdown_table(missed.head(32)))
    lines.extend(
        [
            "",
            "## Notes",
            "Iteration 022 keeps Iteration 017 locked: no strategy re-selection, no private endpoints, no API keys and no live orders.",
            "When snapshots do not overlap historical signal timestamps, rows are tagged `synthetic_microstructure_mapping`; this is a microstructure sensitivity estimate, not a historical L2 backtest.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def collect_loop(config: dict[str, Any], interval_seconds: float, duration_hours: float) -> pd.DataFrame:
    if interval_seconds <= 0 or duration_hours <= 0:
        raise ValueError("interval_seconds and duration_hours must be positive")
    end = time.monotonic() + duration_hours * 3600.0
    frames: list[pd.DataFrame] = []
    while time.monotonic() < end:
        frame = collect_once(config)
        frames.append(frame)
        sleep_for = min(interval_seconds, max(0.0, end - time.monotonic()))
        if sleep_for > 0:
            time.sleep(sleep_for)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_policies(config: dict[str, Any], max_snapshots: int | None = None, max_trades: int | None = None) -> dict[str, Any]:
    validate_order_policy_config(config)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshots(project_path(config["collection"]["snapshot_path"]))
    if max_snapshots is not None:
        snapshots = snapshots.tail(int(max_snapshots)).copy()
    gate = gate_config_from_yaml(config)
    snapshot_summary = summarize_snapshots(snapshots, gate)
    locked = load_locked_017_candidates(config)
    trades = load_locked_017_trades(config, max_trades=max_trades)
    equities = [float(value) for value in config["microstructure_gate"]["account_equity_usdt_grid"]]
    events = order_events_from_trades(trades, equities)
    simulation = simulate_order_policies(snapshots, events, config)
    comparison = summarize_policy_comparison(simulation, config, snapshot_summary)
    signal_attribution = summarize_signal_attribution(simulation)
    slippage = summarize_slippage_by_equity_policy(simulation)
    missed = summarize_missed_fills(simulation)
    heatmap = write_policy_heatmaps(snapshots, simulation, output_dir)
    verdict = decide_policy_verdict(comparison, snapshot_summary, config)

    snapshot_summary.to_csv(output_dir / "collection_quality_summary.csv", index=False)
    comparison.to_csv(output_dir / "policy_comparison.csv", index=False)
    simulation.to_csv(output_dir / "order_execution_simulation.csv", index=False)
    signal_attribution.to_csv(output_dir / "signal_execution_attribution.csv", index=False)
    slippage.to_csv(output_dir / "slippage_by_equity_policy.csv", index=False)
    missed.to_csv(output_dir / "missed_fill_diagnostics.csv", index=False)

    payload = {
        "iteration_name": config["iteration"]["name"],
        "verdict": verdict,
        "locked_candidate_count": int(len(locked)),
        "trade_count": int(len(trades)),
        "order_event_count": int(len(events)),
        "simulated_order_count": int(len(simulation)),
        "collection_quality_summary": snapshot_summary.to_dict("records"),
        "policy_comparison": comparison.to_dict("records"),
        "missed_fill_diagnostics": missed.to_dict("records"),
        "heatmap": str(heatmap),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = write_report(output_dir, payload)
    LOGGER.info("Wrote Iteration 022 outputs to %s", output_dir)
    LOGGER.info("Iteration report: %s", report)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate microstructure order policies for locked Iteration 017.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--collect-loop", action="store_true")
    parser.add_argument("--evaluate-policies", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--max-snapshots", type=int, default=None)
    parser.add_argument("--max-trades", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if not (args.collect_loop or args.evaluate_policies or args.smoke):
        raise SystemExit("Choose --collect-loop, --evaluate-policies or --smoke")
    if args.collect_loop:
        frame = collect_loop(config, args.interval_seconds, args.duration_hours)
        print(json.dumps({"collected_snapshots": int(len(frame))}, indent=2))
    if args.evaluate_policies or args.smoke:
        max_snapshots = args.max_snapshots
        max_trades = args.max_trades
        if args.smoke:
            max_snapshots = 100 if max_snapshots is None else max_snapshots
            max_trades = 50 if max_trades is None else max_trades
        payload = evaluate_policies(config, max_snapshots=max_snapshots, max_trades=max_trades)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
