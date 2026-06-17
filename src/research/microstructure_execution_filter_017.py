from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.validate_data import load_processed
from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.fundamental_trend_escape_martingale_research import build_variant_masks, run_exact_candidate_on_index
from src.research.monthly_target_martingale_research import MonthlyMartingaleCandidate, candidate_from_row
from src.research.walk_forward_martingale_research import make_walk_forward_windows
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/research_iteration_microstructure_execution_filter_017.yaml"


@dataclass(frozen=True)
class MicrostructureGateConfig:
    depth_band_bps: float
    max_spread_bps: float
    max_order_book_share: float
    min_side_depth_usdt_floor: float
    max_abs_depth_imbalance: float
    max_snapshot_age_ms: float
    max_theoretical_slippage_bps: float


def gate_config_from_yaml(config: dict[str, Any]) -> MicrostructureGateConfig:
    gate = config["microstructure_gate"]
    return MicrostructureGateConfig(
        depth_band_bps=float(gate["depth_band_bps"]),
        max_spread_bps=float(gate["max_spread_bps"]),
        max_order_book_share=float(gate["max_order_book_share"]),
        min_side_depth_usdt_floor=float(gate["min_side_depth_usdt_floor"]),
        max_abs_depth_imbalance=float(gate["max_abs_depth_imbalance"]),
        max_snapshot_age_ms=float(gate["max_snapshot_age_ms"]),
        max_theoretical_slippage_bps=float(gate["max_theoretical_slippage_bps"]),
    )


def validate_microstructure_config(config: dict[str, Any]) -> None:
    gate = gate_config_from_yaml(config)
    if gate.depth_band_bps <= 0:
        raise ValueError("microstructure_gate.depth_band_bps must be positive")
    if gate.max_spread_bps < 0:
        raise ValueError("microstructure_gate.max_spread_bps must be non-negative")
    if gate.max_order_book_share <= 0 or gate.max_order_book_share > 1:
        raise ValueError("microstructure_gate.max_order_book_share must stay within (0, 1]")
    if gate.min_side_depth_usdt_floor <= 0:
        raise ValueError("microstructure_gate.min_side_depth_usdt_floor must be positive")
    if gate.max_abs_depth_imbalance < 0 or gate.max_abs_depth_imbalance > 1:
        raise ValueError("microstructure_gate.max_abs_depth_imbalance must stay within [0, 1]")
    if gate.max_snapshot_age_ms <= 0:
        raise ValueError("microstructure_gate.max_snapshot_age_ms must be positive")
    if gate.max_theoretical_slippage_bps < 0:
        raise ValueError("microstructure_gate.max_theoretical_slippage_bps must be non-negative")
    bands = [float(value) for value in config["microstructure_gate"]["depth_bands_bps"]]
    if not bands or any(value <= 0 for value in bands):
        raise ValueError("microstructure_gate.depth_bands_bps must contain positive values")
    equities = [float(value) for value in config["microstructure_gate"]["account_equity_usdt_grid"]]
    if not equities or any(value <= 0 for value in equities):
        raise ValueError("microstructure_gate.account_equity_usdt_grid must contain positive values")


def _json_get(url: str, timeout_seconds: float) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "grid-survival-research/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return json.loads(raw.decode("utf-8")), latency_ms


def fetch_binance_public_json(
    config: dict[str, Any],
    endpoint: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], float, str]:
    collection = config["collection"]
    timeout = float(collection.get("timeout_seconds", 10))
    query = urllib.parse.urlencode(params)
    errors: list[str] = []
    for base_url in collection["base_urls"]:
        url = f"{str(base_url).rstrip('/')}{endpoint}?{query}"
        try:
            payload, latency_ms = _json_get(url, timeout)
            return payload, latency_ms, url
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All Binance public endpoints failed: " + " | ".join(errors))


def parse_depth_levels(levels: list[list[str]] | list[tuple[str, str]]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for price, quantity, *_rest in levels:
        p = float(price)
        q = float(quantity)
        if p > 0 and q > 0:
            parsed.append((p, q))
    return parsed


def quote_depth_within_band(
    levels: list[tuple[float, float]],
    best_price: float,
    band_bps: float,
    side: str,
) -> float:
    if side not in {"bid", "ask"}:
        raise ValueError("side must be bid or ask")
    if best_price <= 0 or band_bps <= 0:
        raise ValueError("best_price and band_bps must be positive")
    if side == "bid":
        threshold = best_price * (1 - band_bps / 10000.0)
        eligible = [(price, qty) for price, qty in levels if price >= threshold and price <= best_price]
    else:
        threshold = best_price * (1 + band_bps / 10000.0)
        eligible = [(price, qty) for price, qty in levels if price <= threshold and price >= best_price]
    return float(sum(price * qty for price, qty in eligible))


def walk_book_slippage_bps(
    levels: list[tuple[float, float]],
    order_notional_usdt: float,
    side: str,
) -> float:
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    if order_notional_usdt <= 0:
        raise ValueError("order_notional_usdt must be positive")
    if not levels:
        return float("inf")
    best_price = float(levels[0][0])
    remaining = float(order_notional_usdt)
    filled_quote = 0.0
    filled_base = 0.0
    for price, quantity in levels:
        level_quote = price * quantity
        consume_quote = min(remaining, level_quote)
        filled_quote += consume_quote
        filled_base += consume_quote / price
        remaining -= consume_quote
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or filled_base <= 0:
        return float("inf")
    avg_price = filled_quote / filled_base
    if side == "buy":
        return float((avg_price / best_price - 1.0) * 10000.0)
    return float((1.0 - avg_price / best_price) * 10000.0)


def normalize_depth_snapshot(
    symbol: str,
    depth_payload: dict[str, Any],
    book_ticker_payload: dict[str, Any],
    depth_bands_bps: list[float],
    source_latency_ms: float,
    depth_source_url: str,
    book_ticker_source_url: str,
    snapshot_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    bids = parse_depth_levels(depth_payload.get("bids", []))
    asks = parse_depth_levels(depth_payload.get("asks", []))
    if not bids or not asks:
        raise ValueError("Binance depth snapshot must contain non-empty bids and asks")
    best_bid = float(book_ticker_payload.get("bidPrice") or bids[0][0])
    best_ask = float(book_ticker_payload.get("askPrice") or asks[0][0])
    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        raise ValueError("Best bid/ask prices must be positive and crossed-free")
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10000.0
    row: dict[str, Any] = {
        "snapshot_time_utc": snapshot_time or pd.Timestamp.now(tz="UTC"),
        "symbol": symbol.upper(),
        "last_update_id": int(depth_payload.get("lastUpdateId", -1)),
        "book_ticker_update_id": int(book_ticker_payload.get("lastUpdateId", -1))
        if str(book_ticker_payload.get("lastUpdateId", "")).lstrip("-").isdigit()
        else -1,
        "best_bid": best_bid,
        "best_bid_qty": float(book_ticker_payload.get("bidQty") or bids[0][1]),
        "best_ask": best_ask,
        "best_ask_qty": float(book_ticker_payload.get("askQty") or asks[0][1]),
        "mid_price": mid,
        "spread_bps": float(spread_bps),
        "source_latency_ms": float(source_latency_ms),
        "depth_source_url": depth_source_url,
        "book_ticker_source_url": book_ticker_source_url,
        "bids_json": json.dumps(bids),
        "asks_json": json.dumps(asks),
    }
    for band in depth_bands_bps:
        label = f"{band:g}".replace(".", "p")
        bid_depth = quote_depth_within_band(bids, best_bid, float(band), "bid")
        ask_depth = quote_depth_within_band(asks, best_ask, float(band), "ask")
        row[f"bid_depth_{label}bps_usdt"] = bid_depth
        row[f"ask_depth_{label}bps_usdt"] = ask_depth
    bid_5 = float(row.get("bid_depth_5bps_usdt", 0.0))
    ask_5 = float(row.get("ask_depth_5bps_usdt", 0.0))
    total_5 = bid_5 + ask_5
    row["depth_imbalance_5bps"] = float((bid_5 - ask_5) / total_5) if total_5 > 0 else np.nan
    return row


def collect_once(config: dict[str, Any]) -> pd.DataFrame:
    validate_microstructure_config(config)
    symbol = str(config["asset"]).upper()
    collection = config["collection"]
    depth, depth_latency, depth_url = fetch_binance_public_json(
        config,
        str(collection["depth_endpoint"]),
        {"symbol": symbol, "limit": int(collection.get("depth_limit", 500))},
    )
    ticker, ticker_latency, ticker_url = fetch_binance_public_json(
        config,
        str(collection["book_ticker_endpoint"]),
        {"symbol": symbol},
    )
    row = normalize_depth_snapshot(
        symbol,
        depth,
        ticker,
        [float(value) for value in config["microstructure_gate"]["depth_bands_bps"]],
        source_latency_ms=depth_latency + ticker_latency,
        depth_source_url=depth_url,
        book_ticker_source_url=ticker_url,
    )
    frame = pd.DataFrame([row])
    append_snapshots(frame, project_path(config["collection"]["snapshot_path"]))
    return frame


def append_snapshots(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()
    combined["snapshot_time_utc"] = pd.to_datetime(combined["snapshot_time_utc"], utc=True)
    combined = combined.sort_values(["snapshot_time_utc", "last_update_id"]).drop_duplicates(
        ["snapshot_time_utc", "last_update_id"],
        keep="last",
    )
    combined.to_parquet(path, index=False)
    return combined


def load_snapshots(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No microstructure snapshots found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Microstructure snapshot file is empty: {path}")
    frame["snapshot_time_utc"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    return frame.sort_values("snapshot_time_utc").reset_index(drop=True)


def depth_column_for_band(side: str, band_bps: float) -> str:
    label = f"{band_bps:g}".replace(".", "p")
    if side == "long":
        return f"ask_depth_{label}bps_usdt"
    if side == "short":
        return f"bid_depth_{label}bps_usdt"
    raise ValueError("side must be long or short")


def book_levels_for_side(snapshot: pd.Series, side: str) -> list[tuple[float, float]]:
    if side == "long":
        return [(float(price), float(qty)) for price, qty in json.loads(str(snapshot["asks_json"]))]
    if side == "short":
        return [(float(price), float(qty)) for price, qty in json.loads(str(snapshot["bids_json"]))]
    raise ValueError("side must be long or short")


def evaluate_gate(
    snapshot: pd.Series,
    side: str,
    account_equity_usdt: float,
    initial_notional_multiplier: float,
    gate: MicrostructureGateConfig,
    snapshot_age_ms: float | None = None,
) -> dict[str, Any]:
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    if account_equity_usdt <= 0 or initial_notional_multiplier <= 0:
        raise ValueError("account_equity_usdt and initial_notional_multiplier must be positive")
    age_ms = float(snapshot_age_ms if snapshot_age_ms is not None else snapshot.get("source_latency_ms", np.nan))
    order_notional = float(account_equity_usdt) * float(initial_notional_multiplier)
    side_depth_col = depth_column_for_band(side, gate.depth_band_bps)
    side_depth = float(snapshot.get(side_depth_col, np.nan))
    total_depth = float(snapshot.get("bid_depth_5bps_usdt", 0.0)) + float(snapshot.get("ask_depth_5bps_usdt", 0.0))
    imbalance = float(snapshot.get("depth_imbalance_5bps", np.nan))
    spread = float(snapshot.get("spread_bps", np.nan))
    min_required_depth = max(float(gate.min_side_depth_usdt_floor), order_notional / float(gate.max_order_book_share))
    order_book_share = order_notional / side_depth if np.isfinite(side_depth) and side_depth > 0 else float("inf")
    levels = book_levels_for_side(snapshot, side) if "bids_json" in snapshot and "asks_json" in snapshot else []
    theoretical_slippage = walk_book_slippage_bps(levels, order_notional, "buy" if side == "long" else "sell")
    reasons: list[str] = []
    if not np.isfinite(age_ms) or age_ms > gate.max_snapshot_age_ms:
        reasons.append("stale_snapshot")
    if not np.isfinite(spread) or spread > gate.max_spread_bps:
        reasons.append("wide_spread")
    if not np.isfinite(side_depth) or side_depth < min_required_depth:
        reasons.append("insufficient_depth")
    if not np.isfinite(order_book_share) or order_book_share > gate.max_order_book_share:
        reasons.append("order_too_large_for_book")
    if not np.isfinite(imbalance) or abs(imbalance) > gate.max_abs_depth_imbalance:
        reasons.append("violent_imbalance")
    if not np.isfinite(theoretical_slippage) or theoretical_slippage > gate.max_theoretical_slippage_bps:
        reasons.append("theoretical_slippage_breach")
    if total_depth <= 0:
        reasons.append("empty_depth")
    return {
        "side": side,
        "account_equity_usdt": float(account_equity_usdt),
        "initial_notional_multiplier": float(initial_notional_multiplier),
        "order_notional_usdt": order_notional,
        "snapshot_age_ms": age_ms,
        "spread_bps": spread,
        "side_depth_usdt": side_depth,
        "min_required_depth_usdt": min_required_depth,
        "order_book_share": float(order_book_share),
        "depth_imbalance_5bps": imbalance,
        "theoretical_slippage_bps": float(theoretical_slippage),
        "authorized": len(reasons) == 0,
        "reject_reasons": ",".join(reasons),
        "reject_reason_primary": reasons[0] if reasons else "",
    }


def load_locked_017_candidates(config: dict[str, Any]) -> pd.DataFrame:
    source_dir = project_path(config["source_iteration"]["output_dir"])
    path = source_dir / "walk_forward_selected_candidates.csv"
    if not path.exists():
        raise FileNotFoundError(f"Iteration 017 selected candidates not found: {path}")
    frame = pd.read_csv(path)
    variant = str(config["source_iteration"]["variant"])
    locked = frame[frame["variant"].astype(str).eq(variant)].copy()
    if locked.empty:
        raise ValueError(f"No locked candidates for variant {variant}")
    return locked.sort_values("fold_id").reset_index(drop=True)


def unique_initial_multipliers(locked_candidates: pd.DataFrame) -> list[float]:
    return sorted(float(value) for value in locked_candidates["base_position_size_pct"].dropna().unique())


def evaluate_snapshot_grid(
    snapshots: pd.DataFrame,
    config: dict[str, Any],
    locked_candidates: pd.DataFrame,
) -> pd.DataFrame:
    gate = gate_config_from_yaml(config)
    equities = [float(value) for value in config["microstructure_gate"]["account_equity_usdt_grid"]]
    multipliers = unique_initial_multipliers(locked_candidates)
    rows: list[dict[str, Any]] = []
    for _, snapshot in snapshots.iterrows():
        for side in ["long", "short"]:
            for equity in equities:
                for multiplier in multipliers:
                    decision = evaluate_gate(snapshot, side, equity, multiplier, gate)
                    rows.append(
                        {
                            "snapshot_time_utc": snapshot["snapshot_time_utc"],
                            "symbol": snapshot["symbol"],
                            "last_update_id": snapshot["last_update_id"],
                            **decision,
                        }
                    )
    return pd.DataFrame(rows)


def build_locked_017_trades(config: dict[str, Any]) -> pd.DataFrame:
    source_config = load_yaml(config["source_iteration"]["config_path"])
    locked = load_locked_017_candidates(config)
    market = prepare_market()
    signal_frame = load_processed(project_path("data/processed/btcusdt_1h.parquet"))
    base_risk = validate_strategy_config(load_strategy_config())
    events, _windows_frame, blackout_masks = build_blackout_bundle(market.index, source_config)
    trend_components = build_trend_escape_components(market, source_config)
    entry_mask, _exit_mask, _reason = build_variant_masks(
        str(config["source_iteration"]["variant"]),
        trend_components["trend_escape"].astype(bool),
        blackout_masks,
    )
    wf = source_config["walk_forward"]
    windows = make_walk_forward_windows(
        market.index,
        train_days=float(wf["train_days"]),
        test_days=float(wf["test_days"]),
        step_days=float(wf["step_days"]),
        embargo_bars=int(wf.get("embargo_bars", 0)),
    )
    rows: list[pd.DataFrame] = []
    for window in windows:
        selected = locked[locked["fold_id"].astype(int).eq(int(window.fold_id))]
        if selected.empty:
            raise ValueError(f"Missing locked candidate for fold {window.fold_id}")
        candidate: MonthlyMartingaleCandidate = candidate_from_row(selected.iloc[0].to_dict())
        _metrics, result = run_exact_candidate_on_index(
            market,
            signal_frame,
            base_risk,
            candidate,
            window.test,
            "test",
            entry_mask=entry_mask,
            exit_mask=None,
            exit_reason="fundamental_trend_escape",
        )
        trades = result.trades.copy()
        if not trades.empty:
            trades.insert(0, "fold_id", int(window.fold_id))
            rows.append(trades)
    trades = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not trades.empty:
        trades["start_timestamp"] = pd.to_datetime(trades["start_timestamp"], utc=True)
    return trades


def attribute_snapshots_to_017_signals(
    snapshots: pd.DataFrame,
    config: dict[str, Any],
    locked_trades: pd.DataFrame,
) -> pd.DataFrame:
    if locked_trades.empty:
        return pd.DataFrame()
    gate = gate_config_from_yaml(config)
    equities = [float(value) for value in config["microstructure_gate"]["account_equity_usdt_grid"]]
    snapshot_cols = [
        "snapshot_time_utc",
        "symbol",
        "last_update_id",
        "spread_bps",
        "bid_depth_5bps_usdt",
        "ask_depth_5bps_usdt",
        "depth_imbalance_5bps",
        "source_latency_ms",
        "bids_json",
        "asks_json",
    ]
    snaps = snapshots[snapshot_cols].sort_values("snapshot_time_utc").copy()
    signals = locked_trades.sort_values("start_timestamp").copy()
    matched = pd.merge_asof(
        signals,
        snaps,
        left_on="start_timestamp",
        right_on="snapshot_time_utc",
        direction="backward",
        tolerance=pd.Timedelta(milliseconds=gate.max_snapshot_age_ms),
    )
    rows: list[dict[str, Any]] = []
    for _, signal in matched.iterrows():
        base = {
            "fold_id": int(signal["fold_id"]),
            "signal_timestamp": signal["start_timestamp"],
            "side": signal["side"],
            "candidate_name": signal.get("name", ""),
            "base_position_size_pct": float(signal.get("base_position_size_pct", np.nan)),
            "snapshot_time_utc": signal.get("snapshot_time_utc", pd.NaT),
            "matched_snapshot": bool(pd.notna(signal.get("snapshot_time_utc", pd.NaT))),
        }
        if not base["matched_snapshot"]:
            for equity in equities:
                rows.append({**base, "account_equity_usdt": equity, "authorized": False, "reject_reasons": "missing_snapshot"})
            continue
        age_ms = (pd.Timestamp(signal["start_timestamp"]) - pd.Timestamp(signal["snapshot_time_utc"])) / pd.Timedelta(
            milliseconds=1
        )
        for equity in equities:
            decision = evaluate_gate(
                signal,
                str(signal["side"]),
                equity,
                float(signal["base_position_size_pct"]),
                gate,
                snapshot_age_ms=float(age_ms),
            )
            rows.append({**base, **decision})
    return pd.DataFrame(rows)


def summarize_snapshots(snapshots: pd.DataFrame, gate: MicrostructureGateConfig) -> pd.DataFrame:
    span_hours = (
        (snapshots["snapshot_time_utc"].max() - snapshots["snapshot_time_utc"].min()) / pd.Timedelta(hours=1)
        if len(snapshots) > 1
        else 0.0
    )
    invalid = (
        snapshots["spread_bps"].isna()
        | snapshots["bid_depth_5bps_usdt"].isna()
        | snapshots["ask_depth_5bps_usdt"].isna()
        | snapshots["source_latency_ms"].astype(float).gt(gate.max_snapshot_age_ms)
    )
    rows = [
        {
            "snapshot_count": int(len(snapshots)),
            "collection_span_hours": float(span_hours),
            "invalid_snapshot_fraction": float(invalid.mean()),
            "spread_bps_median": float(snapshots["spread_bps"].median()),
            "spread_bps_p90": float(snapshots["spread_bps"].quantile(0.90)),
            "spread_bps_p99": float(snapshots["spread_bps"].quantile(0.99)),
            "bid_depth_5bps_median": float(snapshots["bid_depth_5bps_usdt"].median()),
            "ask_depth_5bps_median": float(snapshots["ask_depth_5bps_usdt"].median()),
            "abs_imbalance_5bps_p90": float(snapshots["depth_imbalance_5bps"].abs().quantile(0.90)),
            "source_latency_ms_p90": float(snapshots["source_latency_ms"].quantile(0.90)),
        }
    ]
    return pd.DataFrame(rows)


def summarize_slippage_risk(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["side", "account_equity_usdt", "initial_notional_multiplier"]
    for key, group in decisions.groupby(group_cols, sort=True):
        authorized = group["authorized"].astype(bool)
        rows.append(
            {
                "side": key[0],
                "account_equity_usdt": float(key[1]),
                "initial_notional_multiplier": float(key[2]),
                "decision_count": int(len(group)),
                "authorized_rate": float(authorized.mean()),
                "spread_bps_median": float(group["spread_bps"].median()),
                "side_depth_usdt_median": float(group["side_depth_usdt"].median()),
                "order_book_share_p90": float(group["order_book_share"].replace([np.inf, -np.inf], np.nan).quantile(0.90)),
                "theoretical_slippage_bps_p90": float(
                    group["theoretical_slippage_bps"].replace([np.inf, -np.inf], np.nan).quantile(0.90)
                ),
                "primary_reject_depth_rate": float(group["reject_reasons"].astype(str).str.contains("depth").mean()),
                "primary_reject_spread_rate": float(group["reject_reasons"].astype(str).str.contains("wide_spread").mean()),
                "primary_reject_imbalance_rate": float(
                    group["reject_reasons"].astype(str).str.contains("violent_imbalance").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def decide_microstructure(
    snapshots_summary: pd.DataFrame,
    decisions: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    verdict_cfg = config["verdict"]
    if snapshots_summary.empty or decisions.empty:
        return "needs more collection"
    summary = snapshots_summary.iloc[0]
    if float(summary["collection_span_hours"]) < float(verdict_cfg["min_collection_hours_live_ready"]):
        return "needs more collection"
    if float(summary["invalid_snapshot_fraction"]) > float(verdict_cfg["max_invalid_snapshot_fraction"]):
        return "needs more collection"
    authorized = decisions[decisions["authorized"].astype(bool)]
    if authorized.empty:
        return "execution too thin"
    depth_reject_rate = decisions["reject_reasons"].astype(str).str.contains("insufficient_depth|order_too_large").mean()
    spread_reject_rate = decisions["reject_reasons"].astype(str).str.contains("wide_spread|violent_imbalance").mean()
    if depth_reject_rate > 0.5:
        return "execution too thin"
    if spread_reject_rate > 0.5:
        return "spread/imbalance unstable"
    slippage_breach = authorized["reject_reasons"].astype(str).str.contains("theoretical_slippage_breach").mean()
    if slippage_breach <= float(verdict_cfg["max_authorized_slippage_breach_fraction"]):
        return "microstructure gate live-ready"
    return "execution too thin"


def write_hourly_heatmap(snapshots: pd.DataFrame, output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "microstructure_hourly_heatmaps.png"
    if snapshots.empty:
        return path
    frame = snapshots.copy()
    frame["hour_utc"] = frame["snapshot_time_utc"].dt.hour
    metrics = {
        "spread_bps": "Spread bps",
        "bid_depth_5bps_usdt": "Bid depth 5bps",
        "ask_depth_5bps_usdt": "Ask depth 5bps",
        "depth_imbalance_5bps": "Imbalance 5bps",
    }
    hourly = frame.groupby("hour_utc", sort=True)[list(metrics)].median().reindex(range(24))
    data = hourly.T.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13, 4))
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title("Microstructure median metrics by UTC hour")
    ax.set_xlabel("UTC hour")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(list(metrics.values()))
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(hour) for hour in range(24)])
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(output_dir: Path, payload: dict[str, Any]) -> Path:
    report = output_dir / "execution_gate_report.md"
    snapshot_summary = pd.DataFrame(payload["snapshot_summary"])
    risk = pd.DataFrame(payload["slippage_risk_by_equity"])
    lines = [
        "# Iteration 021 - Microstructure Execution Filter 017",
        "",
        "## Verdict",
        f"`{payload['verdict']}`",
        "",
        "## Snapshot Summary",
        markdown_table(snapshot_summary) if not snapshot_summary.empty else "No snapshots available.",
        "",
        "## Slippage Risk By Equity",
        markdown_table(
            risk[
                [
                    "side",
                    "account_equity_usdt",
                    "initial_notional_multiplier",
                    "authorized_rate",
                    "side_depth_usdt_median",
                    "order_book_share_p90",
                    "theoretical_slippage_bps_p90",
                ]
            ].head(24)
        )
        if not risk.empty
        else "No gate decisions available.",
        "",
        "## Notes",
        "This is a public-market-data execution gate. It does not place orders, does not use API keys, and does not re-select Iteration 017 candidates. A live-ready verdict requires at least 24 hours of collected snapshots.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def evaluate_collected(config: dict[str, Any]) -> dict[str, Any]:
    validate_microstructure_config(config)
    output_dir = project_path(config["iteration"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshots(project_path(config["collection"]["snapshot_path"]))
    locked = load_locked_017_candidates(config)
    gate = gate_config_from_yaml(config)
    snapshot_summary = summarize_snapshots(snapshots, gate)
    decisions = evaluate_snapshot_grid(snapshots, config, locked)
    trades = build_locked_017_trades(config)
    attribution = attribute_snapshots_to_017_signals(snapshots, config, trades)
    risk = summarize_slippage_risk(decisions)
    heatmap = write_hourly_heatmap(snapshots, output_dir)
    verdict = decide_microstructure(snapshot_summary, decisions, config)

    snapshot_summary.to_csv(output_dir / "microstructure_snapshots_summary.csv", index=False)
    decisions.to_csv(output_dir / "microstructure_gate_decisions.csv", index=False)
    attribution.to_csv(output_dir / "microstructure_signal_attribution.csv", index=False)
    risk.to_csv(output_dir / "slippage_risk_by_equity.csv", index=False)
    payload = {
        "iteration_name": config["iteration"]["name"],
        "verdict": verdict,
        "snapshot_summary": snapshot_summary.to_dict("records"),
        "gate_decision_count": int(len(decisions)),
        "signal_attribution_count": int(len(attribution)),
        "matched_signal_attribution_count": int(attribution.get("matched_snapshot", pd.Series(dtype=bool)).sum())
        if not attribution.empty
        else 0,
        "slippage_risk_by_equity": risk.to_dict("records") if not risk.empty else [],
        "hourly_heatmap": str(heatmap),
    }
    (output_dir / "walk_forward_payload.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    report = write_report(output_dir, payload)
    LOGGER.info("Wrote microstructure execution outputs to %s", output_dir)
    LOGGER.info("Execution gate report: %s", report)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect/evaluate BTCUSDT microstructure execution gate for locked 017.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect-once", action="store_true")
    mode.add_argument("--evaluate-collected", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.collect_once:
        frame = collect_once(config)
        hidden_columns = {"bids_json", "asks_json", "depth_source_url", "book_ticker_source_url"}
        display_columns = [column for column in frame.columns if column not in hidden_columns]
        print(frame[display_columns].to_json(orient="records", date_format="iso", indent=2))
    elif args.evaluate_collected:
        payload = evaluate_collected(config)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
