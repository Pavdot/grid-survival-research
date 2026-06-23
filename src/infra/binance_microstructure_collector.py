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

import pandas as pd

from src.research.microstructure_execution_filter_017 import quote_depth_within_band
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/infrastructure_microstructure.yaml"


@dataclass(frozen=True)
class CollectorConfig:
    name: str
    symbol: str
    websocket_url: str
    rest_base_urls: tuple[str, ...]
    rest_depth_endpoint: str
    rest_depth_limit: int
    timeout_seconds: float
    snapshot_interval_seconds: float
    flush_every_rows: int
    max_buffered_events: int
    depth_bands_bps: tuple[float, ...]
    top_levels_to_store: int
    output_dir: Path
    health_path: Path
    reconnect_backoff_seconds: float
    max_snapshot_age_seconds_for_health: float


class LocalOrderBook:
    def __init__(self, bids: dict[float, float], asks: dict[float, float], last_update_id: int):
        self.bids = {float(price): float(qty) for price, qty in bids.items() if float(qty) > 0}
        self.asks = {float(price): float(qty) for price, qty in asks.items() if float(qty) > 0}
        self.last_update_id = int(last_update_id)

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "LocalOrderBook":
        bids = {float(price): float(qty) for price, qty in payload.get("bids", []) if float(qty) > 0}
        asks = {float(price): float(qty) for price, qty in payload.get("asks", []) if float(qty) > 0}
        if not bids or not asks:
            raise ValueError("REST depth snapshot must contain non-empty bids and asks")
        return cls(bids=bids, asks=asks, last_update_id=int(payload["lastUpdateId"]))

    def apply_update(self, event: dict[str, Any]) -> bool:
        first_update = int(event["U"])
        final_update = int(event["u"])
        if final_update <= self.last_update_id:
            return False
        if first_update > self.last_update_id + 1:
            raise ValueError(
                f"Depth stream sequence gap: event U={first_update}, local={self.last_update_id}"
            )
        self._apply_side(self.bids, event.get("b", []))
        self._apply_side(self.asks, event.get("a", []))
        if not self.bids or not self.asks:
            raise ValueError("Local order book became empty after update")
        self.last_update_id = final_update
        return True

    @staticmethod
    def _apply_side(levels: dict[float, float], updates: list[list[str]]) -> None:
        for price_raw, qty_raw, *_rest in updates:
            price = float(price_raw)
            qty = float(qty_raw)
            if qty <= 0:
                levels.pop(price, None)
            else:
                levels[price] = qty

    def bid_levels(self, limit: int | None = None) -> list[tuple[float, float]]:
        levels = sorted(self.bids.items(), key=lambda item: item[0], reverse=True)
        return levels if limit is None else levels[:limit]

    def ask_levels(self, limit: int | None = None) -> list[tuple[float, float]]:
        levels = sorted(self.asks.items(), key=lambda item: item[0])
        return levels if limit is None else levels[:limit]

    def best_bid_ask(self) -> tuple[float, float, float, float]:
        bid_price, bid_qty = self.bid_levels(1)[0]
        ask_price, ask_qty = self.ask_levels(1)[0]
        if ask_price <= bid_price:
            raise ValueError("Local order book is crossed")
        return bid_price, bid_qty, ask_price, ask_qty


def collector_config_from_yaml(config: dict[str, Any]) -> CollectorConfig:
    raw = config["collector"]
    return CollectorConfig(
        name=str(raw["name"]),
        symbol=str(raw["symbol"]).upper(),
        websocket_url=str(raw["websocket_url"]),
        rest_base_urls=tuple(str(value) for value in raw["rest_base_urls"]),
        rest_depth_endpoint=str(raw["rest_depth_endpoint"]),
        rest_depth_limit=int(raw["rest_depth_limit"]),
        timeout_seconds=float(raw["timeout_seconds"]),
        snapshot_interval_seconds=float(raw["snapshot_interval_seconds"]),
        flush_every_rows=int(raw["flush_every_rows"]),
        max_buffered_events=int(raw["max_buffered_events"]),
        depth_bands_bps=tuple(float(value) for value in raw["depth_bands_bps"]),
        top_levels_to_store=int(raw["top_levels_to_store"]),
        output_dir=project_path(raw["output_dir"]),
        health_path=project_path(raw["health_path"]),
        reconnect_backoff_seconds=float(raw["reconnect_backoff_seconds"]),
        max_snapshot_age_seconds_for_health=float(raw["max_snapshot_age_seconds_for_health"]),
    )


def validate_collector_config(config: CollectorConfig) -> None:
    if config.symbol != config.symbol.upper():
        raise ValueError("collector.symbol must be uppercase")
    if "order" in config.rest_depth_endpoint.lower():
        raise ValueError("collector must use public market-data endpoints only")
    if not config.websocket_url.startswith("wss://"):
        raise ValueError("collector.websocket_url must be a secure websocket URL")
    if config.rest_depth_limit <= 0 or config.rest_depth_limit > 5000:
        raise ValueError("collector.rest_depth_limit must stay within (0, 5000]")
    if config.snapshot_interval_seconds <= 0:
        raise ValueError("collector.snapshot_interval_seconds must be positive")
    if config.flush_every_rows <= 0:
        raise ValueError("collector.flush_every_rows must be positive")
    if config.max_buffered_events <= 0:
        raise ValueError("collector.max_buffered_events must be positive")
    if not config.depth_bands_bps or any(value <= 0 for value in config.depth_bands_bps):
        raise ValueError("collector.depth_bands_bps must contain positive values")
    if config.top_levels_to_store <= 0:
        raise ValueError("collector.top_levels_to_store must be positive")


def _json_get(url: str, timeout_seconds: float) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "grid-survival-research/infra"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return json.loads(raw.decode("utf-8")), latency_ms


def fetch_depth_snapshot(config: CollectorConfig) -> tuple[dict[str, Any], float, str]:
    query = urllib.parse.urlencode({"symbol": config.symbol, "limit": config.rest_depth_limit})
    errors: list[str] = []
    for base_url in config.rest_base_urls:
        url = f"{base_url.rstrip('/')}{config.rest_depth_endpoint}?{query}"
        try:
            payload, latency_ms = _json_get(url, config.timeout_seconds)
            return payload, latency_ms, url
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All Binance depth snapshot endpoints failed: " + " | ".join(errors))


def normalize_book_row(
    book: LocalOrderBook,
    config: CollectorConfig,
    source_latency_ms: float,
    stream_event_time_ms: int | None = None,
    event_receive_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    best_bid, best_bid_qty, best_ask, best_ask_qty = book.best_bid_ask()
    mid = (best_bid + best_ask) / 2.0
    bid_levels = book.bid_levels(config.top_levels_to_store)
    ask_levels = book.ask_levels(config.top_levels_to_store)
    row: dict[str, Any] = {
        "snapshot_time_utc": event_receive_time or pd.Timestamp.now(tz="UTC"),
        "symbol": config.symbol,
        "last_update_id": int(book.last_update_id),
        "stream_event_time_utc": pd.to_datetime(stream_event_time_ms, unit="ms", utc=True)
        if stream_event_time_ms is not None
        else pd.NaT,
        "best_bid": best_bid,
        "best_bid_qty": best_bid_qty,
        "best_ask": best_ask,
        "best_ask_qty": best_ask_qty,
        "mid_price": mid,
        "spread_bps": float((best_ask - best_bid) / mid * 10000.0),
        "source_latency_ms": float(source_latency_ms),
        "bids_json": json.dumps(bid_levels),
        "asks_json": json.dumps(ask_levels),
    }
    for band in config.depth_bands_bps:
        label = f"{band:g}".replace(".", "p")
        row[f"bid_depth_{label}bps_usdt"] = quote_depth_within_band(bid_levels, best_bid, float(band), "bid")
        row[f"ask_depth_{label}bps_usdt"] = quote_depth_within_band(ask_levels, best_ask, float(band), "ask")
    bid_5 = float(row.get("bid_depth_5bps_usdt", 0.0))
    ask_5 = float(row.get("ask_depth_5bps_usdt", 0.0))
    total_5 = bid_5 + ask_5
    row["depth_imbalance_5bps"] = float((bid_5 - ask_5) / total_5) if total_5 > 0 else float("nan")
    return row


def daily_output_path(config: CollectorConfig, timestamp: pd.Timestamp) -> Path:
    day = pd.Timestamp(timestamp).tz_convert("UTC").strftime("%Y-%m-%d")
    return config.output_dir / f"{config.symbol.lower()}_depth_{day}.parquet"


def append_daily_rows(rows: list[dict[str, Any]], config: CollectorConfig) -> list[Path]:
    if not rows:
        return []
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["snapshot_time_utc"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    written: list[Path] = []
    for day, group in frame.groupby(frame["snapshot_time_utc"].dt.strftime("%Y-%m-%d")):
        path = config.output_dir / f"{config.symbol.lower()}_depth_{day}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, group], ignore_index=True)
        else:
            combined = group.copy()
        combined["snapshot_time_utc"] = pd.to_datetime(combined["snapshot_time_utc"], utc=True)
        combined = combined.sort_values(["snapshot_time_utc", "last_update_id"]).drop_duplicates(
            ["snapshot_time_utc", "last_update_id"],
            keep="last",
        )
        combined.to_parquet(path, index=False)
        written.append(path)
    return written


def write_health(
    config: CollectorConfig,
    status: str,
    last_snapshot_time: pd.Timestamp | None,
    last_update_id: int | None,
    buffered_rows: int,
    error: str | None = None,
) -> None:
    config.health_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collector": config.name,
        "symbol": config.symbol,
        "status": status,
        "heartbeat_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "last_snapshot_time_utc": pd.Timestamp(last_snapshot_time).isoformat() if last_snapshot_time is not None else None,
        "last_update_id": last_update_id,
        "buffered_rows": int(buffered_rows),
        "error": error,
    }
    config.health_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def healthcheck(config: CollectorConfig, max_age_seconds: float | None = None) -> tuple[bool, str]:
    max_age = float(max_age_seconds or config.max_snapshot_age_seconds_for_health)
    if not config.health_path.exists():
        return False, f"health file missing: {config.health_path}"
    try:
        payload = json.loads(config.health_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid health json: {exc}"
    if payload.get("status") != "running":
        return False, f"collector status is {payload.get('status')}"
    last_raw = payload.get("last_snapshot_time_utc")
    if not last_raw:
        return False, "collector has not written a snapshot yet"
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_raw)) / pd.Timedelta(seconds=1)
    if age > max_age:
        return False, f"last snapshot is stale: {age:.1f}s > {max_age:.1f}s"
    return True, "collector healthy"


class BinanceDepthCollector:
    def __init__(self, config: CollectorConfig):
        validate_collector_config(config)
        self.config = config
        self.book: LocalOrderBook | None = None
        self.rows: list[dict[str, Any]] = []
        self.last_emit_monotonic = 0.0
        self.last_snapshot_time: pd.Timestamp | None = None

    def reset_from_rest(self) -> None:
        payload, latency_ms, url = fetch_depth_snapshot(self.config)
        self.book = LocalOrderBook.from_snapshot(payload)
        now = pd.Timestamp.now(tz="UTC")
        row = normalize_book_row(self.book, self.config, latency_ms, event_receive_time=now)
        self.rows.append(row)
        self.last_snapshot_time = now
        self.last_emit_monotonic = time.monotonic()
        write_health(self.config, "running", self.last_snapshot_time, self.book.last_update_id, len(self.rows))
        LOGGER.info("Seeded local book from %s at update id %s", url, self.book.last_update_id)

    def flush(self) -> None:
        if not self.rows:
            return
        written = append_daily_rows(self.rows, self.config)
        LOGGER.info("Flushed %d microstructure rows to %s", len(self.rows), ", ".join(str(path) for path in written))
        self.rows.clear()
        write_health(
            self.config,
            "running",
            self.last_snapshot_time,
            self.book.last_update_id if self.book is not None else None,
            len(self.rows),
        )

    def process_depth_event(self, event: dict[str, Any], receive_time: pd.Timestamp | None = None) -> bool:
        if self.book is None:
            raise RuntimeError("collector book is not initialized")
        receive_time = receive_time or pd.Timestamp.now(tz="UTC")
        applied = self.book.apply_update(event)
        now_monotonic = time.monotonic()
        if applied and now_monotonic - self.last_emit_monotonic >= self.config.snapshot_interval_seconds:
            event_time = int(event["E"]) if "E" in event else None
            latency_ms = (
                (receive_time - pd.to_datetime(event_time, unit="ms", utc=True)) / pd.Timedelta(milliseconds=1)
                if event_time is not None
                else 0.0
            )
            row = normalize_book_row(
                self.book,
                self.config,
                max(float(latency_ms), 0.0),
                stream_event_time_ms=event_time,
                event_receive_time=receive_time,
            )
            self.rows.append(row)
            self.last_snapshot_time = receive_time
            self.last_emit_monotonic = now_monotonic
            write_health(self.config, "running", self.last_snapshot_time, self.book.last_update_id, len(self.rows))
        if len(self.rows) >= self.config.flush_every_rows:
            self.flush()
        return applied

    def run(self, collect_seconds: float | None = None) -> None:
        deadline = time.monotonic() + float(collect_seconds) if collect_seconds is not None else None
        while True:
            try:
                self._run_websocket(deadline=deadline)
            except KeyboardInterrupt:
                write_health(
                    self.config,
                    "stopped",
                    self.last_snapshot_time,
                    self.book.last_update_id if self.book is not None else None,
                    len(self.rows),
                )
                raise
            except Exception as exc:  # noqa: BLE001 - collector must persist and report.
                write_health(
                    self.config,
                    "error",
                    self.last_snapshot_time,
                    self.book.last_update_id if self.book is not None else None,
                    len(self.rows),
                    error=str(exc),
                )
                LOGGER.exception("Collector loop failed; reconnecting after backoff")
                if deadline is not None and time.monotonic() >= deadline:
                    break
                time.sleep(self.config.reconnect_backoff_seconds)
            if deadline is not None and time.monotonic() >= deadline:
                break
        self.flush()

    def _run_websocket(self, deadline: float | None = None) -> None:
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package.
            raise RuntimeError("websocket-client is required for the live collector") from exc

        ws = websocket.create_connection(self.config.websocket_url, timeout=self.config.timeout_seconds)
        LOGGER.info("Connected to %s", self.config.websocket_url)
        try:
            # Binance local-book sync requires opening the stream before fetching
            # the REST snapshot, so buffered stream events can bridge the snapshot.
            self.reset_from_rest()
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                raw = ws.recv()
                if not raw:
                    continue
                event = json.loads(raw)
                if event.get("e") != "depthUpdate":
                    continue
                self.process_depth_event(event)
        finally:
            ws.close()
            self.flush()


def load_collector_config(path: str | Path = DEFAULT_CONFIG) -> CollectorConfig:
    config = collector_config_from_yaml(load_yaml(path))
    validate_collector_config(config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or check the BTCUSDT Binance microstructure collector.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect-seconds", type=float, default=None)
    parser.add_argument("--seed-once", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--max-age-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_collector_config(args.config)
    if args.healthcheck:
        ok, message = healthcheck(config, args.max_age_seconds)
        print(message)
        raise SystemExit(0 if ok else 1)
    collector = BinanceDepthCollector(config)
    if args.seed_once:
        collector.reset_from_rest()
        collector.flush()
        print(json.dumps({"seeded": True, "health_path": str(config.health_path)}, indent=2))
        return
    if args.run:
        collector.run(collect_seconds=args.collect_seconds)
        return
    raise SystemExit("Choose --run, --seed-once, or --healthcheck")


if __name__ == "__main__":
    main()
