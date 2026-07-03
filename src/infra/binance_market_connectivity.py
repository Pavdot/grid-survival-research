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

from src.research.microstructure_execution_filter_017 import parse_depth_levels, quote_depth_within_band
from src.utils.config_loader import load_yaml, project_path


DEFAULT_CONFIG = "config/binance_market_connectivity.yaml"
PRIVATE_ENDPOINT_MARKERS = (
    "/private",
    "listenkey",
    "/api/v3/order",
    "/fapi/v1/order",
    "/fapi/v1/batchorders",
    "/fapi/v1/account",
    "/fapi/v2/account",
    "/fapi/v3/account",
    "/fapi/v1/position",
    "/fapi/v2/position",
    "/fapi/v3/position",
    "/fapi/v1/balance",
    "/fapi/v2/balance",
    "/sapi/",
)


@dataclass(frozen=True)
class RestCheck:
    venue: str
    name: str
    base_url: str
    endpoint: str
    params: dict[str, Any]
    kind: str
    timeout_seconds: float


@dataclass(frozen=True)
class WebSocketCheck:
    venue: str
    name: str
    url: str
    required_fields: tuple[str, ...]
    expected_event: str | None
    timeout_seconds: float
    optional_no_event: bool


def _contains_private_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PRIVATE_ENDPOINT_MARKERS)


def validate_public_rest_endpoint(endpoint: str) -> None:
    if _contains_private_marker(endpoint):
        raise ValueError(f"Refusing private or trading REST endpoint: {endpoint}")


def validate_public_websocket_url(url: str) -> None:
    if not url.startswith("wss://"):
        raise ValueError("WebSocket URL must use wss://")
    if _contains_private_marker(url):
        raise ValueError(f"Refusing private or trading WebSocket URL: {url}")


def load_checks(path: str | Path = DEFAULT_CONFIG) -> tuple[list[RestCheck], list[WebSocketCheck], Path]:
    config = load_yaml(path)
    defaults = config.get("defaults", {})
    output_dir = project_path(config.get("output", {}).get("dir", "reports/infra/api_connectivity"))
    rest_checks: list[RestCheck] = []
    ws_checks: list[WebSocketCheck] = []
    for venue, venue_config in config.get("venues", {}).items():
        if not bool(venue_config.get("enabled", True)):
            continue
        default_timeout = float(venue_config.get("timeout_seconds", defaults.get("timeout_seconds", 10)))
        default_ws_timeout = float(
            venue_config.get("websocket_timeout_seconds", defaults.get("websocket_timeout_seconds", default_timeout))
        )
        for raw in venue_config.get("rest", []):
            endpoint = str(raw["endpoint"])
            validate_public_rest_endpoint(endpoint)
            rest_checks.append(
                RestCheck(
                    venue=str(venue),
                    name=str(raw["name"]),
                    base_url=str(raw["base_url"]),
                    endpoint=endpoint,
                    params=dict(raw.get("params", {})),
                    kind=str(raw.get("kind", "json")),
                    timeout_seconds=float(raw.get("timeout_seconds", default_timeout)),
                )
            )
        for raw in venue_config.get("websocket", []):
            url = str(raw["url"])
            validate_public_websocket_url(url)
            ws_checks.append(
                WebSocketCheck(
                    venue=str(venue),
                    name=str(raw["name"]),
                    url=url,
                    required_fields=tuple(str(field) for field in raw.get("required_fields", [])),
                    expected_event=str(raw["expected_event"]) if raw.get("expected_event") else None,
                    timeout_seconds=float(raw.get("timeout_seconds", default_ws_timeout)),
                    optional_no_event=bool(raw.get("optional_no_event", False)),
                )
            )
    return rest_checks, ws_checks, output_dir


def _json_get(url: str, timeout_seconds: float) -> tuple[Any, float]:
    start = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "grid-survival-research/connectivity"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return json.loads(raw.decode("utf-8")), latency_ms


def rest_url(check: RestCheck) -> str:
    query = urllib.parse.urlencode(check.params)
    base = f"{check.base_url.rstrip('/')}{check.endpoint}"
    return f"{base}?{query}" if query else base


def summarize_depth_payload(payload: dict[str, Any]) -> dict[str, Any]:
    bids = parse_depth_levels(payload.get("bids", []))
    asks = parse_depth_levels(payload.get("asks", []))
    if not bids or not asks:
        raise ValueError("depth payload must contain non-empty bids and asks")
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    if best_ask <= best_bid:
        raise ValueError("depth payload is crossed")
    mid = (best_bid + best_ask) / 2.0
    bid_depth_5 = quote_depth_within_band(bids, best_bid, 5.0, "bid")
    ask_depth_5 = quote_depth_within_band(asks, best_ask, 5.0, "ask")
    total_depth_5 = bid_depth_5 + ask_depth_5
    return {
        "last_update_id": int(payload.get("lastUpdateId", -1)),
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "spread_bps": float((best_ask - best_bid) / mid * 10000.0),
        "bid_depth_5bps_usdt": float(bid_depth_5),
        "ask_depth_5bps_usdt": float(ask_depth_5),
        "depth_imbalance_5bps": float((bid_depth_5 - ask_depth_5) / total_depth_5) if total_depth_5 > 0 else None,
        "bid_levels": int(len(bids)),
        "ask_levels": int(len(asks)),
    }


def summarize_rest_payload(kind: str, payload: Any) -> dict[str, Any]:
    if kind == "depth":
        if not isinstance(payload, dict):
            raise ValueError("depth REST response must be a JSON object")
        return summarize_depth_payload(payload)
    if kind == "book_ticker":
        if not isinstance(payload, dict):
            raise ValueError("book ticker REST response must be a JSON object")
        best_bid = float(payload["bidPrice"])
        best_ask = float(payload["askPrice"])
        mid = (best_bid + best_ask) / 2.0
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "best_bid_qty": float(payload["bidQty"]),
            "best_ask_qty": float(payload["askQty"]),
            "spread_bps": float((best_ask - best_bid) / mid * 10000.0) if mid > 0 else None,
        }
    if kind == "klines":
        if not isinstance(payload, list):
            raise ValueError("klines REST response must be a JSON list")
        last = payload[-1] if payload else None
        return {
            "row_count": int(len(payload)),
            "last_open_time_utc": pd.to_datetime(int(last[0]), unit="ms", utc=True).isoformat() if last else None,
            "last_close_time_utc": pd.to_datetime(int(last[6]), unit="ms", utc=True).isoformat() if last else None,
        }
    if isinstance(payload, dict):
        return {"keys": ",".join(sorted(str(key) for key in payload.keys())[:20])}
    if isinstance(payload, list):
        return {"row_count": int(len(payload))}
    return {"payload_type": type(payload).__name__}


def run_rest_check(check: RestCheck) -> dict[str, Any]:
    url = rest_url(check)
    row: dict[str, Any] = {
        "check_type": "rest",
        "venue": check.venue,
        "name": check.name,
        "status": "failed",
        "url": url,
        "latency_ms": None,
        "message": "",
    }
    try:
        payload, latency_ms = _json_get(url, check.timeout_seconds)
        row.update(summarize_rest_payload(check.kind, payload))
        row["status"] = "ok"
        row["latency_ms"] = float(latency_ms)
        row["message"] = "ok"
    except Exception as exc:  # noqa: BLE001 - connectivity report must capture all failures.
        row["message"] = str(exc)
    return row


def _unwrap_stream_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("WebSocket message must decode to a JSON object")


def _event_matches(payload: dict[str, Any], expected_event: str | None) -> bool:
    if expected_event is None:
        return True
    return str(payload.get("e")) == expected_event


def _missing_fields(payload: dict[str, Any], required_fields: tuple[str, ...]) -> list[str]:
    return [field for field in required_fields if field not in payload]


def websocket_payload_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "event_type": payload.get("e"),
        "symbol": payload.get("s"),
        "event_time_utc": pd.to_datetime(int(payload["E"]), unit="ms", utc=True).isoformat() if "E" in payload else None,
        "last_update_id": payload.get("u"),
    }
    if {"b", "a"}.issubset(payload.keys()) and isinstance(payload.get("b"), str):
        best_bid = float(payload["b"])
        best_ask = float(payload["a"])
        mid = (best_bid + best_ask) / 2.0
        summary.update(
            {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "best_bid_qty": float(payload.get("B", 0.0)),
                "best_ask_qty": float(payload.get("A", 0.0)),
                "spread_bps": float((best_ask - best_bid) / mid * 10000.0) if mid > 0 else None,
            }
        )
    if "k" in payload and isinstance(payload["k"], dict):
        kline = payload["k"]
        summary.update(
            {
                "kline_interval": kline.get("i"),
                "kline_closed": bool(kline.get("x")),
                "kline_open_time_utc": pd.to_datetime(int(kline["t"]), unit="ms", utc=True).isoformat()
                if "t" in kline
                else None,
            }
        )
    return summary


def run_websocket_check(check: WebSocketCheck) -> dict[str, Any]:
    row: dict[str, Any] = {
        "check_type": "websocket",
        "venue": check.venue,
        "name": check.name,
        "status": "failed",
        "url": check.url,
        "latency_ms": None,
        "message": "",
    }
    start = time.perf_counter()
    try:
        import websocket
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.
        row["message"] = f"websocket-client missing: {exc}"
        return row
    ws = None
    try:
        ws = websocket.create_connection(check.url, timeout=check.timeout_seconds)
        deadline = time.monotonic() + check.timeout_seconds
        last_message = "connected; waiting for event"
        while time.monotonic() < deadline:
            try:
                raw = ws.recv()
            except (TimeoutError, websocket.WebSocketTimeoutException):
                continue
            if not raw:
                continue
            payload = _unwrap_stream_payload(json.loads(raw))
            if not _event_matches(payload, check.expected_event):
                last_message = f"skipped event {payload.get('e')!r}"
                continue
            missing = _missing_fields(payload, check.required_fields)
            if missing:
                raise ValueError(f"WebSocket payload missing fields: {missing}")
            row.update(websocket_payload_summary(payload))
            row["status"] = "ok"
            row["latency_ms"] = float((time.perf_counter() - start) * 1000.0)
            row["message"] = "ok"
            return row
        if check.optional_no_event:
            row["status"] = "no_event"
            row["latency_ms"] = float((time.perf_counter() - start) * 1000.0)
            row["message"] = last_message
        else:
            row["message"] = f"no matching event within {check.timeout_seconds:g}s; {last_message}"
    except Exception as exc:  # noqa: BLE001 - connectivity report must capture all failures.
        row["message"] = str(exc)
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
    return row


def run_connectivity_checks(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    rest_only: bool = False,
    websocket_only: bool = False,
) -> tuple[pd.DataFrame, Path]:
    rest_checks, ws_checks, output_dir = load_checks(config_path)
    rows: list[dict[str, Any]] = []
    if not websocket_only:
        rows.extend(run_rest_check(check) for check in rest_checks)
    if not rest_only:
        rows.extend(run_websocket_check(check) for check in ws_checks)
    frame = pd.DataFrame(rows)
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"binance_market_connectivity_{timestamp}.csv"
    json_path = output_dir / f"binance_market_connectivity_{timestamp}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "config_path": str(project_path(config_path) if not Path(config_path).is_absolute() else Path(config_path)),
                "summary": summarize_statuses(frame),
                "rows": frame.where(pd.notna(frame), None).to_dict(orient="records"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    latest_csv = output_dir / "binance_market_connectivity_latest.csv"
    latest_json = output_dir / "binance_market_connectivity_latest.json"
    frame.to_csv(latest_csv, index=False)
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return frame, output_dir


def summarize_statuses(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"total": 0, "ok": 0, "failed": 0, "no_event": 0}
    counts = frame["status"].value_counts(dropna=False).to_dict()
    return {
        "total": int(len(frame)),
        "ok": int(counts.get("ok", 0)),
        "failed": int(counts.get("failed", 0)),
        "no_event": int(counts.get("no_event", 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check public Binance market-data REST and WebSocket connectivity.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument("--websocket-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rest_only and args.websocket_only:
        raise SystemExit("--rest-only and --websocket-only are mutually exclusive")
    frame, output_dir = run_connectivity_checks(
        args.config,
        rest_only=args.rest_only,
        websocket_only=args.websocket_only,
    )
    print(frame[["check_type", "venue", "name", "status", "latency_ms", "message"]].to_string(index=False))
    print(json.dumps({"output_dir": str(output_dir), "summary": summarize_statuses(frame)}, indent=2))
    failures = frame[frame["status"].eq("failed")] if not frame.empty else frame
    raise SystemExit(1 if not failures.empty else 0)


if __name__ == "__main__":
    main()
