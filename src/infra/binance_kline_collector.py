from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/shadow_live_037.yaml"
PUBLIC_KLINE_ENDPOINTS = {
    "spot": "/api/v3/klines",
    "futures_usdm": "/fapi/v1/klines",
}
PUBLIC_KLINE_HOSTS = {
    "spot": {"api.binance.com", "data-api.binance.vision"},
    "futures_usdm": {"fapi.binance.com"},
}
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
]


@dataclass(frozen=True)
class KlineCollectorConfig:
    name: str
    symbol: str
    interval: str
    rest_base_urls: tuple[str, ...]
    rest_klines_endpoint: str
    rest_limit: int
    timeout_seconds: float
    poll_interval_seconds: float
    lookback_days: float
    output_path: Path
    resampled_1h_path: Path
    health_path: Path
    audit_path: Path
    market: str = "spot"


def interval_to_minutes(interval: str) -> int:
    if interval == "5m":
        return 5
    raise ValueError("Only 5m klines are supported for the shadow live pipeline")


def kline_config_from_yaml(
    config: dict[str, Any],
    timeframe: str | None = None,
    section: str = "kline_collector",
) -> KlineCollectorConfig:
    if section not in config:
        raise ValueError(f"kline config section is missing: {section}")
    raw = config[section]
    interval = str(timeframe or raw.get("interval", "5m"))
    return KlineCollectorConfig(
        name=str(raw.get("name", "btcusdt_closed_kline_collector")),
        symbol=str(raw.get("symbol", "BTCUSDT")).upper(),
        interval=interval,
        rest_base_urls=tuple(str(value) for value in raw.get("rest_base_urls", ["https://api.binance.com"])),
        rest_klines_endpoint=str(raw.get("rest_klines_endpoint", "/api/v3/klines")),
        rest_limit=int(raw.get("rest_limit", 1000)),
        timeout_seconds=float(raw.get("timeout_seconds", 10)),
        poll_interval_seconds=float(raw.get("poll_interval_seconds", 60)),
        lookback_days=float(raw.get("lookback_days", 14)),
        output_path=project_path(raw.get("output_path", "data/live/btcusdt_5m_closed.parquet")),
        resampled_1h_path=project_path(raw.get("resampled_1h_path", "data/live/btcusdt_1h_closed.parquet")),
        health_path=project_path(raw.get("health_path", "data/live/btcusdt_kline_health.json")),
        audit_path=project_path(raw.get("audit_path", "reports/shadow_live_037/kline_data_audit.json")),
        market=str(raw.get("market", "spot")),
    )


def validate_kline_config(config: KlineCollectorConfig) -> None:
    if config.symbol != config.symbol.upper():
        raise ValueError("kline collector symbol must be uppercase")
    if config.market not in PUBLIC_KLINE_ENDPOINTS:
        raise ValueError(f"unsupported public Binance market: {config.market}")
    if "order" in config.rest_klines_endpoint.lower():
        raise ValueError("kline collector must use public market-data endpoints only")
    expected_endpoint = PUBLIC_KLINE_ENDPOINTS[config.market]
    if config.rest_klines_endpoint != expected_endpoint:
        raise ValueError(f"{config.market} kline collector endpoint must be {expected_endpoint}")
    allowed_hosts = PUBLIC_KLINE_HOSTS[config.market]
    for base_url in config.rest_base_urls:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
            raise ValueError(f"unapproved {config.market} public kline base URL: {base_url}")
    if interval_to_minutes(config.interval) != 5:
        raise ValueError("kline collector interval must be 5m")
    if config.rest_limit <= 0 or config.rest_limit > 1000:
        raise ValueError("kline collector rest_limit must stay within (0, 1000]")
    if config.poll_interval_seconds <= 0:
        raise ValueError("kline collector poll_interval_seconds must be positive")
    if config.lookback_days <= 0:
        raise ValueError("kline collector lookback_days must be positive")


def _json_get(url: str, timeout_seconds: float) -> tuple[list[Any], float]:
    start = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "grid-survival-research/kline"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    latency_ms = (time.perf_counter() - start) * 1000.0
    return json.loads(raw.decode("utf-8")), latency_ms


def fetch_klines(
    config: KlineCollectorConfig,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> tuple[list[Any], float, str]:
    query: dict[str, object] = {
        "symbol": config.symbol,
        "interval": config.interval,
        "limit": config.rest_limit,
    }
    if start_time_ms is not None:
        query["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        query["endTime"] = int(end_time_ms)
    encoded = urllib.parse.urlencode(query)
    errors: list[str] = []
    for base_url in config.rest_base_urls:
        url = f"{base_url.rstrip('/')}{config.rest_klines_endpoint}?{encoded}"
        try:
            payload, latency_ms = _json_get(url, config.timeout_seconds)
            if not isinstance(payload, list):
                raise ValueError("Binance klines response must be a list")
            return payload, latency_ms, url
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All Binance kline endpoints failed: " + " | ".join(errors))


def normalize_klines(
    klines: list[Any],
    now: pd.Timestamp | None = None,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    now = now or pd.Timestamp.now(tz="UTC")
    now = pd.Timestamp(now).tz_convert("UTC")
    rows: list[dict[str, Any]] = []
    for item in klines:
        if len(item) < 7:
            raise ValueError("Binance kline row must contain at least 7 fields")
        open_time = pd.to_datetime(int(item[0]), unit="ms", utc=True)
        close_time = pd.to_datetime(int(item[6]), unit="ms", utc=True)
        if close_time >= now:
            continue
        expected_close = open_time + pd.Timedelta(minutes=interval_minutes) - pd.Timedelta(milliseconds=1)
        if close_time != expected_close:
            raise ValueError("Unexpected kline close_time for configured interval")
        rows.append(
            {
                "open_time": open_time,
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "close_time": close_time,
            }
        )
    frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS).set_index(pd.DatetimeIndex([], name="open_time", tz="UTC"))
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    frame = frame.sort_values("open_time").drop_duplicates("open_time", keep="last")
    if frame["open_time"].dt.tz is None:
        raise ValueError("kline open_time must be timezone-aware")
    frame = frame.set_index("open_time", drop=False)
    frame.index.name = "open_time"
    return frame


def load_kline_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=KLINE_COLUMNS).set_index(pd.DatetimeIndex([], name="open_time", tz="UTC"))
    frame = pd.read_parquet(path)
    if "open_time" in frame.columns:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame = frame.set_index("open_time", drop=False)
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame["open_time"] = frame.index
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"kline parquet contains duplicate open_time values: {path}")
    if frame.index.tz is None:
        raise ValueError(f"kline parquet index must be timezone-aware: {path}")
    return frame


def write_kline_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy().reset_index(drop=True)
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True)
    out = out.sort_values("open_time").drop_duplicates("open_time", keep="last")
    out.to_parquet(path, index=False)


def append_kline_rows(rows: pd.DataFrame, config: KlineCollectorConfig) -> pd.DataFrame:
    existing = load_kline_frame(config.output_path)
    combined = pd.concat([existing.reset_index(drop=True), rows.reset_index(drop=True)], ignore_index=True)
    if combined.empty:
        write_kline_frame(existing, config.output_path)
        return existing
    combined["open_time"] = pd.to_datetime(combined["open_time"], utc=True)
    combined["close_time"] = pd.to_datetime(combined["close_time"], utc=True)
    combined = combined.sort_values("open_time").drop_duplicates("open_time", keep="last")
    combined = combined.set_index("open_time", drop=False)
    combined.index.name = "open_time"
    write_kline_frame(combined, config.output_path)
    return combined


def resample_closed_1h_from_5m(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS).set_index(pd.DatetimeIndex([], name="open_time", tz="UTC"))
    data = frame.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    if data.index.tz is None:
        raise ValueError("5m kline index must be timezone-aware")
    data = data.sort_index()
    data["hour"] = data.index.floor("h")
    rows: list[dict[str, Any]] = []
    expected_delta = pd.Timedelta(minutes=5)
    for hour, group in data.groupby("hour", sort=True):
        group = group.sort_index()
        expected_index = pd.date_range(hour, periods=12, freq=expected_delta, tz="UTC")
        if len(group) != 12 or not group.index.equals(expected_index):
            continue
        rows.append(
            {
                "open_time": hour,
                "open": float(group["open"].iloc[0]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": float(group["close"].iloc[-1]),
                "volume": float(group["volume"].sum()),
                "close_time": pd.Timestamp(hour) + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            }
        )
    out = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    if out.empty:
        return pd.DataFrame(columns=KLINE_COLUMNS).set_index(pd.DatetimeIndex([], name="open_time", tz="UTC"))
    out["open_time"] = pd.to_datetime(out["open_time"], utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True)
    out = out.set_index("open_time", drop=False)
    out.index.name = "open_time"
    return out


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_kline_frame(frame: pd.DataFrame, path: Path | None = None, min_coverage: float = 0.99) -> dict[str, Any]:
    if frame.empty:
        return {
            "status": "bad",
            "reason": "empty kline frame",
            "observed_bars": 0,
            "expected_bars": 0,
            "coverage_rate": 0.0,
            "duplicate_count": 0,
            "gap_count": 0,
            "sha256": sha256_file(path) if path is not None else None,
        }
    index = pd.to_datetime(frame.index, utc=True)
    if index.tz is None:
        raise ValueError("kline audit requires timezone-aware index")
    duplicate_count = int(index.duplicated().sum())
    if duplicate_count:
        raise ValueError("kline audit refuses duplicate timestamps")
    expected = pd.date_range(index.min(), index.max(), freq="5min", tz="UTC")
    missing = expected.difference(index)
    observed = int(len(index))
    expected_count = int(len(expected))
    coverage = float(observed / expected_count) if expected_count else 0.0
    status = "healthy" if coverage >= min_coverage and not len(missing) else "degraded"
    if coverage < min_coverage:
        status = "bad"
    return {
        "status": status,
        "reason": "ok" if status == "healthy" else "coverage_or_gaps",
        "start_utc": index.min().isoformat(),
        "end_utc": index.max().isoformat(),
        "observed_bars": observed,
        "expected_bars": expected_count,
        "coverage_rate": coverage,
        "duplicate_count": duplicate_count,
        "gap_count": int(len(missing)),
        "first_gap_utc": missing[0].isoformat() if len(missing) else None,
        "sha256": sha256_file(path) if path is not None else None,
    }


def write_health(config: KlineCollectorConfig, status: str, last_open_time: pd.Timestamp | None, error: str | None = None) -> None:
    config.health_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collector": config.name,
        "market": config.market,
        "symbol": config.symbol,
        "interval": config.interval,
        "status": status,
        "heartbeat_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "last_open_time_utc": pd.Timestamp(last_open_time).isoformat() if last_open_time is not None else None,
        "error": error,
    }
    config.health_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def healthcheck(config: KlineCollectorConfig, max_age_minutes: float | None = None) -> tuple[bool, str]:
    max_age = float(max_age_minutes if max_age_minutes is not None else 15.0)
    if not config.health_path.exists():
        return False, f"kline health file missing: {config.health_path}"
    try:
        payload = json.loads(config.health_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"invalid kline health json: {exc}"
    if payload.get("status") != "running":
        return False, f"kline collector status is {payload.get('status')}"
    last_raw = payload.get("last_open_time_utc")
    if not last_raw:
        return False, "kline collector has not written a closed bar yet"
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_raw)) / pd.Timedelta(minutes=1)
    if age > max_age:
        return False, f"last closed kline is stale: {age:.1f}m > {max_age:.1f}m"
    return True, "kline collector healthy"


def write_audit(config: KlineCollectorConfig, frame: pd.DataFrame) -> dict[str, Any]:
    config.audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_kline_frame(frame, config.output_path)
    config.audit_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    return audit


def seed_klines(config: KlineCollectorConfig) -> dict[str, Any]:
    validate_kline_config(config)
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=config.lookback_days)
    interval_ms = interval_to_minutes(config.interval) * 60 * 1000
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    payload_rows: list[Any] = []
    latency_values: list[float] = []
    source_urls: list[str] = []
    while cursor_ms <= end_ms:
        payload, latency_ms, url = fetch_klines(config, cursor_ms, end_ms)
        if not payload:
            break
        payload_rows.extend(payload)
        latency_values.append(float(latency_ms))
        source_urls.append(url)
        last_open_ms = int(payload[-1][0])
        next_cursor = last_open_ms + interval_ms
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor
        if len(payload) < config.rest_limit:
            break
    rows = normalize_klines(payload_rows, now=end, interval_minutes=interval_to_minutes(config.interval))
    combined = append_kline_rows(rows, config)
    one_hour = resample_closed_1h_from_5m(combined)
    write_kline_frame(one_hour, config.resampled_1h_path)
    write_health(config, "running", combined.index.max() if not combined.empty else None)
    audit = write_audit(config, combined)
    return {
        "rows_fetched": int(len(payload_rows)),
        "rows_closed": int(len(rows)),
        "rows_total": int(len(combined)),
        "rows_1h": int(len(one_hour)),
        "latency_ms_total": float(sum(latency_values)),
        "request_count": int(len(source_urls)),
        "last_source_url": source_urls[-1] if source_urls else "",
        "audit": audit,
    }


def run_loop(config: KlineCollectorConfig, collect_seconds: float | None = None) -> None:
    deadline = time.monotonic() + float(collect_seconds) if collect_seconds is not None else None
    while True:
        try:
            payload = seed_klines(config)
            LOGGER.info("Updated closed klines: %s", payload)
        except Exception as exc:  # noqa: BLE001 - collector must report and continue.
            write_health(config, "error", None, error=str(exc))
            LOGGER.exception("Kline collector update failed")
        if deadline is not None and time.monotonic() >= deadline:
            break
        time.sleep(config.poll_interval_seconds)


def load_kline_config(
    path: str | Path = DEFAULT_CONFIG,
    timeframe: str | None = None,
    section: str = "kline_collector",
) -> KlineCollectorConfig:
    config = kline_config_from_yaml(load_yaml(path), timeframe=timeframe, section=section)
    validate_kline_config(config)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect closed public Binance BTCUSDT 5m klines.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--section", default="kline_collector")
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect-seconds", type=float, default=None)
    parser.add_argument("--healthcheck", action="store_true")
    parser.add_argument("--max-age-minutes", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_kline_config(args.config, timeframe=args.timeframe, section=args.section)
    if args.healthcheck:
        ok, message = healthcheck(config, max_age_minutes=args.max_age_minutes)
        print(message)
        raise SystemExit(0 if ok else 1)
    if args.seed:
        print(json.dumps(seed_klines(config), indent=2, default=str))
        return
    if args.run:
        run_loop(config, collect_seconds=args.collect_seconds)
        return
    raise SystemExit("Choose --seed or --run")


if __name__ == "__main__":
    main()
