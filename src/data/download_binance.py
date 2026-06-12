from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _to_ms(value: str | None) -> int | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int | None,
    end_ms: int | None,
    limit: int,
    base_url: str,
    endpoint: str,
) -> list[list[Any]]:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms
    url = f"{base_url}{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, dict) and "code" in payload:
        raise RuntimeError(f"Binance API error: {payload}")
    return payload


def download_range(
    start: str | None,
    end: str | None,
    output: Path,
    max_batches: int | None = None,
) -> Path:
    config = load_yaml("config/data_sources.yaml")["binance"]
    start_ms = _to_ms(start)
    end_ms = _to_ms(end)
    all_rows: list[list[Any]] = []
    batches = 0
    while True:
        rows = fetch_klines(
            symbol=config["symbol"],
            interval=config["interval"],
            start_ms=start_ms,
            end_ms=end_ms,
            limit=int(config["limit"]),
            base_url=config["base_url"],
            endpoint=config["klines_endpoint"],
        )
        if not rows:
            break
        all_rows.extend(rows)
        batches += 1
        LOGGER.info("Downloaded batch %s with %s rows", batches, len(rows))
        if len(rows) < int(config["limit"]):
            break
        if max_batches is not None and batches >= max_batches:
            break
        start_ms = int(rows[-1][6]) + 1
        if end_ms is not None and start_ms >= end_ms:
            break
        time.sleep(float(config["request_sleep_seconds"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle)
    LOGGER.info("Wrote %s klines to %s", len(all_rows), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public Binance BTCUSDT klines.")
    parser.add_argument("--start", default=None, help="ISO start date, e.g. 2024-01-01T00:00:00Z")
    parser.add_argument("--end", default=None, help="ISO end date")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--output",
        default=str(project_path("data/raw/btcusdt_5m_klines.json")),
    )
    args = parser.parse_args()
    download_range(args.start, args.end, Path(args.output), args.max_batches)


if __name__ == "__main__":
    main()

