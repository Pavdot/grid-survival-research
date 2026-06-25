from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.data.resample_timeframes import resample_closed_ohlcv
from src.data.validate_data import load_processed, validate_ohlcv
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.research.surgical_veto_optimizer_037 import candidate_with_veto
from src.research.zero_fee_p10_optimizer_035 import evaluate_selected, scenario_spec
from src.fundamentals.event_blackout import build_blackout_bundle
from src.utils.config_loader import load_strategy_config, load_yaml, project_path
from src.utils.time_utils import timeframe_to_minutes


DEFAULT_CONFIG = "config/shadow_live_037.yaml"
OUTPUT_ROOT = "research_evidence"
TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h"]
ONE_MINUTE_CACHE = "data/live/btcusdt_1m_037_oos_full.parquet"
SCENARIOS = [
    ("zero_fee_0bps", 0.0, 0.0, 0.0),
    ("zero_fee_0p25bps", 0.0, 0.25, 0.0),
    ("zero_fee_0p5bps", 0.0, 0.5, 0.0),
    ("zero_fee_1bps", 0.0, 1.0, 0.0),
    ("zero_fee_maker_miss10", 0.0, 0.0, 0.10),
    ("realistic_control_fee40_slip2", 0.0004, 2.0, 0.0),
]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_path(value)


def complete_resample_from_5m(frame: pd.DataFrame, timeframe: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resample from 5m and keep only higher-timeframe candles with every 5m bar present."""
    if timeframe == "5m":
        out = frame.copy()
        return out, {
            "timeframe": timeframe,
            "expected_5m_per_signal_bar": 1,
            "raw_signal_bars": int(len(out)),
            "complete_signal_bars": int(len(out)),
            "dropped_incomplete_signal_bars": 0,
        }
    minutes = timeframe_to_minutes(timeframe)
    if minutes % 5 != 0:
        raise ValueError("signal timeframe must be a multiple of 5m")
    expected = minutes // 5
    freq = f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}min"
    sorted_frame = frame.sort_index()
    grouped = sorted_frame.resample(freq, label="right", closed="right", origin="start_day")
    counts = grouped["close"].count()
    raw = resample_closed_ohlcv(sorted_frame, timeframe)
    complete = raw.loc[counts.reindex(raw.index).eq(expected)].copy()
    complete = complete.dropna(subset=["open", "high", "low", "close"])
    validate_ohlcv(complete, timeframe)
    return complete, {
        "timeframe": timeframe,
        "expected_5m_per_signal_bar": int(expected),
        "raw_signal_bars": int(len(raw)),
        "complete_signal_bars": int(len(complete)),
        "dropped_incomplete_signal_bars": int(len(raw) - len(complete)),
        "first_signal_timestamp": str(complete.index.min()) if len(complete) else None,
        "last_signal_timestamp": str(complete.index.max()) if len(complete) else None,
    }


def normalize_binance_klines_1m(rows: list[list[Any]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("Binance 1m kline response is empty")
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    frame[numeric] = frame[numeric].astype(float)
    frame["open_time"] = frame["open_time"].astype("int64")
    frame["close_time"] = frame["close_time"].astype("int64")
    frame["number_of_trades"] = frame["number_of_trades"].astype("int64")
    frame["open_datetime"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_datetime"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame.index = frame["open_datetime"] + pd.Timedelta(minutes=1)
    frame.index.name = "timestamp"
    frame = frame.drop(columns=["ignore"]).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    validate_ohlcv(frame, "1m")
    return frame


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    cursor = pd.Timestamp(start).tz_convert("UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last = pd.Timestamp(end).tz_convert("UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: list[pd.Timestamp] = []
    while cursor <= last:
        months.append(cursor)
        cursor = cursor + pd.DateOffset(months=1)
    return months


def _download_monthly_archive_1m(month_start: pd.Timestamp) -> pd.DataFrame:
    stamp = pd.Timestamp(month_start).strftime("%Y-%m")
    url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{stamp}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return pd.DataFrame()
        raise
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_name) as handle:
            frame = pd.read_csv(handle, header=None)
    if not frame.empty and str(frame.iloc[0, 0]).lower() in {"open_time", "open time"}:
        frame = frame.iloc[1:].reset_index(drop=True)
    frame = frame.iloc[:, :12]
    frame.columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
    frame["close_time"] = pd.to_numeric(frame["close_time"], errors="raise").astype("int64")
    if int(frame["open_time"].max()) > 10_000_000_000_000:
        frame["open_time"] = frame["open_time"] // 1000
        frame["close_time"] = frame["close_time"] // 1000
    return normalize_binance_klines_1m(frame.values.tolist())


def _missing_segments(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if index.empty:
        return []
    ordered = pd.DatetimeIndex(index.sort_values().unique())
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = ordered[0]
    last = ordered[0]
    for ts in ordered[1:]:
        if ts - last > pd.Timedelta(minutes=1):
            segments.append((start, last))
            start = ts
        last = ts
    segments.append((start, last))
    return segments


def fetch_binance_1m_klines_rest(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    rows: list[list[Any]] = []
    cursor_ms = int((start - pd.Timedelta(minutes=1)).timestamp() * 1000)
    end_ms = int((end - pd.Timedelta(minutes=1)).timestamp() * 1000)
    base_urls = ["https://api.binance.com", "https://data-api.binance.vision"]
    while cursor_ms <= end_ms:
        params = urllib.parse.urlencode(
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": 1000,
            }
        )
        last_error: Exception | None = None
        chunk: list[list[Any]] | None = None
        for base_url in base_urls:
            url = f"{base_url}/api/v3/klines?{params}"
            try:
                with urllib.request.urlopen(url, timeout=20) as response:
                    chunk = json.loads(response.read().decode("utf-8"))
                break
            except Exception as exc:  # pragma: no cover - network fallback
                last_error = exc
        if chunk is None:
            raise RuntimeError(f"Unable to fetch Binance 1m klines: {last_error}") from last_error
        if not chunk:
            break
        rows.extend(chunk)
        next_cursor = int(chunk[-1][0]) + 60_000
        if next_cursor <= cursor_ms:
            break
        cursor_ms = next_cursor
        time.sleep(0.01)

    frame = normalize_binance_klines_1m(rows)
    return frame[(frame.index >= start) & (frame.index <= end)].copy()


def fetch_binance_1m_klines(start: pd.Timestamp, end: pd.Timestamp, cache_path: Path) -> pd.DataFrame:
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    if cache_path.exists():
        cached = load_processed(cache_path)
        if cached.index.min() <= start and cached.index.max() >= end:
            return cached[(cached.index >= start) & (cached.index <= end)].copy()

    frames: list[pd.DataFrame] = []
    for month in _month_starts(start, end):
        monthly = _download_monthly_archive_1m(month)
        if not monthly.empty:
            frames.append(monthly)
    combined = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined[(combined.index >= start) & (combined.index <= end)]

    expected = pd.date_range(start, end, freq="1min", tz="UTC", name="timestamp")
    observed = pd.DatetimeIndex(combined.index) if not combined.empty else pd.DatetimeIndex([], tz="UTC", name="timestamp")
    missing = expected.difference(observed)
    if len(missing):
        for segment_start, segment_end in _missing_segments(missing):
            frames.append(fetch_binance_1m_klines_rest(segment_start, segment_end))
    full = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    if full.empty:
        raise ValueError("No Binance 1m data could be fetched")
    full = full[~full.index.duplicated(keep="last")]
    full = full[(full.index >= start) & (full.index <= end)]
    validate_ohlcv(full, "1m")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(cache_path)
    return full.copy()


def locked_037_row(config: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(config["strategy_037"]["candidate"])
    veto = dict(config["strategy_037"]["veto"])
    row = {
        **candidate,
        "candidate_uid": str(veto.get("veto_uid", "veto_none")),
        "blackout_hours": int(config["fundamental_blackout"]["pre_event_hours"]),
        "min_severity": int(config["fundamental_blackout"]["min_severity"]),
        "trend_propagation_bars": int(config["trend_escape"]["propagation_bars"]),
        "breakout_atr_buffer": float(config["trend_escape"]["breakout_atr_buffer"]),
        "min_range_expansion_ratio": float(config["trend_escape"]["min_range_expansion_ratio"]),
        "max_grids_per_month": float(veto.get("max_grids_per_month", 999.0)),
        "pause_after_forced_loss_hours": float(veto.get("pause_after_forced_loss_hours", 0.0)),
        "rolling_7d_loss_threshold": float(veto.get("rolling_7d_loss_threshold", -999.0)),
    }
    return candidate_with_veto(row, veto)


def holdout_index_from_037(market: pd.DataFrame, config: dict[str, Any]) -> pd.Index:
    manifest_path = resolve_path(config["strategy_037"]["source_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = pd.Timestamp(manifest["holdout_start_utc"]).tz_convert("UTC")
    end = pd.Timestamp(manifest["holdout_end_utc"]).tz_convert("UTC")
    index = market.index[(market.index >= start) & (market.index <= end)]
    if index.empty:
        raise ValueError("037 holdout index is empty for the loaded market")
    return index


def evaluation_index_for_window(market: pd.DataFrame, start: str | None, end: str | None, config: dict[str, Any]) -> pd.Index:
    if not start and not end:
        return holdout_index_from_037(market, config)
    if not start or not end:
        raise ValueError("--start and --end must be provided together")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None or end_ts.tzinfo is None:
        raise ValueError("--start and --end must be timezone-aware")
    start_ts = start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_convert("UTC")
    if start_ts >= end_ts:
        raise ValueError("--start must be before --end")
    index = market.index[(market.index >= start_ts) & (market.index <= end_ts)]
    if index.empty:
        raise ValueError("custom evaluation window is empty for the loaded market")
    return index


def load_market_5m() -> pd.DataFrame:
    path = resolve_path("data/processed/btcusdt_5m.parquet")
    frame = load_processed(path)
    validate_ohlcv(frame, "5m")
    return frame


def signal_frame_for_timeframe(raw_5m: pd.DataFrame, timeframe: str, holdout_index: pd.Index) -> tuple[pd.DataFrame, dict[str, Any]]:
    if timeframe == "1m":
        warmup_start = pd.Timestamp(holdout_index.min()) - pd.Timedelta(days=2)
        end = pd.Timestamp(holdout_index.max())
        signal = fetch_binance_1m_klines(warmup_start, end, resolve_path(ONE_MINUTE_CACHE))
        return signal, {
            "timeframe": "1m",
            "expected_5m_per_signal_bar": 0,
            "raw_signal_bars": int(len(signal)),
            "complete_signal_bars": int(len(signal)),
            "dropped_incomplete_signal_bars": 0,
            "first_signal_timestamp": str(signal.index.min()) if len(signal) else None,
            "last_signal_timestamp": str(signal.index.max()) if len(signal) else None,
            "source": "binance_public_rest_cached",
            "cache_path": str(resolve_path(ONE_MINUTE_CACHE)),
            "note": "1m signal is sampled on 5m execution timestamps; RSI window remains 24 bars.",
        }
    signal, audit = complete_resample_from_5m(raw_5m, timeframe)
    audit["source"] = "resampled_from_processed_5m"
    return signal, audit


def write_report(output_dir: Path, summary: pd.DataFrame, signal_audit: pd.DataFrame, manifest: dict[str, Any]) -> None:
    primary = summary[summary["scenario"].eq("zero_fee_0p25bps")].sort_values("monthly_return", ascending=False)
    lines = [
        "# Holdout Timeframe Sensitivity 037",
        "",
        "No parameter reselection. Same locked 037 grid/candidate, same 5m execution data, only the RSI signal timeframe changes.",
        "",
        f"- Holdout: `{manifest['holdout_start_utc']}` to `{manifest['holdout_end_utc']}`",
        f"- Primary scenario: `zero_fee_0p25bps`",
        "",
        "## Primary Holdout Ranking",
        markdown_table(
            primary[
                [
                    "signal_timeframe",
                    "monthly_return",
                    "total_return",
                    "max_drawdown",
                    "orders_per_month",
                    "grids_per_month",
                    "net_pnl_per_order",
                    "number_of_grids",
                ]
            ]
        ),
        "",
        "## Signal Audit",
        markdown_table(signal_audit),
        "",
        "## Notes",
        "- 1h is the control used by iteration 037.",
        "- 2h/4h reuse the same RSI window, so they intentionally test timeframe transfer, not a refit.",
        "- Results are holdout-only and should not be used to select new parameters without a fresh train-only validation.",
    ]
    (output_dir / "holdout_timeframe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(output_dir: Path, summary: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = summary[summary["scenario"].eq("zero_fee_0p25bps")].copy()
    preferred_order = ["1m", "5m", *TIMEFRAMES]
    order = [value for value in preferred_order if value in set(summary["signal_timeframe"].astype(str))]
    primary["signal_timeframe"] = pd.Categorical(primary["signal_timeframe"].astype(str), categories=order, ordered=True)
    primary = primary.sort_values("signal_timeframe")
    plt.figure(figsize=(10, 5))
    plt.bar(primary["signal_timeframe"].astype(str), primary["monthly_return"].astype(float) * 100)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(2, color="#2c7fb8", linestyle="--", linewidth=1.2, label="2% monthly reference")
    plt.ylabel("Holdout monthly return (%)")
    plt.xlabel("Signal timeframe")
    plt.title("037 Holdout Sensitivity: Signal Timeframe")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "holdout_monthly_return_by_timeframe.png", dpi=160)
    plt.close()

    pivot = summary.pivot(index="scenario", columns="signal_timeframe", values="monthly_return").reindex(columns=order)
    plt.figure(figsize=(10, 5))
    im = plt.imshow(pivot.astype(float).to_numpy() * 100, aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, label="Monthly return (%)")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.title("037 Holdout Monthly Return by Scenario and Signal Timeframe")
    plt.tight_layout()
    plt.savefig(figures / "holdout_scenario_timeframe_heatmap.png", dpi=160)
    plt.close()


def run(
    config_path: str = DEFAULT_CONFIG,
    timeframes: list[str] | None = None,
    scenario_names: set[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    selected_timeframes = timeframes or TIMEFRAMES
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(OUTPUT_ROOT) / f"holdout_timeframe_sensitivity_037_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "oos_equity").mkdir(parents=True, exist_ok=True)
    (output_dir / "trades").mkdir(parents=True, exist_ok=True)

    raw_5m = load_market_5m()
    market = prepare_market("btcusdt").reindex(raw_5m.index).dropna(subset=["open", "high", "low", "close"])
    holdout_index = evaluation_index_for_window(market, start, end, config)
    base_risk = validate_strategy_config(load_strategy_config())
    row = locked_037_row(config)
    _events, _event_windows, blackout_masks = build_blackout_bundle(market.index, config)
    trend_components = build_trend_escape_components(market, config)
    entry_mask = fundamental_trend_mask(trend_components["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)

    summary_rows: list[dict[str, Any]] = []
    signal_audit_rows: list[dict[str, Any]] = []
    selected_scenarios = [row for row in SCENARIOS if scenario_names is None or row[0] in scenario_names]
    if not selected_scenarios:
        raise ValueError("No scenarios selected")
    scenarios = [scenario_spec(name, fee, slippage) for name, fee, slippage, _penalty in selected_scenarios]
    penalties = {name: penalty for name, _fee, _slippage, penalty in selected_scenarios}
    for timeframe in selected_timeframes:
        signal_frame, audit = signal_frame_for_timeframe(raw_5m, timeframe, holdout_index)
        signal_audit_rows.append(audit)
        for scenario in scenarios:
            maker_penalty = penalties[scenario.name]
            metrics, trades, equity = evaluate_selected(
                market,
                signal_frame,
                base_risk,
                row,
                holdout_index,
                scenario,
                entry_mask,
                maker_penalty=maker_penalty,
            )
            metrics = dict(metrics)
            metrics.update(
                {
                    "signal_timeframe": timeframe,
                    "scenario": scenario.name,
                    "holdout_start_utc": str(holdout_index.min()),
                    "holdout_end_utc": str(holdout_index.max()),
                }
            )
            summary_rows.append(metrics)
            trades.to_csv(output_dir / "trades" / f"{timeframe}_{scenario.name}_trades.csv", index=False)
            pd.DataFrame({"timestamp": equity.index.astype(str), "equity": equity.to_numpy()}).to_csv(
                output_dir / "oos_equity" / f"{timeframe}_{scenario.name}_equity.csv",
                index=False,
            )

    summary = pd.DataFrame(summary_rows)
    signal_audit = pd.DataFrame(signal_audit_rows)
    summary.to_csv(output_dir / "holdout_timeframe_summary.csv", index=False)
    signal_audit.to_csv(output_dir / "signal_timeframe_audit.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "output_dir": str(output_dir),
        "timeframes": selected_timeframes,
        "holdout_start_utc": str(holdout_index.min()),
        "holdout_end_utc": str(holdout_index.max()),
        "candidate_name": row["name"],
        "parameter_reselection": False,
        "custom_window": bool(start or end),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_figures(output_dir, summary)
    write_report(output_dir, summary, signal_audit, manifest)
    return {"output_dir": str(output_dir), "summary": summary.to_dict("records"), "signal_audit": signal_audit.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout-only signal-timeframe sensitivity for locked strategy 037.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES), help="Comma-separated signal timeframes.")
    parser.add_argument("--primary-only", action="store_true", help="Run only zero_fee_0p25bps.")
    parser.add_argument("--start", default=None, help="Optional timezone-aware evaluation window start.")
    parser.add_argument("--end", default=None, help="Optional timezone-aware evaluation window end.")
    args = parser.parse_args()
    result = run(
        args.config,
        [value.strip() for value in args.timeframes.split(",") if value.strip()],
        {"zero_fee_0p25bps"} if args.primary_only else None,
        start=args.start,
        end=args.end,
    )
    print(json.dumps({"output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
