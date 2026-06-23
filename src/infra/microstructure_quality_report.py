from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.infra.binance_microstructure_collector import collector_config_from_yaml, validate_collector_config
from src.utils.config_loader import load_yaml, project_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
DEFAULT_CONFIG = "config/infrastructure_microstructure.yaml"
REQUIRED_COLUMNS = {
    "snapshot_time_utc",
    "best_bid",
    "best_ask",
    "spread_bps",
    "bid_depth_5bps_usdt",
    "ask_depth_5bps_usdt",
    "depth_imbalance_5bps",
    "source_latency_ms",
}


def load_ws_depth_files(
    output_dir: Path,
    symbol: str,
    days: int | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    paths = sorted(output_dir.glob(f"{symbol.lower()}_depth_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No WS depth parquet files found in {output_dir}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        if not frame.empty:
            frame["source_file"] = path.name
            frames.append(frame)
    if not frames:
        raise ValueError(f"WS depth parquet files are empty in {output_dir}")
    data = pd.concat(frames, ignore_index=True)
    data["snapshot_time_utc"] = pd.to_datetime(data["snapshot_time_utc"], utc=True)
    data = data.sort_values("snapshot_time_utc").drop_duplicates(["snapshot_time_utc", "last_update_id"], keep="last")
    if days is not None and days > 0:
        cutoff = data["snapshot_time_utc"].max() - pd.Timedelta(days=float(days))
        data = data[data["snapshot_time_utc"].ge(cutoff)].copy()
    if max_rows is not None:
        data = data.tail(int(max_rows)).copy()
    return data.reset_index(drop=True)


def validate_quality_input(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"microstructure frame is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("microstructure frame is empty")
    timestamps = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    if timestamps.isna().any():
        raise ValueError("snapshot_time_utc contains invalid timestamps")


def quality_config(raw: dict[str, Any]) -> dict[str, float]:
    q = raw.get("quality", {})
    return {
        "expected_interval_seconds": float(q.get("expected_interval_seconds", 1)),
        "gap_warn_seconds": float(q.get("gap_warn_seconds", 5)),
        "stale_bad_seconds": float(q.get("stale_bad_seconds", 60)),
        "min_coverage_ratio_healthy": float(q.get("min_coverage_ratio_healthy", 0.95)),
        "min_coverage_ratio_degraded": float(q.get("min_coverage_ratio_degraded", 0.80)),
        "max_invalid_fraction_healthy": float(q.get("max_invalid_fraction_healthy", 0.01)),
        "max_invalid_fraction_degraded": float(q.get("max_invalid_fraction_degraded", 0.05)),
    }


def compute_quality_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    validate_quality_input(frame)
    q = quality_config(config)
    data = frame.sort_values("snapshot_time_utc").copy()
    data["snapshot_time_utc"] = pd.to_datetime(data["snapshot_time_utc"], utc=True)
    diffs = data["snapshot_time_utc"].diff().dt.total_seconds().dropna()
    span_seconds = (
        (data["snapshot_time_utc"].max() - data["snapshot_time_utc"].min()) / pd.Timedelta(seconds=1)
        if len(data) > 1
        else 0.0
    )
    expected = int(np.floor(span_seconds / q["expected_interval_seconds"]) + 1) if span_seconds > 0 else len(data)
    expected = max(expected, 1)
    coverage = min(float(len(data) / expected), 1.0)
    crossed = data["best_ask"].astype(float).le(data["best_bid"].astype(float))
    zero_depth = data["bid_depth_5bps_usdt"].astype(float).le(0) | data["ask_depth_5bps_usdt"].astype(float).le(0)
    invalid = (
        data["spread_bps"].isna()
        | data["spread_bps"].astype(float).lt(0)
        | zero_depth
        | data["source_latency_ms"].isna()
        | crossed
    )
    gap_count = int(diffs.gt(q["gap_warn_seconds"]).sum())
    stale_period_count = int(diffs.gt(q["stale_bad_seconds"]).sum())
    invalid_fraction = float(invalid.mean())
    if (
        coverage >= q["min_coverage_ratio_healthy"]
        and invalid_fraction <= q["max_invalid_fraction_healthy"]
        and stale_period_count == 0
        and int(crossed.sum()) == 0
    ):
        score = "healthy"
    elif coverage >= q["min_coverage_ratio_degraded"] and invalid_fraction <= q["max_invalid_fraction_degraded"]:
        score = "degraded"
    else:
        score = "bad"
    return {
        "quality_score": score,
        "snapshot_count": int(len(data)),
        "expected_snapshot_count": int(expected),
        "coverage_ratio": coverage,
        "collection_start_utc": data["snapshot_time_utc"].min().isoformat(),
        "collection_end_utc": data["snapshot_time_utc"].max().isoformat(),
        "collection_span_hours": float(span_seconds / 3600.0),
        "gap_count": gap_count,
        "max_gap_seconds": float(diffs.max()) if not diffs.empty else 0.0,
        "stale_period_count": stale_period_count,
        "invalid_snapshot_fraction": invalid_fraction,
        "crossed_book_count": int(crossed.sum()),
        "zero_depth_count": int(zero_depth.sum()),
        "spread_bps_p50": float(data["spread_bps"].astype(float).median()),
        "spread_bps_p90": float(data["spread_bps"].astype(float).quantile(0.90)),
        "spread_bps_p99": float(data["spread_bps"].astype(float).quantile(0.99)),
        "bid_depth_5bps_p50": float(data["bid_depth_5bps_usdt"].astype(float).median()),
        "ask_depth_5bps_p50": float(data["ask_depth_5bps_usdt"].astype(float).median()),
        "abs_imbalance_5bps_p90": float(data["depth_imbalance_5bps"].astype(float).abs().quantile(0.90)),
        "source_latency_ms_p90": float(data["source_latency_ms"].astype(float).quantile(0.90)),
    }


def hourly_microstructure(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["snapshot_time_utc"] = pd.to_datetime(data["snapshot_time_utc"], utc=True)
    data["hour_utc"] = data["snapshot_time_utc"].dt.hour
    grouped = data.groupby("hour_utc", sort=True).agg(
        snapshot_count=("snapshot_time_utc", "size"),
        spread_bps_p50=("spread_bps", "median"),
        spread_bps_p90=("spread_bps", lambda value: float(pd.Series(value).quantile(0.90))),
        bid_depth_5bps_p50=("bid_depth_5bps_usdt", "median"),
        ask_depth_5bps_p50=("ask_depth_5bps_usdt", "median"),
        abs_imbalance_5bps_p90=("depth_imbalance_5bps", lambda value: float(pd.Series(value).abs().quantile(0.90))),
        source_latency_ms_p90=("source_latency_ms", lambda value: float(pd.Series(value).quantile(0.90))),
    )
    return grouped.reindex(range(24)).reset_index()


def write_quality_heatmap(hourly: pd.DataFrame, output_dir: Path) -> Path:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "microstructure_quality_heatmap.png"
    metrics = [
        "spread_bps_p50",
        "spread_bps_p90",
        "bid_depth_5bps_p50",
        "ask_depth_5bps_p50",
        "abs_imbalance_5bps_p90",
        "source_latency_ms_p90",
    ]
    data = hourly[metrics].T.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title("BTCUSDT microstructure quality by UTC hour")
    ax.set_xlabel("UTC hour")
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(hour) for hour in range(24)])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_dashboard(output_dir: Path, metrics: dict[str, Any], hourly: pd.DataFrame, heatmap_path: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "microstructure_quality_summary.csv"
    hourly_path = output_dir / "microstructure_quality_by_hour.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    hourly.to_csv(hourly_path, index=False)
    md_path = output_dir / "microstructure_quality_report.md"
    html_path = output_dir / "index.html"
    lines = [
        "# Microstructure Quality Dashboard",
        "",
        f"- quality_score: `{metrics['quality_score']}`",
        f"- snapshots: `{metrics['snapshot_count']}` / expected `{metrics['expected_snapshot_count']}`",
        f"- coverage_ratio: `{metrics['coverage_ratio']:.3f}`",
        f"- invalid_snapshot_fraction: `{metrics['invalid_snapshot_fraction']:.3f}`",
        f"- spread p50/p90/p99 bps: `{metrics['spread_bps_p50']:.3f}` / `{metrics['spread_bps_p90']:.3f}` / `{metrics['spread_bps_p99']:.3f}`",
        f"- depth p50 5bps bid/ask: `{metrics['bid_depth_5bps_p50']:.0f}` / `{metrics['ask_depth_5bps_p50']:.0f}` USDT",
        "",
        f"![Microstructure quality heatmap](figures/{heatmap_path.name})",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grid Survival Microstructure Quality</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; color: #172026; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
    td, th {{ border: 1px solid #d4dbe3; padding: 8px; text-align: left; }}
    .score {{ font-weight: 700; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d4dbe3; }}
  </style>
</head>
<body>
  <h1>Grid Survival Microstructure Quality</h1>
  <p class="score">Score: {metrics['quality_score']}</p>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Snapshots</td><td>{metrics['snapshot_count']} / {metrics['expected_snapshot_count']}</td></tr>
    <tr><td>Coverage ratio</td><td>{metrics['coverage_ratio']:.3f}</td></tr>
    <tr><td>Invalid snapshot fraction</td><td>{metrics['invalid_snapshot_fraction']:.3f}</td></tr>
    <tr><td>Spread p50 / p90 / p99 bps</td><td>{metrics['spread_bps_p50']:.3f} / {metrics['spread_bps_p90']:.3f} / {metrics['spread_bps_p99']:.3f}</td></tr>
    <tr><td>Depth p50 5bps bid / ask</td><td>{metrics['bid_depth_5bps_p50']:.0f} / {metrics['ask_depth_5bps_p50']:.0f} USDT</td></tr>
    <tr><td>Gaps / stale periods</td><td>{metrics['gap_count']} / {metrics['stale_period_count']}</td></tr>
  </table>
  <h2>Hourly Heatmap</h2>
  <img src="figures/{heatmap_path.name}" alt="Microstructure quality heatmap">
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return md_path, html_path


def generate_quality_report(
    config: dict[str, Any],
    days: int | None = None,
    max_rows: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    collector = collector_config_from_yaml(config)
    validate_collector_config(collector)
    output_dir = output_dir or project_path(config.get("quality", {}).get("output_dir", "reports/infra/microstructure_quality"))
    frame = load_ws_depth_files(collector.output_dir, collector.symbol, days=days, max_rows=max_rows)
    metrics = compute_quality_metrics(frame, config)
    hourly = hourly_microstructure(frame)
    heatmap = write_quality_heatmap(hourly, output_dir)
    markdown_path, html_path = write_dashboard(output_dir, metrics, hourly, heatmap)
    payload = {
        "metrics": metrics,
        "output_dir": str(output_dir),
        "markdown_report": str(markdown_path),
        "html_dashboard": str(html_path),
        "heatmap": str(heatmap),
    }
    (output_dir / "microstructure_quality_payload.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BTCUSDT microstructure quality reports.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fail-on-bad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = project_path(args.output_dir) if args.output_dir else None
    payload = generate_quality_report(config, days=args.days, max_rows=args.max_rows, output_dir=output_dir)
    print(json.dumps(payload, indent=2, default=str))
    if args.fail_on_bad and payload["metrics"]["quality_score"] == "bad":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
