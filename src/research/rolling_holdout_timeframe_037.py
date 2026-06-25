from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market
from src.research.holdout_timeframe_sensitivity_037 import (
    locked_037_row,
    resolve_path,
    signal_frame_for_timeframe,
)
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.research.surgical_veto_optimizer_037 import DEFAULT_CONFIG as CONFIG_037
from src.research.zero_fee_p10_optimizer_035 import evaluate_selected, scenario_spec
from src.utils.config_loader import load_strategy_config, load_yaml


DEFAULT_CONFIG = "config/shadow_live_037.yaml"
SOURCE_037_DIR = "research_evidence/surgical_veto_optimizer_037_20260624_152430"
OUTPUT_ROOT = "research_evidence"
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h"]


def load_037_oos_bounds(source_dir: str | Path = SOURCE_037_DIR) -> tuple[pd.Timestamp, pd.Timestamp]:
    source = resolve_path(source_dir)
    folds = pd.read_csv(source / "fold_metrics.csv")
    primary = folds[folds["scenario"].eq("zero_fee_0p25bps")].copy()
    if primary.empty:
        raise ValueError("No zero_fee_0p25bps folds found for 037")
    start = pd.to_datetime(primary["test_start"], utc=True).min()
    end = pd.to_datetime(primary["test_end"], utc=True).max()
    return pd.Timestamp(start), pd.Timestamp(end)


def rolling_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    window_days: float = 90.0,
    count: int = 10,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    width = pd.Timedelta(days=float(window_days))
    latest_start = end - width
    if latest_start <= start:
        return [(start, end)]
    if count <= 1:
        return [(latest_start, end)]
    starts_ns = np.linspace(start.value, latest_start.value, int(count))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    seen: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    for value in starts_ns:
        window_start = pd.Timestamp(int(value), tz="UTC").round("5min")
        window_end = window_start + width
        if window_end > end:
            window_end = end
            window_start = end - width
        key = (window_start, window_end)
        if key not in seen:
            windows.append(key)
            seen.add(key)
    return windows


def summarize_by_timeframe(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for timeframe, group in results.groupby("signal_timeframe", sort=False):
        returns = group["monthly_return"].astype(float)
        rows.append(
            {
                "signal_timeframe": timeframe,
                "window_count": int(len(group)),
                "mean_monthly": float(returns.mean()),
                "median_monthly": float(returns.median()),
                "p10_monthly": float(returns.quantile(0.10)),
                "min_monthly": float(returns.min()),
                "max_monthly": float(returns.max()),
                "positive_window_rate": float(returns.gt(0).mean()),
                "above_2pct_rate": float(returns.ge(0.02).mean()),
                "above_10pct_rate": float(returns.ge(0.10).mean()),
                "worst_max_drawdown": float(group["max_drawdown"].astype(float).min()),
                "mean_orders_per_month": float(group["orders_per_month"].astype(float).mean()),
                "mean_grids_per_month": float(group["grids_per_month"].astype(float).mean()),
                "mean_net_pnl_per_order": float(group["net_pnl_per_order"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["median_monthly", "p10_monthly"], ascending=[False, False])


def write_figures(output_dir: Path, results: pd.DataFrame, aggregate: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    pivot = results.pivot(index="window_id", columns="signal_timeframe", values="monthly_return").reindex(columns=TIMEFRAMES)

    plt.figure(figsize=(11, 6))
    image = plt.imshow(pivot.astype(float).to_numpy() * 100, aspect="auto", cmap="RdYlGn")
    plt.colorbar(image, label="Monthly return (%)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    labels = [f"W{int(idx):02d}" for idx in pivot.index]
    plt.yticks(range(len(labels)), labels)
    plt.title("037 Rolling 90d Pseudo-Holdouts by Signal Timeframe")
    plt.tight_layout()
    plt.savefig(figures / "rolling_holdout_timeframe_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    ordered = aggregate.set_index("signal_timeframe").reindex([tf for tf in TIMEFRAMES if tf in set(aggregate["signal_timeframe"])])
    x = np.arange(len(ordered))
    plt.bar(x - 0.2, ordered["median_monthly"].astype(float) * 100, width=0.4, label="Median")
    plt.bar(x + 0.2, ordered["p10_monthly"].astype(float) * 100, width=0.4, label="P10")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(2, color="#2c7fb8", linestyle="--", linewidth=1.1, label="2% monthly")
    plt.xticks(x, ordered.index)
    plt.ylabel("Monthly return (%)")
    plt.title("037 Rolling Holdout Summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "rolling_holdout_median_p10_by_timeframe.png", dpi=160)
    plt.close()


def write_report(output_dir: Path, results: pd.DataFrame, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> None:
    from src.research.fundamental_blackout_martingale_research import markdown_table

    lines = [
        "# 037 Rolling Holdout Timeframe Diagnostic",
        "",
        "Diagnostic only: locked 037 candidate, no parameter reselection, no CPCV/MC/PBO/DSR.",
        "",
        f"- Window count: `{manifest['window_count']}`",
        f"- Window days: `{manifest['window_days']}`",
        f"- Scenario: `{manifest['scenario']}`",
        "",
        "## Aggregate By Timeframe",
        markdown_table(
            aggregate[
                [
                    "signal_timeframe",
                    "median_monthly",
                    "p10_monthly",
                    "mean_monthly",
                    "positive_window_rate",
                    "above_2pct_rate",
                    "above_10pct_rate",
                    "worst_max_drawdown",
                    "mean_orders_per_month",
                ]
            ]
        ),
        "",
        "## Window Results",
        markdown_table(
            results[
                [
                    "window_id",
                    "window_start_utc",
                    "window_end_utc",
                    "signal_timeframe",
                    "monthly_return",
                    "total_return",
                    "max_drawdown",
                    "orders_per_month",
                    "grids_per_month",
                ]
            ].sort_values(["window_id", "signal_timeframe"])
        ),
    ]
    (output_dir / "rolling_holdout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str = DEFAULT_CONFIG,
    source_dir: str = SOURCE_037_DIR,
    timeframes: list[str] | None = None,
    window_days: float = 90.0,
    count: int = 10,
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    selected_timeframes = timeframes or TIMEFRAMES
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(OUTPUT_ROOT) / f"rolling_holdout_timeframe_037_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    raw_5m = pd.read_parquet(resolve_path("data/processed/btcusdt_5m.parquet"))
    raw_5m.index = pd.to_datetime(raw_5m.index, utc=True)
    market = prepare_market("btcusdt").reindex(raw_5m.index).dropna(subset=["open", "high", "low", "close"])
    oos_start, oos_end = load_037_oos_bounds(source_dir)
    windows = rolling_windows(oos_start, oos_end, window_days=window_days, count=count)
    evaluation_index = pd.Index([])
    for start, end in windows:
        evaluation_index = evaluation_index.union(market.index[(market.index >= start) & (market.index <= end)])
    if evaluation_index.empty:
        raise ValueError("rolling holdout evaluation index is empty")

    signal_frames: dict[str, pd.DataFrame] = {}
    signal_audits: list[dict[str, Any]] = []
    for timeframe in selected_timeframes:
        signal_frame, audit = signal_frame_for_timeframe(raw_5m, timeframe, evaluation_index)
        signal_frames[timeframe] = signal_frame
        signal_audits.append(audit)

    _events, _event_windows, blackout_masks = build_blackout_bundle(market.index, config)
    trend_components = build_trend_escape_components(market, config)
    entry_mask = fundamental_trend_mask(trend_components["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)
    row = locked_037_row(config)
    base_risk = validate_strategy_config(load_strategy_config())
    scenario = scenario_spec("zero_fee_0p25bps", 0.0, 0.25)

    results: list[dict[str, Any]] = []
    for window_id, (start, end) in enumerate(windows, start=1):
        split_index = market.index[(market.index >= start) & (market.index <= end)]
        for timeframe in selected_timeframes:
            metrics, _trades, _equity = evaluate_selected(
                market,
                signal_frames[timeframe],
                base_risk,
                row,
                split_index,
                scenario,
                entry_mask,
            )
            metrics.update(
                {
                    "window_id": int(window_id),
                    "window_start_utc": str(start),
                    "window_end_utc": str(end),
                    "signal_timeframe": timeframe,
                }
            )
            results.append(metrics)

    results_frame = pd.DataFrame(results)
    aggregate = summarize_by_timeframe(results_frame)
    signal_audit = pd.DataFrame(signal_audits)
    results_frame.to_csv(output_dir / "rolling_holdout_results.csv", index=False)
    aggregate.to_csv(output_dir / "rolling_holdout_timeframe_aggregate.csv", index=False)
    signal_audit.to_csv(output_dir / "signal_timeframe_audit.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "source_037_dir": str(resolve_path(source_dir)),
        "output_dir": str(output_dir),
        "timeframes": selected_timeframes,
        "window_days": float(window_days),
        "window_count": int(len(windows)),
        "scenario": scenario.name,
        "parameter_reselection": False,
        "anti_overfit_validation": False,
        "windows": [{"window_id": idx + 1, "start": str(start), "end": str(end)} for idx, (start, end) in enumerate(windows)],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_figures(output_dir, results_frame, aggregate)
    write_report(output_dir, results_frame, aggregate, manifest)
    return {"output_dir": str(output_dir), "results": results_frame.to_dict("records"), "aggregate": aggregate.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple 90d pseudo-holdouts for locked 037 timeframes.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", default=SOURCE_037_DIR)
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    parser.add_argument("--window-days", type=float, default=90.0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        source_dir=args.source_dir,
        timeframes=[value.strip() for value in args.timeframes.split(",") if value.strip()],
        window_days=args.window_days,
        count=args.count,
    )
    print(json.dumps({"output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
