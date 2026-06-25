from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.fundamentals.event_blackout import build_blackout_bundle
from src.labeling.grid_risk import validate_strategy_config
from src.regimes.trend_escape import build_trend_escape_components
from src.research.economy_first_research import prepare_market
from src.research.fundamental_blackout_martingale_research import markdown_table
from src.research.holdout_timeframe_sensitivity_037 import (
    complete_resample_from_5m,
    fetch_binance_1m_klines,
    resolve_path,
)
from src.research.range_break_classifier_martingale_research import fundamental_trend_mask
from src.research.surgical_veto_optimizer_037 import DEFAULT_CONFIG as CONFIG_037
from src.research.walk_forward_martingale_research import stitch_oos_equity
from src.research.zero_fee_p10_optimizer_035 import evaluate_selected, fold_row, scenario_spec, summarize_scope
from src.utils.config_loader import load_strategy_config, load_yaml


SOURCE_037_DIR = "research_evidence/surgical_veto_optimizer_037_20260624_152430"
OUTPUT_ROOT = "research_evidence"
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h"]
ONE_MINUTE_FULL_CACHE = "data/live/btcusdt_1m_037_oos_full.parquet"


def _timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def selected_rows_by_fold(source_dir: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    matrix = pd.read_csv(source_dir / "selection_matrix.csv")
    if matrix.empty:
        raise ValueError("037 selection_matrix.csv is empty")
    rows: dict[str, dict[str, Any]] = {}
    for fold_id, group in matrix.groupby(matrix["fold_id"].astype(str), sort=False):
        ordered = group.sort_values(
            ["candidate_selection_rank", "constraint_rank", "candidate_selection_score", "monthly_return", "monthly_p10_chunks"],
            ascending=[True, True, False, False, False],
        )
        row = ordered.iloc[0].to_dict()
        row["candidate_uid"] = str(row.get("candidate_uid", "veto_none"))
        row["name"] = str(row["name"])
        row.setdefault("blackout_hours", int(config["fundamental_blackout"]["pre_event_hours"]))
        row.setdefault("min_severity", int(config["fundamental_blackout"]["min_severity"]))
        row.setdefault("trend_propagation_bars", int(config["trend_escape"]["propagation_bars"]))
        row.setdefault("breakout_atr_buffer", float(config["trend_escape"]["breakout_atr_buffer"]))
        row.setdefault("min_range_expansion_ratio", float(config["trend_escape"]["min_range_expansion_ratio"]))
        row.setdefault("max_grids_per_month", 999.0)
        row.setdefault("pause_after_forced_loss_hours", 0.0)
        row.setdefault("rolling_7d_loss_threshold", -999.0)
        rows[str(fold_id)] = row
    return rows


def fold_windows(source_dir: Path, scenario: str) -> pd.DataFrame:
    folds = pd.read_csv(source_dir / "fold_metrics.csv")
    folds = folds[folds["scenario"].eq(scenario)].copy()
    if folds.empty:
        raise ValueError(f"No 037 fold metrics found for scenario {scenario}")
    folds["fold_key"] = folds["fold_id"].astype(str)
    return folds[
        [
            "fold_key",
            "fold_id",
            "is_holdout",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "monthly_return",
            "total_return",
            "selected_name",
        ]
    ].copy()


def signal_frame_for_full_backtest(raw_5m: pd.DataFrame, timeframe: str, evaluation_index: pd.Index) -> tuple[pd.DataFrame, dict[str, Any]]:
    if timeframe == "1m":
        start = pd.Timestamp(evaluation_index.min()) - pd.Timedelta(days=2)
        end = pd.Timestamp(evaluation_index.max())
        cache = resolve_path(ONE_MINUTE_FULL_CACHE)
        signal = fetch_binance_1m_klines(start, end, cache)
        return signal, {
            "timeframe": "1m",
            "source": "binance_public_rest_cached",
            "cache_path": str(cache),
            "signal_bars": int(len(signal)),
            "first_signal_timestamp": str(signal.index.min()) if len(signal) else None,
            "last_signal_timestamp": str(signal.index.max()) if len(signal) else None,
            "note": "RSI window remains 24 signal bars; 1m means RSI(24 minutes).",
        }
    signal, audit = complete_resample_from_5m(raw_5m, timeframe)
    audit["signal_bars"] = int(len(signal))
    audit["source"] = "resampled_from_processed_5m"
    return signal, audit


def write_figures(output_dir: Path, summary: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = summary[summary["scope"].eq("combined_oos")].copy()
    holdout = summary[summary["scope"].eq("holdout_90d")].copy()
    order = [tf for tf in TIMEFRAMES if tf in set(summary["signal_timeframe"])]
    for frame in [combined, holdout]:
        frame["signal_timeframe"] = pd.Categorical(frame["signal_timeframe"].astype(str), categories=order, ordered=True)
        frame.sort_values("signal_timeframe", inplace=True)

    plt.figure(figsize=(11, 5))
    x = range(len(order))
    combined_values = [float(combined[combined["signal_timeframe"].astype(str).eq(tf)]["monthly_return"].iloc[0]) * 100 for tf in order]
    holdout_values = [float(holdout[holdout["signal_timeframe"].astype(str).eq(tf)]["monthly_return"].iloc[0]) * 100 for tf in order]
    plt.bar([value - 0.18 for value in x], combined_values, width=0.36, label="Combined OOS")
    plt.bar([value + 0.18 for value in x], holdout_values, width=0.36, label="Holdout 90d")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(2, color="#2c7fb8", linestyle="--", linewidth=1.1, label="2% monthly")
    plt.xticks(list(x), order)
    plt.ylabel("Monthly return (%)")
    plt.title("037 Backtest: Signal Timeframe Sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "combined_vs_holdout_monthly_by_timeframe.png", dpi=160)
    plt.close()

    fold_rows = summary[summary["scope"].eq("wf_pre_holdout")].copy()
    if not fold_rows.empty:
        pass


def write_report(output_dir: Path, summary: pd.DataFrame, fold_metrics: pd.DataFrame, signal_audit: pd.DataFrame, manifest: dict[str, Any]) -> None:
    combined = summary[summary["scope"].eq("combined_oos")].sort_values("monthly_return", ascending=False)
    holdout = summary[summary["scope"].eq("holdout_90d")].sort_values("monthly_return", ascending=False)
    lines = [
        "# Iteration 037 Timeframe Backtest",
        "",
        "Replay-only diagnostic backtest. No parameter reselection, no CPCV/MC/PBO/DSR.",
        "",
        f"- Source 037: `{manifest['source_037_dir']}`",
        f"- Scenario: `{manifest['scenario']}`",
        f"- Folds replayed: `{manifest['fold_count']}` including holdout",
        "",
        "## Combined OOS",
        markdown_table(
            combined[
                [
                    "signal_timeframe",
                    "monthly_return",
                    "monthly_p10",
                    "monthly_median",
                    "total_return",
                    "max_drawdown",
                    "positive_fold_rate",
                    "orders_per_month",
                    "grids_per_month",
                    "net_pnl_per_order",
                ]
            ]
        ),
        "",
        "## Holdout 90d",
        markdown_table(
            holdout[
                [
                    "signal_timeframe",
                    "monthly_return",
                    "total_return",
                    "max_drawdown",
                    "orders_per_month",
                    "grids_per_month",
                    "net_pnl_per_order",
                ]
            ]
        ),
        "",
        "## Signal Audit",
        markdown_table(signal_audit),
        "",
        "## Notes",
        "- 1h is the control for iteration 037. If it does not match 037, the replay is invalid.",
        "- 1m uses Binance public 1m klines and samples RSI signals on 5m execution timestamps.",
        "- The RSI window is not rescaled; RSI(24) means 24 bars of the selected signal timeframe.",
    ]
    (output_dir / "timeframe_backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str = CONFIG_037,
    source_dir: str = SOURCE_037_DIR,
    timeframes: list[str] | None = None,
    scenario_name: str = "zero_fee_0p25bps",
    timestamp_override: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    selected_timeframes = timeframes or TIMEFRAMES
    source = resolve_path(source_dir)
    stamp = timestamp_override or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(OUTPUT_ROOT) / f"timeframe_backtest_037_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "oos_equity").mkdir(parents=True, exist_ok=True)
    (output_dir / "trades").mkdir(parents=True, exist_ok=True)

    raw_5m = pd.read_parquet(resolve_path("data/processed/btcusdt_5m.parquet"))
    raw_5m.index = pd.to_datetime(raw_5m.index, utc=True)
    market = prepare_market("btcusdt").reindex(raw_5m.index).dropna(subset=["open", "high", "low", "close"])
    selected_rows = selected_rows_by_fold(source, config)
    windows = fold_windows(source, scenario_name)
    evaluation_index = pd.Index([])
    for row in windows.itertuples(index=False):
        start = _timestamp(row.test_start)
        end = _timestamp(row.test_end)
        evaluation_index = evaluation_index.union(market.index[(market.index >= start) & (market.index <= end)])
    if evaluation_index.empty:
        raise ValueError("No evaluation index could be built from 037 folds")

    _events, _event_windows, blackout_masks = build_blackout_bundle(market.index, config)
    trend_components = build_trend_escape_components(market, config)
    entry_mask = fundamental_trend_mask(trend_components["trend_escape"].astype(bool), blackout_masks).reindex(market.index).fillna(False).astype(bool)
    base_risk = validate_strategy_config(load_strategy_config())
    scenario = scenario_spec(scenario_name, 0.0, 0.25)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    signal_audit_rows: list[dict[str, Any]] = []
    for timeframe in selected_timeframes:
        signal_frame, audit = signal_frame_for_full_backtest(raw_5m, timeframe, evaluation_index)
        signal_audit_rows.append(audit)
        trades_by_fold: list[pd.DataFrame] = []
        wf_equities: list[tuple[int, pd.Series]] = []
        all_equities: list[tuple[int, pd.Series]] = []
        holdout_equity = pd.Series(dtype=float)
        for window in windows.itertuples(index=False):
            fold_key = str(window.fold_key)
            if fold_key not in selected_rows:
                raise ValueError(f"Missing selected 037 row for fold {fold_key}")
            selected = dict(selected_rows[fold_key])
            train_index = market.index[(market.index >= _timestamp(window.train_start)) & (market.index <= _timestamp(window.train_end))]
            test_index = market.index[(market.index >= _timestamp(window.test_start)) & (market.index <= _timestamp(window.test_end))]
            metrics, trades, equity = evaluate_selected(market, signal_frame, base_risk, selected, test_index, scenario, entry_mask)
            metrics["signal_timeframe"] = timeframe
            fold_id: int | str = "holdout_90d" if bool(window.is_holdout) else int(window.fold_id)
            out = fold_row(fold_id, scenario.name, selected, {}, metrics, train_index, test_index)
            out["signal_timeframe"] = timeframe
            fold_rows.append(out)
            scoped_trades = trades.copy()
            scoped_trades.insert(0, "signal_timeframe", timeframe)
            scoped_trades.insert(0, "fold_id", fold_id)
            trades_by_fold.append(scoped_trades)
            stitched_id = 999999 if bool(window.is_holdout) else int(window.fold_id)
            all_equities.append((stitched_id, equity))
            if bool(window.is_holdout):
                holdout_equity = equity
            else:
                wf_equities.append((int(window.fold_id), equity))

        frame = pd.DataFrame([row for row in fold_rows if row["signal_timeframe"] == timeframe])
        trades_frame = pd.concat(trades_by_fold, ignore_index=True) if trades_by_fold else pd.DataFrame()
        trades_frame.to_csv(output_dir / "trades" / f"{timeframe}_{scenario.name}_trades.csv", index=False)
        wf_frame = frame[~frame["is_holdout"].astype(bool)].copy()
        holdout_frame = frame[frame["is_holdout"].astype(bool)].copy()
        wf_trades = trades_frame[~trades_frame["fold_id"].astype(str).eq("holdout_90d")].copy() if not trades_frame.empty else pd.DataFrame()
        holdout_trades = trades_frame[trades_frame["fold_id"].astype(str).eq("holdout_90d")].copy() if not trades_frame.empty else pd.DataFrame()
        wf_equity = stitch_oos_equity(wf_equities)["equity"] if wf_equities else pd.Series(dtype=float)
        all_equity = stitch_oos_equity(all_equities)["equity"] if all_equities else pd.Series(dtype=float)
        for scope, scope_frame, scope_equity, scope_trades in [
            ("wf_pre_holdout", wf_frame, wf_equity, wf_trades),
            ("holdout_90d", holdout_frame, holdout_equity, holdout_trades),
            ("combined_oos", frame, all_equity, trades_frame),
        ]:
            row = summarize_scope(scenario.name, scope_frame, scope_equity, scope_trades, scope)
            row["signal_timeframe"] = timeframe
            summary_rows.append(row)
            if not scope_equity.empty:
                pd.DataFrame({"timestamp": scope_equity.index.astype(str), "equity": scope_equity.to_numpy()}).to_csv(
                    output_dir / "oos_equity" / f"{timeframe}_{scope}_equity.csv",
                    index=False,
                )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows)
    signal_audit = pd.DataFrame(signal_audit_rows)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "timeframe_summary.csv", index=False)
    signal_audit.to_csv(output_dir / "signal_timeframe_audit.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(resolve_path(config_path)),
        "source_037_dir": str(source),
        "scenario": scenario_name,
        "timeframes": selected_timeframes,
        "fold_count": int(len(windows)),
        "parameter_reselection": False,
        "anti_overfit_validation": False,
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_figures(output_dir, summary)
    write_report(output_dir, summary, fold_metrics, signal_audit, manifest)
    return {"output_dir": str(output_dir), "summary": summary.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay iteration 037 selected folds with alternate signal timeframes.")
    parser.add_argument("--config", default=CONFIG_037)
    parser.add_argument("--source-dir", default=SOURCE_037_DIR)
    parser.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        source_dir=args.source_dir,
        timeframes=[value.strip() for value in args.timeframes.split(",") if value.strip()],
    )
    print(json.dumps({"output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
