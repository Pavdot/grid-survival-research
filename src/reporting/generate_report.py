from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.reporting.plot_drawdowns import plot_drawdown_curves
from src.reporting.plot_equity import plot_equity_curves
from src.reporting.plot_feature_importance import plot_feature_importance
from src.utils.config_loader import configured_path, load_strategy_config
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _save_hist(series: pd.Series, title: str, output: Path, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    series.dropna().astype(float).hist(ax=ax, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _save_bar(series: pd.Series, title: str, output: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_confusion_matrix(metrics_path: Path, output: Path) -> None:
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = metrics.get("best_model")
    matrix = metrics.get("metrics", {}).get(best, {}).get("test", {}).get("confusion_matrix")
    if not matrix:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _load_backtest_summary(backtests_dir: Path) -> pd.DataFrame:
    path = backtests_dir / "backtest_summary.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0)
    return pd.DataFrame()


def generate_figures() -> list[str]:
    figures_dir = configured_path("figures_dir")
    backtests_dir = configured_path("backtests_dir")
    model_dir = configured_path("model_reports_dir")
    labels_path = configured_path("labels_dir", "grid_labels.parquet")
    predictions_path = model_dir / "grid_survival_predictions.parquet"
    features_path = configured_path("features_dir", "grid_features.parquet")
    generated: list[str] = []

    equity_files = sorted(backtests_dir.glob("*_equity.csv"))
    if equity_files:
        plot_equity_curves(equity_files, figures_dir / "equity_curve_by_strategy.png")
        plot_drawdown_curves(equity_files, figures_dir / "drawdown_curve_by_strategy.png")
        generated.extend(["equity_curve_by_strategy.png", "drawdown_curve_by_strategy.png"])

    trade_files = sorted(backtests_dir.glob("*_trades.csv"))
    if trade_files:
        trades = pd.concat([pd.read_csv(path).assign(strategy=path.stem.replace("_trades", "")) for path in trade_files])
        if not trades.empty and "realized_pnl" in trades:
            _save_hist(trades["realized_pnl"], "Distribution of returns by grid", figures_dir / "grid_return_distribution.png", "Realized PnL")
            generated.append("grid_return_distribution.png")
            if "start_timestamp" in trades:
                trades["start_timestamp"] = pd.to_datetime(trades["start_timestamp"], utc=True)
                _save_bar(
                    trades.groupby(trades["start_timestamp"].dt.hour)["realized_pnl"].sum(),
                    "PnL by UTC hour",
                    figures_dir / "pnl_by_hour_utc.png",
                    "Realized PnL",
                )
                generated.append("pnl_by_hour_utc.png")

    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        _save_hist(
            predictions["grid_survival_score"].dropna(),
            "Survival score distribution",
            figures_dir / "survival_score_distribution.png",
            "Grid survival score",
        )
        generated.append("survival_score_distribution.png")

    if labels_path.exists():
        labels = pd.read_parquet(labels_path)
        _save_hist(
            labels["max_adverse_excursion"],
            "Max adverse excursion distribution",
            figures_dir / "max_adverse_excursion_distribution.png",
            "MAE",
        )
        generated.append("max_adverse_excursion_distribution.png")

    importance_path = model_dir / "feature_importance.csv"
    if importance_path.exists():
        plot_feature_importance(importance_path, figures_dir / "feature_importance.png")
        generated.append("feature_importance.png")

    _plot_confusion_matrix(model_dir / "model_metrics.json", figures_dir / "confusion_matrix.png")
    if (figures_dir / "confusion_matrix.png").exists():
        generated.append("confusion_matrix.png")

    if features_path.exists() and trade_files:
        features = pd.read_parquet(features_path)
        trades = pd.concat([pd.read_csv(path) for path in trade_files])
        if not trades.empty and "start_timestamp" in trades:
            trades["start_timestamp"] = pd.to_datetime(trades["start_timestamp"], utc=True)
            joined = trades.set_index("start_timestamp").join(features[["is_range_regime"]], how="left")
            regime_pnl = joined.groupby("is_range_regime")["realized_pnl"].sum()
            _save_bar(regime_pnl, "PnL by market regime", figures_dir / "pnl_by_regime.png", "Realized PnL")
            generated.append("pnl_by_regime.png")
    return generated


def generate_report() -> Path:
    figures = generate_figures()
    model_dir = configured_path("model_reports_dir")
    backtests_dir = configured_path("backtests_dir")
    report_path = model_dir / "grid_survival_report.md"
    strategy = load_strategy_config()
    summary = _load_backtest_summary(backtests_dir)
    metrics_path = model_dir / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    lines = [
        "# Grid Survival Report",
        "",
        "## 1. Résumé exécutif",
        "MVP research pipeline for a bounded BTCUSDT grid survival filter. The report is generated from local backtest and model artifacts when available.",
        "",
        "## 2. Hypothèse testée",
        "A lightweight bounded grid is viable only in exploitable range regimes and should be disabled before range-to-trend transitions.",
        "",
        "## 3. Données utilisées",
        "BTCUSDT 5m OHLCV is normalized from Binance klines. 15m, 30m, and 1h candles are rebuilt only from closed 5m candles.",
        "",
        "## 4. Paramètres du grid",
        "```yaml",
        json.dumps(strategy, indent=2),
        "```",
        "",
        "## 5. Méthode de labellisation",
        "Each eligible timestamp launches a bounded long grid simulation with fees, slippage, max exposure, max loss, max holding time, regime stops, and volatility shock stops.",
        "",
        "## 6. Features utilisées",
        "Volatility, trend, range, volume, and session features are computed using closed candles only. Higher timeframe features are forward-filled after their own candle close.",
        "",
        "## 7. Méthode de validation",
        "Train, validation, and test splits are chronological with an embargo. Walk-forward fold boundaries are exported for audit.",
        "",
        "## 8. Résultats des modèles",
        f"Best model: `{metrics.get('best_model', 'not available')}`.",
        "",
        "## 9. Résultats des backtests",
        "```text",
        summary.round(6).to_string() if not summary.empty else "Backtest summary not available yet.",
        "```",
        "",
        "## 10. Comparaison des stratégies",
        "Compare constant grid, light progressive grid, regime-filtered grid, ML-filtered grid, and ML kill-switch grid in the backtest summary.",
        "",
        "## 11. Analyse des drawdowns",
        "See `drawdown_curve_by_strategy.png` and max drawdown metrics.",
        "",
        "## 12. Analyse des faux négatifs dangereux",
        "Dangerous false negatives are reported as authorized failed grids with high adverse excursion, high unrealized drawdown, or max-loss stops.",
        "",
        "## 13. Feature importance",
        "See `feature_importance.csv` and `feature_importance.png`.",
        "",
        "## 14. Analyse par régime de marché",
        "See `pnl_by_regime.png` when backtest trades and regime features are available.",
        "",
        "## 15. Analyse par heure/session",
        "See `pnl_by_hour_utc.png`; session features are available in the feature dataset.",
        "",
        "## 16. Conclusion : edge valide ou non",
        "The MVP provides the framework to decide this from out-of-sample model metrics and comparative backtests. A positive edge requires lower realized/unrealized risk and fewer dangerous grids after filtering.",
        "",
        "## 17. Limites",
        "OHLCV-only labels cannot model intrabar queue priority, funding, order book depth, liquidation clusters, or live execution latency.",
        "",
        "## 18. Prochaines étapes",
        "Add funding/open-interest/liquidations, test alternative bounded exits, add stricter out-of-sample backtest windows, and run sensitivity analysis without touching the final test period.",
        "",
        "## Graphiques générés",
    ]
    if figures:
        lines.extend([f"- `reports/figures/{name}`" for name in sorted(set(figures))])
    else:
        lines.append("- No figures generated yet.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote report to %s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown report and figures.")
    parser.parse_args()
    generate_report()


if __name__ == "__main__":
    main()
