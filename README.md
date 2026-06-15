# Grid Survival Research

Research-only pipeline for testing whether a bounded BTCUSDT grid can survive
specific market regimes. This project does not trade live, does not use private
API keys, and does not contain execution code.

The MVP uses OHLCV data only. It normalizes BTCUSDT 5m Binance klines, rebuilds
15m/30m/1h candles from the 5m source, builds leakage-safe features, simulates
bounded grid outcomes, trains grid-survival classifiers, backtests filtered grid
variants, and writes a Markdown research report.

## Quick Start

```powershell
cd C:\Users\Utilisateur\Downloads\src\grid-survival-research
python -m src.data.load_data
python -m src.data.resample_timeframes
python -m src.features.build_features
python -m src.labeling.simulate_grid_labels
python -m src.models.train_grid_survival_model
python -m src.backtesting.backtest_filtered_grid
python -m src.reporting.generate_report
```

For a faster smoke run, pass `--limit 3000` to the feature, label, model, and
backtest commands.

## Research Iterations

ML-first threshold research:

```powershell
python -m src.research.ml_threshold_research
```

Economy-first grid calibration:

```powershell
python -m src.research.economy_first_research
```

Directional grid research:

```powershell
python -m src.research.directional_grid_research
```

Momentum switch research:

```powershell
python -m src.research.momentum_switch_research
```

Monthly target bounded martingale grid research:

```powershell
python -m src.research.monthly_target_martingale_research
```

Walk-forward martingale research:

```powershell
python -m src.research.walk_forward_martingale_research
```

Fundamental-blackout walk-forward martingale research:

```powershell
python -m src.research.fundamental_blackout_martingale_research
python -m src.research.fundamental_blackout_martingale_research --max-folds 2 --max-candidates 10 --exact-top-n 3
```

Fundamental blackout ablation research:

```powershell
python -m src.research.fundamental_blackout_ablation_research --max-candidates 10 --exact-top-n 3
python -m src.research.fundamental_blackout_ablation_research --max-folds 2 --max-candidates 10 --exact-top-n 3
```

All research iterations write local outputs under `reports/research_iterations/`, which
is ignored by Git except for the directory placeholder.

## Outputs

- `data/processed/btcusdt_5m.parquet`
- `data/processed/btcusdt_15m.parquet`
- `data/processed/btcusdt_30m.parquet`
- `data/processed/btcusdt_1h.parquet`
- `data/features/grid_features.parquet`
- `data/labels/grid_labels.parquet`
- `reports/model_reports/*.joblib`, metrics, predictions, and report files
- `reports/backtests/*.csv`
- `reports/figures/*.png`

## Guardrails

- No exponential martingale and no 1/2/4/8/16 sizing.
- Maximum grid loss, maximum daily loss, maximum exposure, and maximum holding
  time are required.
- Fees and slippage are always included.
- Features use closed candles only and never use future data.
- Labels may use future data because they simulate outcomes after entry.
- Temporal validation is mandatory; random train/test splitting is not used.
- ML filters cannot override risk management.

## Testing

```powershell
python -m unittest discover -s tests
```

If `pytest` is installed, the same tests can also be run with:

```powershell
pytest
```
