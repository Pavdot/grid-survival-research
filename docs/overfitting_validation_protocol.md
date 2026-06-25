# Overfitting Validation Protocol

This project treats walk-forward as necessary but insufficient evidence.

The protocol is informed by `ssrn-4686376.pdf`, "Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment" by Arian, Norouzi Mobarekeh, and Seco.

## Paper Takeaways Applied Here

- Walk-forward is realistic, but it is a single-path test and can understate false discovery risk.
- Combinatorial Purged Cross-Validation (CPCV) is the preferred robustness check because it creates multiple purged backtest paths.
- Probability of Backtest Overfitting (PBO), estimated with Combinatorially Symmetric Cross-Validation (CSCV), must be reported when a strategy was selected from multiple candidates.
- Deflated Sharpe Ratio (DSR) must be reported separately because it adjusts for multiple trials and false discoveries; it is not redundant with PBO.
- Overfitting stability should be assessed through distributions, not only one aggregate walk-forward result.

## Required Gates For Robust Claims

A strategy can no longer be called robust from walk-forward alone.

Minimum evidence:

1. Walk-forward OOS with chronological train/test and embargo.
2. Final holdout untouched by selection.
3. CPCV with purging/embargo and multiple test-group combinations.
4. CSCV/PBO over the candidate universe that was actually searched.
5. DSR or equivalent multiple-testing correction over the same searched candidate universe.
6. Monte Carlo or bootstrap robustness on folds and trades.
7. Execution-cost stress if the strategy depends on low fees/slippage.

## Default Thresholds

The exact thresholds can be tightened per iteration, but a research candidate should fail by default if:

- `PBO > 0.10`
- `DSR statistic < 0`
- `CPCV p25 monthly <= 0`
- any CPCV path ruins equity
- walk-forward is the only validation evidence

For aggressive grid/martingale strategies, the report must also show drawdown and ruin even when drawdown is not the selection objective.

## Implementation

Reusable code lives in:

`src/research/overfitting_validation_protocol.py`

It provides:

- chronological CPCV path construction with observation-level embargo
- CSCV/PBO estimation from a candidate return matrix
- expected max Sharpe and DSR statistic
- a strict evidence decision helper that rejects walk-forward-only claims

## Important Caveat

PBO and DSR are only valid when computed on the candidate universe actually exposed to selection. A selected-fold summary alone is not enough. Future research runners should persist a candidate-by-fold or candidate-by-period OOS return matrix so PBO/DSR can be computed without proxy data.
