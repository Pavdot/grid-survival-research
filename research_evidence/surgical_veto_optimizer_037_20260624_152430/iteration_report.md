# Iteration 037 - Surgical Veto Optimizer

Verdict: `surgical veto candidate`

## Combined OOS
| scenario | monthly_return | monthly_p10 | monthly_median | max_drawdown | positive_fold_rate | orders_per_month | grids_per_month | net_pnl_per_order | effective_leverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero_fee_0bps | 0.184116 | 0.0261784 | 0.10007 | -0.203952 | 0.9375 | 147.025 | 63.2914 | 0.00146242 | 36.121 |
| zero_fee_0p25bps | 0.172546 | 0.0154389 | 0.0871165 | -0.205952 | 0.9375 | 147.025 | 63.2914 | 0.00138334 | 36.121 |
| zero_fee_0p5bps | 0.16091 | 0.00178686 | 0.0741651 | -0.207962 | 0.875 | 147.025 | 63.2914 | 0.00130426 | 36.121 |
| zero_fee_1bps | 0.135555 | -0.0407322 | 0.0482694 | -0.212011 | 0.625 | 147.025 | 63.2914 | 0.00113507 | 36.121 |
| zero_fee_maker_miss10 | 0.127555 | -0.025609 | 0.0627511 | -0.212621 | 0.625 | 147.025 | 63.2914 | 0.00104628 | 36.121 |
| realistic_control_fee40_slip2 | -0.146386 | -0.373336 | -0.168295 | -1.23332 | 0.375 | 147.025 | 63.2914 | -0.000496256 | 36.121 |

## Holdout
| scenario | monthly_return | total_return | max_drawdown | orders_per_month | net_pnl_per_order |
| --- | --- | --- | --- | --- | --- |
| zero_fee_0bps | 0.023392 | 0.0707621 | -0.2445 | 163.686 | 0.000146203 |
| zero_fee_0p25bps | 0.0114201 | 0.0341465 | -0.250405 | 163.686 | 7.05507e-05 |
| zero_fee_0p5bps | -0.000835668 | -0.00246895 | -0.256318 | 163.686 | -5.10113e-06 |
| zero_fee_1bps | -0.0262708 | -0.0756996 | -0.275596 | 163.686 | -0.000156404 |
| zero_fee_maker_miss10 | -0.0264589 | -0.0762276 | -0.261511 | 163.686 | -0.000157495 |
| realistic_control_fee40_slip2 | -0.449075 | -0.82843 | -0.863562 | 163.686 | -0.00171163 |

## Notes
- Grid/sizing/candidates remain locked from fundamental_trend_escape_v2.
- Veto policies only block entries or apply stateful post-loss guards selected on train.
- The score requires return/order retention against the baseline train replay.
