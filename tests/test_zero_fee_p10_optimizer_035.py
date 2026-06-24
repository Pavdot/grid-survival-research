from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.zero_fee_p10_optimizer_035 import (
    candidate_base_size,
    cpcv_process_summary,
    final_verdict,
    forced_loss,
    generate_candidate_universe,
    internal_train_selection_split,
    monte_carlo_summary,
    selection_constraints,
    sizing_sum,
)


def _config() -> dict:
    return {
        "primary_scenario": {"name": "zero_fee_0p25bps", "fee_rate": 0.0, "slippage_bps": 0.25},
        "candidate_search": {
            "random_seed": 35,
            "max_candidates": 10,
            "base_size_safety": 0.95,
            "cooldown_hours": [6],
            "rsi_threshold_pairs": [[35, 65]],
            "max_grids_per_month": [8],
            "blackout_hours": [24],
            "min_severity": [4],
            "trend_propagation_bars": [36],
            "breakout_atr_buffer": [0.25],
            "min_range_expansion_ratio": [1.3],
            "spacing_atr_multipliers": [4.5],
            "take_profit_spacing_multipliers": [4],
            "max_levels": [3],
            "progression_multipliers": [1.25],
            "max_notional_caps": [10],
            "max_holding_hours": [6],
            "pause_after_forced_loss_hours": [12],
            "rolling_7d_loss_thresholds": [-0.03],
        },
        "selection": {
            "min_grids_per_month": 8,
            "max_orders_per_month": 120,
            "max_drawdown": -0.35,
            "require_net_pnl_per_order_positive": True,
        },
        "validation": {
            "cpcv_blocks": 5,
            "cpcv_test_blocks": 2,
            "cpcv_purge_adjacent_blocks": True,
            "monte_carlo_iterations": 200,
            "monte_carlo_seed": 35,
            "robust_monthly_p10_min": 0.0,
            "holdout_min_total_return": 0.0,
            "cpcv_p25_min": 0.0,
            "mc_p05_min": 0.0,
            "positive_probability_min": 0.95,
            "ruin_probability_max": 0.0,
        },
    }


def _base_row() -> dict:
    return {
        "name": "base",
        "side_mode": "mean_reversion_dual",
        "entry_mode": "hourly_extreme",
        "rsi_window": 24,
        "rsi_low": 40,
        "rsi_high": 60,
        "spacing_atr_multiplier": 3,
        "take_profit_spacing_multiplier": 2,
        "max_levels": 5,
        "base_position_size_pct": 2.5,
        "progression_multiplier": 1.55,
        "max_total_exposure_pct": 40,
        "fee_rate": 0.0001,
        "slippage_bps": 0,
        "max_grid_loss_pct": 0.35,
        "max_holding_hours": 6,
        "stop_on_regime_break": False,
        "stop_on_volatility_shock": True,
    }


def test_internal_train_selection_split_is_chronological_without_overlap() -> None:
    index = pd.date_range("2026-01-01", periods=180 * 24 * 12, freq="5min", tz="UTC")
    search, selection = internal_train_selection_split(index, search_days=120, selection_days=60)
    assert not search.empty
    assert not selection.empty
    assert search.max() < selection.min()


def test_candidate_base_size_respects_notional_cap_with_safety() -> None:
    levels = 3
    progression = 1.25
    cap = 10.0
    base = candidate_base_size(cap, levels, progression, 0.95)
    assert np.isclose(base * sizing_sum(levels, progression), 9.5)


def test_generate_candidate_universe_uses_auto_base_size_and_fixed_fee_target() -> None:
    rows = generate_candidate_universe([_base_row()], _config(), 1)
    assert len(rows) == 1
    for row in rows:
        assert row["fee_rate"] == 0.0
        assert row["slippage_bps"] == 0.25
        assert row["base_position_size_pct"] * sizing_sum(row["max_levels"], row["progression_multiplier"]) <= row["max_total_exposure_pct"]


def test_selection_constraints_reject_bad_economics() -> None:
    passed, reasons = selection_constraints(
        {
            "equity_ruined": False,
            "max_drawdown": -0.2,
            "grids_per_month": 9,
            "orders_per_month": 121,
            "net_pnl_per_order": -0.001,
        },
        _config(),
    )
    assert not passed
    assert "max_orders" in reasons
    assert "net_pnl_per_order" in reasons


def test_forced_loss_uses_only_forced_negative_exits() -> None:
    row = pd.Series({"realized_pnl": -0.02, "stopped_by_max_loss": 1})
    assert forced_loss(row)
    row = pd.Series({"realized_pnl": 0.02, "stopped_by_max_loss": 1})
    assert not forced_loss(row)
    row = pd.Series({"realized_pnl": -0.02, "stopped_by_max_loss": 0})
    assert not forced_loss(row)


def test_cpcv_process_summary_builds_purged_block_rows() -> None:
    folds = pd.DataFrame(
        {
            "fold_id": list(range(1, 16)),
            "is_holdout": [False] * 15,
            "monthly_return": np.linspace(-0.01, 0.10, 15),
            "equity_ruined": [False] * 15,
        }
    )
    out = cpcv_process_summary(folds, _config())
    assert len(out) == 10
    assert {"monthly_p25", "positive_rate", "ruin"}.issubset(out.columns)


def test_monte_carlo_summary_reports_fold_and_trade_bootstraps() -> None:
    folds = pd.DataFrame(
        {
            "fold_id": list(range(1, 6)),
            "is_holdout": [False] * 5,
            "monthly_return": [0.01, 0.02, 0.03, 0.04, 0.05],
            "equity_ruined": [False] * 5,
        }
    )
    trades = pd.DataFrame({"realized_pnl": [0.01, -0.002, 0.004, 0.006]})
    out = monte_carlo_summary(folds, trades, _config())
    assert set(out["method"]) == {"fold_bootstrap", "fold_block_bootstrap", "trade_bootstrap"}
    assert out["positive_probability"].between(0, 1).all()


def test_final_verdict_requires_holdout_cpcv_and_mc() -> None:
    summary = pd.DataFrame(
        [
            {
                "scope": "combined_oos",
                "scenario": "zero_fee_0p25bps",
                "monthly_return": 0.05,
                "monthly_p10": 0.01,
                "equity_ruined": False,
                "total_return": 0.4,
            },
            {
                "scope": "holdout_90d",
                "scenario": "zero_fee_0p25bps",
                "monthly_return": 0.02,
                "monthly_p10": 0.0,
                "equity_ruined": False,
                "total_return": 0.05,
            },
        ]
    )
    cpcv = pd.DataFrame({"monthly_p25": [0.01], "ruin": [False]})
    mc = pd.DataFrame({"monthly_p05": [0.01], "positive_probability": [0.99], "ruin_probability": [0.0]})
    assert final_verdict(summary, cpcv, mc, _config()) == "zero-fee robust candidate"
    summary.loc[summary["scope"].eq("holdout_90d"), "total_return"] = -0.01
    assert final_verdict(summary, cpcv, mc, _config()) == "zero-fee fragile but improved"
