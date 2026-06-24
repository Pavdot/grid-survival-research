from __future__ import annotations

import pandas as pd

from src.research.surgical_veto_optimizer_037 import (
    build_veto_mask,
    candidate_selection_score,
    candidate_with_veto,
    generate_veto_policies,
    veto_constraint_audit,
    veto_score,
)


def _config() -> dict:
    return {
        "veto_search": {
            "max_policies": 10,
            "range_expansion_thresholds": [0, 2.0],
            "realized_vol_thresholds": [0],
            "atr_percentile_thresholds": [0],
            "use_breakout_risk": [False, True],
            "use_regime_disallow": [False],
            "max_grids_per_month": [90],
            "pause_after_forced_loss_hours": [0],
            "rolling_7d_loss_thresholds": [-999],
            "min_train_monthly_return": 0.10,
            "min_train_monthly_p10": -0.02,
            "min_baseline_return_retention": 0.75,
            "min_baseline_order_retention": 0.70,
            "min_p10_improvement": -0.02,
            "max_drawdown": -0.40,
        }
    }


def test_generate_veto_policies_includes_none_first() -> None:
    policies = generate_veto_policies(_config())
    assert policies[0]["veto_uid"] == "veto_none"
    assert policies[0]["max_grids_per_month"] == 999.0
    assert any(policy["use_breakout_risk"] for policy in policies)


def test_build_veto_mask_combines_observable_features() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    market = pd.DataFrame(
        {
            "range_expansion_ratio": [1.0, 2.5, 1.0],
            "realized_volatility_ratio": [1.0, 1.0, 3.0],
            "breakout_risk": [0, 0, 1],
            "regime_allows_grid": [1, 1, 1],
            "atr_5m": [10, 11, 12],
        },
        index=index,
    )
    policy = {
        "range_expansion_threshold": 2.0,
        "realized_vol_threshold": 2.0,
        "atr_percentile_threshold": 0.0,
        "use_breakout_risk": True,
        "use_regime_disallow": False,
    }
    assert build_veto_mask(market, policy).tolist() == [False, True, True]


def test_candidate_with_veto_preserves_locked_grid_and_adds_state_fields() -> None:
    locked = {"name": "locked", "take_profit_spacing_multiplier": 2, "max_levels": 5}
    policy = {
        "veto_uid": "v",
        "veto_name": "veto",
        "max_grids_per_month": 90,
        "pause_after_forced_loss_hours": 6,
        "rolling_7d_loss_threshold": -0.08,
    }
    out = candidate_with_veto(locked, policy)
    assert out["take_profit_spacing_multiplier"] == 2
    assert out["max_levels"] == 5
    assert out["max_grids_per_month"] == 90
    assert out["pause_after_forced_loss_hours"] == 6
    assert out["blackout_hours"] == 24
    assert out["min_severity"] == 4
    assert out["trend_propagation_bars"] == 36
    assert out["breakout_atr_buffer"] == 0.25
    assert out["min_range_expansion_ratio"] == 1.3


def test_candidate_selection_score_penalizes_turnover_and_leverage() -> None:
    base = {
        "monthly_return": 0.15,
        "monthly_p10_chunks": 0.03,
        "monthly_median_chunks": 0.10,
        "net_pnl_per_order": 0.001,
        "orders_per_month": 80,
        "effective_leverage": 10,
        "max_drawdown": -0.10,
    }
    expensive = {**base, "orders_per_month": 180, "effective_leverage": 30}
    assert candidate_selection_score(base) > candidate_selection_score(expensive)


def test_veto_constraint_requires_return_and_order_retention() -> None:
    baseline = {"monthly_return": 0.20, "orders_per_month": 100, "monthly_p10_chunks": -0.05}
    metrics = {
        "monthly_return": 0.12,
        "orders_per_month": 60,
        "monthly_p10_chunks": -0.04,
        "max_drawdown": -0.20,
        "equity_ruined": False,
    }
    passed, reasons = veto_constraint_audit(metrics, baseline, _config())
    assert not passed
    assert "return_retention" in reasons
    assert "order_retention" in reasons


def test_veto_constraint_rejects_bad_train_p10() -> None:
    baseline = {"monthly_return": 0.10, "orders_per_month": 100, "monthly_p10_chunks": -0.03}
    metrics = {
        "monthly_return": 0.11,
        "orders_per_month": 100,
        "monthly_p10_chunks": -0.05,
        "max_drawdown": -0.20,
        "equity_ruined": False,
    }
    passed, reasons = veto_constraint_audit(metrics, baseline, _config())
    assert not passed
    assert "min_p10" in reasons


def test_veto_score_rewards_yield_and_p10_improvement() -> None:
    baseline = {"monthly_return": 0.10, "monthly_p10_chunks": -0.10, "orders_per_month": 100}
    low = {"monthly_return": 0.10, "monthly_p10_chunks": -0.08, "monthly_median_chunks": 0.10, "net_pnl_per_order": 0.001, "orders_per_month": 100}
    high = dict(low, monthly_return=0.18, monthly_p10_chunks=-0.02)
    assert veto_score(high, baseline) > veto_score(low, baseline)
