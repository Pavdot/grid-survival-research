from __future__ import annotations

import pandas as pd

from src.research.zero_fee_pareto_optimizer_036 import (
    add_chunk_stats,
    pareto_constraint_audit,
    pareto_score,
    pareto_verdict,
    seed_historical_candidates,
)


def _config() -> dict:
    return {
        "primary_scenario": {"name": "zero_fee_0p25bps", "fee_rate": 0.0, "slippage_bps": 0.25},
        "selection": {
            "min_grids_per_month": 20,
            "max_grids_per_month": 80,
            "min_orders_per_month": 45,
            "max_orders_per_month": 150,
            "min_monthly_return": 0.10,
            "min_monthly_median": 0.06,
            "stability_slices": 3,
            "min_stability_positive_rate": 0.67,
            "min_stability_worst_monthly": -0.12,
            "max_drawdown": -0.40,
            "require_net_pnl_per_order_positive": True,
            "weights": {
                "monthly_return": 0.45,
                "monthly_median": 0.20,
                "monthly_p10": 0.20,
                "net_pnl_per_order": 20.0,
                "drawdown": 0.10,
                "order_distance": 0.05,
            },
        },
        "validation": {
            "min_interesting_monthly": 0.10,
            "min_interesting_median": 0.06,
            "max_p10_floor": -0.08,
            "holdout_min_total_return": 0.0,
        },
    }


def test_pareto_constraints_accept_active_profitable_candidate() -> None:
    passed, reasons = pareto_constraint_audit(
        {
            "equity_ruined": False,
            "max_drawdown": -0.25,
            "monthly_return": 0.12,
            "monthly_median_chunks": 0.08,
            "monthly_p10_chunks": -0.04,
            "grids_per_month": 40,
            "orders_per_month": 90,
            "net_pnl_per_order": 0.001,
        },
        _config(),
    )
    assert passed
    assert reasons == "pass"


def test_pareto_constraints_reject_diluted_low_activity_candidate() -> None:
    passed, reasons = pareto_constraint_audit(
        {
            "equity_ruined": False,
            "max_drawdown": -0.10,
            "monthly_return": 0.02,
            "monthly_median_chunks": 0.01,
            "monthly_p10_chunks": 0.0,
            "grids_per_month": 5,
            "orders_per_month": 12,
            "net_pnl_per_order": 0.002,
        },
        _config(),
    )
    assert not passed
    assert "min_monthly_return" in reasons
    assert "min_orders" in reasons


def test_pareto_constraints_reject_unstable_train_profile() -> None:
    passed, reasons = pareto_constraint_audit(
        {
            "equity_ruined": False,
            "max_drawdown": -0.25,
            "monthly_return": 0.20,
            "monthly_median_chunks": 0.18,
            "monthly_p10_chunks": 0.01,
            "grids_per_month": 40,
            "orders_per_month": 90,
            "net_pnl_per_order": 0.002,
            "stability_positive_rate": 0.33,
            "stability_worst_monthly": -0.30,
        },
        _config(),
    )
    assert not passed
    assert "stability_positive_rate" in reasons
    assert "stability_worst_monthly" in reasons


def test_pareto_score_prefers_yield_when_p10_is_similar() -> None:
    cfg = _config()
    low_yield = {
        "monthly_return": 0.03,
        "monthly_median_chunks": 0.02,
        "monthly_p10_chunks": -0.02,
        "net_pnl_per_order": 0.001,
        "max_drawdown": -0.10,
        "orders_per_month": 80,
    }
    high_yield = dict(low_yield, monthly_return=0.14, monthly_median_chunks=0.08, max_drawdown=-0.20)
    assert pareto_score(high_yield, cfg) > pareto_score(low_yield, cfg)


def test_add_chunk_stats_adds_median_and_p10() -> None:
    index = pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC")
    equity = pd.Series([1 + i * 0.001 for i in range(60)], index=index)
    metrics = add_chunk_stats({}, equity)
    assert metrics["monthly_median_chunks"] > 0
    assert metrics["monthly_p10_chunks"] > 0


def test_seed_historical_candidates_keep_original_grid_with_zero_fee_target() -> None:
    base = {
        "name": "orig",
        "entry_cooldown_hours": 3,
        "rsi_low": 40,
        "rsi_high": 60,
        "spacing_atr_multiplier": 3,
        "take_profit_spacing_multiplier": 2,
        "max_levels": 5,
        "base_position_size_pct": 2.5,
        "progression_multiplier": 1.55,
        "max_total_exposure_pct": 40,
        "max_holding_hours": 6,
    }
    cfg = _config()
    cfg["candidate_search"] = {"max_grids_per_month": [35, 90]}
    seed = seed_historical_candidates([base], cfg)[0]
    assert seed["take_profit_spacing_multiplier"] == 2
    assert seed["max_levels"] == 5
    assert seed["fee_rate"] == 0.0
    assert seed["slippage_bps"] == 0.25
    assert seed["pause_after_forced_loss_hours"] == 0.0


def test_pareto_verdict_flags_candidate_when_yield_and_tail_are_acceptable() -> None:
    summary = pd.DataFrame(
        [
            {
                "scope": "combined_oos",
                "scenario": "zero_fee_0p25bps",
                "monthly_return": 0.12,
                "monthly_median": 0.08,
                "monthly_p10": -0.05,
                "equity_ruined": False,
                "total_return": 1.0,
            },
            {
                "scope": "holdout_90d",
                "scenario": "zero_fee_0p25bps",
                "monthly_return": 0.04,
                "monthly_median": 0.04,
                "monthly_p10": 0.0,
                "equity_ruined": False,
                "total_return": 0.10,
            },
        ]
    )
    assert pareto_verdict(summary, pd.DataFrame(), pd.DataFrame(), _config()) == "pareto candidate - yield preserved"
