from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.cost_budget_hypothesis_ablation_033 import (
    apply_maker_proxy,
    constraint_audit,
    expected_tp_cost_ratio,
    generate_family_universe,
    notional_turnover_proxy,
    order_count,
)
from src.research.monthly_target_martingale_research import candidate_from_row, planned_exposure


def _config() -> dict:
    return {
        "hypotheses": {
            "H1_cost_budget": {"max_cost_pct_per_month": 0.05, "max_notional_turnover_per_month": 80},
            "H2_entry_reduction": {
                "cooldown_hours": [6, 12],
                "rsi_threshold_pairs": [[35, 65], [30, 70]],
                "max_grids_per_month": [15],
            },
            "H3_wider_tp": {
                "take_profit_spacing_multipliers": [4, 6],
                "min_expected_net_tp_to_roundtrip_cost": 3.0,
            },
            "H4_reduced_martingale": {
                "max_notional_caps": [5, 10],
                "max_levels": [2, 3],
                "progression_multipliers": [1.15],
                "base_size_safety": 0.95,
            },
            "H5_maker_first_proxy": {
                "maker_fee_rate": 0.0002,
                "maker_slippage_bps": 0.25,
                "missed_fill_penalties": [0.10, 0.20],
            },
        }
    }


def _base_row() -> dict:
    return {
        "name": "base",
        "side_mode": "mean_reversion_dual",
        "entry_mode": "hourly_extreme",
        "entry_cooldown_hours": 3.0,
        "rsi_window": 24,
        "rsi_low": 40.0,
        "rsi_high": 60.0,
        "spacing_atr_multiplier": 3.0,
        "take_profit_spacing_multiplier": 2.0,
        "max_levels": 5,
        "base_position_size_pct": 2.5,
        "progression_multiplier": 1.55,
        "max_total_exposure_pct": 40.0,
        "fee_rate": 0.0001,
        "slippage_bps": 0.0,
        "max_grid_loss_pct": 0.35,
        "max_holding_hours": 6.0,
        "stop_on_regime_break": False,
        "stop_on_volatility_shock": True,
    }


def test_h2_entry_reduction_only_changes_entry_axes() -> None:
    rows = generate_family_universe([_base_row()], "H2_entry_reduction", _config())
    assert len(rows) == 4
    first = rows[0]
    assert first["entry_cooldown_hours"] in {6.0, 12.0}
    assert (first["rsi_low"], first["rsi_high"]) in {(35.0, 65.0), (30.0, 70.0)}
    assert first["max_grids_per_month"] == 15.0
    assert first["take_profit_spacing_multiplier"] == 2.0
    assert first["max_levels"] == 5
    assert first["progression_multiplier"] == 1.55


def test_h3_wider_tp_only_changes_tp_axis_and_constraint() -> None:
    rows = generate_family_universe([_base_row()], "H3_wider_tp", _config())
    assert {row["take_profit_spacing_multiplier"] for row in rows} == {4.0, 6.0}
    assert all(row["min_expected_net_tp_to_roundtrip_cost"] == 3.0 for row in rows)
    assert all(row["entry_cooldown_hours"] == 3.0 for row in rows)
    assert all(row["max_levels"] == 5 for row in rows)


def test_h4_reduced_martingale_caps_planned_exposure() -> None:
    rows = generate_family_universe([_base_row()], "H4_reduced_martingale", _config())
    assert rows
    for row in rows:
        candidate = candidate_from_row(row)
        assert planned_exposure(candidate) <= float(row["max_total_exposure_pct"]) + 1e-9
        assert row["max_levels"] in {2, 3}
        assert row["progression_multiplier"] == 1.15


def test_cost_budget_rejects_excess_cost_and_turnover() -> None:
    row = generate_family_universe([_base_row()], "H1_cost_budget", _config())[0]
    passed, reasons = constraint_audit(
        row,
        {"cost_total_per_month": 0.06, "notional_turnover_per_month": 100},
        _config(),
    )
    assert not passed
    assert "cost_budget" in reasons
    assert "turnover_budget" in reasons


def test_tp_min_net_ratio_rejects_insufficient_reward_to_cost() -> None:
    row = generate_family_universe([_base_row()], "H3_wider_tp", _config())[0]
    passed, reasons = constraint_audit(row, {"expected_net_tp_to_roundtrip_cost": 2.9}, _config())
    assert not passed
    assert reasons == "min_net_tp_cost_ratio"


def test_order_count_and_turnover_proxy_use_filled_levels_plus_exits() -> None:
    trades = pd.DataFrame(
        {
            "number_of_levels_filled": [1, 3, 2],
            "max_exposure_pct": [2.5, 7.0, 4.0],
        }
    )
    assert order_count(trades) == 9
    assert notional_turnover_proxy(trades) == 27.0


def test_expected_tp_cost_ratio_uses_positive_pnl_over_average_cost() -> None:
    trades = pd.DataFrame(
        {
            "realized_pnl": [0.03, -0.02, 0.06],
            "fees_paid": [0.005, 0.004, 0.006],
            "slippage_paid": [0.005, 0.006, 0.004],
        }
    )
    assert np.isclose(expected_tp_cost_ratio(trades), 4.5)


def test_maker_proxy_penalizes_only_positive_pnl() -> None:
    trades = pd.DataFrame({"realized_pnl": [0.10, -0.04, 0.00]})
    adjusted = apply_maker_proxy(trades, 0.20)
    assert np.allclose(adjusted["realized_pnl"].to_numpy(), [0.08, -0.04, 0.0])
    assert np.allclose(adjusted["maker_missed_fill_penalty"].to_numpy(), [0.02, 0.0, 0.0])
