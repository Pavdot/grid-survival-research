from __future__ import annotations

import pandas as pd
import pytest

from src.research.zero_fee_venue_sensitivity_034 import VenueScenario, load_venue_scenarios, scenario_verdict


def test_load_venue_scenarios_rejects_negative_costs() -> None:
    config = {"scenarios": [{"name": "bad", "fee_rate": -0.1, "slippage_bps": 0, "maker_missed_fill_penalty": 0}]}
    with pytest.raises(ValueError, match="fee_rate"):
        load_venue_scenarios(config)


def test_load_venue_scenarios_rejects_invalid_maker_penalty() -> None:
    config = {"scenarios": [{"name": "bad", "fee_rate": 0, "slippage_bps": 0, "maker_missed_fill_penalty": 1.5}]}
    with pytest.raises(ValueError, match="maker_missed_fill_penalty"):
        load_venue_scenarios(config)


def test_venue_scenario_uses_next_bar_lagged_audit_spec() -> None:
    scenario = VenueScenario("zero", 0.0, 0.25, 0.0, "test")
    spec = scenario.to_scenario_spec()
    assert spec.engine == "audit"
    assert spec.entry_execution_mode == "next_bar_open"
    assert spec.signal_lag_bars == 1
    assert spec.mask_lag_bars == 1
    assert spec.fee_rate == 0.0
    assert spec.slippage_bps == 0.25


def test_scenario_verdict_distinguishes_robust_fragile_and_failed() -> None:
    config = {
        "decision": {
            "robust_monthly_p10_min": 0.0,
            "positive_monthly_min": 0.0,
            "max_drawdown_block": -0.35,
            "holdout_min_total_return": -0.10,
        }
    }
    robust = pd.DataFrame(
        [
            {
                "scope": "combined_oos",
                "scenario": "s",
                "monthly_return": 0.05,
                "monthly_p10": 0.01,
                "max_drawdown": -0.2,
                "equity_ruined": False,
                "total_return": 0.2,
            },
            {
                "scope": "holdout_90d",
                "scenario": "s",
                "monthly_return": 0.03,
                "monthly_p10": 0.0,
                "max_drawdown": -0.1,
                "equity_ruined": False,
                "total_return": 0.04,
            },
        ]
    )
    assert scenario_verdict(robust, "s", config) == "zero_fee_robust_candidate"

    fragile = robust.copy()
    fragile.loc[fragile["scope"].eq("combined_oos"), "monthly_p10"] = -0.02
    assert scenario_verdict(fragile, "s", config) == "zero_fee_fragile_positive"

    failed = robust.copy()
    failed.loc[failed["scope"].eq("combined_oos"), "monthly_return"] = -0.01
    assert scenario_verdict(failed, "s", config) == "zero_fee_not_enough"
