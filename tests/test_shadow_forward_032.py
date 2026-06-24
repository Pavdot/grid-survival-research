from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.paper.shadow_forward_032 import choose_shadow_policy, run_shadow_once, shadow_verdict


def base_config(output_dir: str) -> dict:
    return {
        "shadow": {
            "output_dir": output_dir,
            "infrastructure_config": "config/infrastructure_microstructure.yaml",
            "policy_config": "config/research_iteration_microstructure_order_policy_017.yaml",
            "policy_ablation_dir": output_dir,
            "conditional_execution_dir": output_dir,
            "execution_surface_dir": output_dir,
            "default_policy": "maker_entry_add_taker_exit",
            "primary_equity_usdt": 10000,
            "min_completed_days_for_verdict": 30,
            "max_collector_age_seconds": 20,
            "private_env_vars": ["BINANCE_API_KEY"],
            "live_trading_env_var": "LIVE_TRADING_ENABLED",
        }
    }


class ShadowForward032Tests(unittest.TestCase):
    def test_choose_shadow_policy_uses_real_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy_ablation_summary.csv"
            pd.DataFrame(
                [
                    {"policy": "gate_only_taker", "policy_improves_net": False},
                    {"policy": "sliced_taker", "policy_improves_net": True},
                ]
            ).to_csv(path, index=False)
            config = base_config(tmp)
            self.assertEqual(choose_shadow_policy(config), "sliced_taker")

    def test_shadow_verdict_requires_completed_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            verdict, reason = shadow_verdict({"status": "shadow_ready", "reason": "ready"}, Path(tmp), min_days=30)
            self.assertEqual(verdict, "collecting")
            self.assertIn("below required", reason)

    def test_run_shadow_refuses_live_trading_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LIVE_TRADING_ENABLED": "true"}, clear=True):
                with self.assertRaises(RuntimeError):
                    run_shadow_once(base_config(tmp), duration_days=30)

    def test_run_shadow_refuses_private_key_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"BINANCE_API_KEY": "secret"}, clear=True):
                with self.assertRaises(RuntimeError):
                    run_shadow_once(base_config(tmp), duration_days=30)

    def test_stale_collector_writes_kill_switch_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=True):
                with patch("src.paper.shadow_forward_032.collector_healthcheck", return_value=(False, "stale")):
                    payload = run_shadow_once(base_config(tmp), duration_days=30)
            self.assertEqual(payload["status"]["status"], "shadow_kill_switch")
            self.assertEqual(payload["status"]["reason"], "stale_or_unhealthy_collector")
            self.assertTrue(Path(payload["output_paths"]["report"]).exists())


if __name__ == "__main__":
    unittest.main()
