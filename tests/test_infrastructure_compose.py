from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.utils.config_loader import project_path


class InfrastructureComposeTests(unittest.TestCase):
    def test_default_compose_stack_contains_complete_shadow_dependencies(self) -> None:
        compose = yaml.safe_load(project_path("docker-compose.yml").read_text(encoding="utf-8"))
        services = compose["services"]
        core = {
            name
            for name, service in services.items()
            if not service.get("profiles")
        }
        expected = {
            "collector-btcusdt-klines",
            "collector-btcusdt-futures-klines",
            "collector-btcusdt-futures-depth",
            "shadow-runner-037",
        }
        self.assertEqual(core, expected)
        dependencies = services["shadow-runner-037"]["depends_on"]
        self.assertEqual(set(dependencies), expected - {"shadow-runner-037"})
        self.assertIn("--run", services["shadow-runner-037"]["command"])
        self.assertIn("healthcheck", services["shadow-runner-037"])

    def test_compose_contains_no_private_binance_endpoint_or_key(self) -> None:
        text = project_path("docker-compose.yml").read_text(encoding="utf-8").lower()
        self.assertNotIn("/api/v3/order", text)
        self.assertNotIn("/fapi/v1/order", text)
        self.assertNotIn("binance_api_key", text)
        self.assertNotIn("binance_secret", text)
        self.assertIn("127.0.0.1:8000:8000", text)

    def test_systemd_starts_continuous_shadow_once(self) -> None:
        main = project_path("deploy/systemd/grid-survival-research.service").read_text(encoding="utf-8")
        hourly = project_path("deploy/systemd/grid-survival-ops-hourly.service").read_text(encoding="utf-8")
        self.assertIn("collector-btcusdt-futures-depth", main)
        self.assertIn("collector-btcusdt-futures-klines", main)
        self.assertIn("shadow-runner-037", main)
        self.assertNotIn("shadow-runner-037", hourly)

    def test_live_macro_calendar_extends_through_year_end(self) -> None:
        events = pd.read_csv(project_path("config/fundamental_events_live.csv"))
        timestamps = pd.to_datetime(events["event_time_utc"], utc=True)
        self.assertGreaterEqual(timestamps.max(), pd.Timestamp("2026-12-15T13:30:00Z"))
        self.assertTrue({"macro_fomc", "macro_cpi", "macro_ppi", "macro_jobs"}.issubset(set(events["category"])))


if __name__ == "__main__":
    unittest.main()
