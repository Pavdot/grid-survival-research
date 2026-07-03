from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.infra.shadow_037_preflight import is_lfs_pointer, run_preflight


class Shadow037PreflightTests(unittest.TestCase):
    def test_lfs_pointer_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.csv"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\n"
                "size 123\n",
                encoding="utf-8",
            )
            self.assertTrue(is_lfs_pointer(path))
            path.write_text("real,data\n1,2\n", encoding="utf-8")
            self.assertFalse(is_lfs_pointer(path))

    def test_repository_preflight_is_ready_for_fixed_launch_date(self) -> None:
        payload = run_preflight(
            "config/shadow_live_037.yaml",
            online=False,
            now=pd.Timestamp("2026-06-30T00:00:00Z"),
        )
        self.assertTrue(payload["ready"], payload)
        self.assertEqual(payload["mode"], "paper_only")


if __name__ == "__main__":
    unittest.main()
