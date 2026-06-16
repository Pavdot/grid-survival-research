from __future__ import annotations

import argparse
import json

from src.research.fundamental_blackout_ablation_research import run_iteration


DEFAULT_CONFIG = "config/research_iteration_xauusd_blackout_ablation_martingale.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XAUUSD fundamental-blackout ablation walk-forward research.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--exact-top-n", type=int, default=None)
    args = parser.parse_args()
    payload = run_iteration(
        args.config,
        max_folds=args.max_folds,
        max_candidates=args.max_candidates,
        exact_top_n=args.exact_top_n,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
