from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.overfitting_validation_protocol import (
    combinatorial_purged_paths,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    overfitting_evidence_decision,
    probability_of_backtest_overfitting,
)


def test_combinatorial_purged_paths_have_no_train_test_or_embargo_overlap() -> None:
    paths = combinatorial_purged_paths(80, n_groups=8, n_test_groups=2, embargo_pct=0.02)
    assert len(paths) == 28
    for path in paths:
        train = set(path.train_positions)
        test = set(path.test_positions)
        embargoed = set(path.embargoed_positions)
        assert train.isdisjoint(test)
        assert train.isdisjoint(embargoed)
        assert test
        assert train


def test_pbo_is_high_when_in_sample_winner_reverses_out_of_sample() -> None:
    # Candidate A wins the first half, candidate B wins the second half.
    # CSCV should frequently pick the wrong side out of sample.
    matrix = pd.DataFrame(
        {
            "candidate_a": [0.05] * 8 + [-0.05] * 8,
            "candidate_b": [-0.05] * 8 + [0.05] * 8,
            "candidate_c": [0.00] * 16,
        }
    )
    summary, paths = probability_of_backtest_overfitting(matrix, n_partitions=4, metric="mean")
    assert summary["pbo"] > 0.50
    assert paths["overfit"].any()


def test_pbo_is_low_when_best_candidate_is_stable() -> None:
    matrix = pd.DataFrame(
        {
            "stable_best": [0.03] * 16,
            "weak": [0.00] * 16,
            "bad": [-0.02] * 16,
        }
    )
    summary, _paths = probability_of_backtest_overfitting(matrix, n_partitions=4, metric="mean")
    assert summary["pbo"] == 0.0
    assert summary["relative_rank_median"] > 0.5


def test_expected_max_sharpe_increases_with_number_of_trials() -> None:
    few = expected_max_sharpe(np.array([0.2, 0.3, 0.4]))
    many = expected_max_sharpe(np.linspace(0.2, 0.4, 30))
    assert many > few


def test_deflated_sharpe_penalizes_multiple_trials() -> None:
    returns = pd.Series([0.04, 0.03, 0.02, 0.05, 0.01, 0.04, 0.03, 0.02])
    few_trials = np.array([0.1, 0.2])
    many_high_trials = np.linspace(0.1, 2.5, 40)
    few = deflated_sharpe_ratio(returns, few_trials, periods_per_year=12)
    many = deflated_sharpe_ratio(returns, many_high_trials, periods_per_year=12)
    assert many["expected_max_sharpe"] > few["expected_max_sharpe"]
    assert many["dsr_statistic"] < few["dsr_statistic"]


def test_overfitting_decision_rejects_walk_forward_only_and_bad_metrics() -> None:
    assert (
        overfitting_evidence_decision(
            has_walk_forward=True,
            has_cpcv=False,
            pbo=None,
            dsr_statistic=None,
            cpcv_p25=None,
        )
        == "insufficient anti-overfit evidence: walk-forward only"
    )
    assert (
        overfitting_evidence_decision(
            has_walk_forward=True,
            has_cpcv=True,
            pbo=0.25,
            dsr_statistic=1.0,
            cpcv_p25=0.01,
        )
        == "rejected by PBO"
    )
    assert (
        overfitting_evidence_decision(
            has_walk_forward=True,
            has_cpcv=True,
            pbo=0.05,
            dsr_statistic=0.5,
            cpcv_p25=0.01,
        )
        == "rigorous anti-overfit evidence passed"
    )
