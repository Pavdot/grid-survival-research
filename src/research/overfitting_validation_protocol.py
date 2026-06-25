from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class CpcvPath:
    split_id: int
    train_groups: tuple[int, ...]
    test_groups: tuple[int, ...]
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    embargoed_positions: tuple[int, ...]


def chronological_groups(n_observations: int, n_groups: int) -> dict[int, tuple[int, ...]]:
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    if n_groups <= 1:
        raise ValueError("n_groups must be greater than 1")
    effective_groups = min(int(n_groups), int(n_observations))
    chunks = np.array_split(np.arange(n_observations, dtype=int), effective_groups)
    groups = {idx: tuple(int(value) for value in chunk.tolist()) for idx, chunk in enumerate(chunks)}
    if any(not values for values in groups.values()):
        raise ValueError("chronological grouping produced an empty group")
    return groups


def combinatorial_purged_paths(
    n_observations: int,
    n_groups: int = 8,
    n_test_groups: int = 2,
    embargo_pct: float = 0.02,
) -> list[CpcvPath]:
    if n_test_groups <= 0:
        raise ValueError("n_test_groups must be positive")
    if embargo_pct < 0 or embargo_pct >= 1:
        raise ValueError("embargo_pct must stay in [0, 1)")
    groups = chronological_groups(n_observations, n_groups)
    group_ids = tuple(sorted(groups))
    if n_test_groups >= len(group_ids):
        raise ValueError("n_test_groups must be smaller than effective n_groups")
    embargo = int(math.ceil(n_observations * float(embargo_pct)))
    paths: list[CpcvPath] = []
    for split_id, test_groups_raw in enumerate(itertools.combinations(group_ids, int(n_test_groups))):
        test_groups = tuple(int(value) for value in test_groups_raw)
        test_positions = tuple(sorted(pos for group in test_groups for pos in groups[group]))
        embargoed: set[int] = set(test_positions)
        for pos in test_positions:
            start = max(0, int(pos) - embargo)
            stop = min(n_observations, int(pos) + embargo + 1)
            embargoed.update(range(start, stop))
        train_groups = tuple(group for group in group_ids if group not in test_groups)
        train_positions = tuple(
            pos
            for group in train_groups
            for pos in groups[group]
            if pos not in embargoed
        )
        if not train_positions:
            continue
        paths.append(
            CpcvPath(
                split_id=int(split_id),
                train_groups=train_groups,
                test_groups=test_groups,
                train_positions=train_positions,
                test_positions=test_positions,
                embargoed_positions=tuple(sorted(embargoed - set(test_positions))),
            )
        )
    if not paths:
        raise ValueError("CPCV produced no usable purged paths")
    return paths


def cscv_splits(
    n_observations: int,
    n_partitions: int = 16,
    max_splits: int | None = None,
    random_seed: int = 42,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if n_observations <= 1:
        raise ValueError("n_observations must be greater than 1")
    if n_partitions < 4 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be an even integer >= 4")
    effective_partitions = min(int(n_partitions), int(n_observations))
    if effective_partitions % 2 != 0:
        effective_partitions -= 1
    if effective_partitions < 4:
        raise ValueError("effective CSCV partitions must be at least 4")
    groups = chronological_groups(n_observations, effective_partitions)
    group_ids = tuple(sorted(groups))
    half = len(group_ids) // 2
    combos = list(itertools.combinations(group_ids, half))
    if max_splits is not None and len(combos) > int(max_splits):
        rng = np.random.default_rng(int(random_seed))
        selected = np.sort(rng.choice(len(combos), size=int(max_splits), replace=False))
        combos = [combos[int(index)] for index in selected]
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for train_groups in combos:
        train_set = set(train_groups)
        test_groups = tuple(group for group in group_ids if group not in train_set)
        train_positions = tuple(pos for group in train_groups for pos in groups[group])
        test_positions = tuple(pos for group in test_groups for pos in groups[group])
        if train_positions and test_positions:
            splits.append((train_positions, test_positions))
    if not splits:
        raise ValueError("CSCV produced no splits")
    return splits


def sharpe_score(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return float(np.sign(np.mean(arr)) * np.inf) if float(np.mean(arr)) != 0 else 0.0
    return float(np.mean(arr) / std)


def _metric_score(values: pd.Series | np.ndarray, metric: str) -> float:
    if metric == "mean":
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if len(arr) else 0.0
    if metric == "sharpe":
        return sharpe_score(values)
    raise ValueError(f"Unsupported CSCV metric: {metric}")


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_partitions: int = 16,
    metric: str = "sharpe",
    max_splits: int | None = None,
    random_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if returns_matrix.empty:
        raise ValueError("returns_matrix cannot be empty")
    numeric = returns_matrix.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if numeric.shape[0] < 4 or numeric.shape[1] < 2:
        raise ValueError("PBO requires at least 4 observations and 2 candidates")
    splits = cscv_splits(len(numeric), n_partitions, max_splits=max_splits, random_seed=random_seed)
    rows: list[dict[str, Any]] = []
    n_candidates = numeric.shape[1]
    for split_id, (train_pos, test_pos) in enumerate(splits):
        train = numeric.iloc[list(train_pos)]
        test = numeric.iloc[list(test_pos)]
        train_scores = train.apply(lambda col: _metric_score(col, metric), axis=0)
        test_scores = test.apply(lambda col: _metric_score(col, metric), axis=0)
        selected_candidate = str(train_scores.sort_values(ascending=False).index[0])
        ordered_test = test_scores.sort_values(ascending=True)
        rank_from_worst = int(np.where(ordered_test.index.to_numpy() == selected_candidate)[0][0]) + 1
        relative_rank = rank_from_worst / float(n_candidates + 1)
        relative_rank = min(max(relative_rank, 1e-12), 1.0 - 1e-12)
        logit = math.log(relative_rank / (1.0 - relative_rank))
        rows.append(
            {
                "split_id": int(split_id),
                "selected_candidate": selected_candidate,
                "train_score": float(train_scores[selected_candidate]),
                "test_score": float(test_scores[selected_candidate]),
                "test_rank_from_worst": int(rank_from_worst),
                "candidate_count": int(n_candidates),
                "relative_rank": float(relative_rank),
                "logit": float(logit),
                "overfit": bool(logit < 0),
            }
        )
    paths = pd.DataFrame(rows)
    summary = {
        "method": "CSCV_PBO",
        "n_observations": int(numeric.shape[0]),
        "candidate_count": int(n_candidates),
        "n_partitions": int(n_partitions),
        "split_count": int(len(paths)),
        "metric": metric,
        "pbo": float(paths["overfit"].mean()),
        "logit_median": float(paths["logit"].median()),
        "relative_rank_median": float(paths["relative_rank"].median()),
    }
    return summary, paths


def _sample_skewness(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    centered = arr - np.mean(arr)
    std = np.std(arr, ddof=0)
    if std <= 0:
        return 0.0
    return float(np.mean(centered**3) / std**3)


def _sample_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 3.0
    centered = arr - np.mean(arr)
    std = np.std(arr, ddof=0)
    if std <= 0:
        return 3.0
    return float(np.mean(centered**4) / std**4)


def expected_max_sharpe(trial_sharpes: pd.Series | np.ndarray) -> float:
    values = np.asarray(trial_sharpes, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    mean_sr = float(np.mean(values))
    std_sr = float(np.std(values, ddof=1))
    if std_sr <= 0:
        return mean_sr
    n_trials = float(len(values))
    z_1 = NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    z_2 = NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(mean_sr + std_sr * ((1.0 - EULER_GAMMA) * z_1 + EULER_GAMMA * z_2))


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    trial_sharpes: pd.Series | np.ndarray,
    periods_per_year: float = 12.0,
) -> dict[str, float]:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        raise ValueError("DSR requires at least 3 return observations")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    periodic_sr = sharpe_score(arr)
    candidate_sr = float(periodic_sr * math.sqrt(float(periods_per_year)))
    trial_values = np.asarray(trial_sharpes, dtype=float)
    trial_values = trial_values[np.isfinite(trial_values)]
    sr_star = expected_max_sharpe(trial_values)
    skew = _sample_skewness(arr)
    kurtosis = _sample_kurtosis(arr)
    denominator = 1.0 - skew * candidate_sr + ((kurtosis - 1.0) / 4.0) * candidate_sr**2
    denominator = math.sqrt(max(denominator, 1e-12))
    statistic = (candidate_sr - sr_star) * math.sqrt(len(arr) - 1.0) / denominator
    probability = NORMAL.cdf(statistic)
    return {
        "method": "DSR",
        "observations": float(len(arr)),
        "periods_per_year": float(periods_per_year),
        "candidate_sharpe": float(candidate_sr),
        "expected_max_sharpe": float(sr_star),
        "skewness": float(skew),
        "kurtosis": float(kurtosis),
        "dsr_statistic": float(statistic),
        "dsr_probability": float(probability),
        "trial_count": float(len(trial_values)),
    }


def overfitting_evidence_decision(
    *,
    has_walk_forward: bool,
    has_cpcv: bool,
    pbo: float | None,
    dsr_statistic: float | None,
    cpcv_p25: float | None,
    max_pbo: float = 0.10,
    min_dsr_statistic: float = 0.0,
    min_cpcv_p25: float = 0.0,
) -> str:
    if not has_walk_forward:
        return "insufficient evidence: no walk-forward"
    if not has_cpcv:
        return "insufficient anti-overfit evidence: walk-forward only"
    if pbo is None or dsr_statistic is None:
        return "insufficient anti-overfit evidence: missing PBO/DSR"
    if float(pbo) > float(max_pbo):
        return "rejected by PBO"
    if float(dsr_statistic) < float(min_dsr_statistic):
        return "rejected by DSR"
    if cpcv_p25 is not None and float(cpcv_p25) <= float(min_cpcv_p25):
        return "rejected by CPCV p25"
    return "rigorous anti-overfit evidence passed"
