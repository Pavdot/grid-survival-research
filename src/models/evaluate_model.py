from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    dangerous_mask: pd.Series | None = None,
) -> dict[str, object]:
    y_true_arr = y_true.astype(int).to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, object] = {
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_arr, y_score)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true_arr)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true_arr, y_score))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    if dangerous_mask is not None:
        dangerous = dangerous_mask.reindex(y_true.index).fillna(False).to_numpy(dtype=bool)
        dangerous_count = int(((y_true_arr == 0) & dangerous).sum())
        allowed_dangerous = int(((y_pred == 1) & (y_true_arr == 0) & dangerous).sum())
        metrics["dangerous_false_negative_rate"] = (
            float(allowed_dangerous / dangerous_count) if dangerous_count else 0.0
        )
        metrics["dangerous_cases"] = dangerous_count
        metrics["allowed_dangerous_cases"] = allowed_dangerous
    return metrics


def calibration_table(y_true: pd.Series, y_score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    table = pd.DataFrame({"y_true": y_true.astype(int), "score": y_score})
    table["bucket"] = pd.cut(table["score"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return table.groupby("bucket", observed=False).agg(
        mean_score=("score", "mean"),
        observed_survival=("y_true", "mean"),
        count=("y_true", "size"),
    )


def write_json(data: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)

