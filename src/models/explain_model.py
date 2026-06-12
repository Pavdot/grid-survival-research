from __future__ import annotations

import numpy as np
import pandas as pd


def extract_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        importance = np.zeros(len(feature_names), dtype=float)
    out = pd.DataFrame({"feature": feature_names, "importance": importance})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)

