from __future__ import annotations

import argparse

import joblib
import pandas as pd

from src.utils.config_loader import configured_path
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def predict_scores(model_path: str | None = None, output_name: str = "grid_survival_predictions.parquet") -> pd.DataFrame:
    features = pd.read_parquet(configured_path("features_dir", "grid_features.parquet"))
    if model_path is None:
        model_path = str(configured_path("model_reports_dir", "best_grid_survival_model.joblib"))
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    scores = model.predict_proba(features[feature_columns])[:, 1]
    predictions = pd.DataFrame({"grid_survival_score": scores}, index=features.index)
    output = configured_path("model_reports_dir", output_name)
    predictions.to_parquet(output)
    LOGGER.info("Wrote predictions to %s", output)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict grid survival scores.")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    predict_scores(args.model_path)


if __name__ == "__main__":
    main()

