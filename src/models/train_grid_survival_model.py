from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

from src.backtesting.walk_forward import temporal_train_validation_test_split, walk_forward_splits
from src.models.evaluate_model import calibration_table, evaluate_binary_classifier, write_json
from src.models.explain_model import extract_feature_importance
from src.utils.config_loader import configured_path, load_model_config, load_settings, load_strategy_config
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _load_training_frame(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(configured_path("features_dir", "grid_features.parquet"))
    labels = pd.read_parquet(configured_path("labels_dir", "grid_labels.parquet"))
    if limit is not None:
        features = features.iloc[:limit]
        labels = labels.loc[labels.index <= features.index[-1]]
    common = features.index.intersection(labels.index)
    features = features.loc[common].sort_index()
    labels = labels.loc[common].sort_index()
    return features, labels


def _feature_columns(features: pd.DataFrame, model_config: dict) -> list[str]:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    drop_columns = set(model_config["features"].get("drop_columns", []))
    min_fraction = float(model_config["features"]["min_non_null_fraction"])
    columns = [
        col
        for col in numeric
        if col not in drop_columns and features[col].notna().mean() >= min_fraction
    ]
    if not columns:
        raise ValueError("No usable feature columns after non-null filtering")
    return columns


def _build_models(model_config: dict) -> dict[str, Pipeline]:
    random_state = int(model_config["models"]["random_state"])
    log_cfg = model_config["models"]["logistic_regression"]
    rf_cfg = model_config["models"]["random_forest"]
    dt_cfg = model_config["models"]["decision_tree"]
    gb_cfg = model_config["models"]["boosted_tree"]

    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=int(log_cfg["max_iter"]),
                        class_weight=log_cfg["class_weight"],
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(rf_cfg["n_estimators"]),
                        max_depth=int(rf_cfg["max_depth"]),
                        min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
                        class_weight=rf_cfg["class_weight"],
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "decision_tree_shallow": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=int(dt_cfg["max_depth"]),
                        min_samples_leaf=int(dt_cfg["min_samples_leaf"]),
                        class_weight=dt_cfg["class_weight"],
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        models["hist_gradient_boosting_fallback"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=int(gb_cfg["max_iter"]),
                        max_leaf_nodes=int(gb_cfg["max_leaf_nodes"]),
                        learning_rate=float(gb_cfg["learning_rate"]),
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def _dangerous_mask(labels: pd.DataFrame, model_config: dict) -> pd.Series:
    cfg = model_config["dangerous_case"]
    return (
        (labels["grid_survived"].astype(int) == 0)
        & (
            labels["max_adverse_excursion"].ge(float(cfg["max_adverse_excursion_threshold"]))
            | labels["unrealized_drawdown_max"].ge(float(cfg["unrealized_drawdown_threshold"]))
            | labels["stopped_by_max_loss"].astype(bool)
        )
    )


def train_models(limit: int | None = None) -> dict[str, object]:
    settings = load_settings()
    model_config = load_model_config()
    strategy_config = load_strategy_config()
    threshold = float(strategy_config["model_filter"]["min_survival_probability_open"])
    features, labels = _load_training_frame(limit=limit)
    y = labels["grid_survived"].astype(int)
    feature_columns = _feature_columns(features, model_config)
    X = features[feature_columns]

    split = temporal_train_validation_test_split(
        X.index,
        train_fraction=float(settings["validation"]["train_fraction"]),
        validation_fraction=float(settings["validation"]["validation_fraction"]),
        embargo_bars=int(settings["validation"]["embargo_bars"]),
    )
    dangerous = _dangerous_mask(labels, model_config)
    output_dir = configured_path("model_reports_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _build_models(model_config)
    metrics_by_model: dict[str, object] = {}
    best_name = ""
    best_score = -1.0
    best_model = None

    for name, model in models.items():
        LOGGER.info("Training %s on %s rows", name, len(split.train))
        model.fit(X.loc[split.train], y.loc[split.train])
        val_score = model.predict_proba(X.loc[split.validation])[:, 1]
        test_score = model.predict_proba(X.loc[split.test])[:, 1]
        val_metrics = evaluate_binary_classifier(
            y.loc[split.validation],
            val_score,
            threshold,
            dangerous_mask=dangerous.loc[split.validation],
        )
        test_metrics = evaluate_binary_classifier(
            y.loc[split.test],
            test_score,
            threshold,
            dangerous_mask=dangerous.loc[split.test],
        )
        metrics_by_model[name] = {"validation": val_metrics, "test": test_metrics}
        selection_score = val_metrics.get("pr_auc")
        selection_score = float(selection_score) if selection_score is not None else float(val_metrics["f1_score"])
        if selection_score > best_score:
            best_name = name
            best_score = selection_score
            best_model = model
        joblib.dump({"model": model, "feature_columns": feature_columns}, output_dir / f"{name}.joblib")

    if best_model is None:
        raise RuntimeError("No model was trained")

    joblib.dump(
        {"model": best_model, "feature_columns": feature_columns, "model_name": best_name},
        output_dir / "best_grid_survival_model.joblib",
    )

    predictions = pd.DataFrame(index=X.index)
    predictions["dataset_split"] = "unused"
    predictions.loc[split.train, "dataset_split"] = "train"
    predictions.loc[split.validation, "dataset_split"] = "validation"
    predictions.loc[split.test, "dataset_split"] = "test"
    predictions["grid_survival_score"] = pd.NA
    predictions.loc[split.validation, "grid_survival_score"] = best_model.predict_proba(X.loc[split.validation])[:, 1]
    predictions.loc[split.test, "grid_survival_score"] = best_model.predict_proba(X.loc[split.test])[:, 1]
    predictions.to_parquet(output_dir / "grid_survival_predictions.parquet")

    importance = extract_feature_importance(best_model, feature_columns)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    calibration_table(
        y.loc[split.test],
        best_model.predict_proba(X.loc[split.test])[:, 1],
    ).to_csv(output_dir / "calibration_curve.csv")

    folds = walk_forward_splits(
        X.index,
        train_bars=int(settings["validation"]["walk_forward_train_bars"]),
        test_bars=int(settings["validation"]["walk_forward_test_bars"]),
        step_bars=int(settings["validation"]["walk_forward_step_bars"]),
        embargo_bars=int(settings["validation"]["embargo_bars"]),
    )
    fold_summary = pd.DataFrame(
        {
            "fold": i,
            "train_start": fold.train.min(),
            "train_end": fold.train.max(),
            "test_start": fold.test.min(),
            "test_end": fold.test.max(),
            "train_rows": len(fold.train),
            "test_rows": len(fold.test),
        }
        for i, fold in enumerate(folds, start=1)
    )
    fold_summary.to_csv(output_dir / "walk_forward_folds.csv", index=False)

    payload = {
        "best_model": best_name,
        "selection_metric": "validation_pr_auc",
        "threshold": threshold,
        "feature_count": len(feature_columns),
        "metrics": metrics_by_model,
        "xgboost_available": XGBClassifier is not None,
    }
    write_json(payload, output_dir / "model_metrics.json")
    LOGGER.info("Best model: %s", best_name)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train grid survival models with temporal validation.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    train_models(limit=args.limit)


if __name__ == "__main__":
    main()

