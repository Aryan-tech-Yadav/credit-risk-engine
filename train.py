import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import joblib
from pathlib import Path
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_data(df: pd.DataFrame, config: dict):
    """Split into train/test BEFORE any preprocessing or resampling."""
    target = "TARGET"

    feature_cols = (
        config["features"]["numerical"]
        + config["features"]["categorical"]
    )

    X = df[feature_cols]
    y = df[target]

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train default rate: {y_train.mean():.4f}")
    print(f"Test default rate:  {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols


def train_models(X_train, y_train, X_test, y_test, scale_pos_weight):
    """Train and compare multiple models."""
    results = {}

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_train, y_train)
    results["LogisticRegression"] = evaluate_model(
        lr, X_test, y_test, "LogisticRegression"
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results["RandomForest"] = evaluate_model(
        rf, X_test, y_test, "RandomForest"
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device="cpu",
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    results["XGBoost"] = evaluate_model(
        xgb_model,
        X_test,
        y_test,
        "XGBoost",
    )

    return results, xgb_model


def evaluate_model(model, X_test, y_test, name):
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")

    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\n  Confusion Matrix:\n  {confusion_matrix(y_test, y_pred)}")

    return metrics


def train_full_pipeline(config_path="config.yaml"):
    """Complete training pipeline."""
    config = load_config(config_path)

    # 1. Load data
    from src.data.load_data import load_raw_data

    df = load_raw_data(config)

    # 2. Feature engineering
    from src.features.feature_engineering import FeatureEngineer

    df = FeatureEngineer.create_features(df)

    # 3. Prepare data
    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(
        df,
        config,
    )

    # 4. Preprocess
    from src.preprocessing.preprocessor import Preprocessor

    preprocessor = Preprocessor(num_cols, cat_cols)
    preprocessor.build()

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5. Compute class weight from training data only
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

    # 6. Train models
    results, best_model = train_models(
        X_train_proc,
        y_train,
        X_test_proc,
        y_test,
        scale_pos_weight,
    )

    # 7. Save artifacts
    Path("models").mkdir(exist_ok=True)

    best_model.save_model("models/xgboost_model.json")
    preprocessor.save("models/preprocessing.pkl")
    joblib.dump(
        preprocessor.feature_names,
        "models/feature_names.pkl",
    )

    print("\nTraining complete. Model saved to models/")

    return (
        results,
        best_model,
        preprocessor,
        X_test_proc,
        y_test,
    )


if __name__ == "__main__":
    import sys

    config_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "config.yaml"
    )

    train_full_pipeline(config_path)