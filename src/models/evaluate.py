import pandas as pd
import numpy as np
import xgboost as xgb
from src.preprocessing.preprocessor import Preprocessor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt
# seaborn removed: Matplotlib-only evaluation
import joblib
from pathlib import Path


class ModelEvaluator:
    """
    Post-training evaluation utilities for the XGBoost credit risk model.
    """

    def __init__(self, model, X_test, y_test, feature_names: list):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names

    def get_predictions(self):
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        return y_pred, y_prob

    def classification_metrics(self) -> dict:
        y_pred, y_prob = self.get_predictions()
        metrics = {
            "roc_auc": roc_auc_score(self.y_test, y_prob),
            "pr_auc": average_precision_score(self.y_test, y_prob),
            "precision": precision_score(self.y_test, y_pred),
            "recall": recall_score(self.y_test, y_pred),
            "f1": f1_score(self.y_test, y_pred),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
        }
        return metrics

    def classification_report_str(self) -> str:
        y_pred, _ = self.get_predictions()
        return classification_report(self.y_test, y_pred)

    def plot_roc_curve(self, save_path: str = None) -> plt.Figure:
        _, y_prob = self.get_predictions()
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        auc = roc_auc_score(self.y_test, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    def plot_confusion_matrix(self, save_path: str = None) -> plt.Figure:
        y_pred, _ = self.get_predictions()
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(["No Default", "Default"])
        ax.set_yticklabels(["No Default", "Default"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    def plot_score_distribution(self, save_path: str = None) -> plt.Figure:
        _, y_prob = self.get_predictions()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            y_prob[self.y_test == 0], bins=50, alpha=0.6, label="No Default", color="green"
        )
        ax.hist(
            y_prob[self.y_test == 1], bins=50, alpha=0.6, label="Default", color="red"
        )
        ax.set_xlabel("Predicted Probability of Default")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution by Class")
        ax.legend()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        return fig

    def evaluate_all(self, output_dir: str = "models/evaluation"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        metrics = self.classification_metrics()

        self.plot_roc_curve(f"{output_dir}/roc_curve.png")
        self.plot_confusion_matrix(f"{output_dir}/confusion_matrix.png")
        self.plot_score_distribution(f"{output_dir}/score_distribution.png")

        with open(f"{output_dir}/classification_report.txt", "w") as f:
            f.write(self.classification_report_str())

        return metrics


def evaluate_from_artifacts(
    model_path="models/xgboost_model.json",
    preprocessor_path="models/preprocessing.pkl",
    test_data_path=None,
    config_path="config.yaml",
):
    """Load saved artifacts and run evaluation on test data."""
    import yaml
    from src.data.load_data import load_raw_data
    from src.features.feature_engineering import FeatureEngineer
    from train import prepare_data

    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = load_raw_data(config)
    df = FeatureEngineer.create_features(df)

    X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(df, config)

    preprocessor = Preprocessor.load(preprocessor_path)

    X_test_proc = preprocessor.transform(X_test)
   

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    evaluator = ModelEvaluator(model, X_test_proc, y_test, preprocessor.feature_names)
    return evaluator.evaluate_all()



