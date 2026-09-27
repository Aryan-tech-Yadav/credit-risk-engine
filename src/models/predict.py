import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from src.preprocessing.preprocessor import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.decision.policy_engine import PolicyEngine
from src.explainability.shap_explainer import SHAPExplainer
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class CreditRiskPredictor:
    """
    End-to-end predictor that loads saved model artifacts and runs
    inference on new applicant data.
    """

    def __init__(
        self,
        model_path="models/xgboost_model.json",
        preprocessor_path="models/preprocessing.pkl",
        feature_names_path="models/feature_names.pkl",
        config_path="config.yaml",
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.preprocessor = Preprocessor.load(preprocessor_path)
        self.feature_names = joblib.load(feature_names_path)
        self.policy = PolicyEngine(
            low_threshold=self.config["policy"]["low_risk_threshold"],
            medium_threshold=self.config["policy"]["medium_risk_threshold"],
        )

    def predict(self, df: pd.DataFrame) -> dict:
        """Run a single applicant through the full pipeline."""
        df_fe = FeatureEngineer.create_features(df)
        feature_cols = (
            self.config["features"]["numerical"]
            + self.config["features"]["categorical"]
        )

        full_features = df_fe.reindex(columns=feature_cols).copy()

        for col in self.config["features"]["numerical"]:
            full_features[col] = pd.to_numeric(
                full_features[col], errors="coerce"
            )

        for col in self.config["features"]["categorical"]:
            full_features[col] = full_features[col].fillna("Unknown")
        
        X_proc = self.preprocessor.transform(full_features)
        pd_prob = float(self.model.predict_proba(X_proc)[0, 1])

        decision = self.policy.decide(pd_prob)
        return decision

    def predict_with_shap(self, df: pd.DataFrame) -> dict:
        """Run prediction with SHAP local explanation."""
        df_fe = FeatureEngineer.create_features(df)

        feature_cols = (
            self.config["features"]["numerical"]
            + self.config["features"]["categorical"]
        )

        full_features = df_fe.reindex(columns=feature_cols).copy()

        for col in self.config["features"]["numerical"]:
            full_features[col] = pd.to_numeric(
                full_features[col], errors="coerce"
            )

        for col in self.config["features"]["categorical"]:
            full_features[col] = full_features[col].fillna("Unknown")

        X_proc = self.preprocessor.transform(full_features)
        pd_prob = float(self.model.predict_proba(X_proc)[0, 1])
        decision = self.policy.decide(pd_prob)

        explainer = SHAPExplainer("models/xgboost_model.json", self.feature_names)
        local_exp = explainer.local_explanation(X_proc)

        decision["shap_explanation"] = {
            "top_positive": local_exp["top_positive"][:5],
            "top_negative": local_exp["top_negative"][:5],
        }
        return decision


def predict_single(input_dict: dict, config_path="config.yaml") -> dict:
    """Convenience function to predict on a single applicant dict."""
    config = load_config(config_path)
    predictor = CreditRiskPredictor(config_path=config_path)
    df = pd.DataFrame([input_dict])
    return predictor.predict(df)
