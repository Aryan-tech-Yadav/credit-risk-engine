import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb


class SHAPExplainer:
    """
    Wrapper around SHAP TreeExplainer for XGBoost credit risk models.
    Provides both global and local (per-applicant) explanations.
    """

    def __init__(self, model_path: str, feature_names: list):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)

    @staticmethod
    def _get_expected_value(explainer):
        val = explainer.expected_value
        if isinstance(val, np.ndarray):
            return float(val[0])
        return float(val)

    def global_explanation(self, X: pd.DataFrame, max_display: int = 15):
        """Return SHAP values and summary plot for the whole dataset."""
        shap_values = self.explainer.shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        return shap_values

    def local_explanation(self, x_single: pd.DataFrame) -> dict:
        """
        Explain a single prediction.
        Returns dict with:
          - base_value: expected model output
          - shap_values: array of feature contributions
          - top_positive: features that INCREASE default risk
          - top_negative: features that DECREASE default risk
        """
        shap_values = self.explainer.shap_values(x_single)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        sv = shap_values[0] if shap_values.ndim > 1 else shap_values
        base_value = self._get_expected_value(self.explainer)

        contributions = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": sv,
                "abs_shap": np.abs(sv),
            }
        ).sort_values("abs_shap", ascending=False)

        top_pos = contributions[contributions["shap_value"] > 0].head(5)
        top_neg = contributions[contributions["shap_value"] < 0].head(5)

        return {
            "base_value": base_value,
            "shap_values": sv,
            "top_positive": top_pos.to_dict("records"),
            "top_negative": top_neg.to_dict("records"),
            "all_contributions": contributions,
        }

    def waterfall_plot(self, x_single: pd.DataFrame, save_path: str = None):
        """Generate a SHAP waterfall plot for a single prediction."""
        shap_values = self.explainer.shap_values(x_single)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        base_value = self._get_expected_value(self.explainer)
        values = shap_values[0] if shap_values.ndim > 1 else shap_values

        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=values,
                base_values=base_value,
                data=x_single.iloc[0].values,
                feature_names=self.feature_names,
            ),
            show=False,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
