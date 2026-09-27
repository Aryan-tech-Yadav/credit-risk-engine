import numpy as np


class PolicyEngine:
    """
    Business rule layer that sits ON TOP of the ML model.

    The ML model outputs P(Default). This engine converts that probability
    into a risk score, risk band, and recommendation based on configurable
    business policy — NOT hardcoded banking standards.

    Thresholds here are DEMO values. In real lending, they would be
    calibrated against portfolio-level expected loss, regulatory capital,
    and business risk appetite.
    """

    def __init__(self, low_threshold: float = 0.10, medium_threshold: float = 0.25):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def probability_to_score(self, pd: float) -> int:
        """
        Convert default probability to a 0-100 risk score.
        Higher score = lower risk (more creditworthy).

        Score = (1 - PD) * 100
        """
        pd = np.clip(pd, 0.0, 1.0)
        return int(round((1.0 - pd) * 100))

    def classify_risk(self, pd: float) -> str:
        """Map probability to risk band."""
        if pd < self.low_threshold:
            return "LOW"
        elif pd < self.medium_threshold:
            return "MEDIUM"
        else:
            return "HIGH"

    def recommend_action(self, risk_band: str) -> str:
        """Map risk band to a recommended action (for human review)."""
        mapping = {
            "LOW": "RECOMMEND: APPROVE",
            "MEDIUM": "RECOMMEND: REVIEW",
            "HIGH": "RECOMMEND: DECLINE",
        }
        return mapping.get(risk_band, "RECOMMEND: MANUAL REVIEW")

    def decide(self, pd: float) -> dict:
        """
        Full decision pipeline for a single applicant.
        Returns dict with probability, score, band, action, and reason.
        """
        score = self.probability_to_score(pd)
        band = self.classify_risk(pd)
        action = self.recommend_action(band)

        return {
            "probability_of_default": round(pd, 4),
            "risk_score": score,
            "risk_band": band,
            "recommendation": action,
        }

    def explain_decision(self, pd: float, top_factors: list) -> str:
        """Generate a human-readable explanation string."""
        band = self.classify_risk(pd)
        action = self.recommend_action(band)
        reasons = "\n".join(
            [f"  • {f['feature']}: {f['shap_value']:+.3f}" for f in top_factors[:3]]
        )
        return f"{action}\nRisk Band: {band}\nTop factors:\n{reasons}"
