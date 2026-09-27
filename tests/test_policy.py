import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.decision.policy_engine import PolicyEngine


def test_low_risk():
    policy = PolicyEngine(low_threshold=0.10, medium_threshold=0.25)
    result = policy.decide(0.05)
    assert result["risk_band"] == "LOW"
    assert "APPROVE" in result["recommendation"]
    assert result["risk_score"] == 95


def test_medium_risk():
    policy = PolicyEngine(low_threshold=0.10, medium_threshold=0.25)
    result = policy.decide(0.18)
    assert result["risk_band"] == "MEDIUM"
    assert "REVIEW" in result["recommendation"]


def test_high_risk():
    policy = PolicyEngine(low_threshold=0.10, medium_threshold=0.25)
    result = policy.decide(0.60)
    assert result["risk_band"] == "HIGH"
    assert "DECLINE" in result["recommendation"]


def test_score_boundaries():
    policy = PolicyEngine()
    assert policy.probability_to_score(0.0) == 100
    assert policy.probability_to_score(1.0) == 0
    assert policy.probability_to_score(0.5) == 50


def test_score_clipping():
    policy = PolicyEngine()
    assert policy.probability_to_score(-0.5) == 100
    assert policy.probability_to_score(1.5) == 0


def test_decision_keys():
    policy = PolicyEngine()
    result = policy.decide(0.3)
    assert set(result.keys()) == {
        "probability_of_default",
        "risk_score",
        "risk_band",
        "recommendation",
    }


def test_explain_decision():
    policy = PolicyEngine(low_threshold=0.10, medium_threshold=0.25)
    top_factors = [
        {"feature": "DTI", "shap_value": 0.15},
        {"feature": "EXT_SOURCE_MEAN", "shap_value": -0.20},
        {"feature": "AGE_YEARS", "shap_value": -0.05},
    ]
    explanation = policy.explain_decision(0.3, top_factors)
    assert "LOW" not in explanation or "MEDIUM" in explanation or "HIGH" in explanation
