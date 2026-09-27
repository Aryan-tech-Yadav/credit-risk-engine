import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.preprocessing.preprocessor import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.decision.policy_engine import PolicyEngine
from src.explainability.shap_explainer import SHAPExplainer

# ── Page config ──
st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1f77b4; }
    .risk-card {
        padding: 20px; border-radius: 12px; text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .risk-low { background: linear-gradient(135deg, #d4edda, #c3e6cb); }
    .risk-medium { background: linear-gradient(135deg, #fff3cd, #ffeeba); }
    .risk-high { background: linear-gradient(135deg, #f8d7da, #f5c6cb); }
</style>
""",
    unsafe_allow_html=True,
)

# ── Load artifacts (cached) ──
@st.cache_resource
def load_artifacts():
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_model.json")
    preprocessor = Preprocessor.load("models/preprocessing.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, preprocessor, feature_names

model, preprocessor, feature_names = load_artifacts()
policy = PolicyEngine(low_threshold=0.10, medium_threshold=0.25)

# ── Sidebar inputs ──
st.sidebar.header("Applicants Information")

income = st.sidebar.number_input(
    "Annual Income", min_value=10000, value=500000, step=10000
)
loan_amount = st.sidebar.number_input(
    "Loan Amount", min_value=10000, value=300000, step=10000
)
annuity = st.sidebar.number_input(
    "Monthly EMI", min_value=1000, value=15000, step=500
)
age = st.sidebar.slider("Age (years)", 18, 70, 30)
employment_years = st.sidebar.slider("Employment (years)", 0, 40, 3)
family_members = st.sidebar.slider("Family Members", 1, 10, 2)

gender = st.sidebar.selectbox("Gender", ["M", "F"])
own_car = st.sidebar.selectbox("Owns Car?", ["Y", "N"])
own_realty = st.sidebar.selectbox("Owns Realty?", ["Y", "N"])
contract_type = st.sidebar.selectbox(
    "Contract Type", ["Cash loans", "Revolving loans"]
)
education = st.sidebar.selectbox(
    "Education",
    [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree",
    ],
)
family_status = st.sidebar.selectbox(
    "Family Status",
    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
)
occupation = st.sidebar.selectbox(
    "Occupation",
    [
        "Laborers",
        "Sales staff",
        "Core staff",
        "Managers",
        "Drivers",
        "High skill tech staff",
        "Accountants",
        "Medicine staff",
        "Security staff",
        "Cooking staff",
        "Cleaning staff",
        "Private service staff",
        "Low-skill Laborers",
        "Waiters/barmen staff",
        "Secretaries",
        "Realty agents",
        "HR staff",
        "IT staff",
        "Unknown",
    ],
)

# ── Build input row ──
input_dict = {
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": loan_amount,
    "AMT_ANNUITY": annuity,
    "DAYS_BIRTH": -int(age * 365.25),
    "DAYS_EMPLOYED": -int(employment_years * 365.25),
    "CNT_FAM_MEMBERS": float(family_members),
    "CODE_GENDER": gender,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "NAME_CONTRACT_TYPE": contract_type,
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family_status,
    "NAME_HOUSING_TYPE": "House / apartment",
    "OCCUPATION_TYPE": occupation,
    "EXT_SOURCE_1": np.nan,
    "EXT_SOURCE_2": np.nan,
    "EXT_SOURCE_3": np.nan,
}

# ── Main panel ──
st.markdown(
    '<p class="main-header">Credit Risk Decision Support System</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "Enter applicant details in the sidebar, then click **Analyze**."
)

if st.button("Analyze Application", type="primary", use_container_width=True):
    input_df = pd.DataFrame([input_dict])

    try:
        # Feature engineering
        input_fe = FeatureEngineer.create_features(input_df)

        # Align to preprocessor input columns
        input_cols = preprocessor.numerical_cols + preprocessor.categorical_cols
        full_input = pd.DataFrame(index=[0])
        for col in input_cols:
            if col in input_fe.columns:
                full_input[col] = input_fe[col].values
            else:
                if col in preprocessor.numerical_cols:
                    full_input[col] = 0.0
                else:
                    full_input[col] = "Unknown"

        # Preprocess
        X_proc = preprocessor.transform(full_input)

        # Predict
        pd_prob = float(model.predict_proba(X_proc)[0, 1])

        # Policy engine
        decision = policy.decide(pd_prob)

        # SHAP explanation
        try:
            explainer = SHAPExplainer(
                "models/xgboost_model.json", feature_names
            )
            local_exp = explainer.local_explanation(X_proc)
            top_pos = local_exp["top_positive"][:3]
            top_neg = local_exp["top_negative"][:3]
        except Exception as e:
            top_pos, top_neg = [], []
            st.warning(f"SHAP explanation unavailable: {e}")

        # ── Display Results ──
        col1, col2, col3 = st.columns(3)

        risk_class = (
            "risk-low"
            if decision["risk_band"] == "LOW"
            else "risk-medium"
            if decision["risk_band"] == "MEDIUM"
            else "risk-high"
        )

        with col1:
            st.markdown(
                f"""
                <div class="risk-card {risk_class}">
                    <h4>Default Probability</h4>
                    <h1>{decision['probability_of_default']:.1%}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="risk-card {risk_class}">
                    <h4>Risk Score</h4>
                    <h1>{decision['risk_score']} / 100</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="risk-card {risk_class}">
                    <h4>Risk Band</h4>
                    <h1>{decision['risk_band']}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Recommendation
        st.markdown("---")
        if "APPROVE" in decision["recommendation"]:
            st.success(f"### {decision['recommendation']}")
        elif "REVIEW" in decision["recommendation"]:
            st.warning(f"### {decision['recommendation']}")
        else:
            st.error(f"### {decision['recommendation']}")

        st.caption(
            "This is a decision-support recommendation, not a final lending decision."
        )

        # ── Explainability ──
        st.markdown("---")
        st.subheader("Why this decision? (SHAP Explainability)")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Risk-Increasing Factors**")
            if top_pos:
                for f in top_pos:
                    st.markdown(f"- **{f['feature']}**: {f['shap_value']:+.4f}")
            else:
                st.info("No major risk-increasing factors detected.")

        with col_b:
            st.markdown("**Risk-Reducing Factors**")
            if top_neg:
                for f in top_neg:
                    st.markdown(f"- **{f['feature']}**: {f['shap_value']:+.4f}")
            else:
                st.info("No major risk-reducing factors detected.")

        # ── Input summary ──
        with st.expander("View Applicant Summary"):
            st.dataframe(
                pd.DataFrame([input_dict]).T.rename(columns={0: "Value"})
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

else:
    st.info(
        "Fill in the applicant details in the sidebar and click **Analyze Application**."
    )

    # Show project architecture
    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown(
        """
        ```
        Customer Data -> Validation -> Preprocessing -> Feature Engineering
                 -> XGBoost Model -> P(Default)
                 -> Policy Engine (Risk Score + Band + Recommendation)
                 -> SHAP Explainability (Why this decision?)
                 -> Streamlit Dashboard
        ```
        """
    )

# ── Footer ──
st.markdown("---")
st.caption(
    "Credit Risk Engine v1.0 | XGBoost + SHAP + Streamlit | For educational/demo purposes only."
)
