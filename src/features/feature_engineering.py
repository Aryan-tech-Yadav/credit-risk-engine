import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Create financial ratio features from raw credit application data.
    Supports both full DataFrames and single-row prediction inputs.
    """

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Ensure calculations work for both DataFrame rows and single-row inputs
        for col in [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CNT_FAM_MEMBERS",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Monthly income
        df["MONTHLY_INCOME"] = df["AMT_INCOME_TOTAL"] / 12.0

        # Debt-to-Income Ratio
        monthly_income = df["MONTHLY_INCOME"].replace(0, np.nan)
        df["DTI"] = df["AMT_ANNUITY"] / monthly_income

        # Loan-to-Income Ratio
        income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
        df["LOAN_INCOME_RATIO"] = df["AMT_CREDIT"] / income

        # EMI-to-Income Ratio
        df["EMI_INCOME_RATIO"] = df["AMT_ANNUITY"] / income

        # Credit Term in months
        annuity = df["AMT_ANNUITY"].replace(0, np.nan)
        df["CREDIT_TERM"] = df["AMT_CREDIT"] / annuity

        # Age in years
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

        # Employment stability
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

        # Income per family member
        family_members = df["CNT_FAM_MEMBERS"].replace(0, np.nan)
        df["INCOME_PER_FAMILY"] = (
            df["AMT_INCOME_TOTAL"] / family_members
        )

        # External credit score average
        ext_cols = [
            c for c in df.columns
            if c.startswith("EXT_SOURCE_")
        ]

        if ext_cols:
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

        print(f"Created features. Shape: {df.shape}")

        return df