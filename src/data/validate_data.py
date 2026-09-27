import pandas as pd
import numpy as np


class DataValidator:
    """Basic validation for credit application data."""

    REQUIRED_COLS = ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]

    @staticmethod
    def validate(df: pd.DataFrame) -> dict:
        report = {"passed": True, "issues": []}

        missing_cols = [c for c in DataValidator.REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            report["passed"] = False
            report["issues"].append(f"Missing required columns: {missing_cols}")

        for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    report["issues"].append(f"{col}: {null_count} nulls")

        if "AMT_INCOME_TOTAL" in df.columns:
            neg_income = (df["AMT_INCOME_TOTAL"] <= 0).sum()
            if neg_income > 0:
                report["issues"].append(
                    f"AMT_INCOME_TOTAL: {neg_income} non-positive values"
                )

        if "TARGET" in df.columns:
            unique_targets = df["TARGET"].unique()
            if not set(unique_targets).issubset({0, 1}):
                report["passed"] = False
                report["issues"].append(
                    f"TARGET has unexpected values: {unique_targets}"
                )

        return report
