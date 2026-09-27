import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path


class Preprocessor:
    """
    Build and apply preprocessing pipeline.
    Handles: imputation, scaling (numerical), one-hot encoding (categorical).
    """

    def __init__(self, numerical_cols: list, categorical_cols: list):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.pipeline = None
        self.feature_names = None

    def build(self):
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.pipeline = ColumnTransformer(
            [
                ("num", num_pipe, self.numerical_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ]
        )
        return self

    def fit_transform(self, X: pd.DataFrame):
        X_transformed = self.pipeline.fit_transform(X)
        self.feature_names = self._get_feature_names()
        return pd.DataFrame(
            X_transformed, columns=self.feature_names, index=X.index
        )

    def transform(self, X: pd.DataFrame):
        X_transformed = self.pipeline.transform(X)
        return pd.DataFrame(
            X_transformed, columns=self.feature_names, index=X.index
        )

    def _get_feature_names(self):
        names = list(self.numerical_cols)
        ohe = self.pipeline.named_transformers_["cat"].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(self.categorical_cols)
        names.extend(cat_names)
        return names

    def save(self, path: str = "models/preprocessing.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "numerical_cols": self.numerical_cols,
                "categorical_cols": self.categorical_cols,
            },
            path,
        )
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str = "models/preprocessing.pkl"):
        data = joblib.load(path)
        obj = Preprocessor(
            data["numerical_cols"], data["categorical_cols"]
        )
        obj.pipeline = data["pipeline"]
        obj.feature_names = data["feature_names"]
        return obj
