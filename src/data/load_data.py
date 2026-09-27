import pandas as pd
from pathlib import Path


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the raw application_train.csv from Home Credit dataset."""
    path = Path(config["data"]["raw_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from Kaggle first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_info(df: pd.DataFrame) -> None:
    """Print quick info about the dataframe."""
    print("Shape:", df.shape)
    print("\nTarget distribution:")
    print(df["TARGET"].value_counts(normalize=True))
    print("\nMissing values (top 10):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
