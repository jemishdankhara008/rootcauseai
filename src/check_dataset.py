"""Small inspection script for understanding the raw dataset before cleaning."""

from __future__ import annotations

import pandas as pd

from src.config import get_project_paths


def describe_raw_dataset() -> None:
    """Print a lightweight summary of the raw Kaggle dataset."""
    paths = get_project_paths()
    if not paths.raw_dataset.exists():
        raise FileNotFoundError(f"Raw dataset not found at {paths.raw_dataset}")

    df = pd.read_csv(paths.raw_dataset)

    print("Dataset shape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(20))


if __name__ == "__main__":
    describe_raw_dataset()
