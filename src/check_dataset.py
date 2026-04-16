"""Small inspection script for understanding the raw dataset before cleaning."""

import pandas as pd

df = pd.read_csv("data/rows.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))
