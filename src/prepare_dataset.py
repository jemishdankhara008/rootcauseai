"""Prepare filtered training datasets from the raw complaint export."""

import os
import pandas as pd

os.makedirs("data", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/rows.csv", low_memory=False)

df = df[
    ["Consumer complaint narrative", "Issue", "Product"]
].dropna()

df = df.rename(columns={
    "Consumer complaint narrative": "complaint_text",
    "Issue": "issue",
    "Product": "product"
})

# Drop rows that survived dropna but still contain empty-looking strings.
df = df[df["complaint_text"].astype(str).str.strip() != ""]
df = df[df["issue"].astype(str).str.strip() != ""]
df = df[df["product"].astype(str).str.strip() != ""]

print("After cleaning:", df.shape)

# Limit the label space so the product model has enough support per class.
top_products = df["product"].value_counts().head(10).index
df_products = df[df["product"].isin(top_products)].copy()

# Downsample very large datasets to keep training fast and reproducible.
product_sample_size = min(40000, len(df_products))
df_products = df_products.sample(n=product_sample_size, random_state=42)

# Build the issue dataset from the filtered product set to keep the pipeline aligned.
issue_counts = df_products["issue"].value_counts()
top_issues = issue_counts[issue_counts >= 80].index

df_issues = df_products[df_products["issue"].isin(top_issues)].copy()
# Use a second sample cap so the issue classifier remains tractable.
issue_sample_size = min(25000, len(df_issues))
df_issues = df_issues.sample(n=issue_sample_size, random_state=42)

df_products.to_csv("data/train_products.csv", index=False)
df_issues.to_csv("data/train_issues.csv", index=False)

print("Saved:")
print("data/train_products.csv")
print("data/train_issues.csv")
print("\nTop products:")
print(df_products["product"].value_counts().head(10))
print("\nTop issues:")
print(df_issues["issue"].value_counts().head(15))
