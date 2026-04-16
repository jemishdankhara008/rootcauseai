"""Quick artifact check script for verifying required training outputs exist."""

import json
import os

print("Checking model artifacts...\n")

files = [
    "data/train_products.csv",
    "data/train_issues.csv",
    "models/product_model.pkl",
    "models/product_vectorizer.pkl",
    "models/issue_model.pkl",
    "models/issue_vectorizer.pkl",
    "models/metadata.json"
]

for file in files:
    print(f"{file}: {'FOUND' if os.path.exists(file) else 'MISSING'}")

if os.path.exists("models/metadata.json"):
    # Pretty-print metadata so version and training context are easy to inspect.
    print("\nMetadata:")
    with open("models/metadata.json", "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2))
