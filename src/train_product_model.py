"""Train the product classifier used by the inference pipeline."""

import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import preprocess_dataframe

os.makedirs("models", exist_ok=True)

print("Loading product training data...")
df = pd.read_csv("data/train_products.csv")

# Text cleaning is shared so training and inference treat complaints consistently.
df = preprocess_dataframe(df, text_column="complaint_text")

X = df["clean_text"]
y = df["product"]

vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
X_vec = vectorizer.fit_transform(X)

# Stratification preserves the product distribution across train/test splits.
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic regression is a strong baseline for sparse TF-IDF text features.
model = LogisticRegression(max_iter=1200, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Product Model Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "models/product_model.pkl")
joblib.dump(vectorizer, "models/product_vectorizer.pkl")

print("Saved product model and vectorizer.")
