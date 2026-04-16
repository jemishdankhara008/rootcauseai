"""Train the issue classifier used by the inference pipeline."""

import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import preprocess_dataframe

os.makedirs("models", exist_ok=True)

print("Loading issue training data...")
df = pd.read_csv("data/train_issues.csv")

# Keep preprocessing identical to the inference path for stable features.
df = preprocess_dataframe(df, text_column="complaint_text")

X = df["clean_text"]
y = df["issue"]

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
X_vec = vectorizer.fit_transform(X)

# Stratification helps evaluation reflect the production label mix.
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Class balancing improves recall for less common issue types.
model = LogisticRegression(max_iter=1200, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Issue Model Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, "models/issue_model.pkl")
joblib.dump(vectorizer, "models/issue_vectorizer.pkl")

print("Saved issue model and vectorizer.")
