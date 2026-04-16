"""Legacy single-model training script for issue classification experiments."""

import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import preprocess_dataframe


def train():
    """Train the original issue classifier and save its artifacts."""
    if not os.path.exists("data/clean_complaints.csv"):
        raise FileNotFoundError(
            "clean_complaints.csv not found. Run prepare_dataset.py first."
        )

    print("Loading cleaned dataset...")
    df = pd.read_csv("data/clean_complaints.csv")

    print("Dataset shape:", df.shape)

    # Final safeguard: stratified splitting requires at least two samples per label.
    issue_counts = df["issue"].value_counts()
    valid_issues = issue_counts[issue_counts >= 2].index
    df = df[df["issue"].isin(valid_issues)].copy()

    print("Dataset shape after final label filtering:", df.shape)

    print("Preprocessing text...")
    df = preprocess_dataframe(df, text_column="complaint_text")

    X = df["clean_text"]
    y = df["issue"]

    print("Creating TF-IDF features...")
    # Unigrams and bigrams capture both single keywords and short complaint phrases.
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2
    )
    X_vec = vectorizer.fit_transform(X)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Logistic Regression model...")
    # Balanced class weights reduce bias toward the most frequent issue labels.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rootcause_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("\nSaved model to models/rootcause_model.pkl")
    print("Saved vectorizer to models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    train()
