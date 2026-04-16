"""Build and save the TF-IDF retrieval index used for nearest-neighbor lookup."""

import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_dataframe

os.makedirs("models", exist_ok=True)

print("Loading issue training data...")
df = pd.read_csv("data/train_issues.csv")

# Reuse the same cleaning logic as the classifiers so feature spaces stay aligned.
df = preprocess_dataframe(df, text_column="complaint_text")

retrieval_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

# Persist both the sparse matrix and the source rows needed to explain matches.
X_retrieval = retrieval_vectorizer.fit_transform(df["clean_text"])

# Save every artifact needed to reproduce retrieval at inference time.
joblib.dump(retrieval_vectorizer, "models/retrieval_vectorizer.pkl")
joblib.dump(X_retrieval, "models/retrieval_matrix.pkl")
df.to_pickle("models/retrieval_df.pkl")

print("Saved retrieval index artifacts:")
print("- models/retrieval_vectorizer.pkl")
print("- models/retrieval_matrix.pkl")
print("- models/retrieval_df.pkl")
