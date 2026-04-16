"""Retrieve historically similar complaints using TF-IDF cosine similarity."""

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import clean_text

# Load retrieval artifacts once so repeated searches stay inexpensive.
retrieval_vectorizer = joblib.load("models/retrieval_vectorizer.pkl")
retrieval_matrix = joblib.load("models/retrieval_matrix.pkl")
retrieval_df = pd.read_pickle("models/retrieval_df.pkl")


def get_similar_complaints(text: str, top_k: int = 3):
    """Return the most similar complaints with labels and rounded similarity."""
    cleaned = clean_text(text)
    query_vec = retrieval_vectorizer.transform([cleaned])

    # Compare the incoming complaint against every indexed historical complaint.
    similarities = cosine_similarity(query_vec, retrieval_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = retrieval_df.iloc[idx]
        # The response format is designed to be consumed directly by the UI.
        results.append({
            "complaint_text": row["complaint_text"],
            "issue": row["issue"],
            "product": row["product"],
            "similarity": round(float(similarities[idx]), 4)
        })

    return results
