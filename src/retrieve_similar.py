"""Retrieve historically similar complaints using TF-IDF cosine similarity."""

from __future__ import annotations

from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from src.artifacts import ArtifactBundle, load_artifacts
from src.preprocess import clean_text


def get_similar_complaints(
    text: str,
    top_k: int = 3,
    bundle: ArtifactBundle | None = None,
) -> list[dict[str, Any]]:
    """Return the most similar complaints with labels and rounded similarity."""
    artifacts = bundle or load_artifacts()
    cleaned = clean_text(text)
    query_vec = artifacts.retrieval_vectorizer.transform([cleaned])

    similarities = cosine_similarity(query_vec, artifacts.retrieval_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for idx in top_indices:
        row = artifacts.retrieval_df.iloc[idx]
        results.append(
            {
                "complaint_text": row["complaint_text"],
                "issue": row["issue"],
                "product": row["product"],
                "similarity": round(float(similarities[idx]), 4),
            }
        )

    return results
