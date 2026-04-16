"""Build and save the TF-IDF retrieval index used for nearest-neighbor lookup."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TrainingConfig, get_project_paths
from src.preprocess import preprocess_dataframe


def build_retrieval_index(
    source_df: pd.DataFrame | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, int]:
    """Build and persist retrieval artifacts from the issue dataset."""
    config = config or TrainingConfig()
    paths = get_project_paths()
    paths.models_dir.mkdir(parents=True, exist_ok=True)

    if source_df is None:
        if not paths.issue_dataset.exists():
            raise FileNotFoundError(
                f"Missing {paths.issue_dataset}. Run `python -m src.prepare_dataset` first."
            )
        source_df = pd.read_csv(paths.issue_dataset)

    processed_df = preprocess_dataframe(source_df, text_column="complaint_text")
    vectorizer = TfidfVectorizer(
        max_features=config.retrieval_max_features,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_document_frequency,
    )
    retrieval_matrix = vectorizer.fit_transform(processed_df["clean_text"])

    joblib.dump(vectorizer, paths.retrieval_vectorizer)
    joblib.dump(retrieval_matrix, paths.retrieval_matrix)
    processed_df.to_pickle(paths.retrieval_dataframe)

    summary = {
        "rows": int(processed_df.shape[0]),
        "columns": int(retrieval_matrix.shape[1]),
    }
    print(f"Saved retrieval artifacts to {paths.models_dir}")
    return summary


if __name__ == "__main__":
    build_retrieval_index()
