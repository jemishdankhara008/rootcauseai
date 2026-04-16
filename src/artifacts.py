"""Helpers for loading serialized model artifacts lazily and safely."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from src.config import get_project_paths


@dataclass(frozen=True)
class ArtifactBundle:
    """All persisted assets required for end-to-end inference."""

    product_model: Any
    product_vectorizer: Any
    issue_model: Any
    issue_vectorizer: Any
    retrieval_vectorizer: Any
    retrieval_matrix: Any
    retrieval_df: pd.DataFrame
    metadata: dict


@lru_cache(maxsize=1)
def load_artifacts() -> ArtifactBundle:
    """Load trained artifacts once per process with clear validation errors."""
    paths = get_project_paths()
    required_paths = {
        "product model": paths.product_model,
        "product vectorizer": paths.product_vectorizer,
        "issue model": paths.issue_model,
        "issue vectorizer": paths.issue_vectorizer,
        "retrieval vectorizer": paths.retrieval_vectorizer,
        "retrieval matrix": paths.retrieval_matrix,
        "retrieval dataframe": paths.retrieval_dataframe,
    }

    missing = [f"{name} ({path})" for name, path in required_paths.items() if not path.exists()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "Required project artifacts are missing.\n"
            "Run `python -m src.train_model` from the repository root first.\n"
            f"Missing:\n{joined}"
        )

    metadata: dict = {}
    if paths.metadata_path.exists():
        with paths.metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

    return ArtifactBundle(
        product_model=joblib.load(paths.product_model),
        product_vectorizer=joblib.load(paths.product_vectorizer),
        issue_model=joblib.load(paths.issue_model),
        issue_vectorizer=joblib.load(paths.issue_vectorizer),
        retrieval_vectorizer=joblib.load(paths.retrieval_vectorizer),
        retrieval_matrix=joblib.load(paths.retrieval_matrix),
        retrieval_df=pd.read_pickle(paths.retrieval_dataframe),
        metadata=metadata,
    )
