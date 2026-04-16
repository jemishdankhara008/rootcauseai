"""Centralized project configuration and path helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout used across training, inference, and interfaces."""

    root_dir: Path
    data_dir: Path
    models_dir: Path
    app_dir: Path
    reports_dir: Path
    reports_assets_dir: Path

    raw_dataset: Path
    cleaned_dataset: Path
    product_dataset: Path
    issue_dataset: Path

    product_model: Path
    product_vectorizer: Path
    issue_model: Path
    issue_vectorizer: Path
    retrieval_vectorizer: Path
    retrieval_matrix: Path
    retrieval_dataframe: Path
    metadata_path: Path
    metrics_path: Path
    legacy_model: Path
    legacy_vectorizer: Path


@dataclass(frozen=True)
class TrainingConfig:
    """Training and evaluation parameters shared by the pipeline."""

    random_state: int = 42
    top_product_count: int = 10
    min_issue_frequency: int = 80
    product_sample_size: int = 40000
    issue_sample_size: int = 25000
    test_size: float = 0.2
    product_max_features: int = 12000
    issue_max_features: int = 15000
    retrieval_max_features: int = 20000
    ngram_min: int = 1
    ngram_max: int = 2
    min_document_frequency: int = 2
    logistic_max_iter: int = 1200
    low_confidence_default: float = 0.45
    high_confidence_default: float = 0.70


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime behavior toggles for interfaces and optional integrations."""

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
    top_k_similar_cases: int = int(os.getenv("TOP_K_SIMILAR_CASES", "3"))


def get_project_paths() -> ProjectPaths:
    """Return resolved project paths rooted at the repository directory."""
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    models_dir = root_dir / "models"
    app_dir = root_dir / "app"
    reports_dir = root_dir / "reports"
    reports_assets_dir = reports_dir / "assets"

    return ProjectPaths(
        root_dir=root_dir,
        data_dir=data_dir,
        models_dir=models_dir,
        app_dir=app_dir,
        reports_dir=reports_dir,
        reports_assets_dir=reports_assets_dir,
        raw_dataset=data_dir / "rows.csv",
        cleaned_dataset=data_dir / "clean_complaints.csv",
        product_dataset=data_dir / "train_products.csv",
        issue_dataset=data_dir / "train_issues.csv",
        product_model=models_dir / "product_model.pkl",
        product_vectorizer=models_dir / "product_vectorizer.pkl",
        issue_model=models_dir / "issue_model.pkl",
        issue_vectorizer=models_dir / "issue_vectorizer.pkl",
        retrieval_vectorizer=models_dir / "retrieval_vectorizer.pkl",
        retrieval_matrix=models_dir / "retrieval_matrix.pkl",
        retrieval_dataframe=models_dir / "retrieval_df.pkl",
        metadata_path=models_dir / "metadata.json",
        metrics_path=models_dir / "metrics.json",
        legacy_model=models_dir / "rootcause_model.pkl",
        legacy_vectorizer=models_dir / "tfidf_vectorizer.pkl",
    )


def training_config_dict() -> dict:
    """Expose the training configuration as a serializable dictionary."""
    return asdict(TrainingConfig())
