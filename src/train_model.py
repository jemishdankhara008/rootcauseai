"""Train the complete RootCause AI pipeline end-to-end."""

from __future__ import annotations

from src.build_retrieval_index import build_retrieval_index
from src.evaluate_models import evaluate_pipeline
from src.prepare_dataset import prepare_datasets
from src.train_issue_model import train_issue_model
from src.train_product_model import train_product_model


def train_pipeline() -> dict:
    """Prepare data, train models, build retrieval, and evaluate the pipeline."""
    prepare_summary = prepare_datasets()
    product_artifacts = train_product_model()
    issue_artifacts = train_issue_model()
    retrieval_summary = build_retrieval_index()
    evaluation_summary = evaluate_pipeline(
        product_artifacts=product_artifacts,
        issue_artifacts=issue_artifacts,
        retrieval_summary=retrieval_summary,
    )

    return {
        "prepare_summary": prepare_summary,
        "retrieval_summary": retrieval_summary,
        "evaluation_summary": evaluation_summary,
    }


if __name__ == "__main__":
    train_pipeline()
