"""Evaluate trained artifacts and write reproducible model metrics."""

from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TrainingConfig, get_project_paths, training_config_dict
from src.predict import _classify_confidence, _score_issue_candidates
from src.train_issue_model import IssueTrainingArtifacts
from src.train_product_model import ProductTrainingArtifacts


def _derive_confidence_thresholds(scores: list[float], is_correct: list[bool], config: TrainingConfig) -> dict[str, float]:
    """Derive low/high thresholds from observed end-to-end correctness."""
    correct_scores = [score for score, correct in zip(scores, is_correct) if correct]
    incorrect_scores = [score for score, correct in zip(scores, is_correct) if not correct]

    if not correct_scores or not incorrect_scores:
        return {
            "low": config.low_confidence_default,
            "high": config.high_confidence_default,
        }

    low_threshold = round(float(np.quantile(incorrect_scores, 0.70)), 4)
    high_threshold = round(float(np.quantile(correct_scores, 0.40)), 4)
    if high_threshold <= low_threshold:
        low_threshold = config.low_confidence_default
        high_threshold = config.high_confidence_default

    return {"low": low_threshold, "high": high_threshold}


def evaluate_pipeline(
    product_artifacts: ProductTrainingArtifacts,
    issue_artifacts: IssueTrainingArtifacts,
    retrieval_summary: dict[str, int] | None = None,
    config: TrainingConfig | None = None,
) -> dict:
    """Evaluate the end-to-end pipeline and persist metrics and metadata."""
    config = config or TrainingConfig()
    paths = get_project_paths()

    product_test = product_artifacts.test_df.copy()
    issue_test = issue_artifacts.test_df.copy()
    retrieval_vectorizer = TfidfVectorizer(
        max_features=config.retrieval_max_features,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_document_frequency,
    )
    retrieval_matrix = retrieval_vectorizer.fit_transform(issue_artifacts.train_df["clean_text"])

    product_x = product_artifacts.vectorizer.transform(product_test["clean_text"])
    product_predictions = product_artifacts.model.predict(product_x)
    product_probabilities = product_artifacts.model.predict_proba(product_x)
    product_accuracy = float(accuracy_score(product_test["product"], product_predictions))

    end_to_end_scores: list[float] = []
    end_to_end_correct: list[bool] = []
    issue_correct_count = 0
    joint_correct_count = 0

    provisional_metadata = {
        "product_issue_priors": issue_artifacts.compatibility_map,
        "confidence_thresholds": {
            "low": config.low_confidence_default,
            "high": config.high_confidence_default,
        },
    }

    issue_x = issue_artifacts.vectorizer.transform(issue_test["clean_text"])
    raw_issue_probabilities = issue_artifacts.model.predict_proba(issue_x)

    for row, raw_probs in zip(issue_test.itertuples(index=False), raw_issue_probabilities):
        product_vec = product_artifacts.vectorizer.transform([row.clean_text])
        product_probs = product_artifacts.model.predict_proba(product_vec)[0]
        product_idx = int(np.argmax(product_probs))
        product_prediction = str(product_artifacts.model.classes_[product_idx])
        product_conf = float(product_probs[product_idx])
        top_two_product_probs = product_probs[np.argsort(product_probs)[::-1][:2]]
        product_margin = float(np.ptp(top_two_product_probs)) if len(top_two_product_probs) == 2 else product_conf

        predicted_issue, issue_conf, _, compatibility_score = _score_issue_candidates(
            predicted_product=product_prediction,
            issue_classes=issue_artifacts.model.classes_,
            issue_probabilities=raw_probs,
            metadata=provisional_metadata,
        )

        query_vec = retrieval_vectorizer.transform([row.clean_text])
        similarities = cosine_similarity(query_vec, retrieval_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:3]
        avg_similarity = float(np.mean(similarities[top_indices])) if len(top_indices) else 0.0

        overall_confidence = float(
            (product_conf * 0.35)
            + (issue_conf * 0.30)
            + (avg_similarity * 0.15)
            + (compatibility_score * 0.10)
            + (product_margin * 0.10)
        )

        issue_correct = predicted_issue == row.issue
        joint_correct = issue_correct and product_prediction == row.product

        issue_correct_count += int(issue_correct)
        joint_correct_count += int(joint_correct)
        end_to_end_scores.append(overall_confidence)
        end_to_end_correct.append(joint_correct)

    thresholds = _derive_confidence_thresholds(end_to_end_scores, end_to_end_correct, config)
    confidence_bands = [
        _classify_confidence(score, {"confidence_thresholds": thresholds})[0]
        for score in end_to_end_scores
    ]

    band_summary: dict[str, dict[str, float]] = {}
    for band_name in ("Low", "Medium", "High"):
        indices = [idx for idx, band in enumerate(confidence_bands) if band == band_name]
        if not indices:
            band_summary[band_name] = {"count": 0, "joint_accuracy": 0.0}
            continue
        joint_accuracy = sum(end_to_end_correct[idx] for idx in indices) / len(indices)
        band_summary[band_name] = {
            "count": len(indices),
            "joint_accuracy": round(float(joint_accuracy), 4),
        }

    metrics = {
        "product_model": product_artifacts.metrics,
        "issue_model": issue_artifacts.metrics,
        "end_to_end": {
            "product_accuracy_on_product_holdout": round(product_accuracy, 4),
            "issue_accuracy": round(issue_correct_count / len(issue_test), 4),
            "joint_accuracy": round(joint_correct_count / len(issue_test), 4),
            "confidence_band_summary": band_summary,
        },
        "retrieval": retrieval_summary or {},
    }

    metadata = {
        "training_config": training_config_dict(),
        "product_labels": sorted(product_artifacts.processed_df["product"].unique().tolist()),
        "issue_labels": sorted(issue_artifacts.processed_df["issue"].unique().tolist()),
        "product_train_rows": int(product_artifacts.processed_df.shape[0]),
        "issue_train_rows": int(issue_artifacts.processed_df.shape[0]),
        "product_issue_priors": issue_artifacts.compatibility_map,
        "confidence_thresholds": thresholds,
    }

    with paths.metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved evaluation metrics and metadata.")
    print(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "metadata": metadata}


if __name__ == "__main__":
    print(
        "Run `python -m src.train_model` to prepare datasets, train models, "
        "build retrieval, and write metrics in one step."
    )
