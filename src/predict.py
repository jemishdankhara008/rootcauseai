"""End-to-end inference entry point for complaint classification and enrichment."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.artifacts import ArtifactBundle, load_artifacts
from src.config import RuntimeConfig
from src.openai_helper import analyze_complaint_with_openai
from src.preprocess import clean_text
from src.retrieve_similar import get_similar_complaints


def _top_k_predictions(classes: np.ndarray, probabilities: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Return the top-k labels and probabilities sorted from strongest to weakest."""
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [(str(classes[index]), float(probabilities[index])) for index in top_indices]


def _score_issue_candidates(
    predicted_product: str,
    issue_classes: np.ndarray,
    issue_probabilities: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[str, float, list[tuple[str, float]], float]:
    """Adjust issue probabilities using product-specific compatibility priors."""
    compatibility_map = metadata.get("product_issue_priors", {})
    product_issue_priors = compatibility_map.get(predicted_product, {})
    adjusted_scores: list[tuple[str, float]] = []

    for issue_name, raw_probability in zip(issue_classes, issue_probabilities):
        prior = float(product_issue_priors.get(str(issue_name), 0.0))
        compatibility_boost = 0.30 + prior if prior > 0 else 0.05
        adjusted_scores.append((str(issue_name), float(raw_probability) * compatibility_boost))

    adjusted_scores.sort(key=lambda item: item[1], reverse=True)
    total_adjusted = sum(score for _, score in adjusted_scores) or 1.0
    normalized_scores = [
        (issue_name, round(score / total_adjusted, 6)) for issue_name, score in adjusted_scores[:3]
    ]

    final_issue, best_adjusted_score = adjusted_scores[0]
    compatibility_score = min(
        1.0,
        float(product_issue_priors.get(final_issue, 0.0) * len(product_issue_priors) * 0.5),
    )
    issue_confidence = float(best_adjusted_score / total_adjusted)
    return final_issue, issue_confidence, normalized_scores, compatibility_score


def _classify_confidence(overall_confidence: float, metadata: dict[str, Any]) -> tuple[str, bool]:
    """Map the overall confidence score into a reviewer-facing band."""
    thresholds = metadata.get("confidence_thresholds", {})
    low_threshold = float(thresholds.get("low", 0.45))
    high_threshold = float(thresholds.get("high", 0.70))

    if overall_confidence < low_threshold:
        return "Low", True
    if overall_confidence < high_threshold:
        return "Medium", False
    return "High", False


def predict_complaint(text: str, include_llm: bool = True) -> dict[str, Any]:
    """Predict product and issue labels, retrieve context, and add analyst notes."""
    if not text or not text.strip():
        raise ValueError("Complaint text must be a non-empty string.")

    artifacts: ArtifactBundle = load_artifacts()
    runtime = RuntimeConfig()
    cleaned = clean_text(text)

    product_vec = artifacts.product_vectorizer.transform([cleaned])
    product_probabilities = artifacts.product_model.predict_proba(product_vec)[0]
    top3_products = _top_k_predictions(artifacts.product_model.classes_, product_probabilities)
    product_pred, product_conf = top3_products[0]
    top_two_product_probs = product_probabilities[np.argsort(product_probabilities)[::-1][:2]]
    product_margin = float(np.ptp(top_two_product_probs)) if len(top_two_product_probs) == 2 else product_conf

    issue_vec = artifacts.issue_vectorizer.transform([cleaned])
    raw_issue_probabilities = artifacts.issue_model.predict_proba(issue_vec)[0]
    raw_top_issues = _top_k_predictions(artifacts.issue_model.classes_, raw_issue_probabilities)
    raw_issue_pred = raw_top_issues[0][0]

    final_issue, issue_conf, top3_issues, compatibility_score = _score_issue_candidates(
        predicted_product=product_pred,
        issue_classes=artifacts.issue_model.classes_,
        issue_probabilities=raw_issue_probabilities,
        metadata=artifacts.metadata,
    )

    similar_cases = get_similar_complaints(text, top_k=runtime.top_k_similar_cases, bundle=artifacts)
    avg_similarity = float(
        sum(case["similarity"] for case in similar_cases) / len(similar_cases)
    ) if similar_cases else 0.0

    overall_confidence = float(
        (product_conf * 0.35)
        + (issue_conf * 0.30)
        + (avg_similarity * 0.15)
        + (compatibility_score * 0.10)
        + (product_margin * 0.10)
    )

    confidence_band, review_flag = _classify_confidence(overall_confidence, artifacts.metadata)
    final_product = product_pred if confidence_band != "Low" else "Uncertain Category"
    displayed_issue = final_issue if confidence_band != "Low" else "Needs Reclassification"

    llm_result = (
        analyze_complaint_with_openai(
            complaint_text=text,
            predicted_product=product_pred,
            predicted_issue=final_issue,
            confidence=overall_confidence,
            similar_cases=similar_cases,
            review_flag=review_flag,
        )
        if include_llm
        else {
            "summary": "LLM enrichment skipped for this run.",
            "urgency": "Unknown",
            "explanation": "Prediction generated without calling OpenAI.",
            "case_note": "Local-only execution path.",
            "recommended_action": "Review retrieved examples if additional context is needed.",
            "needs_human_review": review_flag,
            "llm_status": "skipped",
            "llm_error": "",
        }
    )

    return {
        "predicted_product": final_product,
        "predicted_issue": displayed_issue,
        "raw_product_prediction": product_pred,
        "raw_issue_prediction": raw_issue_pred,
        "product_aware_issue_prediction": final_issue,
        "product_confidence": round(product_conf, 4),
        "issue_confidence": round(issue_conf, 4),
        "product_margin": round(product_margin, 4),
        "compatibility_score": round(compatibility_score, 4),
        "retrieval_similarity": round(avg_similarity, 4),
        "overall_confidence": round(overall_confidence, 4),
        "confidence_band": confidence_band,
        "review_flag": review_flag,
        "similar_cases": similar_cases,
        "top3_products": top3_products,
        "top3_issues": top3_issues,
        "summary": llm_result.get("summary", ""),
        "urgency": llm_result.get("urgency", ""),
        "explanation": llm_result.get("explanation", ""),
        "case_note": llm_result.get("case_note", ""),
        "recommended_action": llm_result.get("recommended_action", ""),
        "needs_human_review": llm_result.get("needs_human_review", review_flag),
        "llm_status": llm_result.get("llm_status", "unknown"),
        "llm_error": llm_result.get("llm_error", ""),
    }


def main() -> None:
    """CLI entry point for quick local inference smoke tests."""
    parser = argparse.ArgumentParser(description="Run RootCause AI on a complaint.")
    parser.add_argument("--text", required=True, help="Complaint text to analyze.")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the optional OpenAI narrative generation step.",
    )
    args = parser.parse_args()

    result = predict_complaint(args.text, include_llm=not args.skip_llm)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
