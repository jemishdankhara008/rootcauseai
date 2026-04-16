"""End-to-end inference entry point for complaint classification and enrichment."""

import joblib
import numpy as np

from preprocess import clean_text
from retrieve_similar import get_similar_complaints
from openai_helper import analyze_complaint_with_openai

# Load all serialized artifacts once at import time so UI requests stay fast.
product_model = joblib.load("models/product_model.pkl")
product_vectorizer = joblib.load("models/product_vectorizer.pkl")

issue_model = joblib.load("models/issue_model.pkl")
issue_vectorizer = joblib.load("models/issue_vectorizer.pkl")


def predict_complaint(text: str) -> dict:
    """Predict product/issue labels, score confidence, and attach LLM insights."""
    # Normalize the raw complaint before sending it through any vectorizer.
    cleaned = clean_text(text)

    # Product prediction
    # The product model estimates the broad complaint category first.
    product_vec = product_vectorizer.transform([cleaned])
    product_pred = product_model.predict(product_vec)[0]
    product_probs = product_model.predict_proba(product_vec)[0]
    product_conf = float(np.max(product_probs))

    # Issue prediction
    # The issue model predicts the finer-grained problem within the product area.
    issue_vec = issue_vectorizer.transform([cleaned])
    issue_pred = issue_model.predict(issue_vec)[0]
    issue_probs = issue_model.predict_proba(issue_vec)[0]
    issue_conf = float(np.max(issue_probs))

    # Top-3 predictions
    # Keep the top alternatives so the UI can show near-miss classes for review.
    top3_prod_idx = np.argsort(product_probs)[::-1][:3]
    top3_products = [
        (product_model.classes_[i], float(product_probs[i]))
        for i in top3_prod_idx
    ]

    top3_issue_idx = np.argsort(issue_probs)[::-1][:3]
    top3_issues = [
        (issue_model.classes_[i], float(issue_probs[i]))
        for i in top3_issue_idx
    ]

    # Retrieval adds historical context so the final confidence is not based on
    # classifier probabilities alone.
    similar_cases = get_similar_complaints(text, top_k=3)
    avg_similarity = sum(case["similarity"] for case in similar_cases) / len(similar_cases)

    # Blend the two classifiers with similarity search into one review score.
    overall_confidence = (product_conf * 0.35) + (issue_conf * 0.45) + (avg_similarity * 0.20)

    review_flag = False
    confidence_band = "High"

    # Low-scoring predictions are intentionally downgraded to force manual review.
    if overall_confidence < 0.45:
        review_flag = True
        confidence_band = "Low"
    elif overall_confidence < 0.65:
        confidence_band = "Medium"

    # Confidence-aware final displayed labels
    final_product = product_pred
    final_issue = issue_pred

    # Replace risky labels with safe placeholders when the score is too weak.
    if overall_confidence < 0.45:
        final_product = "Uncertain Category"
        final_issue = "Needs Reclassification"

    # Ask the LLM for human-friendly narrative fields after structured prediction
    # is complete, so the ML outputs remain the system of record.
    llm_result = analyze_complaint_with_openai(
        complaint_text=text,
        predicted_product=product_pred,
        predicted_issue=issue_pred,
        confidence=overall_confidence,
        similar_cases=similar_cases,
        review_flag=review_flag
    )

    return {
        "predicted_product": final_product,
        "predicted_issue": final_issue,
        "raw_product_prediction": product_pred,
        "raw_issue_prediction": issue_pred,
        "product_confidence": round(product_conf, 4),
        "issue_confidence": round(issue_conf, 4),
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
        "needs_human_review": llm_result.get("needs_human_review", review_flag)
    }


if __name__ == "__main__":
    # Handy smoke test for local development.
    sample = "A debt collector keeps contacting me about a debt that is not mine and refuses to provide validation."
    result = predict_complaint(sample)

    for key, value in result.items():
        print(f"{key}: {value}")
