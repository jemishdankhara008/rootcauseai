"""OpenAI wrapper that turns structured model outputs into analyst-facing notes."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from src.config import RuntimeConfig

load_dotenv()


def _fallback_payload(reason: str, review_flag: bool) -> dict[str, Any]:
    """Return a stable response when LLM enrichment is unavailable."""
    return {
        "summary": "LLM enrichment is unavailable for this run.",
        "urgency": "Unknown",
        "explanation": reason,
        "case_note": "Generated from the rule-based fallback path.",
        "recommended_action": "Review the model outputs and retrieved cases manually.",
        "needs_human_review": True if review_flag else False,
        "llm_status": "fallback",
        "llm_error": reason,
    }


def analyze_complaint_with_openai(
    complaint_text: str,
    predicted_product: str,
    predicted_issue: str,
    confidence: float,
    similar_cases: list[dict[str, Any]],
    review_flag: bool,
) -> dict[str, Any]:
    """Ask the LLM for concise commentary while preserving a fixed JSON shape."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_payload("OPENAI_API_KEY is not configured.", review_flag)

    runtime = RuntimeConfig()
    client = OpenAI(api_key=api_key, timeout=runtime.openai_timeout_seconds)

    similar_cases_text = "\n".join(
        [
            f"- Similarity: {case['similarity']}, Product: {case['product']}, Issue: {case['issue']}"
            for case in similar_cases
        ]
    )

    messages = [
        {
            "role": "developer",
            "content": (
                "You are an AI analyst for RootCause AI. "
                "Return concise, professional outputs only. "
                "If confidence is low, explicitly recommend human review."
            ),
        },
        {
            "role": "user",
            "content": f"""
Complaint:
{complaint_text}

Predicted product:
{predicted_product}

Predicted issue:
{predicted_issue}

Overall confidence:
{confidence:.2%}

Needs human review:
{review_flag}

Top similar historical cases:
{similar_cases_text}
""",
        },
    ]

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "complaint_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "urgency": {"type": "string"},
                    "explanation": {"type": "string"},
                    "case_note": {"type": "string"},
                    "recommended_action": {"type": "string"},
                    "needs_human_review": {"type": "boolean"},
                },
                "required": [
                    "summary",
                    "urgency",
                    "explanation",
                    "case_note",
                    "recommended_action",
                    "needs_human_review",
                ],
                "additionalProperties": False,
            },
        },
    }

    try:
        response = client.chat.completions.create(
            model=runtime.openai_model,
            messages=messages,
            response_format=schema,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        parsed["llm_status"] = "success"
        parsed["llm_error"] = ""
        return parsed
    except Exception as exc:
        return _fallback_payload(f"OpenAI error: {exc}", review_flag)
