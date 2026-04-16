"""OpenAI wrapper that turns structured model outputs into analyst-facing notes."""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# A missing key will be handled later by the try/except fallback path.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_complaint_with_openai(
    complaint_text: str,
    predicted_product: str,
    predicted_issue: str,
    confidence: float,
    similar_cases: list,
    review_flag: bool
) -> dict:
    """Ask the LLM for concise commentary while preserving a fixed JSON shape."""

    # Only send compact summaries of retrieved matches to keep the prompt focused.
    similar_cases_text = "\n".join([
        f"- Similarity: {case['similarity']}, Product: {case['product']}, Issue: {case['issue']}"
        for case in similar_cases
    ])

    messages = [
        {
            "role": "developer",
            "content": (
                "You are an AI analyst for a complaint intelligence system called RootCause AI. "
                "Return concise, professional outputs only."
            )
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
"""
        }
    ]

    # Force a predictable schema so the UI does not need to parse free-form prose.
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
                    "needs_human_review": {"type": "boolean"}
                },
                "required": [
                    "summary",
                    "urgency",
                    "explanation",
                    "case_note",
                    "recommended_action",
                    "needs_human_review"
                ],
                "additionalProperties": False
            }
        }
    }

    try:
        # Structured output keeps the app response predictable for the UI layer.
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            response_format=schema
        )

        # The API returns the structured object as a JSON string in message content.
        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        # Fall back to safe defaults so classification results still render.
        return {
            "summary": "Unable to generate summary.",
            "urgency": "Unknown",
            "explanation": f"OpenAI error: {str(e)}",
            "case_note": "Unable to generate case note.",
            "recommended_action": "Manual review recommended.",
            "needs_human_review": True
        }
