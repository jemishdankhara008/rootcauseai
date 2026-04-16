"""FastAPI interface for RootCause AI."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_complaint

app = FastAPI(
    title="RootCause AI API",
    description="REST API for complaint classification, retrieval, and optional LLM enrichment.",
    version="1.0.0",
)


class ComplaintRequest(BaseModel):
    """Payload accepted by the prediction endpoint."""

    complaint_text: str = Field(..., min_length=10, description="Complaint narrative to analyze.")
    include_llm: bool = Field(default=True, description="Whether to call OpenAI for narrative insights.")


class ComplaintResponse(BaseModel):
    """Typed API response envelope."""

    result: dict[str, Any]


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Simple readiness endpoint for local validation."""
    return {"status": "ok"}


@app.post("/predict", response_model=ComplaintResponse)
def predict(request: ComplaintRequest) -> ComplaintResponse:
    """Run the full inference pipeline on a complaint."""
    try:
        result = predict_complaint(request.complaint_text, include_llm=request.include_llm)
        return ComplaintResponse(result=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
