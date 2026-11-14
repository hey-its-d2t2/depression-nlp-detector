"""Pydantic request/response schemas for the prediction API."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The text to analyse for depression signals.",
        examples=["I have been feeling very low lately and can't see any hope."],
    )
    threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (default 0.5). Lower → higher recall.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be blank or whitespace only.")
        return v.strip()


class BatchPredictionRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyse (max 100 per request).",
    )
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("texts list must not be empty")
        return [t.strip() for t in v if t.strip()]


class ProbabilityBreakdown(BaseModel):
    non_depressed: float
    depressed: float


class PredictionResponse(BaseModel):
    text_preview: str = Field(description="First 100 chars of the input text.")
    label: int = Field(description="Predicted class: 0=non-depressed, 1=depressed.")
    label_text: str = Field(description="Human-readable label.")
    confidence: float = Field(description="Confidence score for the predicted class.")
    probabilities: ProbabilityBreakdown
    threshold_used: float
    model_name: str
    disclaimer: str = (
        "This tool is for research purposes only and is NOT a substitute "
        "for professional mental health evaluation."
    )


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
    model_name: str
