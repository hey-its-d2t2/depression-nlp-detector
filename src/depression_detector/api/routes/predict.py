"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from depression_detector.api.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    ProbabilityBreakdown,
)
from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predictions"])

# Global model registry — populated on startup by main.py
_model = None


def get_model():
    if _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run the training pipeline first.",
        )
    return _model


def set_model(model) -> None:
    global _model
    _model = model


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict depression signals in a single text",
)
async def predict_single(
    request: PredictionRequest,
    model=Depends(get_model),
) -> PredictionResponse:
    logger.info(f"Single prediction request — text_len={len(request.text)}")
    try:
        result = model.predict_one(request.text)
    except Exception as exc:
        logger.exception(f"Prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    return PredictionResponse(
        text_preview=request.text[:100],
        label=result["label"],
        label_text=result["label_text"],
        confidence=result["confidence"],
        probabilities=ProbabilityBreakdown(
            non_depressed=result["probabilities"]["non_depressed"],
            depressed=result["probabilities"]["depressed"],
        ),
        threshold_used=request.threshold,
        model_name=model.model_name,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predict depression signals in multiple texts (max 100)",
)
async def predict_batch(
    request: BatchPredictionRequest,
    model=Depends(get_model),
) -> BatchPredictionResponse:
    logger.info(f"Batch prediction request — n_texts={len(request.texts)}")

    if len(request.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size cannot exceed 100 texts.",
        )

    try:
        probas = model.predict_proba(request.texts)
    except Exception as exc:
        logger.exception(f"Batch prediction failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    responses = []
    for text, proba in zip(request.texts, probas):
        label = int(proba[1] >= request.threshold)
        responses.append(
            PredictionResponse(
                text_preview=text[:100],
                label=label,
                label_text="depressed" if label == 1 else "non-depressed",
                confidence=float(proba[label]),
                probabilities=ProbabilityBreakdown(
                    non_depressed=float(proba[0]),
                    depressed=float(proba[1]),
                ),
                threshold_used=request.threshold,
                model_name=model.model_name,
            )
        )

    return BatchPredictionResponse(
        results=responses,
        total=len(responses),
        model_name=model.model_name,
    )


@router.get("/models", summary="List available trained models")
async def list_models(model=Depends(get_model)) -> dict:
    return {
        "active_model": model.model_name,
        "is_fitted": model.is_fitted,
    }
