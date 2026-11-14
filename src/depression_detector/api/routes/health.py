"""Health and readiness endpoints."""

from __future__ import annotations

import platform
import sys
from datetime import datetime

from fastapi import APIRouter

import depression_detector

router = APIRouter(tags=["health"])

_start_time = datetime.utcnow()


@router.get("/health", summary="Liveness probe")
async def health_check() -> dict:
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready", summary="Readiness probe — model loaded check")
async def readiness(model_registry=None) -> dict:
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}


@router.get("/info", summary="Application metadata")
async def app_info() -> dict:
    return {
        "name": "Depression NLP Detector",
        "version": depression_detector.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "uptime_seconds": (datetime.utcnow() - _start_time).total_seconds(),
    }
