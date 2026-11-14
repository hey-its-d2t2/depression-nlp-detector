"""Abstract base class for all depression detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class BaseDepressionModel(ABC):
    """Contract that every model in this system must satisfy.

    Provides a consistent interface for training, inference, persistence,
    and introspection regardless of the underlying algorithm.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._is_fitted = False

    @abstractmethod
    def fit(self, texts: list[str], labels: list[int]) -> "BaseDepressionModel":
        """Train the model."""

    @abstractmethod
    def predict(self, texts: list[str]) -> np.ndarray:
        """Return binary predictions (0 or 1) for each text."""

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probability estimates, shape (n_samples, 2)."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the model artefact to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseDepressionModel":
        """Load a persisted model from disk."""

    def predict_one(self, text: str) -> dict:
        """Convenience wrapper for a single text prediction."""
        proba = self.predict_proba([text])[0]
        label = int(np.argmax(proba))
        return {
            "label": label,
            "label_text": "depressed" if label == 1 else "non-depressed",
            "confidence": float(proba[label]),
            "probabilities": {"non_depressed": float(proba[0]), "depressed": float(proba[1])},
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name!r}, fitted={self._is_fitted})"
