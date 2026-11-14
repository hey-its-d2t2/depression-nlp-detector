"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from depression_detector.data.dataset import DepressionDataset
from depression_detector.data.preprocessing import TextPreprocessor, PreprocessorConfig
from depression_detector.features.text_features import TfidfFeatureExtractor
from depression_detector.features.psychological_features import PsychologicalFeatureExtractor
from depression_detector.models.traditional.sklearn_classifier import TraditionalClassifier


@pytest.fixture(scope="session")
def sample_texts() -> list[str]:
    return [
        "I feel so hopeless and empty every single day. Nothing matters anymore.",
        "I can't sleep and I cry all the time. I feel so worthless and alone.",
        "Had a great day at work! Really enjoying my new project.",
        "Spent time with family. Everything is going really well for me.",
        "I'm trapped in this darkness and I don't see any way out.",
        "Looking forward to the weekend, planning to go hiking with friends.",
    ]


@pytest.fixture(scope="session")
def sample_labels() -> list[int]:
    return [1, 1, 0, 0, 1, 0]


@pytest.fixture(scope="session")
def synthetic_dataset() -> DepressionDataset:
    return DepressionDataset.generate_synthetic(n_samples=200, random_seed=42)


@pytest.fixture(scope="session")
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor(PreprocessorConfig())


@pytest.fixture(scope="session")
def tfidf_extractor() -> TfidfFeatureExtractor:
    return TfidfFeatureExtractor(
        word_max_features=1000,
        char_max_features=500,
        use_char_ngrams=True,
    )


@pytest.fixture(scope="session")
def psych_extractor() -> PsychologicalFeatureExtractor:
    return PsychologicalFeatureExtractor(use_vader=False)


@pytest.fixture(scope="session")
def fitted_model(sample_texts, sample_labels) -> TraditionalClassifier:
    # Use a minimal model for fast tests
    model = TraditionalClassifier(
        classifier_name="logistic_regression",
        word_max_features=500,
        use_char_ngrams=False,
        use_psychological_features=False,
    )
    # Pad to have enough samples for LR
    texts = sample_texts * 10
    labels = sample_labels * 10
    model.fit(texts, labels)
    return model
