"""TF-IDF and character n-gram feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)


class TfidfFeatureExtractor:
    """Dual-channel TF-IDF extractor: word n-grams + character n-grams.

    Character n-grams capture morphological patterns (e.g., "less", "ful")
    that are predictive of emotional language even after stemming.
    """

    def __init__(
        self,
        word_max_features: int = 50000,
        word_ngram_range: tuple[int, int] = (1, 3),
        char_max_features: int = 30000,
        char_ngram_range: tuple[int, int] = (3, 5),
        sublinear_tf: bool = True,
        min_df: int = 2,
        use_char_ngrams: bool = True,
    ) -> None:
        self.use_char_ngrams = use_char_ngrams

        self._word_vectorizer = TfidfVectorizer(
            max_features=word_max_features,
            ngram_range=word_ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            analyzer="word",
            strip_accents="unicode",
            decode_error="replace",
        )

        if use_char_ngrams:
            self._char_vectorizer = TfidfVectorizer(
                max_features=char_max_features,
                ngram_range=char_ngram_range,
                sublinear_tf=sublinear_tf,
                min_df=min_df,
                analyzer="char_wb",
                strip_accents="unicode",
                decode_error="replace",
            )
        else:
            self._char_vectorizer = None

        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfFeatureExtractor":
        logger.info(f"Fitting TF-IDF on {len(texts)} texts")
        self._word_vectorizer.fit(texts)
        if self.use_char_ngrams and self._char_vectorizer:
            self._char_vectorizer.fit(texts)
        self._fitted = True
        logger.info(
            f"Word vocab size: {len(self._word_vectorizer.vocabulary_)} | "
            + (
                f"Char vocab size: {len(self._char_vectorizer.vocabulary_)}"
                if self.use_char_ngrams
                else "char n-grams disabled"
            )
        )
        return self

    def transform(self, texts: list[str]) -> csr_matrix:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        word_features = self._word_vectorizer.transform(texts)
        if self.use_char_ngrams and self._char_vectorizer:
            char_features = self._char_vectorizer.transform(texts)
            return hstack([word_features, char_features])
        return word_features

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        return self.fit(texts).transform(texts)

    def get_feature_names(self) -> list[str]:
        word_names = self._word_vectorizer.get_feature_names_out().tolist()
        if self.use_char_ngrams and self._char_vectorizer:
            char_names = [f"char_{n}" for n in self._char_vectorizer.get_feature_names_out()]
            return word_names + char_names
        return word_names

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"word": self._word_vectorizer, "char": self._char_vectorizer, "fitted": self._fitted},
            path,
        )
        logger.info(f"TF-IDF extractor saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TfidfFeatureExtractor":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._word_vectorizer = data["word"]
        obj._char_vectorizer = data["char"]
        obj._fitted = data["fitted"]
        obj.use_char_ngrams = data["char"] is not None
        logger.info(f"TF-IDF extractor loaded from {path}")
        return obj
