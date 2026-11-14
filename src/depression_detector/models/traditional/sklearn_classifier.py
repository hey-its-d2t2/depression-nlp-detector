"""Traditional ML classifier with feature fusion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy.sparse import hstack, issparse, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from depression_detector.data.preprocessing import TextPreprocessor, PreprocessorConfig
from depression_detector.features.text_features import TfidfFeatureExtractor
from depression_detector.features.psychological_features import PsychologicalFeatureExtractor
from depression_detector.models.base_model import BaseDepressionModel
from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)

_CLASSIFIERS = {
    "logistic_regression": lambda cw: LogisticRegression(
        C=1.0, max_iter=1000, class_weight=cw, solver="lbfgs"
    ),
    "svm": lambda cw: CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=2000, class_weight=cw), cv=3, method="sigmoid"
    ),
    "random_forest": lambda cw: RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight=cw, n_jobs=-1, random_state=42
    ),
    "gradient_boosting": lambda _: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
    ),
    "ensemble": lambda cw: None,  # built separately below
}


class TraditionalClassifier(BaseDepressionModel):
    """Feature-fusion traditional ML classifier.

    Architecture:
        TextPreprocessor → TF-IDF (word + char n-grams)  ─┐
                                                           ├→ concat → Classifier
        TextPreprocessor → PsychologicalFeatures           ─┘

    The dual-channel approach combines distributional semantics (TF-IDF)
    with manually engineered clinical linguistic features for best F1
    without requiring a GPU.
    """

    def __init__(
        self,
        classifier_name: str = "logistic_regression",
        word_max_features: int = 50000,
        word_ngram_range: tuple[int, int] = (1, 3),
        char_max_features: int = 30000,
        char_ngram_range: tuple[int, int] = (3, 5),
        use_char_ngrams: bool = True,
        use_psychological_features: bool = True,
        class_weight: str = "balanced",
        random_seed: int = 42,
    ) -> None:
        super().__init__(model_name=f"traditional_{classifier_name}")
        self.classifier_name = classifier_name
        self.class_weight = class_weight
        self.use_psychological_features = use_psychological_features
        self.random_seed = random_seed

        self._preprocessor = TextPreprocessor(PreprocessorConfig(remove_stopwords=False))
        self._tfidf = TfidfFeatureExtractor(
            word_max_features=word_max_features,
            word_ngram_range=word_ngram_range,
            char_max_features=char_max_features,
            char_ngram_range=char_ngram_range,
            use_char_ngrams=use_char_ngrams,
        )
        self._psych = PsychologicalFeatureExtractor(use_vader=True) if use_psychological_features else None
        self._clf = self._build_classifier()

    def _build_classifier(self):
        if self.classifier_name == "ensemble":
            lr = LogisticRegression(C=1.0, max_iter=1000, class_weight=self.class_weight, solver="lbfgs", n_jobs=-1)
            svm = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, class_weight=self.class_weight), cv=3)
            rf = RandomForestClassifier(n_estimators=200, class_weight=self.class_weight, n_jobs=-1, random_state=self.random_seed)
            return VotingClassifier(
                estimators=[("lr", lr), ("svm", svm), ("rf", rf)],
                voting="soft",
                n_jobs=-1,
            )
        if self.classifier_name not in _CLASSIFIERS:
            raise ValueError(f"Unknown classifier '{self.classifier_name}'. Choose from: {list(_CLASSIFIERS)}")
        return _CLASSIFIERS[self.classifier_name](self.class_weight)

    def _build_features(self, texts: list[str], fit: bool = False):
        # Minimal cleaning for TF-IDF (preserve important tokens)
        cleaned = [self._preprocessor.clean(t) for t in texts]

        if fit:
            tfidf_feats = self._tfidf.fit_transform(cleaned)
        else:
            tfidf_feats = self._tfidf.transform(cleaned)

        if self._psych is not None:
            psych_feats = self._psych.transform(texts)  # use raw text for psych features
            return hstack([tfidf_feats, csr_matrix(psych_feats)])
        return tfidf_feats

    def fit(self, texts: list[str], labels: list[int]) -> "TraditionalClassifier":
        logger.info(f"Training {self.classifier_name} on {len(texts)} samples")
        X = self._build_features(texts, fit=True)
        self._clf.fit(X, labels)
        self._is_fitted = True
        logger.info("Training complete")
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        X = self._build_features(texts, fit=False)
        return self._clf.predict(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self._build_features(texts, fit=False)
        return self._clf.predict_proba(X)

    def get_top_features(self, n: int = 20) -> dict:
        """Return top predictive features for interpretability."""
        if not hasattr(self._clf, "coef_"):
            return {}
        feature_names = self._tfidf.get_feature_names()
        if self._psych:
            feature_names += self._psych.feature_names
        coef = self._clf.coef_.ravel()
        top_pos_idx = np.argsort(coef)[-n:][::-1]
        top_neg_idx = np.argsort(coef)[:n]
        return {
            "depression_indicators": [(feature_names[i], float(coef[i])) for i in top_pos_idx if i < len(feature_names)],
            "non_depression_indicators": [(feature_names[i], float(coef[i])) for i in top_neg_idx if i < len(feature_names)],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self._clf,
                "tfidf": self._tfidf,
                "psych": self._psych,
                "preprocessor": self._preprocessor,
                "config": {
                    "classifier_name": self.classifier_name,
                    "use_psychological_features": self.use_psychological_features,
                    "class_weight": self.class_weight,
                },
            },
            path,
            compress=3,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TraditionalClassifier":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model_name = f"traditional_{data['config']['classifier_name']}"
        obj._clf = data["clf"]
        obj._tfidf = data["tfidf"]
        obj._psych = data["psych"]
        obj._preprocessor = data["preprocessor"]
        obj.classifier_name = data["config"]["classifier_name"]
        obj.use_psychological_features = data["config"]["use_psychological_features"]
        obj.class_weight = data["config"]["class_weight"]
        obj._is_fitted = True
        logger.info(f"Model loaded from {path}")
        return obj
