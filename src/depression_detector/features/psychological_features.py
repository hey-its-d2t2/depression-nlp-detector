"""LIWC-inspired psychological and linguistic feature extraction.

Features are grounded in peer-reviewed depression NLP research:
- Rude & Gortner (2004): increased first-person singular pronouns
- Stirman & Pennebaker (2001): reduced social reference words
- Pennebaker et al. (2001): LIWC word categories
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lexical dictionaries (curated word lists)
# ---------------------------------------------------------------------------

_FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
_FIRST_PERSON_PLURAL = {"we", "us", "our", "ours", "ourselves"}
_SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
_THIRD_PERSON = {"he", "she", "they", "him", "her", "them", "his", "hers", "their"}

_NEGATIVE_EMOTION = {
    "sad", "unhappy", "miserable", "depressed", "hopeless", "worthless", "empty",
    "lonely", "alone", "isolated", "numb", "lost", "broken", "hurt", "pain",
    "suffer", "suffering", "despair", "despaired", "desperate", "anxious",
    "afraid", "fear", "scared", "terrified", "angry", "rage", "furious",
    "hate", "loathe", "disgust", "shame", "guilt", "regret", "sorry",
    "tired", "exhausted", "drained", "useless", "failure", "failed",
    "stupid", "ugly", "fat", "burden", "invisible", "trapped", "stuck",
}

_POSITIVE_EMOTION = {
    "happy", "joy", "joyful", "glad", "grateful", "thankful", "love",
    "excited", "amazing", "wonderful", "fantastic", "great", "good",
    "pleased", "proud", "content", "peaceful", "calm", "hope", "hopeful",
    "optimistic", "cheerful", "blissful", "elated", "enthusiastic",
    "motivated", "inspired", "satisfied", "fulfilled",
}

_COGNITIVE_DISTORTIONS = {
    "always", "never", "everyone", "nobody", "nothing", "everything",
    "completely", "totally", "absolutely", "impossible", "must", "should",
    "have to", "can't", "cannot", "won't", "worthless", "failure",
    "useless", "pointless", "hopeless", "helpless", "catastrophe",
    "disaster", "terrible", "horrible", "awful",
}

_DEATH_RELATED = {
    "die", "death", "dead", "dying", "suicide", "suicidal", "kill",
    "end it", "end my life", "not alive", "no longer here", "disappear",
    "gone", "afterlife", "cemetery", "funeral", "overdose",
}

_SOCIAL_PROCESS = {
    "talk", "share", "listen", "friend", "family", "people", "together",
    "meet", "help", "support", "community", "connect", "relationship",
    "group", "team",
}

_NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "nothing", "nobody",
    "nowhere", "none", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "wasn't", "aren't", "weren't",
}

_FUTURE_TENSE_MARKERS = {
    "will", "shall", "going to", "plan to", "hope to", "expect to",
    "intend to", "about to",
}

_PAST_TENSE_INDICATORS = {
    "was", "were", "had", "did", "used to", "before", "ago", "previously",
    "once", "last", "yesterday", "earlier",
}


class PsychologicalFeatureExtractor:
    """Extracts clinical linguistic features relevant to depression detection.

    Feature groups:
        1. Pronoun ratios (self-focus indicator)
        2. Affect word counts (positive / negative / death)
        3. Cognitive distortion markers
        4. Social process word density
        5. Negation density
        6. Temporal orientation (past vs future tense)
        7. Text statistics (length, sentence count, avg sentence length)
        8. VADER sentiment scores
    """

    def __init__(self, use_vader: bool = True) -> None:
        self.use_vader = use_vader
        self._vader = None
        if use_vader:
            self._init_vader()

    def _init_vader(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("vaderSentiment not installed — VADER features disabled")
            self.use_vader = False

    @property
    def feature_names(self) -> list[str]:
        names = [
            "ratio_first_person_singular",
            "ratio_first_person_plural",
            "ratio_second_person",
            "ratio_third_person",
            "ratio_negative_emotion",
            "ratio_positive_emotion",
            "ratio_death_related",
            "ratio_cognitive_distortion",
            "ratio_social_process",
            "ratio_negation",
            "ratio_future_tense",
            "ratio_past_tense",
            "text_length",
            "sentence_count",
            "avg_sentence_length",
            "type_token_ratio",
            "avg_word_length",
        ]
        if self.use_vader:
            names += ["vader_neg", "vader_neu", "vader_pos", "vader_compound"]
        return names

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _ratio(self, tokens: list[str], word_set: set[str]) -> float:
        if not tokens:
            return 0.0
        return sum(1 for t in tokens if t in word_set) / len(tokens)

    def extract_one(self, text: str) -> dict[str, float]:
        if not text or not text.strip():
            return {name: 0.0 for name in self.feature_names}

        tokens = self._tokenize(text)
        n_tokens = len(tokens) or 1

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = len(sentences) or 1

        features: dict[str, float] = {
            "ratio_first_person_singular": self._ratio(tokens, _FIRST_PERSON_SINGULAR),
            "ratio_first_person_plural": self._ratio(tokens, _FIRST_PERSON_PLURAL),
            "ratio_second_person": self._ratio(tokens, _SECOND_PERSON),
            "ratio_third_person": self._ratio(tokens, _THIRD_PERSON),
            "ratio_negative_emotion": self._ratio(tokens, _NEGATIVE_EMOTION),
            "ratio_positive_emotion": self._ratio(tokens, _POSITIVE_EMOTION),
            "ratio_death_related": self._ratio(tokens, _DEATH_RELATED),
            "ratio_cognitive_distortion": self._ratio(tokens, _COGNITIVE_DISTORTIONS),
            "ratio_social_process": self._ratio(tokens, _SOCIAL_PROCESS),
            "ratio_negation": self._ratio(tokens, _NEGATION_WORDS),
            "ratio_future_tense": self._ratio(tokens, _FUTURE_TENSE_MARKERS),
            "ratio_past_tense": self._ratio(tokens, _PAST_TENSE_INDICATORS),
            "text_length": float(len(text)),
            "sentence_count": float(n_sentences),
            "avg_sentence_length": float(len(text)) / n_sentences,
            "type_token_ratio": float(len(set(tokens))) / n_tokens,
            "avg_word_length": float(sum(len(t) for t in tokens)) / n_tokens,
        }

        if self.use_vader and self._vader:
            scores = self._vader.polarity_scores(text)
            features["vader_neg"] = scores["neg"]
            features["vader_neu"] = scores["neu"]
            features["vader_pos"] = scores["pos"]
            features["vader_compound"] = scores["compound"]

        return features

    def transform(self, texts: list[str]) -> np.ndarray:
        """Extract features for a list of texts.

        Returns shape (n_samples, n_features).
        """
        rows = [list(self.extract_one(t).values()) for t in texts]
        return np.array(rows, dtype=np.float32)

    def transform_as_dataframe(self, texts: list[str]):
        """Return features as a polars DataFrame."""
        import polars as pl
        rows = [self.extract_one(t) for t in texts]
        return pl.DataFrame(rows)
