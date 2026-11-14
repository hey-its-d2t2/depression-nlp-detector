"""Text cleaning and normalization pipeline."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy NLTK resource bootstrap
_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
]


def _ensure_nltk_resources() -> None:
    for resource_path, resource_name in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)


# Negation words — preserving these improves sentiment detection accuracy
_NEGATION_WORDS = {
    "no", "not", "never", "neither", "nor", "nowhere", "nothing",
    "nobody", "none", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "wasn't", "aren't", "weren't", "haven't",
    "hasn't", "hadn't", "wouldn't", "shouldn't", "couldn't",
}

# Depression-related stopword exceptions (keep them even if in stopwords)
_KEEP_WORDS = _NEGATION_WORDS | {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "very", "too", "again", "always", "never", "alone", "empty",
    "hopeless", "worthless", "useless",
}


@dataclass
class PreprocessorConfig:
    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_emails: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = True
    remove_extra_whitespace: bool = True
    lemmatize: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    max_token_length: int = 50
    preserve_negations: bool = True
    expand_contractions: bool = True


_CONTRACTIONS: dict[str, str] = {
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would",
    "they'd": "they would", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "doesn't": "does not",
    "don't": "do not", "didn't": "did not", "won't": "will not",
    "wouldn't": "would not", "can't": "cannot", "couldn't": "could not",
    "shouldn't": "should not", "mightn't": "might not",
    "mustn't": "must not", "that's": "that is", "what's": "what is",
    "there's": "there is", "here's": "here is", "let's": "let us",
}


class TextPreprocessor:
    """Multi-stage text preprocessing pipeline optimized for mental health NLP.

    Preserves negation words and depression-related vocabulary that generic
    NLP pipelines would otherwise strip, maintaining clinical signal fidelity.
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.config = config or PreprocessorConfig()
        _ensure_nltk_resources()
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = self._build_stop_words()

    def _build_stop_words(self) -> set[str]:
        stops = set(stopwords.words("english"))
        if self.config.preserve_negations:
            stops -= _KEEP_WORDS
        return stops

    # ------------------------------------------------------------------
    # Individual transformation steps (composable, testable)
    # ------------------------------------------------------------------

    def remove_urls(self, text: str) -> str:
        return re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

    def remove_html_tags(self, text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    def remove_emails(self, text: str) -> str:
        return re.sub(r"\S+@\S+\.\S+", " ", text)

    def expand_contractions(self, text: str) -> str:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in _CONTRACTIONS) + r")\b",
            re.IGNORECASE,
        )
        return pattern.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text)

    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    def normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Apply surface-level cleaning without tokenization."""
        if not text or not isinstance(text, str):
            return ""

        if self.config.remove_urls:
            text = self.remove_urls(text)
        if self.config.remove_html:
            text = self.remove_html_tags(text)
        if self.config.remove_emails:
            text = self.remove_emails(text)
        if self.config.expand_contractions:
            text = self.expand_contractions(text)
        if self.config.lowercase:
            text = text.lower()
        if self.config.remove_punctuation:
            text = self.remove_punctuation(text)
        if self.config.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        if self.config.remove_extra_whitespace:
            text = self.normalize_whitespace(text)

        return text

    def tokenize_and_normalize(self, text: str) -> list[str]:
        """Tokenize cleaned text and apply stopword removal + lemmatization."""
        tokens = word_tokenize(text)

        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        # Length filter
        tokens = [
            t for t in tokens
            if self.config.min_token_length <= len(t) <= self.config.max_token_length
        ]

        if self.config.lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def process(self, text: str) -> str:
        """Full pipeline: clean → tokenize → normalize → rejoin."""
        cleaned = self.clean(text)
        tokens = self.tokenize_and_normalize(cleaned)
        return " ".join(tokens)

    def process_batch(self, texts: list[str]) -> list[str]:
        """Process a list of texts."""
        return [self.process(t) for t in texts]

    def get_raw_tokens(self, text: str) -> list[str]:
        """Return tokens from cleaned text without stopword removal (for feature extraction)."""
        cleaned = self.clean(text)
        return word_tokenize(cleaned)
