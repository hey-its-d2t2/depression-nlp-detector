"""Dataset abstraction for depression detection (polars-backed)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplit:
    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)

    def to_dict(self) -> dict:
        return {"text": self.texts, "label": self.labels}


class DepressionDataset:
    """Dataset manager for depression detection.

    Stores data internally as Python lists for maximum compatibility.
    Uses polars for file I/O operations.
    """

    LABEL_MAP = {0: "non-depressed", 1: "depressed"}

    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        random_seed: int = 42,
    ) -> None:
        self.text_column = text_column
        self.label_column = label_column
        self.random_seed = random_seed
        self._texts: list[str] = []
        self._labels: list[int] = []

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs) -> "DepressionDataset":
        import polars as pl
        ds = cls(**kwargs)
        df = pl.read_csv(str(path))
        ds._load_from_polars(df)
        logger.info(f"Loaded dataset from {path}: {len(ds)} samples")
        return ds

    @classmethod
    def from_dataframe(cls, df, **kwargs) -> "DepressionDataset":
        """Accept a polars DataFrame, dict, or any iterable of dicts."""
        import polars as pl
        ds = cls(**kwargs)
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        ds._load_from_polars(df)
        return ds

    @classmethod
    def generate_synthetic(cls, n_samples: int = 2000, **kwargs) -> "DepressionDataset":
        ds = cls(**kwargs)
        seed = kwargs.get("random_seed", 42)
        rows = _generate_synthetic_depression_data(n_samples, seed)
        ds._texts = [r["text"] for r in rows]
        ds._labels = [r["label"] for r in rows]
        logger.info(f"Generated synthetic dataset: {len(ds)} samples")
        return ds

    def _load_from_polars(self, df) -> None:
        import polars as pl
        tc, lc = self.text_column, self.label_column
        assert tc in df.columns, f"Missing text column '{tc}'. Available: {df.columns}"
        assert lc in df.columns, f"Missing label column '{lc}'. Available: {df.columns}"
        df = df.drop_nulls(subset=[tc, lc])
        df = df.with_columns([
            pl.col(tc).cast(pl.Utf8),
            pl.col(lc).cast(pl.Int32),
        ])
        self._texts = df[tc].to_list()
        self._labels = df[lc].to_list()
        from collections import Counter
        logger.info(
            f"Dataset validated — {len(self._texts)} samples | "
            f"class distribution: {dict(Counter(self._labels))}"
        )

    def split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        from sklearn.model_selection import train_test_split

        X, y = self._texts, np.array(self._labels)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        val_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=self.random_seed,
            stratify=y_trainval,
        )
        logger.info(f"Split → train:{len(X_train)} | val:{len(X_val)} | test:{len(X_test)}")
        return (
            DatasetSplit(X_train, y_train.tolist()),
            DatasetSplit(X_val, y_val.tolist()),
            DatasetSplit(X_test, y_test.tolist()),
        )

    def save_csv(self, path: str | Path) -> None:
        import polars as pl
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({self.text_column: self._texts, self.label_column: self._labels}).write_csv(str(path))
        logger.info(f"Dataset saved to {path}")

    def class_distribution(self) -> dict[str, int]:
        from collections import Counter
        counts = Counter(self._labels)
        return {self.LABEL_MAP.get(k, str(k)): v for k, v in counts.items()}

    @property
    def dataframe(self):
        import polars as pl
        return pl.DataFrame({self.text_column: self._texts, self.label_column: self._labels})

    def __len__(self) -> int:
        return len(self._texts)


# ---------------------------------------------------------------------------
# Synthetic data generation helpers
# ---------------------------------------------------------------------------

_DEPRESSED_TEMPLATES = [
    "I feel so {adj} and {adj2}. Nothing seems to {verb} anymore.",
    "Every day I wake up feeling {adj}. I don't see the point of {activity}.",
    "I can't stop thinking about how {adj} everything is. I feel {adj2} inside.",
    "Nobody understands how {adj} I feel. I'm so {adj2} all the time.",
    "I've been feeling {adj} for weeks. I can't {verb} or {verb2}.",
    "Sometimes I think I'm {adj} and no one would even notice if I disappeared.",
    "I cry all the time and I don't even know why. I feel so {adj} and {adj2}.",
    "My life feels {adj}. I've lost all interest in {activity}.",
    "I feel like a {noun} — {adj} and completely {adj2}.",
    "I haven't been able to {verb} properly in days. I feel {adj}.",
    "Everything feels {adj}. I'm so tired of feeling this way.",
    "I feel so alone even when I'm surrounded by people. It's {adj}.",
    "I keep thinking that I'm {adj} and nobody cares about me.",
    "I can't concentrate, can't sleep, can't {verb}. I'm {adj}.",
    "I feel empty inside. Like nothing will ever get better. It's {adj}.",
    "I am so {adj2}. There is no hope left for me.",
    "What's the point of anything? I am just so {adj} and {adj2}.",
    "I've been struggling with {mood} for so long. I can't escape this {adj} feeling.",
    "I don't deserve happiness. I am a {noun} to everyone around me.",
    "I wake up every morning wishing I hadn't. I am so {adj}.",
    "Nothing I do matters. I'm so {adj} and {adj2} all the time.",
    "I've stopped caring about {activity}. I feel {adj} and invisible.",
    "I can't stop crying. I feel {adj2} and I don't know why.",
    "My mind is full of {adj2} thoughts. I can't escape them.",
    "I'm exhausted from pretending to be okay. Inside I'm so {adj} and {adj2}.",
    "I feel {adj} and nobody cares about me at all.",
    "I feel like a {noun} to everyone. I wish I could disappear.",
    "Every single day I feel {adj}. There is no hope left.",
    "I am so {adj2}. Life has no meaning anymore.",
    "I feel {adj} and {adj2}. No one understands what I am going through.",
    "I am a {noun} to my family. They would be better off without me.",
    "I cannot function. I feel {adj} and completely {adj2}.",
    "What is wrong with me? I feel so {adj} all the time.",
    "I feel {adj}. Nothing will ever change. It is {adj2}.",
    "I have been feeling {adj} for months. I see no way out.",
]

_NON_DEPRESSED_TEMPLATES = [
    "Had a great day today! Really enjoyed {activity}.",
    "Feeling {adj} about the upcoming {event}. Can't wait!",
    "Just finished {activity} and feeling accomplished.",
    "Spent time with family today, it was {adj} and {adj2}.",
    "Looking forward to the weekend. Planning to {verb}.",
    "The weather is {adj} today. Perfect for {activity}.",
    "Made some progress on my {project} today. Feeling {adj}.",
    "Had a productive meeting. Things are going {adv}.",
    "Tried a new recipe and it turned out {adj}. {adj2} day overall.",
    "Finished reading a {adj} book. Highly recommend it.",
    "Exercise was {adj} this morning. Feeling {adj2} and energized.",
    "Caught up with an old friend today. It was really {adj}.",
    "Had a {adj} conversation about {topic} with my colleagues.",
    "Successfully completed my tasks for the day. Feeling {adj}.",
    "Planning a trip to {place}. Really excited about it.",
    "Life is {adj} right now. Grateful for all the good things.",
    "Had a {adj} dinner with the family. Kids were so {adj2}.",
    "My new {project} is coming together nicely. Feeling very {adj}.",
    "Went for a morning run and it was {adj}. Great way to start the day.",
    "Reconnected with old friends over the weekend. It was {adj} and {adj2}.",
    "Things at work are going {adv}. Really proud of what we have accomplished.",
    "Enjoying a {adj} cup of coffee and reading a {adj2} book.",
    "Completed my workout and feeling {adj} and energized.",
    "Had such a {adj} time at {place} today. Memories to cherish.",
    "Feeling grateful and {adj} today. Life is good.",
]

_DEPRESSED_WORDS = {
    "adj": ["worthless", "hopeless", "empty", "broken", "numb", "exhausted", "isolated",
            "invisible", "trapped", "useless", "utterly alone", "hollow", "defeated",
            "devastated", "completely lost"],
    "adj2": ["lost", "defeated", "overwhelmed", "miserable", "deeply depressed", "sad",
             "helpless", "desperate", "terribly anxious", "completely broken",
             "utterly hopeless", "deeply unhappy"],
    "verb": ["sleep properly", "eat normally", "function", "concentrate", "cope",
             "think clearly", "get out of bed", "work", "feel happy", "stop crying"],
    "verb2": ["focus on anything", "enjoy anything", "stop crying", "feel normal",
              "see the point", "carry on"],
    "noun": ["burden", "failure", "ghost", "empty shell", "disappointment"],
    "activity": ["life", "going out", "socializing", "work", "my hobbies",
                 "seeing friends", "getting up in the morning"],
    "mood": ["sadness", "hopelessness", "darkness", "emptiness", "despair",
             "anxiety", "overwhelming grief"],
}

_NON_DEPRESSED_WORDS = {
    "adj": ["great", "wonderful", "amazing", "fantastic", "excellent", "happy",
            "lovely", "brilliant", "enjoyable", "peaceful", "joyful", "pleasant"],
    "adj2": ["refreshed", "positive", "energetic", "grateful", "content",
             "motivated", "inspired", "fulfilled", "cheerful", "enthusiastic"],
    "adv": ["really well", "smoothly", "perfectly", "excellently", "great"],
    "event": ["weekend trip", "holiday", "project launch", "family gathering", "celebration"],
    "activity": ["cooking", "reading", "hiking", "yoga", "painting", "working out",
                 "gardening", "cycling", "swimming", "photography"],
    "project": ["project", "assignment", "hobby", "personal goal", "side project"],
    "place": ["the mountains", "the beach", "Europe", "a new city", "the park",
              "a lovely cafe", "the countryside"],
    "topic": ["technology", "science", "art", "music", "history", "travel", "food"],
    "verb": ["relax", "travel", "meet friends", "explore", "try something new",
             "spend quality time", "learn a new skill"],
}


def _fill_template(template: str, word_dict: dict, rng: random.Random) -> str:
    result = template
    for key, options in word_dict.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, rng.choice(options), 1)
    return result


def _generate_synthetic_depression_data(n_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    half = n_samples // 2
    for _ in range(half):
        tpl = rng.choice(_DEPRESSED_TEMPLATES)
        rows.append({"text": _fill_template(tpl, _DEPRESSED_WORDS, rng), "label": 1})
    for _ in range(n_samples - half):
        tpl = rng.choice(_NON_DEPRESSED_TEMPLATES)
        rows.append({"text": _fill_template(tpl, _NON_DEPRESSED_WORDS, rng), "label": 0})
    rng.shuffle(rows)
    return rows
