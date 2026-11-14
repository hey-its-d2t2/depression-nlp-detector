"""Unified data loading interface supporting multiple sources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from depression_detector.data.dataset import DepressionDataset
from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads depression detection datasets from various sources.

    Supported sources:
        - Local CSV / JSON / Parquet files (via polars)
        - Synthetic data (for demos and testing)
        - HuggingFace Hub datasets (requires `datasets` package)
    """

    def __init__(
        self,
        text_column: str = "text",
        label_column: str = "label",
        random_seed: int = 42,
    ) -> None:
        self.text_column = text_column
        self.label_column = label_column
        self.random_seed = random_seed

    def _kwargs(self) -> dict:
        return {
            "text_column": self.text_column,
            "label_column": self.label_column,
            "random_seed": self.random_seed,
        }

    def from_csv(self, path: str | Path) -> DepressionDataset:
        logger.info(f"Loading CSV dataset from {path}")
        return DepressionDataset.from_csv(path, **self._kwargs())

    def from_json(self, path: str | Path) -> DepressionDataset:
        import polars as pl
        logger.info(f"Loading JSON dataset from {path}")
        if str(path).endswith(".jsonl"):
            df = pl.read_ndjson(str(path))
        else:
            df = pl.read_json(str(path))
        return DepressionDataset.from_dataframe(df, **self._kwargs())

    def from_parquet(self, path: str | Path) -> DepressionDataset:
        import polars as pl
        logger.info(f"Loading Parquet dataset from {path}")
        df = pl.read_parquet(str(path))
        return DepressionDataset.from_dataframe(df, **self._kwargs())

    def from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> DepressionDataset:
        """Load a dataset from HuggingFace Hub.

        Recommended public dataset:
            depression_reddit_cleaned  — binary labels (0/1)

        Add your HF token for private repos via hf_token parameter
        or HUGGINGFACE_TOKEN env variable.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install the 'datasets' package: pip install datasets") from None

        logger.info(f"Loading HuggingFace dataset: {dataset_name} (split={split})")
        hf_ds = load_dataset(dataset_name, split=split, token=hf_token)
        if max_samples:
            hf_ds = hf_ds.select(range(min(max_samples, len(hf_ds))))

        # Use polars output if available, else fallback to pandas conversion
        try:
            import polars as pl
            df = hf_ds.to_polars()
        except AttributeError:
            import polars as pl
            df = pl.from_pandas(hf_ds.to_pandas())

        tc = text_column or self.text_column
        lc = label_column or self.label_column
        if tc != "text" or lc != "label":
            df = df.rename({tc: "text", lc: "label"})

        kwargs = {**self._kwargs(), "text_column": "text", "label_column": "label"}
        return DepressionDataset.from_dataframe(df[["text", "label"]], **kwargs)

    def synthetic(self, n_samples: int = 2000) -> DepressionDataset:
        logger.info(f"Generating synthetic dataset with {n_samples} samples")
        return DepressionDataset.generate_synthetic(n_samples=n_samples, **self._kwargs())

    def load(self, path: str | Path) -> DepressionDataset:
        """Auto-detect format and load."""
        path = Path(path)
        ext = path.suffix.lower()
        dispatch = {
            ".csv": self.from_csv,
            ".json": self.from_json,
            ".jsonl": self.from_json,
            ".parquet": self.from_parquet,
        }
        if ext not in dispatch:
            raise ValueError(f"Unsupported file format: {ext}. Expected: {list(dispatch)}")
        return dispatch[ext](path)
