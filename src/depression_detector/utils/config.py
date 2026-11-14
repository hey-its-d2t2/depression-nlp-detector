"""Type-safe configuration management using pydantic-settings.

Nested models use BaseModel (not BaseSettings) so that only the root
AppConfig reads from environment / .env files, avoiding duplicate env-var
lookups and prefix collisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    max_text_length: int = 512
    min_text_length: int = 10
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed: int = 42
    target_column: str = "label"
    text_column: str = "text"
    positive_class: int = 1

    @field_validator("test_size", "val_size")
    @classmethod
    def validate_split_ratio(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Split ratio must be between 0 and 1, got {v}")
        return v


class TraditionalModelConfig(BaseModel):
    classifier: str = "logistic_regression"
    tfidf_max_features: int = 50000
    tfidf_ngram_range: tuple[int, int] = (1, 3)
    tfidf_sublinear_tf: bool = True
    use_char_ngrams: bool = True
    char_ngram_range: tuple[int, int] = (3, 5)
    use_psychological_features: bool = True
    class_weight: str = "balanced"


class BertModelConfig(BaseModel):
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.3
    gradient_clipping: float = 1.0


class ModelConfig(BaseModel):
    model_type: str = "traditional"
    model_dir: Path = Path("models/saved")
    traditional: TraditionalModelConfig = Field(default_factory=TraditionalModelConfig)
    bert: BertModelConfig = Field(default_factory=BertModelConfig)


class TrainingConfig(BaseModel):
    n_folds: int = 5
    scoring_metric: str = "f1"
    n_jobs: int = -1
    verbose: int = 1
    use_smote: bool = True
    early_stopping_patience: int = 3


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"
    max_text_length: int = 2000
    rate_limit_per_minute: int = 60
    allowed_origins: list[str] = ["*"]


class AppConfig(BaseSettings):
    """Root configuration — reads from .env and environment variables.

    Nested config sections are populated via double-underscore delimiter:
        DATA__RANDOM_SEED=99
        MODEL__MODEL_TYPE=bert
        API__PORT=9000
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Depression NLP Detector"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/app.log"

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: APIConfig = Field(default_factory=APIConfig)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache the application configuration (singleton)."""
    return AppConfig()
