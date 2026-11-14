"""BERT-based depression classifier (requires torch + transformers)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from depression_detector.models.base_model import BaseDepressionModel
from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)


class BertDepressionClassifier(BaseDepressionModel):
    """Fine-tuned BERT classifier for depression detection.

    Fine-tunes a pre-trained BERT model with a linear classification head.
    Supports mixed-precision training (fp16) on CUDA devices.

    Requires: torch, transformers, accelerate
    Install: pip install -r requirements-deep.txt
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        dropout: float = 0.3,
        gradient_clipping: float = 1.0,
        num_labels: int = 2,
        random_seed: int = 42,
    ) -> None:
        super().__init__(model_name=f"bert_{model_name}")
        self._check_dependencies()

        self.hf_model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.dropout = dropout
        self.gradient_clipping = gradient_clipping
        self.num_labels = num_labels
        self.random_seed = random_seed

        self._tokenizer = None
        self._model = None
        self._device = self._get_device()

    @staticmethod
    def _check_dependencies() -> None:
        try:
            import torch
            import transformers
        except ImportError as e:
            raise ImportError(
                "BERT classifier requires torch and transformers. "
                "Install with: pip install -r requirements-deep.txt"
            ) from e

    @staticmethod
    def _get_device():
        import torch
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        logger.info("GPU not available, using CPU")
        return torch.device("cpu")

    def _load_tokenizer_and_model(self) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        logger.info(f"Loading {self.hf_model_name} from HuggingFace Hub")
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.hf_model_name,
            num_labels=self.num_labels,
            hidden_dropout_prob=self.dropout,
            attention_probs_dropout_prob=self.dropout,
        ).to(self._device)

    def _tokenize(self, texts: list[str]):
        import torch

        encoding = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in encoding.items()}

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        val_texts: Optional[list[str]] = None,
        val_labels: Optional[list[int]] = None,
    ) -> "BertDepressionClassifier":
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_linear_schedule_with_warmup

        self._load_tokenizer_and_model()
        torch.manual_seed(self.random_seed)

        logger.info(f"Fine-tuning {self.hf_model_name} for {self.epochs} epochs on {len(texts)} samples")

        # Build DataLoader
        encodings = self._tokenizer(
            texts, truncation=True, padding=True,
            max_length=self.max_length, return_tensors="pt"
        )
        label_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], label_tensor
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer & scheduler
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        total_steps = len(dataloader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        for epoch in range(self.epochs):
            self._model.train()
            total_loss = 0.0
            for batch in dataloader:
                input_ids, attention_mask, batch_labels = [b.to(self._device) for b in batch]
                optimizer.zero_grad()
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.gradient_clipping)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.epochs} — avg loss: {avg_loss:.4f}")

        self._is_fitted = True
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(texts), axis=1)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch

        if self._model is None or not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba()")
        self._model.eval()
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]
                inputs = self._tokenize(batch)
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        logger.info(f"BERT model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BertDepressionClassifier":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        obj = cls.__new__(cls)
        obj.model_name = f"bert_{path}"
        obj._device = cls._get_device()
        obj._tokenizer = AutoTokenizer.from_pretrained(str(path))
        obj._model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(obj._device)
        obj._is_fitted = True
        obj.batch_size = 16
        obj.max_length = 128
        logger.info(f"BERT model loaded from {path}")
        return obj
