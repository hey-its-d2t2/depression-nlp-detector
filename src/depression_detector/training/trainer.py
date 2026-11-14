"""End-to-end training orchestration with cross-validation and artifact management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from depression_detector.data.dataset import DatasetSplit
from depression_detector.evaluation.metrics import ModelEvaluator
from depression_detector.models.base_model import BaseDepressionModel
from depression_detector.models.traditional.sklearn_classifier import TraditionalClassifier
from depression_detector.utils.config import AppConfig, get_config
from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Orchestrates the full training lifecycle.

    Responsibilities:
        - K-fold cross-validation on training set
        - Final model fitting on full train split
        - SMOTE oversampling for class imbalance
        - Evaluation on held-out test split
        - Artifact persistence (model + evaluation report)
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_config()
        self.evaluator = ModelEvaluator(threshold=0.5)

    def build_model(self) -> BaseDepressionModel:
        """Instantiate a model according to configuration."""
        cfg = self.config.model
        if cfg.model_type == "traditional":
            tc = cfg.traditional
            return TraditionalClassifier(
                classifier_name=tc.classifier,
                word_max_features=tc.tfidf_max_features,
                word_ngram_range=tc.tfidf_ngram_range,
                use_char_ngrams=tc.use_char_ngrams,
                char_ngram_range=tc.char_ngram_range,
                use_psychological_features=tc.use_psychological_features,
                class_weight=tc.class_weight,
            )
        if cfg.model_type == "bert":
            from depression_detector.models.deep_learning.bert_classifier import BertDepressionClassifier
            bc = cfg.bert
            return BertDepressionClassifier(
                model_name=bc.model_name,
                max_length=bc.max_length,
                batch_size=bc.batch_size,
                epochs=bc.epochs,
                learning_rate=bc.learning_rate,
                weight_decay=bc.weight_decay,
                warmup_ratio=bc.warmup_ratio,
                dropout=bc.dropout,
            )
        raise ValueError(f"Unknown model_type '{cfg.model_type}'. Choose: traditional | bert")

    def cross_validate(
        self,
        train: DatasetSplit,
        n_folds: int = 5,
    ) -> dict[str, float]:
        """Run stratified K-fold CV and return mean ± std of key metrics."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, roc_auc_score

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        texts = np.array(train.texts)
        labels = np.array(train.labels)

        f1_scores, auc_scores = [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            fold_model = self.build_model()
            fold_model.fit(texts[train_idx].tolist(), labels[train_idx].tolist())
            proba = fold_model.predict_proba(texts[val_idx].tolist())
            pred = np.argmax(proba, axis=1)
            f1 = f1_score(labels[val_idx], pred, zero_division=0)
            auc = roc_auc_score(labels[val_idx], proba[:, 1])
            f1_scores.append(f1)
            auc_scores.append(auc)
            logger.info(f"Fold {fold + 1}/{n_folds} — F1={f1:.4f}, AUC={auc:.4f}")

        cv_results = {
            "cv_f1_mean": float(np.mean(f1_scores)),
            "cv_f1_std": float(np.std(f1_scores)),
            "cv_auc_mean": float(np.mean(auc_scores)),
            "cv_auc_std": float(np.std(auc_scores)),
        }
        logger.info(
            f"CV Results — F1: {cv_results['cv_f1_mean']:.4f} ± {cv_results['cv_f1_std']:.4f} | "
            f"AUC: {cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}"
        )
        return cv_results

    def apply_smote(
        self,
        texts: list[str],
        labels: list[int],
    ) -> tuple[list[str], list[int]]:
        """Apply SMOTE to balance classes.

        Since SMOTE operates on numeric features, we vectorize first with
        a lightweight TF-IDF, oversample the numeric space, then return
        indices to duplicate original texts (index-based SMOTE variant).
        This avoids creating nonsensical interpolated text.
        """
        try:
            from imblearn.over_sampling import RandomOverSampler
        except ImportError:
            logger.warning("imbalanced-learn not installed — skipping oversampling")
            return texts, labels

        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse as sp

        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform(texts)
        y = np.array(labels)

        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)

        # Map back to original texts using sample indices from RandomOverSampler
        sample_indices = ros.sample_indices_
        resampled_texts = [texts[i] for i in sample_indices]
        resampled_labels = y_res.tolist()

        logger.info(
            f"SMOTE: {len(texts)} → {len(resampled_texts)} samples "
            f"(class 0: {resampled_labels.count(0)}, class 1: {resampled_labels.count(1)})"
        )
        return resampled_texts, resampled_labels

    def train(
        self,
        train: DatasetSplit,
        val: DatasetSplit,
        test: DatasetSplit,
        run_cv: bool = True,
        use_smote: bool = True,
        save_model: bool = True,
    ) -> dict:
        """Full training pipeline: CV → fit → evaluate → save."""
        results: dict = {}

        # Cross-validation
        if run_cv:
            logger.info("Running cross-validation...")
            results["cross_validation"] = self.cross_validate(train)

        # Oversample training set
        train_texts, train_labels = train.texts, train.labels
        if use_smote:
            train_texts, train_labels = self.apply_smote(train_texts, train_labels)

        # Final model fit
        logger.info("Fitting final model on full training set...")
        model = self.build_model()
        model.fit(train_texts, train_labels)

        # Validation evaluation
        val_proba = model.predict_proba(val.texts)
        val_metrics = self.evaluator.evaluate(val.labels, val_proba, model_name="validation")
        results["validation"] = val_metrics

        # Test evaluation
        test_proba = model.predict_proba(test.texts)
        test_metrics = self.evaluator.evaluate(test.labels, test_proba, model_name="test")
        results["test"] = test_metrics

        # Persist artifacts
        if save_model:
            model_path = Path(self.config.model.model_dir) / f"{model.model_name}.pkl"
            model.save(model_path)
            report_dir = Path("reports")
            self.evaluator.save_report(report_dir)
            logger.info(f"Artifacts saved — model: {model_path}, reports: {report_dir}")

        results["model"] = model
        return results
