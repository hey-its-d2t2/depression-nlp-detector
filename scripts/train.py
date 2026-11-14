#!/usr/bin/env python
"""Training entry-point script.

Usage examples:
    # Train on synthetic data (demo, no data required):
    python scripts/train.py

    # Train on a local CSV file:
    python scripts/train.py --data data/raw/depression_dataset.csv

    # Train on HuggingFace dataset:
    python scripts/train.py --hf-dataset depression_reddit_cleaned

    # Use BERT model:
    python scripts/train.py --model-type bert

    # Skip cross-validation (faster):
    python scripts/train.py --no-cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depression_detector.data.data_loader import DataLoader
from depression_detector.training.trainer import ModelTrainer
from depression_detector.utils.config import get_config
from depression_detector.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the depression detection model")
    p.add_argument("--data", type=Path, default=None, help="Path to local CSV dataset")
    p.add_argument("--hf-dataset", default=None, help="HuggingFace dataset name")
    p.add_argument("--hf-token", default=None, help="HuggingFace API token")
    p.add_argument("--model-type", choices=["traditional", "bert"], default="traditional")
    p.add_argument("--classifier", default="logistic_regression",
                   choices=["logistic_regression", "svm", "random_forest", "gradient_boosting", "ensemble"])
    p.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    p.add_argument("--no-smote", action="store_true", help="Skip SMOTE oversampling")
    p.add_argument("--n-samples", type=int, default=2000, help="Synthetic dataset size")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level, "logs/train.log")
    logger = get_logger("train")

    config = get_config()
    # Apply CLI overrides to config
    config.model.model_type = args.model_type
    if args.model_type == "traditional":
        config.model.traditional.classifier = args.classifier

    # ---- Load dataset -------------------------------------------------------
    loader = DataLoader(random_seed=config.data.random_seed)

    if args.data:
        logger.info(f"Loading dataset from {args.data}")
        dataset = loader.load(args.data)
    elif args.hf_dataset:
        logger.info(f"Loading HuggingFace dataset: {args.hf_dataset}")
        dataset = loader.from_huggingface(args.hf_dataset, hf_token=args.hf_token)
    else:
        logger.info(f"No dataset specified — generating {args.n_samples} synthetic samples")
        dataset = loader.synthetic(n_samples=args.n_samples)

    logger.info(f"Dataset class distribution: {dataset.class_distribution()}")

    # ---- Split dataset -------------------------------------------------------
    train, val, test = dataset.split(
        test_size=config.data.test_size,
        val_size=config.data.val_size,
    )

    # ---- Train ---------------------------------------------------------------
    trainer = ModelTrainer(config=config)
    results = trainer.train(
        train=train,
        val=val,
        test=test,
        run_cv=not args.no_cv,
        use_smote=not args.no_smote,
        save_model=True,
    )

    # ---- Report ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    if "cross_validation" in results:
        cv = results["cross_validation"]
        logger.info(f"CV F1:  {cv['cv_f1_mean']:.4f} ± {cv['cv_f1_std']:.4f}")
        logger.info(f"CV AUC: {cv['cv_auc_mean']:.4f} ± {cv['cv_auc_std']:.4f}")
    test_r = results["test"]
    logger.info(f"TEST Accuracy:  {test_r['accuracy']:.4f}")
    logger.info(f"TEST Precision: {test_r['precision']:.4f}")
    logger.info(f"TEST Recall:    {test_r['recall']:.4f}")
    logger.info(f"TEST F1:        {test_r['f1']:.4f}")
    logger.info(f"TEST AUC-ROC:   {test_r['roc_auc']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
