#!/usr/bin/env python
"""Evaluation script — run on a test set and produce a detailed report.

Usage:
    python scripts/evaluate.py --model models/saved/traditional_logistic_regression.pkl \
                                --data data/processed/test.csv \
                                --output reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depression_detector.data.data_loader import DataLoader
from depression_detector.evaluation.metrics import ModelEvaluator
from depression_detector.models.traditional.sklearn_classifier import TraditionalClassifier
from depression_detector.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained depression detection model")
    p.add_argument("--model", type=Path, required=True, help="Path to saved model .pkl")
    p.add_argument("--data", type=Path, required=False, help="Test CSV path (falls back to synthetic)")
    p.add_argument("--output", type=Path, default=Path("reports"), help="Report output directory")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level, "logs/evaluate.log")
    logger = get_logger("evaluate")

    logger.info(f"Loading model from {args.model}")
    model = TraditionalClassifier.load(args.model)

    loader = DataLoader()
    if args.data and args.data.exists():
        dataset = loader.load(args.data)
    else:
        logger.warning("No test data provided — using 500 synthetic samples")
        dataset = loader.synthetic(n_samples=500)

    _, _, test = dataset.split()

    evaluator = ModelEvaluator(threshold=args.threshold)
    results = evaluator.evaluate(test.labels, model.predict_proba(test.texts), model_name=model.model_name)

    # Threshold sweep
    import polars as pl
    sweep_rows = evaluator.threshold_sweep(test.labels, model.predict_proba(test.texts))
    sweep_path = args.output / f"{model.model_name}_threshold_sweep.csv"
    args.output.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(sweep_rows).write_csv(str(sweep_path))
    logger.info(f"Threshold sweep saved to {sweep_path}")

    evaluator.save_report(args.output)

    logger.info("Evaluation complete.")
    logger.info(f"F1={results['f1']:.4f} | AUC={results['roc_auc']:.4f}")

    # Feature importance (traditional models only)
    if hasattr(model, "get_top_features"):
        top = model.get_top_features(n=20)
        fi_path = args.output / f"{model.model_name}_feature_importance.json"
        with open(fi_path, "w") as f:
            json.dump(top, f, indent=2)
        logger.info(f"Feature importance saved to {fi_path}")


if __name__ == "__main__":
    main()
