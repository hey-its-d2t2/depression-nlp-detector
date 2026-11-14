#!/usr/bin/env python
"""CLI prediction script.

Usage:
    # Predict from stdin / single string:
    python scripts/predict.py --text "I feel empty and hopeless every day."

    # Predict from a text file (one sample per line):
    python scripts/predict.py --file samples.txt

    # Use a specific model and threshold:
    python scripts/predict.py --text "..." --model models/saved/model.pkl --threshold 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depression_detector.models.traditional.sklearn_classifier import TraditionalClassifier
from depression_detector.utils.logger import get_logger, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run depression detection on text")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text to predict")
    group.add_argument("--file", type=Path, help="File with one text per line")
    p.add_argument("--model", type=Path, default=None, help="Path to trained model .pkl")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output-json", type=Path, default=None, help="Save predictions to JSON file")
    return p.parse_args()


def _find_latest_model() -> Path | None:
    model_dir = Path("models/saved")
    models = sorted(model_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0] if models else None


def main() -> None:
    args = parse_args()
    setup_logger("INFO")
    logger = get_logger("predict")

    model_path = args.model or _find_latest_model()
    if not model_path:
        logger.error("No trained model found. Run scripts/train.py first.")
        sys.exit(1)

    model = TraditionalClassifier.load(model_path)

    texts = [args.text] if args.text else Path(args.file).read_text().splitlines()
    texts = [t.strip() for t in texts if t.strip()]

    logger.info(f"Predicting {len(texts)} samples with threshold={args.threshold}")
    probas = model.predict_proba(texts)

    results = []
    for text, proba in zip(texts, probas):
        label = int(proba[1] >= args.threshold)
        results.append({
            "text_preview": text[:80] + ("..." if len(text) > 80 else ""),
            "label": label,
            "label_text": "depressed" if label == 1 else "non-depressed",
            "confidence": float(proba[label]),
            "prob_depressed": float(proba[1]),
        })
        print(
            f"  [{results[-1]['label_text'].upper():>14}] "
            f"confidence={results[-1]['confidence']:.3f} | "
            f"{results[-1]['text_preview']}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Predictions saved to {args.output_json}")


if __name__ == "__main__":
    main()
