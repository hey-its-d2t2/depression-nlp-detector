"""Model evaluation: metrics, reports, and visualisations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from depression_detector.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Computes, formats, and persists comprehensive evaluation metrics."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._results: dict = {}

    def evaluate(
        self,
        y_true: list[int] | np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "model",
    ) -> dict:
        from sklearn.metrics import (
            accuracy_score, classification_report, confusion_matrix,
            f1_score, precision_score, recall_score, roc_auc_score,
            average_precision_score,
        )

        y_true = np.asarray(y_true)
        y_pred = (y_proba[:, 1] >= self.threshold).astype(int)

        results = {
            "model_name": model_name,
            "threshold": self.threshold,
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "n_negative": int((y_true == 0).sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba[:, 1])),
            "avg_precision": float(average_precision_score(y_true, y_proba[:, 1])),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred,
                target_names=["non-depressed", "depressed"],
                output_dict=True,
            ),
        }

        self._results[model_name] = results
        self._log_summary(results)
        return results

    def _log_summary(self, r: dict) -> None:
        logger.info(
            f"[{r['model_name']}] "
            f"Acc={r['accuracy']:.4f} | "
            f"P={r['precision']:.4f} | "
            f"R={r['recall']:.4f} | "
            f"F1={r['f1']:.4f} | "
            f"AUC={r['roc_auc']:.4f}"
        )

    def threshold_sweep(
        self,
        y_true: list[int] | np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[list[float]] = None,
    ) -> list[dict]:
        """Evaluate metrics across a range of decision thresholds."""
        from sklearn.metrics import f1_score, precision_score, recall_score

        y_true = np.asarray(y_true)
        thresholds = thresholds or [i / 20 for i in range(1, 20)]
        rows = []
        for t in thresholds:
            y_pred = (y_proba[:, 1] >= t).astype(int)
            rows.append({
                "threshold": t,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            })
        return rows

    def save_report(self, output_dir: str | Path, model_name: Optional[str] = None) -> None:
        """Save evaluation artefacts (JSON report + confusion matrix plot)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, r in self._results.items():
            if model_name and name != model_name:
                continue

            report_path = output_dir / f"{name}_report.json"
            with open(report_path, "w") as f:
                json.dump(r, f, indent=2)
            logger.info(f"Report saved to {report_path}")

            cm = np.array(r["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Non-Depressed", "Depressed"],
            )
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(f"{name} — Confusion Matrix")
            plt.tight_layout()
            plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
            plt.close()

    def compare_models(self) -> list[dict]:
        """Return a comparison list of all evaluated models, sorted by F1."""
        rows = [
            {
                "model": name,
                "accuracy": r["accuracy"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "roc_auc": r["roc_auc"],
                "avg_precision": r["avg_precision"],
            }
            for name, r in self._results.items()
        ]
        return sorted(rows, key=lambda x: x["f1"], reverse=True)
