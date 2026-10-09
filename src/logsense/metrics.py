"""Shared classification metrics, so baselines (Phase 2) and the fine-tuned model (Phase 3)
are scored the same way."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class F1, and confusion matrix (rows = true, cols = predicted).

    Macro-F1 averages F1 over classes with equal weight, so a model that ignores one class
    is penalized even if overall accuracy looks high.
    """
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_f1": {label: float(f) for label, f in zip(labels, per_class, strict=True)},
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }
