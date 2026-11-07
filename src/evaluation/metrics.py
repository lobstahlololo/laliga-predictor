"""Evaluation metrics for three-way football outcome probabilities.

All functions accept a ``classes`` array (the model's ``classes_``) and align
probability columns to the true labels, so a missing class in a small fold
cannot silently shift columns.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.utils.config import LABEL_ORDER, OUTCOME_LABELS, OUTCOME_SHORT

EPS = 1e-12


def _align(
    y_true: Iterable[int],
    y_proba: np.ndarray,
    classes: Iterable[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=int)
    classes_arr = np.asarray(list(classes), dtype=int)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != len(classes_arr):
        raise ValueError(
            f"Probability matrix shape {proba.shape} does not match "
            f"{len(classes_arr)} classes."
        )
    if len(y_true_arr) != len(proba):
        raise ValueError("y_true and y_proba have different lengths.")
    if not np.isfinite(proba).all():
        raise ValueError("Probability matrix contains NaN/inf.")
    if (proba < -1e-8).any():
        raise ValueError("Probability matrix contains negative values.")
    return y_true_arr, proba, classes_arr


def align_proba_to_labels(
    y_proba: np.ndarray, classes: Iterable[int], labels: Iterable[int]
) -> np.ndarray:
    """Reorder model probability columns into ``labels`` order.

    Columns for a label the model never saw are filled with zeros (they
    cannot occur), keeping log-loss well-defined for small folds.
    """
    classes_arr = list(classes)
    labels_arr = list(labels)
    out = np.zeros((len(y_proba), len(labels_arr)), dtype=float)
    for j, label in enumerate(labels_arr):
        if label in classes_arr:
            out[:, j] = y_proba[:, classes_arr.index(label)]
    return out


def multiclass_log_loss(
    y_true: Iterable[int],
    y_proba: np.ndarray,
    classes: Iterable[int],
) -> float:
    """Label-aligned multiclass log-loss.

    Implemented explicitly rather than delegating to ``sklearn.log_loss`` so
    the probability columns are guaranteed to match the label order (our
    labels are 2/1/0, which is not lexicographic).
    """
    y_true_arr, proba, classes_arr = _align(y_true, y_proba, classes)
    index = {int(c): i for i, c in enumerate(classes_arr)}
    if any(int(v) not in index for v in y_true_arr):
        raise ValueError("y_true contains a label absent from classes.")
    true_probs = np.array([
        proba[row, index[int(label)]] for row, label in enumerate(y_true_arr)
    ])
    return float(-np.mean(np.log(np.clip(true_probs, EPS, 1.0))))


def multiclass_brier_score(
    y_true: Iterable[int],
    y_proba: np.ndarray,
    classes: Iterable[int],
) -> float:
    """Mean squared error across the three one-hot probability columns.

    Lower is better; a perfectly calibrated deterministic forecast scores 0.
    """
    y_true_arr, proba, classes_arr = _align(y_true, y_proba, classes)
    one_hot = (y_true_arr[:, None] == classes_arr[None, :]).astype(float)
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def calibration_table(
    y_true: Iterable[int],
    y_proba: np.ndarray,
    classes: Iterable[int],
    n_bins: int = 10,
) -> pd.DataFrame:
    """One-vs-rest reliability bins for every outcome class.

    Returns columns: class, bin, count, mean_confidence, observed_rate.
    """
    y_true_arr, proba, classes_arr = _align(y_true, y_proba, classes)
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for j, cls in enumerate(classes_arr):
        conf = proba[:, j]
        observed = (y_true_arr == cls).astype(float)
        bin_idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            rows.append({
                "class": int(cls),
                "label": OUTCOME_LABELS.get(int(cls), str(cls)),
                "bin": b,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "observed_rate": float(observed[mask].mean()),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(
    y_true: Iterable[int],
    y_proba: np.ndarray,
    classes: Iterable[int],
    n_bins: int = 10,
) -> float:
    """Count-weighted mean gap between confidence and observed frequency."""
    table = calibration_table(y_true, y_proba, classes, n_bins)
    if table.empty:
        return float("nan")
    weights = table["count"] / table["count"].sum()
    gaps = (table["mean_confidence"] - table["observed_rate"]).abs()
    return float((weights * gaps).sum())


def evaluate_probabilities(
    y_true: Iterable[int],
    y_proba: np.ndarray,
    classes: Iterable[int],
    model_name: str = "model",
) -> dict:
    """Full metric bundle for one model on one window.

    Includes accuracy, log-loss, multiclass Brier, per-class
    precision/recall/F1, draw recall, confusion matrix and calibration.
    """
    y_true_arr, proba, classes_arr = _align(y_true, y_proba, classes)
    labels = [c for c in LABEL_ORDER if c in set(classes_arr) or c in set(y_true_arr)]
    if not labels:
        labels = sorted(set(classes_arr))

    proba_aligned = align_proba_to_labels(proba, classes_arr, labels)
    # Re-normalise in case a label was absent from the model's classes.
    totals = proba_aligned.sum(axis=1, keepdims=True)
    proba_aligned = np.divide(
        proba_aligned, totals, out=np.zeros_like(proba_aligned), where=totals > 0
    )
    predicted = np.asarray(labels)[proba_aligned.argmax(axis=1)]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, predicted, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame({
        "outcome": [OUTCOME_LABELS.get(c, str(c)) for c in labels],
        "code": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    cm = pd.DataFrame(
        confusion_matrix(y_true_arr, predicted, labels=labels),
        index=[f"Actual {OUTCOME_LABELS.get(c, c)}" for c in labels],
        columns=[f"Pred {OUTCOME_LABELS.get(c, c)}" for c in labels],
    )

    draw_idx = labels.index(1) if 1 in labels else None
    metrics = {
        "model": model_name,
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(accuracy_score(y_true_arr, predicted)),
        "log_loss": multiclass_log_loss(y_true_arr, proba_aligned, labels),
        "brier_score": multiclass_brier_score(y_true_arr, proba_aligned, labels),
        "macro_f1": float(np.mean(f1)),
        "draw_recall": float(recall[draw_idx]) if draw_idx is not None else float("nan"),
        "draw_precision": float(precision[draw_idx]) if draw_idx is not None else float("nan"),
        "ece": expected_calibration_error(y_true_arr, proba_aligned, labels),
        "prediction_distribution": {
            OUTCOME_SHORT.get(c, str(c)).lower(): float((predicted == c).mean())
            for c in labels
        },
        "confusion_matrix": cm,
        "per_class": per_class,
        "classes": labels,
    }
    return metrics


def metrics_to_row(metrics: dict) -> dict:
    """Flatten a metrics dict into a CSV-friendly row."""
    return {
        "model": metrics["model"],
        "n_samples": metrics["n_samples"],
        "accuracy": metrics["accuracy"],
        "log_loss": metrics["log_loss"],
        "brier_score": metrics["brier_score"],
        "macro_f1": metrics["macro_f1"],
        "draw_recall": metrics["draw_recall"],
        "draw_precision": metrics["draw_precision"],
        "ece": metrics["ece"],
        **{f"pred_share_{key}": value for key, value in metrics["prediction_distribution"].items()},
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Comparison table sorted by log-loss (primary) then Brier (tiebreak)."""
    rows = [metrics_to_row(r) for r in results]
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            ["log_loss", "brier_score"], ascending=True
        ).reset_index(drop=True)
    return table
