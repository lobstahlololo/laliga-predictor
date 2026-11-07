"""Validation-only probability calibration for three-way outcomes.

Method selection uses only the validation season: the first part fits a
calibrator and the later part scores it. The untouched test season is used
once, after the method has been chosen, to compare calibrated and uncalibrated
probabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import evaluate_probabilities, metrics_to_row
from src.utils.config import CALIBRATION_HOLDOUT_FRACTION, LABEL_ORDER

logger = logging.getLogger(__name__)

CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
EPS = 1e-6


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


@dataclass
class IdentityCalibrator:
    method: str = "none"

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        proba = np.asarray(probabilities, dtype=float)
        proba = np.clip(proba, 1e-12, None)
        return proba / proba.sum(axis=1, keepdims=True)


class _BinarySigmoid:
    """Platt scaling on the logit of one class probability."""

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.constant: float | None = None

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> "_BinarySigmoid":
        target = np.asarray(target, dtype=int)
        if len(np.unique(target)) < 2:
            self.constant = float(target.mean())
            return self
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.model.fit(_logit(probabilities).reshape(-1, 1), target)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(probabilities), self.constant, dtype=float)
        if self.model is None:
            return np.clip(probabilities, 0.0, 1.0)
        return self.model.predict_proba(
            _logit(probabilities).reshape(-1, 1)
        )[:, 1]


class _BinaryIsotonic:
    """Isotonic regression for one class probability."""

    def __init__(self) -> None:
        self.model: IsotonicRegression | None = None
        self.constant: float | None = None

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> "_BinaryIsotonic":
        target = np.asarray(target, dtype=int)
        x = np.asarray(probabilities, dtype=float)
        if len(np.unique(target)) < 2 or len(np.unique(x)) < 2:
            self.constant = float(target.mean())
            return self
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.model.fit(x, target)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(probabilities), self.constant, dtype=float)
        if self.model is None:
            return np.clip(probabilities, 0.0, 1.0)
        return np.clip(self.model.predict(np.asarray(probabilities, dtype=float)), 0.0, 1.0)


class MulticlassCalibrator:
    """One-vs-rest calibration followed by row normalisation."""

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"sigmoid", "isotonic"}:
            raise ValueError(f"Unsupported calibration method: {method}")
        self.method = method
        self.calibrators: list[_BinarySigmoid | _BinaryIsotonic] = []

    def fit(
        self,
        probabilities: np.ndarray,
        y: np.ndarray,
        classes: list[int] | None = None,
    ) -> "MulticlassCalibrator":
        proba = np.asarray(probabilities, dtype=float)
        target = np.asarray(y, dtype=int)
        classes = list(classes or LABEL_ORDER)
        self.calibrators = []
        for j, label in enumerate(classes):
            calibrator = _BinarySigmoid() if self.method == "sigmoid" else _BinaryIsotonic()
            calibrator.fit(proba[:, j], (target == int(label)).astype(int))
            self.calibrators.append(calibrator)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        proba = np.asarray(probabilities, dtype=float)
        if not self.calibrators:
            raise RuntimeError("Calibrator must be fitted before transform().")
        out = np.column_stack([
            calibrator.transform(proba[:, j])
            for j, calibrator in enumerate(self.calibrators)
        ])
        out = np.clip(out, 1e-9, None)
        totals = out.sum(axis=1, keepdims=True)
        # A pathological all-zero row falls back to the raw distribution.
        bad = (totals[:, 0] <= 0) | ~np.isfinite(totals[:, 0])
        if bad.any():
            fallback = np.clip(proba[bad], 1e-9, None)
            fallback = fallback / fallback.sum(axis=1, keepdims=True)
            out[bad] = fallback
            totals = out.sum(axis=1, keepdims=True)
        return out / totals


def fit_calibrator(
    method: str,
    probabilities: np.ndarray,
    y: np.ndarray,
    classes: list[int] | None = None,
):
    """Fit one of ``none``, ``sigmoid`` or ``isotonic``."""
    if method == "none":
        return IdentityCalibrator().fit(probabilities, y)
    return MulticlassCalibrator(method).fit(probabilities, y, classes)


def select_calibration_method(
    validation_probabilities: np.ndarray,
    y_validation: np.ndarray,
    validation_dates,
    classes: list[int] | None = None,
    holdout_fraction: float = CALIBRATION_HOLDOUT_FRACTION,
) -> tuple[str, pd.DataFrame]:
    """Choose a calibrator using only a chronological validation split.

    The early validation rows fit each calibrator and the later rows score it.
    This prevents the final test season from choosing the calibration method.
    """
    proba = np.asarray(validation_probabilities, dtype=float)
    y = np.asarray(y_validation, dtype=int)
    dates = pd.to_datetime(pd.Series(validation_dates)).to_numpy()
    order = np.argsort(dates, kind="stable")
    proba, y = proba[order], y[order]
    n = len(y)
    split = int(round(n * (1.0 - holdout_fraction)))
    if split < 20 or n - split < 20:
        return "none", pd.DataFrame([{
            "method": "none", "n_fit": n, "n_eval": 0,
            "log_loss": np.nan, "brier_score": np.nan, "ece": np.nan,
            "accuracy": np.nan, "note": "Validation window too small to select.",
        }])

    rows = []
    for method in CALIBRATION_METHODS:
        calibrator = fit_calibrator(method, proba[:split], y[:split], classes)
        calibrated = calibrator.transform(proba[split:])
        metrics = evaluate_probabilities(
            y[split:], calibrated, list(classes or LABEL_ORDER), method
        )
        rows.append({
            "method": method,
            "n_fit": split,
            "n_eval": n - split,
            "log_loss": metrics["log_loss"],
            "brier_score": metrics["brier_score"],
            "ece": metrics["ece"],
            "accuracy": metrics["accuracy"],
            "draw_recall": metrics["draw_recall"],
            "note": "Chronological validation holdout.",
        })
    table = pd.DataFrame(rows).sort_values(["log_loss", "brier_score"]).reset_index(drop=True)
    best = str(table.iloc[0]["method"])
    logger.info("Calibration selected on validation holdout: %s", best)
    return best, table


def compare_calibrated_test(
    y_test,
    uncalibrated_probabilities: np.ndarray,
    fitted_calibrators: dict[str, object],
    classes: list[int] | None = None,
) -> pd.DataFrame:
    """One-shot comparison of uncalibrated and calibrated test probabilities."""
    labels = list(classes or LABEL_ORDER)
    rows = []
    entries = {"none": uncalibrated_probabilities, **{
        method: calibrator.transform(uncalibrated_probabilities)
        for method, calibrator in fitted_calibrators.items()
    }}
    for method, probabilities in entries.items():
        metrics = evaluate_probabilities(y_test, probabilities, labels, method)
        row = metrics_to_row(metrics)
        row["calibration"] = method
        row.pop("model", None)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["log_loss", "brier_score"]).reset_index(drop=True)
