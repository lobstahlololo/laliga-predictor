"""Simple, fixed baselines for honest chronological comparison.

All baselines are evaluated on the same train/validation/test seasons as the
tree models. They are fixed before seeing test results and are never tuned on
the test season.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import evaluate_probabilities, metrics_to_row
from src.evaluation.splits import chronological_split
from src.features.preprocessing import FeaturePreprocessor
from src.utils.config import (
    BASELINE_LOGISTIC_MAX_ITER,
    BASELINE_TEST_PATH,
    BASELINE_VALIDATION_PATH,
    FEATURE_COLUMNS,
    LABEL_ORDER,
    RANDOM_STATE,
    TEST_SEASON,
    TRAIN_SEASONS,
    VALIDATION_SEASON,
)

logger = logging.getLogger(__name__)

BASELINE_LABELS = list(LABEL_ORDER)  # [2, 1, 0] = [home, draw, away]


def class_prior(y: pd.Series | np.ndarray) -> np.ndarray:
    """Empirical class frequencies in ``[home, draw, away]`` order."""
    arr = np.asarray(y, dtype=int)
    counts = np.array([(arr == label).sum() for label in BASELINE_LABELS], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.full(len(BASELINE_LABELS), 1.0 / len(BASELINE_LABELS))
    return counts / total


def majority_class(y: pd.Series | np.ndarray) -> int:
    """Most frequent training outcome; ties resolve in ``[home, draw, away]`` order."""
    prior = class_prior(y)
    return int(BASELINE_LABELS[int(np.argmax(prior))])


def _one_hot(label: int, n: int) -> np.ndarray:
    out = np.zeros((n, len(BASELINE_LABELS)), dtype=float)
    out[:, BASELINE_LABELS.index(int(label))] = 1.0
    return out


def _market_probabilities(meta: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    columns = ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]
    if not all(c in meta.columns for c in columns):
        return np.tile(fallback, (len(meta), 1))
    values = meta[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    missing = ~np.isfinite(values).all(axis=1) | (values.sum(axis=1) <= 0)
    values[missing] = fallback
    values = np.clip(values, 1e-9, None)
    values = values / values.sum(axis=1, keepdims=True)
    return values


def _fit_logistic(X_train: pd.DataFrame, y_train: pd.Series):
    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS)
    X_scaled = preprocessor.fit_transform(X_train)
    model = LogisticRegression(
        max_iter=BASELINE_LOGISTIC_MAX_ITER,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_scaled, y_train)
    return model, preprocessor


def _predict_logistic(model, preprocessor, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(preprocessor.transform(X))
    out = np.zeros((len(X), len(BASELINE_LABELS)), dtype=float)
    index = {int(c): i for i, c in enumerate(model.classes_)}
    for j, label in enumerate(BASELINE_LABELS):
        out[:, j] = proba[:, index[label]] if label in index else 0.0
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def fixed_baseline_probabilities(
    name: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    y_train: pd.Series,
) -> np.ndarray:
    """Probabilities for the non-learned baselines, aligned to [home, draw, away]."""
    n = len(X)
    prior = class_prior(y_train)
    if name == "Majority class":
        return _one_hot(majority_class(y_train), n)
    if name == "Home always":
        return _one_hot(2, n)
    if name == "Historical frequencies":
        return np.tile(prior, (n, 1))
    if name == "Market odds":
        return _market_probabilities(meta, prior)
    raise ValueError(f"Unknown fixed baseline: {name}")


FIXED_BASELINES = [
    "Majority class",
    "Home always",
    "Historical frequencies",
    "Market odds",
]


def run_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    save: bool = True,
) -> dict:
    """Fit and score every baseline on identical chronological windows.

    Validation is used for comparison only. Each baseline is then refit on
    train + validation and scored once on test. No baseline is selected using
    test data.
    """
    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx = splits["train"].index
    val_idx = splits["validation"].index
    test_idx = splits["test"].index

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    meta_val, meta_test = meta.loc[val_idx], meta.loc[test_idx]

    validation_rows: list[dict] = []
    test_rows: list[dict] = []

    for name in FIXED_BASELINES:
        val_proba = fixed_baseline_probabilities(name, X_val, meta_val, y_train)
        metrics = evaluate_probabilities(y_val, val_proba, BASELINE_LABELS, name)
        row = metrics_to_row(metrics)
        row["evaluation"] = "validation"
        validation_rows.append(row)

        test_proba = fixed_baseline_probabilities(name, X_test, meta_test, y_train)
        metrics = evaluate_probabilities(y_test, test_proba, BASELINE_LABELS, name)
        row = metrics_to_row(metrics)
        row["evaluation"] = "test"
        test_rows.append(row)

    # Logistic regression: fit on train for validation, refit on train+val for
    # the single test score.
    logistic, logistic_pre = _fit_logistic(X_train, y_train)
    val_proba = _predict_logistic(logistic, logistic_pre, X_val)
    metrics = evaluate_probabilities(
        y_val, val_proba, BASELINE_LABELS, "Logistic regression"
    )
    row = metrics_to_row(metrics)
    row["evaluation"] = "validation"
    validation_rows.append(row)

    hist_idx = train_idx.append(val_idx)
    final_pre = FeaturePreprocessor(FEATURE_COLUMNS)
    X_hist = final_pre.fit_transform(X.loc[hist_idx])
    final_logistic = LogisticRegression(
        max_iter=BASELINE_LOGISTIC_MAX_ITER,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    final_logistic.fit(X_hist, y.loc[hist_idx])
    test_proba = _predict_logistic(final_logistic, final_pre, X_test)
    metrics = evaluate_probabilities(
        y_test, test_proba, BASELINE_LABELS, "Logistic regression"
    )
    row = metrics_to_row(metrics)
    row["evaluation"] = "test"
    test_rows.append(row)

    validation = pd.DataFrame(validation_rows).sort_values("log_loss").reset_index(drop=True)
    test = pd.DataFrame(test_rows).sort_values("log_loss").reset_index(drop=True)

    if save:
        BASELINE_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        validation.to_csv(BASELINE_VALIDATION_PATH, index=False)
        test.to_csv(BASELINE_TEST_PATH, index=False)
        logger.info("Saved baseline validation/test tables to %s", BASELINE_VALIDATION_PATH.parent)

    return {
        "validation": validation,
        "test": test,
        "splits": splits,
        "logistic_model": final_logistic,
        "logistic_preprocessor": final_pre,
    }
