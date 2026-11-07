"""Rolling-origin (walk-forward) backtesting for both models.

Each fold fits a fresh preprocessor and model on all seasons before the test
season, then scores the following season. This is the honest estimate of how
the approach would have performed in production over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_probabilities, metrics_to_row
from src.evaluation.splits import walk_forward_folds
from src.features.preprocessing import FeaturePreprocessor
from src.utils.config import FEATURE_COLUMNS, MIN_TRAIN_SEASONS

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """A model factory plus how it should be fitted."""

    name: str
    factory: Callable[[], object]
    uses_sample_weight: bool = False


def balanced_sample_weights(y: pd.Series | np.ndarray) -> np.ndarray:
    """Inverse-frequency weights: n / (n_classes * class_count)."""
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    weight_map = {
        int(c): len(y_arr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)
    }
    return np.array([weight_map[int(v)] for v in y_arr], dtype=float)


def fit_model(spec: ModelSpec, model, X_train: pd.DataFrame, y_train: pd.Series):
    """Fit one model, applying balanced weights where the spec asks for them."""
    if spec.uses_sample_weight:
        weights = balanced_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)
    return model


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    specs: list[ModelSpec],
    min_train_seasons: int = MIN_TRAIN_SEASONS,
    include_test_season: bool = False,
    test_season: str | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate every model on every rolling-origin fold.

    Returns ``(results_dataframe, metrics_list)``. ``include_test_season`` is
    False by default so the final test season stays untouched; the final test
    score is produced by ``train_model`` after model selection.
    """
    folds = walk_forward_folds(meta, min_train_seasons=min_train_seasons)
    if not include_test_season and test_season is not None:
        folds = [f for f in folds if f.test_season != test_season]
    if not folds:
        logger.warning("Not enough seasons for walk-forward evaluation.")
        return pd.DataFrame(), []

    rows: list[dict] = []
    detail: list[dict] = []
    for fold in folds:
        train_idx, test_idx = fold.train_index, fold.test_index
        preprocessor = FeaturePreprocessor(FEATURE_COLUMNS)
        X_train = preprocessor.fit_transform(X.loc[train_idx])
        X_test = preprocessor.transform(X.loc[test_idx])
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        for spec in specs:
            model = spec.factory()
            fit_model(spec, model, X_train, y_train)
            proba = model.predict_proba(X_test)
            metrics = evaluate_probabilities(
                y_test, proba, model.classes_, model_name=spec.name
            )
            row = metrics_to_row(metrics)
            row.update({
                "fold": fold.fold,
                "test_season": fold.test_season,
                "train_seasons": "+".join(fold.train_seasons),
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
            })
            rows.append(row)
            detail.append({
                "fold": fold,
                "model": spec.name,
                "metrics": metrics,
            })
            logger.info(
                "%s | %s | log-loss=%.4f acc=%.4f brier=%.4f",
                fold.describe(), spec.name, metrics["log_loss"],
                metrics["accuracy"], metrics["brier_score"],
            )

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(["fold", "model"]).reset_index(drop=True)
    return results, detail


def summarise_walk_forward(results: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each metric per model across folds."""
    if results.empty:
        return results
    metric_cols = [
        c for c in ("accuracy", "log_loss", "brier_score", "draw_recall", "ece")
        if c in results.columns
    ]
    grouped = results.groupby("model")[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index().sort_values("log_loss_mean")
