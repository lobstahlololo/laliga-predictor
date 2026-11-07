"""Constrained probability ensembles with validation-only weight selection.

Weights are non-negative, sum to one, and are chosen by evaluating a fixed grid
of simplex weights on validation predictions only. The test season is never an
input to this module's selection function. Equal weights are part of the grid,
so a selected ensemble can never be worse than equal weighting on the same
validation rows.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    evaluate_probabilities,
    multiclass_brier_score,
    multiclass_log_loss,
)
from src.utils.config import LABEL_ORDER

EPS = 1e-12


def normalise_probabilities(probabilities: np.ndarray) -> np.ndarray:
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError(f"Expected an (n, 3) probability matrix, got {proba.shape}.")
    proba = np.clip(proba, EPS, None)
    return proba / proba.sum(axis=1, keepdims=True)


def blend_probabilities(
    probabilities: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted probability blend in ``[home, draw, away]`` order."""
    names = [name for name, value in weights.items() if float(value) > 0]
    if not names:
        raise ValueError("At least one positive ensemble weight is required.")
    missing = [name for name in names if name not in probabilities]
    if missing:
        raise ValueError(f"Missing component probabilities: {missing}")
    arrays = [normalise_probabilities(probabilities[name]) for name in names]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("Ensemble components have different row counts.")
    total = sum(float(weights[name]) for name in names)
    blended = sum(
        normalise_probabilities(probabilities[name]) * (float(weights[name]) / total)
        for name in names
    )
    return normalise_probabilities(blended)


def _simplex_grid(n_components: int, step: float) -> list[tuple[float, ...]]:
    steps = int(round(1.0 / step))
    if steps < 1:
        raise ValueError("step must be <= 1 and divide 1 reasonably.")
    out = []
    for values in product(range(steps + 1), repeat=n_components):
        if sum(values) != steps:
            continue
        out.append(tuple(value / steps for value in values))
    return out


def select_ensemble_weights(
    probabilities: dict[str, np.ndarray],
    y: np.ndarray,
    *,
    step: float = 0.05,
    top_n: int = 20,
) -> dict:
    """Choose non-negative simplex weights on validation data.

    Returns the best weights, the validation table, and the equal-weight
    reference. The grid is deterministic; ties resolve toward the earlier
    component order in the supplied dictionary.
    """
    names = list(probabilities)
    if not names:
        raise ValueError("No ensemble components supplied.")
    lengths = {len(normalise_probabilities(probabilities[name])) for name in names}
    if len(lengths) != 1:
        raise ValueError("Ensemble components have different row counts.")
    y_array = np.asarray(y, dtype=int)
    if len(y_array) != next(iter(lengths)):
        raise ValueError("Labels and component probabilities have different lengths.")

    rows = []
    for values in _simplex_grid(len(names), step):
        weights = dict(zip(names, values))
        blended = blend_probabilities(probabilities, weights)
        loss = multiclass_log_loss(y_array, blended, LABEL_ORDER)
        brier = multiclass_brier_score(y_array, blended, LABEL_ORDER)
        rows.append({
            "weights": weights,
            "log_loss": float(loss),
            "brier_score": float(brier),
            **{f"weight_{name}": float(weight) for name, weight in weights.items()},
        })
    table = pd.DataFrame(rows).sort_values(
        ["log_loss", "brier_score"]
    ).reset_index(drop=True)
    equal = dict(zip(names, [1.0 / len(names)] * len(names)))
    equal_proba = blend_probabilities(probabilities, equal)
    equal_metrics = evaluate_probabilities(
        y_array, equal_proba, LABEL_ORDER, "Equal-weight ensemble"
    )
    best = table.iloc[0]
    best_weights = {
        name: float(best[f"weight_{name}"]) for name in names
    }
    best_weights = {name: weight for name, weight in best_weights.items() if weight > 0}
    return {
        "weights": best_weights,
        "all_weights": {name: float(best[f"weight_{name}"]) for name in names},
        "validation_log_loss": float(best["log_loss"]),
        "validation_brier_score": float(best["brier_score"]),
        "equal_weights": equal,
        "equal_validation_log_loss": float(equal_metrics["log_loss"]),
        "equal_validation_brier_score": float(equal_metrics["brier_score"]),
        "table": table.head(top_n).reset_index(drop=True),
        "grid_step": float(step),
        "selection": "validation log-loss; test not used",
    }
