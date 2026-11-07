"""Prediction uncertainty and model-agreement diagnostics.

A high model probability is not the same thing as certainty. This module keeps
five separate signals:

* probability confidence (the largest class probability);
* normalised entropy (spread across the three outcomes);
* the margin between the top two probabilities;
* agreement between different model families;
* data/history reliability (missing inputs, Elo rating deviation, matches seen).

The resulting levels are descriptive. They are not calibrated confidence
intervals and must not be presented as guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import LABEL_ORDER, OUTCOME_LABELS

EPS = 1e-12


def _normalise(probabilities: np.ndarray) -> np.ndarray:
    proba = np.clip(np.asarray(probabilities, dtype=float), EPS, None)
    return proba / proba.sum(axis=1, keepdims=True)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    proba = _normalise(probabilities)
    terms = np.where(proba > 0, proba * np.log(proba), 0.0)
    return -terms.sum(axis=1) / np.log(3.0)


def prediction_margin(probabilities: np.ndarray) -> np.ndarray:
    proba = np.sort(_normalise(probabilities), axis=1)
    return proba[:, -1] - proba[:, -2]


def model_agreement(component_probabilities: list[np.ndarray]) -> np.ndarray:
    """One minus mean pairwise total-variation distance.

    1.0 means every component gives the same distribution; 0.0 is maximal
    disagreement. The measure is descriptive and does not weight model quality.
    """
    if not component_probabilities:
        return np.ones(0, dtype=float)
    stack = np.stack([_normalise(p) for p in component_probabilities], axis=0)
    if stack.shape[0] == 1:
        return np.ones(stack.shape[1], dtype=float)
    distances = []
    for i in range(stack.shape[0]):
        for j in range(i + 1, stack.shape[0]):
            distances.append(0.5 * np.abs(stack[i] - stack[j]).sum(axis=1))
    return 1.0 - np.mean(np.vstack(distances), axis=0)


def _level(entropy: float, agreement: float, confidence: float) -> str:
    if entropy >= 0.90 or agreement <= 0.65 or confidence < 0.45:
        return "high"
    if entropy >= 0.80 or agreement <= 0.80 or confidence < 0.55:
        return "medium"
    return "low"


def _reliability(min_elo_matches: float, min_poisson_matches: float, known_both: bool) -> str:
    matches = min(min_elo_matches, min_poisson_matches)
    if not known_both or matches < 5:
        return "low"
    if matches < 20:
        return "medium"
    return "high"


def build_uncertainty_table(
    meta: pd.DataFrame,
    ensemble_probabilities: np.ndarray,
    component_probabilities: dict[str, np.ndarray],
    elo_features: pd.DataFrame,
    poisson_predictions: pd.DataFrame,
    base_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build one uncertainty row per fixture and a measured summary."""
    if len(meta) != len(ensemble_probabilities):
        raise ValueError("Metadata and ensemble probabilities have different lengths.")
    ensemble = _normalise(ensemble_probabilities)
    components = list(component_probabilities.values())
    agreement = model_agreement(components)
    entropy = normalized_entropy(ensemble)
    margin = prediction_margin(ensemble)
    confidence = ensemble.max(axis=1)
    stack = np.stack([_normalise(p) for p in components], axis=0) if components else np.zeros((1, len(meta), 3))
    component_std = stack.std(axis=0).mean(axis=1) if components else np.zeros(len(meta))
    component_deviation = (
        np.mean(np.abs(stack - ensemble[None, :, :]), axis=(0, 2))
        if components else np.zeros(len(meta))
    )

    elo = elo_features.reindex(meta["match_id"].to_numpy())
    poisson = poisson_predictions.reindex(meta["match_id"].to_numpy())
    base_missing = (
        base_features.isna().sum(axis=1).to_numpy()
        if base_features is not None else np.zeros(len(meta), dtype=int)
    )

    rows = []
    for i, (_, row) in enumerate(meta.iterrows()):
        home = str(row["home_team"])
        away = str(row["away_team"])
        min_elo_matches = float(
            min(
                elo.iloc[i].get("home_elo_matches", 0),
                elo.iloc[i].get("away_elo_matches", 0),
            )
        )
        min_poisson_matches = float(
            min(
                poisson.iloc[i].get("home_matches", 0),
                poisson.iloc[i].get("away_matches", 0),
            )
        )
        known_both = bool(
            poisson.iloc[i].get("home_known", False)
            and poisson.iloc[i].get("away_known", False)
        )
        actual = int(row["match_outcome"]) if pd.notna(row["match_outcome"]) else None
        rows.append({
            "match_id": row["match_id"],
            "date": row["date"],
            "season": row.get("season"),
            "home_team": home,
            "away_team": away,
            "actual_outcome": actual,
            "prob_home": float(ensemble[i, 0]),
            "prob_draw": float(ensemble[i, 1]),
            "prob_away": float(ensemble[i, 2]),
            "confidence": float(confidence[i]),
            "entropy": float(entropy[i]),
            "margin": float(margin[i]),
            "model_agreement": float(agreement[i]) if len(agreement) else 1.0,
            "component_std": float(component_std[i]),
            "component_deviation": float(component_deviation[i]),
            "home_elo_rd": float(elo.iloc[i].get("home_elo_rd", np.nan)),
            "away_elo_rd": float(elo.iloc[i].get("away_elo_rd", np.nan)),
            "min_elo_matches": min_elo_matches,
            "min_poisson_matches": min_poisson_matches,
            "poisson_known_both": known_both,
            "missing_base_features": int(base_missing[i]),
            "historical_reliability": _reliability(
                min_elo_matches, min_poisson_matches, known_both
            ),
            "uncertainty_level": _level(float(entropy[i]), float(agreement[i]), float(confidence[i])),
        })

    table = pd.DataFrame(rows)
    if "actual_outcome" in table and table["actual_outcome"].notna().any():
        actual = table["actual_outcome"].astype("Int64")
        probs = table[["prob_home", "prob_draw", "prob_away"]].to_numpy(float)
        index = {label: j for j, label in enumerate(LABEL_ORDER)}
        true_prob = np.array([
            probs[i, index[int(label)]] if pd.notna(label) else np.nan
            for i, label in enumerate(actual)
        ])
        table["row_log_loss"] = -np.log(np.clip(true_prob, EPS, 1.0))
        table["model_correct"] = (
            table[["prob_home", "prob_draw", "prob_away"]].to_numpy(float).argmax(axis=1)
            == np.array([index[int(label)] if pd.notna(label) else -1 for label in actual])
        ).astype(int)
    else:
        table["row_log_loss"] = np.nan
        table["model_correct"] = np.nan

    summary = {
        "n": int(len(table)),
        "level_counts": table["uncertainty_level"].value_counts().to_dict(),
        "mean_confidence": float(table["confidence"].mean()),
        "mean_entropy": float(table["entropy"].mean()),
        "mean_agreement": float(table["model_agreement"].mean()),
        "high_uncertainty_share": float((table["uncertainty_level"] == "high").mean()),
        "reliability_counts": table["historical_reliability"].value_counts().to_dict(),
        "correlation_entropy_loss": float(
            table[["entropy", "row_log_loss"]].corr(method="spearman").iloc[0, 1]
        ) if table["row_log_loss"].notna().any() else None,
        "correlation_agreement_loss": float(
            table[["model_agreement", "row_log_loss"]].corr(method="spearman").iloc[0, 1]
        ) if table["row_log_loss"].notna().any() else None,
        "mean_loss_by_level": table.groupby("uncertainty_level")["row_log_loss"].mean().to_dict(),
        "accuracy_by_level": table.groupby("uncertainty_level")["model_correct"].mean().to_dict(),
        "note": (
            "Uncertainty levels are descriptive diagnostics, not calibrated "
            "confidence intervals. High probability is not certainty."
        ),
    }
    return table, summary
