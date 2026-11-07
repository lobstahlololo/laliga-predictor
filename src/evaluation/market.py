"""Market-odds extraction, overround removal and benchmark analysis.

Bookmaker decimal odds imply probabilities that sum to more than one. Every
comparison here explicitly removes that overround first. Market odds are treated
as a strong benchmark, not as an opponent to be beaten by narrative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import (
    align_proba_to_labels,
    evaluate_probabilities,
    metrics_to_row,
)
from src.utils.config import LABEL_ORDER, OUTCOME_LABELS, RANDOM_STATE

EPS = 1e-9
MARKET_COLUMNS = ("market_home", "market_draw", "market_away")


def implied_probabilities(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return raw implied probabilities and overround from decimal odds.

    Raw implied probability is ``1 / decimal_odds``. The overround is
    ``sum(raw) - 1``; probabilities after removal are ``raw / sum(raw)``.
    Invalid rows fall back to a uniform distribution and overround zero.
    """
    odds = np.asarray(odds, dtype=float)
    if odds.ndim != 2 or odds.shape[1] != 3:
        raise ValueError(f"Expected an (n, 3) odds matrix, got {odds.shape}.")
    valid = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
    raw = np.full_like(odds, np.nan, dtype=float)
    raw[valid] = 1.0 / odds[valid]
    totals = raw.sum(axis=1)
    valid &= np.isfinite(totals) & (totals > 0)
    out = np.full_like(odds, 1.0 / 3.0, dtype=float)
    overround = np.zeros(len(odds), dtype=float)
    out[valid] = raw[valid] / totals[valid, None]
    overround[valid] = totals[valid] - 1.0
    return out, overround


def market_from_meta(meta: pd.DataFrame, fallback: np.ndarray | None = None) -> pd.DataFrame:
    """Extract normalized market probabilities from a metadata frame."""
    odds_columns = ["odds_home", "odds_draw", "odds_away"]
    if all(column in meta.columns for column in odds_columns):
        probabilities, overround = implied_probabilities(
            meta[odds_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
    else:
        implied = ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]
        if not all(column in meta.columns for column in implied):
            raise ValueError("Metadata has neither odds nor implied-probability columns.")
        raw = meta[implied].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        bad = ~np.isfinite(raw).all(axis=1) | (raw.sum(axis=1) <= 0)
        raw[bad] = (
            np.asarray(fallback, dtype=float)
            if fallback is not None else np.full(3, 1.0 / 3.0)
        )
        raw = np.clip(raw, EPS, None)
        totals = raw.sum(axis=1)
        probabilities = raw / totals[:, None]
        overround = totals - 1.0
    if fallback is not None:
        bad = ~np.isfinite(probabilities).all(axis=1) | (probabilities.sum(axis=1) <= 0)
        probabilities[bad] = np.asarray(fallback, dtype=float)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        probabilities, columns=list(MARKET_COLUMNS), index=meta.index
    )
    frame["overround"] = overround
    frame["raw_implied_sum"] = 1.0 + overround
    return frame


def overround_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-season overround statistics from decimal odds."""
    required = {"season", "odds_home", "odds_draw", "odds_away"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Overround input missing: {sorted(missing)}")
    probabilities, overround = implied_probabilities(
        raw[["odds_home", "odds_draw", "odds_away"]].to_numpy(float)
    )
    frame = raw[["season"]].copy()
    frame["overround"] = overround
    frame["raw_implied_sum"] = 1.0 + overround
    frame["valid"] = np.isfinite(probabilities).all(axis=1)
    rows = []
    for season, part in frame.groupby("season"):
        valid = part[part["valid"]]
        rows.append({
            "season": season,
            "matches": int(len(part)),
            "valid_odds": int(len(valid)),
            "mean_overround": float(valid["overround"].mean()) if not valid.empty else np.nan,
            "median_overround": float(valid["overround"].median()) if not valid.empty else np.nan,
            "min_overround": float(valid["overround"].min()) if not valid.empty else np.nan,
            "max_overround": float(valid["overround"].max()) if not valid.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def compare_model_to_market(
    y_true,
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    model_name: str,
) -> dict:
    """Model and market metrics plus probability-difference diagnostics."""
    y = np.asarray(y_true, dtype=int)
    model = np.asarray(model_probabilities, dtype=float)
    market = np.asarray(market_probabilities, dtype=float)
    if model.shape != market.shape:
        raise ValueError("Model and market probability matrices differ in shape.")
    model_metrics = evaluate_probabilities(y, model, LABEL_ORDER, model_name)
    market_metrics = evaluate_probabilities(y, market, LABEL_ORDER, "Market odds")
    diff = model - market
    row = {
        "model": model_name,
        "n_samples": int(len(y)),
        "model_log_loss": model_metrics["log_loss"],
        "market_log_loss": market_metrics["log_loss"],
        "delta_log_loss": model_metrics["log_loss"] - market_metrics["log_loss"],
        "model_brier_score": model_metrics["brier_score"],
        "market_brier_score": market_metrics["brier_score"],
        "delta_brier_score": model_metrics["brier_score"] - market_metrics["brier_score"],
        "model_accuracy": model_metrics["accuracy"],
        "market_accuracy": market_metrics["accuracy"],
        "mean_abs_probability_difference": float(np.mean(np.abs(diff))),
        "max_abs_probability_difference": float(np.max(np.abs(diff))),
        "mean_bias_home": float(diff[:, 0].mean()),
        "mean_bias_draw": float(diff[:, 1].mean()),
        "mean_bias_away": float(diff[:, 2].mean()),
        "beats_market_log_loss": bool(model_metrics["log_loss"] < market_metrics["log_loss"]),
    }
    return row


def probability_range_performance(
    y_true,
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Observed frequency vs model and market in predicted-probability bins."""
    y = np.asarray(y_true, dtype=int)
    model = np.asarray(model_probabilities, dtype=float)
    market = np.asarray(market_probabilities, dtype=float)
    labels = list(LABEL_ORDER)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for j, label in enumerate(labels):
        bin_idx = np.clip(
            np.digitize(model[:, j], edges[1:-1], right=True), 0, n_bins - 1
        )
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            rows.append({
                "model": model_name,
                "outcome": OUTCOME_LABELS.get(label, str(label)),
                "code": label,
                "bin": b,
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": int(mask.sum()),
                "model_mean_probability": float(model[mask, j].mean()),
                "market_mean_probability": float(market[mask, j].mean()),
                "observed_rate": float((y[mask] == label).mean()),
                "model_gap": float(model[mask, j].mean() - (y[mask] == label).mean()),
                "market_gap": float(market[mask, j].mean() - (y[mask] == label).mean()),
            })
    return pd.DataFrame(rows)


def market_differences_frame(
    meta: pd.DataFrame,
    model_probabilities: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Per-fixture market-versus-model probabilities and differences."""
    market = market_from_meta(meta)
    model = np.asarray(model_probabilities, dtype=float)
    if len(model) != len(meta):
        raise ValueError("Model probabilities and metadata have different lengths.")
    out = meta[[
        "match_id", "date", "season", "home_team", "away_team", "match_outcome",
    ]].copy()
    out["model"] = model_name
    for j, column in enumerate(MARKET_COLUMNS):
        out[column] = market[column].to_numpy(float)
        out[f"model_{column.split('_')[1]}"] = model[:, j]
        out[f"diff_{column.split('_')[1]}"] = model[:, j] - market[column].to_numpy(float)
    out["market_pick"] = np.asarray(LABEL_ORDER)[
        market[list(MARKET_COLUMNS)].to_numpy(float).argmax(axis=1)
    ]
    out["model_pick"] = np.asarray(LABEL_ORDER)[model.argmax(axis=1)]
    out["disagrees"] = out["market_pick"] != out["model_pick"]
    out["actual_label"] = out["match_outcome"].map(
        {label: OUTCOME_LABELS[label] for label in LABEL_ORDER}
    )
    return out


def disagreement_analysis(
    y_true,
    model_probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Compare model and market on the rows where their picks disagree."""
    y = np.asarray(y_true, dtype=int)
    model = np.asarray(model_probabilities, dtype=float)
    market = np.asarray(market_probabilities, dtype=float)
    model_pick = np.asarray(LABEL_ORDER)[model.argmax(axis=1)]
    market_pick = np.asarray(LABEL_ORDER)[market.argmax(axis=1)]
    rows = []
    for label, mask in (
        ("all", np.ones(len(y), dtype=bool)),
        ("disagreement", model_pick != market_pick),
        ("agreement", model_pick == market_pick),
    ):
        if not mask.any():
            rows.append({
                "model": model_name,
                "subset": label,
                "n": 0,
                "model_log_loss": np.nan,
                "market_log_loss": np.nan,
                "model_accuracy": np.nan,
                "market_accuracy": np.nan,
                "model_beats_market": False,
            })
            continue
        model_metrics = evaluate_probabilities(y[mask], model[mask], LABEL_ORDER, model_name)
        market_metrics = evaluate_probabilities(y[mask], market[mask], LABEL_ORDER, "Market odds")
        rows.append({
            "model": model_name,
            "subset": label,
            "n": int(mask.sum()),
            "model_log_loss": model_metrics["log_loss"],
            "market_log_loss": market_metrics["log_loss"],
            "model_accuracy": model_metrics["accuracy"],
            "market_accuracy": market_metrics["accuracy"],
            "model_beats_market": bool(model_metrics["log_loss"] < market_metrics["log_loss"]),
        })
    return pd.DataFrame(rows)


def _log_features(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0))


def fit_market_incremental_stack(
    validation_probabilities: dict[str, np.ndarray],
    test_probabilities: dict[str, np.ndarray],
    y_validation,
    y_test,
    market_validation: np.ndarray,
    market_test: np.ndarray,
    *,
    model_name: str = "Market + model stack",
) -> dict:
    """Fit a validation-only logistic stack and score it once on test.

    The stack asks a narrow question: conditional on the market's probabilities,
    do the supplied model probabilities add information? It is a diagnostic, not
    a deployed ensemble, and validation is its only fitting window.
    """
    y_val = np.asarray(y_validation, dtype=int)
    y_test_array = np.asarray(y_test, dtype=int)
    names = list(validation_probabilities)
    if not names:
        raise ValueError("No component probabilities supplied to the stack.")
    X_val = np.column_stack(
        [_log_features(market_validation)]
        + [_log_features(validation_probabilities[name]) for name in names]
    )
    X_test = np.column_stack(
        [_log_features(market_test)]
        + [_log_features(test_probabilities[name]) for name in names]
    )
    if len(X_val) != len(y_val) or len(X_test) != len(y_test_array):
        raise ValueError("Stack features and labels have different lengths.")
    model = LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_val, y_val)
    raw = model.predict_proba(X_test)
    probabilities = align_proba_to_labels(raw, model.classes_, LABEL_ORDER)
    totals = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities, totals, out=np.zeros_like(probabilities), where=totals > 0
    )
    metrics = evaluate_probabilities(
        y_test_array, probabilities, LABEL_ORDER, model_name
    )
    market_metrics = evaluate_probabilities(
        y_test_array, market_test, LABEL_ORDER, "Market odds"
    )
    row = metrics_to_row(metrics)
    row.update({
        "market_log_loss": market_metrics["log_loss"],
        "market_brier_score": market_metrics["brier_score"],
        "delta_log_loss_vs_market": metrics["log_loss"] - market_metrics["log_loss"],
        "delta_brier_vs_market": metrics["brier_score"] - market_metrics["brier_score"],
        "components": "+".join(names),
        "fit_period": "validation",
        "test_used_for_fit": False,
    })
    return {
        "metrics": row,
        "probabilities": probabilities,
        "model": model,
    }
