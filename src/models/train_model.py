"""Train and evaluate classifiers for La Liga match outcomes.

Ingests the standardized train/test CSVs from ``data/processed/`` (produced by
``src/features/build_features.py``), tunes an XGBoost Classifier and a Random
Forest Classifier with RandomizedSearchCV, and reports accuracy, log-loss,
per-class precision/recall/F1 and a confusion matrix (console text plus a
matplotlib PNG saved to ``models/evaluation/``).

Scoring/optimization design:

* The search refits **strictly on ``neg_log_loss``** — accuracy is logged for
  context only. Log-loss punishes overconfident models and models that
  collapse onto majority classes (e.g. never predicting Draw).
* Class balancing: Random Forest uses ``class_weight="balanced"``; XGBoost is
  fitted with balanced ``sample_weight`` computed from the training label
  counts, so the underrepresented Draw class carries proportionally more
  mathematical weight in the multi:softprob objective.
* The final winner serialized with joblib is the model with the **lowest
  log-loss on the held-out test season**.
* Self-correcting feedback: before the training matrix is initialized the
  ledger ``data/processed/prediction_history.csv`` is read and the rolling
  Draw-class precision over the last 30 settled entries is measured. When
  that precision falls below 25% (the model is overestimating draw edges
  and losing money on value bets) the Draw class (label 1) coefficients are
  raised by 15% for both models — XGBoost's per-row ``sample_weight`` and the
  Random Forest class-cost penalty — so the next optimization loop is more
  conservative. With fewer than 5 settled entries (brand-new ledger) or a
  precision at/above the floor, everything runs at the baseline
  ``balanced_subsample`` state.

The split is time-based: the historical seasons (23/24 + 24/25 + 25/26) form
the training set and the current 26/27 season is the held-out test set. See
``src/features/build_features.py`` for where the split is applied.

Usage:
    python src/models/train_model.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVAL_DIR = PROJECT_ROOT / "models" / "evaluation"
SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "saved_models"
BEST_MODEL_PATH = SAVED_MODEL_DIR / "best_model.pkl"

X_TRAIN_PATH = PROCESSED_DIR / "X_train_scaled.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test_scaled.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

# Prediction ledger written by predict_upcoming.py / backfill_results.py.
LEDGER_PATH = PROCESSED_DIR / "prediction_history.csv"

# match_outcome codes -> human-readable labels.
OUTCOME_LABELS = {2: "Home Win", 1: "Draw", 0: "Away Win"}
LABEL_ORDER = [2, 1, 0]  # display order: Home Win, Draw, Away Win

RANDOM_STATE = 42
N_SEARCH_ITER = 40        # RandomizedSearchCV candidates per model
CV_FOLDS = 5              # stratified folds inside the training window
SEARCH_JOBS = -1          # parallelize the search itself

# --- Self-correcting ledger feedback ---------------------------------------
DRAW_CLASS = 1              # match_outcome code for a Draw (class 1)
DRAW_WINDOW = 30            # rolling window of settled ledger entries
MIN_SETTLED_ENTRIES = 5     # below this -> baseline adjustment state
DRAW_PRECISION_FLOOR = 0.25 # below this -> penalize the Draw class
DRAW_PENALTY_FACTOR = 1.15  # +15% on Draw coefficients when triggered

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the scaled feature matrices and targets from data/processed/."""
    missing = [
        p for p in (X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing processed files: "
            + ", ".join(str(p) for p in missing)
            + ". Run `python src/features/build_features.py` first."
        )

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["match_outcome"].astype(int)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["match_outcome"].astype(int)

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature matrices and target vectors have mismatched lengths.")

    logger.info(
        "Loaded data: train=%s (y: %s), test=%s (y: %s)",
        X_train.shape,
        y_train.value_counts().sort_index().to_dict(),
        X_test.shape,
        y_test.value_counts().sort_index().to_dict(),
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Model search spaces
# ---------------------------------------------------------------------------

def read_draw_performance() -> dict:
    """Read the ledger and measure rolling Draw precision over its tail.

    Must run **before** the training matrix is initialized. Filters
    ``data/processed/prediction_history.csv`` to settled rows (non-blank
    ``actual_outcome``), keeps the last ``DRAW_WINDOW`` entries and computes
    Draw-class precision: of the rows whose model argmax was the Draw, the
    share that actually finished as a Draw.

    Any unusable state — missing/unreadable ledger, missing columns, fewer
    than ``MIN_SETTLED_ENTRIES`` settled rows (e.g. a brand-new file), no
    Draw predictions or unparseable cells — degrades gracefully to the
    baseline adjustment state (``adjust=False``) instead of raising.
    """
    stats: dict = {
        "settled_entries": 0,
        "window_entries": 0,
        "draw_predictions": 0,
        "draw_precision": None,
        "adjust": False,
        "reason": "",
    }

    def baseline(reason: str) -> dict:
        stats["reason"] = reason
        logger.info("Draw feedback: %s", reason)
        return stats

    if not LEDGER_PATH.exists():
        return baseline(f"ledger {LEDGER_PATH} not found - baseline state")

    try:
        ledger = pd.read_csv(LEDGER_PATH, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001 - feedback must never crash training
        return baseline(f"ledger unreadable ({exc}) - baseline state")

    prob_cols = ["model_prob_home", "model_prob_draw", "model_prob_away"]
    if "actual_outcome" not in ledger.columns or not all(
        c in ledger.columns for c in prob_cols
    ):
        return baseline("ledger missing required columns - baseline state")

    settled = ledger[ledger["actual_outcome"].astype(str).str.strip() != ""].copy()
    stats["settled_entries"] = len(settled)
    if len(settled) < MIN_SETTLED_ENTRIES:
        return baseline(
            f"only {len(settled)} settled entries "
            f"(< {MIN_SETTLED_ENTRIES}) - baseline state"
        )

    window = settled.tail(DRAW_WINDOW)
    probs = window[prob_cols].apply(pd.to_numeric, errors="coerce")
    actual = pd.to_numeric(window["actual_outcome"], errors="coerce")
    usable = probs.notna().all(axis=1) & actual.notna()
    probs, actual = probs[usable], actual[usable]
    stats["window_entries"] = len(probs)
    if not usable.any():
        return baseline("no parseable settled rows in window - baseline state")

    # Model's top pick per row over [home, draw, away] -> index 1 == Draw.
    predicted = probs.to_numpy().argmax(axis=1)
    draw_mask = predicted == 1
    n_draw_pred = int(draw_mask.sum())
    stats["draw_predictions"] = n_draw_pred
    if n_draw_pred == 0:
        return baseline("no Draw predictions in window - baseline state")

    precision = float((actual.to_numpy()[draw_mask] == DRAW_CLASS).mean())
    stats["draw_precision"] = precision

    if precision < DRAW_PRECISION_FLOOR:
        stats["adjust"] = True
        stats["reason"] = (
            f"Draw precision {precision:.1%} < {DRAW_PRECISION_FLOOR:.0%} floor "
            f"over last {len(probs)} settled entries "
            f"({n_draw_pred} Draw picks) - applying +"
            f"{(DRAW_PENALTY_FACTOR - 1) * 100:.0f}% Draw penalty"
        )
    else:
        stats["reason"] = (
            f"Draw precision {precision:.1%} >= {DRAW_PRECISION_FLOOR:.0%} floor "
            f"over last {len(probs)} settled entries "
            f"({n_draw_pred} Draw picks) - baseline weights retained"
        )
    logger.info("Draw feedback: %s", stats["reason"])
    return stats


def build_search_spaces(rf_class_weight="balanced_subsample") -> dict:
    """Return {name: (estimator, param_distributions)} for tuning.

    Estimators use n_jobs=1 so the outer RandomizedSearchCV (SEARCH_JOBS=-1)
    can parallelize across candidates without oversubscribing cores.

    ``rf_class_weight`` is the Random Forest cost penalty: the baseline
    ``"balanced_subsample"`` string, or an explicit per-class dict when the
    ledger feedback raised the Draw-class penalty.
    """
    # XGBoost: multi:softprob + mlogloss; draw significance comes from the
    # balanced sample weights supplied at fit time (see balanced_sample_weights).
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_space = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.03, 0.05, 0.08, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.5, 1.0, 2.0],
        # Caps per-class update weight per step, stabilizing training against
        # the skewed multi-class (home-heavy) distribution.
        "max_delta_step": list(range(1, 6)),  # integers 1..5
    }

    # Random Forest: per-bag class balancing so Draw is never free to ignore.
    rf = RandomForestClassifier(
        class_weight=rf_class_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf_space = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [None, 5, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, None],
    }

    return {
        "XGBoost": (xgb, xgb_space),
        "Random Forest": (rf, rf_space),
    }


def balanced_sample_weights(y: pd.Series, draw_multiplier: float = 1.0) -> np.ndarray:
    """Inverse-frequency sample weights: n / (n_classes * class_count).

    ``draw_multiplier`` scales the Draw class (1) coefficient: 1.0 keeps the
    baseline, 1.15 applies the ledger-driven +15% penalty.
    """
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    weight_map = {
        c: len(y_arr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)
    }
    if draw_multiplier != 1.0 and DRAW_CLASS in weight_map:
        weight_map[DRAW_CLASS] *= draw_multiplier
    weights = np.array([weight_map[v] for v in y_arr], dtype=float)
    logger.info(
        "Balanced sample weights per class%s: %s",
        " (Draw-adjusted)" if draw_multiplier != 1.0 else "",
        {int(c): round(weight_map[c], 3) for c in classes},
    )
    return weights


def rf_class_cost(y: pd.Series, draw_multiplier: float = 1.0):
    """Random Forest class-cost penalty for the current feedback state.

    Baseline is the ``"balanced_subsample"`` string. When the ledger
    feedback triggers, returns an explicit balanced per-class dict with the
    Draw class (1) scaled by ``draw_multiplier`` (1.15 = +15% cost penalty).
    """
    if draw_multiplier == 1.0:
        return "balanced_subsample"
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    cost = {
        int(c): len(y_arr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)
    }
    cost[DRAW_CLASS] = cost.get(DRAW_CLASS, 1.0) * draw_multiplier
    logger.info("Random Forest Draw-adjusted class cost: %s", cost)
    return cost


def log_training_coefficients(
    draw_stats: dict, sample_weight: np.ndarray, y: pd.Series, rf_cw
) -> None:
    """Log the final adjusted training weight coefficients to the console.

    Called at the beginning of the fit sequence via formal Python logging so
    every pipeline run records exactly which coefficients shaped the fit.
    """
    y_arr = np.asarray(y)
    per_class = {
        int(c): round(float(sample_weight[y_arr == c].mean()), 4)
        for c in np.unique(y_arr)
    }
    precision = draw_stats["draw_precision"]
    logger.info("=" * 64)
    logger.info("FIT SEQUENCE: final adjusted training weight coefficients")
    logger.info("  Ledger feedback  : %s", draw_stats["reason"])
    logger.info(
        "  Settled entries  : %d (window %d, Draw picks %d)",
        draw_stats["settled_entries"],
        draw_stats["window_entries"],
        draw_stats["draw_predictions"],
    )
    logger.info(
        "  Draw precision   : %s",
        "n/a" if precision is None else f"{precision:.1%}",
    )
    logger.info("  XGBoost sample_weight per class : %s", per_class)
    logger.info("  RandomForest class_weight       : %s", rf_cw)
    logger.info("=" * 64)


def tune_model(
    name: str,
    estimator,
    space: dict,
    X_train,
    y_train,
    fit_kwargs: dict | None = None,
):
    """Run RandomizedSearchCV optimizing cross-validated ``neg_log_loss``.

    ``fit_kwargs`` allows passing balanced ``sample_weight`` (XGBoost);
    RandomForest balancing is handled via its ``class_weight`` parameter.
    """
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=N_SEARCH_ITER,
        scoring={
            "log_loss": "neg_log_loss",   # primary: the optimization target
            "accuracy": "accuracy",       # diagnostics only, never refit on
        },
        refit="log_loss",  # strictly: pick/refit the best by lowest CV log-loss
        cv=StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE,
        n_jobs=SEARCH_JOBS,
        verbose=0,
        return_train_score=False,
    )
    logger.info(
        "Tuning %s (%d candidates, %d-fold stratified CV, refit on neg_log_loss)...",
        name, N_SEARCH_ITER, CV_FOLDS,
    )
    search.fit(X_train, y_train, **(fit_kwargs or {}))
    logger.info(
        "%s best params: %s", name,
        {k: v for k, v in sorted(search.best_params_.items())},
    )
    logger.info(
        "%s best CV log-loss=%.4f, CV accuracy=%.4f",
        name,
        -search.best_score_,  # refit metric is neg_log_loss
        search.cv_results_["mean_test_accuracy"][
            int(search.best_index_)
        ],
    )
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: pd.DataFrame, model_name: str, accuracy: float
) -> Path:
    """Render a confusion matrix PNG with raw counts *and* row-wise
    percentage distributions overlaid on every cell."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(cm.columns)),
        yticks=range(len(cm.index)),
        xticklabels=cm.columns,
        yticklabels=cm.index,
        ylabel="Actual",
        xlabel="Predicted",
        title=f"{model_name} — Confusion Matrix (acc={accuracy:.3f})",
    )

    # Annotate each cell with the raw count and the row-wise percentage
    # distribution (share of the actual class predicted as this column).
    counts = cm.values.astype(int)
    row_totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_pct = np.where(
            row_totals > 0, counts / row_totals * 100.0, 0.0
        )
    thresh = counts.max() / 2 if counts.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{counts[i, j]}\n({row_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if counts[i, j] > thresh else "black",
            )

    fig.tight_layout()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"confusion_matrix_{'_'.join(model_name.lower().split())}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate(name: str, model, y_true: pd.Series, X_test: pd.DataFrame) -> dict:
    """Score one tuned model: accuracy, log-loss, report, confusion matrix."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_true, y_pred)
    # Log-loss measures probabilistic confidence; classes absent from the
    # test fold would break alignment, so map labels to column indexes.
    class_index = {c: i for i, c in enumerate(model.classes_)}
    labels_present = sorted(set(y_true) | set(y_pred))
    proba_cols = [class_index[c] for c in labels_present if c in class_index]
    ll = log_loss(
        y_true,
        y_proba[:, proba_cols],
        labels=labels_present,
    )

    labels = [l for l in LABEL_ORDER if l in set(y_true) | set(y_pred)]
    target_names = [OUTCOME_LABELS[l] for l in labels]

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"Actual {n}" for n in target_names],
        columns=[f"Pred {n}" for n in target_names],
    )

    print("\n" + "=" * 64)
    print(f"MODEL: {name} (tuned)")
    print("=" * 64)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Log-Loss:  {ll:.4f}  (lower = better calibrated probabilities)")
    print("\nPrecision / Recall / F1 per class:")
    print(report)
    print("Confusion matrix:")
    print(cm.to_string())

    path = plot_confusion_matrix(cm, name, accuracy)
    print(f"Confusion matrix figure saved to: {path}")
    return {
        "model": name,
        "accuracy": accuracy,
        "log_loss": ll,
        "confusion_matrix": cm,
        "estimator": model,
    }


def main() -> int:
    """Entry point: read ledger feedback, tune models, persist the best."""
    # 1) Dynamic ledger read: MUST precede training-matrix initialization so
    #    settled financial performance can steer this run's coefficients.
    draw_stats = read_draw_performance()
    draw_multiplier = DRAW_PENALTY_FACTOR if draw_stats["adjust"] else 1.0

    X_train, y_train, X_test, y_test = load_processed()

    # Balanced sample weights give the underrepresented Draw class more
    # mathematical significance inside XGBoost's objective (Random Forest gets
    # its balancing from class_weight). Both carry the ledger-driven Draw
    # multiplier when the feedback mechanism triggers.
    sample_weight = balanced_sample_weights(y_train, draw_multiplier)
    rf_cw = rf_class_cost(y_train, draw_multiplier)
    sample_weights = {"sample_weight": sample_weight}

    # 3) Formal logging of the final adjusted coefficients at the start of
    #    the fit sequence.
    log_training_coefficients(draw_stats, sample_weight, y_train, rf_cw)

    results = []
    for name, (estimator, space) in build_search_spaces(rf_cw).items():
        kwargs = sample_weights if name == "XGBoost" else None
        best_est = tune_model(
            name, estimator, space, X_train, y_train, fit_kwargs=kwargs
        )
        results.append(evaluate(name, best_est, y_test, X_test))

    print("\n" + "=" * 64)
    print("SUMMARY (time-based split: train 23/24+24/25+25/26 -> test 26/27)")
    print("=" * 64)
    for res in results:
        print(
            f"  {res['model']:<15} accuracy = {res['accuracy']:.4f}   "
            f"log-loss = {res['log_loss']:.4f}"
        )

    # Persist the winner: strictly the lowest log-loss on the test season.
    best = min(results, key=lambda r: r["log_loss"])
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["estimator"], BEST_MODEL_PATH)
    print(
        f"\nBest model: {best['model']} -> saved via joblib to {BEST_MODEL_PATH}"
    )
    logger.info(
        "Serialized model classes: %s", list(best["estimator"].classes_)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
