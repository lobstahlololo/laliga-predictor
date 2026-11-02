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

# match_outcome codes -> human-readable labels.
OUTCOME_LABELS = {2: "Home Win", 1: "Draw", 0: "Away Win"}
LABEL_ORDER = [2, 1, 0]  # display order: Home Win, Draw, Away Win

RANDOM_STATE = 42
N_SEARCH_ITER = 40        # RandomizedSearchCV candidates per model
CV_FOLDS = 5              # stratified folds inside the training window
SEARCH_JOBS = -1          # parallelize the search itself

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

def build_search_spaces() -> dict:
    """Return {name: (estimator, param_distributions)} for tuning.

    Estimators use n_jobs=1 so the outer RandomizedSearchCV (SEARCH_JOBS=-1)
    can parallelize across candidates without oversubscribing cores.
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
        class_weight="balanced_subsample",
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


def balanced_sample_weights(y: pd.Series) -> np.ndarray:
    """Inverse-frequency sample weights: n / (n_classes * class_count)."""
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    weight_map = {
        c: len(y_arr) / (len(classes) * cnt) for c, cnt in zip(classes, counts)
    }
    weights = np.array([weight_map[v] for v in y_arr], dtype=float)
    logger.info(
        "Balanced sample weights per class: %s",
        {int(c): round(weight_map[c], 3) for c in classes},
    )
    return weights


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
    """Entry point: load data, tune both models, report, persist the best."""
    X_train, y_train, X_test, y_test = load_processed()

    # Balanced sample weights give the underrepresented Draw class more
    # mathematical significance inside XGBoost's objective (Random Forest gets
    # its balancing from class_weight="balanced").
    sample_weights = {"sample_weight": balanced_sample_weights(y_train)}

    results = []
    for name, (estimator, space) in build_search_spaces().items():
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
