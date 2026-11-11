"""Train, select and honestly evaluate La Liga outcome models.

Methodology (the core of the project):

1. **Seasonal split** — earlier seasons train (23/24 + 24/25), the penultimate
   season validates (25/26), and the newest season is an untouched test set
   (26/27).
2. **Time-ordered tuning** — RandomizedSearchCV uses ``TimeSeriesSplit``
   (never shuffled folds) and optimises multiclass log-loss.
3. **Validation-only selection** — the model with the lowest validation
   log-loss wins (Brier score breaks ties). Accuracy alone is never used.
4. **One final test score** — both models are refitted on train+validation
   and scored once on the held-out test season. The winner was already fixed
   before this score existed.
5. **Walk-forward backtest** — rolling-origin folds over earlier seasons
   provide a second, season-by-season view of stability.

Artifacts:
    models/saved_models/best_model.pkl
    models/saved_models/model_random_forest.pkl / model_xgboost.pkl
    models/saved_models/model_metadata.json
    data/processed/preprocessor.joblib
    models/evaluation/*.csv, *.png, evaluation_summary.json

Usage:
    python -m src.models.train_model
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.backtesting import (  # noqa: E402
    ModelSpec,
    balanced_sample_weights,
    run_walk_forward,
    summarise_walk_forward,
)
from src.evaluation.metrics import (  # noqa: E402
    compare_models,
    evaluate_probabilities,
    metrics_to_row,
)
from src.evaluation.plots import (  # noqa: E402
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_metrics,
    plot_season_performance,
    plot_walk_forward,
)
from src.evaluation.splits import chronological_split  # noqa: E402
from src.features.build_features import (  # noqa: E402
    build_model_frame,
    load_raw,
    save_feature_reference,
)
from src.features.preprocessing import FeaturePreprocessor  # noqa: E402
from src.utils.config import (  # noqa: E402
    # Retained as the live deployment targets; training itself only writes
    # versioned candidates under SAVED_MODEL_DIR.
    BEST_MODEL_PATH,
    CV_FOLDS,
    EVAL_DIR,
    EVALUATION_REPORT_PATH,
    FEATURE_COLUMNS,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    N_SEARCH_ITER,
    PRIMARY_METRIC,
    RANDOM_STATE,
    SAVED_MODEL_DIR,
    SEASON_METRICS_PATH,
    TEST_SEASON,
    TIEBREAK_METRIC,
    TRAIN_SEASONS,
    VALIDATION_SEASON,
    WALK_FORWARD_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_estimator(name: str, params: dict | None = None, *, class_weight=None):
    """Create a model with fixed reproducibility and balancing settings."""
    params = dict(params or {})
    if name == "Random Forest":
        return RandomForestClassifier(
            **params,
            class_weight=class_weight or "balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "XGBoost":
        return XGBClassifier(
            **params,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model name: {name}")


def build_search_spaces() -> dict:
    """Hyperparameter distributions for both candidate models."""
    xgb_space = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 3],
        "reg_lambda": [1.0, 2.0],
        "max_delta_step": [1, 3, 5],
    }
    rf_space = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, None],
    }
    return {
        "XGBoost": (build_estimator("XGBoost"), xgb_space),
        "Random Forest": (build_estimator("Random Forest"), rf_space),
    }


def spec_from_params(name: str, params: dict) -> ModelSpec:
    """Wrap a tuned parameter set in a fresh-model factory."""
    return ModelSpec(
        name=name,
        factory=lambda: build_estimator(name, params),
        uses_sample_weight=(name == "XGBoost"),
    )


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_model(
    name: str,
    estimator,
    space: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_search_iter: int = N_SEARCH_ITER,
    cv_folds: int = CV_FOLDS,
) -> tuple[dict, dict]:
    """Time-ordered randomized search optimising cross-validated log-loss.

    Rows are already chronological. ``TimeSeriesSplit`` only ever validates on
    the future of each training fold, so the search cannot peek forwards.
    """
    fit_kwargs = (
        {"sample_weight": balanced_sample_weights(y_train)}
        if name == "XGBoost" else {}
    )
    cv = TimeSeriesSplit(n_splits=cv_folds)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=n_search_iter,
        scoring={"log_loss": "neg_log_loss", "accuracy": "accuracy"},
        refit="log_loss",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )
    logger.info(
        "Tuning %s: %d candidates, %d time-ordered folds, metric=neg_log_loss",
        name, n_search_iter, cv_folds,
    )
    search.fit(X_train, y_train, **fit_kwargs)
    best_idx = int(search.best_index_)
    info = {
        "params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "cv_log_loss": float(-search.best_score_),
        "cv_accuracy": float(search.cv_results_["mean_test_accuracy"][best_idx]),
        "cv_folds": cv_folds,
        "n_candidates": n_search_iter,
    }
    logger.info(
        "%s best params: %s | CV log-loss=%.4f acc=%.4f",
        name, info["params"], info["cv_log_loss"], info["cv_accuracy"],
    )
    return info["params"], info


# ---------------------------------------------------------------------------
# Training / selection
# ---------------------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _scalar_metrics(metrics: dict) -> dict:
    return {
        k: _jsonable(v)
        for k, v in metrics.items()
        if not isinstance(v, (pd.DataFrame, list, dict, np.ndarray))
        and not k.startswith("_")
    }


def train_and_select(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    n_search_iter: int = N_SEARCH_ITER,
    cv_folds: int = CV_FOLDS,
    save: bool = True,
) -> dict:
    """Run tuning, validation selection, final test scoring and backtests."""
    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx = splits["train"].index
    val_idx = splits["validation"].index
    test_idx = splits["test"].index

    X_train_raw, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val_raw, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test_raw, y_test = X.loc[test_idx], y.loc[test_idx]

    # Preprocessing is fitted on the training window ONLY.
    train_pre = FeaturePreprocessor(FEATURE_COLUMNS)
    X_train = train_pre.fit_transform(X_train_raw)
    X_val = train_pre.transform(X_val_raw)

    tuned: dict[str, dict] = {}
    validation_results: list[dict] = []

    for name, (estimator, space) in build_search_spaces().items():
        params, info = tune_model(
            name, estimator, space, X_train, y_train,
            n_search_iter=n_search_iter, cv_folds=cv_folds,
        )
        tuned[name] = {"params": params, "tuning": info}

        # Validation score for the tuned candidate.
        model = build_estimator(name, params)
        if name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=balanced_sample_weights(y_train))
        else:
            model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        metrics = evaluate_probabilities(y_val, proba, model.classes_, model_name=name)
        metrics["_y_true"] = y_val.to_numpy()
        metrics["_y_proba"] = proba
        metrics["_classes"] = [int(c) for c in model.classes_]
        validation_results.append(metrics)
        tuned[name]["validation_metrics"] = _scalar_metrics(metrics)
        logger.info(
            "%s validation: log-loss=%.4f acc=%.4f brier=%.4f draw-recall=%.3f",
            name, metrics["log_loss"], metrics["accuracy"],
            metrics["brier_score"], metrics["draw_recall"],
        )

    validation_table = compare_models(validation_results)
    if save:
        # ``save=False`` must not write anything: the test suite trains on a
        # synthetic frame and would otherwise overwrite the committed
        # evaluation artifacts in ``models/evaluation``.
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        validation_table.to_csv(
            EVAL_DIR / "validation_model_comparison.csv", index=False
        )

    # Selection happens HERE, on validation data only.
    ordered = sorted(
        validation_results,
        key=lambda m: (m[PRIMARY_METRIC], m[TIEBREAK_METRIC]),
    )
    best_validation = ordered[0]
    best_name = best_validation["model"]
    logger.info(
        "Selected %s on validation %s=%.4f (Brier=%.4f). "
        "Test season has not been used for selection.",
        best_name, PRIMARY_METRIC, best_validation[PRIMARY_METRIC],
        best_validation[TIEBREAK_METRIC],
    )

    # Final fit on train + validation, using preprocessing fitted on that same
    # historical window. The test season remains untouched until scoring.
    hist_idx = train_idx.append(val_idx)
    final_pre = FeaturePreprocessor(FEATURE_COLUMNS)
    X_hist = final_pre.fit_transform(X.loc[hist_idx])
    y_hist = y.loc[hist_idx]
    X_test = final_pre.transform(X_test_raw)

    test_results: list[dict] = []
    final_models: dict[str, object] = {}
    for name, info in tuned.items():
        model = build_estimator(name, info["params"])
        if name == "XGBoost":
            model.fit(X_hist, y_hist, sample_weight=balanced_sample_weights(y_hist))
        else:
            model.fit(X_hist, y_hist)
        final_models[name] = model
        proba = model.predict_proba(X_test)
        metrics = evaluate_probabilities(y_test, proba, model.classes_, model_name=name)
        metrics["_y_true"] = y_test.to_numpy()
        metrics["_y_proba"] = proba
        metrics["_classes"] = [int(c) for c in model.classes_]
        test_results.append(metrics)
        info["test_metrics"] = _scalar_metrics(metrics)
        logger.info(
            "%s TEST (scored once): log-loss=%.4f acc=%.4f brier=%.4f draw-recall=%.3f",
            name, metrics["log_loss"], metrics["accuracy"],
            metrics["brier_score"], metrics["draw_recall"],
        )

    test_table = compare_models(test_results)
    test_table.insert(0, "selected_on_validation", test_table["model"] == best_name)

    # Walk-forward backtest: excludes the final test season by default.
    specs = [spec_from_params(name, info["params"]) for name, info in tuned.items()]
    wf_results, wf_detail = run_walk_forward(
        X, y, meta, specs,
        min_train_seasons=1,
        include_test_season=False,
        test_season=TEST_SEASON,
    )
    wf_summary = summarise_walk_forward(wf_results)

    # Season-by-season metrics: walk-forward folds plus the final test season.
    season_cols = [
        "model", "accuracy", "log_loss", "brier_score", "draw_recall",
        "macro_f1", "ece", "n_samples",
    ]
    season_rows = []
    if not wf_results.empty:
        for _, row in wf_results.iterrows():
            record = {c: row.get(c) for c in season_cols if c in row.index}
            record["season"] = row["test_season"]
            record["evaluation"] = "walk_forward"
            season_rows.append(record)
    for metrics in test_results:
        row = metrics_to_row(metrics)
        record = {c: row.get(c) for c in season_cols if c in row}
        record["season"] = TEST_SEASON
        record["evaluation"] = "final_test"
        season_rows.append(record)
    season_table = pd.DataFrame(season_rows)

    trained_at = datetime.now(timezone.utc)
    version_slug = best_name.lower().replace(" ", "_")
    model_version = f"{version_slug}-{trained_at:%Y%m%dT%H%M%SZ}"

    selected_model = final_models[best_name]
    metadata = {
        "model_name": best_name,
        "model_version": model_version,
        "trained_at": trained_at.isoformat(),
        "random_state": RANDOM_STATE,
        "feature_names": FEATURE_COLUMNS,
        "classes": [int(c) for c in selected_model.classes_],
        "parameters": tuned[best_name]["params"],
        "tuning": tuned[best_name]["tuning"],
        "selection": {
            "metric": PRIMARY_METRIC,
            "tiebreak": TIEBREAK_METRIC,
            "validation_log_loss": best_validation["log_loss"],
            "validation_brier": best_validation["brier_score"],
            "validation_accuracy": best_validation["accuracy"],
            "test_used_for_selection": False,
        },
        "split": {
            "train_seasons": TRAIN_SEASONS,
            "validation_season": VALIDATION_SEASON,
            "test_season": TEST_SEASON,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "train_date_min": str(pd.to_datetime(meta.loc[train_idx, "date"]).min().date()),
            "train_date_max": str(pd.to_datetime(meta.loc[train_idx, "date"]).max().date()),
            "validation_date_max": str(pd.to_datetime(meta.loc[val_idx, "date"]).max().date()),
            "test_date_min": str(pd.to_datetime(meta.loc[test_idx, "date"]).min().date()),
            "test_date_max": str(pd.to_datetime(meta.loc[test_idx, "date"]).max().date()),
        },
        "validation_metrics": tuned[best_name]["validation_metrics"],
        "test_metrics": tuned[best_name]["test_metrics"],
        "all_models": {
            name: {
                "parameters": info["params"],
                "validation_metrics": info["validation_metrics"],
                "test_metrics": info["test_metrics"],
            }
            for name, info in tuned.items()
        },
        "walk_forward": {
            "folds": int(wf_results["fold"].nunique()) if not wf_results.empty else 0,
            "mean_log_loss": {
                row["model"]: float(row["log_loss_mean"])
                for _, row in wf_summary.iterrows()
            } if not wf_summary.empty else {},
        },
    }

    if save:
        _save_artifacts(
            selected_model, best_name, final_models, final_pre, metadata,
            validation_results, test_results, test_table, season_table,
            wf_results, wf_summary, X_hist,
        )

    return {
        "metadata": metadata,
        "validation_results": validation_results,
        "test_results": test_results,
        "comparison": test_table,
        "season_metrics": season_table,
        "walk_forward": wf_results,
        "walk_forward_summary": wf_summary,
        "tuned": tuned,
        "models": final_models,
        "preprocessor": final_pre,
    }


def _save_artifacts(
    selected_model,
    best_name: str,
    final_models: dict,
    preprocessor: FeaturePreprocessor,
    metadata: dict,
    validation_results: list[dict],
    test_results: list[dict],
    test_table: pd.DataFrame,
    season_table: pd.DataFrame,
    wf_results: pd.DataFrame,
    wf_summary: pd.DataFrame,
    X_hist: pd.DataFrame,
) -> None:
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in final_models.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(model, SAVED_MODEL_DIR / f"model_{slug}.pkl")

    # Training writes a versioned candidate. The live inference artifacts are
    # only replaced by an explicit promotion (or the very first registered
    # model), so a newer experiment can never silently become production.
    version = metadata["model_version"]
    candidate_model_path = SAVED_MODEL_DIR / f"candidate_{version}.pkl"
    candidate_pre_path = SAVED_MODEL_DIR / f"candidate_preprocessor_{version}.joblib"
    joblib.dump(selected_model, candidate_model_path)
    preprocessor.save(candidate_pre_path)

    # Portable, repo-relative paths: metadata is committed and read on Linux CI.
    metadata["selected_model_path"] = candidate_model_path.relative_to(PROJECT_ROOT).as_posix()
    metadata["preprocessor_path"] = candidate_pre_path.relative_to(PROJECT_ROOT).as_posix()

    from src.models import registry

    entry, created = registry.register_model(
        metadata,
        model_path=candidate_model_path,
        preprocessor_path=candidate_pre_path,
    )
    production = registry.current_production()
    if production and production["version"] == version:
        registry.restore_artifacts(version)
        metadata["promotion_status"] = "production"
        logger.warning(
            "No previous production model existed; %s is now the initial "
            "production model.", version,
        )
    else:
        metadata["promotion_status"] = "candidate"
        metadata["production_version"] = production["version"] if production else None
        logger.info(
            "Registered %s as a candidate (new=%s). Production %s is unchanged; "
            "promote with `python -m src.models.registry promote %s --reason \"...\"`.",
            version, created, metadata["production_version"], version,
        )
    logger.info("Saved candidate artifacts and evaluation outputs for %s", version)

    test_table.to_csv(MODEL_COMPARISON_PATH, index=False)
    season_table.to_csv(SEASON_METRICS_PATH, index=False)
    if not wf_results.empty:
        wf_results.to_csv(WALK_FORWARD_PATH, index=False)
        wf_summary.to_csv(EVAL_DIR / "walk_forward_summary.csv", index=False)
    save_feature_reference(X_hist)

    # ---- Plots ---------------------------------------------------------
    for metrics in test_results:
        name = metrics["model"]
        plot_confusion_matrix(
            metrics["confusion_matrix"], f"{name} — Test", metrics["accuracy"]
        )
        plot_per_class_metrics(metrics["per_class"], f"{name} — Test")
        if all(k in metrics for k in ("_y_true", "_y_proba", "_classes")):
            plot_calibration_curves(
                metrics["_y_true"], metrics["_y_proba"], metrics["_classes"],
                f"{name} — Test",
            )
    plot_model_comparison(test_table)
    if not wf_results.empty:
        plot_walk_forward(wf_results)
    if not season_table.empty:
        plot_season_performance(season_table)

    # Feature importance for tree models (human-readable model insight).
    importance_rows = []
    for name, model in final_models.items():
        values = getattr(model, "feature_importances_", None)
        if values is not None:
            importance_rows.extend({
                "model": name,
                "feature": feature,
                "importance": float(value),
            } for feature, value in zip(FEATURE_COLUMNS, values))
    if importance_rows:
        pd.DataFrame(importance_rows).sort_values(
            ["model", "importance"], ascending=[True, False]
        ).to_csv(EVAL_DIR / "feature_importance.csv", index=False)

    summary = {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "trained_at": metadata["trained_at"],
        "selection": metadata["selection"],
        "split": metadata["split"],
        "validation_comparison": [
            {k: _jsonable(v) for k, v in metrics_to_row(m).items()}
            for m in validation_results
        ],
        "test_comparison": [
            {k: _jsonable(v) for k, v in metrics_to_row(m).items()}
            for m in test_results
        ],
        "walk_forward_summary": (
            wf_summary.to_dict(orient="records") if not wf_summary.empty else []
        ),
        "feature_count": len(FEATURE_COLUMNS),
        "known_limitations": [
            "Test season is in progress; final metrics will change as more "
            "matches are played.",
            "Draw probabilities remain the hardest class and are typically "
            "the least well calibrated.",
            "Historical odds are Bet365 snapshots and may contain closing-line "
            "bias unavailable at prediction time.",
        ],
    }
    EVALUATION_REPORT_PATH.write_text(
        json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8"
    )



def main() -> int:
    logger.info("Loading data and engineering features...")
    raw = load_raw()
    X, y, meta = build_model_frame(raw)
    result = train_and_select(X, y, meta)
    metadata = result["metadata"]

    print("\n" + "=" * 72)
    print("FINAL SUMMARY — validation-only selection, untouched test season")
    print("=" * 72)
    print(
        f"Selected model : {metadata['model_name']} ({metadata['model_version']})"
    )
    print(f"Promotion      : {metadata.get('promotion_status', 'unknown')}")
    print(
        f"Selection      : validation log-loss="
        f"{metadata['selection']['validation_log_loss']:.4f}, "
        f"Brier={metadata['selection']['validation_brier']:.4f}"
    )
    print("\nValidation comparison:")
    print(compare_models(result["validation_results"]).to_string(index=False))
    print("\nTest comparison (scored once, never used for selection):")
    print(result["comparison"].to_string(index=False))
    if not result["walk_forward"].empty:
        print("\nWalk-forward folds (earlier seasons only):")
        print(
            result["walk_forward"][
                ["fold", "test_season", "model", "log_loss", "accuracy", "brier_score"]
            ].to_string(index=False)
        )
    print(f"\nArtifacts: {SAVED_MODEL_DIR}, {EVAL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
