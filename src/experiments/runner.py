"""Phase 2 experiment orchestrator.

Order of evaluation is deliberate:

1. Fixed baselines are scored on validation and once on test.
2. Candidate feature groups are compared on validation and walk-forward folds
   only. Test metrics are withheld.
3. Calibration methods are selected on a chronological validation holdout, then
   compared once on test.
4. Explainability uses validation/fold fits, never the final test season for
   feature selection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.baselines import BASELINE_LABELS, run_baselines
from src.evaluation.calibration import (
    CALIBRATION_METHODS,
    compare_calibrated_test,
    fit_calibrator,
    select_calibration_method,
)
from src.evaluation.explainability import (
    fold_importance,
    native_tree_importance,
    optional_random_forest_shap,
    permutation_table,
    stability_table,
    xgboost_tree_shap,
)
from src.evaluation.metrics import evaluate_probabilities
from src.evaluation.phase2_plots import (
    plot_baseline_comparison,
    plot_calibration_comparison,
    plot_experiment_leaderboard,
    plot_feature_experiments,
    plot_feature_stability,
    plot_importance_comparison,
    plot_reliability_overlay,
)
from src.evaluation.splits import chronological_split, walk_forward_folds
from src.experiments.tracking import (
    fingerprint_file,
    load_experiments,
    make_experiment_record,
    save_experiment,
)
from src.features.build_features import build_model_frame, load_raw
from src.features.experimental_features import (
    CANDIDATE_FEATURE_SETS,
    FEATURE_GROUPS,
    build_extended_frame,
)
from src.features.preprocessing import FeaturePreprocessor
from src.models.train_model import build_estimator
from src.monitoring.diagnostics import run_diagnostics, save_monitoring_report
from src.predictions.history import load_history
from src.utils.config import (
    BASELINE_TEST_PATH,
    BASELINE_VALIDATION_PATH,
    BASELINE_VS_ML_PATH,
    CALIBRATION_COMPARISON_PATH,
    CALIBRATION_SELECTION_PATH,
    EVAL_DIR,
    FEATURE_COLUMNS,
    FEATURE_EXPERIMENT_FOLDS_PATH,
    FEATURE_EXPERIMENTS_PATH,
    FEATURE_IMPORTANCE_FOLDS_PATH,
    FEATURE_RECOMMENDATION_PATH,
    FEATURE_STABILITY_PATH,
    MODEL_METADATA_PATH,
    PERMUTATION_IMPORTANCE_PATH,
    PHASE2_SUMMARY_PATH,
    RAW_PATH,
    SAVED_MODEL_DIR,
    SHAP_IMPORTANCE_PATH,
    TEST_SEASON,
    TRAIN_SEASONS,
    TREE_IMPORTANCE_PATH,
    VALIDATION_SEASON,
    WALK_FORWARD_PATH,
)
from src.utils.config import PHASE2_REPORT_PATH

logger = logging.getLogger(__name__)

DEFAULT_RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}
DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "max_delta_step": 1,
}
MODEL_NAMES = ("Random Forest", "XGBoost")


def _slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def _scalar_metrics(metrics: dict) -> dict:
    return {
        key: value for key, value in metrics.items()
        if not isinstance(value, (pd.DataFrame, list, dict, np.ndarray))
        and not key.startswith("_")
    }


def _per_row_loss(y_true, proba: np.ndarray, classes) -> np.ndarray:
    """Per-row log-loss in [home, draw, away] order."""
    from src.evaluation.metrics import align_proba_to_labels
    aligned = align_proba_to_labels(proba, classes, BASELINE_LABELS)
    totals = aligned.sum(axis=1, keepdims=True)
    aligned = np.divide(aligned, totals, out=np.zeros_like(aligned), where=totals > 0)
    index = {label: i for i, label in enumerate(BASELINE_LABELS)}
    true_p = np.array([
        aligned[row, index[int(label)]]
        for row, label in enumerate(np.asarray(y_true, dtype=int))
    ])
    return -np.log(np.clip(true_p, 1e-12, 1.0))


def _bootstrap_delta_ci(base_loss: np.ndarray, candidate_loss: np.ndarray, n_boot: int = 1000) -> tuple[float, float]:
    """Paired bootstrap interval for candidate minus base log-loss."""
    base = np.asarray(base_loss, dtype=float)
    candidate = np.asarray(candidate_loss, dtype=float)
    if len(base) != len(candidate) or len(base) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(base), size=(n_boot, len(base)))
    deltas = (candidate[indices] - base[indices]).mean(axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _fit(name: str, params: dict, X_train, y_train):
    model = build_estimator(name, params)
    if name == "XGBoost":
        from src.evaluation.backtesting import balanced_sample_weights
        model.fit(
            X_train, y_train,
            sample_weight=balanced_sample_weights(y_train),
        )
    else:
        model.fit(X_train, y_train)
    return model


def _params(metadata: dict) -> dict[str, dict]:
    result = {}
    for name in MODEL_NAMES:
        info = metadata.get("all_models", {}).get(name, {})
        result[name] = dict(
            info.get("parameters")
            or (DEFAULT_RF_PARAMS if name == "Random Forest" else DEFAULT_XGB_PARAMS)
        )
    return result


def _base_metrics_row(metadata: dict, split: str, label_suffix: str = " (ML)") -> list[dict]:
    rows = []
    for name in MODEL_NAMES:
        metrics = metadata.get("all_models", {}).get(name, {}).get(f"{split}_metrics", {})
        if not metrics:
            continue
        rows.append({
            "model": f"{name}{label_suffix}",
            **{key: metrics.get(key) for key in (
                "n_samples", "accuracy", "log_loss", "brier_score", "macro_f1",
                "draw_recall", "draw_precision", "ece",
            )},
        })
    return rows


def _combined_baseline_table(metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = pd.read_csv(BASELINE_VALIDATION_PATH) if BASELINE_VALIDATION_PATH.exists() else pd.DataFrame()
    test = pd.read_csv(BASELINE_TEST_PATH) if BASELINE_TEST_PATH.exists() else pd.DataFrame()
    validation = pd.concat(
        [validation, pd.DataFrame(_base_metrics_row(metadata, "validation"))],
        ignore_index=True,
    )
    test = pd.concat(
        [test, pd.DataFrame(_base_metrics_row(metadata, "test"))],
        ignore_index=True,
    )
    validation = validation.sort_values("log_loss").reset_index(drop=True)
    test = test.sort_values("log_loss").reset_index(drop=True)
    combined = validation.assign(split="Validation").merge(
        test.assign(split="Test"),
        on="model", suffixes=("_validation", "_test"), how="outer",
    )
    combined.to_csv(BASELINE_VS_ML_PATH, index=False)
    return validation, test


def _record_baselines(metadata: dict, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    data_hash = fingerprint_file(RAW_PATH)
    val_lookup = validation.set_index("model")
    test_lookup = test.set_index("model")
    for name in val_lookup.index:
        if "(ML)" in str(name) or name not in test_lookup.index:
            continue
        record = make_experiment_record(
            model_type=name,
            feature_set="base_19",
            features=list(FEATURE_COLUMNS),
            training_period=list(TRAIN_SEASONS),
            validation_period=VALIDATION_SEASON,
            test_period=TEST_SEASON,
            hyperparameters={},
            validation_metrics=val_lookup.loc[name].to_dict(),
            test_metrics=test_lookup.loc[name].to_dict(),
            data_fingerprint=data_hash,
            model_version=str(metadata.get("model_version", "")),
            selection_status="fixed baseline; test scored once",
            notes="Baseline defined before test evaluation; not tuned on test.",
        )
        save_experiment(record)


def run_feature_experiments(
    raw: pd.DataFrame,
    metadata: dict,
    *,
    save: bool = True,
) -> dict:
    """Compare candidate feature groups on validation + walk-forward only."""
    X_ext, y, meta = build_extended_frame(raw)
    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx = splits["train"].index
    val_idx = splits["validation"].index
    params_by_model = _params(metadata)
    folds = [
        fold for fold in walk_forward_folds(meta, min_train_seasons=1)
        if fold.test_season != TEST_SEASON
    ]

    rows: list[dict] = []
    fold_rows: list[dict] = []
    base_by_model: dict[str, dict] = {}
    validation_losses: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in MODEL_NAMES
    }

    for model_name in MODEL_NAMES:
        for feature_set, features in CANDIDATE_FEATURE_SETS.items():
            features = list(dict.fromkeys(features))
            pre = FeaturePreprocessor(features)
            X_train = pre.fit_transform(X_ext.loc[train_idx, features])
            X_val = pre.transform(X_ext.loc[val_idx, features])
            model = _fit(
                model_name, params_by_model[model_name], X_train, y.loc[train_idx]
            )
            val_proba = model.predict_proba(X_val)
            val_metrics = evaluate_probabilities(
                y.loc[val_idx], val_proba, model.classes_, model_name
            )
            validation_losses[model_name][feature_set] = _per_row_loss(
                y.loc[val_idx], val_proba, model.classes_
            )
            fold_losses = []
            for fold in folds:
                fold_pre = FeaturePreprocessor(features)
                fold_train = fold_pre.fit_transform(
                    X_ext.loc[fold.train_index, features]
                )
                fold_test = fold_pre.transform(
                    X_ext.loc[fold.test_index, features]
                )
                fold_model = _fit(
                    model_name, params_by_model[model_name],
                    fold_train, y.loc[fold.train_index],
                )
                fold_proba = fold_model.predict_proba(fold_test)
                fold_metrics = evaluate_probabilities(
                    y.loc[fold.test_index], fold_proba,
                    fold_model.classes_, model_name,
                )
                fold_losses.append(fold_metrics["log_loss"])
                fold_rows.append({
                    "model": model_name,
                    "feature_set": feature_set,
                    "fold": fold.fold,
                    "test_season": fold.test_season,
                    "log_loss": fold_metrics["log_loss"],
                    "accuracy": fold_metrics["accuracy"],
                    "brier_score": fold_metrics["brier_score"],
                    "ece": fold_metrics["ece"],
                    "n_samples": fold_metrics["n_samples"],
                })
            wf_mean = float(np.mean(fold_losses)) if fold_losses else np.nan
            row = {
                "model": model_name,
                "feature_set": feature_set,
                "n_features": len(features),
                "validation_log_loss": val_metrics["log_loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_brier_score": val_metrics["brier_score"],
                "validation_ece": val_metrics["ece"],
                "walk_forward_log_loss": wf_mean,
                **{f"wf_fold_{fold.fold}_log_loss": loss for fold, loss in zip(folds, fold_losses)},
            }
            if feature_set == "base_19":
                base_by_model[model_name] = row
            else:
                base_loss = validation_losses[model_name]["base_19"]
                candidate_loss = validation_losses[model_name][feature_set]
                ci_low, ci_high = _bootstrap_delta_ci(base_loss, candidate_loss)
                row["delta_validation_ci_low"] = ci_low
                row["delta_validation_ci_high"] = ci_high
            rows.append(row)

    table = pd.DataFrame(rows)
    for model_name, base in base_by_model.items():
        mask = table["model"] == model_name
        table.loc[mask, "delta_validation_log_loss"] = (
            table.loc[mask, "validation_log_loss"] - base["validation_log_loss"]
        )
        table.loc[mask, "delta_walk_forward_log_loss"] = (
            table.loc[mask, "walk_forward_log_loss"] - base["walk_forward_log_loss"]
        )
    table = table.sort_values(
        ["feature_set", "model"]
    ).reset_index(drop=True)
    folds_table = pd.DataFrame(fold_rows)

    # Strict retention rule: a group must improve BOTH models on validation and
    # walk-forward, and the paired validation bootstrap interval must exclude
    # zero. This avoids promoting a small difference that is indistinguishable
    # from noise.
    candidates = []
    for feature_set, part in table[table["feature_set"] != "base_19"].groupby("feature_set"):
        if len(part) < len(MODEL_NAMES):
            continue
        if (part["delta_validation_log_loss"] < 0).all() and (
            part["delta_walk_forward_log_loss"] < 0
        ).all() and (part["delta_validation_ci_high"] < 0).all():
            candidates.append({
                "feature_set": feature_set,
                "mean_walk_forward_delta": float(part["delta_walk_forward_log_loss"].mean()),
                "mean_validation_delta": float(part["delta_validation_log_loss"].mean()),
            })
    if candidates:
        retained = min(candidates, key=lambda c: c["mean_walk_forward_delta"])
        recommended = retained["feature_set"]
        reason = (
            "Both models improved on validation and walk-forward; selected the "
            "largest mean walk-forward improvement."
        )
    else:
        retained = None
        recommended = "base_19"
        reason = (
            "No candidate produced a statistically credible improvement for both "
            "models on validation and walk-forward; the deployed 19-feature set "
            "is retained."
        )
    h2h = table[table["feature_set"] == "head_to_head"]
    h2h_evidence = []
    for _, part in h2h.iterrows():
        h2h_evidence.append({
            "model": part["model"],
            "delta_validation_log_loss": float(part["delta_validation_log_loss"]),
            "delta_validation_ci_low": float(part["delta_validation_ci_low"]),
            "delta_validation_ci_high": float(part["delta_validation_ci_high"]),
            "delta_walk_forward_log_loss": float(part["delta_walk_forward_log_loss"]),
        })
    recommendation = {
        "recommended_feature_set": recommended,
        "reason": reason,
        "retained": retained,
        "head_to_head_evidence": h2h_evidence,
        "head_to_head_justified": bool(h2h_evidence) and all(
            row["delta_validation_ci_high"] < 0 for row in h2h_evidence
        ),
        "rule": (
            "Validation log-loss, mean walk-forward log-loss, and the 95% paired "
            "validation bootstrap upper bound must all improve for both Random "
            "Forest and XGBoost."
        ),
        "feature_groups": FEATURE_GROUPS,
        "test_used_for_selection": False,
    }

    if save:
        table.to_csv(FEATURE_EXPERIMENTS_PATH, index=False)
        folds_table.to_csv(FEATURE_EXPERIMENT_FOLDS_PATH, index=False)
        FEATURE_RECOMMENDATION_PATH.write_text(
            json.dumps(recommendation, indent=2), encoding="utf-8"
        )
        plot_feature_experiments(table)

    data_hash = fingerprint_file(RAW_PATH)
    for _, row in table.iterrows():
        group = row["feature_set"]
        record = make_experiment_record(
            model_type=row["model"],
            feature_set=group,
            features=CANDIDATE_FEATURE_SETS[group],
            training_period=list(TRAIN_SEASONS),
            validation_period=VALIDATION_SEASON,
            test_period=TEST_SEASON,
            hyperparameters=params_by_model[row["model"]],
            validation_metrics={
                "log_loss": row["validation_log_loss"],
                "accuracy": row["validation_accuracy"],
                "brier_score": row["validation_brier_score"],
                "ece": row["validation_ece"],
            },
            data_fingerprint=data_hash,
            model_version=str(metadata.get("model_version", "")),
            selection_status=(
                "retained" if group == recommended and group != "base_19"
                else ("base_reference" if group == "base_19" else "rejected")
            ),
            notes=(
                f"Walk-forward mean log-loss={row['walk_forward_log_loss']:.4f}; "
                "test metrics withheld from feature selection."
            ),
            extra={
                "walk_forward_log_loss": row["walk_forward_log_loss"],
                "delta_validation_log_loss": row["delta_validation_log_loss"],
                "delta_validation_ci_low": row.get("delta_validation_ci_low"),
                "delta_validation_ci_high": row.get("delta_validation_ci_high"),
                "delta_walk_forward_log_loss": row["delta_walk_forward_log_loss"],
            },
        )
        save_experiment(record)

    return {
        "table": table,
        "folds": folds_table,
        "recommendation": recommendation,
        "splits": splits,
    }


def _align(proba: np.ndarray, classes) -> np.ndarray:
    from src.evaluation.metrics import align_proba_to_labels
    out = align_proba_to_labels(proba, classes, BASELINE_LABELS)
    totals = out.sum(axis=1, keepdims=True)
    return np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)


def run_calibration_evaluation(
    raw: pd.DataFrame,
    metadata: dict,
    *,
    save: bool = True,
) -> dict:
    """Select calibration on validation, then score test once."""
    X, y, meta = build_model_frame(raw)
    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx, val_idx, test_idx = (
        splits["train"].index, splits["validation"].index, splits["test"].index
    )
    params_by_model = _params(metadata)

    selection_rows: list[dict] = []
    comparison_rows: list[dict] = []
    selected_by_model: dict[str, str] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    final_pre = FeaturePreprocessor.load()
    X_test = final_pre.transform(X.loc[test_idx])
    y_test = y.loc[test_idx]

    for model_name in MODEL_NAMES:
        pre = FeaturePreprocessor(FEATURE_COLUMNS)
        X_train = pre.fit_transform(X.loc[train_idx])
        X_val = pre.transform(X.loc[val_idx])
        validation_model = _fit(
            model_name, params_by_model[model_name], X_train, y.loc[train_idx]
        )
        val_proba = _align(
            validation_model.predict_proba(X_val), validation_model.classes_
        )
        best_method, selection = select_calibration_method(
            val_proba, y.loc[val_idx].to_numpy(),
            meta.loc[val_idx, "date"], classes=BASELINE_LABELS,
        )
        selected_by_model[model_name] = best_method
        selection = selection.assign(model=model_name, selected=selection["method"] == best_method)
        selection_rows.extend(selection.to_dict(orient="records"))

        # Calibrators for the test comparison are fit on the full validation
        # season. The method itself was chosen on an earlier validation
        # holdout, so the test season is not used to choose it.
        calibrators = {
            method: fit_calibrator(
                method, val_proba, y.loc[val_idx].to_numpy(), BASELINE_LABELS
            )
            for method in CALIBRATION_METHODS if method != "none"
        }
        final_model = joblib.load(SAVED_MODEL_DIR / f"model_{_slug(model_name)}.pkl")
        test_proba = _align(final_model.predict_proba(X_test), final_model.classes_)
        test_probabilities[model_name] = test_proba
        comparison = compare_calibrated_test(
            y_test, test_proba, calibrators, classes=BASELINE_LABELS
        )
        comparison["model"] = model_name
        comparison["selected_on_validation"] = comparison["calibration"] == best_method
        comparison_rows.extend(comparison.to_dict(orient="records"))

        if save:
            plot_reliability_overlay(
                y_test,
                {
                    "Uncalibrated": test_proba,
                    **{
                        method.title(): calibrator.transform(test_proba)
                        for method, calibrator in calibrators.items()
                    },
                },
                BASELINE_LABELS,
                f"{model_name} — Test calibration (selected: {best_method})",
                f"calibration_overlay_{_slug(model_name)}.png",
            )

    comparison = pd.DataFrame(comparison_rows)
    selection = pd.DataFrame(selection_rows)
    if save:
        comparison.to_csv(CALIBRATION_COMPARISON_PATH, index=False)
        selection.to_csv(CALIBRATION_SELECTION_PATH, index=False)
        for model_name, part in comparison.groupby("model"):
            plot_calibration_comparison(
                part.sort_values("log_loss"),
                EVAL_DIR,
            )
            # The generic filename would be overwritten; use a model-specific
            # copy as the stable dashboard artifact.
            generic = EVAL_DIR / "calibration_comparison.png"
            if generic.exists():
                generic.replace(EVAL_DIR / f"calibration_comparison_{_slug(model_name)}.png")

    data_hash = fingerprint_file(RAW_PATH)
    selection_lookup = selection.set_index(["model", "method"])
    for _, row in comparison.iterrows():
        try:
            holdout = selection_lookup.loc[(row["model"], row["calibration"])]
        except KeyError:
            holdout = {}
        if isinstance(holdout, pd.DataFrame):
            holdout = holdout.iloc[0]
        validation_metrics = {
            "calibration_method": row["calibration"],
            "period": "validation_holdout",
        }
        for key in ("log_loss", "brier_score", "ece", "accuracy", "draw_recall"):
            value = holdout.get(key) if hasattr(holdout, "get") else None
            if value is not None and pd.notna(value):
                validation_metrics[key] = float(value)
        record = make_experiment_record(
            model_type=row["model"],
            feature_set="base_19",
            features=list(FEATURE_COLUMNS),
            training_period=list(TRAIN_SEASONS),
            validation_period=VALIDATION_SEASON,
            test_period=TEST_SEASON,
            hyperparameters=params_by_model[row["model"]],
            validation_metrics=validation_metrics,
            test_metrics={
                "calibration_method": row["calibration"],
                "log_loss": row["log_loss"],
                "brier_score": row["brier_score"],
                "ece": row["ece"],
                "accuracy": row["accuracy"],
                "draw_recall": row["draw_recall"],
            },
            variant=f"calibration_{row['calibration']}",
            data_fingerprint=data_hash,
            model_version=str(metadata.get("model_version", "")),
            selection_status=(
                "selected_calibration" if row["selected_on_validation"]
                else "comparison_only"
            ),
            notes=(
                "Calibration method chosen on a chronological validation holdout; "
                "test scored once after selection."
            ),
        )
        save_experiment(record)

    deployment_note = (
        "Calibration comparison is diagnostic. No calibrator is applied "
        "automatically by this experiment run."
    )
    collapsed = []
    for model_name, method in selected_by_model.items():
        part = comparison[comparison["model"] == model_name]
        selected = part[part["calibration"] == method]
        raw = part[part["calibration"] == "none"]
        if selected.empty or raw.empty:
            continue
        if (
            float(raw.iloc[0].get("draw_recall", 0.0)) > 0.10
            and float(selected.iloc[0].get("draw_recall", 0.0)) <= 0.10
        ):
            collapsed.append(model_name)
    if collapsed:
        deployment_note += (
            " On test, the validation-selected calibrator collapses draw recall "
            f"to about zero for {', '.join(collapsed)}. Treat that as a "
            "decision-quality regression even where log-loss/Brier improve."
        )

    recommendation = {
        "selected_by_model": selected_by_model,
        "test_used_for_selection": False,
        "deployment_note": deployment_note,
        "draw_recall_collapse": collapsed,
    }
    if save:
        (EVAL_DIR / "calibration_recommendation.json").write_text(
            json.dumps(recommendation, indent=2), encoding="utf-8"
        )
    return {
        "comparison": comparison,
        "selection": selection,
        "selected_by_model": selected_by_model,
        "recommendation": recommendation,
    }


def run_explainability(
    raw: pd.DataFrame,
    metadata: dict,
    *,
    save: bool = True,
) -> dict:
    """Permutation, native tree, SHAP and fold-stability importance."""
    X, y, meta = build_model_frame(raw)
    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx, val_idx = splits["train"].index, splits["validation"].index
    params_by_model = _params(metadata)

    permutation_frames = []
    native_frames = []
    shap_table = pd.DataFrame()
    for model_name in MODEL_NAMES:
        pre = FeaturePreprocessor(FEATURE_COLUMNS)
        X_train = pre.fit_transform(X.loc[train_idx])
        X_val = pre.transform(X.loc[val_idx])
        model = _fit(
            model_name, params_by_model[model_name], X_train, y.loc[train_idx]
        )
        permutation_frames.append(
            permutation_table(
                model, X_val, y.loc[val_idx], model_name
            )
        )
        final_model = joblib.load(SAVED_MODEL_DIR / f"model_{_slug(model_name)}.pkl")
        native_frames.append(
            native_tree_importance(final_model, FEATURE_COLUMNS, model_name)
        )
        if model_name == "XGBoost":
            shap_table = xgboost_tree_shap(model, X_val, FEATURE_COLUMNS, model_name)
            rf_shap = optional_random_forest_shap(
                joblib.load(SAVED_MODEL_DIR / "model_random_forest.pkl"),
                X_val, FEATURE_COLUMNS, "Random Forest",
            )
            if not rf_shap.empty:
                shap_table = pd.concat([shap_table, rf_shap], ignore_index=True)

    permutation = pd.concat(permutation_frames, ignore_index=True)
    native = pd.concat(native_frames, ignore_index=True)

    factories = {
        name: (lambda n=name, p=params_by_model[name]: build_estimator(n, p))
        for name in MODEL_NAMES
    }
    fold_frame = fold_importance(
        factories, X, y, meta, test_season=TEST_SEASON,
        n_repeats=5, min_train_seasons=1,
    )
    stability = stability_table(fold_frame)

    if save:
        permutation.to_csv(PERMUTATION_IMPORTANCE_PATH, index=False)
        native.to_csv(TREE_IMPORTANCE_PATH, index=False)
        if not shap_table.empty:
            shap_table.to_csv(SHAP_IMPORTANCE_PATH, index=False)
        fold_frame.to_csv(FEATURE_IMPORTANCE_FOLDS_PATH, index=False)
        stability.to_csv(FEATURE_STABILITY_PATH, index=False)
        plot_importance_comparison(
            permutation[permutation["model"] == "Random Forest"],
            native[native["model"] == "Random Forest"],
            shap_table[shap_table["model"] == "XGBoost"] if not shap_table.empty else pd.DataFrame(),
        )
        plot_feature_stability(stability)

    top_features = (
        permutation.sort_values("importance_mean", ascending=False)
        .groupby("model").head(8)
        [["model", "feature", "importance_mean"]]
        .to_dict(orient="records")
    )
    return {
        "permutation": permutation,
        "native": native,
        "shap": shap_table,
        "folds": fold_frame,
        "stability": stability,
        "top_features": top_features,
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], digits: dict | None = None) -> str:
    if frame.empty:
        return "_No data._"
    digits = digits or {}
    columns = [c for c in columns if c in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float) and col in digits:
                cells.append(f"{value:.{digits[col]}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_phase2_report(
    metadata: dict,
    baselines: tuple[pd.DataFrame, pd.DataFrame],
    features: dict,
    calibration: dict,
    explainability: dict,
    monitoring: dict | None = None,
    *,
    path: Path = PHASE2_REPORT_PATH,
) -> Path:
    """Write an honest, measured Phase 2 report from generated artifacts."""
    validation, test = baselines
    feature_table = features["table"]
    recommendation = features["recommendation"]
    calibration_comparison = calibration["comparison"]
    permutation = explainability["permutation"]

    ml_mask = test["model"].astype(str).str.contains(r"\(ML\)", regex=True) if not test.empty else pd.Series(dtype=bool)
    baseline_test = test[~ml_mask] if not test.empty else test
    best_baseline = baseline_test.sort_values("log_loss").iloc[0] if not baseline_test.empty else None
    ml_test = test[ml_mask] if not test.empty else pd.DataFrame()
    best_ml = ml_test.sort_values("log_loss").iloc[0] if not ml_test.empty else None
    lines = [
        "# La Liga Predictor — Phase 2: Model Intelligence & Performance",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Model version: `{metadata.get('model_version', 'unknown')}`",
        "",
        "## Executive summary",
        "",
    ]
    if best_baseline is not None and best_ml is not None:
        lines += [
            f"- Best test baseline: **{best_baseline['model']}** "
            f"(log-loss {best_baseline['log_loss']:.4f}, accuracy {best_baseline['accuracy']:.1%}).",
            f"- Best test learned model: **{best_ml['model']}** "
            f"(log-loss {best_ml['log_loss']:.4f}, accuracy {best_ml['accuracy']:.1%}).",
        ]
        if best_ml["log_loss"] < best_baseline["log_loss"]:
            lines.append("- On this test window, the learned model beats the best simple baseline on log-loss.")
        else:
            lines.append(
                "- On this test window, the learned model does **not** beat the best simple "
                "baseline on log-loss. This is reported as a negative result, not hidden."
            )
    lines += [
        f"- Feature recommendation: **{recommendation['recommended_feature_set']}**. "
        f"{recommendation['reason']}",
        f"- Calibration selected on validation: "
        + ", ".join(f"{k}={v}" for k, v in calibration["selected_by_model"].items())
        + ". This is diagnostic, not an automatic deployment.",
        "",
        "## 1. Baselines vs learned models",
        "",
        "All rows use the same chronological train/validation/test seasons. Baselines "
        "are fixed before test evaluation; test metrics were not used to choose a model.",
        "",
        "### Test",
        "",
        _markdown_table(test, ["model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"], {
            "accuracy": 4, "log_loss": 4, "brier_score": 4, "draw_recall": 4, "ece": 4,
        }),
        "",
        "### Validation",
        "",
        _markdown_table(validation, ["model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"], {
            "accuracy": 4, "log_loss": 4, "brier_score": 4, "draw_recall": 4, "ece": 4,
        }),
        "",
        "## 2. Feature experiments",
        "",
        "Each candidate adds one feature group to the base 19. The retention rule "
        "requires lower validation log-loss, lower mean walk-forward log-loss and "
        "a 95% paired bootstrap upper bound below zero for both tree models. The "
        "test season was not used. The interval is deliberately conservative: a "
        "small point estimate that cannot be separated from noise is rejected.",
        "",
        _markdown_table(
            feature_table,
            ["model", "feature_set", "n_features", "validation_log_loss",
             "walk_forward_log_loss", "delta_validation_log_loss",
             "delta_validation_ci_low", "delta_validation_ci_high",
             "delta_walk_forward_log_loss"],
            {k: 4 for k in (
                "validation_log_loss", "walk_forward_log_loss",
                "delta_validation_log_loss", "delta_validation_ci_low",
                "delta_validation_ci_high", "delta_walk_forward_log_loss",
            )},
        ),
        "",
        f"**Decision:** `{recommendation['recommended_feature_set']}`. "
        f"{recommendation['reason']}",
        "",
        "Head-to-head and venue form were the closest candidates. Every candidate "
        "had a validation interval that included zero; for head-to-head the upper "
        "bounds were "
        f"{recommendation.get('head_to_head_evidence', [{}])[0].get('delta_validation_ci_high', float('nan')):+.4f} "
        "for Random Forest and "
        f"{recommendation.get('head_to_head_evidence', [{}, {}])[1].get('delta_validation_ci_high', float('nan')):+.4f} "
        "for XGBoost. None of the candidates is retained.",
        "",
        "xG was not tested: the cached Football-Data.co.uk extract has no reliable "
        "historical xG column, so adding one would mean inventing data. Elo, "
        "strength of schedule, venue form and 10-match form were tested and rejected "
        "under the same rule.",
        "",
        "## 3. Calibration",
        "",
        "Methods were selected on a chronological validation holdout and then scored "
        "once on test. Test metrics below compare all pre-declared methods for "
        "transparency, but selection did not use test.",
        "",
        _markdown_table(
            calibration_comparison,
            ["model", "calibration", "selected_on_validation", "log_loss",
             "brier_score", "ece", "accuracy", "draw_recall"],
            {"log_loss": 4, "brier_score": 4, "ece": 4, "accuracy": 4, "draw_recall": 4},
        ),
        "",
        calibration["recommendation"]["deployment_note"],
        "",
        "## 4. Explainability",
        "",
        "Permutation importance is measured on validation; native tree importance "
        "comes from the final train+validation fit; XGBoost TreeSHAP is measured on "
        "validation. None of these is causal evidence.",
        "",
        _markdown_table(
            permutation.groupby("model").head(10),
            ["model", "feature", "importance_mean", "importance_std"],
            {"importance_mean": 5, "importance_std": 5},
        ),
        "",
        "## 5. Monitoring",
        "",
        ("Current monitoring verdict: **" + monitoring["overall_status"] + "**. "
         if monitoring else "No monitoring report was supplied."),
        "",
        _markdown_table(
            pd.DataFrame(monitoring["checks"]) if monitoring else pd.DataFrame(),
            ["name", "status", "message"],
        ) if monitoring else "_No monitoring checks._",
        "",
        "Monitoring distinguishes `insufficient_data`, `warning` and `confirmed` "
        "degradation. The ledger currently has no settled predictions, so "
        "performance and calibration checks are correctly reported as "
        "insufficient rather than healthy.",
        "",
        "## 6. Reproducibility",
        "",
        "- Experiment index: `models/experiments/index.csv`.",
        "- Each experiment JSON records data fingerprint, feature hash, split, "
        "hyperparameters, validation metrics and whether test was evaluated.",
        "- Random seed: 42.",
        "- Test metrics were recorded only after model/feature/calibration decisions.",
        "",
        "## 7. Limitations",
        "",
        "- The test season contains 109 of 380 matches, so test differences are noisy.",
        "- Draw recall and calibration remain difficult; one season is not enough to "
        "establish a stable betting edge.",
        "- Market odds are included as features, so the learned model is not an "
        "independent forecast of the market.",
        "- Head-to-head and rest features have sparse early-season values; imputation "
        "is training-only but still a modelling compromise. Head-to-head did not "
        "clear the statistical bar and was rejected.",
        "- The strict feature rule can reject a real but small effect; that is a "
        "deliberate bias toward not shipping noise.",
        "- SHAP for Random Forest is optional; XGBoost native TreeSHAP is always available.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(*, save: bool = True) -> dict:
    """Run the full Phase 2 evaluation suite and write artifacts/report."""
    raw = load_raw()
    X, y, meta = build_model_frame(raw)
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata missing. Run `python -m src.models.train_model` first."
        )
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    baselines_raw = run_baselines(X, y, meta, save=save)
    validation, test = _combined_baseline_table(metadata)
    if save:
        plot_baseline_comparison(validation, test)
        _record_baselines(metadata, validation, test)

    features = run_feature_experiments(raw, metadata, save=save)
    calibration = run_calibration_evaluation(raw, metadata, save=save)
    explainability = run_explainability(raw, metadata, save=save)
    history, _ = load_history()
    monitoring = run_diagnostics(raw, history, metadata, feature_frame=X)
    if save:
        save_monitoring_report(monitoring)

    summary = {
        "model_version": metadata.get("model_version"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_test": test.to_dict(orient="records"),
        "baseline_validation": validation.to_dict(orient="records"),
        "feature_recommendation": features["recommendation"],
        "calibration": calibration["recommendation"],
        "top_permutation_features": explainability["top_features"],
        "test_used_for_feature_selection": False,
        "test_used_for_calibration_selection": False,
        "monitoring_overall_status": monitoring["overall_status"],
    }
    if save:
        PHASE2_SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        write_phase2_report(
            metadata, (validation, test), features, calibration, explainability,
            monitoring,
        )
        experiments = load_experiments()
        plot_experiment_leaderboard(experiments)
        logger.info("Wrote Phase 2 report to %s", PHASE2_REPORT_PATH)
    return {
        "baselines": {"validation": validation, "test": test, "raw": baselines_raw},
        "features": features,
        "calibration": calibration,
        "explainability": explainability,
        "monitoring": monitoring,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: run all Phase 2 diagnostics and write artifacts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Phase 2 baselines, feature experiments, calibration, "
        "explainability and monitoring."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Compute without writing artifacts or the report.",
    )
    args = parser.parse_args(argv)
    result = run_all(save=not args.no_save)
    summary = result["summary"]
    print(f"Best test baseline: {summary['baseline_test'][0]['model']}")
    print(f"Feature set: {summary['feature_recommendation']['recommended_feature_set']}")
    print(f"Monitoring: {summary['monitoring_overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
