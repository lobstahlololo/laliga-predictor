"""Out-of-sample feature importance and stability diagnostics.

The primary signal is permutation importance on a held-out season. Impurity
importance is included as a model-native comparison, and XGBoost's native
TreeSHAP contributions provide a third view without adding a heavyweight
dependency. None of these measures establishes causation.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.evaluation.splits import walk_forward_folds
from src.features.preprocessing import FeaturePreprocessor
from src.utils.config import FEATURE_COLUMNS, PERMUTATION_REPEATS, RANDOM_STATE

logger = logging.getLogger(__name__)


def native_tree_importance(
    model,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    model_name: str = "model",
) -> pd.DataFrame:
    """Tree impurity/gain importance, normalized to sum to one per model."""
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    values = np.asarray(values, dtype=float)
    if values.shape[0] != len(list(feature_names)):
        raise ValueError("Feature importance length does not match feature names.")
    total = values.sum()
    if total > 0:
        values = values / total
    return pd.DataFrame({
        "model": model_name,
        "feature": list(feature_names),
        "importance": values,
    }).sort_values("importance", ascending=False, ignore_index=True)


def permutation_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_repeats: int = PERMUTATION_REPEATS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance using multiclass log-loss on held-out rows.

    Lower (more negative) scores are worse, so a larger positive importance
    means permuting that feature damaged the model more.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_log_loss",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return pd.DataFrame({
        "model": model_name,
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


def xgboost_tree_shap(
    model,
    X: pd.DataFrame,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    model_name: str = "XGBoost",
) -> pd.DataFrame:
    """Mean absolute TreeSHAP contribution across the three outcome classes.

    XGBoost's native ``pred_contribs`` is exact TreeSHAP for the fitted trees
    and needs no extra package. The trailing bias column is excluded.
    """
    import xgboost as xgb

    booster = model.get_booster()
    contributions = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim == 3:
        # (rows, classes, features + bias)
        values = np.abs(contributions[:, :, :-1]).mean(axis=(0, 1))
    elif contributions.ndim == 2:
        values = np.abs(contributions[:, :-1]).mean(axis=0)
    else:
        raise ValueError(f"Unexpected SHAP contribution shape: {contributions.shape}")
    names = list(feature_names)
    if len(values) != len(names):
        raise ValueError("SHAP contribution count does not match feature names.")
    frame = pd.DataFrame({
        "model": model_name,
        "feature": names,
        "mean_abs_shap": values,
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    if frame["mean_abs_shap"].sum() > 0:
        frame["share"] = frame["mean_abs_shap"] / frame["mean_abs_shap"].sum()
    else:
        frame["share"] = 0.0
    return frame


def optional_random_forest_shap(
    model,
    X: pd.DataFrame,
    feature_names: Iterable[str] = FEATURE_COLUMNS,
    model_name: str = "Random Forest",
) -> pd.DataFrame:
    """RF SHAP when the optional ``shap`` package is installed.

    Returns an empty frame with a ``note`` column when unavailable. The main
    artifacts use XGBoost TreeSHAP instead of making ``shap`` mandatory.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        return pd.DataFrame(columns=["model", "feature", "mean_abs_shap", "note"])
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    # RF multiclass may return (classes, rows, features) or (rows, features, classes).
    if values.ndim == 3:
        if values.shape[0] == len(getattr(model, "classes_", [])):
            values = np.abs(values).mean(axis=(0, 1))
        else:
            values = np.abs(values).mean(axis=(0, 2))
    else:
        values = np.abs(values).mean(axis=0)
    names = list(feature_names)
    return pd.DataFrame({
        "model": model_name,
        "feature": names,
        "mean_abs_shap": values[: len(names)],
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def fold_importance(
    model_factories: dict[str, Callable[[], object]],
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    *,
    test_season: str,
    n_repeats: int = 5,
    min_train_seasons: int = 1,
) -> pd.DataFrame:
    """Importance across walk-forward folds, excluding the final test season.

    Both impurity and permutation importance are computed from each fold's
    own training fit and its out-of-sample test season.
    """
    folds = [
        fold for fold in walk_forward_folds(meta, min_train_seasons=min_train_seasons)
        if fold.test_season != test_season
    ]
    rows: list[dict] = []
    for fold in folds:
        pre = FeaturePreprocessor(list(X.columns))
        X_train = pre.fit_transform(X.loc[fold.train_index])
        X_test = pre.transform(X.loc[fold.test_index])
        y_train = y.loc[fold.train_index]
        y_test = y.loc[fold.test_index]
        for model_name, factory in model_factories.items():
            model = factory()
            model.fit(X_train, y_train)
            native = native_tree_importance(model, X.columns, model_name)
            for _, row in native.iterrows():
                rows.append({
                    "fold": fold.fold,
                    "test_season": fold.test_season,
                    "model": model_name,
                    "feature": row["feature"],
                    "method": "tree",
                    "importance": float(row["importance"]),
                })
            perm = permutation_table(
                model, X_test, y_test, model_name, n_repeats=n_repeats
            )
            for _, row in perm.iterrows():
                rows.append({
                    "fold": fold.fold,
                    "test_season": fold.test_season,
                    "model": model_name,
                    "feature": row["feature"],
                    "method": "permutation",
                    "importance": float(row["importance_mean"]),
                })
    return pd.DataFrame(rows)


def stability_table(fold_importance_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise importance stability across folds for each model/method."""
    if fold_importance_frame.empty:
        return pd.DataFrame()
    rows = []
    for (model, method), part in fold_importance_frame.groupby(["model", "method"]):
        pivot = part.pivot_table(
            index="feature", columns="test_season", values="importance", aggfunc="mean"
        )
        if pivot.empty:
            continue
        ranks = pivot.rank(ascending=False, axis=0, method="average")
        for feature in pivot.index:
            values = pivot.loc[feature].dropna()
            rank_values = ranks.loc[feature].dropna()
            mean_importance = float(values.mean()) if not values.empty else np.nan
            std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            rows.append({
                "model": model,
                "method": method,
                "feature": feature,
                "folds": int(len(values)),
                "mean_importance": mean_importance,
                "std_importance": std,
                "cv_importance": std / abs(mean_importance) if mean_importance else np.nan,
                "mean_rank": float(rank_values.mean()) if not rank_values.empty else np.nan,
                "rank_std": float(rank_values.std(ddof=0)) if len(rank_values) > 1 else 0.0,
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(
        ["model", "method", "mean_rank"], ascending=[True, True, True]
    ).reset_index(drop=True)
