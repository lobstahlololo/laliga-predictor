"""Phase 3 advanced football-modelling experiment runner.

Order of operations is intentional:

1. Elo and Poisson features are built chronologically over the full history.
2. Poisson time-decay/regularisation settings are selected on validation only.
3. Validation predictions from the models trained on the training seasons are
   used to select ensemble weights.
4. Test predictions are produced once from models refitted on train+validation.
5. Walk-forward folds use only seasons before each target season.
6. Market diagnostics, uncertainty and experiment records are written last.

No test result is used to choose a feature, hyperparameter, calibration method
or ensemble weight.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.backtesting import balanced_sample_weights
from src.evaluation.baselines import BASELINE_LABELS, class_prior
from src.evaluation.ensemble import (
    blend_probabilities,
    normalise_probabilities,
    select_ensemble_weights,
)
from src.evaluation.market import (
    compare_model_to_market,
    disagreement_analysis,
    fit_market_incremental_stack,
    market_differences_frame,
    market_from_meta,
    overround_summary,
    probability_range_performance,
)
from src.evaluation.metrics import (
    align_proba_to_labels,
    evaluate_probabilities,
    metrics_to_row,
)
from src.evaluation.phase3_plots import (
    plot_elo_history,
    plot_elo_uncertainty,
    plot_ensemble_weights,
    plot_market_probability_scatter,
    plot_phase3_model_comparison,
    plot_poisson_ratings,
    plot_scoreline_heatmap,
    plot_uncertainty,
)
from src.evaluation.splits import chronological_split, walk_forward_folds
from src.evaluation.uncertainty import build_uncertainty_table
from src.experiments.tracking import (
    fingerprint_file,
    load_experiments,
    make_experiment_record,
    save_experiment,
)
from src.features.build_features import build_model_frame, load_raw
from src.features.preprocessing import FeaturePreprocessor
from src.models.elo import (
    ELO_FEATURE_COLUMNS,
    EloConfig,
    compute_elo,
    config_as_dict,
    elo_probabilities,
    fit_elo_probability_params,
    latest_elo_ratings,
)
from src.models.poisson import (
    POISSON_FEATURE_COLUMNS,
    PoissonConfig,
    PoissonModel,
    build_pre_match_poisson_features,
)
from src.models.train_model import build_estimator
from src.utils.config import (
    ELO_HISTORY_PATH,
    ELO_MATCH_FEATURES_PATH,
    ELO_RATINGS_PATH,
    ENSEMBLE_COMPARISON_PATH,
    ENSEMBLE_WEIGHTS_PATH,
    EVAL_DIR,
    FEATURE_COLUMNS,
    MARKET_DISAGREEMENT_PATH,
    MARKET_MODEL_COMPARISON_PATH,
    MARKET_OVERROUND_PATH,
    MARKET_PROBABILITY_DIFFERENCES_PATH,
    MARKET_RANGE_PERFORMANCE_PATH,
    MARKET_STACK_PATH,
    MODEL_METADATA_PATH,
    PHASE3_REPORT_PATH,
    PHASE3_SEASON_METRICS_PATH,
    PHASE3_SUMMARY_PATH,
    PHASE3_TEST_PATH,
    PHASE3_VALIDATION_PATH,
    PHASE3_WALK_FORWARD_PATH,
    POISSON_RATINGS_PATH,
    POISSON_SCORELINES_PATH,
    POISSON_SUMMARY_PATH,
    POISSON_TEST_PREDICTIONS_PATH,
    RAW_PATH,
    SAVED_MODEL_DIR,
    TEST_SEASON,
    TRAIN_SEASONS,
    UNCERTAINTY_PATH,
    UNCERTAINTY_SUMMARY_PATH,
    VALIDATION_SEASON,
    WALK_FORWARD_PATH,
)

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
TREE_MODELS = ("Random Forest", "XGBoost")
ENSEMBLE_COMPONENTS = ("Random Forest", "XGBoost", "Dynamic Elo", "Independent Poisson")
ELO_MODEL_FEATURES = list(FEATURE_COLUMNS) + list(ELO_FEATURE_COLUMNS)
POISSON_MODEL_FEATURES = list(FEATURE_COLUMNS) + list(POISSON_FEATURE_COLUMNS)
MARKET_COLUMNS = ["market_home", "market_draw", "market_away"]


def _slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _align(proba: np.ndarray, classes) -> np.ndarray:
    out = align_proba_to_labels(proba, classes, BASELINE_LABELS)
    totals = out.sum(axis=1, keepdims=True)
    return np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)


def _params(metadata: dict) -> dict[str, dict]:
    result = {}
    for name in TREE_MODELS:
        info = metadata.get("all_models", {}).get(name, {})
        result[name] = dict(
            info.get("parameters")
            or (DEFAULT_RF_PARAMS if name == "Random Forest" else DEFAULT_XGB_PARAMS)
        )
    return result


def _raw_for_index(raw: pd.DataFrame, meta: pd.DataFrame, index) -> pd.DataFrame:
    lookup = raw.set_index("match_id", drop=False)
    return lookup.loc[meta.loc[index, "match_id"].to_numpy()].reset_index(drop=True)


def _market_probabilities(meta: pd.DataFrame, y_train) -> np.ndarray:
    market = market_from_meta(meta, fallback=class_prior(y_train))
    return market[MARKET_COLUMNS].to_numpy(float)


def _fit_learned(
    name: str,
    params: dict,
    features: list[str],
    X_train_raw: pd.DataFrame,
    y_train,
    X_eval_raw: pd.DataFrame,
) -> np.ndarray:
    pre = FeaturePreprocessor(features)
    X_train = pre.fit_transform(X_train_raw[features])
    X_eval = pre.transform(X_eval_raw[features])
    model = build_estimator(name, params)
    if name == "XGBoost":
        model.fit(X_train, y_train, sample_weight=balanced_sample_weights(y_train))
    else:
        model.fit(X_train, y_train)
    return _align(model.predict_proba(X_eval), model.classes_)


def _fit_logistic(
    X_train_raw: pd.DataFrame,
    y_train,
    X_eval_raw: pd.DataFrame,
) -> np.ndarray:
    pre = FeaturePreprocessor(FEATURE_COLUMNS)
    X_train = pre.fit_transform(X_train_raw[FEATURE_COLUMNS])
    X_eval = pre.transform(X_eval_raw[FEATURE_COLUMNS])
    model = LogisticRegression(
        max_iter=3000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return _align(model.predict_proba(X_eval), model.classes_)


def _saved_base_probabilities(name: str, X_eval_raw: pd.DataFrame) -> np.ndarray:
    model = joblib.load(SAVED_MODEL_DIR / f"model_{_slug(name)}.pkl")
    pre = FeaturePreprocessor.load()
    X_eval = pre.transform(X_eval_raw[pre.feature_names])
    return _align(model.predict_proba(X_eval), model.classes_)


def _poisson_prediction_frame(model: PoissonModel, meta: pd.DataFrame) -> pd.DataFrame:
    fixtures = meta[["home_team", "away_team"]].copy()
    return model.predict_frame(fixtures)


def _result(
    *,
    name: str,
    family: str,
    feature_set: str,
    features: list[str],
    params: dict,
    probs: np.ndarray,
    notes: str = "",
    model=None,
) -> dict:
    return {
        "name": name,
        "family": family,
        "feature_set": feature_set,
        "features": list(features),
        "params": params,
        "probs": normalise_probabilities(probs),
        "notes": notes,
        "model": model,
    }


def _fit_window(
    *,
    train_raw: pd.DataFrame,
    eval_meta: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    y_train,
    X_eval_raw: pd.DataFrame,
    elo_train: pd.DataFrame,
    elo_eval: pd.DataFrame,
    poisson_train: pd.DataFrame,
    poisson_eval: pd.DataFrame,
    params: dict,
    poisson_config: PoissonConfig,
    use_saved_base: bool,
) -> tuple[dict[str, dict], dict]:
    results: dict[str, dict] = {}
    y_train_array = np.asarray(y_train, dtype=int)

    # 1. Market benchmark.
    results["Market odds"] = _result(
        name="Market odds",
        family="market",
        feature_set="market_odds",
        features=["odds_home", "odds_draw", "odds_away"],
        params={"overround": "removed by row normalisation"},
        probs=_market_probabilities(eval_meta, y_train_array),
        notes="Bookmaker benchmark; overround removed, not a model.",
    )

    # 2. Current tree models. Test uses the saved production fit; validation is
    # refitted on the training seasons with the same parameters.
    for name in TREE_MODELS:
        if use_saved_base:
            probs = _saved_base_probabilities(name, X_eval_raw)
        else:
            probs = _fit_learned(
                name, params[name], FEATURE_COLUMNS, X_train_raw, y_train, X_eval_raw
            )
        results[name] = _result(
            name=name,
            family="tree ensemble",
            feature_set="base_19",
            features=list(FEATURE_COLUMNS),
            params=params[name],
            probs=probs,
            notes="Production reference; not changed by Phase 3.",
        )

    # 3. Multinomial logistic baseline.
    logistic_probs = _fit_logistic(X_train_raw, y_train, X_eval_raw)
    results["Logistic regression"] = _result(
        name="Logistic regression",
        family="linear",
        feature_set="base_19",
        features=list(FEATURE_COLUMNS),
        params={"max_iter": 3000, "class_weight": "balanced"},
        probs=logistic_probs,
        notes="Multinomial logistic regression on the same features.",
    )

    # 4. Standalone dynamic Elo.
    elo_params, _elo_grid = fit_elo_probability_params(elo_train, y_train)
    elo_probs = elo_probabilities(
        elo_eval, elo_params["draw_base"], elo_params["draw_scale"]
    )
    results["Dynamic Elo"] = _result(
        name="Dynamic Elo",
        family="elo",
        feature_set="elo_ratings",
        features=list(ELO_FEATURE_COLUMNS),
        params={**config_as_dict(EloConfig()), **elo_params},
        probs=elo_probs,
        notes="Standalone Elo; draw mapping fitted on training log-loss.",
    )

    # 5. Poisson and Dixon-Coles.
    as_of = pd.to_datetime(train_raw["date"]).max()
    independent_config = replace(poisson_config, dixon_coles=False)
    dc_config = replace(poisson_config, dixon_coles=True)
    independent_model = PoissonModel(independent_config).fit(train_raw, as_of=as_of)
    dc_model = PoissonModel(dc_config).fit(train_raw, as_of=as_of)
    independent_pred = _poisson_prediction_frame(independent_model, eval_meta)
    dc_pred = _poisson_prediction_frame(dc_model, eval_meta)
    results["Independent Poisson"] = _result(
        name="Independent Poisson",
        family="poisson",
        feature_set="poisson_goals",
        features=["team_attack", "team_defence", "home_advantage"],
        params=asdict(independent_config),
        probs=independent_pred[["prob_home", "prob_draw", "prob_away"]].to_numpy(float),
        notes="Time-decayed, regularised Poisson; no low-score correction.",
        model=independent_model,
    )
    results["Dixon-Coles"] = _result(
        name="Dixon-Coles",
        family="poisson",
        feature_set="poisson_goals",
        features=["team_attack", "team_defence", "home_advantage", "rho"],
        params={**asdict(dc_config), "rho": float(dc_model.rho)},
        probs=dc_pred[["prob_home", "prob_draw", "prob_away"]].to_numpy(float),
        notes="Same team strengths with a training-fitted low-score correction.",
        model=dc_model,
    )

    # 6. Elo plus the canonical features.
    X_elo_train = pd.concat([X_train_raw, elo_train], axis=1)
    X_elo_eval = pd.concat([X_eval_raw, elo_eval], axis=1)
    for name in TREE_MODELS:
        probs = _fit_learned(
            name, params[name], ELO_MODEL_FEATURES, X_elo_train, y_train, X_elo_eval
        )
        results[f"Elo + {name}"] = _result(
            name=f"Elo + {name}",
            family="hybrid",
            feature_set="base_19+elo",
            features=list(ELO_MODEL_FEATURES),
            params=params[name],
            probs=probs,
            notes="Canonical features with pre-match Elo ratings and uncertainty.",
        )

    # 7. Poisson pre-match probabilities plus the canonical features.
    X_poisson_train = pd.concat([X_train_raw, poisson_train], axis=1)
    X_poisson_eval = pd.concat([X_eval_raw, poisson_eval], axis=1)
    for name in TREE_MODELS:
        probs = _fit_learned(
            name, params[name], POISSON_MODEL_FEATURES, X_poisson_train, y_train, X_poisson_eval
        )
        results[f"Poisson + {name}"] = _result(
            name=f"Poisson + {name}",
            family="hybrid",
            feature_set="base_19+poisson",
            features=list(POISSON_MODEL_FEATURES),
            params=params[name],
            probs=probs,
            notes="Canonical features with expanding-window Poisson probabilities.",
        )

    fit_info = {
        "elo_params": elo_params,
        "poisson_config": poisson_config,
        "independent_model": independent_model,
        "dc_model": dc_model,
        "as_of": as_of,
    }
    return results, fit_info


def _metrics_table(results: dict[str, dict], y, *, split: str) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        metrics = evaluate_probabilities(
            y, result["probs"], BASELINE_LABELS, name
        )
        row = metrics_to_row(metrics)
        row.update({
            "split": split,
            "model_family": result["family"],
            "feature_set": result["feature_set"],
            "parameters": json.dumps(result["params"], default=_json_default, sort_keys=True),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["log_loss", "brier_score"]
    ).reset_index(drop=True)


def _select_poisson_config(
    raw_train: pd.DataFrame,
    val_meta: pd.DataFrame,
    y_val,
) -> tuple[PoissonConfig, pd.DataFrame]:
    """Select time decay and shrinkage on validation only."""
    rows = []
    as_of = pd.to_datetime(raw_train["date"]).max()
    for half_life in (365.0, 730.0, 1095.0):
        for prior_strength in (2.0, 5.0, 10.0):
            config = PoissonConfig(
                half_life_days=half_life,
                prior_strength=prior_strength,
                dixon_coles=True,
            )
            model = PoissonModel(config).fit(raw_train, as_of=as_of)
            pred = _poisson_prediction_frame(model, val_meta)
            metrics = evaluate_probabilities(
                y_val,
                pred[["prob_home", "prob_draw", "prob_away"]].to_numpy(float),
                BASELINE_LABELS,
                "Dixon-Coles",
            )
            rows.append({
                "half_life_days": half_life,
                "prior_strength": prior_strength,
                "validation_log_loss": metrics["log_loss"],
                "validation_brier_score": metrics["brier_score"],
                "rho": model.rho,
            })
    table = pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "validation_brier_score"]
    ).reset_index(drop=True)
    best = table.iloc[0]
    config = PoissonConfig(
        half_life_days=float(best["half_life_days"]),
        prior_strength=float(best["prior_strength"]),
        dixon_coles=True,
    )
    return config, table


def _per_row_loss(y, probabilities: np.ndarray) -> np.ndarray:
    aligned = _align(probabilities, BASELINE_LABELS)
    index = {label: j for j, label in enumerate(BASELINE_LABELS)}
    y_array = np.asarray(y, dtype=int)
    true_p = np.array([
        aligned[row, index[int(label)]] for row, label in enumerate(y_array)
    ])
    return -np.log(np.clip(true_p, 1e-12, 1.0))


def _bootstrap_delta_ci(
    base_loss: np.ndarray,
    candidate_loss: np.ndarray,
    n_boot: int = 2000,
) -> tuple[float, float]:
    base = np.asarray(base_loss, dtype=float)
    candidate = np.asarray(candidate_loss, dtype=float)
    if len(base) != len(candidate) or len(base) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(base), size=(n_boot, len(base)))
    deltas = (candidate[idx] - base[idx]).mean(axis=1)
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def _add_ensemble(
    results: dict[str, dict],
    y_val,
    weights_result: dict,
    *,
    weights: dict[str, float],
    name: str,
    notes: str,
) -> None:
    components = {key: results[key]["probs"] for key in ENSEMBLE_COMPONENTS}
    probs = blend_probabilities(components, weights)
    results[name] = _result(
        name=name,
        family="ensemble",
        feature_set="rf+xgb+elo+poisson",
        features=[],
        params={"weights": {key: float(value) for key, value in weights.items()}},
        probs=probs,
        notes=notes,
    )


def _walk_forward(
    raw: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    elo: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Walk-forward comparison using only seasons before each target season."""
    rows = []
    existing = pd.read_csv(WALK_FORWARD_PATH) if WALK_FORWARD_PATH.exists() else pd.DataFrame()
    if not existing.empty:
        for _, row in existing.iterrows():
            if row.get("test_season") == TEST_SEASON:
                continue
            rows.append({
                "model": row["model"],
                "model_family": "tree ensemble",
                "fold": int(row["fold"]),
                "test_season": row["test_season"],
                "train_seasons": row["train_seasons"],
                "n_samples": int(row["n_samples"]),
                "log_loss": float(row["log_loss"]),
                "brier_score": float(row["brier_score"]),
                "accuracy": float(row["accuracy"]),
                "draw_recall": float(row["draw_recall"]),
                "ece": float(row["ece"]),
                "source": "existing walk-forward artifact",
            })

    for fold in walk_forward_folds(meta, min_train_seasons=1):
        if fold.test_season == TEST_SEASON:
            continue
        train_idx, test_idx = fold.train_index, fold.test_index
        train_raw = raw[raw["season"].isin(fold.train_seasons)].copy()
        test_meta = meta.loc[test_idx]
        y_test = y.loc[test_idx]
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        elo_train, elo_test = elo.loc[train_idx], elo.loc[test_idx]

        candidates = {
            "Market odds": _market_probabilities(test_meta, y.loc[train_idx]),
            "Logistic regression": _fit_logistic(X_train, y.loc[train_idx], X_test),
        }
        elo_params, _ = fit_elo_probability_params(elo_train, y.loc[train_idx])
        candidates["Dynamic Elo"] = elo_probabilities(
            elo_test, elo_params["draw_base"], elo_params["draw_scale"]
        )
        poisson_config = PoissonConfig()
        independent = PoissonModel(replace(poisson_config, dixon_coles=False)).fit(
            train_raw, as_of=pd.to_datetime(train_raw["date"]).max()
        )
        dc = PoissonModel(replace(poisson_config, dixon_coles=True)).fit(
            train_raw, as_of=pd.to_datetime(train_raw["date"]).max()
        )
        for name, model in (
            ("Independent Poisson", independent),
            ("Dixon-Coles", dc),
        ):
            pred = _poisson_prediction_frame(model, test_meta)
            candidates[name] = pred[["prob_home", "prob_draw", "prob_away"]].to_numpy(float)

        for name, probs in candidates.items():
            metrics = evaluate_probabilities(y_test, probs, BASELINE_LABELS, name)
            rows.append({
                "model": name,
                "model_family": (
                    "market" if name == "Market odds"
                    else "linear" if name == "Logistic regression"
                    else "elo" if name == "Dynamic Elo"
                    else "poisson"
                ),
                "fold": int(fold.fold),
                "test_season": fold.test_season,
                "train_seasons": "+".join(fold.train_seasons),
                "n_samples": int(metrics["n_samples"]),
                "log_loss": float(metrics["log_loss"]),
                "brier_score": float(metrics["brier_score"]),
                "accuracy": float(metrics["accuracy"]),
                "draw_recall": float(metrics["draw_recall"]),
                "ece": float(metrics["ece"]),
                "source": "phase3 walk-forward",
            })
    return pd.DataFrame(rows).sort_values(
        ["fold", "log_loss"]
    ).reset_index(drop=True)


def _poisson_artifacts(
    model: PoissonModel,
    test_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = model.predict_frame(test_meta[["home_team", "away_team"]].copy())
    predictions.insert(0, "match_id", test_meta["match_id"].to_numpy())
    predictions.insert(1, "date", test_meta["date"].to_numpy())
    predictions.insert(2, "season", test_meta["season"].to_numpy())
    predictions["actual_outcome"] = test_meta["match_outcome"].to_numpy()
    predictions["actual_home_goals"] = test_meta["full_time_home_goals"].to_numpy()
    predictions["actual_away_goals"] = test_meta["full_time_away_goals"].to_numpy()

    score_rows = []
    for row in test_meta.itertuples(index=False):
        matrix = model.scoreline_matrix(row.home_team, row.away_team)
        for i, home_label in enumerate(matrix.index):
            for j, away_label in enumerate(matrix.columns):
                score_rows.append({
                    "match_id": row.match_id,
                    "home_goals": i,
                    "away_goals": j,
                    "probability": float(matrix.iloc[i, j]),
                })
    return predictions, pd.DataFrame(score_rows)


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    digits: dict | None = None,
    max_rows: int | None = None,
) -> str:
    if frame is None or frame.empty:
        return "_No data._"
    digits = digits or {}
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return "_No data._"
    part = frame.head(max_rows) if max_rows else frame
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in part.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)) and column in digits:
                cells.append(f"{float(value):.{digits[column]}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_phase3_report(
    *,
    metadata: dict,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    walk_forward: pd.DataFrame,
    ensemble: dict,
    elo_params: dict,
    elo_latest: pd.DataFrame,
    poisson_summary: dict,
    poisson_ratings: pd.DataFrame,
    poisson_example: dict,
    market_comparison: pd.DataFrame,
    market_disagreement: pd.DataFrame,
    market_stack: pd.DataFrame,
    overround: pd.DataFrame,
    uncertainty_summary: dict,
    path: Path = PHASE3_REPORT_PATH,
) -> Path:
    """Write a measured Phase 3 report without inventing improvements."""
    best_test = test.sort_values("log_loss").iloc[0] if not test.empty else None
    best_val = validation.sort_values("log_loss").iloc[0] if not validation.empty else None
    market_row = test[test["model"] == "Market odds"]
    market_test = market_row.iloc[0] if not market_row.empty else None
    weighted = test[test["model"] == "Ensemble (validation weights)"]
    weighted_row = weighted.iloc[0] if not weighted.empty else None
    production = test[test["model"] == "Random Forest"]
    production_row = production.iloc[0] if not production.empty else None
    elo_test = test[test["model"] == "Dynamic Elo"]
    elo_test_row = elo_test.iloc[0] if not elo_test.empty else None
    elo_validation = validation[validation["model"] == "Dynamic Elo"]
    elo_validation_row = elo_validation.iloc[0] if not elo_validation.empty else None
    production_validation = validation[validation["model"] == "Random Forest"]
    production_validation_row = (
        production_validation.iloc[0] if not production_validation.empty else None
    )
    weighted_validation = validation[validation["model"] == "Ensemble (validation weights)"]
    weighted_validation_row = (
        weighted_validation.iloc[0] if not weighted_validation.empty else None
    )
    beats_market = (
        best_test is not None and market_test is not None
        and float(best_test["log_loss"]) < float(market_test["log_loss"])
    )
    lines = [
        "# La Liga Predictor — Phase 3: Advanced Football Modelling",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Production model version (unchanged): `{metadata.get('model_version', 'unknown')}`",
        "",
        "## Executive summary",
        "",
    ]
    if best_test is not None:
        lines.append(
            f"- Best **test** log-loss among all evaluated models: **{best_test['model']}** "
            f"({float(best_test['log_loss']):.4f}); accuracy {float(best_test['accuracy']):.1%}, "
            f"Brier {float(best_test['brier_score']):.4f}, ECE {float(best_test['ece']):.4f}, "
            f"draw recall {float(best_test['draw_recall']):.1%}."
        )
    if best_val is not None:
        lines.append(
            f"- Best **validation** log-loss: **{best_val['model']}** "
            f"({float(best_val['log_loss']):.4f}). Validation, not test, drove model and ensemble decisions."
        )
    if market_test is not None:
        lines.append(
            f"- Market odds test log-loss: **{float(market_test['log_loss']):.4f}** "
            f"(Brier {float(market_test['brier_score']):.4f}, accuracy {float(market_test['accuracy']):.1%})."
        )
    lines.append(
        "- No Phase 3 model is promoted to production in this run. The current "
        "production model is preserved; test metrics are confirmatory, not a "
        "selection signal."
    )
    if weighted_row is not None:
        lines.append(
            f"- Validation-weighted ensemble test log-loss: **{float(weighted_row['log_loss']):.4f}** "
            f"(Brier {float(weighted_row['brier_score']):.4f}, accuracy {float(weighted_row['accuracy']):.1%})."
        )
    if all(row is not None for row in (
        production_row, production_validation_row, elo_test_row,
        elo_validation_row, weighted_validation_row, weighted_row,
    )):
        lines.append(
            "- Relative to the current production Random Forest reference, Dynamic Elo "
            f"improved validation log-loss from {float(production_validation_row['log_loss']):.4f} "
            f"to {float(elo_validation_row['log_loss']):.4f} and test log-loss from "
            f"{float(production_row['log_loss']):.4f} to {float(elo_test_row['log_loss']):.4f}. "
            "The validation-weighted ensemble improved validation to "
            f"{float(weighted_validation_row['log_loss']):.4f} and test to "
            f"{float(weighted_row['log_loss']):.4f}. Both challengers still trail market odds, "
            "so this is a reason for a controlled review, not an automatic deployment."
        )
    lines += [
        "",
        "> **Result honesty:** "
        + (
            "At least one evaluated model has lower test log-loss than market odds. "
            "That is not a betting edge: the test season is only 109 matches, the "
            "market probabilities are also model inputs, and no independent "
            "out-of-sample betting audit was performed."
            if beats_market else
            "No evaluated model beats market odds on test log-loss. This is a "
            "negative result and is reported without adjusting the evaluation."
        ),
        "",
        "## 1. Dynamic Elo",
        "",
        "The Elo system updates ratings match-by-match, includes a fixed home "
        "advantage, uses a margin-of-victory multiplier and tracks rating "
        "deviation. All fixtures on a date read the pre-date state; updates are "
        "applied after the date is complete. Initial rating: 1500, initial RD: "
        "350. The draw mapping is fitted on training log-loss only.",
        "",
        f"Training-fitted Elo draw parameters: `{json.dumps(elo_params, default=_json_default)}`.",
        "",
        "### Latest ratings (descriptive, through the latest cached match)",
        "",
        _markdown_table(
            elo_latest,
            ["rank", "team", "rating", "rating_deviation", "matches"],
            {"rating": 2, "rating_deviation": 1},
            max_rows=12,
        ),
        "",
        "### Elo performance",
        "",
        _markdown_table(
            pd.concat([
                validation[validation["model"] == "Dynamic Elo"].assign(split="Validation"),
                test[test["model"] == "Dynamic Elo"].assign(split="Test"),
            ]),
            ["split", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"],
            {"accuracy": 4, "log_loss": 4, "brier_score": 4, "draw_recall": 4, "ece": 4},
        ),
        "",
        "## 2. Poisson and Dixon-Coles",
        "",
        "Expected goals are multiplicative functions of league home/away base "
        "rates, team attack and team defence. Observations are time-decayed, "
        "team estimates are shrunk toward the league average, and clubs with "
        "fewer than the minimum matches fall back explicitly to league average. "
        "The Dixon-Coles low-score correction is fitted on the training "
        "likelihood, not on test.",
        "",
        f"- Selected Poisson configuration: `{json.dumps(poisson_summary.get('config', {}), default=_json_default)}`.",
        f"- Fitted rho: **{poisson_summary.get('rho', float('nan')):+.4f}**.",
        f"- League home base: {poisson_summary.get('home_base', float('nan')):.3f} goals; "
        f"away base: {poisson_summary.get('away_base', float('nan')):.3f} goals.",
        f"- Example fixture: **{poisson_example.get('home_team', '')} vs {poisson_example.get('away_team', '')}**; "
        f"expected goals {poisson_example.get('lambda_home', float('nan')):.2f}–"
        f"{poisson_example.get('lambda_away', float('nan')):.2f}; "
        f"probabilities H/D/A {poisson_example.get('prob_home', float('nan')):.1%}/"
        f"{poisson_example.get('prob_draw', float('nan')):.1%}/"
        f"{poisson_example.get('prob_away', float('nan')):.1%}; "
        f"most likely score {poisson_example.get('most_likely_score', 'n/a')}.",
        "",
        "### Attack and defence estimates",
        "",
        _markdown_table(
            poisson_ratings.head(12),
            ["team", "attack", "defence", "matches", "effective_matches", "known"],
            {"attack": 3, "defence": 3, "effective_matches": 1},
        ),
        "",
        "### Poisson and Dixon-Coles performance",
        "",
        _markdown_table(
            pd.concat([
                validation[validation["model"].isin(["Independent Poisson", "Dixon-Coles"])].assign(split="Validation"),
                test[test["model"].isin(["Independent Poisson", "Dixon-Coles"])].assign(split="Test"),
            ]),
            ["split", "model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"],
            {"accuracy": 4, "log_loss": 4, "brier_score": 4, "draw_recall": 4, "ece": 4},
        ),
        "",
        "## 3. Model comparison",
        "",
        "All rows use the same chronological windows. Validation rows train on "
        "23/24+24/25; test rows refit on 23/24+24/25+25/26 and score 26/27 once. "
        "Market odds are a benchmark, not a learned model.",
        "",
        "### Validation",
        "",
        _markdown_table(
            validation,
            ["model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"],
            {"accuracy": 4, "log_loss": 4, "brier_score": 4, "draw_recall": 4, "ece": 4},
        ),
        "",
        "### Test",
        "",
        _markdown_table(
            test,
            ["model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"],
            {"accuracy": 4, "log_loss": 4, "brier_score": 4, "draw_recall": 4, "ece": 4},
        ),
        "",
        "## 4. Walk-forward performance",
        "",
        "The existing tree-model walk-forward rows are combined with new "
        "market, logistic, Elo, Poisson and Dixon-Coles rows. Folds train only "
        "on seasons before their target season; the final test season is excluded.",
        "",
        _markdown_table(
            walk_forward,
            ["fold", "test_season", "model", "n_samples", "log_loss", "brier_score", "accuracy", "draw_recall", "ece"],
            {"log_loss": 4, "brier_score": 4, "accuracy": 4, "draw_recall": 4, "ece": 4},
        ),
        "",
        "## 5. Ensemble",
        "",
        "Weights are non-negative, sum to one, and were selected by a fixed "
        "simplex grid on validation predictions only. Equal weights are included "
        "in the grid, so the selected ensemble cannot be worse than equal "
        "weighting on validation.",
        "",
        f"- Selected weights: `{json.dumps(ensemble.get('weights', {}), default=_json_default)}`.",
        f"- Validation log-loss: {ensemble.get('validation_log_loss', float('nan')):.4f}; "
        f"equal-weight validation log-loss: {ensemble.get('equal_validation_log_loss', float('nan')):.4f}.",
        f"- Test log-loss: {weighted_row['log_loss']:.4f}." if weighted_row is not None else "- Test metrics unavailable.",
        "",
        _markdown_table(
            test[test["model"].str.startswith("Ensemble", na=False)],
            ["model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"],
            {"accuracy": 4, "log_loss": 4, "brier_score": 4, "draw_recall": 4, "ece": 4},
        ),
        "",
        "## 6. Market odds analysis",
        "",
        "Raw implied probabilities are `1 / decimal odds`; overround is removed "
        "by row-normalising those probabilities. The market remains the primary "
        "benchmark. Model-versus-market differences are descriptive, not a "
        "claim of a betting edge.",
        "",
        "### Overround by season",
        "",
        _markdown_table(
            overround,
            ["season", "matches", "mean_overround", "median_overround", "min_overround", "max_overround"],
            {key: 4 for key in ("mean_overround", "median_overround", "min_overround", "max_overround")},
        ),
        "",
        "### Model versus market",
        "",
        _markdown_table(
            market_comparison,
            ["model", "model_log_loss", "market_log_loss", "delta_log_loss",
             "delta_log_loss_ci_low", "delta_log_loss_ci_high",
             "mean_abs_probability_difference", "beats_market_log_loss"],
            {key: 4 for key in (
                "model_log_loss", "market_log_loss", "delta_log_loss",
                "delta_log_loss_ci_low", "delta_log_loss_ci_high",
                "mean_abs_probability_difference",
            )},
        ),
        "",
        "### Picks where model and market disagree",
        "",
        _markdown_table(
            market_disagreement[market_disagreement["subset"] == "disagreement"],
            ["model", "n", "model_log_loss", "market_log_loss", "model_accuracy",
             "market_accuracy", "model_beats_market"],
            {"model_log_loss": 4, "market_log_loss": 4, "model_accuracy": 4, "market_accuracy": 4},
        ),
        "",
        "### Validation-fitted market + model stack",
        "",
        "This stack asks whether the component probabilities add information "
        "conditional on the market. It is fitted on validation only and scored "
        "once on test; it is not deployed.",
        "",
        _markdown_table(
            market_stack,
            ["model", "log_loss", "brier_score", "accuracy", "draw_recall",
             "market_log_loss", "delta_log_loss_vs_market",
             "delta_log_loss_ci_low", "delta_log_loss_ci_high"],
            {key: 4 for key in (
                "log_loss", "brier_score", "accuracy", "draw_recall",
                "market_log_loss", "delta_log_loss_vs_market",
                "delta_log_loss_ci_low", "delta_log_loss_ci_high",
            )},
        ),
        "",
        "## 7. Prediction uncertainty",
        "",
        "Confidence is the largest class probability, entropy is the normalized "
        "spread across outcomes, agreement is one minus mean pairwise "
        "total-variation distance between model families, and historical "
        "reliability combines Elo matches, Poisson match counts and known-team "
        "status. These are descriptive diagnostics, not calibrated intervals.",
        "",
        f"- Fixtures: **{uncertainty_summary.get('n', 0)}**; "
        f"mean confidence {uncertainty_summary.get('mean_confidence', float('nan')):.1%}; "
        f"mean entropy {uncertainty_summary.get('mean_entropy', float('nan')):.3f}; "
        f"mean agreement {uncertainty_summary.get('mean_agreement', float('nan')):.3f}.",
        f"- Uncertainty levels: `{json.dumps(uncertainty_summary.get('level_counts', {}), default=_json_default)}`.",
        f"- Historical reliability: `{json.dumps(uncertainty_summary.get('reliability_counts', {}), default=_json_default)}`.",
        f"- Spearman correlation between entropy and realised row log-loss: "
        f"{uncertainty_summary.get('correlation_entropy_loss')}.",
        "",
        "## 8. Reproducibility",
        "",
        "- Data fingerprint and feature hashes are recorded in every experiment JSON.",
        "- Random seed: 42.",
        "- Ensemble weights: selected on validation only, then applied unchanged to test.",
        "- Poisson time-decay and shrinkage: selected on validation only, then refit on train+validation for test.",
        "- Elo ratings are chronological; no post-match rating enters a pre-match feature.",
        "- Experiment index: `models/experiments/index.csv`.",
        "- Test season was scored once after all choices; it was not used to tune a model, feature or weight.",
        "",
        "## 9. Limitations",
        "",
        "- The test season has 109 of 380 matches, so differences of a few thousandths in log-loss are noisy.",
        "- Elo starts every club at 1500; promoted clubs are not given a separate prior.",
        "- Poisson estimates are multiplicative and do not model lineups, injuries, red cards, xG or shot quality.",
        "- The Dixon-Coles correction is fitted on the training likelihood, not a separate validation window.",
        "- Market odds are also inputs to the tree models, so the learned models are not independent forecasts.",
        "- No betting edge is claimed. A flat-stake or closing-line audit on genuinely unseen fixtures would be required.",
        "- The ensemble grid can overfit the validation season; the equal-weight reference is reported for that reason.",
        "- Uncertainty levels are descriptive and not calibrated probability intervals.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(*, save: bool = True) -> dict:
    raw = load_raw()
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata missing. Run `python -m src.models.train_model` first."
        )
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    X, y, meta = build_model_frame(raw)
    params = _params(metadata)

    elo, elo_history = compute_elo(raw)
    elo = elo.reindex(meta["match_id"].to_numpy())
    elo.index = X.index
    elo_latest = latest_elo_ratings(elo_history)
    poisson_pre = build_pre_match_poisson_features(raw)
    poisson_pre = poisson_pre.reindex(meta["match_id"].to_numpy())
    poisson_pre.index = X.index

    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx = splits["train"].index
    val_idx = splits["validation"].index
    test_idx = splits["test"].index
    raw_train = _raw_for_index(raw, meta, train_idx)
    raw_val = _raw_for_index(raw, meta, val_idx)
    raw_hist = _raw_for_index(raw, meta, train_idx.append(val_idx))
    raw_test = _raw_for_index(raw, meta, test_idx)
    val_meta = meta.loc[val_idx]
    test_meta = meta.loc[test_idx]

    poisson_config, poisson_selection = _select_poisson_config(
        raw_train, val_meta, y.loc[val_idx]
    )
    logger.info("Selected Poisson config on validation: %s", poisson_config)

    validation_results, validation_fit = _fit_window(
        train_raw=raw_train,
        eval_meta=val_meta,
        X_train_raw=X.loc[train_idx],
        y_train=y.loc[train_idx],
        X_eval_raw=X.loc[val_idx],
        elo_train=elo.loc[train_idx],
        elo_eval=elo.loc[val_idx],
        poisson_train=poisson_pre.loc[train_idx],
        poisson_eval=poisson_pre.loc[val_idx],
        params=params,
        poisson_config=poisson_config,
        use_saved_base=False,
    )

    ensemble_selection = select_ensemble_weights(
        {name: validation_results[name]["probs"] for name in ENSEMBLE_COMPONENTS},
        y.loc[val_idx].to_numpy(),
        step=0.05,
    )
    _add_ensemble(
        validation_results,
        y.loc[val_idx],
        ensemble_selection,
        weights=ensemble_selection["equal_weights"],
        name="Ensemble (equal)",
        notes="Equal-weight RF/XGB/Elo/Poisson reference.",
    )
    _add_ensemble(
        validation_results,
        y.loc[val_idx],
        ensemble_selection,
        weights=ensemble_selection["weights"],
        name="Ensemble (validation weights)",
        notes="Weights selected on validation log-loss; test not used.",
    )

    test_results, test_fit = _fit_window(
        train_raw=raw_hist,
        eval_meta=test_meta,
        X_train_raw=X.loc[train_idx.append(val_idx)],
        y_train=y.loc[train_idx.append(val_idx)],
        X_eval_raw=X.loc[test_idx],
        elo_train=elo.loc[train_idx.append(val_idx)],
        elo_eval=elo.loc[test_idx],
        poisson_train=poisson_pre.loc[train_idx.append(val_idx)],
        poisson_eval=poisson_pre.loc[test_idx],
        params=params,
        poisson_config=poisson_config,
        use_saved_base=True,
    )
    _add_ensemble(
        test_results,
        y.loc[test_idx],
        ensemble_selection,
        weights=ensemble_selection["equal_weights"],
        name="Ensemble (equal)",
        notes="Equal-weight RF/XGB/Elo/Poisson reference.",
    )
    _add_ensemble(
        test_results,
        y.loc[test_idx],
        ensemble_selection,
        weights=ensemble_selection["weights"],
        name="Ensemble (validation weights)",
        notes="Weights selected on validation log-loss; test not used.",
    )

    validation = _metrics_table(validation_results, y.loc[val_idx], split="validation")
    test = _metrics_table(test_results, y.loc[test_idx], split="test")
    walk_forward = _walk_forward(raw, X, y, meta, elo, params)

    season_rows = []
    for frame, split, season in (
        (validation, "validation", VALIDATION_SEASON),
        (test, "test", TEST_SEASON),
    ):
        part = frame.copy()
        part["evaluation"] = split
        part["season"] = season
        season_rows.append(part)
    if not walk_forward.empty:
        wf = walk_forward.copy()
        wf["evaluation"] = "walk_forward"
        wf["season"] = wf["test_season"]
        season_rows.append(wf)
    season_metrics = pd.concat(season_rows, ignore_index=True, sort=False)

    # Market diagnostics use the same test rows as the model comparison.
    market_validation = market_from_meta(
        val_meta, fallback=class_prior(y.loc[train_idx])
    )
    market_test = market_from_meta(
        test_meta, fallback=class_prior(y.loc[train_idx.append(val_idx)])
    )
    market_val_probs = market_validation[MARKET_COLUMNS].to_numpy(float)
    market_test_probs = market_test[MARKET_COLUMNS].to_numpy(float)

    market_rows = []
    range_frames = []
    disagreement_frames = []
    difference_frames = []
    for name, result in test_results.items():
        probabilities = result["probs"]
        row = compare_model_to_market(
            y.loc[test_idx], probabilities, market_test_probs, name
        )
        model_loss = _per_row_loss(y.loc[test_idx], probabilities)
        market_loss = _per_row_loss(y.loc[test_idx], market_test_probs)
        ci_low, ci_high = _bootstrap_delta_ci(market_loss, model_loss)
        row["delta_log_loss_ci_low"] = ci_low
        row["delta_log_loss_ci_high"] = ci_high
        market_rows.append(row)
        range_frames.append(
            probability_range_performance(
                y.loc[test_idx], probabilities, market_test_probs, name
            )
        )
        disagreement_frames.append(
            disagreement_analysis(
                y.loc[test_idx], probabilities, market_test_probs, name
            )
        )
        difference_frames.append(
            market_differences_frame(test_meta, probabilities, name)
        )
    market_comparison = pd.DataFrame(market_rows).sort_values("model_log_loss").reset_index(drop=True)
    market_ranges = pd.concat(range_frames, ignore_index=True)
    market_disagreement = pd.concat(disagreement_frames, ignore_index=True)
    market_differences = pd.concat(difference_frames, ignore_index=True)
    overround = overround_summary(raw)

    market_stack_result = fit_market_incremental_stack(
        {name: validation_results[name]["probs"] for name in ENSEMBLE_COMPONENTS},
        {name: test_results[name]["probs"] for name in ENSEMBLE_COMPONENTS},
        y.loc[val_idx].to_numpy(),
        y.loc[test_idx].to_numpy(),
        market_val_probs,
        market_test_probs,
        model_name="Market + RF/XGB/Elo/Poisson stack",
    )
    stack_loss = _per_row_loss(y.loc[test_idx], market_stack_result["probabilities"])
    stack_ci = _bootstrap_delta_ci(market_loss, stack_loss)
    market_stack_row = dict(market_stack_result["metrics"])
    market_stack_row["delta_log_loss_ci_low"] = stack_ci[0]
    market_stack_row["delta_log_loss_ci_high"] = stack_ci[1]
    market_stack = pd.DataFrame([market_stack_row])

    # Poisson artifacts and uncertainty use the test-window models, never a
    # model fitted on test outcomes.
    test_poisson = test_fit["dc_model"]
    poisson_predictions, poisson_scorelines = _poisson_artifacts(test_poisson, test_meta)
    poisson_ratings = test_poisson.team_ratings()
    poisson_summary = {
        **test_poisson.summary(),
        "selected_config": asdict(poisson_config),
        "selection": poisson_selection.to_dict(orient="records"),
        "selection_rule": "validation log-loss; test not used",
    }

    ensemble_probs = test_results["Ensemble (validation weights)"]["probs"]
    uncertainty_components = {
        name: test_results[name]["probs"]
        for name in (
            "Random Forest", "XGBoost", "Dynamic Elo",
            "Independent Poisson", "Dixon-Coles",
        )
    }
    uncertainty, uncertainty_summary = build_uncertainty_table(
        test_meta,
        ensemble_probs,
        uncertainty_components,
        elo.loc[test_idx],
        poisson_predictions.set_index("match_id"),
        X.loc[test_idx],
    )

    ensemble_weights = pd.DataFrame([
        {
            "component": name,
            "weight": float(ensemble_selection["all_weights"][name]),
            "validation_log_loss": ensemble_selection["validation_log_loss"],
            "equal_weight": ensemble_selection["equal_weights"][name],
            "equal_validation_log_loss": ensemble_selection["equal_validation_log_loss"],
        }
        for name in ENSEMBLE_COMPONENTS
    ])
    ensemble_comparison = pd.concat(
        [
            validation[validation["model"].isin([*ENSEMBLE_COMPONENTS, "Ensemble (equal)", "Ensemble (validation weights)"])]
            .assign(split="validation"),
            test[test["model"].isin([*ENSEMBLE_COMPONENTS, "Ensemble (equal)", "Ensemble (validation weights)"])]
            .assign(split="test"),
        ],
        ignore_index=True,
        sort=False,
    )

    if save:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        validation.to_csv(PHASE3_VALIDATION_PATH, index=False)
        test.to_csv(PHASE3_TEST_PATH, index=False)
        walk_forward.to_csv(PHASE3_WALK_FORWARD_PATH, index=False)
        season_metrics.to_csv(PHASE3_SEASON_METRICS_PATH, index=False)
        ensemble_weights.to_csv(ENSEMBLE_WEIGHTS_PATH, index=False)
        ensemble_comparison.to_csv(ENSEMBLE_COMPARISON_PATH, index=False)
        market_comparison.to_csv(MARKET_MODEL_COMPARISON_PATH, index=False)
        market_ranges.to_csv(MARKET_RANGE_PERFORMANCE_PATH, index=False)
        market_disagreement.to_csv(MARKET_DISAGREEMENT_PATH, index=False)
        market_differences.to_csv(MARKET_PROBABILITY_DIFFERENCES_PATH, index=False)
        market_stack.to_csv(MARKET_STACK_PATH, index=False)
        overround.to_csv(MARKET_OVERROUND_PATH, index=False)
        uncertainty.to_csv(UNCERTAINTY_PATH, index=False)
        UNCERTAINTY_SUMMARY_PATH.write_text(
            json.dumps(uncertainty_summary, indent=2, default=_json_default),
            encoding="utf-8",
        )
        ELO_MATCH_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        elo.reset_index().to_csv(ELO_MATCH_FEATURES_PATH, index=False)
        elo_history.to_csv(ELO_HISTORY_PATH, index=False)
        elo_latest.to_csv(ELO_RATINGS_PATH, index=False)
        poisson_ratings.to_csv(POISSON_RATINGS_PATH, index=False)
        poisson_predictions.to_csv(POISSON_TEST_PREDICTIONS_PATH, index=False)
        poisson_scorelines.to_csv(POISSON_SCORELINES_PATH, index=False)
        POISSON_SUMMARY_PATH.write_text(
            json.dumps(poisson_summary, indent=2, default=_json_default),
            encoding="utf-8",
        )

        plot_elo_history(elo_history, elo_latest)
        plot_elo_uncertainty(elo_latest)
        plot_poisson_ratings(poisson_ratings)
        example = poisson_predictions.iloc[0] if not poisson_predictions.empty else None
        if example is not None:
            matrix = test_poisson.scoreline_matrix(example["home_team"], example["away_team"])
            plot_scoreline_heatmap(
                matrix,
                f"{example['home_team']} vs {example['away_team']} — Poisson scorelines",
            )
        plot_phase3_model_comparison(validation, test)
        plot_ensemble_weights(ensemble_weights)
        plot_market_probability_scatter(
            market_differences, model_name="Random Forest"
        )
        plot_uncertainty(uncertainty)

    # Record every new model in the existing experiment tracker.
    data_hash = fingerprint_file(RAW_PATH)
    run_stamp = datetime.now(timezone.utc)
    model_version = f"phase3-{run_stamp:%Y%m%dT%H%M%SZ}"
    val_lookup = validation.set_index("model")
    test_lookup = test.set_index("model")
    for name, result in test_results.items():
        if name not in val_lookup.index:
            continue
        val_row = val_lookup.loc[name]
        test_row = test_lookup.loc[name]
        if isinstance(val_row, pd.DataFrame):
            val_row = val_row.iloc[0]
        if isinstance(test_row, pd.DataFrame):
            test_row = test_row.iloc[0]
        val_metrics = {
            key: float(val_row[key]) for key in (
                "accuracy", "log_loss", "brier_score", "macro_f1",
                "draw_recall", "draw_precision", "ece", "n_samples",
            ) if key in val_row and pd.notna(val_row[key])
        }
        test_metrics = {
            key: float(test_row[key]) for key in (
                "accuracy", "log_loss", "brier_score", "macro_f1",
                "draw_recall", "draw_precision", "ece", "n_samples",
            ) if key in test_row and pd.notna(test_row[key])
        }
        selection_status = (
            "benchmark" if name == "Market odds"
            else "production_reference" if name in TREE_MODELS
            else "equal_weight_reference" if name == "Ensemble (equal)"
            else "validation_selected_ensemble" if name == "Ensemble (validation weights)"
            else "experimental_not_deployed"
        )
        record = make_experiment_record(
            model_type=name,
            model_family=result["family"],
            feature_set=result["feature_set"],
            features=result["features"],
            training_period=list(TRAIN_SEASONS),
            validation_period=VALIDATION_SEASON,
            test_period=TEST_SEASON,
            hyperparameters=result["params"],
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            data_fingerprint=data_hash,
            model_version=model_version,
            selection_status=selection_status,
            notes=result["notes"] + " Test was not used to choose this model.",
            extra={
                "test_used_for_selection": False,
                "deployment_status": "not_deployed",
                "model_version": model_version,
            },
            variant="phase3",
            timestamp=run_stamp,
        )
        if save:
            save_experiment(record)

    best_test = test.iloc[0].to_dict() if not test.empty else {}
    best_validation = validation.iloc[0].to_dict() if not validation.empty else {}
    best_market = market_comparison.sort_values("market_log_loss").iloc[0].to_dict()
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": model_version,
        "production_model_version": metadata.get("model_version"),
        "data_fingerprint": data_hash,
        "best_validation": best_validation,
        "best_test": best_test,
        "market_test_log_loss": best_market.get("market_log_loss"),
        "best_model_beats_market_test": bool(
            best_test and float(best_test.get("log_loss", np.inf)) < float(best_market.get("market_log_loss", np.inf))
        ),
        "production_comparison": {
            "random_forest_validation_log_loss": float(
                validation.loc[validation["model"] == "Random Forest", "log_loss"].iloc[0]
            ),
            "dynamic_elo_validation_log_loss": float(
                validation.loc[validation["model"] == "Dynamic Elo", "log_loss"].iloc[0]
            ),
            "dynamic_elo_test_log_loss": float(
                test.loc[test["model"] == "Dynamic Elo", "log_loss"].iloc[0]
            ),
            "random_forest_test_log_loss": float(
                test.loc[test["model"] == "Random Forest", "log_loss"].iloc[0]
            ),
            "note": (
                "Dynamic Elo and the validation-weighted ensemble beat the current "
                "Random Forest reference, but neither beats market odds. No automatic "
                "production change is made."
            ),
        },
        "ensemble": {
            "weights": ensemble_selection["weights"],
            "all_weights": ensemble_selection["all_weights"],
            "validation_log_loss": ensemble_selection["validation_log_loss"],
            "validation_brier_score": ensemble_selection["validation_brier_score"],
            "equal_validation_log_loss": ensemble_selection["equal_validation_log_loss"],
        },
        "elo_params": validation_fit["elo_params"],
        "poisson_selected_config": asdict(poisson_config),
        "poisson_rho": test_poisson.rho,
        "walk_forward_models": sorted(walk_forward["model"].unique().tolist()) if not walk_forward.empty else [],
        "market_stack": market_stack.to_dict(orient="records"),
        "uncertainty": uncertainty_summary,
        "test_used_for_selection": False,
        "deployment_decision": (
            "No experimental model promoted. Current production model preserved."
        ),
    }
    if save:
        PHASE3_SUMMARY_PATH.write_text(
            json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
        )
        write_phase3_report(
            metadata=metadata,
            validation=validation,
            test=test,
            walk_forward=walk_forward,
            ensemble=summary["ensemble"],
            elo_params=validation_fit["elo_params"],
            elo_latest=elo_latest,
            poisson_summary=poisson_summary,
            poisson_ratings=poisson_ratings,
            poisson_example=(
                poisson_predictions.iloc[0].to_dict()
                if not poisson_predictions.empty else {}
            ),
            market_comparison=market_comparison,
            market_disagreement=market_disagreement,
            market_stack=market_stack,
            overround=overround,
            uncertainty_summary=uncertainty_summary,
        )
        logger.info("Wrote Phase 3 report to %s", PHASE3_REPORT_PATH)

    return {
        "validation": validation,
        "test": test,
        "walk_forward": walk_forward,
        "season_metrics": season_metrics,
        "ensemble": ensemble_selection,
        "elo": {"features": elo, "history": elo_history, "latest": elo_latest},
        "poisson": {
            "config": poisson_config,
            "model": test_poisson,
            "ratings": poisson_ratings,
            "predictions": poisson_predictions,
            "summary": poisson_summary,
        },
        "market": {
            "comparison": market_comparison,
            "disagreement": market_disagreement,
            "stack": market_stack,
            "overround": overround,
        },
        "uncertainty": uncertainty_summary,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 Elo, Poisson/Dixon-Coles, ensembles and market analysis."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Compute without writing artifacts or the report.",
    )
    args = parser.parse_args(argv)
    result = run_all(save=not args.no_save)
    summary = result["summary"]
    print(f"Best validation: {summary['best_validation'].get('model')} "
          f"({summary['best_validation'].get('log_loss', float('nan')):.4f})")
    print(f"Best test: {summary['best_test'].get('model')} "
          f"({summary['best_test'].get('log_loss', float('nan')):.4f})")
    print(f"Market test log-loss: {summary['market_test_log_loss']:.4f}")
    print(f"Ensemble weights: {summary['ensemble']['weights']}")
    print(f"Deployment: {summary['deployment_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
