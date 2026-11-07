"""Cached data access for the Streamlit dashboard.

All pages read through these helpers so model metrics, history normalization
and feature calculations stay consistent. Cache keys include file mtimes, so
regenerating models/data refreshes the dashboard without restarting it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_probabilities  # noqa: E402
from src.features.build_features import (  # noqa: E402
    build_model_frame,
    build_team_history,
    load_raw as load_raw_file,
)
from src.predictions import history as hist  # noqa: E402
from src.predictions.fixtures import load_upcoming_fixtures  # noqa: E402
from src.predictions.predictor import (  # noqa: E402
    load_artifacts,
    load_raw as load_raw_file_predictor,
    predict_fixtures,
)
from src.utils.config import (  # noqa: E402
    BASELINE_TEST_PATH,
    BASELINE_VALIDATION_PATH,
    BASELINE_VS_ML_PATH,
    CALIBRATION_COMPARISON_PATH,
    CALIBRATION_DEPLOYMENT_PATH,
    CALIBRATION_SELECTION_PATH,
    DATA_QUALITY_PATH,
    ELO_HISTORY_PATH,
    ELO_RATINGS_PATH,
    ENSEMBLE_COMPARISON_PATH,
    ENSEMBLE_WEIGHTS_PATH,
    EXPERIMENTS_INDEX_PATH,
    FEATURE_EXPERIMENT_FOLDS_PATH,
    FEATURE_EXPERIMENTS_PATH,
    FEATURE_IMPORTANCE_FOLDS_PATH,
    FEATURE_RECOMMENDATION_PATH,
    FEATURE_REFERENCE_PATH,
    FEATURE_STABILITY_PATH,
    LEDGER_PATH,
    MARKET_DISAGREEMENT_PATH,
    MARKET_MODEL_COMPARISON_PATH,
    MARKET_OVERROUND_PATH,
    MARKET_PROBABILITY_DIFFERENCES_PATH,
    MARKET_STACK_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MONITORING_REPORT_PATH,
    PIPELINE_RUNS_PATH,
    REGISTRY_PATH,
    PERMUTATION_IMPORTANCE_PATH,
    PHASE2_SUMMARY_PATH,
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
    SEASON_METRICS_PATH,
    SHAP_IMPORTANCE_PATH,
    STALE_DATA_DAYS,
    TEST_SEASON,
    TREE_IMPORTANCE_PATH,
    UNCERTAINTY_PATH,
    UNCERTAINTY_SUMMARY_PATH,
    VALIDATION_SEASON,
    WALK_FORWARD_PATH,
    FEATURE_COLUMNS,
)
from src.data.validation import validate_matches  # noqa: E402
from src.monitoring.diagnostics import run_diagnostics  # noqa: E402


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner="Loading match data…")
def load_raw(mtime: float = 0.0) -> pd.DataFrame:
    return load_raw_file(RAW_PATH)


@st.cache_data(show_spinner="Loading prediction ledger…")
def load_history(mtime: float = 0.0) -> pd.DataFrame:
    history, _changed = hist.load_history(migrate=True)
    return history


@st.cache_data(show_spinner=False)
def load_metadata(mtime: float = 0.0) -> dict:
    if not MODEL_METADATA_PATH.exists():
        return {}
    try:
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@st.cache_data(show_spinner=False)
def load_comparison(mtime: float = 0.0) -> pd.DataFrame:
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(MODEL_COMPARISON_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_season_metrics(mtime: float = 0.0) -> pd.DataFrame:
    if not SEASON_METRICS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SEASON_METRICS_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_walk_forward(mtime: float = 0.0) -> pd.DataFrame:
    if not WALK_FORWARD_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(WALK_FORWARD_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_quality_report(mtime: float = 0.0) -> dict:
    if not DATA_QUALITY_PATH.exists():
        return {}
    try:
        return json.loads(DATA_QUALITY_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@st.cache_data(show_spinner=False)
def load_feature_reference(mtime: float = 0.0) -> pd.DataFrame:
    if not FEATURE_REFERENCE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(FEATURE_REFERENCE_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_baseline_validation(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(BASELINE_VALIDATION_PATH)


@st.cache_data(show_spinner=False)
def load_baseline_test(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(BASELINE_TEST_PATH)


@st.cache_data(show_spinner=False)
def load_baseline_vs_ml(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(BASELINE_VS_ML_PATH)


@st.cache_data(show_spinner=False)
def load_feature_experiments(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(FEATURE_EXPERIMENTS_PATH)


@st.cache_data(show_spinner=False)
def load_feature_experiment_folds(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(FEATURE_EXPERIMENT_FOLDS_PATH)


@st.cache_data(show_spinner=False)
def load_calibration_comparison(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(CALIBRATION_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_calibration_selection(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(CALIBRATION_SELECTION_PATH)


@st.cache_data(show_spinner=False)
def load_permutation_importance(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(PERMUTATION_IMPORTANCE_PATH)


@st.cache_data(show_spinner=False)
def load_tree_importance(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(TREE_IMPORTANCE_PATH)


@st.cache_data(show_spinner=False)
def load_shap_importance(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(SHAP_IMPORTANCE_PATH)


@st.cache_data(show_spinner=False)
def load_feature_stability(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(FEATURE_STABILITY_PATH)


@st.cache_data(show_spinner=False)
def load_experiments(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(EXPERIMENTS_INDEX_PATH)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@st.cache_data(show_spinner=False)
def load_feature_recommendation(mtime: float = 0.0) -> dict:
    return _read_json(FEATURE_RECOMMENDATION_PATH)


@st.cache_data(show_spinner=False)
def load_calibration_recommendation(mtime: float = 0.0) -> dict:
    return _read_json(CALIBRATION_DEPLOYMENT_PATH)


@st.cache_data(show_spinner=False)
def load_phase2_summary(mtime: float = 0.0) -> dict:
    return _read_json(PHASE2_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_phase3_summary(mtime: float = 0.0) -> dict:
    return _read_json(PHASE3_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_phase3_validation(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(PHASE3_VALIDATION_PATH)


@st.cache_data(show_spinner=False)
def load_phase3_test(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(PHASE3_TEST_PATH)


@st.cache_data(show_spinner=False)
def load_phase3_walk_forward(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(PHASE3_WALK_FORWARD_PATH)


@st.cache_data(show_spinner=False)
def load_phase3_season_metrics(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(PHASE3_SEASON_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_elo_history(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(ELO_HISTORY_PATH)


@st.cache_data(show_spinner=False)
def load_elo_ratings(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(ELO_RATINGS_PATH)


@st.cache_data(show_spinner=False)
def load_poisson_ratings(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(POISSON_RATINGS_PATH)


@st.cache_data(show_spinner=False)
def load_poisson_predictions(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(POISSON_TEST_PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_poisson_scorelines(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(POISSON_SCORELINES_PATH)


@st.cache_data(show_spinner=False)
def load_poisson_summary(mtime: float = 0.0) -> dict:
    return _read_json(POISSON_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_ensemble_weights(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(ENSEMBLE_WEIGHTS_PATH)


@st.cache_data(show_spinner=False)
def load_ensemble_comparison(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(ENSEMBLE_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_market_comparison(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(MARKET_MODEL_COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_market_differences(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(MARKET_PROBABILITY_DIFFERENCES_PATH)


@st.cache_data(show_spinner=False)
def load_market_disagreement(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(MARKET_DISAGREEMENT_PATH)


@st.cache_data(show_spinner=False)
def load_market_overround(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(MARKET_OVERROUND_PATH)


@st.cache_data(show_spinner=False)
def load_market_stack(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(MARKET_STACK_PATH)


@st.cache_data(show_spinner=False)
def load_uncertainty(mtime: float = 0.0) -> pd.DataFrame:
    return _read_csv(UNCERTAINTY_PATH)


@st.cache_data(show_spinner=False)
def load_uncertainty_summary(mtime: float = 0.0) -> dict:
    return _read_json(UNCERTAINTY_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_monitoring_report(mtime: float = 0.0) -> dict:
    return _read_json(MONITORING_REPORT_PATH)


@st.cache_data(show_spinner=False)
def monitoring_diagnostics(raw_mtime: float, ledger_mtime: float, model_mtime: float) -> dict:
    raw = load_raw(raw_mtime)
    history = load_history(ledger_mtime)
    metadata = load_metadata(model_mtime)
    try:
        X, _y, _meta = build_model_frame(raw)
    except Exception:  # noqa: BLE001
        X = None
    return run_diagnostics(raw, history, metadata, feature_frame=X)


@st.cache_data(show_spinner="Scoring upcoming fixtures…")
def get_upcoming_predictions(raw_mtime: float, model_mtime: float):
    fixtures, source = load_upcoming_fixtures()
    if not fixtures:
        return [], [], source
    try:
        model, preprocessor, metadata = load_artifacts()
        raw = load_raw_file_predictor()
        predictions, errors = predict_fixtures(
            fixtures, model=model, preprocessor=preprocessor,
            metadata=metadata, raw=raw,
        )
        return predictions, errors, source
    except Exception as exc:  # noqa: BLE001 - page shows actionable error
        return [], [{"fixture": "all", "error": str(exc)}], source


def available_teams(raw: pd.DataFrame) -> list[str]:
    return sorted(set(raw["home_team"].astype(str)) | set(raw["away_team"].astype(str)))


def team_recent(history: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    rows = history[history["team"] == team].sort_values("date").tail(n).copy()
    rows["result"] = np.select(
        [rows["goals_scored"] > rows["goals_conceded"],
         rows["goals_scored"] == rows["goals_conceded"]],
        ["W", "D"], default="L",
    )
    rows["points"] = rows["result"].map({"W": 3, "D": 1, "L": 0})
    return rows.sort_values("date", ascending=False)


def form_string(history: pd.DataFrame, team: str, n: int = 5) -> str:
    recent = team_recent(history, team, n)
    return " ".join(recent.sort_values("date")["result"].tolist()) or "n/a"


def team_summary(raw: pd.DataFrame, history: pd.DataFrame, team: str) -> dict:
    rows = history[history["team"] == team].copy()
    if rows.empty:
        return {}
    rows["points"] = np.select(
        [rows["goals_scored"] > rows["goals_conceded"],
         rows["goals_scored"] == rows["goals_conceded"]],
        [3, 1], default=0,
    )
    rows["result"] = np.select(
        [rows["goals_scored"] > rows["goals_conceded"],
         rows["goals_scored"] == rows["goals_conceded"]],
        ["W", "D"], default="L",
    )
    home = rows[rows["side"] == "home"]
    away = rows[rows["side"] == "away"]
    return {
        "matches": len(rows),
        "wins": int((rows["result"] == "W").sum()),
        "draws": int((rows["result"] == "D").sum()),
        "losses": int((rows["result"] == "L").sum()),
        "goals_for": int(rows["goals_scored"].sum()),
        "goals_against": int(rows["goals_conceded"].sum()),
        "gf_per_match": float(rows["goals_scored"].mean()),
        "ga_per_match": float(rows["goals_conceded"].mean()),
        "ppg": float(rows["points"].mean()),
        "home_ppg": float(home["points"].mean()) if not home.empty else float("nan"),
        "away_ppg": float(away["points"].mean()) if not away.empty else float("nan"),
        "home_matches": len(home),
        "away_matches": len(away),
        "form": " ".join(rows.sort_values("date").tail(5)["result"].tolist()),
    }


def team_rolling(history: pd.DataFrame, team: str, window: int = 5) -> pd.DataFrame:
    rows = history[history["team"] == team].sort_values("date").copy()
    rows["points"] = np.select(
        [rows["goals_scored"] > rows["goals_conceded"],
         rows["goals_scored"] == rows["goals_conceded"]],
        [3, 1], default=0,
    )
    rows["rolling_points"] = rows["points"].rolling(window, min_periods=1).mean()
    rows["rolling_gf"] = rows["goals_scored"].rolling(window, min_periods=1).mean()
    rows["rolling_ga"] = rows["goals_conceded"].rolling(window, min_periods=1).mean()
    return rows


def team_season_table(history: pd.DataFrame, team: str) -> pd.DataFrame:
    rows = history[history["team"] == team].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["season"] = pd.to_datetime(rows["date"]).dt.year.where(
        pd.to_datetime(rows["date"]).dt.month >= 7,
        pd.to_datetime(rows["date"]).dt.year - 1,
    )
    rows["season_label"] = rows["season"].map(lambda y: f"{y % 100:02d}/{(y + 1) % 100:02d}")
    rows["points"] = np.select(
        [rows["goals_scored"] > rows["goals_conceded"],
         rows["goals_scored"] == rows["goals_conceded"]],
        [3, 1], default=0,
    )
    grouped = rows.groupby("season_label").agg(
        matches=("match_id", "count"),
        goals_for=("goals_scored", "sum"),
        goals_against=("goals_conceded", "sum"),
        points=("points", "sum"),
    ).reset_index()
    grouped["ppg"] = grouped["points"] / grouped["matches"]
    return grouped.sort_values("season_label")


def history_derived(mtime: float = 0.0) -> pd.DataFrame:
    return hist.add_derived_columns(load_history(mtime))


def history_summary(mtime: float = 0.0) -> dict:
    return hist.performance_summary(load_history(mtime))


def model_rows(metadata: dict) -> pd.DataFrame:
    """Validation/test rows for every trained candidate."""
    rows = []
    for name, info in metadata.get("all_models", {}).items():
        for split in ("validation_metrics", "test_metrics"):
            metrics = info.get(split, {})
            if not metrics:
                continue
            rows.append({
                "model": name,
                "split": "Validation" if split.startswith("validation") else "Test",
                **{k: metrics.get(k) for k in (
                    "accuracy", "log_loss", "brier_score", "macro_f1",
                    "draw_recall", "draw_precision", "ece", "n_samples",
                )},
            })
    return pd.DataFrame(rows)


def calibration_by_period(history: pd.DataFrame) -> pd.DataFrame:
    """ECE of settled ledger predictions grouped by month."""
    df = hist.add_derived_columns(history)
    settled = df[df["actual_label"].astype(str) != ""].copy()
    if settled.empty:
        return pd.DataFrame()
    settled["period"] = settled["predicted_at_ts"].dt.to_period("M").astype(str)
    rows = []
    for period, part in settled.groupby("period"):
        probs = part[hist.PROB_COLUMNS].to_numpy(float)
        y = part["actual_outcome"].astype(int).to_numpy()
        if not np.isfinite(probs).all():
            continue
        metrics = evaluate_probabilities(y, probs, [2, 1, 0], period)
        rows.append({
            "period": period,
            "n": metrics["n_samples"],
            "ece": metrics["ece"],
            "log_loss": metrics["log_loss"],
            "accuracy": metrics["accuracy"],
        })
    return pd.DataFrame(rows)


def quality_warnings(raw: pd.DataFrame) -> list[tuple[str, str]]:
    """Live data-quality and staleness warnings."""
    warnings: list[tuple[str, str]] = []
    if raw.empty:
        return [("error", "Raw dataset is empty. Run the data pipeline.")]

    report = validate_matches(raw)
    if report["duplicate_fixture_ids"]:
        warnings.append(
            ("error", f"{report['duplicate_fixture_ids']} duplicate fixture IDs.")
        )
    if report["invalid_dates"]:
        warnings.append(("error", f"{report['invalid_dates']} invalid match dates."))
    if report["inconsistent_results"]:
        warnings.append(
            ("error", f"{report['inconsistent_results']} score/result inconsistencies.")
        )
    if report["missing_odds"]:
        warnings.append(
            ("warning", f"{report['missing_odds']} matches have missing odds.")
        )

    max_date = pd.to_datetime(raw["date"], errors="coerce").max()
    if pd.notna(max_date):
        age = (pd.Timestamp.today().normalize() - max_date.normalize()).days
        if age > STALE_DATA_DAYS:
            warnings.append(
                ("warning",
                 f"Latest match is {age} days old (threshold {STALE_DATA_DAYS}). "
                 "Run `python -m src.data.make_dataset` to refresh.")
            )
        else:
            warnings.append(("ok", f"Latest match is {age} day(s) old."))

    current_test = raw[raw.get("season") == TEST_SEASON] if "season" in raw.columns else raw.iloc[0:0]
    if len(current_test) < 380:
        warnings.append(
            ("info",
             f"Test season {TEST_SEASON} has {len(current_test)} of 380 matches; "
             "test metrics are provisional.")
        )
    return warnings


def drift_table(raw: pd.DataFrame, recent_n: int = 60) -> pd.DataFrame:
    """Standardized mean shift of recent raw features vs training reference."""
    reference = load_feature_reference(_mtime(FEATURE_REFERENCE_PATH))
    if reference.empty or raw.empty:
        return pd.DataFrame()
    try:
        from src.features.preprocessing import FeaturePreprocessor

        X, _y, meta = build_model_frame(raw)
        recent_idx = meta.sort_values("date").tail(recent_n).index
        # feature_reference.csv stores moments of the training-standardized
        # matrix, so the recent window must be scaled with the same fitted
        # preprocessor before comparing means.
        recent = FeaturePreprocessor.load().transform(X.loc[recent_idx])
    except Exception:  # noqa: BLE001
        return pd.DataFrame()

    rows = []
    for _, ref in reference.iterrows():
        feature = ref["feature"]
        if feature not in recent.columns:
            continue
        std = float(ref["std"]) if float(ref["std"]) > 0 else 1.0
        recent_mean = float(recent[feature].mean())
        shift = (recent_mean - float(ref["mean"])) / std
        rows.append({
            "feature": feature,
            "train_mean": float(ref["mean"]),
            "recent_mean": recent_mean,
            "shift_sd": shift,
            "abs_shift": abs(shift),
            "status": "Alert" if abs(shift) > 2 else ("Watch" if abs(shift) > 1 else "OK"),
        })
    return pd.DataFrame(rows).sort_values("abs_shift", ascending=False).reset_index(drop=True)


def artifact_status() -> dict:
    """Presence/mtime of core artifacts, for empty-state guidance."""
    return {
        "raw": RAW_PATH.exists(),
        "ledger": LEDGER_PATH.exists(),
        "model": (ROOT / "models" / "saved_models" / "best_model.pkl").exists(),
        "metadata": MODEL_METADATA_PATH.exists(),
        "evaluation": MODEL_COMPARISON_PATH.exists(),
    }


def season_metric_options() -> list[str]:
    return [TEST_SEASON, VALIDATION_SEASON]


@st.cache_data(show_spinner=False)
def load_registry_state(mtime: float = 0.0) -> dict:
    """Model registry status, tolerant of a missing registry file."""
    try:
        from src.models.registry import status

        return status()
    except Exception as exc:  # noqa: BLE001 - page shows an empty state
        return {"error": str(exc), "production": None, "n_versions": 0}


@st.cache_data(show_spinner=False)
def load_registry_versions(mtime: float = 0.0) -> pd.DataFrame:
    try:
        from src.models.registry import list_models

        return list_models()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_pipeline_runs(limit: int = 20, mtime: float = 0.0) -> list[dict]:
    from src.pipeline.automation import load_runs

    return load_runs(limit=limit)
