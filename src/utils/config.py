"""Central configuration for the La Liga predictor.

Every module imports paths, feature definitions and modelling constants from
here so that training, inference, evaluation and the dashboard cannot drift
apart. No credentials or machine-specific absolute paths live in this file.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_PATH = RAW_DIR / "laliga_matches.csv"
FIXTURES_PATH = PROJECT_ROOT / "fixtures.txt"
UPCOMING_FIXTURES_PATH = RAW_DIR / "upcoming_fixtures.csv"
DATA_QUALITY_PATH = PROCESSED_DIR / "data_quality.json"
FEATURE_REFERENCE_PATH = PROCESSED_DIR / "feature_reference.csv"

LEDGER_PATH = PROCESSED_DIR / "prediction_history.csv"
SCALER_PATH = PROCESSED_DIR / "feature_scaler.joblib"

MODEL_DIR = PROJECT_ROOT / "models"
SAVED_MODEL_DIR = MODEL_DIR / "saved_models"
EVAL_DIR = MODEL_DIR / "evaluation"
BEST_MODEL_PATH = SAVED_MODEL_DIR / "best_model.pkl"
MODEL_METADATA_PATH = SAVED_MODEL_DIR / "model_metadata.json"
CALIBRATOR_PATH = SAVED_MODEL_DIR / "calibrator.joblib"
CALIBRATOR_METADATA_PATH = SAVED_MODEL_DIR / "calibrator_metadata.json"
MODEL_COMPARISON_PATH = EVAL_DIR / "model_comparison.csv"
SEASON_METRICS_PATH = EVAL_DIR / "season_metrics.csv"
WALK_FORWARD_PATH = EVAL_DIR / "walk_forward_results.csv"
EVALUATION_REPORT_PATH = EVAL_DIR / "evaluation_summary.json"

# Phase 2 artifacts: baselines, explainability, calibration, experiments.
BASELINE_VALIDATION_PATH = EVAL_DIR / "baseline_validation.csv"
BASELINE_TEST_PATH = EVAL_DIR / "baseline_test.csv"
BASELINE_COMPARISON_PATH = EVAL_DIR / "baseline_comparison.png"
PERMUTATION_IMPORTANCE_PATH = EVAL_DIR / "permutation_importance.csv"
TREE_IMPORTANCE_PATH = EVAL_DIR / "tree_importance.csv"
FEATURE_IMPORTANCE_FOLDS_PATH = EVAL_DIR / "feature_importance_folds.csv"
FEATURE_STABILITY_PATH = EVAL_DIR / "feature_stability.csv"
SHAP_IMPORTANCE_PATH = EVAL_DIR / "shap_importance.csv"
BASELINE_VS_ML_PATH = EVAL_DIR / "baseline_vs_ml.csv"
FEATURE_EXPERIMENTS_PATH = EVAL_DIR / "feature_experiments.csv"
FEATURE_EXPERIMENT_FOLDS_PATH = EVAL_DIR / "feature_experiment_folds.csv"
FEATURE_RECOMMENDATION_PATH = EVAL_DIR / "feature_recommendation.json"
CALIBRATION_COMPARISON_PATH = EVAL_DIR / "calibration_comparison.csv"
CALIBRATION_SELECTION_PATH = EVAL_DIR / "calibration_selection.csv"
CALIBRATION_DEPLOYMENT_PATH = EVAL_DIR / "calibration_recommendation.json"
PHASE2_SUMMARY_PATH = EVAL_DIR / "phase2_summary.json"

# Phase 3 artifacts: Elo, Poisson/Dixon-Coles, market analysis and uncertainty.
ELO_HISTORY_PATH = EVAL_DIR / "elo_history.csv"
ELO_RATINGS_PATH = EVAL_DIR / "elo_ratings.csv"
ELO_MATCH_FEATURES_PATH = EVAL_DIR / "elo_match_features.csv"
POISSON_RATINGS_PATH = EVAL_DIR / "poisson_team_ratings.csv"
POISSON_TEST_PREDICTIONS_PATH = EVAL_DIR / "poisson_test_predictions.csv"
POISSON_SCORELINES_PATH = EVAL_DIR / "poisson_test_scorelines.csv"
POISSON_SUMMARY_PATH = EVAL_DIR / "poisson_summary.json"
PHASE3_SUMMARY_PATH = EVAL_DIR / "phase3_summary.json"
PHASE3_VALIDATION_PATH = EVAL_DIR / "phase3_validation.csv"
PHASE3_TEST_PATH = EVAL_DIR / "phase3_test.csv"
PHASE3_WALK_FORWARD_PATH = EVAL_DIR / "phase3_walk_forward.csv"
PHASE3_SEASON_METRICS_PATH = EVAL_DIR / "phase3_season_metrics.csv"
ENSEMBLE_WEIGHTS_PATH = EVAL_DIR / "ensemble_weights.csv"
ENSEMBLE_COMPARISON_PATH = EVAL_DIR / "ensemble_comparison.csv"
MARKET_OVERROUND_PATH = EVAL_DIR / "market_overround.csv"
MARKET_MODEL_COMPARISON_PATH = EVAL_DIR / "market_model_comparison.csv"
MARKET_PROBABILITY_DIFFERENCES_PATH = EVAL_DIR / "market_probability_differences.csv"
MARKET_RANGE_PERFORMANCE_PATH = EVAL_DIR / "market_range_performance.csv"
MARKET_DISAGREEMENT_PATH = EVAL_DIR / "market_disagreement.csv"
MARKET_STACK_PATH = EVAL_DIR / "market_incremental_stack.csv"
UNCERTAINTY_PATH = EVAL_DIR / "prediction_uncertainty.csv"
UNCERTAINTY_SUMMARY_PATH = EVAL_DIR / "uncertainty_summary.json"

EXPERIMENTS_DIR = MODEL_DIR / "experiments"
EXPERIMENTS_INDEX_PATH = EXPERIMENTS_DIR / "index.csv"

# Phase 4: model registry and operational artifacts.
REGISTRY_DIR = MODEL_DIR / "registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"
REGISTRY_VERSIONS_DIR = REGISTRY_DIR / "versions"

LOGS_DIR = PROJECT_ROOT / "logs"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.log"

REPORTS_DIR = PROJECT_ROOT / "reports"
PIPELINE_RUNS_PATH = REPORTS_DIR / "pipeline_runs.jsonl"
PHASE2_REPORT_PATH = REPORTS_DIR / "phase2_report.md"
PHASE3_REPORT_PATH = REPORTS_DIR / "phase3_report.md"
MONITORING_REPORT_PATH = REPORTS_DIR / "monitoring_report.json"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# ---------------------------------------------------------------------------
# Dataset / split configuration
# ---------------------------------------------------------------------------

# Football-Data.co.uk season codes, oldest -> newest. The last season is the
# untouched test season; the one before it is the validation season.
SEASONS = ["2324", "2425", "2526", "2627"]
SEASON_CODES = {s: f"{s[:2]}/{s[2:]}" for s in SEASONS}

# The split is seasonal, never random: earlier seasons train, the penultimate
# season validates, and the newest season is scored exactly once.
TEST_SEASON = "26/27"
VALIDATION_SEASON = "25/26"
TRAIN_SEASONS = ["23/24", "24/25"]

# Walk-forward backtesting: fit on all seasons before the target season and
# predict the target season. The first fold needs at least this many seasons.
MIN_TRAIN_SEASONS = 1

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 5
SHORT_WINDOW = 3

HOME_REST_MIN_DAYS = 5
AWAY_REST_MAX_DAYS = 3

IMPLIED_PROB_COLS = ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]

# The canonical 19-feature order. This exact order is used for training,
# scaling, inference and the dashboard. It matches the historical pipeline
# (rolling 5-match block, 3-match momentum block, market probabilities, goal
# differentials, rest advantage, home advantage).
FEATURE_COLUMNS = [
    "away_form_goals_conceded",
    "away_form_goals_scored",
    "away_form_shots_on_target",
    "away_form_total_shots",
    "away_short_goals_conceded",
    "away_short_goals_scored",
    "home_form_goals_conceded",
    "home_form_goals_scored",
    "home_form_shots_on_target",
    "home_form_total_shots",
    "home_short_goals_conceded",
    "home_short_goals_scored",
    "implied_prob_home",
    "implied_prob_draw",
    "implied_prob_away",
    "home_goal_diff_5m",
    "away_goal_diff_5m",
    "short_rest_advantage",
    "home_advantage",
]

# ---------------------------------------------------------------------------
# Target / display labels
# ---------------------------------------------------------------------------

# FTR codes: H=2, D=1, A=0. The numerical order (Away < Draw < Home) is
# arbitrary but must never change without retraining.
OUTCOME_AWAY = 0
OUTCOME_DRAW = 1
OUTCOME_HOME = 2

OUTCOME_LABELS = {2: "Home Win", 1: "Draw", 0: "Away Win"}
OUTCOME_SHORT = {2: "Home", 1: "Draw", 0: "Away"}
LABEL_ORDER = [2, 1, 0]

# ---------------------------------------------------------------------------
# Modelling / evaluation
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SEARCH_ITER = 24
CV_FOLDS = 3

# Phase 2 experiment settings. These are fixed before seeing test results.
PERMUTATION_REPEATS = 20
BASELINE_LOGISTIC_MAX_ITER = 3000
CALIBRATION_HOLDOUT_FRACTION = 0.4
MONITORING_MIN_SETTLED = 30
MONITORING_CONFIRM_SETTLED = 100

# Selection metric: validation multiclass log-loss. Accuracy alone is never
# used to pick a model. Brier score breaks ties.
PRIMARY_METRIC = "log_loss"
TIEBREAK_METRIC = "brier_score"

# Prediction history / value betting.
VALUE_BET_THRESHOLD = 0.05
FLAT_STAKE = 10.0

# Data quality thresholds.
MAX_MISSING_FEATURE_FRACTION = 0.05
STALE_DATA_DAYS = 14
