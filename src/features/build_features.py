"""Feature engineering for La Liga match outcomes.

For every match the pipeline computes each team's rolling statistics using
only matches played *before* kickoff (a ``shift(1)`` boundary), then combines
the home and away blocks into the canonical 19-feature vector defined in
``src.utils.config.FEATURE_COLUMNS``:

* 5-match rolling goals scored/conceded, total shots and shots on target
* 3-match momentum for goals scored/conceded
* Bet365 implied market probabilities
* 5-match goal differentials
* short-rest advantage flag
* home advantage

The module deliberately separates clean chronological data from modelling:
``build_model_frame`` returns features/target/metadata, while ``main`` writes
the legacy processed CSV artifacts used by the CLI. Scaling and median
imputation are delegated to :class:`src.features.preprocessing.FeaturePreprocessor`
and are always fitted on the training window only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocessing import FeaturePreprocessor  # noqa: E402
from src.utils.config import (  # noqa: E402
    AWAY_REST_MAX_DAYS,
    BASELINE_FEATURE_GROUP,
    FEATURE_COLUMNS,
    FEATURE_REFERENCE_PATH,
    HOME_REST_MIN_DAYS,
    IMPLIED_PROB_COLS,
    PROCESSED_DIR,
    RAW_PATH,
    ROLLING_WINDOW,
    SCALER_PATH,
    SHORT_WINDOW,
    TEST_SEASON,
    TRAIN_SEASONS,
    VALIDATION_SEASON,
)
from src.evaluation.splits import chronological_split  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the cleaned master dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.make_dataset` first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("Loaded %d matches from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Team history / rolling features
# ---------------------------------------------------------------------------

def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format history: one row per (match, team, side).

    Also computes rest days as the gap to the team's previous match. This is
    the single source of truth shared with inference.
    """
    required = {
        "match_id", "date", "home_team", "away_team",
        "full_time_home_goals", "full_time_away_goals",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw dataset missing columns for history: {sorted(missing)}")

    df = df.sort_values("date").reset_index(drop=True)
    home = pd.DataFrame({
        "match_id": df["match_id"],
        "date": df["date"],
        "team": df["home_team"],
        "side": "home",
        "goals_scored": df["full_time_home_goals"],
        "goals_conceded": df["full_time_away_goals"],
        "total_shots": df["home_shots"],
        "shots_on_target": df["home_shots_on_target"],
    })
    away = pd.DataFrame({
        "match_id": df["match_id"],
        "date": df["date"],
        "team": df["away_team"],
        "side": "away",
        "goals_scored": df["full_time_away_goals"],
        "goals_conceded": df["full_time_home_goals"],
        "total_shots": df["away_shots"],
        "shots_on_target": df["away_shots_on_target"],
    })
    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])
    long["rest_days"] = long.groupby("team")["date"].diff().dt.days
    return long


def compute_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Attach rolling form columns for both sides of every match.

    ``shift(1)`` guarantees the current match is excluded from its own
    features. Form carries across season boundaries because the long history
    is chronological, but future matches are never visible.
    """
    long = build_team_history(df)
    df = df.sort_values("date").reset_index(drop=True).copy()

    metric_cols = ["goals_scored", "goals_conceded", "total_shots", "shots_on_target"]
    for col in metric_cols:
        long[f"form_{col}"] = long.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
    for col in ("goals_scored", "goals_conceded"):
        long[f"short_{col}"] = long.groupby("team")[col].transform(
            lambda s: s.shift(1).rolling(SHORT_WINDOW, min_periods=1).mean()
        )

    frames = []
    for side in ("home", "away"):
        block = long[long["side"] == side].copy()
        rename = {f"form_{c}": f"{side}_form_{c}" for c in metric_cols}
        rename.update({
            f"short_{c}": f"{side}_short_{c}"
            for c in ("goals_scored", "goals_conceded")
        })
        rename["rest_days"] = f"{side}_rest_days"
        block = block.rename(columns=rename)
        frames.append(block[["match_id"] + list(rename.values())])

    out = df.merge(frames[0], on="match_id", how="left").merge(
        frames[1], on="match_id", how="left"
    )
    return out


def short_rest_flag(home_rest_days, away_rest_days):
    """1 when home is rested (>=5 days) and away is fatigued (<=3 days)."""
    if isinstance(home_rest_days, pd.Series):
        return (
            (home_rest_days >= HOME_REST_MIN_DAYS)
            & (away_rest_days <= AWAY_REST_MAX_DAYS)
        ).fillna(False).astype(int)
    if pd.isna(home_rest_days) or pd.isna(away_rest_days):
        return 0
    return int(
        home_rest_days >= HOME_REST_MIN_DAYS
        and away_rest_days <= AWAY_REST_MAX_DAYS
    )


def rest_days_as_of(history: pd.DataFrame, team: str, as_of) -> int:
    """Days since ``team``'s most recent match on or before ``as_of``.

    The cutoff prevents future matches from influencing rest days when a
    historical fixture is predicted or backtested.
    """
    cutoff = pd.Timestamp(as_of)
    dates = history.loc[
        (history["team"] == team) & (history["date"] <= cutoff), "date"
    ]
    if dates.empty:
        raise ValueError(f"No historical matches found for '{team}'.")
    return int((cutoff - dates.max()).days)


# ---------------------------------------------------------------------------
# Model frame
# ---------------------------------------------------------------------------

def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features while preserving metadata columns."""
    out = df.copy()
    out["home_advantage"] = 1.0
    out["home_goal_diff_5m"] = (
        out["home_form_goals_scored"] - out["home_form_goals_conceded"]
    )
    out["away_goal_diff_5m"] = (
        out["away_form_goals_scored"] - out["away_form_goals_conceded"]
    )
    if {"home_rest_days", "away_rest_days"} <= set(out.columns):
        out["short_rest_advantage"] = short_rest_flag(
            out["home_rest_days"], out["away_rest_days"]
        )
    else:
        out["short_rest_advantage"] = 0
        logger.warning("Rest-day columns missing; short_rest_advantage=0.")
    return out


def build_model_frame(
    df: pd.DataFrame,
    feature_groups: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return ``(X, y, metadata)`` aligned on the same rows.

    ``feature_groups`` is optional and defaults to the 19-feature baseline, so
    every existing caller and the production preprocessor contract are
    unchanged. Requesting ``xg`` or ``player_availability`` adds that group's
    columns; an unavailable source raises instead of inventing values. Missing
    values are kept for the preprocessor to impute with training medians, so
    no information from validation/test influences imputation.
    """
    groups = list(feature_groups or [BASELINE_FEATURE_GROUP])
    prepared = add_engineered_columns(compute_team_form(df))
    if groups != [BASELINE_FEATURE_GROUP]:
        from src.features.optional_features import (
            attach_optional_features,
            resolve_feature_columns,
        )

        columns, _statuses = resolve_feature_columns(groups)
        prepared = attach_optional_features(
            prepared, [g for g in groups if g != BASELINE_FEATURE_GROUP]
        )
    else:
        columns = list(FEATURE_COLUMNS)
    missing = [c for c in columns if c not in prepared.columns]
    if missing:
        raise ValueError(f"Engineered frame is missing features: {missing}")

    meta_cols = [
        c for c in (
            "match_id", "date", "season", "home_team", "away_team",
            "match_outcome", "full_time_home_goals", "full_time_away_goals",
            "full_time_result", "odds_home", "odds_draw", "odds_away",
            "implied_prob_home", "implied_prob_draw", "implied_prob_away",
            "home_form_goals_scored", "away_form_goals_scored",
            "home_form_goals_conceded", "away_form_goals_conceded",
        ) if c in prepared.columns
    ]
    meta = prepared[meta_cols].copy()
    y = prepared["match_outcome"].astype(int)
    X = prepared[columns].apply(pd.to_numeric, errors="coerce")

    valid_target = y.notna()
    X, y, meta = X[valid_target], y[valid_target], meta[valid_target]
    if len(X) == 0:
        raise ValueError("No rows with a valid match outcome.")
    logger.info("Built model frame: %d rows x %d features", *X.shape)
    return X, y, meta


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Backward-compatible ``(X, y)`` wrapper used by older callers/tests."""
    X, y, _ = build_model_frame(df)
    return X, y


# ---------------------------------------------------------------------------
# Scaling / artifact persistence
# ---------------------------------------------------------------------------

def save_feature_reference(X_train: pd.DataFrame, path: Path = FEATURE_REFERENCE_PATH) -> None:
    """Persist training feature stats for dashboard drift monitoring."""
    reference = pd.DataFrame({
        "feature": X_train.columns,
        "mean": X_train.mean().values,
        "std": X_train.std(ddof=0).values,
        "p05": X_train.quantile(0.05).values,
        "p50": X_train.quantile(0.50).values,
        "p95": X_train.quantile(0.95).values,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(path, index=False)


def main() -> int:
    """Build features and persist train/validation/test artifacts.

    Only the training window is used to fit imputation and scaling. The
    validation and test CSVs are transformed with those same training
    statistics; the untouched test season is never inspected during fitting.
    """
    df = load_raw()
    X, y, meta = build_model_frame(df)

    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx = splits["train"].index
    val_idx = splits["validation"].index
    test_idx = splits["test"].index
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Season split is empty. Present seasons: "
            f"{sorted(meta['season'].dropna().unique())}."
        )

    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS)
    X_train_scaled = preprocessor.fit_transform(X.loc[train_idx])
    X_val_scaled = preprocessor.transform(X.loc[val_idx])
    X_test_scaled = preprocessor.transform(X.loc[test_idx])

    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]
    y_test = y.loc[test_idx]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "X_train_scaled.csv": X_train_scaled,
        "y_train.csv": y_train.to_frame("match_outcome"),
        "X_validation_scaled.csv": X_val_scaled,
        "y_validation.csv": y_val.to_frame("match_outcome"),
        "X_test_scaled.csv": X_test_scaled,
        "y_test.csv": y_test.to_frame("match_outcome"),
        "model_metadata_train.csv": meta.loc[train_idx].reset_index(drop=True),
        "model_metadata_validation.csv": meta.loc[val_idx].reset_index(drop=True),
        "model_metadata_test.csv": meta.loc[test_idx].reset_index(drop=True),
    }
    for name, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / name, index=False)
        logger.info("Wrote %s (%s)", PROCESSED_DIR / name, frame.shape)

    preprocessor.save()
    # Legacy artifact kept so older scripts can still find a scaler.
    import joblib
    joblib.dump(preprocessor.scaler, SCALER_PATH)
    save_feature_reference(X.loc[train_idx])

    logger.info(
        "Split: train=%d %s | validation=%d %s | test=%d %s",
        len(train_idx), splits["train_seasons"],
        len(val_idx), VALIDATION_SEASON,
        len(test_idx), TEST_SEASON,
    )
    logger.info("Feature pipeline complete (%d features).", len(FEATURE_COLUMNS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
