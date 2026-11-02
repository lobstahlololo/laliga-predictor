"""Build rolling form features and scaled train/test matrices for La Liga.

Loads ``data/raw/laliga_matches.csv`` and, for every match, computes each
team's 5-match rolling averages (computed strictly from *previous* matches to
avoid leakage) for:

    * goals scored
    * goals conceded
    * total shots
    * shots on target

plus a 3-match short-term rolling block for goals scored/conceded to capture
sudden momentum shifts, the explicit 5-match goal differential per side
(``home_goal_diff_5m`` / ``away_goal_diff_5m``), and a binary
``short_rest_advantage`` flag (home rested >= 5 days while away is fatigued
at <= 3 days, typically after midweek competition). The Bet365 implied market
probabilities engineered in ``make_dataset.py`` are included as per-match
features. Distinct Home/Away feature blocks are created, plus a binary
``home_advantage`` column. The time-based split keeps the historical seasons
(23/24, 24/25, 25/26) for training and the current season (26/27) for
testing. Features are scaled
with a StandardScaler **fit on the training window only**, converted back to
plain DataFrames and written to ``data/processed/`` as standard CSVs:

    * ``X_train_scaled.csv`` / ``y_train.csv``
    * ``X_test_scaled.csv``  / ``y_test.csv``

Usage:
    python src/features/build_features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "laliga_matches.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SCALER_PATH = PROCESSED_DIR / "feature_scaler.joblib"

TRAIN_SEASONS = {"23/24", "24/25", "25/26"}
TEST_SEASONS = {"26/27"}

ROLLING_WINDOW = 5   # medium-term form
SHORT_WINDOW = 3     # short-term momentum (goals only)

# Rest-advantage rule. Shared constants with predict_upcoming.py so training
# and inference apply the exact same thresholds (single source of truth).
HOME_REST_MIN_DAYS = 5   # home team needs >= this many rest days
AWAY_REST_MAX_DAYS = 3   # away team considered fatigued at <= this many

# Market features carried over from make_dataset.py (1 / Bet365 decimal odds).
IMPLIED_PROB_COLS = ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """Load the master raw dataset."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Run `python src/data/make_dataset.py` first."
        )
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    logger.info("Loaded %d matches from %s", len(df), RAW_PATH)
    return df


def build_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format match history: one row per (match, team, side).

    Columns: ``match_id``, ``date``, ``team``, ``side``, ``goals_scored``,
    ``goals_conceded``, ``total_shots``, ``shots_on_target``. This is the
    single source of truth for rolling metrics, shared with the production
    inference script (``predict_upcoming.py``).
    """
    df = df.sort_values("date").reset_index(drop=True)
    home = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["home_team"],
            "side": "home",
            "goals_scored": df["full_time_home_goals"],
            "goals_conceded": df["full_time_away_goals"],
            "total_shots": df["home_shots"],
            "shots_on_target": df["home_shots_on_target"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "team": df["away_team"],
            "side": "away",
            "goals_scored": df["full_time_away_goals"],
            "goals_conceded": df["full_time_home_goals"],
            "total_shots": df["away_shots"],
            "shots_on_target": df["away_shots_on_target"],
        }
    )
    long = pd.concat([home, away], ignore_index=True).sort_values(
        ["team", "date"]
    )
    # Rest days: match date minus the same team's previous match date.
    long["rest_days"] = long.groupby("team")["date"].diff().dt.days
    return long


def compute_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Attach 5-match rolling form columns for home and away sides.

    For each match the rolling window uses only the team's *prior* matches
    (``shift(1)`` before rolling) so no future information leaks into the
    features. The two season frames are concatenated chronologically so form
    carries across season boundaries.
    """
    long = build_team_history(df)
    df = df.sort_values("date").reset_index(drop=True).copy()

    metric_cols = [
        "goals_scored",
        "goals_conceded",
        "total_shots",
        "shots_on_target",
    ]
    # Per-team rolling means over the previous ROLLING_WINDOW matches.
    for col in metric_cols:
        long[f"form_{col}"] = (
            long.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        )

    # Short-term momentum: 3-match rolling goals scored/conceded, capturing
    # sudden shifts from injuries, tactics or confidence.
    for col in ("goals_scored", "goals_conceded"):
        long[f"short_{col}"] = (
            long.groupby("team")[col]
            .transform(lambda s: s.shift(1).rolling(SHORT_WINDOW, min_periods=1).mean())
        )

    short_cols = ["goals_scored", "goals_conceded"]

    # Pivot back into distinct home/away feature blocks.
    prefixes = {"home": "home", "away": "away"}
    frames = []
    for side in ("home", "away"):
        block = long[long["side"] == side].copy()
        rename = {
            f"form_{col}": f"{prefixes[side]}_form_{col}" for col in metric_cols
        }
        rename.update({
            f"short_{col}": f"{prefixes[side]}_short_{col}" for col in short_cols
        })
        rename["rest_days"] = f"{prefixes[side]}_rest_days"
        block = block.rename(columns=rename)
        block = block[["match_id"] + list(rename.values())]
        frames.append(block)

    home_form, away_form = frames
    out = df.merge(home_form, on="match_id", how="left").merge(
        away_form, on="match_id", how="left"
    )
    return out


def short_rest_flag(home_rest_days, away_rest_days):
    """1 when the home side is well rested and the away side is fatigued.

    Works on pandas Series (training, vectorized) and plain numbers
    (inference, scalar). Uses the shared HOME_REST_MIN_DAYS /
    AWAY_REST_MAX_DAYS constants so both paths encode the same rule.
    """
    if isinstance(home_rest_days, pd.Series):
        return (
            (home_rest_days >= HOME_REST_MIN_DAYS)
            & (away_rest_days <= AWAY_REST_MAX_DAYS)
        ).astype(int)
    return int(
        home_rest_days >= HOME_REST_MIN_DAYS
        and away_rest_days <= AWAY_REST_MAX_DAYS
    )


def rest_days_as_of(history: pd.DataFrame, team: str, as_of) -> int:
    """Days of rest for ``team`` as of ``as_of`` (match/fixture date).

    Same formula as training's per-team date diff: as_of minus the team's
    most recent match date. ``history`` is the long frame from
    :func:`build_team_history`.
    """
    dates = history.loc[history["team"] == team, "date"]
    if dates.empty:
        raise ValueError(f"No historical matches found for '{team}'.")
    return int((pd.Timestamp(as_of) - dates.max()).days)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble X (numeric features) and y (match_outcome)."""
    form_cols = sorted(
        c for c in df.columns if "_form_" in c or "_short_" in c
    )
    df = df.copy()
    # Simple binary home-advantage flag (always 1 for the home side).
    df["home_advantage"] = 1

    # Derived: explicit 5-match goal differential per side (scored - conceded).
    df["home_goal_diff_5m"] = (
        df["home_form_goals_scored"] - df["home_form_goals_conceded"]
    )
    df["away_goal_diff_5m"] = (
        df["away_form_goals_scored"] - df["away_form_goals_conceded"]
    )

    # Derived: rest asymmetry (home >= 5 rest days, away <= 3).
    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        df["short_rest_advantage"] = short_rest_flag(
            df["home_rest_days"], df["away_rest_days"]
        )
    else:
        df["short_rest_advantage"] = 0
        logger.warning("Rest-day columns missing; short_rest_advantage=0.")

    # Market features (implied probabilities) are per-match, not rolling.
    market_cols = [c for c in IMPLIED_PROB_COLS if c in df.columns]
    missing = [c for c in IMPLIED_PROB_COLS if c not in df.columns]
    if missing:
        logger.warning("Implied probability columns missing from raw data: %s", missing)
    derived_cols = ["home_goal_diff_5m", "away_goal_diff_5m", "short_rest_advantage"]
    feature_cols = form_cols + market_cols + derived_cols + ["home_advantage"]

    X = df[feature_cols].astype(float)
    y = df["match_outcome"].astype(int)

    # Structural objects (team names, dates, keys, raw result codes) are
    # already excluded by selecting only `feature_cols`.
    before = len(X)
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    dropped = before - len(X)
    if dropped:
        logger.info(
            "Dropped %d matches with incomplete rolling history or missing odds "
            "(first %d matches per team).", dropped, ROLLING_WINDOW,
        )
    return X, y


def scale_and_save(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standard-scale features (fit on train only) and save everything.

    The scaled numpy arrays are converted *back* into standard Pandas
    DataFrames with the descriptive feature headers re-attached before being
    written as ordinary CSV files.

    The fitted scaler itself is persisted with joblib so the production
    inference script (``predict_upcoming.py``) can transform new fixtures
    with the exact same train-window statistics.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train.values)
    test_scaled = scaler.transform(X_test.values)

    # Convert scaled arrays back into DataFrames with real headers.
    X_train_scaled = pd.DataFrame(train_scaled, columns=list(X_train.columns))
    X_test_scaled = pd.DataFrame(test_scaled, columns=list(X_test.columns))

    y_train_frame = y_train.to_frame(name="match_outcome")
    y_test_frame = y_test.to_frame(name="match_outcome")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "X_train_scaled.csv": X_train_scaled,
        "y_train.csv": y_train_frame,
        "X_test_scaled.csv": X_test_scaled,
        "y_test.csv": y_test_frame,
    }
    for name, frame in outputs.items():
        frame.to_csv(PROCESSED_DIR / name, index=False)
        logger.info("Wrote %s (%s)", PROCESSED_DIR / name, frame.shape)

    joblib.dump(scaler, SCALER_PATH)
    logger.info("Wrote fitted scaler to %s", SCALER_PATH)

    return X_train_scaled, X_test_scaled


def main() -> int:
    """Entry point: build features, scale, and persist processed CSVs."""
    df = load_raw()
    df = compute_team_form(df)

    # --- Time-based split: history -> train, current season -> test ---
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Time-based split failed. Seasons present: {sorted(df['season'].unique())}. "
            f"Expected train={sorted(TRAIN_SEASONS)}, test={sorted(TEST_SEASONS)}."
        )
    logger.info(
        "Time-based split: train=%d matches (%s), test=%d matches (%s)",
        len(train_df), sorted(TRAIN_SEASONS),
        len(test_df), sorted(TEST_SEASONS),
    )

    X_train, y_train = build_feature_matrix(train_df)
    X_test, y_test = build_feature_matrix(test_df)

    # Align columns between splits (defensive; they should already match).
    shared = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[shared], X_test[shared]

    logger.info("Feature columns (%d): %s", len(shared), shared)

    scale_and_save(X_train, X_test, y_train, y_test)
    logger.info("Feature pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
