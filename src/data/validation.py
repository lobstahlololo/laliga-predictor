"""Validation and cleaning for the raw La Liga match dataset.

The functions here are intentionally pure: they take a DataFrame and return a
cleaned DataFrame plus a JSON-serialisable quality report. Ingestion, the
dashboard's monitoring page and the tests all use the same rules, so a
"valid" dataset means the same thing everywhere.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import OUTCOME_LABELS
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "full_time_home_goals",
    "full_time_away_goals",
    "full_time_result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "odds_home",
    "odds_draw",
    "odds_away",
]

NUMERIC_COLUMNS = [
    "full_time_home_goals",
    "full_time_away_goals",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "odds_home",
    "odds_draw",
    "odds_away",
]

FTR_TO_OUTCOME = {"H": 2, "D": 1, "A": 0}


def season_from_date(date: pd.Timestamp) -> str:
    """Derive the European season label (e.g. ``24/25``) from a match date."""
    year = date.year if date.month >= 7 else date.year - 1
    return f"{year % 100:02d}/{(year + 1) % 100:02d}"


def parse_match_dates(values) -> pd.Series:
    """Parse ISO dates and Football-Data ``dd/mm/yyyy`` dates unambiguously.

    A single ``dayfirst`` flag cannot handle both formats: it misreads ISO
    dates. ISO-looking strings are parsed without ``dayfirst``; everything else
    is treated as day-first (the upstream Football-Data convention).
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    text = series.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    if iso.any():
        result.loc[iso] = pd.to_datetime(text.loc[iso], errors="coerce")
    other = ~iso
    if other.any():
        result.loc[other] = pd.to_datetime(
            text.loc[other], errors="coerce", dayfirst=True
        )
    return result


def build_fixture_ids(df: pd.DataFrame) -> pd.Series:
    """Stable fixture IDs: ``YYYYMMDD_Home_v_Away``.

    Uses canonical team names, so the same fixture always produces the same
    ID regardless of how the user typed the club name.
    """
    dates = parse_match_dates(df["date"])
    home = df["home_team"].map(canonical_team)
    away = df["away_team"].map(canonical_team)
    return (
        dates.dt.strftime("%Y%m%d")
        + "_"
        + home.astype(str)
        + "_v_"
        + away.astype(str)
    )


def validate_matches(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a data-quality report without modifying the frame."""
    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "columns_present": [],
        "missing_columns": [],
        "missing_values": {},
        "duplicate_fixture_ids": 0,
        "invalid_dates": 0,
        "inconsistent_results": 0,
        "unknown_outcomes": 0,
        "missing_odds": 0,
        "warnings": [],
    }
    if df.empty:
        report["warnings"].append("Dataset is empty.")
        return report

    report["columns_present"] = [c for c in REQUIRED_COLUMNS if c in df.columns]
    report["missing_columns"] = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if report["missing_columns"]:
        report["warnings"].append(
            f"Missing required columns: {report['missing_columns']}"
        )

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            report["missing_values"][col] = n_missing

    if "match_id" in df.columns:
        report["duplicate_fixture_ids"] = int(df["match_id"].duplicated().sum())

    if "date" in df.columns:
        dates = parse_match_dates(df["date"])
        report["invalid_dates"] = int(dates.isna().sum())
        if report["invalid_dates"]:
            report["warnings"].append(
                f"{report['invalid_dates']} rows have an unparseable date."
            )

    if "match_outcome" in df.columns:
        invalid = ~df["match_outcome"].isin(list(OUTCOME_LABELS))
        report["unknown_outcomes"] = int(invalid.sum())
        if report["unknown_outcomes"]:
            report["warnings"].append(
                f"{report['unknown_outcomes']} rows have an unknown match outcome."
            )

    if {"full_time_home_goals", "full_time_away_goals", "match_outcome"} <= set(df.columns):
        goals_home = pd.to_numeric(df["full_time_home_goals"], errors="coerce")
        goals_away = pd.to_numeric(df["full_time_away_goals"], errors="coerce")
        expected = np.select(
            [goals_home > goals_away, goals_home == goals_away],
            [2, 1],
            default=0,
        )
        inconsistent = (expected != df["match_outcome"]) & goals_home.notna() & goals_away.notna()
        report["inconsistent_results"] = int(inconsistent.sum())
        if report["inconsistent_results"]:
            report["warnings"].append(
                f"{report['inconsistent_results']} rows have a result that "
                "contradicts the scoreline."
            )

    odds_cols = ["odds_home", "odds_draw", "odds_away"]
    if all(c in df.columns for c in odds_cols):
        report["missing_odds"] = int(df[odds_cols].isna().any(axis=1).sum())

    return report


def clean_matches(df: pd.DataFrame, *, drop_inconsistent: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalise, deduplicate and validate a raw match frame.

    Steps:
    1. Parse dates and derive/normalise the season label.
    2. Canonicalise team names and rebuild stable fixture IDs.
    3. Coerce statistics and odds to numeric.
    4. Drop rows with unusable dates/teams/outcomes and exact duplicates.
    5. Optionally drop rows whose FTR code contradicts the scoreline.

    The returned report records what was removed so nothing disappears
    silently.
    """
    if df.empty:
        return df.copy(), validate_matches(df)

    out = df.copy()
    original_rows = len(out)

    for col in ("home_team", "away_team"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().map(canonical_team)

    out["date"] = parse_match_dates(out["date"])
    invalid_dates = out["date"].isna()
    if invalid_dates.any():
        logger.warning("Dropping %d rows with invalid dates.", int(invalid_dates.sum()))
    out = out[~invalid_dates].copy()

    out = out[
        out["home_team"].astype(str).str.strip().ne("")
        & out["away_team"].astype(str).str.strip().ne("")
        & out["home_team"].astype(str).ne(out["away_team"].astype(str))
    ].copy()

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "full_time_result" in out.columns:
        out["full_time_result"] = out["full_time_result"].astype(str).str.strip().str.upper()

    # Rebuild the target from the FTR code so labels are always consistent.
    if "full_time_result" in out.columns:
        out["match_outcome"] = out["full_time_result"].map(FTR_TO_OUTCOME)
        out = out[out["match_outcome"].notna()].copy()
        out["match_outcome"] = out["match_outcome"].astype(int)

    out["season"] = out["date"].map(season_from_date)
    out["match_id"] = build_fixture_ids(out)

    # Implied market probabilities: 1 / decimal odds, zero odds rejected.
    for odds_col, prob_col in (
        ("odds_home", "implied_prob_home"),
        ("odds_draw", "implied_prob_draw"),
        ("odds_away", "implied_prob_away"),
    ):
        if odds_col in out.columns:
            out[prob_col] = 1.0 / out[odds_col].where(out[odds_col] > 0)

    if drop_inconsistent and {"full_time_home_goals", "full_time_away_goals", "match_outcome"} <= set(out.columns):
        expected = np.select(
            [out["full_time_home_goals"] > out["full_time_away_goals"],
             out["full_time_home_goals"] == out["full_time_away_goals"]],
            [2, 1],
            default=0,
        )
        inconsistent = (
            (expected != out["match_outcome"])
            & out["full_time_home_goals"].notna()
            & out["full_time_away_goals"].notna()
        )
        if inconsistent.any():
            logger.warning(
                "Dropping %d rows whose result contradicts the scoreline.",
                int(inconsistent.sum()),
            )
        out = out[~inconsistent].copy()

    # Exact duplicate fixtures (same date + pairing) are removed, keeping the
    # first occurrence. match_id is deterministic, so this is idempotent.
    out = out.drop_duplicates(subset=["match_id"], keep="first")
    out = out.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="first"
    )

    out = out.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    ordered = ["match_id", "date", "season", "home_team", "away_team"]
    ordered += [c for c in out.columns if c not in ordered]
    out = out[ordered]

    report = validate_matches(out)
    report["rows_removed"] = int(original_rows - len(out))
    report["cleaned_rows"] = int(len(out))
    if report["rows_removed"]:
        logger.info("Cleaning removed %d rows.", report["rows_removed"])
    return out, report
