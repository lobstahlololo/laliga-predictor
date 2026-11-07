"""Data cleaning, IDs and validation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.make_dataset import COLUMN_MAP
from src.data.validation import (
    build_fixture_ids,
    clean_matches,
    season_from_date,
    validate_matches,
)
from src.utils.teams import canonical_team, resolve_team


def test_team_normalization_and_aliases():
    assert canonical_team("Atletico Madrid") == "Ath Madrid"
    assert canonical_team("Real Betis") == "Betis"
    assert canonical_team("  FC Barcelona ") == "Barcelona"
    assert canonical_team("Nonexistent FC") == "Nonexistent FC"


def test_resolve_team_alias_and_fuzzy():
    available = ["Real Madrid", "Betis", "Ath Madrid", "Barcelona"]
    assert resolve_team("Real Betis", available) == "Betis"
    assert resolve_team("atletico madrid", available) == "Ath Madrid"
    assert resolve_team("Barcelon", available) == "Barcelona"
    with pytest.raises(ValueError):
        resolve_team("", available)


def test_fixture_ids_are_stable_and_canonical():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "home_team": ["Real Betis"],
        "away_team": ["Atletico Madrid"],
    })
    ids = build_fixture_ids(df)
    assert ids.iloc[0] == "20240102_Betis_v_Ath Madrid"
    assert build_fixture_ids(df).iloc[0] == ids.iloc[0]


def test_season_from_date():
    assert season_from_date(pd.Timestamp("2024-08-01")) == "24/25"
    assert season_from_date(pd.Timestamp("2025-02-01")) == "24/25"
    assert season_from_date(pd.Timestamp("2026-11-01")) == "26/27"


def test_clean_deduplicates_idempotently(raw_matches):
    doubled = pd.concat([raw_matches, raw_matches], ignore_index=True)
    cleaned, report = clean_matches(doubled)
    assert len(cleaned) == len(raw_matches)
    assert report["cleaned_rows"] == len(raw_matches)

    cleaned_again, _ = clean_matches(cleaned)
    assert len(cleaned_again) == len(cleaned)
    assert cleaned_again["match_id"].is_unique


def test_clean_sorts_chronologically_and_derives_targets(raw_matches):
    cleaned, _ = clean_matches(raw_matches.sample(frac=1, random_state=1))
    assert cleaned["date"].is_monotonic_increasing
    assert cleaned["season"].notna().all()
    assert set(cleaned["match_outcome"].unique()) <= {0, 1, 2}


def test_validation_flags_inconsistent_results(raw_matches):
    # Break the scoreline so it contradicts the FTR code (deriving the target
    # from FTR is intentional, so tampering with match_outcome alone is repaired).
    broken = raw_matches.copy()
    idx = broken.index[0]
    broken.loc[idx, "full_time_result"] = "H"
    broken.loc[idx, "full_time_home_goals"] = 0
    broken.loc[idx, "full_time_away_goals"] = 3
    report = validate_matches(broken.assign(
        match_outcome=broken["full_time_result"].map({"H": 2, "D": 1, "A": 0})
    ))
    assert report["inconsistent_results"] == 1

    cleaned, _ = clean_matches(broken)
    assert len(cleaned) == len(raw_matches) - 1


def test_invalid_dates_and_missing_columns_are_reported(raw_matches):
    broken = raw_matches.copy()
    broken["date"] = broken["date"].astype(object)
    broken.loc[broken.index[0], "date"] = "not-a-date"
    report = validate_matches(broken)
    assert report["invalid_dates"] >= 1

    missing = raw_matches.drop(columns=["odds_home"])
    report = validate_matches(missing)
    assert "odds_home" in report["missing_columns"]


def test_required_column_map_is_complete():
    assert set(COLUMN_MAP) >= {
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "B365H", "B365D", "B365A",
    }


def test_implied_probabilities_are_inverse_odds(raw_matches):
    row = raw_matches.iloc[0]
    if pd.notna(row["odds_home"]) and row["odds_home"] > 0:
        assert row["implied_prob_home"] == pytest.approx(1 / row["odds_home"])
