"""Shared synthetic fixtures for the La Liga predictor test suite.

The synthetic dataset is deterministic and contains four seasons, so tests do
not depend on network access or the real cached data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.validation import clean_matches  # noqa: E402
from src.features.build_features import build_model_frame  # noqa: E402

TEAMS = [
    "Real Madrid", "Barcelona", "Ath Madrid", "Sevilla",
    "Betis", "Valencia", "Sociedad", "Villarreal",
]
SEASONS = [(2023, "23/24"), (2024, "24/25"), (2025, "25/26"), (2026, "26/27")]


def _season_matches(start_year: int, season_label: str) -> list[dict]:
    """One round-robin season: every team plays once per round.

    Avoiding two matches for the same team on the same date keeps the
    historical ordering unambiguous, which is essential for leakage tests.
    """
    rows = []
    pairs = []
    n = len(TEAMS)
    for round_no in range(n - 1):
        rotation = TEAMS[round_no:] + TEAMS[:round_no]
        for k in range(n // 2):
            home, away = rotation[k], rotation[n - 1 - k]
            if round_no % 2:
                home, away = away, home
            pairs.append((home, away, round_no))
    for home, away, round_no in pairs:
        i, j = TEAMS.index(home), TEAMS.index(away)
        goals_home = (i * 2 + j + start_year) % 4
        goals_away = (i + j * 3 + start_year) % 3
        if goals_home > goals_away:
            result = "H"
        elif goals_home == goals_away:
            result = "D"
        else:
            result = "A"
        # Deterministic, plausible odds with a small home edge.
        odds_home = round(1.55 + 0.28 * ((j - i) % 5), 2)
        odds_draw = round(3.10 + 0.12 * ((i + j) % 4), 2)
        odds_away = round(1.85 + 0.32 * ((i - j) % 5), 2)
        rows.append({
            "date": f"{start_year}-08-{12 + round_no:02d}",
            "home_team": home,
            "away_team": away,
            "full_time_home_goals": goals_home,
            "full_time_away_goals": goals_away,
            "full_time_result": result,
            "home_shots": 6 + goals_home * 3 + (i % 3),
            "away_shots": 5 + goals_away * 3 + (j % 3),
            "home_shots_on_target": 2 + goals_home,
            "away_shots_on_target": 1 + goals_away,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "season": season_label,
        })
    return rows


@pytest.fixture(scope="session")
def raw_matches() -> pd.DataFrame:
    """112 cleaned synthetic matches across four seasons."""
    rows = []
    for year, label in SEASONS:
        rows.extend(_season_matches(year, label))
    raw = pd.DataFrame(rows)
    cleaned, _ = clean_matches(raw)
    assert len(cleaned) == len(rows)
    return cleaned


@pytest.fixture(scope="session")
def model_frame(raw_matches):
    return build_model_frame(raw_matches)


@pytest.fixture()
def tiny_history(raw_matches):
    """A small, played prediction ledger row for reconciliation tests."""
    from src.predictions.history import build_prediction_record

    match = raw_matches.iloc[5]
    return build_prediction_record(
        home_team=match["home_team"],
        away_team=match["away_team"],
        match_date=f"{match['date']:%Y-%m-%d}",
        predicted_at=f"{match['date']:%Y-%m-%d}",
        model_name="Random Forest",
        model_version="test-v1",
        prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
    )
