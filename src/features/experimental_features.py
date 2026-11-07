"""Leakage-safe candidate feature groups for Phase 2 experiments.

Nothing here is deployed automatically. The module builds *candidate* features
alongside the canonical 19-feature vector so the experiment runner can test
each group out-of-sample. A group is only promoted if it improves validation
and walk-forward log-loss; the test season is never used to choose features.

Design notes
------------
* Rolling features use ``shift(1)`` before every window, exactly like the base
  pipeline.
* Elo and head-to-head state are computed date-by-date. All matches on a date
  read the ratings/history that existed *before* that date, then updates are
  applied after the whole date has been processed. This avoids same-day leakage.
* Expected goals are not included because the cached Football-Data.co.uk
  extract does not contain a reliable historical xG column. We do not invent
  one.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.features.build_features import build_model_frame, build_team_history
from src.utils.config import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Candidate feature groups
# ---------------------------------------------------------------------------

LONG_FORM_COLUMNS = [
    "home_form10_goals_scored",
    "home_form10_goals_conceded",
    "home_form10_shots",
    "away_form10_goals_scored",
    "away_form10_goals_conceded",
    "away_form10_shots",
]

VENUE_FORM_COLUMNS = [
    "home_venue_form_goals_scored",
    "home_venue_form_goals_conceded",
    "home_venue_form_points",
    "away_venue_form_goals_scored",
    "away_venue_form_goals_conceded",
    "away_venue_form_points",
]

GOAL_TREND_COLUMNS = [
    "home_attack_trend_3v5",
    "home_defense_trend_3v5",
    "away_attack_trend_3v5",
    "away_defense_trend_3v5",
]

REST_COLUMNS = [
    "home_rest_days_clean",
    "away_rest_days_clean",
    "rest_difference",
]

ELO_COLUMNS = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_difference",
]

SOS_COLUMNS = [
    "home_sos_elo_5",
    "away_sos_elo_5",
    "sos_elo_difference",
]

H2H_COLUMNS = [
    "h2h_home_points_5",
    "h2h_home_goal_diff_5",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "long_form": LONG_FORM_COLUMNS,
    "venue_form": VENUE_FORM_COLUMNS,
    "goal_trends": GOAL_TREND_COLUMNS,
    "rest_detail": REST_COLUMNS,
    "elo": ELO_COLUMNS,
    "strength_of_opposition": SOS_COLUMNS,
    "head_to_head": H2H_COLUMNS,
}

EXTENDED_COLUMNS = [
    column for columns in FEATURE_GROUPS.values() for column in columns
]

# Stable candidate sets used by the experiment runner.
CANDIDATE_FEATURE_SETS: dict[str, list[str]] = {
    "base_19": list(FEATURE_COLUMNS),
    **{
        name: list(FEATURE_COLUMNS) + list(columns)
        for name, columns in FEATURE_GROUPS.items()
    },
    "all_extended": list(FEATURE_COLUMNS) + list(EXTENDED_COLUMNS),
}

GROUP_DESCRIPTIONS = {
    "base_19": "Canonical 19 pre-kickoff features.",
    "long_form": "10-match rolling goals and shot volume in addition to 5m/3m form.",
    "venue_form": "Separate home-only and away-only form for each club.",
    "goal_trends": "3-match versus 5-match attack/defence momentum.",
    "rest_detail": "Continuous rest days for each side and the rest difference.",
    "elo": "Pre-match Elo ratings and the rating difference.",
    "strength_of_opposition": "Average pre-match Elo of each side's last five opponents.",
    "head_to_head": "Prior head-to-head points and goal difference, home perspective.",
    "all_extended": "All supported candidates together (diagnostic only).",
}


def _rolling_prior(series: pd.Series, window: int) -> pd.Series:
    """Mean of the previous ``window`` values, never including the current row."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _elo_and_h2h(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-match Elo and prior head-to-head records without leakage."""
    df = raw.sort_values(["date", "match_id"]).reset_index(drop=True).copy()
    elo: dict[str, float] = {}
    h2h: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rows: list[dict] = []
    k_factor = 20.0
    home_advantage = 60.0

    for date, group in df.groupby("date", sort=True):
        pending_updates: list[tuple[str, str, int, int, float, float]] = []

        for row in group.itertuples(index=False):
            home = str(row.home_team)
            away = str(row.away_team)
            home_elo = float(elo.get(home, 1500.0))
            away_elo = float(elo.get(away, 1500.0))
            expected_home = 1.0 / (
                1.0 + 10.0 ** (-((home_elo + home_advantage - away_elo) / 400.0))
            )

            key = tuple(sorted((home, away)))
            prior = h2h[key]
            # Home team's perspective across previous meetings at either venue.
            points: list[float] = []
            diffs: list[float] = []
            for meeting in prior[-5:]:
                if home == meeting["home"]:
                    gf, ga = meeting["home_goals"], meeting["away_goals"]
                else:
                    gf, ga = meeting["away_goals"], meeting["home_goals"]
                points.append(3.0 if gf > ga else (1.0 if gf == ga else 0.0))
                diffs.append(float(gf - ga))

            rows.append({
                "match_id": row.match_id,
                "home_elo_pre": home_elo,
                "away_elo_pre": away_elo,
                "elo_difference": home_elo - away_elo,
                "home_sos_elo_5": np.nan,
                "away_sos_elo_5": np.nan,
                "sos_elo_difference": np.nan,
                "h2h_home_points_5": float(np.mean(points)) if points else np.nan,
                "h2h_home_goal_diff_5": float(np.mean(diffs)) if diffs else np.nan,
                "h2h_matches": len(prior),
            })

            home_goals = int(row.full_time_home_goals)
            away_goals = int(row.full_time_away_goals)
            if home_goals > away_goals:
                score_home = 1.0
            elif home_goals == away_goals:
                score_home = 0.5
            else:
                score_home = 0.0
            pending_updates.append((
                home, away, home_goals, away_goals, expected_home, date,
            ))

        # Apply all results only after every fixture on this date has been
        # described by its pre-match state.
        for home, away, home_goals, away_goals, expected_home, _date in pending_updates:
            home_elo = float(elo.get(home, 1500.0))
            away_elo = float(elo.get(away, 1500.0))
            if home_goals > away_goals:
                score_home = 1.0
            elif home_goals == away_goals:
                score_home = 0.5
            else:
                score_home = 0.0
            delta = k_factor * (score_home - expected_home)
            elo[home] = home_elo + delta
            elo[away] = away_elo - delta
            h2h[tuple(sorted((home, away)))].append({
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            })

    return pd.DataFrame(rows).set_index("match_id")


def build_experimental_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row of candidate features per ``match_id``.

    The frame contains every column in :data:`EXTENDED_COLUMNS`, indexed by
    ``match_id``. Missing values (e.g. first match for head-to-head) are left
    as ``NaN`` for the training-only preprocessor to impute.
    """
    required = {
        "match_id", "date", "home_team", "away_team",
        "full_time_home_goals", "full_time_away_goals",
        "home_shots", "away_shots", "odds_home", "odds_draw", "odds_away",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw frame missing columns for experiments: {sorted(missing)}")

    long = build_team_history(raw).sort_values(["team", "date"]).copy()
    long["points"] = np.select(
        [long["goals_scored"] > long["goals_conceded"],
         long["goals_scored"] == long["goals_conceded"]],
        [3.0, 1.0], default=0.0,
    )

    for col in ("goals_scored", "goals_conceded", "total_shots"):
        long[f"form10_{col}"] = long.groupby("team", group_keys=False)[col].transform(
            lambda s: _rolling_prior(s, 10)
        )

    # Venue-specific form: home form only in home matches, away form only in
    # away matches. Grouping by (team, side) cannot mix venues.
    side = long.sort_values(["team", "side", "date"]).copy()
    for col in ("goals_scored", "goals_conceded", "points"):
        side[f"venue_form_{col}"] = side.groupby(
            ["team", "side"], group_keys=False
        )[col].transform(lambda s: _rolling_prior(s, 5))

    home = side[side["side"] == "home"].set_index("match_id")
    away = side[side["side"] == "away"].set_index("match_id")

    elo = _elo_and_h2h(raw)
    extra = elo.copy()

    # Opponent Elo is the pre-match rating of the other side. It is used for a
    # rolling strength-of-opposition average, never the current opponent's
    # post-match rating.
    opponent_elo = {}
    for match_id, row in elo.iterrows():
        opponent_elo[(match_id, "home")] = float(row["away_elo_pre"])
        opponent_elo[(match_id, "away")] = float(row["home_elo_pre"])
    long["opponent_elo"] = [
        opponent_elo.get((mid, side_name), np.nan)
        for mid, side_name in zip(long["match_id"], long["side"])
    ]
    long = long.sort_values(["team", "date"])
    long["sos_elo_5"] = long.groupby("team", group_keys=False)["opponent_elo"].transform(
        lambda s: _rolling_prior(s, 5)
    )
    sos_home = long[long["side"] == "home"].set_index("match_id")["sos_elo_5"]
    sos_away = long[long["side"] == "away"].set_index("match_id")["sos_elo_5"]
    extra["home_sos_elo_5"] = sos_home
    extra["away_sos_elo_5"] = sos_away
    extra["sos_elo_difference"] = extra["home_sos_elo_5"] - extra["away_sos_elo_5"]

    mapping = {
        "home_form10_goals_scored": home["form10_goals_scored"],
        "home_form10_goals_conceded": home["form10_goals_conceded"],
        "home_form10_shots": home["form10_total_shots"],
        "away_form10_goals_scored": away["form10_goals_scored"],
        "away_form10_goals_conceded": away["form10_goals_conceded"],
        "away_form10_shots": away["form10_total_shots"],
        "home_venue_form_goals_scored": home["venue_form_goals_scored"],
        "home_venue_form_goals_conceded": home["venue_form_goals_conceded"],
        "home_venue_form_points": home["venue_form_points"],
        "away_venue_form_goals_scored": away["venue_form_goals_scored"],
        "away_venue_form_goals_conceded": away["venue_form_goals_conceded"],
        "away_venue_form_points": away["venue_form_points"],
        "home_rest_days_clean": home["rest_days"],
        "away_rest_days_clean": away["rest_days"],
    }
    for name, series in mapping.items():
        extra[name] = series

    extra["rest_difference"] = (
        extra["home_rest_days_clean"] - extra["away_rest_days_clean"]
    )

    return extra.reindex(columns=EXTENDED_COLUMNS)


def build_extended_frame(
    raw: pd.DataFrame,
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return ``(X_extended, y, metadata)`` aligned to the base model frame.

    ``extra_columns`` defaults to every supported candidate. The canonical base
    columns always come first, so ordering is deterministic.
    """
    X, y, meta = build_model_frame(raw)
    extras = build_experimental_features(raw)
    selected = list(EXTENDED_COLUMNS if extra_columns is None else extra_columns)
    unknown = [c for c in selected if c not in extras.columns]
    if unknown:
        raise ValueError(f"Unknown experimental features: {unknown}")
    extras = extras.reindex(meta["match_id"].to_numpy())
    extras.index = X.index

    # Goal trends require base rolling columns, which live in the model frame.
    prepared = pd.concat([X, extras[selected]], axis=1)
    if "home_attack_trend_3v5" in prepared.columns:
        prepared["home_attack_trend_3v5"] = (
            prepared["home_short_goals_scored"] - prepared["home_form_goals_scored"]
        )
        prepared["home_defense_trend_3v5"] = (
            prepared["home_short_goals_conceded"] - prepared["home_form_goals_conceded"]
        )
        prepared["away_attack_trend_3v5"] = (
            prepared["away_short_goals_scored"] - prepared["away_form_goals_scored"]
        )
        prepared["away_defense_trend_3v5"] = (
            prepared["away_short_goals_conceded"] - prepared["away_form_goals_conceded"]
        )
    return prepared, y, meta
