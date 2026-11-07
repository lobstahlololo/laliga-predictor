"""Chronologically safe dynamic Elo ratings for La Liga.

The rating system is deliberately simple enough to audit:

* every club starts at 1500 with a high rating deviation;
* the home side receives a fixed rating bonus before the expected score is
  calculated;
* the rating change uses the World Football Elo margin-of-victory multiplier;
* the effective K factor shrinks as a club's rating deviation falls, so early
  results move ratings more than later, better-established results;
* inactivity increases rating uncertainty, and each match reduces it;
* all fixtures on a date read the state that existed before that date. Updates
  are applied only after every fixture on the date has been described.

No function here sees a future match when producing a pre-match rating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import multiclass_log_loss
from src.utils.config import LABEL_ORDER

ELO_FEATURE_COLUMNS = [
    "home_elo_pre",
    "away_elo_pre",
    "elo_difference",
    "home_elo_rd",
    "away_elo_rd",
    "elo_rd_difference",
    "home_elo_matches",
    "away_elo_matches",
    "elo_expected_home",
]


@dataclass(frozen=True)
class EloConfig:
    """Fixed, declared-before-evaluation Elo settings."""

    initial_rating: float = 1500.0
    initial_rating_deviation: float = 350.0
    minimum_rating_deviation: float = 40.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    mov_enabled: bool = True
    mov_cap: float = 3.0
    uncertainty_horizon_days: float = 365.0
    inactivity_inflation: float = 50.0
    rd_shrink: float = 800.0


def expected_home_score(
    home_rating: float,
    away_rating: float,
    home_advantage: float = 60.0,
) -> float:
    """Standard Elo expected score for the home team, including home edge."""
    difference = float(home_rating) + float(home_advantage) - float(away_rating)
    return float(1.0 / (1.0 + 10.0 ** (-difference / 400.0)))


def margin_of_victory_multiplier(goal_difference: int, cap: float = 3.0) -> float:
    """World Football Elo-style margin multiplier.

    A one-goal result counts once, two goals 1.5 times, and larger wins scale
    with the margin. The cap prevents a single blowout from dominating a club's
    rating. Draws always use 1.0.
    """
    difference = abs(int(goal_difference))
    if difference <= 1:
        value = 1.0
    elif difference == 2:
        value = 1.5
    else:
        value = (11.0 + difference) / 8.0
    return float(min(cap, value))


def _rating_deviation_before(
    previous_rd: float | None,
    previous_date: pd.Timestamp | None,
    match_date: pd.Timestamp,
    config: EloConfig,
) -> float:
    if previous_rd is None or previous_date is None:
        return float(config.initial_rating_deviation)
    elapsed = max(0.0, (match_date - previous_date).days)
    inflated = np.sqrt(
        previous_rd**2
        + (elapsed / config.uncertainty_horizon_days) * config.inactivity_inflation**2
    )
    return float(min(config.initial_rating_deviation, inflated))


def _rating_deviation_after(rd_before: float, config: EloConfig) -> float:
    variance = 1.0 / max(rd_before, 1e-6) ** 2 + 1.0 / config.rd_shrink**2
    return float(max(config.minimum_rating_deviation, np.sqrt(1.0 / variance)))


def compute_elo(
    raw: pd.DataFrame,
    config: EloConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pre-match Elo features and a date-by-date rating history.

    Returns
    -------
    features:
        One row per ``match_id`` with only pre-match information.
    history:
        Long rating snapshots after each match date: date, team, rating,
        rating_deviation and matches.
    """
    config = config or EloConfig()
    required = {
        "match_id", "date", "home_team", "away_team",
        "full_time_home_goals", "full_time_away_goals",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Elo input missing columns: {sorted(missing)}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["date", "match_id"]).reset_index(drop=True)

    ratings: dict[str, float] = {}
    deviations: dict[str, float] = {}
    matches: dict[str, int] = {}
    last_dates: dict[str, pd.Timestamp] = {}
    feature_rows: list[dict] = []
    history_rows: list[dict] = []

    for match_date, group in frame.groupby("date", sort=True):
        pending: list[dict] = []

        for row in group.itertuples(index=False):
            home = str(row.home_team)
            away = str(row.away_team)
            home_pre = float(ratings.get(home, config.initial_rating))
            away_pre = float(ratings.get(away, config.initial_rating))
            home_rd = _rating_deviation_before(
                deviations.get(home), last_dates.get(home), match_date, config
            )
            away_rd = _rating_deviation_before(
                deviations.get(away), last_dates.get(away), match_date, config
            )
            expected = expected_home_score(home_pre, away_pre, config.home_advantage)
            feature_rows.append({
                "match_id": str(row.match_id),
                "home_elo_pre": home_pre,
                "away_elo_pre": away_pre,
                "elo_difference": home_pre - away_pre,
                "home_elo_rd": home_rd,
                "away_elo_rd": away_rd,
                "elo_rd_difference": home_rd - away_rd,
                "home_elo_matches": int(matches.get(home, 0)),
                "away_elo_matches": int(matches.get(away, 0)),
                "elo_expected_home": expected,
            })
            pending.append({
                "home": home,
                "away": away,
                "home_goals": int(row.full_time_home_goals),
                "away_goals": int(row.full_time_away_goals),
                "home_pre": home_pre,
                "away_pre": away_pre,
                "home_rd": home_rd,
                "away_rd": away_rd,
                "expected": expected,
            })

        # Apply the date's results only after every fixture has been described.
        for item in pending:
            home_goals = item["home_goals"]
            away_goals = item["away_goals"]
            if home_goals > away_goals:
                home_score = 1.0
            elif home_goals == away_goals:
                home_score = 0.5
            else:
                home_score = 0.0
            goal_difference = home_goals - away_goals
            mov = (
                margin_of_victory_multiplier(goal_difference, config.mov_cap)
                if config.mov_enabled and goal_difference != 0
                else 1.0
            )
            home_k = config.k_factor * np.clip(item["home_rd"] / 200.0, 0.5, 1.5)
            away_k = config.k_factor * np.clip(item["away_rd"] / 200.0, 0.5, 1.5)
            delta = home_score - item["expected"]
            home_change = float(home_k * mov * delta)
            away_change = float(-away_k * mov * delta)
            ratings[item["home"]] = item["home_pre"] + home_change
            ratings[item["away"]] = item["away_pre"] + away_change
            deviations[item["home"]] = _rating_deviation_after(item["home_rd"], config)
            deviations[item["away"]] = _rating_deviation_after(item["away_rd"], config)
            matches[item["home"]] = matches.get(item["home"], 0) + 1
            matches[item["away"]] = matches.get(item["away"], 0) + 1
            last_dates[item["home"]] = match_date
            last_dates[item["away"]] = match_date

        for team in sorted({item["home"] for item in pending} | {item["away"] for item in pending}):
            history_rows.append({
                "date": match_date,
                "team": team,
                "rating": float(ratings[team]),
                "rating_deviation": float(deviations[team]),
                "matches": int(matches[team]),
            })

    features = pd.DataFrame(feature_rows).set_index("match_id")
    history = pd.DataFrame(history_rows)
    return features[ELO_FEATURE_COLUMNS], history


def latest_elo_ratings(history: pd.DataFrame) -> pd.DataFrame:
    """Latest rating snapshot for every club that has appeared."""
    if history.empty:
        return pd.DataFrame(columns=["team", "rating", "rating_deviation", "matches"])
    latest = (
        history.sort_values(["team", "date"])
        .groupby("team", as_index=False)
        .tail(1)
        .copy()
    )
    latest["rank"] = latest["rating"].rank(ascending=False, method="min").astype(int)
    return latest.sort_values(["rank", "team"]).reset_index(drop=True)


def _probabilities(
    expected_home: np.ndarray,
    rating_difference: np.ndarray,
    draw_base: float,
    draw_scale: float,
) -> np.ndarray:
    expected = np.clip(np.asarray(expected_home, dtype=float), 1e-6, 1.0 - 1e-6)
    difference = np.asarray(rating_difference, dtype=float)
    draw = draw_base * np.exp(-((difference / max(draw_scale, 1e-6)) ** 2))
    draw = np.clip(draw, 0.03, 0.60)
    home = np.clip(expected - 0.5 * draw, 1e-9, None)
    away = np.clip(1.0 - expected - 0.5 * draw, 1e-9, None)
    out = np.column_stack([home, draw, away])
    return out / out.sum(axis=1, keepdims=True)


def elo_probabilities(
    features: pd.DataFrame,
    draw_base: float = 0.28,
    draw_scale: float = 220.0,
) -> np.ndarray:
    """Turn Elo features into ``[home, draw, away]`` probabilities."""
    required = {"elo_expected_home", "elo_difference"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Elo feature frame missing: {sorted(missing)}")
    return _probabilities(
        features["elo_expected_home"].to_numpy(float),
        features["elo_difference"].to_numpy(float),
        draw_base,
        draw_scale,
    )


def fit_elo_probability_params(
    features: pd.DataFrame,
    y: Iterable[int],
) -> tuple[dict, pd.DataFrame]:
    """Fit the two draw-shape parameters on training data only.

    Elo ratings themselves are not fitted to outcomes here; only the mapping
    from an Elo difference to a three-way probability is tuned. A small grid is
    used so the choice is transparent and reproducible.
    """
    y_array = np.asarray(y, dtype=int)
    if len(features) != len(y_array):
        raise ValueError("Elo features and labels have different lengths.")
    rows = []
    for draw_base in np.arange(0.18, 0.361, 0.02):
        for draw_scale in (80.0, 120.0, 160.0, 200.0, 240.0, 300.0, 380.0):
            probabilities = _probabilities(
                features["elo_expected_home"].to_numpy(float),
                features["elo_difference"].to_numpy(float),
                float(draw_base),
                float(draw_scale),
            )
            loss = multiclass_log_loss(y_array, probabilities, LABEL_ORDER)
            rows.append({
                "draw_base": float(draw_base),
                "draw_scale": float(draw_scale),
                "log_loss": float(loss),
            })
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    best = table.iloc[0]
    params = {
        "draw_base": float(best["draw_base"]),
        "draw_scale": float(best["draw_scale"]),
        "selection": "training log-loss grid; test not used",
    }
    return params, table


def config_as_dict(config: EloConfig) -> dict:
    return asdict(config)
