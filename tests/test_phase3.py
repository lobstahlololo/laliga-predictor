"""Phase 3 tests: Elo chronology, Poisson validity, ensembles and market odds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ensemble import (
    blend_probabilities,
    select_ensemble_weights,
)
from src.evaluation.market import (
    implied_probabilities,
    market_from_meta,
    overround_summary,
)
from src.models.elo import (
    ELO_FEATURE_COLUMNS,
    EloConfig,
    compute_elo,
    elo_probabilities,
    fit_elo_probability_params,
)
from src.models.poisson import (
    POISSON_FEATURE_COLUMNS,
    PoissonConfig,
    PoissonModel,
    build_pre_match_poisson_features,
)


def _one_match(
    home="Home",
    away="Away",
    home_goals=2,
    away_goals=0,
    date="2025-08-01",
    match_id="m1",
) -> pd.DataFrame:
    return pd.DataFrame([{
        "match_id": match_id,
        "date": date,
        "home_team": home,
        "away_team": away,
        "full_time_home_goals": home_goals,
        "full_time_away_goals": away_goals,
        "full_time_result": "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A"),
        "home_shots": 10,
        "away_shots": 5,
        "home_shots_on_target": 5,
        "away_shots_on_target": 2,
        "odds_home": 2.0,
        "odds_draw": 3.2,
        "odds_away": 4.0,
        "season": "25/26",
        "match_outcome": 2 if home_goals > away_goals else (1 if home_goals == away_goals else 0),
    }])


def test_elo_updates_are_chronological_and_uncertainty_falls():
    raw = _one_match()
    features, history = compute_elo(raw)
    assert list(features.columns) == ELO_FEATURE_COLUMNS
    assert len(features) == 1
    home = history[history["team"] == "Home"].iloc[-1]
    away = history[history["team"] == "Away"].iloc[-1]
    assert home["rating"] > 1500
    assert away["rating"] < 1500
    assert home["rating_deviation"] < 350
    assert away["rating_deviation"] < 350
    assert int(home["matches"]) == 1
    probs = elo_probabilities(features, 0.28, 220.0)
    assert probs.shape == (1, 3)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_elo_does_not_use_future_matches(raw_matches):
    full, _ = compute_elo(raw_matches)
    cutoff = pd.Timestamp("2025-01-01")
    truncated = raw_matches[pd.to_datetime(raw_matches["date"]) < cutoff]
    cut, _ = compute_elo(truncated)
    common = sorted(set(full.index) & set(cut.index))
    assert common
    pd.testing.assert_frame_equal(
        full.loc[common], cut.loc[common], check_dtype=False
    )


def test_elo_probability_parameters_are_fitted_on_training_only(raw_matches):
    features, _ = compute_elo(raw_matches)
    y = raw_matches.set_index("match_id").loc[features.index, "match_outcome"].to_numpy()
    params, table = fit_elo_probability_params(features.head(70), y[:70])
    assert {"draw_base", "draw_scale"}.issubset(params)
    assert len(table) > 1
    assert np.isfinite(table["log_loss"]).all()
    assert table["log_loss"].min() <= table["log_loss"].max()


def test_poisson_matrix_is_normalized_and_outcomes_aggregate(raw_matches):
    model = PoissonModel(PoissonConfig(dixon_coles=True)).fit(
        raw_matches, as_of=pd.to_datetime(raw_matches["date"]).max()
    )
    matrix = model.scoreline_matrix("Real Madrid", "Barcelona")
    values = matrix.to_numpy(float)
    assert values.shape == (11, 11)
    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert values.sum() == pytest.approx(1.0, abs=1e-9)
    probs = model.outcome_probabilities("Real Madrid", "Barcelona")
    assert probs.shape == (3,)
    assert (probs >= 0).all()
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert probs[0] == pytest.approx(np.tril(values, -1).sum(), abs=1e-9)
    assert probs[1] == pytest.approx(np.trace(values), abs=1e-9)
    assert probs[2] == pytest.approx(np.triu(values, 1).sum(), abs=1e-9)


def test_poisson_dixon_coles_changes_low_score_cells(raw_matches):
    independent = PoissonModel(PoissonConfig(dixon_coles=False)).fit(raw_matches)
    dc = PoissonModel(PoissonConfig(
        dixon_coles=True, rho_min=0.1, rho_max=0.1, rho_steps=1
    )).fit(raw_matches)
    assert independent.rho == 0.0
    assert dc.rho == pytest.approx(0.1)
    base = independent.scoreline_matrix("Real Madrid", "Barcelona").to_numpy(float)
    adjusted = dc.scoreline_matrix("Real Madrid", "Barcelona").to_numpy(float)
    assert adjusted.sum() == pytest.approx(1.0, abs=1e-9)
    assert not np.allclose(base, adjusted)


def test_poisson_sparse_and_unknown_teams_fall_back_safely():
    tiny = _one_match(home="Only Home", away="Only Away")
    model = PoissonModel(PoissonConfig(min_matches=3)).fit(tiny)
    prediction = model.predict_fixture("Unknown FC", "Only Home")
    assert prediction["home_known"] is False
    assert prediction["home_matches"] == 0
    assert np.isfinite([
        prediction["prob_home"], prediction["prob_draw"], prediction["prob_away"],
    ]).all()
    assert sum([
        prediction["prob_home"], prediction["prob_draw"], prediction["prob_away"],
    ]) == pytest.approx(1.0, abs=1e-9)


def test_pre_match_poisson_features_do_not_use_future_matches(raw_matches):
    full = build_pre_match_poisson_features(raw_matches)
    cutoff = pd.Timestamp("2025-01-01")
    truncated = raw_matches[pd.to_datetime(raw_matches["date"]) < cutoff]
    cut = build_pre_match_poisson_features(truncated)
    common = sorted(set(full.index) & set(cut.index))
    assert common
    pd.testing.assert_frame_equal(full.loc[common], cut.loc[common], check_dtype=False)
    assert list(full.columns) == POISSON_FEATURE_COLUMNS


def test_ensemble_weights_are_valid_and_beat_equal_on_validation():
    rng = np.random.default_rng(11)
    y = rng.choice([2, 1, 0], size=180, p=[0.45, 0.25, 0.30])
    good = np.full((180, 3), 0.10)
    true_index = np.argmax(y[:, None] == np.array([2, 1, 0]), axis=1)
    good[np.arange(180), true_index] = 0.80
    good = good / good.sum(axis=1, keepdims=True)
    weak = rng.dirichlet([5, 5, 5], size=180)
    result = select_ensemble_weights({"good": good, "weak": weak}, y, step=0.1)
    weights = result["weights"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in weights.values())
    assert result["validation_log_loss"] <= result["equal_validation_log_loss"] + 1e-12
    blended = blend_probabilities({"good": good, "weak": weak}, weights)
    np.testing.assert_allclose(blended.sum(axis=1), 1.0, atol=1e-9)


def test_market_overround_is_removed(raw_matches):
    odds = np.array([[2.0, 3.0, 4.0], [1.5, 4.0, 6.0]])
    probabilities, overround = implied_probabilities(odds)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert overround[0] == pytest.approx(1 / 2 + 1 / 3 + 1 / 4 - 1)
    assert overround[1] == pytest.approx(1 / 1.5 + 1 / 4 + 1 / 6 - 1)
    assert (probabilities >= 0).all()

    meta = raw_matches.head(6).copy()
    market = market_from_meta(meta)
    np.testing.assert_allclose(
        market[["market_home", "market_draw", "market_away"]].sum(axis=1),
        1.0, atol=1e-9,
    )
    summary = overround_summary(raw_matches)
    assert {"season", "mean_overround"}.issubset(summary.columns)
    assert len(summary) == raw_matches["season"].nunique()


def test_phase3_predictions_are_reproducible(raw_matches):
    first = PoissonModel(PoissonConfig()).fit(raw_matches)
    second = PoissonModel(PoissonConfig()).fit(raw_matches)
    assert first.predict_fixture("Real Madrid", "Barcelona") == second.predict_fixture(
        "Real Madrid", "Barcelona"
    )
    np.testing.assert_allclose(
        first.scoreline_matrix("Real Madrid", "Barcelona").to_numpy(float),
        second.scoreline_matrix("Real Madrid", "Barcelona").to_numpy(float),
        atol=1e-12,
    )
    elo_one, _ = compute_elo(raw_matches)
    elo_two, _ = compute_elo(raw_matches)
    pd.testing.assert_frame_equal(elo_one, elo_two)
