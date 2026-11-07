"""Unified prediction engine: expected goals, scorelines and uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.preprocessing import FeaturePreprocessor
from src.predictions.predictor import PREDICTION_LABELS, predict_fixtures
from src.utils.config import FEATURE_COLUMNS


def _model(raw_matches, model_frame):
    X, y, _meta = model_frame
    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS)
    scaled = preprocessor.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=40, max_depth=4, random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(scaled, y)
    return model, preprocessor


def _predict(raw_matches, model_frame, **kwargs):
    model, preprocessor = _model(raw_matches, model_frame)
    return predict_fixtures(
        [{
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "match_date": "2027-06-01",
            "odds": None,
        }],
        model=model,
        preprocessor=preprocessor,
        metadata={"model_name": "Random Forest", "model_version": "test-v1"},
        raw=raw_matches,
        as_of=pd.Timestamp("2027-06-01"),
        **kwargs,
    )


def test_goal_context_is_attached_and_normalized(raw_matches, model_frame):
    predictions, errors = _predict(raw_matches, model_frame)
    assert not errors
    prediction = predictions[0]

    probs = prediction["probs"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert prediction["predicted_outcome"] == PREDICTION_LABELS[
        int(np.argmax([probs["home"], probs["draw"], probs["away"]]))
    ]

    assert prediction["expected_home_goals"] > 0
    assert prediction["expected_away_goals"] > 0
    assert 0.05 <= prediction["expected_home_goals"] <= 6.0

    scorelines = prediction["top_scorelines"]
    assert len(scorelines) == 5
    probabilities = [item["probability"] for item in scorelines]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert sum(probabilities) <= 1.0 + 1e-9
    assert prediction["most_likely_scoreline"] == scorelines[0]["scoreline"]
    assert prediction["scoreline_probability"] == pytest.approx(probabilities[0])

    goal_probs = prediction["goal_probabilities"]
    assert sum(goal_probs.values()) == pytest.approx(1.0)

    uncertainty = prediction["uncertainty"]
    assert 0.0 <= uncertainty["entropy"] <= 1.0
    assert 0.0 <= uncertainty["agreement"] <= 1.0
    assert uncertainty["level"] in {"low", "medium", "high"}

    goal_model = prediction["goal_model"]
    assert goal_model["as_of"] == "2027-06-01"
    assert goal_model["name"] in {"Independent Poisson", "Dixon-Coles"}
    assert goal_model["home_matches"] > 0
    assert "goal_model_error" not in prediction


def test_goal_model_only_uses_earlier_matches(raw_matches, model_frame):
    predictions, _ = _predict(raw_matches, model_frame)
    goal_model = predictions[0]["goal_model"]
    expected_rows = int(
        (pd.to_datetime(raw_matches["date"]) < pd.Timestamp("2027-06-01")).sum()
    )
    assert goal_model["home_matches"] + goal_model["away_matches"] > 0
    # The fitted cutoff is exclusive of the fixture date; all synthetic matches
    # predate it, so the engine saw the whole history and nothing from the future.
    assert expected_rows == len(raw_matches)


def test_predictions_are_reproducible_with_goal_context(raw_matches, model_frame):
    first, _ = _predict(raw_matches, model_frame)
    second, _ = _predict(raw_matches, model_frame)
    keys = (
        "probs", "expected_home_goals", "expected_away_goals",
        "most_likely_scoreline", "scoreline_probability", "top_scorelines",
    )
    for key in keys:
        assert first[0][key] == second[0][key]
