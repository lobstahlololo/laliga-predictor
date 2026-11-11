"""FastAPI endpoints using the shared prediction/statistics engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api.main import Services, app, get_services
from src.features.preprocessing import FeaturePreprocessor
from src.predictions.history import build_prediction_record, normalize_history
from src.utils.config import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def services(model_frame, raw_matches) -> Services:
    X, y, _meta = model_frame
    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS)
    scaled = preprocessor.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=30, max_depth=4, random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(scaled, y)

    record = build_prediction_record(
        home_team="Real Madrid",
        away_team="Barcelona",
        match_date="2026-08-20",
        predicted_at="2026-08-01T00:00:00+00:00",
        model_name="Random Forest",
        model_version="test-v1",
        prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
    )
    history = normalize_history(pd.DataFrame([record]))
    history.loc[0, "actual_outcome"] = "2"
    history.loc[0, "actual_home_goals"] = "2"
    history.loc[0, "actual_away_goals"] = "0"
    history.loc[0, "correct"] = "1"
    history.loc[0, "settled_at"] = "2026-08-21T00:00:00+00:00"

    return Services(
        model=model,
        preprocessor=preprocessor,
        metadata={
            "model_name": "Random Forest",
            "model_version": "test-v1",
            "validation_metrics": {"log_loss": 1.0, "brier_score": 0.6},
            "test_metrics": {"log_loss": 1.05, "brier_score": 0.61},
        },
        raw=raw_matches,
        history=history,
    )


@pytest.fixture()
def client(services):
    app.dependency_overrides[get_services] = lambda: services
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_reports_artifacts_and_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["artifacts"]) == {"raw", "model", "preprocessor", "metadata"}
    assert body["api_version"]


def test_model_endpoint_exposes_feature_contract(client):
    response = client.get("/model")
    assert response.status_code == 200
    body = response.json()
    assert body["feature_count"] == len(FEATURE_COLUMNS)
    assert body["feature_names"] == FEATURE_COLUMNS
    assert body["metadata"]["model_version"] == "test-v1"


def test_predict_returns_probabilities_goals_and_scorelines(client):
    response = client.post(
        "/predict",
        json={"home_team": "Real Madrid", "away_team": "Barcelona"},
    )
    assert response.status_code == 200
    body = response.json()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["predicted_outcome"] in {"Home Win", "Draw", "Away Win"}
    assert body["expected_home_goals"] > 0
    assert body["expected_away_goals"] > 0
    assert body["top_scorelines"]
    assert body["most_likely_scoreline"] == body["top_scorelines"][0]["scoreline"]
    assert body["goal_model"]["name"] in {"Independent Poisson", "Dixon-Coles"}
    assert body["uncertainty"]["level"] in {"low", "medium", "high"}
    assert body["model_version"] == "test-v1"
    assert body["predicted_at"]


def test_predict_rejects_unknown_team_and_bad_input(client):
    unknown = client.post(
        "/predict", json={"home_team": "Fake FC", "away_team": "Barcelona"}
    )
    assert unknown.status_code == 422

    same_team = client.post(
        "/predict", json={"home_team": "Barcelona", "away_team": "Barcelona"}
    )
    assert same_team.status_code == 422

    partial_odds = client.post(
        "/predict",
        json={"home_team": "Real Madrid", "away_team": "Barcelona", "odds_home": 2.0},
    )
    assert partial_odds.status_code == 422

    missing = client.post("/predict", json={"home_team": "Real Madrid"})
    assert missing.status_code == 422


def test_teams_and_team_stats(client):
    teams = client.get("/teams")
    assert teams.status_code == 200
    names = teams.json()
    assert "Real Madrid" in names and "Barcelona" in names

    stats = client.get("/teams/Barcelona/stats")
    assert stats.status_code == 200
    body = stats.json()
    assert body["team"] == "Barcelona"
    assert body["matches"] > 0
    assert body["elo_rating"] is not None
    assert body["attack"] is not None and body["defence"] is not None
    assert set(body["form"].split()) <= {"W", "D", "L"}

    missing = client.get("/teams/Not A Team/stats")
    assert missing.status_code == 404


def test_performance_and_registry_endpoints(client):
    performance = client.get("/performance")
    assert performance.status_code == 200
    body = performance.json()
    assert body["summary"]["predictions"] == 1
    assert body["summary"]["settled"] == 1
    assert body["summary"]["accuracy"] == pytest.approx(1.0)

    registry = client.get("/registry")
    assert registry.status_code == 200
    assert "versions" in registry.json()


def test_fixtures_endpoint_uses_configured_fixtures(client, monkeypatch):
    import src.api.main as api_main

    monkeypatch.setattr(
        api_main,
        "load_upcoming_fixtures",
        lambda: ([
            {"home_team": "Real Madrid", "away_team": "Barcelona",
             "match_date": "", "odds": None},
        ], "test"),
    )
    response = client.get("/fixtures")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["home_team"] == "Real Madrid"

    monkeypatch.setattr(api_main, "load_upcoming_fixtures", lambda: ([], "none"))
    assert client.get("/fixtures").json() == []


def test_runs_endpoint_returns_a_list(client):
    response = client.get("/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ---------------------------------------------------------------------------
# Caching integration
# ---------------------------------------------------------------------------

def _cached_payload():
    return {"home_team": "Real Madrid", "away_team": "Barcelona"}


@pytest.fixture()
def cached_client(services, monkeypatch):
    """An app with its own cache and a counting prediction engine."""
    from src.api import main as api_main
    from src.api.cache import InMemoryCache

    calls = {"n": 0}
    real_predict = api_main.predict_fixtures

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(api_main, "predict_fixtures", counting)
    app = api_main.create_app(cache=InMemoryCache(default_ttl=60))
    app.dependency_overrides[get_services] = lambda: services
    with TestClient(app) as client:
        yield client, calls, services
    app.dependency_overrides.clear()


def test_repeated_prediction_is_served_from_cache(cached_client):
    client, calls, _services = cached_client
    first = client.post("/predict", json=_cached_payload())
    second = client.post("/predict", json=_cached_payload())

    assert first.status_code == 200
    assert first.json() == second.json()
    assert calls["n"] == 1


def test_model_version_change_invalidates_cached_prediction(cached_client):
    client, calls, services = cached_client
    original = services.metadata["model_version"]
    try:
        assert client.post("/predict", json=_cached_payload()).status_code == 200
        services.metadata["model_version"] = original + "-promoted"
        assert client.post("/predict", json=_cached_payload()).status_code == 200
        assert calls["n"] == 2
    finally:
        services.metadata["model_version"] = original


def test_kickoff_and_odds_change_cache_identity(cached_client):
    client, calls, _services = cached_client
    first = {"home_team": "Real Madrid", "away_team": "Barcelona", "match_date": "2026-11-08"}
    second = {"home_team": "Real Madrid", "away_team": "Barcelona", "match_date": "2026-11-15"}
    with_odds = {
        "home_team": "Real Madrid", "away_team": "Barcelona",
        "match_date": "2026-11-08", "odds_home": 2.1, "odds_draw": 3.4, "odds_away": 3.6,
    }

    assert client.post("/predict", json=first).status_code == 200
    assert client.post("/predict", json=second).status_code == 200
    assert client.post("/predict", json=with_odds).status_code == 200
    assert calls["n"] == 3


class _BrokenRedis:
    def get(self, key):
        raise ConnectionError("redis down")

    def setex(self, key, ttl, value):
        raise ConnectionError("redis down")

    def delete(self, key):
        raise ConnectionError("redis down")

    def scan_iter(self, match=None):
        return []


def test_api_still_predicts_when_cache_backend_fails(services):
    from src.api import main as api_main
    from src.api.cache import RedisCache

    app = api_main.create_app(cache=RedisCache(_BrokenRedis()))
    app.dependency_overrides[get_services] = lambda: services
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=_cached_payload())
            assert response.status_code == 200
            assert sum(response.json()["probabilities"].values()) == pytest.approx(1.0)
    finally:
        app.dependency_overrides.clear()


def test_fixtures_endpoint_is_cached(services, monkeypatch):
    from src.api import main as api_main
    from src.api.cache import InMemoryCache

    calls = {"n": 0}
    monkeypatch.setattr(
        api_main,
        "load_upcoming_fixtures",
        lambda: ([{"home_team": "Real Madrid", "away_team": "Barcelona",
                   "match_date": "", "odds": None}], "test"),
    )
    real_predict = api_main.predict_fixtures

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_predict(*args, **kwargs)

    monkeypatch.setattr(api_main, "predict_fixtures", counting)
    app = api_main.create_app(cache=InMemoryCache(default_ttl=60))
    app.dependency_overrides[get_services] = lambda: services
    try:
        with TestClient(app) as client:
            assert client.get("/fixtures").status_code == 200
            assert client.get("/fixtures").status_code == 200
        assert calls["n"] == 1
    finally:
        app.dependency_overrides.clear()
