"""Phase 8: calendar refresh, prediction lifecycle and reconciliation.

Every test here is offline: calendar batches are injected as
``FixtureRecord`` objects and the shared prediction engine is stubbed, so no
provider, model or registry is ever reached over the network.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.api.main import Services, app, get_services
from src.data import fixture_provider as fp
from src.features.preprocessing import FeaturePreprocessor
from src.models import registry
from src.monitoring import live_performance
from src.pipeline import automation
from src.predictions import history as hist
from src.predictions import lifecycle as lc
from src.utils.config import (
    FEATURE_COLUMNS,
    REGISTRY_DIR,
    SAVED_MODEL_DIR,
)

TEAMS = [
    "Real Madrid", "Barcelona", "Ath Madrid", "Sevilla",
    "Betis", "Valencia", "Sociedad", "Villarreal",
]
NOW = datetime(2026, 11, 12, 6, 15, tzinfo=timezone.utc)


def _calendar(
    days: int = 56,
    *,
    start: str = "2027-01-04",
    statuses: dict[int, str] | None = None,
    teams: list[str] | None = None,
) -> list[fp.FixtureRecord]:
    """A deterministic, unique full-calendar-shaped batch (no network)."""
    teams = teams or TEAMS
    statuses = statuses or {}
    records: list[fp.FixtureRecord] = []
    for day in range(days):
        rotation = teams[day % len(teams):] + teams[:day % len(teams)]
        date = pd.Timestamp(start) + pd.Timedelta(days=day)
        for slot in range(len(teams) // 2):
            home, away = rotation[slot], rotation[len(teams) - 1 - slot]
            records.append(fp.FixtureRecord(
                home_team=home,
                away_team=away,
                kickoff_utc=f"{date:%Y-%m-%d}T20:00:00Z",
                provider="openfootball",
                provider_event_id=f"of-{day}-{slot}",
                status=statuses.get(day, "scheduled"),
                fetched_at="2026-11-12T06:00:00+00:00",
            ))
    return records


def _refresh(tmp_path: Path, records, **kwargs) -> dict:
    return lc.refresh_fixture_calendar(
        store_path=tmp_path / "fixtures.csv",
        calendar_path=tmp_path / "calendar.json",
        state_path=tmp_path / "state.json",
        revisions_path=tmp_path / "revisions.jsonl",
        records=records,
        now=NOW,
        record_telemetry=False,
        **kwargs,
    )


def _record(**overrides) -> dict:
    base = dict(
        home_team="Real Madrid",
        away_team="Barcelona",
        match_date="2027-01-04",
        predicted_at="2026-11-12T06:15:00+00:00",
        model_name="Random Forest",
        model_version="prod-v1",
        prob_home=0.5,
        prob_draw=0.3,
        prob_away=0.2,
        market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
    )
    base.update(overrides)
    return hist.build_prediction_record(**base)


# ---------------------------------------------------------------------------
# Scheduled calendar refresh
# ---------------------------------------------------------------------------

def test_refresh_persists_verified_store_and_is_idempotent(tmp_path):
    records = _calendar(days=5)
    first = _refresh(tmp_path, records, min_fixtures=1)
    assert first["status"] == "ok" and first["verified"] is True
    stored_first = (tmp_path / "fixtures.csv").read_bytes()
    assert first["stored"] == len(records)
    assert first["schedule_changes"] == 0

    second = _refresh(tmp_path, records, min_fixtures=1)
    assert second["status"] == "ok" and second["verified"] is True
    assert second["schedule_changes"] == 0
    assert (tmp_path / "fixtures.csv").read_bytes() == stored_first
    assert lc.load_schedule_revisions(tmp_path / "revisions.jsonl") == []

    state = lc.load_refresh_state(tmp_path / "state.json")
    assert state["verified"] is True
    assert state["fetched_at"] and state["verified_at"]
    freshness = lc.calendar_freshness(state, now=NOW)
    assert freshness["verified"] is True and freshness["stale"] is False


def test_provider_outage_keeps_store_and_marks_unverified(tmp_path, monkeypatch):
    records = _calendar(days=3)
    _refresh(tmp_path, records, min_fixtures=1)
    before = (tmp_path / "fixtures.csv").read_bytes()

    def outage():
        raise RuntimeError("calendar host unavailable")

    class Broken:
        name = "openfootball"

        def fetch_payload(self):
            raise RuntimeError("calendar host unavailable")

    result = lc.refresh_fixture_calendar(
        store_path=tmp_path / "fixtures.csv",
        calendar_path=tmp_path / "calendar.json",
        state_path=tmp_path / "state.json",
        revisions_path=tmp_path / "revisions.jsonl",
        provider=Broken(),
        now=NOW,
        record_telemetry=False,
    )
    assert result["status"] == "fallback"
    assert result["verified"] is False
    assert "calendar host unavailable" in " ".join(result["errors"])
    assert (tmp_path / "fixtures.csv").read_bytes() == before
    state = lc.load_refresh_state(tmp_path / "state.json")
    assert state["verified"] is False
    # The last successful verification timestamp is never erased.
    assert state["verified_at"]
    assert lc.calendar_freshness(state, now=NOW)["verified"] is False


def test_partial_calendar_is_rejected_without_overwriting(tmp_path):
    _refresh(tmp_path, _calendar(days=56), min_fixtures=1)
    before = (tmp_path / "fixtures.csv").read_bytes()

    partial = _calendar(days=1)  # 4 fixtures
    result = _refresh(tmp_path, partial, min_fixtures=200)
    assert result["status"] == "invalid"
    assert result["verified"] is False
    assert any("full-season minimum" in error for error in result["errors"])
    assert (tmp_path / "fixtures.csv").read_bytes() == before
    assert lc.load_refresh_state(tmp_path / "state.json")["verified"] is False


def test_empty_calendar_response_is_rejected(tmp_path):
    result = _refresh(tmp_path, [], min_fixtures=1)
    assert result["status"] == "invalid"
    assert any("empty" in error for error in result["errors"])
    assert not (tmp_path / "fixtures.csv").exists()


def test_reschedule_is_recorded_and_predictions_are_versioned(tmp_path, monkeypatch):
    original = _calendar(days=2)
    _refresh(tmp_path, original, min_fixtures=1)

    moved = _calendar(days=2, start="2027-01-06")
    result = _refresh(tmp_path, moved, min_fixtures=1)
    revisions = lc.load_schedule_revisions(tmp_path / "revisions.jsonl")
    assert result["schedule_changes"] >= 1
    assert any(item["change_type"] == "rescheduled" for item in revisions)
    change = next(item for item in revisions if item["change_type"] == "rescheduled")
    assert change["previous_match_date"] != change["current_match_date"]

    store = fp.load_fixture_rows(tmp_path / "fixtures.csv")
    assert not store["match_date"].eq("").any()
    assert "2027-01-06" in set(store["match_date"])

    # An earlier prediction snapshot survives the reschedule; the next cycle
    # appends a new snapshot that supersedes it.
    monkeypatch.setattr(lc, "predict_fixtures", _fake_predict)
    ledger = tmp_path / "history.csv"
    fixtures = _slate(("Real Madrid", "Barcelona", "2027-01-04"))
    _, first = lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        history=hist.empty_history(), fixtures=fixtures,
        ledger_path=ledger, now=NOW,
    )
    assert first["appended"] == 1

    rescheduled = _slate(("Real Madrid", "Barcelona", "2027-01-06"))
    _, second = lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        fixtures=rescheduled, ledger_path=ledger,
        now=NOW + timedelta(hours=1),
    )
    assert second["appended"] == 1
    stored, _ = hist.load_history(ledger)
    assert len(stored) == 2
    newer = stored[stored["match_date"] == "2027-01-06"].iloc[0]
    older = stored[stored["match_date"] == "2027-01-04"].iloc[0]
    assert newer["prediction_id"] != older["prediction_id"]
    assert newer["supersedes_prediction_id"] == older["prediction_id"]
    assert older["prediction_status"] == hist.STATUS_PENDING  # history preserved


# ---------------------------------------------------------------------------
# Full-slate prediction cycles
# ---------------------------------------------------------------------------

def _slate(*rows) -> list[dict]:
    return [
        {
            "home_team": home,
            "away_team": away,
            "match_date": date,
            "kickoff_utc": f"{date}T20:00:00Z",
            "status": "scheduled",
            "fixture_id": hist.make_fixture_id(home, away, date),
            "provider": "openfootball",
        }
        for home, away, date in rows
    ]


def _fake_predict(fixtures, **kwargs):
    predictions = []
    for fixture in fixtures:
        predictions.append({
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "match_date": fixture["match_date"],
            "predicted_at": kwargs.get("predicted_at") or NOW.isoformat(),
            "model_name": "Random Forest",
            "model_version": "prod-v1",
            "probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
            "market": {
                "implied_prob_home": 0.45,
                "implied_prob_draw": 0.28,
                "implied_prob_away": 0.27,
            },
            "expected_home_goals": 1.4,
            "expected_away_goals": 0.9,
            "most_likely_scoreline": "1-0",
            "fixture_id": fixture["fixture_id"],
            "kickoff_utc": fixture.get("kickoff_utc", ""),
            "fixture_source": fixture.get("fixture_source", ""),
        })
    return predictions, []


def test_full_slate_predicts_once_per_fixture_per_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "predict_fixtures", _fake_predict)
    ledger = tmp_path / "history.csv"
    fixtures = _slate(
        ("Real Madrid", "Barcelona", "2027-01-04"),
        ("Sevilla", "Betis", "2027-01-05"),
    ) + _slate(("Real Madrid", "Barcelona", "2027-01-04"))  # duplicate discovery

    _, first = lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        history=hist.empty_history(), fixtures=fixtures,
        ledger_path=ledger, now=NOW,
    )
    assert first["eligible"] == 2
    assert first["excluded"]["duplicate"] == 1
    assert first["appended"] == 2

    _, again = lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        fixtures=fixtures, ledger_path=ledger, now=NOW,
    )
    assert again["appended"] == 0
    assert again["note"]

    stored, _ = hist.load_history(ledger)
    assert len(stored) == 2
    assert stored.groupby("fixture_id").size().max() == 1
    assert set(stored["prediction_cycle"]) == {lc.cycle_id(NOW)}
    assert (stored["kickoff_utc"].astype(str).str.strip() != "").all()


def test_kickoff_in_the_past_is_not_predicted(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "predict_fixtures", _fake_predict)
    past = _slate(("Real Madrid", "Barcelona", "2026-11-01"))
    future = _slate(("Sevilla", "Betis", "2027-02-01"))
    fixtures = past + future
    eligible, selection = lc.eligible_fixtures(fixtures, now=NOW)
    assert [item["match_date"] for item in eligible] == ["2027-02-01"]
    assert selection["past"] == 1

    _, detail = lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        history=hist.empty_history(), fixtures=fixtures,
        ledger_path=tmp_path / "history.csv", now=NOW,
    )
    assert detail["appended"] == 1


def test_new_model_version_appends_without_touching_old_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "predict_fixtures", _fake_predict)
    ledger = tmp_path / "history.csv"
    fixtures = _slate(("Real Madrid", "Barcelona", "2027-01-04"))
    lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        history=hist.empty_history(), fixtures=fixtures,
        ledger_path=ledger, now=NOW,
    )

    def other_model(fixtures, **kwargs):
        predictions, errors = _fake_predict(fixtures, **kwargs)
        for prediction in predictions:
            prediction["model_version"] = "prod-v2"
        return predictions, errors

    monkeypatch.setattr(lc, "predict_fixtures", other_model)
    _, detail = lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        fixtures=fixtures, ledger_path=ledger, now=NOW + timedelta(hours=1),
    )
    assert detail["appended"] == 1
    stored, _ = hist.load_history(ledger)
    assert set(stored["model_version"]) == {"prod-v1", "prod-v2"}
    assert len(stored) == 2


# ---------------------------------------------------------------------------
# Result reconciliation
# ---------------------------------------------------------------------------

def _save(records, path: Path) -> None:
    hist.save_history(hist.normalize_history(pd.DataFrame(records)), path)


def _raw(match_id, home, away, goals, date="2027-01-04", result=None) -> pd.DataFrame:
    home_goals, away_goals = goals
    if result is None:
        result = "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
    return pd.DataFrame([{
        "match_id": match_id,
        "date": date,
        "home_team": home,
        "away_team": away,
        "full_time_home_goals": home_goals,
        "full_time_away_goals": away_goals,
        "full_time_result": result,
    }])


def test_verified_result_settles_once(tmp_path):
    ledger = tmp_path / "history.csv"
    record = _record()
    _save([record], ledger)
    raw = _raw(record["fixture_id"], "Real Madrid", "Barcelona", (2, 0))

    _, first = lc.settle_results(
        raw, ledger_path=ledger, corrections_path=tmp_path / "corrections.jsonl", now=NOW
    )
    assert first["settled"] == 1
    stored, _ = hist.load_history(ledger)
    assert stored.iloc[0]["actual_outcome"] == "2"
    assert stored.iloc[0]["result_status"] == hist.RESULT_FINAL
    assert stored.iloc[0]["result_source"] == lc.DEFAULT_RESULT_SOURCE
    assert stored.iloc[0]["settled_at"]
    assert stored.iloc[0]["prediction_status"] == hist.STATUS_SETTLED

    _, second = lc.settle_results(
        raw, ledger_path=ledger, corrections_path=tmp_path / "corrections.jsonl", now=NOW
    )
    assert second["settled"] == 0
    assert second["already_settled"] == 1
    stored_again, _ = hist.load_history(ledger)
    assert len(stored_again) == 1


def test_inconsistent_or_missing_score_never_settles(tmp_path):
    ledger = tmp_path / "history.csv"
    record = _record()
    _save([record], ledger)
    raw = _raw(
        record["fixture_id"], "Real Madrid", "Barcelona", (2, 0), result="A"
    )
    _, stats = lc.settle_results(
        raw, ledger_path=ledger, corrections_path=tmp_path / "corrections.jsonl"
    )
    assert stats["settled"] == 0
    stored, _ = hist.load_history(ledger)
    assert str(stored.iloc[0]["actual_outcome"]).strip() == ""


def test_corrected_result_keeps_previous_record(tmp_path):
    ledger = tmp_path / "history.csv"
    corrections = tmp_path / "corrections.jsonl"
    record = _record()
    _save([record], ledger)

    lc.settle_results(
        _raw(record["fixture_id"], "Real Madrid", "Barcelona", (2, 0)),
        ledger_path=ledger, corrections_path=corrections, now=NOW,
    )
    _, stats = lc.settle_results(
        _raw(record["fixture_id"], "Real Madrid", "Barcelona", (2, 1)),
        ledger_path=ledger, corrections_path=corrections,
        now=NOW + timedelta(hours=2),
    )
    assert stats["corrected"] == 1
    stored, _ = hist.load_history(ledger)
    row = stored.iloc[0]
    assert (row["actual_home_goals"], row["actual_away_goals"]) == ("2", "1")
    assert row["result_status"] == hist.RESULT_CORRECTED
    assert row["correction_count"] == "1"
    assert row["corrected_at"]
    assert len(stored) == 1  # append-only ledger, row preserved

    log = lc.load_result_corrections(corrections)
    assert len(log) == 1
    assert log[0]["previous_home_goals"] == "2"
    assert log[0]["previous_away_goals"] == "0"
    assert log[0]["new_away_goals"] == "1"
    assert log[0]["fixture_id"] == record["fixture_id"]


def test_postponed_and_cancelled_fixtures_are_not_settled(tmp_path):
    ledger = tmp_path / "history.csv"
    postponed = _record(match_date="2027-01-05")
    cancelled = _record(
        home_team="Sevilla", away_team="Betis", match_date="2027-01-06"
    )
    _save([postponed, cancelled], ledger)
    fixtures = pd.DataFrame([
        {
            "fixture_id": postponed["fixture_id"], "home_team": "Real Madrid",
            "away_team": "Barcelona", "match_date": "2027-01-05",
            "status": "postponed",
        },
        {
            "fixture_id": cancelled["fixture_id"], "home_team": "Sevilla",
            "away_team": "Betis", "match_date": "2027-01-06",
            "status": "cancelled",
        },
    ])

    _, stats = lc.settle_results(
        pd.DataFrame(), fixtures=fixtures, ledger_path=ledger,
        corrections_path=tmp_path / "corrections.jsonl", now=NOW,
    )
    assert stats["postponed"] == 1
    assert stats["cancelled"] == 1 and stats["voided"] == 1
    assert stats["settled"] == 0

    stored, _ = hist.load_history(ledger)
    postponed_row = stored[stored["fixture_id"] == postponed["fixture_id"]].iloc[0]
    cancelled_row = stored[stored["fixture_id"] == cancelled["fixture_id"]].iloc[0]
    assert str(postponed_row["actual_outcome"]).strip() == ""
    assert postponed_row["result_status"] == "postponed"
    assert str(cancelled_row["actual_outcome"]).strip() == ""
    assert cancelled_row["prediction_status"] == hist.STATUS_VOID
    assert cancelled_row["result_status"] == "cancelled"


def test_result_without_a_matching_fixture_stays_pending(tmp_path):
    ledger = tmp_path / "history.csv"
    record = _record()
    _save([record], ledger)
    raw = _raw("20270104_Sevilla_v_Betis", "Sevilla", "Betis", (1, 0))
    _, stats = lc.settle_results(
        raw, ledger_path=ledger, corrections_path=tmp_path / "corrections.jsonl"
    )
    assert stats["settled"] == 0
    assert stats["pending"] == 1


# ---------------------------------------------------------------------------
# Live performance
# ---------------------------------------------------------------------------

def _settled_history() -> pd.DataFrame:
    first = _record(predicted_at="2026-11-01T10:00:00+00:00", match_date="2026-11-02")
    first.update({
        "actual_outcome": "2", "actual_home_goals": "2", "actual_away_goals": "0",
        "correct": "1", "settled_at": "2026-11-03T00:00:00+00:00",
        "result_status": hist.RESULT_FINAL, "result_source": "football-data.co.uk",
    })
    second = _record(
        home_team="Sevilla", away_team="Betis", match_date="2025-12-01",
        predicted_at="2025-11-30T10:00:00+00:00", model_version="prod-v0",
        prob_home=0.3, prob_draw=0.4, prob_away=0.3,
    )
    second.update({
        "actual_outcome": "1", "actual_home_goals": "1", "actual_away_goals": "1",
        "correct": "1", "settled_at": "2025-12-02T00:00:00+00:00",
        "result_status": hist.RESULT_FINAL, "result_source": "football-data.co.uk",
    })
    return hist.normalize_history(pd.DataFrame([first, second]))


def test_live_performance_separates_splits_and_flags_small_samples(tmp_path):
    history = _settled_history()
    metadata = {
        "validation_metrics": {"n_samples": 380, "log_loss": 1.02, "accuracy": 0.52},
        "test_metrics": {"n_samples": 119, "log_loss": 1.11, "accuracy": 0.49},
    }
    report = live_performance.build_live_performance(
        history, metadata=metadata, min_settled=30
    )
    assert report["splits"]["validation"]["source"] == "model metadata"
    assert report["splits"]["historical_test"]["n"] == 119
    assert report["splits"]["live"]["n"] == 2
    assert report["splits"]["live"]["status"] == "insufficient_data"
    assert report["status"] == "insufficient_data"
    assert report["counts"]["settled_rows"] == 2
    seasons = {block["season"] for block in report["by_season"]}
    versions = {block["model_version"] for block in report["by_model_version"]}
    assert {"26/27", "25/26"} <= seasons
    assert {"prod-v1", "prod-v0"} <= versions

    paths = live_performance.write_live_performance(
        report,
        json_path=tmp_path / "live.json",
        markdown_path=tmp_path / "live.md",
    )
    assert all(path.exists() for path in paths)
    assert "insufficient_data" in (tmp_path / "live.md").read_text(encoding="utf-8")


def test_live_performance_without_history_is_explicitly_insufficient():
    report = live_performance.build_live_performance(
        hist.empty_history(), metadata={}, min_settled=30
    )
    assert report["splits"]["live"]["status"] == "insufficient_data"
    assert report["splits"]["validation"]["status"] == "not_available"
    assert report["counts"]["settled_rows"] == 0


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------

def test_pipeline_order_is_calendar_first():
    order = list(automation.STEP_ORDER)
    assert order.index("fixtures") < order.index("data")
    assert order.index("data") < order.index("coverage") < order.index("predict")
    assert order.index("predict") < order.index("backfill") < order.index("quality")
    assert order.index("quality") < order.index("reports") < order.index("monitoring")
    assert order.index("fixtures") == 0


def test_calendar_refresh_step_is_opt_in(monkeypatch):
    monkeypatch.delenv("LALIGA_CALENDAR_REFRESH", raising=False)
    detail = automation.step_fixtures()
    assert "note" in detail


def test_calendar_refresh_step_delegates_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("LALIGA_CALENDAR_REFRESH", "1")
    sentinel = {"status": "ok", "verified": True, "stored": 380}
    monkeypatch.setattr(lc, "refresh_fixture_calendar", lambda: sentinel)
    assert automation.step_fixtures() == sentinel


# ---------------------------------------------------------------------------
# API, dashboard and registry safety
# ---------------------------------------------------------------------------

def _services(model_frame, raw_matches) -> Services:
    X, y, _meta = model_frame
    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS)
    scaled = preprocessor.fit_transform(X)
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(scaled, y)
    return Services(
        model=model,
        preprocessor=preprocessor,
        metadata={"model_name": "Random Forest", "model_version": "prod-v1"},
        raw=raw_matches,
        history=_settled_history(),
    )


def test_lifecycle_api_endpoints_are_additive(model_frame, raw_matches):
    services = _services(model_frame, raw_matches)
    app.dependency_overrides[get_services] = lambda: services
    try:
        with TestClient(app) as client:
            history = client.get("/predictions/history")
            assert history.status_code == 200
            assert history.json()[0]["model_version"] in {"prod-v1", "prod-v0"}

            fixture_id = "20261102_Real Madrid_v_Barcelona"
            per_fixture = client.get(f"/fixtures/{fixture_id}/predictions")
            assert per_fixture.status_code == 200
            assert per_fixture.json()[0]["fixture_id"] == fixture_id

            assert client.get("/results/settled").status_code == 200
            live = client.get("/performance/live")
            assert live.status_code == 200
            assert "splits" in live.json()
            assert client.get("/fixtures/upcoming").status_code == 200
            assert client.get("/fixtures/schedule-changes").status_code == 200

            # Existing contracts are untouched.
            assert client.get("/performance").status_code == 200
            assert client.get("/health").status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_operations_dashboard_page_renders():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    page = AppTest.from_file(
        str(root / "dashboard" / "pages" / "8_Fixture_Operations.py"),
        default_timeout=90,
    )
    page.run()
    assert not page.exception, [str(item.value) for item in page.exception]


def _digest(*directories: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    for root in directories:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_lifecycle_never_touches_registry_snapshots(tmp_path, monkeypatch):
    before = _digest(REGISTRY_DIR, SAVED_MODEL_DIR)
    if not before:
        pytest.skip("no committed model registry in this checkout")

    monkeypatch.setattr(lc, "predict_fixtures", _fake_predict)
    store = tmp_path / "fixtures.csv"
    _refresh(tmp_path, _calendar(days=4), min_fixtures=1)
    ledger = tmp_path / "history.csv"
    fixtures = _slate(("Real Madrid", "Barcelona", "2027-01-04"))
    lc.generate_full_slate(
        model=None, preprocessor=None, metadata={}, raw=pd.DataFrame(),
        history=hist.empty_history(), fixtures=fixtures,
        ledger_path=ledger, now=NOW,
    )
    lc.settle_results(
        _raw(hist.make_fixture_id("Real Madrid", "Barcelona", "2027-01-04"),
             "Real Madrid", "Barcelona", (2, 0)),
        ledger_path=ledger, corrections_path=tmp_path / "corrections.jsonl",
        now=NOW,
    )
    live_performance.build_live_performance(hist.load_history(ledger)[0], metadata={})
    assert store.exists()

    assert _digest(REGISTRY_DIR, SAVED_MODEL_DIR) == before
    assert registry.status()["production_version"]
