"""Fixture discovery: normalisation, merge, rate limits and fallback."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import fixture_provider as fp
from src.predictions import fixtures as fixtures_module


class StubProvider:
    name = "stub"

    def __init__(self, records):
        self.records = records
        self.calls = 0

    def fetch_upcoming(self):
        self.calls += 1
        if isinstance(self.records, Exception):
            raise self.records
        return self.records


def _record(**overrides) -> fp.FixtureRecord:
    data = dict(
        home_team="Real Madrid",
        away_team="Barcelona",
        kickoff_utc="2026-11-08T18:30:00Z",
        provider="thesportsdb",
        provider_event_id="evt-1",
        status="scheduled",
        fetched_at="2026-11-01T00:00:00+00:00",
    )
    data.update(overrides)
    return fp.FixtureRecord(**data)


def test_thesportsdb_payload_is_normalised_and_missing_fields_skipped():
    payload = {"events": [
        {
            "idEvent": "1", "strHomeTeam": "FC Barcelona", "strAwayTeam": "Real Betis",
            "dateEvent": "2026-11-08", "strTime": "18:30:00", "strStatus": "",
        },
        {
            "idEvent": "2", "strHomeTeam": "", "strAwayTeam": "Valencia",
            "dateEvent": "2026-11-09", "strTime": "20:00:00",
        },
        {
            "idEvent": "3", "strHomeTeam": "Girona", "strAwayTeam": "Sevilla",
            "dateEvent": "2026-11-10", "strTime": "21:00:00", "strStatus": "Postponed",
        },
    ]}
    records = fp.records_from_thesportsdb(payload)
    assert [r.home_team for r in records] == ["Barcelona", "Girona"]
    assert records[0].away_team == "Betis"
    assert records[0].kickoff_utc == "2026-11-08T18:30:00Z"
    assert records[0].match_date == "2026-11-08"
    assert records[1].status == "postponed"
    assert all(r.provider == fp.THESPORTSDB for r in records)


def test_thesportsdb_null_events_is_an_empty_list():
    assert fp.records_from_thesportsdb({"events": None}) == []


def test_api_football_payload_is_normalised():
    payload = {
        "errors": {},
        "response": [{
            "fixture": {
                "id": 9001,
                "date": "2026-11-08T18:30:00+00:00",
                "status": {"short": "NS", "long": "Not Started"},
            },
            "teams": {
                "home": {"name": "Atletico Madrid"},
                "away": {"name": "Real Sociedad"},
            },
        }],
    }
    records = fp.records_from_api_football(payload)
    assert len(records) == 1
    record = records[0]
    assert record.home_team == "Ath Madrid"
    assert record.away_team == "Sociedad"
    assert record.provider_event_id == "9001"
    assert record.kickoff_utc == "2026-11-08T18:30:00Z"

    with pytest.raises(fp.FixtureProviderError):
        fp.records_from_api_football({"errors": {"token": "invalid"}, "response": []})


def test_stable_fixture_id_ignores_kickoff_time_but_not_date():
    first = fp.stable_fixture_id("Real Betis", "Atletico Madrid", "2026-11-08")
    second = fp.stable_fixture_id("Real Betis", "Ath Madrid", "2026-11-08")
    assert first == second
    assert first != fp.stable_fixture_id(
        "Real Betis", "Ath Madrid", "2026-11-09"
    )


def test_merge_updates_changed_kickoff_and_keeps_unreturned_fixtures():
    first_batch = fp.merge_fixtures(
        pd.DataFrame(columns=fp.FIXTURE_COLUMNS),
        [_record()],
    )
    assert len(first_batch) == 1

    incoming = [
        _record(kickoff_utc="2026-11-08T21:00:00Z"),  # time moved
        _record(kickoff_utc="2026-11-08T21:00:00Z"),  # duplicate in batch
        _record(
            home_team="Girona", away_team="Sevilla", provider_event_id="evt-2",
            status="cancelled", kickoff_utc="2026-11-10T21:00:00Z",
        ),
    ]
    merged = fp.merge_fixtures(first_batch, incoming)
    assert len(merged) == 2
    event_one = merged[merged["provider_event_id"] == "evt-1"]
    assert len(event_one) == 1
    assert event_one.iloc[0]["kickoff_utc"] == "2026-11-08T21:00:00Z"
    assert merged[merged["provider_event_id"] == "evt-2"].iloc[0]["status"] == "cancelled"


def test_merge_drops_finished_events_but_keeps_stored_history_untouched():
    existing = fp.merge_fixtures(
        pd.DataFrame(columns=fp.FIXTURE_COLUMNS),
        [_record(provider_event_id="old", kickoff_utc="2026-10-01T18:00:00Z")],
    )
    merged = fp.merge_fixtures(
        existing,
        [_record(provider_event_id="done", status="Match Finished")],
    )
    assert len(merged) == 1
    assert merged.iloc[0]["provider_event_id"] == "old"


def test_rate_limiter_waits_when_window_is_full():
    now = [0.0]
    slept = []

    def clock():
        return now[0]

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = fp.RateLimiter(2, 10, clock=clock, sleep=sleep)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()
    assert slept and slept[0] > 9


def _response(status, payload=None, headers=None):
    class Response:
        def __init__(self):
            self.status_code = status
            self.headers = headers or {}
            self._payload = payload

        def json(self):
            return self._payload

    return Response()


def test_http_retries_on_rate_limit_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response(429, headers={"Retry-After": "1"})
        return _response(200, {"events": []})

    monkeypatch.setattr(fp.requests, "get", fake_get)
    payload = fp._request_json(
        "https://example.test", limiter=fp.RateLimiter(100), sleep=lambda _s: None
    )
    assert payload == {"events": []}
    assert calls["n"] == 2


def test_http_gives_up_on_client_error(monkeypatch):
    monkeypatch.setattr(fp.requests, "get", lambda *_a, **_k: _response(401, {}))
    with pytest.raises(fp.FixtureProviderError, match="HTTP 401"):
        fp._request_json(
            "https://example.test", limiter=fp.RateLimiter(100), sleep=lambda _s: None
        )


def test_refresh_writes_atomically_and_reports_counts(tmp_path):
    path = tmp_path / "upcoming_fixtures.csv"
    result = fp.refresh_upcoming_fixtures(path, provider=StubProvider([_record()]))
    assert result.status == "ok"
    assert result.fetched == 1 and result.written == 1
    stored = pd.read_csv(path, dtype=str)
    assert len(stored) == 1
    assert stored.iloc[0]["home_team"] == "Real Madrid"
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_provider_failure_keeps_existing_fixtures(tmp_path):
    path = tmp_path / "upcoming_fixtures.csv"
    fp.refresh_upcoming_fixtures(path, provider=StubProvider([_record()]))
    before = path.read_text(encoding="utf-8")

    result = fp.refresh_upcoming_fixtures(
        path, provider=StubProvider(RuntimeError("rate limit exceeded"))
    )
    assert result.status == "fallback"
    assert result.kept == 1
    assert "rate limit" in result.errors[0]
    assert path.read_text(encoding="utf-8") == before


def test_discovery_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv(fp.FIXTURE_DISCOVERY_ENV, "0")
    result = fp.refresh_upcoming_fixtures(tmp_path / "upcoming_fixtures.csv")
    assert result.status == "disabled"


def test_unknown_provider_is_reported_without_network(tmp_path, monkeypatch):
    # Discovery must be enabled for the provider name to be validated at all;
    # CI sets LALIGA_FIXTURE_DISCOVERY=0 globally, so set it explicitly here.
    monkeypatch.setenv(fp.FIXTURE_DISCOVERY_ENV, "1")
    monkeypatch.setenv(fp.FIXTURE_PROVIDER_ENV, "not-a-provider")
    result = fp.refresh_upcoming_fixtures(tmp_path / "upcoming_fixtures.csv")
    assert result.status in {"disabled", "error"}
    assert "Unknown fixture provider" in " ".join(result.errors)


def test_load_upcoming_fixtures_filters_cancelled_and_dedupes(tmp_path, monkeypatch):
    path = tmp_path / "upcoming_fixtures.csv"
    pd.DataFrame([
        {
            "fixture_id": "fx_1", "provider": "thesportsdb", "provider_event_id": "1",
            "home_team": "Real Madrid", "away_team": "Barcelona",
            "kickoff_utc": "2026-11-08T18:30:00Z", "match_date": "2026-11-08",
            "status": "scheduled", "source": "test", "fetched_at": "x",
        },
        {
            "fixture_id": "fx_1", "provider": "thesportsdb", "provider_event_id": "1",
            "home_team": "Real Madrid", "away_team": "Barcelona",
            "kickoff_utc": "2026-11-08T21:00:00Z", "match_date": "2026-11-08",
            "status": "scheduled", "source": "test", "fetched_at": "x",
        },
        {
            "fixture_id": "fx_2", "provider": "thesportsdb", "provider_event_id": "2",
            "home_team": "Girona", "away_team": "Sevilla",
            "kickoff_utc": "2026-11-10T21:00:00Z", "match_date": "2026-11-10",
            "status": "cancelled", "source": "test", "fetched_at": "x",
        },
    ]).to_csv(path, index=False)
    monkeypatch.setattr(fixtures_module, "UPCOMING_FIXTURES_PATH", path)

    fixtures, source = fixtures_module.load_upcoming_fixtures()
    assert source == str(path)
    assert len(fixtures) == 1
    assert fixtures[0]["kickoff_utc"] == "2026-11-08T21:00:00Z"
    assert fixtures[0]["match_date"] == "2026-11-08"
    assert fixtures[0]["fixture_id"] == "fx_1"


def test_step_data_records_fixture_refresh_and_tolerates_failure(
    raw_matches, monkeypatch
):
    import src.data.fixture_provider as provider
    import src.data.make_dataset as make_dataset
    from src.pipeline import automation

    monkeypatch.setattr(
        make_dataset, "build_master_dataset",
        lambda: (raw_matches, {"seasons_fetched": ["26/27"], "seasons_failed": []}),
    )
    monkeypatch.setattr(make_dataset, "write_atomic", lambda *_a, **_k: None)
    monkeypatch.setattr(make_dataset, "write_quality_report", lambda *_a, **_k: None)

    fake = fp.FixtureDiscoveryResult(status="ok", provider="stub", fetched=2, written=2)
    monkeypatch.setattr(provider, "refresh_upcoming_fixtures", lambda: fake)
    detail = automation.step_data()
    assert detail["rows"] == len(raw_matches)
    assert detail["fixtures"]["status"] == "ok"
    assert detail["fixtures"]["fetched"] == 2

    def boom():
        raise RuntimeError("provider down")

    monkeypatch.setattr(provider, "refresh_upcoming_fixtures", boom)
    detail = automation.step_data()
    assert detail["fixtures"]["status"] == "error"
    assert "provider down" in detail["fixtures"]["errors"][0]
