"""Season-calendar parsing, comparison and coverage verification tests.

Every test is offline: provider payloads are in-memory fixtures and no HTTP
client is reachable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data import fixture_coverage as coverage_module
from src.data import fixture_provider as fp
from src.data import football_calendar as fc

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)

# The 20 clubs exactly as openfootball/football.json spells them (2026-27).
OPENFOOTBALL_TEAMS = [
    "Athletic Club",
    "CA Osasuna",
    "Club Atlético de Madrid",
    "Deportivo Alavés",
    "Elche CF",
    "FC Barcelona",
    "Getafe CF",
    "Levante UD",
    "Málaga CF",
    "RC Celta de Vigo",
    "RC Deportivo La Coruña",
    "RCD Espanyol de Barcelona",
    "Rayo Vallecano de Madrid",
    "Real Betis Balompié",
    "Real Madrid CF",
    "Real Racing Club de Santander",
    "Real Sociedad de Fútbol",
    "Sevilla FC",
    "Valencia CF",
    "Villarreal CF",
]

DATASET_TEAMS = {
    "Alaves", "Ath Bilbao", "Ath Madrid", "Barcelona", "Betis", "Celta",
    "Elche", "Espanol", "Getafe", "La Coruna", "Levante", "Malaga",
    "Osasuna", "Real Madrid", "Santander", "Sevilla", "Sociedad",
    "Valencia", "Vallecano", "Villarreal",
}


def _payload() -> dict:
    return {
        "name": "Spain Primera División 2026/27",
        "matches": [
            {
                "round": "Matchday 13", "date": "2026-11-15", "time": "19:30",
                "team1": "FC Barcelona", "team2": "Real Madrid CF",
            },
            {
                "round": "Matchday 13", "date": "2026-11-15",
                "team1": "Athletic Club", "team2": "RCD Espanyol de Barcelona",
            },
            {
                "round": "Matchday 13", "date": "2026-11-16",
                "team1": "Sevilla FC", "team2": "Real Betis Balompié",
            },
            {
                "round": "Matchday 14", "date": "2026-11-22", "time": "21:00",
                "team1": "Club Atlético de Madrid", "team2": "Valencia CF",
            },
            {
                "round": "Matchday 1", "date": "2026-08-15",
                "team1": "Deportivo Alavés", "team2": "Getafe CF",
                "score": {"ft": [3, 0]},
            },
            {
                "round": "Matchday 1", "date": "2026-08-15",
                "team1": "Málaga CF", "team2": "Rayo Vallecano de Madrid",
                "score": [1, 1],
            },
        ],
    }


def _store(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=fp.FIXTURE_COLUMNS)


def _row(**overrides) -> dict:
    data = dict(
        fixture_id="", provider="openfootball", provider_event_id="",
        home_team="Barcelona", away_team="Real Madrid",
        kickoff_utc="2026-11-15T19:30:00Z", match_date="2026-11-15",
        status="scheduled", source="openfootball:es.1", fetched_at="2026-11-12T00:00:00+00:00",
    )
    data.update(overrides)
    return data


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"date": pd.Timestamp("2025-08-15"), "home_team": "Barcelona",
         "away_team": "Sevilla", "season": "25/26", "ftr": "H",
         "home_goals": 2, "away_goals": 0},
        {"date": pd.Timestamp("2026-09-01"), "home_team": "Barcelona",
         "away_team": "Betis", "season": "26/27", "ftr": "D",
         "home_goals": 1, "away_goals": 1},
    ])


def _block_network(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("calendar tests must not use the network")

    monkeypatch.setattr(fp.requests, "get", boom)
    monkeypatch.setattr(fp, "_request_json", boom)


# ---------------------------------------------------------------------------
# Payload normalisation
# ---------------------------------------------------------------------------

def test_openfootball_payload_normalises_teams_dates_and_status():
    records = fp.records_from_openfootball(_payload())

    assert len(records) == 6
    assert [r.home_team for r in records[:2]] == ["Barcelona", "Ath Bilbao"]
    assert records[2].away_team == "Betis"
    assert records[0].kickoff_utc == "2026-11-15T19:30:00Z"
    assert records[0].match_date == "2026-11-15"
    # A match without a published kickoff time keeps the date, not a fake clock.
    assert records[1].kickoff_utc == "2026-11-15T00:00:00Z"
    assert records[1].match_date == "2026-11-15"
    assert records[0].provider == fp.OPENFOOTBALL
    assert records[0].source == "openfootball:es.1"
    assert records[0].provider_event_id == ""

    # Played matches and unplayed matches are distinguishable from the feed.
    assert records[4].status == "finished"
    assert records[5].status == "finished"  # older list-shaped score form
    assert records[0].status == "scheduled"


def test_stable_ids_are_shared_with_the_rest_of_the_pipeline():
    records = fp.records_from_openfootball(_payload())
    same = fp.stable_fixture_id("FC Barcelona", "Real Madrid CF", "2026-11-15")
    assert records[0].fixture_id == same

    # A time-only move keeps identity; a date change does not.
    moved = fp.records_from_openfootball({
        "matches": [{
            "date": "2026-11-15", "time": "22:00",
            "team1": "FC Barcelona", "team2": "Real Madrid CF",
        }]
    })[0]
    assert moved.fixture_id == records[0].fixture_id
    rescheduled = fp.records_from_openfootball({
        "matches": [{
            "date": "2026-11-16", "time": "22:00",
            "team1": "FC Barcelona", "team2": "Real Madrid CF",
        }]
    })[0]
    assert rescheduled.fixture_id != records[0].fixture_id


def test_every_openfootball_club_maps_to_the_dataset_spelling():
    mapped = {fp.canonical_team(name) for name in OPENFOOTBALL_TEAMS}
    assert mapped == DATASET_TEAMS


def test_openfootball_skips_incomplete_fixtures_and_raises_on_bad_payloads():
    records = fp.records_from_openfootball({"matches": [
        {"date": "2026-11-15", "team1": "", "team2": "Valencia CF"},
        {"date": "2026-11-15", "team1": "Valencia CF", "team2": "Valencia CF"},
        "not-a-match",
        {"date": "2026-11-15", "team1": "Valencia CF", "team2": "Elche CF"},
    ]})
    assert len(records) == 1

    for bad in (None, {}, {"matches": {}}, {"matches": "nope"}):
        with pytest.raises(fp.FixtureProviderError):
            fp.records_from_openfootball(bad)


# ---------------------------------------------------------------------------
# Provider behaviour
# ---------------------------------------------------------------------------

def test_openfootball_provider_filters_finished_matches(monkeypatch):
    provider = fp.OpenFootballProvider(season="2026-27")
    monkeypatch.setattr(provider, "fetch_payload", lambda: _payload())

    records = provider.fetch_upcoming()

    assert provider.covers_full_season is True
    assert provider.source_url.endswith("/2026-27/es.1.json")
    assert len(records) == 4
    assert all(record.status != "finished" for record in records)


def test_refresh_reports_full_season_scope_only_for_calendar_providers(monkeypatch, tmp_path):
    class PartialProvider:
        name = "thesportsdb"

        def fetch_upcoming(self):
            return [fp.records_from_openfootball(_payload())[0]]

    # Drive the provider with a patched payload so no request is made.
    calendar_provider = fp.OpenFootballProvider(season="2026-27")
    monkeypatch.setattr(calendar_provider, "fetch_upcoming",
                        lambda: fp.records_from_openfootball(_payload())[:4])
    result = fp.refresh_upcoming_fixtures(
        tmp_path / "calendar.csv", providers=[calendar_provider],
    )
    assert result.status == "ok"
    assert result.schedule_scope == "full_season_calendar"
    assert result.written == 4

    partial = fp.refresh_upcoming_fixtures(
        tmp_path / "partial.csv", providers=[PartialProvider()],
    )
    assert partial.schedule_scope == "partial_provider_window"


def test_provider_environment_supports_a_priority_list(monkeypatch):
    monkeypatch.setenv(fp.FIXTURE_DISCOVERY_ENV, "1")
    monkeypatch.setenv(fp.FIXTURE_PROVIDER_ENV, "openfootball, thesportsdb,OPENFOOTBALL")

    assert fp.configured_provider_names() == ["openfootball", "thesportsdb"]
    providers = fp.providers_from_environment()
    assert [provider.name for provider in providers] == ["openfootball", "thesportsdb"]

    monkeypatch.setenv(fp.FIXTURE_PROVIDER_ENV, "none")
    assert fp.configured_provider_names() == []
    assert fp.providers_from_environment() == []

    monkeypatch.setenv(fp.FIXTURE_PROVIDER_ENV, "none,thesportsdb")
    assert fp.configured_provider_names() == ["thesportsdb"]

    monkeypatch.setenv(fp.FIXTURE_PROVIDER_ENV, "thesportsdb,not-a-provider")
    with pytest.raises(fp.FixtureProviderError, match="Unknown fixture provider"):
        fp.configured_provider_names()

    monkeypatch.setenv(fp.FIXTURE_DISCOVERY_ENV, "0")
    monkeypatch.setenv(fp.FIXTURE_PROVIDER_ENV, "openfootball")
    assert fp.configured_provider_names() == []


def test_default_provider_list_starts_with_the_calendar(monkeypatch):
    monkeypatch.delenv(fp.FIXTURE_PROVIDER_ENV, raising=False)
    monkeypatch.delenv(fp.FIXTURE_DISCOVERY_ENV, raising=False)
    assert fp.configured_provider_names() == ["openfootball", "thesportsdb"]


def test_time_less_calendar_row_is_merged_into_the_timed_provider_row():
    """A date-only calendar row must not duplicate a fixture a provider timed."""
    calendar_row = _row(
        fixture_id=fp.stable_fixture_id("Barcelona", "Villarreal", "2026-11-22"),
        home_team="Barcelona", away_team="Villarreal", provider="openfootball",
        kickoff_utc="2026-11-22T00:00:00Z", match_date="2026-11-22",
    )
    incoming = fp.FixtureRecord(
        home_team="Barcelona", away_team="Villarreal",
        kickoff_utc="2026-11-22T16:00:00Z", provider="thesportsdb",
        provider_event_id="2506289", source="thesportsdb:eventsnextleague",
    )

    merged = fp.merge_fixtures(_store([calendar_row]), [incoming])

    assert len(merged) == 1
    assert merged.iloc[0]["kickoff_utc"] == "2026-11-22T16:00:00Z"
    assert merged.iloc[0]["provider"] == "thesportsdb"
    assert merged.iloc[0]["fixture_id"] == calendar_row["fixture_id"]
    assert coverage_module.detect_fixture_conflicts(merged).empty


def test_genuine_kickoff_time_disagreement_is_preserved():
    existing = _row(
        fixture_id=fp.stable_fixture_id("Barcelona", "Villarreal", "2026-11-22"),
        home_team="Barcelona", away_team="Villarreal", provider="openfootball",
        kickoff_utc="2026-11-22T18:00:00Z", match_date="2026-11-22",
    )
    incoming = fp.FixtureRecord(
        home_team="Barcelona", away_team="Villarreal",
        kickoff_utc="2026-11-22T16:00:00Z", provider="thesportsdb",
        provider_event_id="2506289", source="thesportsdb:eventsnextleague",
    )

    merged = fp.merge_fixtures(_store([existing]), [incoming])

    assert len(merged) == 2
    assert set(merged["kickoff_utc"]) == {"2026-11-22T18:00:00Z", "2026-11-22T16:00:00Z"}
    conflicts = coverage_module.detect_fixture_conflicts(merged)
    assert len(conflicts) == 1
    assert conflicts.iloc[0]["conflict"] == "kickoff_time"


# ---------------------------------------------------------------------------
# Calendar cache and comparison
# ---------------------------------------------------------------------------

def test_calendar_cache_roundtrip(tmp_path):
    path = tmp_path / "season_calendar.json"
    fc.save_calendar(
        _payload(), path=path, source="test:es.1", season="2026-27",
        fetched_at="2026-11-12T00:00:00+00:00",
    )

    frame, meta = fc.load_calendar(path)

    assert len(frame) == 6
    assert meta["source"] == "test:es.1"
    assert meta["season"] == "2026-27"
    assert meta["matches"] == 6
    assert json.loads(path.read_text(encoding="utf-8"))["payload"]["name"].startswith("Spain")


def test_calendar_cache_missing_or_corrupt_is_empty_not_invented(tmp_path):
    frame, meta = fc.load_calendar(tmp_path / "absent.json")
    assert frame.empty and meta == {}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    frame, meta = fc.load_calendar(corrupt)
    assert frame.empty and meta == {}


def test_comparison_reports_missing_rescheduled_and_unexpected():
    calendar = fc.calendar_frame(_payload())
    stored = _store([
        _row(),  # Barcelona vs Real Madrid, matched
        _row(home_team="Ath Bilbao", away_team="Espanol",
             kickoff_utc="2026-11-15T00:00:00Z"),
        _row(home_team="Ath Madrid", away_team="Valencia",
             kickoff_utc="2026-11-23T21:00:00Z", match_date="2026-11-23"),
        _row(home_team="Elche", away_team="Osasuna",
             kickoff_utc="2026-11-20T20:00:00Z", match_date="2026-11-20"),
    ])

    comparison = fc.compare_with_calendar(stored, calendar, now=NOW, horizon_days=14)

    assert comparison["verified"] is True
    assert comparison["calendar_upcoming"] == 4
    assert comparison["stored_upcoming"] == 4
    assert comparison["matched"] == 2
    assert comparison["missing"] == 2
    assert comparison["unexpected"] == 1
    assert comparison["rescheduled"] == 1
    assert comparison["coverage_fraction"] == 0.5
    assert comparison["rescheduled_fixtures"][0]["calendar_date"] == "2026-11-22"
    assert comparison["rescheduled_fixtures"][0]["stored_date"] == "2026-11-23"
    assert {"Sevilla", "Betis"} <= set(comparison["teams_without_fixtures"])
    assert comparison["within_horizon"]["calendar_upcoming"] == 4


def test_comparison_is_complete_when_every_calendar_fixture_is_stored():
    calendar = fc.calendar_frame(_payload())
    fixtures = fp.records_from_openfootball(_payload())
    stored = _store([
        record.as_row() for record in fixtures if record.status != "finished"
    ])

    comparison = fc.compare_with_calendar(stored, calendar, now=NOW)

    assert comparison["missing"] == 0
    assert comparison["unexpected"] == 0
    assert comparison["coverage_fraction"] == 1.0
    assert comparison["teams_without_fixtures"] == []
    assert comparison["teams_with_partial_coverage"] == []


def test_comparison_ignores_past_and_finished_fixtures():
    calendar = fc.calendar_frame(_payload())
    stored = _store([
        _row(home_team="Alaves", away_team="Getafe", status="finished",
             kickoff_utc="2026-08-15T00:00:00Z", match_date="2026-08-15"),
    ])

    comparison = fc.compare_with_calendar(stored, calendar, now=NOW)

    assert comparison["stored_upcoming"] == 0
    assert comparison["missing"] == 4


# ---------------------------------------------------------------------------
# Coverage audit integration
# ---------------------------------------------------------------------------

def test_coverage_audit_stays_unknown_without_a_calendar():
    """A provider window on its own never proves completeness."""
    report, _ = coverage_module.audit_fixture_coverage(
        _raw_frame(), _store([_row()]), now=NOW, full_season_threshold=1,
    )

    assert report["upcoming"]["schedule_completeness"] == "unknown_partial_provider_window"
    assert report["upcoming"]["expected_fixtures"] is None
    assert report["upcoming"]["calendar_comparison"] is None


def test_coverage_audit_marks_verified_completeness_with_a_calendar(tmp_path):
    report, coverage = coverage_module.audit_fixture_coverage(
        _raw_frame(),
        _store([_row()]),
        now=NOW,
        full_season_threshold=1,
        calendar=fc.calendar_frame(_payload()),
        calendar_meta={"source": "test:es.1", "fetched_at": "2026-11-12T00:00:00+00:00"},
    )

    upcoming = report["upcoming"]
    assert upcoming["schedule_completeness"] == "verified_against_calendar"
    assert upcoming["expected_fixtures"] == 4
    assert upcoming["calendar_source"] == "test:es.1"
    assert upcoming["calendar_comparison"]["missing"] == 3
    assert report["status"] == "warning"
    assert any("missing from the store" in warning for warning in report["warnings"])

    paths = coverage_module.write_coverage_reports(
        report, coverage,
        json_path=tmp_path / "coverage.json",
        csv_path=tmp_path / "coverage.csv",
        markdown_path=tmp_path / "coverage.md",
    )
    assert all(path.exists() for path in paths)


def test_coverage_reports_include_the_calendar_section(tmp_path):
    report, coverage = coverage_module.audit_fixture_coverage(
        _raw_frame(), _store([_row()]), now=NOW, full_season_threshold=1,
        calendar=fc.calendar_frame(_payload()),
        calendar_meta={"source": "test:es.1", "fetched_at": "2026-11-12T00:00:00+00:00"},
    )
    paths = coverage_module.write_coverage_reports(
        report,
        coverage,
        json_path=tmp_path / "coverage.json",
        csv_path=tmp_path / "coverage.csv",
        markdown_path=tmp_path / "coverage.md",
    )
    text = (tmp_path / "coverage.md").read_text(encoding="utf-8")
    assert "Season calendar comparison" in text
    assert "Missing fixtures" in text
    payload = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert payload["upcoming"]["calendar_comparison"]["verified"] is True
    assert all(path.exists() for path in paths)


def test_refresh_calendar_caches_the_payload(monkeypatch, tmp_path):
    provider = fp.OpenFootballProvider(season="2026-27")
    monkeypatch.setattr(provider, "fetch_payload", lambda: _payload())

    frame, meta = fc.refresh_calendar(path=tmp_path / "cal.json", provider=provider)

    assert len(frame) == 6
    assert meta["matches"] == 6
    assert (tmp_path / "cal.json").exists()
    assert meta["source"].endswith("/2026-27/es.1.json")
