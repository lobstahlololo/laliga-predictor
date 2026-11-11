"""Phase 6 audits: data quality, fixture coverage and historical sources."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data import data_quality as dq
from src.data import fixture_coverage as fc
from src.data import fixture_provider as fp
from src.data import historical_sources as hs
from src.utils.config import FEATURE_COLUMNS


NOW = datetime(2026, 11, 10, tzinfo=timezone.utc)


def test_data_quality_clean_dataset_has_no_errors(raw_matches):
    thresholds = dq.QualityThresholds(
        min_season_matches=0,
        max_season_matches=10_000,
        stale_days=100_000,
        max_missing_odds_fraction=1.0,
        max_missing_fraction=1.0,
        min_feature_coverage=0.0,
    )
    report, tables = dq.audit_dataset(raw_matches, thresholds=thresholds, now=NOW)
    assert report["status"] == "ok"
    assert report["summary"]["errors"] == 0
    assert report["duplicates"]["duplicate_fixture_rows"] == 0
    assert report["invalid_scores"]["negative"] == 0
    assert report["chronology"]["ordering_violations"] == 0
    assert set(tables) >= {"by_column", "by_season", "features"}
    assert list(tables["by_column"].columns) == [
        "column", "rows", "missing", "missing_fraction"
    ]


def test_data_quality_detects_anomalies_with_actionable_codes(raw_matches):
    broken = raw_matches.head(30).copy()
    first, second = broken.index[0], broken.index[1]
    broken.loc[first, "home_team"] = "Real Betis"  # non-canonical spelling
    broken.loc[first, "full_time_home_goals"] = -1
    broken.loc[first, "odds_home"] = np.nan
    broken.loc[second, "date"] = pd.Timestamp("2099-01-01")
    broken = pd.concat([broken, broken.iloc[[2]]], ignore_index=True)

    feature_frame = pd.DataFrame(
        0.0, index=broken.index, columns=FEATURE_COLUMNS
    )
    report, _tables = dq.audit_dataset(
        broken,
        feature_frame=feature_frame,
        thresholds=dq.QualityThresholds(min_season_matches=0, max_season_matches=10000),
        now=NOW,
    )
    codes = {finding["code"] for finding in report["findings"]}
    assert report["status"] == "error"
    assert report["duplicates"]["duplicate_fixture_rows"] >= 1
    assert report["invalid_scores"]["negative"] >= 1
    assert report["team_names"]["inconsistent_rows"] >= 1
    assert report["missing_odds"]["rows"] >= 1
    assert report["invalid_dates"]["future"] >= 1
    assert {"duplicate_fixtures", "invalid_scores", "team_name_inconsistency",
            "invalid_dates"} <= codes


def test_data_quality_thresholds_are_configurable(raw_matches):
    strict = dq.QualityThresholds(min_season_matches=340)
    report, _ = dq.audit_dataset(raw_matches, thresholds=strict, now=NOW)
    codes = {finding["code"] for finding in report["findings"]}
    assert "season_too_small" in codes
    assert report["status"] == "warning"
    assert report["thresholds"]["min_season_matches"] == 340


def test_data_quality_feature_missingness_is_reported(raw_matches):
    features = pd.DataFrame(
        1.0, index=raw_matches.index, columns=FEATURE_COLUMNS
    )
    features.loc[:, "implied_prob_home"] = np.nan
    report, tables = dq.audit_dataset(
        raw_matches,
        feature_frame=features,
        thresholds=dq.QualityThresholds(min_season_matches=0, max_season_matches=10000),
        now=NOW,
    )
    assert "implied_prob_home" in report["features"]["features_over_threshold"]
    assert any(
        finding["code"] == "feature_missingness" for finding in report["findings"]
    )
    assert not tables["features"].empty


def test_data_quality_writes_strict_json_csv_and_markdown(tmp_path, raw_matches):
    report, tables = dq.audit_dataset(
        raw_matches,
        thresholds=dq.QualityThresholds(min_season_matches=0, max_season_matches=10000),
        now=NOW,
    )
    paths = dq.write_quality_reports(
        report,
        tables,
        json_path=tmp_path / "quality.json",
        columns_path=tmp_path / "columns.csv",
        seasons_path=tmp_path / "seasons.csv",
        markdown_path=tmp_path / "quality.md",
    )
    assert all(path.exists() for path in paths)
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["rows"] == len(raw_matches)
    assert "NaN" not in (tmp_path / "quality.json").read_text(encoding="utf-8")
    assert "Historical Data Quality" in (tmp_path / "quality.md").read_text(encoding="utf-8")


def _fixture_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "fixture_id": "fx_a", "provider": "thesportsdb", "provider_event_id": "1",
            "home_team": "Real Madrid", "away_team": "Barcelona",
            "kickoff_utc": "2026-11-20T18:00:00Z", "match_date": "2026-11-20",
            "status": "scheduled", "source": "test", "fetched_at": "2026-11-09T00:00:00Z",
        },
        {
            "fixture_id": "fx_a", "provider": "api-football", "provider_event_id": "9001",
            "home_team": "Real Madrid", "away_team": "Barcelona",
            "kickoff_utc": "2026-11-20T21:00:00Z", "match_date": "2026-11-20",
            "status": "scheduled", "source": "test", "fetched_at": "2026-11-09T00:00:00Z",
        },
        {
            "fixture_id": "", "provider": "thesportsdb", "provider_event_id": "2",
            "home_team": "Betis", "away_team": "Valencia",
            "kickoff_utc": "", "match_date": "",
            "status": "postponed", "source": "test", "fetched_at": "2026-11-09T00:00:00Z",
        },
    ], columns=fp.FIXTURE_COLUMNS)


def test_fixture_coverage_detects_missing_team_duplicates_and_conflicts(tmp_path, raw_matches):
    latest = raw_matches[raw_matches["season"] == "26/27"]
    latest = latest[~latest["home_team"].eq("Barcelona") & ~latest["away_team"].eq("Barcelona")]
    raw = pd.concat(
        [raw_matches[raw_matches["season"] != "26/27"], latest], ignore_index=True
    )
    from pathlib import Path

    telemetry = fc.ProviderTelemetry(Path(tmp_path) / "provider_status.json")
    telemetry.record("thesportsdb", "ok", fetched=2)
    report, coverage = fc.audit_fixture_coverage(
        raw,
        _fixture_rows(),
        now=NOW,
        telemetry=telemetry,
        full_season_threshold=25,
    )
    assert report["historical"]["missing_teams"] == ["Barcelona"]
    assert report["upcoming"]["duplicate_fixture_ids"] >= 1
    assert report["upcoming"]["conflicting_fixtures"] >= 1
    assert report["upcoming"]["missing_dates"] == 1
    assert report["upcoming"]["schedule_completeness"] == "unknown_partial_provider_window"
    assert report["upcoming"]["expected_fixtures"] is None
    assert report["provider_telemetry"]["providers"]["thesportsdb"]["last_status"] == "ok"
    assert any("incomplete" in warning or "not proof" in warning for warning in report["warnings"])
    assert {"season", "team", "matches", "upcoming_fixtures"} <= set(coverage.columns)
    assert coverage["has_upcoming_fixture"].dtype == bool


def test_fixture_coverage_writes_reports(tmp_path, raw_matches):
    report, coverage = fc.audit_fixture_coverage(
        raw_matches, _fixture_rows(), now=NOW, full_season_threshold=25
    )
    paths = fc.write_coverage_reports(
        report,
        coverage,
        json_path=tmp_path / "coverage.json",
        csv_path=tmp_path / "coverage.csv",
        markdown_path=tmp_path / "coverage.md",
    )
    assert all(path.exists() for path in paths)
    payload = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert payload["upcoming"]["n"] == 3
    assert "Fixture Coverage" in (tmp_path / "coverage.md").read_text(encoding="utf-8")


def test_provider_telemetry_tracks_freshness_and_consecutive_failures(tmp_path):
    telemetry = fc.ProviderTelemetry(tmp_path / "status.json")
    telemetry.record("thesportsdb", "ok", fetched=4)
    telemetry.record("thesportsdb", "failed", errors=["HTTP 429"])
    telemetry.record("thesportsdb", "failed", errors=["timeout"])
    summary = telemetry.summary()
    entry = summary["providers"]["thesportsdb"]
    assert entry["total_requests"] == 3
    assert entry["total_failures"] == 2
    assert entry["consecutive_failures"] == 2
    assert entry["last_success_at"] is not None
    assert entry["last_fixture_count"] == 4
    assert len(summary["recent_attempts"]) == 3

    telemetry.record("thesportsdb", "ok", fetched=2)
    entry = telemetry.summary()["providers"]["thesportsdb"]
    assert entry["consecutive_failures"] == 0
    assert entry["last_fixture_count"] == 2


def test_multi_provider_merge_preserves_ids_and_keeps_conflicting_kickoffs():
    from src.data.fixture_provider import FixtureRecord, merge_provider_batches

    first = FixtureRecord(
        home_team="Real Madrid", away_team="Barcelona",
        kickoff_utc="2026-11-20T18:00:00Z", provider="thesportsdb",
        provider_event_id="1",
    )
    existing = fp.merge_fixtures(pd.DataFrame(columns=fp.FIXTURE_COLUMNS), [first])
    stable_id = existing.iloc[0]["fixture_id"]

    same_time = FixtureRecord(
        home_team="Real Betis", away_team="Barcelona",  # different fixture
        kickoff_utc="2026-11-21T18:00:00Z", provider="api-football",
        provider_event_id="9002",
    )
    # A second provider returning the same fixture with an empty kickoff must
    # not erase a known kickoff or change the stable ID.
    incomplete = FixtureRecord(
        home_team="Real Madrid", away_team="Barcelona",
        kickoff_utc="", provider="api-football", provider_event_id="9001",
    )
    merged = merge_provider_batches(existing, {"api-football": [incomplete, same_time]})
    real_madrid = merged[
        merged["home_team"].eq("Real Madrid") & merged["away_team"].eq("Barcelona")
    ]
    assert len(real_madrid) == 1
    assert real_madrid.iloc[0]["fixture_id"] == stable_id
    assert real_madrid.iloc[0]["kickoff_utc"] == "2026-11-20T18:00:00Z"

    conflicting = FixtureRecord(
        home_team="Real Madrid", away_team="Barcelona",
        kickoff_utc="2026-11-20T21:00:00Z", provider="api-football",
        provider_event_id="9001",
    )
    with_conflict = merge_provider_batches(existing, {"api-football": [conflicting]})
    same_fixture = with_conflict[with_conflict["fixture_id"] == stable_id]
    assert len(same_fixture) == 2
    conflicts = fc.detect_fixture_conflicts(with_conflict)
    assert not conflicts.empty
    assert set(conflicts["conflict"]) == {"kickoff_time"}


def test_fixture_status_lifecycle_postponed_rescheduled_cancelled_completed():
    from src.data.fixture_provider import FixtureRecord

    def record(status: str, kickoff: str = "2026-11-20T18:00:00Z") -> FixtureRecord:
        return FixtureRecord(
            home_team="Real Madrid", away_team="Barcelona",
            kickoff_utc=kickoff, provider="thesportsdb",
            provider_event_id="lifecycle", status=status,
        )

    frame = fp.merge_fixtures(pd.DataFrame(columns=fp.FIXTURE_COLUMNS), [record("scheduled")])
    stable_id = frame.iloc[0]["fixture_id"]

    postponed = fp.merge_fixtures(frame, [record("postponed")])
    assert postponed.iloc[0]["status"] == "postponed"

    rescheduled = fp.merge_fixtures(
        postponed, [record("scheduled", "2026-11-22T20:00:00Z")]
    )
    assert len(rescheduled) == 1
    assert rescheduled.iloc[0]["fixture_id"] == stable_id
    assert rescheduled.iloc[0]["kickoff_utc"] == "2026-11-22T20:00:00Z"

    cancelled = fp.merge_fixtures(
        rescheduled, [record("cancelled", "2026-11-22T20:00:00Z")]
    )
    assert cancelled.iloc[0]["status"] == "cancelled"

    # A completed event is history and must not enter the upcoming store; the
    # existing stored row is not silently deleted or rewritten.
    completed = fp.merge_fixtures(
        cancelled, [record("finished", "2026-11-22T20:00:00Z")]
    )
    assert len(completed) == 1
    assert completed.iloc[0]["status"] == "cancelled"
    assert completed.iloc[0]["fixture_id"] == stable_id



def test_partial_provider_failure_keeps_the_successful_batch(tmp_path):
    from src.data.fixture_provider import FixtureRecord

    class Stub:
        def __init__(self, name, payload):
            self.name = name
            self.payload = payload

        def fetch_upcoming(self):
            if isinstance(self.payload, Exception):
                raise self.payload
            return self.payload

    good = Stub("thesportsdb", [FixtureRecord(
        home_team="Real Madrid", away_team="Barcelona",
        kickoff_utc="2026-11-20T18:00:00Z", provider="thesportsdb",
        provider_event_id="1",
    )])
    bad = Stub("api-football", RuntimeError("quota exceeded"))
    path = tmp_path / "fixtures.csv"
    result = fp.refresh_upcoming_fixtures(
        path, providers=[good, bad], status_path=tmp_path / "status.json"
    )
    assert result.status == "partial"
    assert result.fetched == 1
    assert result.failed_providers == ["api-football"]
    assert len(pd.read_csv(path)) == 1
    telemetry = fc.ProviderTelemetry(tmp_path / "status.json").summary()
    assert telemetry["providers"]["api-football"]["total_failures"] == 1
    assert telemetry["providers"]["thesportsdb"]["last_status"] == "ok"


def test_historical_sources_without_extracts_are_reported_unavailable(tmp_path, raw_matches):
    report, tables = hs.audit_historical_sources(
        raw_matches,
        xg_path=tmp_path / "missing_xg.csv",
        availability_path=tmp_path / "missing_availability.csv",
        now=NOW,
    )
    assert report["network_requests"] == 0
    assert report["production_model_unchanged"] is True
    assert report["coverage"]["xg"]["available"] is False
    assert report["coverage"]["player_availability"]["available"] is False
    assert {finding["kind"] for finding in report["findings"]} == {
        "xg", "player_availability"
    }
    assert tables["xg_coverage"].empty
    assert tables["availability_coverage"].empty


def test_historical_xg_coverage_is_measured_and_time_aware(tmp_path, raw_matches):
    raw = raw_matches[raw_matches["season"] == "23/24"].copy()
    first = raw.iloc[0]
    before = first["date"] - pd.Timedelta(days=1)
    xg = pd.DataFrame([
        {
            "match_id": "prior", "team": first["home_team"], "date": f"{before:%Y-%m-%d}",
            "available_at": f"{before:%Y-%m-%d}T22:00:00Z",
            "xg_for": 1.4, "xg_against": 0.9,
        },
        {
            "match_id": "prior", "team": first["away_team"], "date": f"{before:%Y-%m-%d}",
            "available_at": f"{before:%Y-%m-%d}T22:00:00Z",
            "xg_for": 1.1, "xg_against": 1.2,
        },
    ])
    xg_path = tmp_path / "xg.csv"
    xg.to_csv(xg_path, index=False)
    report, tables = hs.audit_historical_sources(
        raw,
        xg_path=xg_path,
        availability_path=tmp_path / "missing.csv",
        now=NOW,
    )
    assert report["coverage"]["xg"]["available"] is True
    assert report["coverage"]["xg"]["matches_with_pre_kickoff_source"] >= 2
    assert report["coverage"]["player_availability"]["available"] is False
    assert not tables["xg_coverage"].empty
    assert tables["xg_coverage"]["coverage_fraction"].max() > 0

    paths = hs.write_historical_source_report(
        report, tables,
        json_path=tmp_path / "sources.json",
        markdown_path=tmp_path / "sources.md",
    )
    assert all(path.exists() for path in paths)
    payload = json.loads((tmp_path / "sources.json").read_text(encoding="utf-8"))
    assert payload["coverage"]["xg"]["available"] is True
