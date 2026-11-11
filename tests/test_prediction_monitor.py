"""Prediction monitoring: counts, sample-size guards, errors and drift."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.fixture_provider import FIXTURE_COLUMNS
from src.monitoring.prediction_monitor import (
    monitor_predictions,
    write_prediction_monitoring_report,
)
from src.predictions.history import build_prediction_record, reconcile_results

NOW = datetime(2027, 1, 5, tzinfo=timezone.utc)


def _settled_history(raw_matches, n: int):
    records = []
    for row in raw_matches.head(n).itertuples(index=False):
        match_date = f"{row.date:%Y-%m-%d}"
        records.append(build_prediction_record(
            home_team=row.home_team,
            away_team=row.away_team,
            match_date=match_date,
            predicted_at=match_date,
            model_name="Random Forest",
            model_version="test-v1",
            prob_home=0.50,
            prob_draw=0.30,
            prob_away=0.20,
        ))
    history = pd.DataFrame(records)
    settled, stats = reconcile_results(
        history, raw_matches, as_of=pd.Timestamp("2027-01-01")
    )
    assert stats["settled"] == n
    return settled


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame([{
        "fixture_id": "fx_unpredicted",
        "provider": "thesportsdb",
        "provider_event_id": "99",
        "home_team": "Real Madrid",
        "away_team": "Barcelona",
        "kickoff_utc": "2027-02-01T18:00:00Z",
        "match_date": "2027-02-01",
        "status": "scheduled",
        "source": "test",
        "fetched_at": "2027-01-04T00:00:00Z",
    }], columns=FIXTURE_COLUMNS)


COVERAGE = {
    "status": "ok",
    "upcoming": {"n": 1, "schedule_completeness": "unknown_partial_provider_window"},
    "warnings": [],
}


def test_insufficient_settled_sample_is_explicit(raw_matches):
    history = _settled_history(raw_matches, 5)
    report = monitor_predictions(
        history,
        fixtures=_fixtures(),
        raw=raw_matches,
        now=NOW,
        min_settled=30,
        coverage=COVERAGE,
    )
    assert report["live_performance_reliable"] is False
    assert report["conclusions_allowed"] is False
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["settled_performance"]["status"] == "insufficient_data"
    assert report["metrics"].get("reliable") is False
    assert report["counts"]["settled_rows"] == 5


def test_reliable_metrics_counts_and_missing_predictions(raw_matches):
    history = _settled_history(raw_matches, 40)
    report = monitor_predictions(
        history,
        fixtures=_fixtures(),
        raw=raw_matches,
        now=NOW,
        min_settled=30,
        coverage=COVERAGE,
    )
    assert report["live_performance_reliable"] is True
    assert report["conclusions_allowed"] is True
    assert report["counts"]["settled_rows"] == 40
    assert report["counts"]["missing_predictions"] == 1
    assert report["counts"]["invalid_probability_rows"] == 0
    assert np.isfinite(report["metrics"]["log_loss"])
    assert np.isfinite(report["metrics"]["brier_score"])
    assert 0 <= report["metrics"]["accuracy"] <= 1
    assert report["per_class"]
    checks = {check["name"]: check["status"] for check in report["checks"]}
    assert checks["settled_performance"] in {"ok", "warning"}
    assert checks["missing_predictions"] == "warning"


def test_invalid_probabilities_are_prediction_errors(raw_matches):
    history = _settled_history(raw_matches, 10)
    history.loc[history.index[0], "model_prob_home"] = "-0.5"
    report = monitor_predictions(
        history,
        fixtures=_fixtures(),
        raw=raw_matches,
        now=NOW,
        min_settled=30,
        coverage=COVERAGE,
    )
    assert report["counts"]["invalid_probability_rows"] >= 1
    assert report["overall_status"] == "error"
    checks = {check["name"]: check["status"] for check in report["checks"]}
    assert checks["prediction_errors"] == "error"


def test_monitoring_report_is_strict_json(tmp_path, raw_matches):
    history = _settled_history(raw_matches, 10)
    report = monitor_predictions(
        history,
        fixtures=_fixtures(),
        raw=raw_matches,
        now=NOW,
        min_settled=30,
        coverage=COVERAGE,
    )
    paths = write_prediction_monitoring_report(
        report,
        json_path=tmp_path / "monitor.json",
        markdown_path=tmp_path / "monitor.md",
    )
    assert all(path.exists() for path in paths)
    payload = json.loads((tmp_path / "monitor.json").read_text(encoding="utf-8"))
    assert payload["minimum_settled_sample"] == 30
    assert "NaN" not in (tmp_path / "monitor.json").read_text(encoding="utf-8")
    assert "Prediction Monitoring" in (tmp_path / "monitor.md").read_text(encoding="utf-8")
