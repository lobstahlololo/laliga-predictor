"""Update-cycle retries, run log persistence and failure notification."""

from __future__ import annotations

import pytest

from src.pipeline import automation


def test_step_retries_until_success_and_records_attempts():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary failure")
        return {"value": 1}

    result = automation._run_step("monitoring", flaky, retries=2)
    assert result.status == automation.STATUS_OK
    assert result.attempts == 3
    assert calls["n"] == 3


def test_step_fails_after_retry_budget():
    def always_fails():
        raise RuntimeError("permanent failure")

    result = automation._run_step("data", always_fails, retries=1)
    assert result.status == automation.STATUS_FAILED
    assert result.attempts == 2
    assert "permanent failure" in result.detail["error"]


def test_run_cycle_is_logged_and_stops_after_failure(tmp_path, monkeypatch):
    calls = []

    def ok_step():
        calls.append("ok")
        return {"rows": 1}

    def bad_step():
        calls.append("bad")
        raise RuntimeError("boom")

    def never_called():
        calls.append("never")
        return {}

    monkeypatch.setitem(automation.STEP_FUNCTIONS, "reports", ok_step)
    monkeypatch.setitem(automation.STEP_FUNCTIONS, "backfill", bad_step)
    monkeypatch.setitem(automation.STEP_FUNCTIONS, "monitoring", never_called)

    runs_path = tmp_path / "runs.jsonl"
    run = automation.run_cycle(
        ["reports", "backfill", "monitoring"],
        retries=0,
        notify=False,
        runs_path=runs_path,
    )

    assert run["status"] == automation.STATUS_FAILED
    assert run["failed_steps"] == ["backfill"]
    assert calls == ["ok", "bad"]  # later steps do not run
    stored = automation.load_runs(path=runs_path)
    assert len(stored) == 1
    assert stored[0]["run_id"] == run["run_id"]
    # The run log is append-only.
    automation.append_run_record(run, runs_path)
    assert len(automation.load_runs(path=runs_path)) == 2


def test_notification_is_optional(monkeypatch):
    monkeypatch.delenv(automation.ALERT_WEBHOOK_ENV, raising=False)
    run = {"run_id": "test", "steps": [{"name": "data", "status": "failed"}]}
    assert automation.notify_failure(run) is False


def test_unknown_step_is_rejected():
    with pytest.raises(ValueError, match="Unknown pipeline steps"):
        automation.run_cycle(["not-a-step"], notify=False)
