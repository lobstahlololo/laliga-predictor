"""Automated update cycle for the La Liga predictor.

One cycle performs the repeated operational work for a season in progress:

1. ``data`` — refresh played results from Football-Data, validate, write atomically.
2. ``predict`` — score the configured upcoming fixtures and append new ledger rows.
3. ``backfill`` — reconcile settled results without overwriting history.
4. ``reports`` — regenerate ledger performance reports.
5. ``monitoring`` — refresh drift/calibration/performance diagnostics.

Everything is idempotent: re-running a cycle refreshes data, de-duplicates
predictions by fixture/timestamp/version and only fills blank results. Steps are
retried on failure, logged to ``logs/pipeline.log`` and summarised in
``reports/pipeline_runs.jsonl`` (append-only). If ``LALIGA_ALERT_WEBHOOK`` is
set, a failed cycle is posted there; without it the failure is logged only.

No fixture, result or statistic is ever invented: an empty fixture list skips
prediction, and reconciliation only uses played matches.

Usage:
    python -m src.pipeline.automation
    python -m src.pipeline.automation --steps data,backfill,monitoring
    python -m src.pipeline.automation --retries 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_model_frame  # noqa: E402
from src.predictions import history as hist  # noqa: E402
from src.predictions.fixtures import load_upcoming_fixtures  # noqa: E402
from src.predictions.predictor import (  # noqa: E402
    load_artifacts,
    load_raw,
    predict_fixtures,
    predictions_to_records,
)
from src.predictions.reports import write_reports  # noqa: E402
from src.utils.config import (  # noqa: E402
    DATA_QUALITY_PATH,
    MODEL_METADATA_PATH,
    PIPELINE_LOG_PATH,
    PIPELINE_RUNS_PATH,
    PROVIDER_STATUS_PATH,
    RAW_PATH,
)

logger = logging.getLogger(__name__)

STEP_ORDER = ("data", "quality", "coverage", "predict", "backfill", "reports", "monitoring")
ALERT_WEBHOOK_ENV = "LALIGA_ALERT_WEBHOOK"
ALERT_TIMEOUT_ENV = "LALIGA_ALERT_TIMEOUT"
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


# ---------------------------------------------------------------------------
# Logging / persistence
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """Attach one console handler and one rotating file handler (idempotent)."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            PIPELINE_LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        ))
        root.addHandler(file_handler)
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        for h in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
        ))
        root.addHandler(console)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StepResult:
    name: str
    status: str
    detail: dict = field(default_factory=dict)
    attempts: int = 0
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def append_run_record(record: dict, path: Path = PIPELINE_RUNS_PATH) -> None:
    """Append one cycle summary as a JSON line (never rewrites old runs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def load_runs(limit: int | None = 20, path: Path = PIPELINE_RUNS_PATH) -> list[dict]:
    """Read recent cycle records, newest last."""
    if not path.exists():
        return []
    runs = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    except Exception as exc:  # noqa: BLE001 - a bad log must not hide newer runs
        logger.warning("Could not parse pipeline run log: %s", exc)
    return runs[-limit:] if limit else runs


def notify_failure(run: dict) -> bool:
    """Post a failure summary to ``LALIGA_ALERT_WEBHOOK`` when configured.

    Returns True when a notification was delivered. Missing configuration is
    not an error: the failure remains in the log and run record.
    """
    webhook = os.environ.get(ALERT_WEBHOOK_ENV, "").strip()
    if not webhook:
        logger.info(
            "%s is not set; failure recorded locally only.", ALERT_WEBHOOK_ENV
        )
        return False
    failed = [s["name"] for s in run.get("steps", []) if s.get("status") == STATUS_FAILED]
    message = (
        f"La Liga pipeline run {run.get('run_id')} failed at "
        f"{', '.join(failed) or 'an unknown step'}."
    )
    payload = json.dumps({"text": message, "run": run}).encode("utf-8")
    timeout = float(os.environ.get(ALERT_TIMEOUT_ENV, "10"))
    request = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            response.read()
        logger.info("Failure notification delivered.")
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("Could not deliver failure notification: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_data() -> dict:
    """Refresh, validate and atomically persist the match dataset."""
    from src.data.make_dataset import (
        build_master_dataset,
        write_atomic,
        write_quality_report,
    )
    from src.data.validation import validate_matches

    master, quality = build_master_dataset()
    if master.empty:
        raise RuntimeError("Downloaded+cleaned dataset is empty; refusing to write.")
    validation = validate_matches(master)
    if validation["duplicate_fixture_ids"]:
        raise RuntimeError("Refreshed dataset contains duplicate fixture IDs.")
    if validation["invalid_dates"]:
        raise RuntimeError("Refreshed dataset contains invalid dates.")

    write_atomic(master, RAW_PATH)
    write_quality_report(quality, DATA_QUALITY_PATH)
    logger.info("Data refreshed: %d matches", len(master))

    # Automated fixture discovery is best-effort: a provider outage or rate
    # limit keeps the previously stored fixtures and never fails the data step.
    try:
        from src.data.fixture_provider import refresh_upcoming_fixtures

        fixture_status = refresh_upcoming_fixtures().as_dict()
    except Exception as exc:  # noqa: BLE001 - fixtures are a soft dependency
        logger.warning("Fixture discovery skipped: %s", exc)
        fixture_status = {"status": "error", "errors": [str(exc)]}

    # Provider telemetry is opt-in for scheduled/operations runs. Ordinary
    # unit tests and local cycles do not write operational state unless asked.
    if os.environ.get("LALIGA_FIXTURE_TELEMETRY", "0").strip() == "1":
        try:
            from src.data.fixture_coverage import ProviderTelemetry

            ProviderTelemetry(PROVIDER_STATUS_PATH).record(
                str(fixture_status.get("provider") or "unknown"),
                str(fixture_status.get("status", "unknown")),
                fetched=int(fixture_status.get("fetched", 0) or 0),
                written=int(fixture_status.get("written", 0) or 0),
                errors=fixture_status.get("errors", []),
            )
        except Exception as exc:  # noqa: BLE001 - telemetry is best effort
            logger.warning("Could not record fixture provider telemetry: %s", exc)

    return {
        "rows": int(len(master)),
        "seasons_fetched": quality.get("seasons_fetched", []),
        "seasons_failed": quality.get("seasons_failed", []),
        "rows_removed": quality.get("rows_removed", 0),
        "date_max": quality.get("date_max"),
        "fixtures": fixture_status,
    }


def step_predict(fixture_path: str | None = None) -> dict:
    """Score upcoming fixtures and append new snapshots to the ledger."""
    if fixture_path:
        from src.predictions.fixtures import read_fixture_file

        fixtures = read_fixture_file(fixture_path)
        source = fixture_path
    else:
        fixtures, source = load_upcoming_fixtures()
    if not fixtures:
        return {"predictions": 0, "source": source, "note": "no upcoming fixtures"}

    model, preprocessor, metadata = load_artifacts()
    raw = load_raw()
    predictions, errors = predict_fixtures(
        fixtures, model=model, preprocessor=preprocessor, metadata=metadata, raw=raw,
    )
    records = predictions_to_records(predictions)
    appended, _ = hist.append_predictions(records)
    return {
        "predictions": len(predictions),
        "appended": appended,
        "errors": len(errors),
        "source": source,
        "model_version": metadata.get("model_version"),
    }


def step_backfill() -> dict:
    """Fill blank results from played matches, leaving settled rows untouched."""
    if not RAW_PATH.exists():
        raise RuntimeError(f"Raw dataset not found at {RAW_PATH}.")
    raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
    _, stats = hist.update_results(raw)
    return stats


def step_reports() -> dict:
    history, _ = hist.load_history()
    if history.empty:
        return {"files": 0, "note": "ledger is empty"}
    written = write_reports(history)
    return {"files": len(written)}


def step_quality() -> dict:
    """Audit historical data quality and write JSON/CSV/Markdown reports."""
    from src.data.data_quality import audit_dataset, write_quality_reports

    report, tables = audit_dataset()
    paths = write_quality_reports(report, tables)
    return {
        "status": report["status"],
        "errors": report["summary"]["errors"],
        "warnings": report["summary"]["warnings"],
        "reports": len(paths),
    }


def step_coverage() -> dict:
    """Regenerate historical/upcoming fixture coverage reports."""
    from src.data.fixture_coverage import audit_fixture_coverage, write_coverage_reports

    report, coverage = audit_fixture_coverage()
    paths = write_coverage_reports(report, coverage)
    return {
        "status": report["status"],
        "upcoming_fixtures": report["upcoming"]["n"],
        "warnings": len(report["warnings"]),
        "reports": len(paths),
    }


def step_monitoring() -> dict:
    from src.monitoring.diagnostics import run_diagnostics, save_monitoring_report
    from src.monitoring.prediction_monitor import (
        monitor_predictions,
        write_prediction_monitoring_report,
    )

    raw = load_raw()
    history, _ = hist.load_history()
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    try:
        feature_frame, _y, _meta = build_model_frame(raw)
    except Exception as exc:  # noqa: BLE001 - drift check degrades, cycle continues
        logger.warning("Could not build features for monitoring: %s", exc)
        feature_frame = None
    report = run_diagnostics(raw, history, metadata, feature_frame=feature_frame)
    save_monitoring_report(report)

    prediction_report = monitor_predictions(history, raw=raw)
    write_prediction_monitoring_report(prediction_report)
    return {
        "overall_status": report["overall_status"],
        "checks": {c["name"]: c["status"] for c in report["checks"]},
        "prediction_monitoring": prediction_report["overall_status"],
        "live_performance_reliable": prediction_report["live_performance_reliable"],
    }


STEP_FUNCTIONS = {
    "data": step_data,
    "quality": step_quality,
    "coverage": step_coverage,
    "predict": step_predict,
    "backfill": step_backfill,
    "reports": step_reports,
    "monitoring": step_monitoring,
}


def _run_step(name: str, function, retries: int) -> StepResult:
    attempt = 0
    started = time.perf_counter()
    last_error = ""
    while attempt <= retries:
        attempt += 1
        try:
            detail = function()
            return StepResult(
                name=name,
                status=STATUS_SKIPPED if detail.get("note") else STATUS_OK,
                detail=detail,
                attempts=attempt,
                duration_seconds=round(time.perf_counter() - started, 2),
            )
        except Exception as exc:  # noqa: BLE001 - recorded and retried
            last_error = str(exc)
            logger.exception("Step %s failed (attempt %d): %s", name, attempt, exc)
            if attempt <= retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    return StepResult(
        name=name,
        status=STATUS_FAILED,
        detail={"error": last_error},
        attempts=attempt,
        duration_seconds=round(time.perf_counter() - started, 2),
    )


def run_cycle(
    steps: list[str] | None = None,
    *,
    retries: int = 1,
    fixture_path: str | None = None,
    notify: bool = True,
    runs_path: Path = PIPELINE_RUNS_PATH,
) -> dict:
    """Run one idempotent update cycle and record the outcome."""
    selected = list(steps or STEP_ORDER)
    unknown = [s for s in selected if s not in STEP_FUNCTIONS]
    if unknown:
        raise ValueError(f"Unknown pipeline steps: {unknown}")

    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    started_at = _now()
    logger.info("Starting pipeline run %s: %s", run_id, ", ".join(selected))

    results: list[StepResult] = []
    for name in selected:
        function = STEP_FUNCTIONS[name]
        if name == "predict":
            result = _run_step(name, lambda: function(fixture_path), retries)
        else:
            result = _run_step(name, function, retries)
        results.append(result)
        logger.info(
            "Step %s: %s (%.2fs, %d attempt(s))",
            name, result.status, result.duration_seconds, result.attempts,
        )
        if result.status == STATUS_FAILED:
            break

    failed = [r.name for r in results if r.status == STATUS_FAILED]
    run = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": _now(),
        "status": STATUS_FAILED if failed else STATUS_OK,
        "steps_requested": selected,
        "steps": [r.as_dict() for r in results],
        "failed_steps": failed,
    }
    if failed and notify:
        run["notified"] = notify_failure(run)
    append_run_record(run, runs_path)
    logger.info("Pipeline run %s finished: %s", run_id, run["status"])
    return run


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="La Liga automated update cycle")
    parser.add_argument(
        "--steps",
        default=",".join(STEP_ORDER),
        help=f"Comma-separated steps from {', '.join(STEP_ORDER)}",
    )
    parser.add_argument("--retries", type=int, default=1, help="Extra attempts per step")
    parser.add_argument("--fixtures", default=None, help="Optional fixture file path")
    parser.add_argument("--no-notify", action="store_true", help="Skip webhook alerts")
    args = parser.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    run = run_cycle(
        steps, retries=max(args.retries, 0), fixture_path=args.fixtures,
        notify=not args.no_notify,
    )
    print(json.dumps(run, indent=2, default=str))
    return 0 if run["status"] == STATUS_OK else 1


if __name__ == "__main__":
    sys.exit(main())
