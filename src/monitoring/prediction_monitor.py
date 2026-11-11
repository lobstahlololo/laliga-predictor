"""Production prediction monitoring with explicit sample-size guards.

This monitor tracks the operational health of the prediction ledger and the
data feeding it:

* prediction, settled, pending and missing-prediction counts;
* invalid probability rows and overdue settlements (prediction errors);
* log-loss, Brier score, accuracy, calibration and per-outcome metrics;
* drift between an earlier and a more recent settled window;
* data and fixture freshness plus fixture coverage.

Every performance statement is gated by a minimum settled sample. Below that
threshold the metrics are still reported as arithmetic, but are explicitly
marked ``reliable=False`` and cannot be read as conclusions. No check here
retrains, promotes or rewrites a model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.fixture_coverage import ProviderTelemetry, audit_fixture_coverage
from src.data.fixture_provider import load_fixture_rows
from src.evaluation.baselines import BASELINE_LABELS
from src.evaluation.metrics import evaluate_probabilities, metrics_to_row
from src.predictions.history import add_derived_columns
from src.utils.config import (
    LEDGER_PATH,
    MONITORING_CONFIRM_SETTLED,
    MONITORING_MIN_SETTLED,
    PREDICTION_MONITORING_MARKDOWN_PATH,
    PREDICTION_MONITORING_PATH,
    PROVIDER_STATUS_PATH,
    RAW_PATH,
    STALE_DATA_DAYS,
    UPCOMING_FIXTURES_PATH,
)
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

OK = "ok"
WARNING = "warning"
CONFIRMED = "confirmed"
INSUFFICIENT = "insufficient_data"
ERROR = "error"
PROB_COLUMNS = ["model_prob_home", "model_prob_draw", "model_prob_away"]
TERMINAL_STATUSES = {"finished", "cancelled", "canceled", "abandoned"}
EPS = 1e-12


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if not isinstance(value, (str, bytes, dict, list)) and pd.isna(value):
        return None
    return value


def _check(name: str, status: str, message: str, **details: Any) -> dict:
    return {"name": name, "status": status, "message": message, "details": _json_safe(details)}


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 3))
    return frame[PROB_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _valid_probability_mask(probabilities: np.ndarray) -> np.ndarray:
    if len(probabilities) == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(probabilities).all(axis=1)
    non_negative = (probabilities >= 0).all(axis=1)
    positive_sum = probabilities.sum(axis=1) > 0
    return finite & non_negative & positive_sum


def _settled(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "actual_label" not in frame:
        return frame.iloc[0:0].copy()
    return frame[frame["actual_label"].astype(str).str.strip() != ""].copy()


def _metrics(y, probabilities: np.ndarray, label: str) -> dict:
    return evaluate_probabilities(y, probabilities, BASELINE_LABELS, label)


def _per_row_log_loss(y, probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(probabilities, EPS, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    labels = np.asarray(y, dtype=int)
    index = {label: j for j, label in enumerate(BASELINE_LABELS)}
    true_prob = probs[np.arange(len(labels)), [index[int(v)] for v in labels]]
    return -np.log(np.clip(true_prob, EPS, 1.0))


def _paired_ci(earlier: np.ndarray, recent: np.ndarray, *, draws: int = 1000, seed: int = 42) -> dict:
    if len(earlier) < 5 or len(recent) < 5:
        return {"delta": None, "ci_low": None, "ci_high": None, "ci_excludes_zero": None}
    delta = float(np.mean(recent) - np.mean(earlier))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for i in range(draws):
        boot[i] = (
            rng.choice(recent, size=len(recent), replace=True).mean()
            - rng.choice(earlier, size=len(earlier), replace=True).mean()
        )
    low, high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    return {
        "delta": delta,
        "ci_low": low,
        "ci_high": high,
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def _ledger_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    if frame.empty:
        return set()
    keys = set()
    for row in frame.itertuples(index=False):
        date = ""
        for candidate in (getattr(row, "match_date", ""),):
            parsed = pd.to_datetime(candidate, errors="coerce")
            if pd.notna(parsed):
                date = f"{parsed:%Y-%m-%d}"
        keys.add((
            canonical_team(getattr(row, "home_team", "")),
            canonical_team(getattr(row, "away_team", "")),
            date,
        ))
    return keys


def _fixture_keys(fixtures: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    keys = []
    for row in fixtures.itertuples(index=False):
        date = str(getattr(row, "match_date", "") or "").strip()
        if not date:
            kickoff = pd.to_datetime(getattr(row, "kickoff_utc", ""), errors="coerce")
            if pd.notna(kickoff):
                date = f"{kickoff:%Y-%m-%d}"
        parsed = pd.to_datetime(date, errors="coerce")
        normalized = f"{parsed:%Y-%m-%d}" if pd.notna(parsed) else ""
        keys.append((
            canonical_team(getattr(row, "home_team", "")),
            canonical_team(getattr(row, "away_team", "")),
            normalized,
            str(getattr(row, "status", "") or "").lower(),
        ))
    return keys


def monitor_predictions(
    history: pd.DataFrame | None = None,
    *,
    fixtures: pd.DataFrame | None = None,
    raw: pd.DataFrame | None = None,
    now: datetime | None = None,
    min_settled: int = MONITORING_MIN_SETTLED,
    confirm_settled: int = MONITORING_CONFIRM_SETTLED,
    stale_days: int = STALE_DATA_DAYS,
    settlement_grace_days: int = 2,
    log_loss_warning: float = 1.20,
    ece_warning: float = 0.10,
    drift_warning: float = 0.05,
    coverage: dict | None = None,
) -> dict:
    """Build the production monitoring report without changing any artifact."""
    now = _now(now)
    now_naive = pd.Timestamp(now)
    if now_naive.tzinfo is not None:
        now_naive = now_naive.tz_convert("UTC").tz_localize(None)
    if history is None:
        from src.predictions.history import load_history

        history, _ = load_history()
    if raw is None and Path(RAW_PATH).exists():
        try:
            raw = pd.read_csv(RAW_PATH, parse_dates=["date"])
        except Exception as exc:  # noqa: BLE001 - freshness check degrades
            logger.warning("Could not read raw data for monitoring: %s", exc)
            raw = pd.DataFrame()
    raw = pd.DataFrame() if raw is None else raw
    fixtures_available = fixtures is not None
    if fixtures is None:
        fixtures = load_fixture_rows()
    fixtures = fixtures.copy()

    frame = add_derived_columns(history) if history is not None and not history.empty else pd.DataFrame()
    counts = {
        "prediction_rows": int(len(frame)),
        "unique_fixtures": int(frame["fixture_id"].nunique()) if not frame.empty else 0,
        "multiple_snapshot_fixtures": 0,
        "settled_rows": 0,
        "pending_rows": 0,
        "settled_unique_fixtures": 0,
        "missing_predictions": 0,
        "overdue_pending": 0,
        "invalid_probability_rows": 0,
        "prediction_errors": 0,
    }
    checks: list[dict] = []
    metrics: dict = {}
    per_class: list[dict] = []
    drift: dict = {}

    if frame.empty:
        checks.append(_check(
            "prediction_counts", INSUFFICIENT,
            "No predictions are logged; no performance claim can be made.",
        ))
    else:
        counts["multiple_snapshot_fixtures"] = int(
            (frame.groupby("fixture_id").size() > 1).sum()
        )
        probabilities = _probabilities(frame)
        valid = _valid_probability_mask(probabilities)
        counts["invalid_probability_rows"] = int((~valid).sum())
        settled = _settled(frame)
        counts["settled_rows"] = int(len(settled))
        counts["pending_rows"] = int(len(frame) - len(settled))
        counts["settled_unique_fixtures"] = int(settled["fixture_id"].nunique())

        ledger_keys = _ledger_keys(frame)
        upcoming_keys = [
            key for key in _fixture_keys(fixtures)
            if key[3] not in TERMINAL_STATUSES
        ] if not fixtures.empty else []
        missing = [
            key for key in upcoming_keys
            if (key[0], key[1], key[2]) not in ledger_keys
            and key[2] != ""
        ]
        counts["missing_predictions"] = int(len(missing))

        grace = now_naive.normalize() - pd.Timedelta(days=settlement_grace_days)
        if not settled.empty:
            pass
        pending = frame[~frame.index.isin(settled.index)]
        pending_dates = pd.to_datetime(pending["match_date"], errors="coerce")
        counts["overdue_pending"] = int((pending_dates < grace).sum())
        counts["prediction_errors"] = int(
            counts["invalid_probability_rows"] + counts["overdue_pending"]
        )

        checks.append(_check(
            "prediction_counts", OK,
            f"{counts['prediction_rows']} prediction rows cover "
            f"{counts['unique_fixtures']} fixtures; {counts['settled_rows']} settled, "
            f"{counts['pending_rows']} pending.",
            **counts,
        ))
        checks.append(_check(
            "missing_predictions",
            WARNING if counts["missing_predictions"] else OK,
            (
                f"{counts['missing_predictions']} configured upcoming fixture(s) "
                "have no prediction."
                if counts["missing_predictions"] else
                "Every configured dated upcoming fixture has a prediction."
            ),
            missing=counts["missing_predictions"],
        ))
        checks.append(_check(
            "prediction_errors",
            ERROR if counts["invalid_probability_rows"] else WARNING if counts["overdue_pending"] else OK,
            (
                f"{counts['invalid_probability_rows']} row(s) have invalid probabilities; "
                f"{counts['overdue_pending']} settled-age row(s) are still pending."
            ),
            invalid_probability_rows=counts["invalid_probability_rows"],
            overdue_pending=counts["overdue_pending"],
        ))

        if len(settled):
            settled_probs = _probabilities(settled)
            valid_settled = _valid_probability_mask(settled_probs)
            settled = settled[valid_settled].copy()
            settled_probs = settled_probs[valid_settled]
            y = settled["actual_outcome"].astype(int).to_numpy()
            n = len(settled)
            if n >= 2:
                result = _metrics(y, settled_probs, "live settled predictions")
                metrics = {
                    "n": int(n),
                    "accuracy": float(result["accuracy"]),
                    "log_loss": float(result["log_loss"]),
                    "brier_score": float(result["brier_score"]),
                    "macro_f1": float(result["macro_f1"]),
                    "draw_recall": float(result["draw_recall"]),
                    "ece": float(result["ece"]),
                }
                per_class = result["per_class"].to_dict(orient="records")
                reliable = n >= min_settled
                metrics["reliable"] = bool(reliable)
                metrics["min_settled"] = int(min_settled)
                checks.append(_check(
                    "settled_performance",
                    (
                        OK if metrics["log_loss"] <= log_loss_warning
                        else WARNING if reliable else INSUFFICIENT
                    ) if reliable else INSUFFICIENT,
                    (
                        f"{n} settled predictions: log-loss {metrics['log_loss']:.4f}, "
                        f"Brier {metrics['brier_score']:.4f}, accuracy "
                        f"{metrics['accuracy']:.1%}. "
                        + ("" if reliable else f"Below {min_settled}; unreliable.")
                    ),
                    **{k: v for k, v in metrics.items() if k != "reliable"},
                ))
                checks.append(_check(
                    "calibration",
                    (OK if metrics["ece"] <= ece_warning else WARNING) if reliable else INSUFFICIENT,
                    (
                        f"ECE {metrics['ece']:.4f} over {n} settled predictions."
                        + ("" if reliable else f" Below {min_settled}; not a conclusion.")
                    ),
                    **{k: v for k, v in metrics.items() if k != "reliable"},
                ))

                # Drift: compare earlier and recent settled windows.
                ordered = settled.sort_values(
                    ["match_date_ts", "predicted_at_ts"], na_position="last"
                )
                half = n // 2
                if half >= 10:
                    earlier_probs = _probabilities(ordered.iloc[:half])
                    recent_probs = _probabilities(ordered.iloc[half:])
                    earlier_y = ordered.iloc[:half]["actual_outcome"].astype(int).to_numpy()
                    recent_y = ordered.iloc[half:]["actual_outcome"].astype(int).to_numpy()
                    earlier_metrics = _metrics(earlier_y, earlier_probs, "earlier")
                    recent_metrics = _metrics(recent_y, recent_probs, "recent")
                    interval = _paired_ci(
                        _per_row_log_loss(earlier_y, earlier_probs),
                        _per_row_log_loss(recent_y, recent_probs),
                    )
                    drift = {
                        "earlier_n": int(half),
                        "recent_n": int(n - half),
                        "earlier_log_loss": float(earlier_metrics["log_loss"]),
                        "recent_log_loss": float(recent_metrics["log_loss"]),
                        "log_loss_delta": float(
                            recent_metrics["log_loss"] - earlier_metrics["log_loss"]
                        ),
                        "earlier_accuracy": float(earlier_metrics["accuracy"]),
                        "recent_accuracy": float(recent_metrics["accuracy"]),
                        **interval,
                    }
                    confirmed = (
                        n >= confirm_settled
                        and drift["log_loss_delta"] >= drift_warning
                        and bool(interval["ci_excludes_zero"])
                        and interval["delta"] is not None
                        and interval["delta"] > 0
                    )
                    status = (
                        CONFIRMED if confirmed
                        else WARNING if drift["log_loss_delta"] >= drift_warning
                        else OK
                    )
                    checks.append(_check(
                        "performance_drift", status,
                        f"Recent log-loss is {drift['recent_log_loss']:.4f} vs earlier "
                        f"{drift['earlier_log_loss']:.4f} "
                        f"(delta {drift['log_loss_delta']:+.4f}; "
                        f"{drift['recent_n']} recent / {drift['earlier_n']} earlier).",
                        **drift,
                    ))
                else:
                    checks.append(_check(
                        "performance_drift", INSUFFICIENT,
                        f"Only {n} settled rows; need at least 20 for a two-window comparison.",
                    ))
            else:
                checks.append(_check(
                    "settled_performance", INSUFFICIENT,
                    f"Only {n} settled prediction(s); metrics are not meaningful.",
                ))
        else:
            checks.append(_check(
                "settled_performance", INSUFFICIENT,
                "No settled predictions yet; performance and calibration are unknown.",
            ))

    # Data and fixture freshness.
    freshness: dict = {
        "latest_match_date": None,
        "match_age_days": None,
        "latest_prediction_at": None,
        "prediction_age_days": None,
        "fixtures_newest_fetched_at": None,
        "fixtures_age_days": None,
    }
    stale_reasons: list[str] = []
    if not raw.empty and "date" in raw:
        latest = pd.to_datetime(raw["date"], errors="coerce").max()
        if pd.notna(latest):
            age = (now_naive.normalize() - latest.normalize()).days
            freshness["latest_match_date"] = f"{latest:%Y-%m-%d}"
            freshness["match_age_days"] = int(age)
            if age > stale_days:
                stale_reasons.append(f"latest match is {age} days old")
    if not frame.empty and "predicted_at_ts" in frame:
        latest_pred = frame["predicted_at_ts"].max()
        if pd.notna(latest_pred):
            stamp = pd.Timestamp(latest_pred)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            age = (pd.Timestamp(now) - stamp).days
            freshness["latest_prediction_at"] = f"{stamp:%Y-%m-%dT%H:%M:%SZ}"
            freshness["prediction_age_days"] = int(age)
            if age > stale_days:
                stale_reasons.append(f"latest prediction is {age} days old")
    if fixtures_available and not fixtures.empty and "fetched_at" in fixtures:
        stamps = pd.to_datetime(fixtures["fetched_at"], errors="coerce", utc=True).dropna()
        if not stamps.empty:
            latest_fetch = stamps.max()
            age = (pd.Timestamp(now) - latest_fetch).days
            freshness["fixtures_newest_fetched_at"] = f"{latest_fetch:%Y-%m-%dT%H:%M:%SZ}"
            freshness["fixtures_age_days"] = int(age)
            if age > stale_days:
                stale_reasons.append(f"fixture snapshot is {age} days old")
    checks.append(_check(
        "data_freshness",
        WARNING if stale_reasons else OK if freshness["latest_match_date"] else INSUFFICIENT,
        "; ".join(stale_reasons) if stale_reasons else "Data and fixture freshness are within threshold.",
        **freshness,
    ))

    # Fixture coverage is delegated to the dedicated audit.
    if coverage is None:
        try:
            coverage, _coverage_table = audit_fixture_coverage(
                raw, fixtures,
                now=now,
                telemetry=ProviderTelemetry(Path(PROVIDER_STATUS_PATH)),
                stale_days=stale_days,
            )
        except Exception as exc:  # noqa: BLE001 - coverage failure is reported
            coverage = {"status": ERROR, "warnings": [str(exc)], "upcoming": {}}
    checks.append(_check(
        "fixture_coverage",
        WARNING if coverage.get("status") in {"warning", "error"} else OK,
        f"{coverage.get('upcoming', {}).get('n', 0)} upcoming fixture(s) stored; "
        f"schedule completeness is "
        f"{coverage.get('upcoming', {}).get('schedule_completeness', 'unknown')}.",
        warnings=coverage.get("warnings", [])[:10],
    ))

    statuses = [check["status"] for check in checks]
    if ERROR in statuses:
        overall = ERROR
    elif CONFIRMED in statuses:
        overall = CONFIRMED
    elif WARNING in statuses:
        overall = WARNING
    elif INSUFFICIENT in statuses:
        overall = INSUFFICIENT
    else:
        overall = OK

    reliable = bool(metrics.get("reliable", False))
    report = {
        "generated_at": now.isoformat(timespec="seconds"),
        "overall_status": overall,
        "live_performance_reliable": reliable,
        "conclusions_allowed": reliable,
        "minimum_settled_sample": int(min_settled),
        "counts": counts,
        "metrics": metrics,
        "per_class": per_class,
        "drift": drift,
        "freshness": freshness,
        "fixture_coverage": {
            "status": coverage.get("status"),
            "upcoming": coverage.get("upcoming", {}),
            "warnings": coverage.get("warnings", []),
        },
        "checks": checks,
        "notes": [
            "Live metrics below the minimum settled sample are not conclusions.",
            "A confirmed drift label requires a sufficiently large sample and a "
            "supportive bootstrap interval.",
            "This monitor never retrains, promotes or rewrites a model.",
        ],
        "ledger_path": str(LEDGER_PATH),
        "raw_path": str(RAW_PATH),
        "fixtures_path": str(UPCOMING_FIXTURES_PATH),
    }
    return _json_safe(report)


def write_prediction_monitoring_report(
    report: dict,
    *,
    json_path: Path = PREDICTION_MONITORING_PATH,
    markdown_path: Path = PREDICTION_MONITORING_MARKDOWN_PATH,
) -> list[Path]:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8")

    checks = pd.DataFrame(report["checks"])
    metrics = report.get("metrics", {})
    counts = report.get("counts", {})
    lines = [
        "# La Liga Predictor — Prediction Monitoring",
        "",
        f"Generated: {report['generated_at']}",
        f"Overall status: **{report['overall_status']}**",
        "",
        f"> Live performance reliable: **{report['live_performance_reliable']}**. "
        f"Minimum settled sample: {report['minimum_settled_sample']}. "
        "Below that threshold, metrics are arithmetic only and not conclusions.",
        "",
        "## Prediction ledger",
        "",
        f"- Prediction rows: {counts.get('prediction_rows', 0)}",
        f"- Unique fixtures: {counts.get('unique_fixtures', 0)}",
        f"- Multi-snapshot fixtures: {counts.get('multiple_snapshot_fixtures', 0)}",
        f"- Settled rows: {counts.get('settled_rows', 0)}",
        f"- Pending rows: {counts.get('pending_rows', 0)}",
        f"- Missing predictions: {counts.get('missing_predictions', 0)}",
        f"- Overdue pending: {counts.get('overdue_pending', 0)}",
        f"- Invalid probability rows: {counts.get('invalid_probability_rows', 0)}",
        "",
        "## Settled performance",
        "",
    ]
    if metrics:
        lines += [
            f"- n: {metrics.get('n')}",
            f"- Log-loss: {metrics.get('log_loss'):.4f}",
            f"- Brier score: {metrics.get('brier_score'):.4f}",
            f"- Accuracy: {metrics.get('accuracy'):.1%}",
            f"- Macro F1: {metrics.get('macro_f1'):.4f}",
            f"- Draw recall: {metrics.get('draw_recall'):.1%}",
            f"- ECE: {metrics.get('ece'):.4f}",
            f"- Reliable: **{metrics.get('reliable')}**",
        ]
    else:
        lines.append("- No settled predictions with usable probabilities.")
    lines += ["", "## Checks", ""]
    if not checks.empty:
        lines += ["| check | status | message |", "|---|---|---|"]
        for _, row in checks.iterrows():
            lines.append(f"| {row['name']} | {row['status']} | {row['message']} |")
    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in report.get("notes", [])]
    lines += [""]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, markdown_path]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    report = monitor_predictions()
    paths = write_prediction_monitoring_report(report)
    print(json.dumps({
        "overall_status": report["overall_status"],
        "live_performance_reliable": report["live_performance_reliable"],
        "counts": report["counts"],
        "reports": [str(path) for path in paths],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
