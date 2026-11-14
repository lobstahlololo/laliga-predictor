"""Phase 8 fixture operations and prediction lifecycle.

This module owns the operational state around predictions:

* **Calendar refresh** — fetch the OpenFootball full-season calendar, validate
  it, normalise teams, merge it into the fixture store without discarding valid
  stored fixtures, record schedule revisions and provider telemetry, and persist
  a refresh manifest so completeness claims require a verified refresh.
* **Prediction cycles** — select every eligible upcoming fixture once, generate
  one current prediction per fixture per cycle, and link each new snapshot to
  the snapshot it supersedes (kickoff change or later cycle).
* **Result reconciliation** — settle only verified final scores, keep
  postponed/abandoned/cancelled fixtures out of the settled set, prevent
  duplicate settlement, and append an immutable correction record whenever an
  official result is amended.
* **Queries** — upcoming fixture status, per-fixture prediction history,
  schedule revisions and settled results for the API and dashboard.

Nothing here trains, promotes or rewrites a model, and no fixture, kickoff time
or result is ever invented.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.    fixture_provider import (
    FIXTURE_COLUMNS,
    FixtureRecord,
    is_date_only_kickoff,
    load_fixture_rows,
    merge_fixtures,
    records_from_openfootball,
    write_fixtures_atomic,
)
from src.data.football_calendar import save_calendar
from src.predictions import history as hist
from src.predictions.fixtures import load_upcoming_fixtures
from src.predictions.predictor import predict_fixtures, predictions_to_records
from src.utils.config import (
    CALENDAR_MIN_FIXTURES,
    CALENDAR_STALE_HOURS,
    FIXTURE_REFRESH_STATE_PATH,
    FIXTURE_SCHEDULE_REVISIONS_PATH,
    LEDGER_PATH,
    PROVIDER_STATUS_PATH,
    RESULT_CORRECTIONS_PATH,
    SEASON_CALENDAR_PATH,
    UPCOMING_FIXTURES_PATH,
)
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

OPENFOOTBALL_PROVIDER = "openfootball"
TERMINAL_FIXTURE_STATUSES = {
    "finished", "cancelled", "canceled", "abandoned", "ft", "aet", "pen",
}
POSTPONED_STATUSES = {"postponed", "pst", "suspended", "delayed"}
CANCELLED_STATUSES = {"cancelled", "canceled"}
ABANDONED_STATUSES = {"abandoned"}
RESULT_DATE_TOLERANCE_DAYS = 3
DEFAULT_RESULT_SOURCE = "football-data.co.uk"


# ---------------------------------------------------------------------------
# Small atomic helpers
# ---------------------------------------------------------------------------

def _now(now: datetime | None = None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _iso(moment: datetime | None = None) -> str:
    return _now(moment).isoformat(timespec="seconds")


def _atomic_json(payload: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def _append_jsonl(record: dict, path: Path) -> None:
    """Append one immutable line to an append-only operations log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    except Exception as exc:  # noqa: BLE001 - a bad ops log must not crash a cycle
        logger.warning("Could not parse %s: %s", path, exc)
    return rows


def _parse_ts(value, *, utc: bool = False) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=utc)
    if parsed is pd.NaT or pd.isna(parsed):
        return None
    return parsed


# ---------------------------------------------------------------------------
# Calendar refresh state
# ---------------------------------------------------------------------------

def load_refresh_state(path: Path = FIXTURE_REFRESH_STATE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - corrupt state means "unverified"
        logger.warning("Could not read calendar refresh state %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def calendar_freshness(
    state: dict | None = None,
    *,
    now: datetime | None = None,
    stale_hours: int = CALENDAR_STALE_HOURS,
) -> dict:
    """Describe whether the stored calendar can support a coverage claim."""
    state = load_refresh_state() if state is None else state
    verified_at = _parse_ts(state.get("verified_at") or "", utc=True)
    if verified_at is None:
        return {
            "verified": False,
            "status": state.get("status", "never_refreshed"),
            "age_hours": None,
            "stale": True,
            "stale_hours": int(stale_hours),
            "fetched_at": state.get("fetched_at"),
            "verified_at": None,
        }
    age_hours = (_now(now) - verified_at).total_seconds() / 3600
    stale = age_hours > stale_hours
    return {
        "verified": bool(state.get("verified")) and not stale,
        "status": "stale" if stale else state.get("status", "unknown"),
        "age_hours": round(age_hours, 2),
        "stale": bool(stale),
        "stale_hours": int(stale_hours),
        "fetched_at": state.get("fetched_at"),
        "verified_at": state.get("verified_at"),
    }


def validate_calendar(
    frame: pd.DataFrame | None,
    *,
    min_fixtures: int = CALENDAR_MIN_FIXTURES,
) -> dict:
    """Validate a provider calendar before it may touch the fixture store.

    A reduced payload (a provider window, a truncated download or an empty
    response) is rejected as a whole: it must never overwrite verified fixtures.
    """
    errors: list[str] = []
    warnings: list[str] = []
    count = 0 if frame is None else int(len(frame))
    if count == 0:
        errors.append("calendar payload is empty")
    elif count < min_fixtures:
        errors.append(
            f"calendar payload has {count} fixtures, below the "
            f"{min_fixtures}-fixture full-season minimum"
        )
    if frame is not None and not frame.empty:
        for column in ("home_team", "away_team"):
            if column not in frame.columns:
                errors.append(f"calendar payload is missing '{column}'")
        if "home_team" in frame.columns and "away_team" in frame.columns:
            blank = (
                frame["home_team"].astype(str).str.strip().eq("")
                | frame["away_team"].astype(str).str.strip().eq("")
            )
            if blank.any():
                errors.append(f"{int(blank.sum())} calendar fixtures have a blank team")
        if "match_date" in frame.columns:
            dates = pd.to_datetime(
                frame["match_date"].astype(str).str.strip(), errors="coerce"
            )
            invalid = dates.isna() & frame["match_date"].astype(str).str.strip().ne("")
            if invalid.any():
                errors.append(f"{int(invalid.sum())} calendar fixtures have invalid dates")
            if dates.isna().mean() > 0.2:
                warnings.append("more than 20% of calendar fixtures have no usable date")
        if "fixture_id" in frame.columns:
            duplicates = int(frame["fixture_id"].duplicated().sum())
            if duplicates:
                warnings.append(f"{duplicates} duplicate fixture IDs in the calendar")
    return {
        "valid": not errors,
        "fixtures": count,
        "errors": errors,
        "warnings": warnings,
    }


def detect_schedule_changes(
    previous: pd.DataFrame | None,
    current: pd.DataFrame | None,
    *,
    detected_at: str = "",
) -> list[dict]:
    """Compare two fixture snapshots by pairing and report real changes."""
    if previous is None or current is None or previous.empty or current.empty:
        return []

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        data = frame.copy()
        for column in FIXTURE_COLUMNS:
            if column not in data.columns:
                data[column] = ""
        data["status"] = data["status"].astype(str).str.lower().str.strip()
        data = data[~data["status"].isin(TERMINAL_FIXTURE_STATUSES)]
        return data

    old = prepare(previous)
    new = prepare(current)
    new_by_pair = {
        (canonical_team(row["home_team"]), canonical_team(row["away_team"])): row
        for row in new.to_dict("records")
    }
    changes: list[dict] = []
    for row in old.to_dict("records"):
        key = (canonical_team(row["home_team"]), canonical_team(row["away_team"]))
        replacement = new_by_pair.get(key)
        if replacement is None:
            continue
        previous_kickoff = str(row.get("kickoff_utc", "")).strip()
        current_kickoff = str(replacement.get("kickoff_utc", "")).strip()
        previous_date = str(row.get("match_date", "")).strip()
        current_date = str(replacement.get("match_date", "")).strip()
        previous_status = str(row.get("status", "") or "scheduled").strip()
        current_status = str(replacement.get("status", "") or "scheduled").strip()

        if previous_date and current_date and previous_date != current_date:
            change = "rescheduled"
        elif previous_kickoff and current_kickoff and previous_kickoff != current_kickoff:
            change = "kickoff_time_changed"
        elif previous_status != current_status:
            change = "status_changed"
        else:
            continue

        changes.append({
            "detected_at": detected_at or _iso(),
            "fixture_id": str(replacement.get("fixture_id", "")).strip(),
            "previous_fixture_id": str(row.get("fixture_id", "")).strip(),
            "home_team": key[0],
            "away_team": key[1],
            "change_type": change,
            "previous_kickoff_utc": previous_kickoff,
            "current_kickoff_utc": current_kickoff,
            "previous_match_date": previous_date,
            "current_match_date": current_date,
            "previous_status": previous_status,
            "current_status": current_status,
            "provider": str(replacement.get("provider", "")).strip(),
        })
    return changes


def refresh_fixture_calendar(
    *,
    store_path: Path = UPCOMING_FIXTURES_PATH,
    calendar_path: Path = SEASON_CALENDAR_PATH,
    state_path: Path = FIXTURE_REFRESH_STATE_PATH,
    revisions_path: Path = FIXTURE_SCHEDULE_REVISIONS_PATH,
    provider=None,
    records: list[FixtureRecord] | None = None,
    now: datetime | None = None,
    min_fixtures: int = CALENDAR_MIN_FIXTURES,
    record_telemetry: bool = True,
) -> dict:
    """Fetch, validate, normalise and merge the season calendar.

    ``records`` lets tests (and offline maintenance) supply an already-fetched
    batch, so no test needs network access. Any fetch failure leaves the store
    untouched and is reported as ``fallback``; an invalid or partial payload is
    reported as ``invalid`` and never written over valid fixtures.
    """
    store_path = Path(store_path)
    moment = _now(now)
    fetched_at = _iso(moment)
    previous_state = load_refresh_state(state_path)
    previous_store = load_fixture_rows(store_path)
    detail: dict = {
        "status": "fallback",
        "verified": False,
        "provider": OPENFOOTBALL_PROVIDER,
        "fetched": 0,
        "stored": int(len(previous_store)),
        "kept": int(len(previous_store)),
        "schedule_changes": 0,
        "errors": [],
        "warnings": [],
        "fetched_at": fetched_at,
        "verified_at": previous_state.get("verified_at"),
        "source": previous_state.get("source"),
        "store_path": str(store_path),
    }

    payload = None
    if records is None:
        try:
            from src.data.fixture_provider import OpenFootballProvider

            provider = provider or OpenFootballProvider()
            payload = provider.fetch_payload()
            records = records_from_openfootball(payload)
        except Exception as exc:  # noqa: BLE001 - an outage must be survivable
            detail["errors"] = [f"openfootball: {exc}"]
            logger.warning("Calendar refresh failed, keeping stored fixtures: %s", exc)
            _record_telemetry("failed", 0, 0, [str(exc)], enabled=record_telemetry)
            _write_state(state_path, detail, previous_state, moment, verified=False)
            return detail

    incoming = list(records or [])
    frame = pd.DataFrame(
        [record.as_row() for record in incoming], columns=FIXTURE_COLUMNS
    )
    validation = validate_calendar(frame, min_fixtures=min_fixtures)
    detail["fetched"] = len(frame)
    detail["warnings"] = list(validation["warnings"])
    detail["source"] = str(getattr(provider, "source_url", "") or detail["source"] or "")

    if not validation["valid"]:
        detail["status"] = "invalid"
        detail["errors"] = list(validation["errors"])
        detail["kept"] = int(len(previous_store))
        logger.warning(
            "Calendar rejected (%s); stored fixtures preserved.",
            "; ".join(validation["errors"]),
        )
        _record_telemetry(
            "invalid", len(frame), 0, validation["errors"], enabled=record_telemetry
        )
        _write_state(state_path, detail, previous_state, moment, verified=False)
        return detail

    merged = merge_fixtures(previous_store, incoming)
    changes = detect_schedule_changes(previous_store, merged, detected_at=fetched_at)
    for change in changes:
        _append_jsonl(change, revisions_path)

    write_fixtures_atomic(merged, store_path)
    if payload is not None:
        try:
            save_calendar(
                payload,
                path=calendar_path,
                source=str(getattr(provider, "source_url", "") or detail["source"] or ""),
                season=str(getattr(provider, "season", "") or ""),
                fetched_at=fetched_at,
            )
        except Exception as exc:  # noqa: BLE001 - cache write must not lose fixtures
            logger.warning("Could not cache the season calendar: %s", exc)
            detail["warnings"].append(f"calendar cache not written: {exc}")

    detail.update({
        "status": "ok",
        "verified": True,
        "fetched": len(incoming),
        "stored": int(len(merged)),
        "schedule_changes": len(changes),
        "verified_at": fetched_at,
    })
    _record_telemetry(
        "ok", len(incoming), int(len(merged)), [], enabled=record_telemetry
    )
    logger.info(
        "Calendar verified: %d fixtures, %d stored, %d schedule change(s).",
        len(incoming), len(merged), len(changes),
    )
    _write_state(state_path, detail, previous_state, moment, verified=True)
    return detail


def _record_telemetry(
    status: str,
    fetched: int,
    written: int,
    errors: list[str],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    try:
        from src.data.fixture_coverage import ProviderTelemetry

        ProviderTelemetry(PROVIDER_STATUS_PATH).record(
            OPENFOOTBALL_PROVIDER,
            status,
            fetched=int(fetched),
            written=int(written),
            errors=list(errors),
        )
    except Exception as exc:  # noqa: BLE001 - telemetry is best effort
        logger.warning("Could not record calendar provider telemetry: %s", exc)


def _write_state(
    state_path: Path,
    detail: dict,
    previous_state: dict,
    moment: datetime,
    *,
    verified: bool,
) -> None:
    """Persist the refresh manifest; a failure never erases the last success."""
    state = {
        "status": detail.get("status", "unknown"),
        "verified": bool(verified),
        "fetched_at": detail.get("fetched_at"),
        "verified_at": detail.get("verified_at") if verified
        else previous_state.get("verified_at"),
        "provider": detail.get("provider"),
        "source": detail.get("source"),
        "fetched": int(detail.get("fetched", 0)),
        "stored": int(detail.get("stored", 0)),
        "kept": int(detail.get("kept", 0)),
        "schedule_changes": int(detail.get("schedule_changes", 0)),
        "errors": list(detail.get("errors", [])),
        "warnings": list(detail.get("warnings", [])),
        "store_path": detail.get("store_path"),
        "checked_at": _iso(moment),
        "stale_hours": CALENDAR_STALE_HOURS,
    }
    try:
        _atomic_json(state, state_path)
    except Exception as exc:  # noqa: BLE001 - state write must not break a cycle
        logger.warning("Could not persist calendar refresh state: %s", exc)


# ---------------------------------------------------------------------------
# Prediction slate and cycles
# ---------------------------------------------------------------------------

def cycle_id(now: datetime | None = None) -> str:
    """Stable identifier for one prediction cycle (UTC, second resolution)."""
    return f"{_now(now).astimezone(timezone.utc):%Y%m%dT%H%M%SZ}"


def is_kickoff_past(fixture: dict, *, now: datetime | None = None) -> bool:
    """True when the stored kickoff (or date) has already passed."""
    moment = _now(now)
    now_naive = pd.Timestamp(moment)
    if now_naive.tzinfo is not None:
        now_naive = now_naive.tz_convert("UTC").tz_localize(None)
    kickoff = _parse_ts(fixture.get("kickoff_utc"), utc=True)
    if kickoff is not None:
        kickoff_naive = kickoff.tz_convert("UTC").tz_localize(None)
        return bool(kickoff_naive <= now_naive)
    date = _parse_ts(fixture.get("match_date"))
    if date is not None:
        return bool(date.normalize() < now_naive.normalize())
    return False


def eligible_fixtures(
    fixtures: list[dict] | None = None,
    *,
    now: datetime | None = None,
) -> tuple[list[dict], dict]:
    """Every upcoming fixture that may be predicted exactly once.

    Finished/cancelled/abandoned fixtures and fixtures whose kickoff has passed
    are excluded, so no post-kickoff information can enter a prediction.
    Postponed fixtures stay eligible when they still have a future date; a
    fixture without any date is kept with ``kickoff_provenance="unknown"`` and is
    scored date-agnostically rather than dropped or invented.
    """
    if fixtures is None:
        fixtures, source = load_upcoming_fixtures()
    else:
        source = "provided"
    selected: list[dict] = []
    index: dict[str, int] = {}
    excluded = {"past": 0, "terminal": 0, "invalid": 0, "duplicate": 0}

    for fixture in fixtures:
        home = str(fixture.get("home_team", "")).strip()
        away = str(fixture.get("away_team", "")).strip()
        if not home or not away or canonical_team(home) == canonical_team(away):
            excluded["invalid"] += 1
            continue
        status = str(fixture.get("status", "") or "scheduled").strip().lower()
        if status in TERMINAL_FIXTURE_STATUSES:
            excluded["terminal"] += 1
            continue
        match_date = str(fixture.get("match_date", "")).strip()
        raw_kickoff = str(fixture.get("kickoff_utc", "")).strip()
        kickoff = _parse_ts(raw_kickoff, utc=True)
        date_only = is_date_only_kickoff(raw_kickoff) if raw_kickoff else True
        if not match_date and kickoff is not None:
            match_date = f"{kickoff:%Y-%m-%d}"
        if (not date_only and is_kickoff_past(fixture, now=now)) or (
            date_only and match_date and is_kickoff_past(
                {"match_date": match_date}, now=now
            )
        ):
            excluded["past"] += 1
            continue

        fixture_id = str(fixture.get("fixture_id", "")).strip() or hist.make_fixture_id(
            home, away, match_date
        )
        record = {
            **fixture,
            "home_team": canonical_team(home),
            "away_team": canonical_team(away),
            "match_date": match_date,
            "fixture_id": fixture_id,
            "fixture_source": str(
                fixture.get("provider") or fixture.get("fixture_source") or "fixture-store"
            ),
            "kickoff_utc": str(fixture.get("kickoff_utc", "")).strip(),
            "kickoff_provenance": (
                "provider_timestamp" if kickoff is not None and not date_only
                else "date_only" if match_date else "unknown"
            ),
        }
        if fixture_id in index:
            excluded["duplicate"] += 1
            selected[index[fixture_id]] = record
            continue
        index[fixture_id] = len(selected)
        selected.append(record)

    return selected, {"source": source, "eligible": len(selected), **excluded}


def latest_snapshots(history: pd.DataFrame) -> pd.DataFrame:
    """Current prediction per fixture (the newest snapshot)."""
    if history is None or history.empty:
        return hist.empty_history()
    frame = hist.add_derived_columns(history)
    ordered = frame.sort_values(
        ["predicted_at_ts", "match_date_ts"], na_position="first"
    )
    return ordered.drop_duplicates("fixture_id", keep="last").reset_index(drop=True)


def generate_full_slate(
    *,
    model,
    preprocessor,
    metadata: dict | None = None,
    raw: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    fixtures: list[dict] | None = None,
    ledger_path: Path = LEDGER_PATH,
    now: datetime | None = None,
) -> tuple[list[dict], dict]:
    """Generate at most one prediction per eligible fixture in this cycle."""
    moment = _now(now)
    cycle = cycle_id(moment)
    predicted_at = _iso(moment)
    if history is None:
        history, _ = hist.load_history(ledger_path)

    slate, selection = eligible_fixtures(fixtures, now=moment)
    already: set[str] = set()
    if history is not None and not history.empty:
        frame = hist.add_derived_columns(history)
        mask = frame["prediction_cycle"].astype(str).str.strip() == cycle
        already = set(frame.loc[mask, "fixture_id"].astype(str))
    pending = [fixture for fixture in slate if fixture["fixture_id"] not in already]

    detail = {
        "cycle": cycle,
        "eligible": len(slate),
        "already_predicted_in_cycle": len(already),
        "predictions": 0,
        "appended": 0,
        "errors": 0,
        "source": selection["source"],
        "excluded": {
            key: value for key, value in selection.items()
            if key not in {"source", "eligible"}
        },
        "kickoff_provenance": {},
    }
    detail["kickoff_provenance"] = {
        provenance: sum(
            1 for fixture in slate if fixture["kickoff_provenance"] == provenance
        )
        for provenance in ("provider_timestamp", "date_only", "unknown")
    }
    if not pending:
        detail["note"] = "every eligible fixture already has a prediction this cycle"
        return [], detail

    predictions, errors = predict_fixtures(
        pending,
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        raw=raw,
        predicted_at=predicted_at,
    )
    records = predictions_to_records(predictions)
    by_pair = {
        (fixture["home_team"], fixture["away_team"]): fixture for fixture in pending
    }
    current = latest_snapshots(history) if history is not None else hist.empty_history()
    previous_by_fixture = (
        {row["fixture_id"]: row for row in current.to_dict("records")}
        if not current.empty else {}
    )
    # A reschedule changes the date-derived fixture ID, so the pairing is the
    # fallback identity used to link the update to the snapshot it supersedes.
    previous_by_pair = (
        {
            (canonical_team(row["home_team"]), canonical_team(row["away_team"])): row
            for row in current.to_dict("records")
        }
        if not current.empty else {}
    )

    enriched: list[dict] = []
    for record in records:
        fixture = by_pair.get((record["home_team"], record["away_team"]), {})
        record["prediction_cycle"] = cycle
        record["prediction_status"] = hist.STATUS_PENDING
        record["kickoff_utc"] = fixture.get("kickoff_utc", "")
        record["fixture_source"] = fixture.get("fixture_source", "")
        record["schedule_revision"] = fixture.get("schedule_revision", 0)
        previous = previous_by_fixture.get(record["fixture_id"])
        if previous is None:
            previous = previous_by_pair.get(
                (record["home_team"], record["away_team"])
            )
        if previous is not None and str(previous.get("prediction_cycle", "")) != cycle:
            record["supersedes_prediction_id"] = str(
                previous.get("prediction_id", "")
            )
        # The prediction ID includes the cycle, so it is recomputed here rather
        # than in the engine (which does not know about cycles).
        record["prediction_id"] = hist.make_prediction_id(
            record["fixture_id"], record["predicted_at"],
            record["model_version"], cycle,
        )
        enriched.append(record)

    appended, _ = hist.append_predictions(enriched, ledger_path)
    detail.update({
        "predictions": len(predictions),
        "appended": int(appended),
        "errors": len(errors),
        "unresolved": [
            {"fixture": item.get("fixture"), "error": item.get("error")}
            for item in errors[:20]
        ],
    })
    return predictions, detail


# ---------------------------------------------------------------------------
# Result reconciliation
# ---------------------------------------------------------------------------

def _int_or_none(value) -> int | None:
    parsed = hist._safe_float(value)
    return None if parsed is None else int(parsed)


def _outcome_from_goals(home: int | None, away: int | None) -> int | None:
    if home is None or away is None:
        return None
    return 2 if home > away else 1 if home == away else 0


def verified_results_from_raw(
    raw: pd.DataFrame,
    *,
    source: str = DEFAULT_RESULT_SOURCE,
) -> pd.DataFrame:
    """Extract only fully played, internally consistent final scores.

    A row without both goals, or whose stored result contradicts its goals, is
    skipped rather than repaired: results are never inferred.
    """
    columns = [
        "fixture_id", "match_date", "home_team", "away_team", "status",
        "home_goals", "away_goals", "verified", "source",
    ]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)
    frame = raw.copy()
    rows: list[dict] = []
    for _, match in frame.iterrows():
        home_goals = _int_or_none(match.get("full_time_home_goals"))
        away_goals = _int_or_none(match.get("full_time_away_goals"))
        if home_goals is None or away_goals is None:
            continue
        outcome = _outcome_from_goals(home_goals, away_goals)
        stored = match.get("match_outcome")
        if stored is None or (isinstance(stored, float) and not np.isfinite(stored)):
            stored = match.get("full_time_result")
            stored_code = hist.FTR_TO_OUTCOME.get(str(stored).strip().upper())
        else:
            stored_code = _int_or_none(stored)
        if stored_code is not None and stored_code != outcome:
            logger.warning(
                "Skipping inconsistent score for %s vs %s (%s-%s).",
                match.get("home_team"), match.get("away_team"),
                home_goals, away_goals,
            )
            continue
        date = _parse_ts(match.get("date"))
        match_date = f"{date:%Y-%m-%d}" if date is not None else ""
        home = canonical_team(match.get("home_team"))
        away = canonical_team(match.get("away_team"))
        fixture_id = str(match.get("match_id", "") or "").strip()
        if not fixture_id:
            fixture_id = hist.make_fixture_id(home, away, match_date)
        rows.append({
            "fixture_id": fixture_id,
            "match_date": match_date,
            "home_team": home,
            "away_team": away,
            "status": "finished",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "verified": True,
            "source": source,
        })
    return pd.DataFrame(rows, columns=columns)


def status_results_from_store(
    fixtures: pd.DataFrame | None,
    *,
    source: str = "fixture-store",
) -> pd.DataFrame:
    """Non-final fixture states observed by the provider (never a score)."""
    columns = [
        "fixture_id", "match_date", "home_team", "away_team", "status",
        "home_goals", "away_goals", "verified", "source",
    ]
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for _, fixture in fixtures.iterrows():
        status = str(fixture.get("status", "") or "").strip().lower()
        if status in POSTPONED_STATUSES:
            mapped = "postponed"
        elif status in CANCELLED_STATUSES:
            mapped = "cancelled"
        elif status in ABANDONED_STATUSES:
            mapped = "abandoned"
        else:
            continue
        rows.append({
            "fixture_id": str(fixture.get("fixture_id", "")).strip(),
            "match_date": str(fixture.get("match_date", "")).strip(),
            "home_team": canonical_team(fixture.get("home_team")),
            "away_team": canonical_team(fixture.get("away_team")),
            "status": mapped,
            "home_goals": None,
            "away_goals": None,
            "verified": True,
            "source": source,
        })
    return pd.DataFrame(rows, columns=columns)


def _index_results(results: pd.DataFrame) -> tuple[dict, dict]:
    by_id: dict[str, dict] = {}
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for row in results.to_dict("records"):
        fixture_id = str(row.get("fixture_id", "")).strip()
        if fixture_id:
            by_id[fixture_id] = row
        by_pair.setdefault(
            (canonical_team(row.get("home_team")), canonical_team(row.get("away_team"))),
            [],
        ).append(row)
    return by_id, by_pair


def _find_result(
    row: pd.Series,
    by_id: dict,
    by_pair: dict,
    *,
    tolerance_days: int = RESULT_DATE_TOLERANCE_DAYS,
) -> dict | None:
    fixture_id = str(row.get("fixture_id", "")).strip()
    if fixture_id and fixture_id in by_id:
        return by_id[fixture_id]
    candidates = by_pair.get((
        canonical_team(row.get("home_team")),
        canonical_team(row.get("away_team")),
    ), [])
    target = _parse_ts(row.get("match_date"))
    best, best_gap = None, None
    for candidate in candidates:
        if not bool(candidate.get("verified", True)):
            continue
        candidate_date = _parse_ts(candidate.get("match_date"))
        if target is None or candidate_date is None:
            gap = 0 if target is None and candidate_date is None else None
            if gap is None:
                continue
        else:
            gap = abs(int((candidate_date - target).days))
            if gap > tolerance_days:
                continue
        if best is None or gap < best_gap:
            best, best_gap = candidate, gap
    return best


def reconcile_verified_results(
    history: pd.DataFrame,
    results: pd.DataFrame,
    *,
    corrections_path: Path = RESULT_CORRECTIONS_PATH,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Settle only verified results; append corrections instead of rewriting.

    Returns ``(updated_history, stats)``. Earlier rows are not deleted: a
    corrected result keeps the row and appends a correction record containing
    the previous values, the new values, the source and the timestamp.
    """
    stats = {
        "settled": 0, "corrected": 0, "already_settled": 0, "pending": 0,
        "voided": 0, "postponed": 0, "cancelled": 0, "abandoned": 0,
        "unmatched": 0, "results": 0 if results is None else int(len(results)),
    }
    if history is None or history.empty:
        return hist.empty_history(), stats
    out = history.copy()
    if results is None or results.empty:
        stats["pending"] = int(
            (out["actual_outcome"].astype(str).str.strip() == "").sum()
        )
        return out, stats

    by_id, by_pair = _index_results(results)
    stamp = _iso(now)

    for idx in out.index:
        settled = str(out.at[idx, "actual_outcome"]).strip() != ""
        result = _find_result(out.loc[idx], by_id, by_pair)
        if result is None:
            if settled:
                stats["already_settled"] += 1
            else:
                stats["pending"] += 1
                stats["unmatched"] += 1
            continue

        status = str(result.get("status", "")).strip().lower()
        verified = bool(result.get("verified", False))
        if not verified:
            stats["pending"] += 1
            continue

        if status in {"postponed", "suspended"}:
            stats["postponed"] += 1
            if settled:
                stats["already_settled"] += 1
            else:
                out.at[idx, "result_status"] = "postponed"
                out.at[idx, "result_source"] = str(result.get("source", ""))
                stats["pending"] += 1
            continue
        if status in {"cancelled", "canceled"}:
            stats["cancelled"] += 1
            stats["voided"] += 1
            if not settled:
                out.at[idx, "result_status"] = "cancelled"
                out.at[idx, "prediction_status"] = hist.STATUS_VOID
                out.at[idx, "result_source"] = str(result.get("source", ""))
            else:
                stats["already_settled"] += 1
            continue
        if status == "abandoned":
            stats["abandoned"] += 1
            stats["voided"] += 1
            if not settled:
                out.at[idx, "result_status"] = "abandoned"
                out.at[idx, "prediction_status"] = hist.STATUS_VOID
                out.at[idx, "result_source"] = str(result.get("source", ""))
            else:
                stats["already_settled"] += 1
            continue
        if status != "finished":
            stats["pending"] += 1
            continue

        home_goals = _int_or_none(result.get("home_goals"))
        away_goals = _int_or_none(result.get("away_goals"))
        outcome = _outcome_from_goals(home_goals, away_goals)
        if outcome is None:
            stats["pending"] += 1
            continue

        new_home, new_away = str(home_goals), str(away_goals)
        if not settled:
            out.at[idx, "actual_outcome"] = str(outcome)
            out.at[idx, "actual_home_goals"] = new_home
            out.at[idx, "actual_away_goals"] = new_away
            out.at[idx, "correct"] = str(
                int(out.at[idx, "predicted_outcome"] == hist._to_label(outcome))
            )
            out.at[idx, "result_source"] = str(result.get("source", ""))
            out.at[idx, "result_status"] = hist.RESULT_FINAL
            out.at[idx, "settled_at"] = stamp
            out.at[idx, "prediction_status"] = hist.STATUS_SETTLED
            out.at[idx, "correction_count"] = "0"
            if not str(out.at[idx, "match_date"]).strip():
                out.at[idx, "match_date"] = str(result.get("match_date", ""))
            stats["settled"] += 1
            continue

        previous_home = str(out.at[idx, "actual_home_goals"]).strip()
        previous_away = str(out.at[idx, "actual_away_goals"]).strip()
        previous_outcome = str(out.at[idx, "actual_outcome"]).strip()
        changed = (
            previous_home != new_home
            or previous_away != new_away
            or previous_outcome != str(outcome)
        )
        if not changed:
            stats["already_settled"] += 1
            continue

        _append_jsonl({
            "correction_id": hist.make_prediction_id(
                out.at[idx, "fixture_id"],
                f"correction:{stamp}:{stats['corrected']}",
                out.at[idx, "model_version"],
            ),
            "corrected_at": stamp,
            "fixture_id": str(out.at[idx, "fixture_id"]),
            "prediction_id": str(out.at[idx, "prediction_id"]),
            "source": str(result.get("source", "")),
            "previous_outcome": previous_outcome,
            "previous_home_goals": previous_home,
            "previous_away_goals": previous_away,
            "new_outcome": str(outcome),
            "new_home_goals": new_home,
            "new_away_goals": new_away,
        }, corrections_path)
        out.at[idx, "actual_outcome"] = str(outcome)
        out.at[idx, "actual_home_goals"] = new_home
        out.at[idx, "actual_away_goals"] = new_away
        out.at[idx, "correct"] = str(
            int(out.at[idx, "predicted_outcome"] == hist._to_label(outcome))
        )
        out.at[idx, "result_source"] = str(result.get("source", ""))
        out.at[idx, "result_status"] = hist.RESULT_CORRECTED
        out.at[idx, "correction_count"] = str(
            int(_int_or_none(out.at[idx, "correction_count"]) or 0) + 1
        )
        out.at[idx, "corrected_at"] = stamp
        stats["corrected"] += 1

    return out, stats


def settle_results(
    raw: pd.DataFrame,
    *,
    fixtures: pd.DataFrame | None = None,
    ledger_path: Path = LEDGER_PATH,
    corrections_path: Path = RESULT_CORRECTIONS_PATH,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reconcile the ledger against verified final scores and persist changes."""
    history, _ = hist.load_history(ledger_path)
    results = verified_results_from_raw(raw)
    statuses = status_results_from_store(fixtures)
    if not statuses.empty:
        results = pd.concat([results, statuses], ignore_index=True)
    updated, stats = reconcile_verified_results(
        history, results, corrections_path=corrections_path, now=now
    )
    if stats["settled"] or stats["corrected"] or stats["voided"] or stats["postponed"]:
        hist.save_history(updated, ledger_path)
    return updated, stats


# ---------------------------------------------------------------------------
# Queries for API / dashboard
# ---------------------------------------------------------------------------

def load_schedule_revisions(
    path: Path = FIXTURE_SCHEDULE_REVISIONS_PATH,
    *,
    limit: int | None = None,
) -> list[dict]:
    revisions = _read_jsonl(path)
    if limit:
        return revisions[-limit:]
    return revisions


def load_result_corrections(
    path: Path = RESULT_CORRECTIONS_PATH,
    *,
    limit: int | None = None,
) -> list[dict]:
    corrections = _read_jsonl(path)
    if limit:
        return corrections[-limit:]
    return corrections


def upcoming_fixture_status(
    history: pd.DataFrame | None = None,
    *,
    fixtures: list[dict] | None = None,
    now: datetime | None = None,
    limit: int = 200,
    store_path: Path = UPCOMING_FIXTURES_PATH,
) -> list[dict]:
    """Upcoming fixtures joined with their current prediction status."""
    if fixtures is None:
        stored = load_fixture_rows(store_path)
        fixtures = []
        for row in stored.to_dict("records"):
            fixtures.append({
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "match_date": row.get("match_date"),
                "kickoff_utc": row.get("kickoff_utc"),
                "status": row.get("status"),
                "fixture_id": row.get("fixture_id"),
                "provider": row.get("provider"),
                "fixture_source": row.get("source"),
            })
    slate, _selection = eligible_fixtures(fixtures, now=now)
    if history is None:
        history, _ = hist.load_history()
    current = latest_snapshots(history)
    by_fixture = (
        {str(row["fixture_id"]): row for row in current.to_dict("records")}
        if not current.empty else {}
    )
    revisions = load_schedule_revisions()
    changed_pairs = {
        (canonical_team(item.get("home_team")), canonical_team(item.get("away_team")))
        for item in revisions
    }
    rows: list[dict] = []
    for fixture in slate[:limit]:
        prediction = by_fixture.get(fixture["fixture_id"], {})
        rows.append({
            "fixture_id": fixture["fixture_id"],
            "home_team": fixture["home_team"],
            "away_team": fixture["away_team"],
            "match_date": fixture["match_date"],
            "kickoff_utc": fixture["kickoff_utc"],
            "kickoff_provenance": fixture["kickoff_provenance"],
            "fixture_status": str(fixture.get("status", "") or "scheduled"),
            "fixture_source": fixture["fixture_source"],
            "schedule_changed": (
                canonical_team(fixture["home_team"]),
                canonical_team(fixture["away_team"]),
            ) in changed_pairs,
            "predicted": bool(prediction),
            "prediction_id": str(prediction.get("prediction_id", "")),
            "prediction_cycle": str(prediction.get("prediction_cycle", "")),
            "predicted_at": str(prediction.get("predicted_at", "")),
            "model_version": str(prediction.get("model_version", "")),
            "prediction_status": str(prediction.get("prediction_status", "")),
            "predicted_outcome": str(prediction.get("predicted_outcome", "")),
            "probabilities": (
                {
                    "home": float(prediction["model_prob_home"]),
                    "draw": float(prediction["model_prob_draw"]),
                    "away": float(prediction["model_prob_away"]),
                }
                if prediction and pd.notna(prediction.get("model_prob_home"))
                else None
            ),
            "expected_home_goals": prediction.get("expected_home_goals") or None,
            "expected_away_goals": prediction.get("expected_away_goals") or None,
            "most_likely_scoreline": prediction.get("most_likely_scoreline") or None,
        })
    return rows


def prediction_history(
    history: pd.DataFrame | None = None,
    *,
    fixture_id: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """Prediction snapshots, newest first, optionally for one fixture."""
    if history is None:
        history, _ = hist.load_history()
    if history is None or history.empty:
        return []
    frame = hist.add_derived_columns(history)
    if fixture_id:
        frame = frame[frame["fixture_id"].astype(str) == str(fixture_id)]
    if frame.empty:
        return []
    frame = frame.sort_values("predicted_at_ts", ascending=False).head(limit)
    columns = [
        "prediction_id", "fixture_id", "prediction_cycle", "match_date",
        "kickoff_utc", "home_team", "away_team", "predicted_at", "model_name",
        "model_version", "predicted_outcome", "confidence",
        "model_prob_home", "model_prob_draw", "model_prob_away",
        "expected_home_goals", "expected_away_goals", "most_likely_scoreline",
        "fixture_source", "prediction_status", "schedule_revision",
        "supersedes_prediction_id", "actual_outcome",
        "actual_home_goals", "actual_away_goals", "result_source",
        "result_status", "settled_at", "correction_count", "corrected_at",
    ]
    rows = []
    for record in frame.to_dict("records"):
        item = {}
        for column in columns:
            value = record.get(column, "")
            if isinstance(value, float) and not np.isfinite(value):
                value = None
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif not isinstance(value, (str, list, dict)) and pd.isna(value):
                value = None
            item[column] = value
        rows.append(item)
    return rows


def settled_results(
    history: pd.DataFrame | None = None,
    *,
    limit: int = 200,
) -> list[dict]:
    """Settled rows with their verified result and correction history."""
    if history is None:
        history, _ = hist.load_history()
    if history is None or history.empty:
        return []
    frame = hist.settled_only(history)
    if frame.empty:
        return []
    derived = hist.add_derived_columns(frame)
    derived = derived.sort_values("settled_at", ascending=False).head(limit)
    rows = []
    for record in derived.to_dict("records"):
        rows.append({
            "fixture_id": str(record.get("fixture_id", "")),
            "prediction_id": str(record.get("prediction_id", "")),
            "home_team": str(record.get("home_team", "")),
            "away_team": str(record.get("away_team", "")),
            "match_date": str(record.get("match_date", "")),
            "predicted_outcome": str(record.get("predicted_outcome", "")),
            "actual_outcome": hist._to_label(record.get("actual_outcome")),
            "actual_home_goals": record.get("actual_home_goals"),
            "actual_away_goals": record.get("actual_away_goals"),
            "correct": record.get("correct"),
            "model_version": str(record.get("model_version", "")),
            "result_source": str(record.get("result_source", "")),
            "result_status": str(record.get("result_status", "")),
            "settled_at": str(record.get("settled_at", "")),
            "correction_count": record.get("correction_count"),
            "corrected_at": str(record.get("corrected_at", "")),
        })
    return rows
