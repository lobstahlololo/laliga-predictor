"""Automated upcoming-fixture discovery with a pluggable provider.

The module separates three concerns so they can be tested and replaced
independently:

1. :class:`TheSportsDBProvider` / :class:`ApiFootballProvider` fetch a JSON
   payload and normalise it into :class:`FixtureRecord` rows.
2. :func:`merge_fixtures` reconciles a fresh batch with the previously stored
   fixtures (kickoff changes, postponements, duplicates, provider omissions).
3. :func:`refresh_upcoming_fixtures` performs the network call, merges and
   writes ``data/raw/upcoming_fixtures.csv`` atomically. Any failure leaves the
   existing file untouched, so the pipeline falls back to the last known
   fixtures instead of inventing or deleting them.

Provider notes (verified against the public documentation, November 2026):

* TheSportsDB v1 uses a free shared key (``123``; the older development key
  ``3`` still works for many endpoints) and is limited to 30 requests/minute.
  ``eventsnextleague.php?id=4335`` is the free La Liga "next events" endpoint,
  capped at a small number of events and with no historical guarantees. A
  personal/premium key raises the limits and is read from
  ``LALIGA_FOOTBALL_API_KEY``.
* API-Football's free plan allows 100 requests/day (10/minute) and a
  restrictive set of seasons; the next-fixtures endpoint only needs the
  current season. It requires a registered key, supplied through the same
  environment variable.

Neither free tier is assumed to provide historical xG. Historical xG and
player-availability features live in separate, optional files (see
``src.features.optional_features``) and are reported as unavailable when the
repository has no such data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd
import requests

from src.utils.config import (
    FIXTURE_API_BASE_ENV,
    FIXTURE_API_KEY_ENV,
    FIXTURE_DISCOVERY_ENV,
    FIXTURE_LEAGUE_ID_ENV,
    FIXTURE_PROVIDER_ENV,
    UPCOMING_FIXTURES_PATH,
)
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

THESPORTSDB = "thesportsdb"
API_FOOTBALL = "api-football"
DISABLED_PROVIDERS = {"", "none", "off", "disabled", "local"}

REQUEST_TIMEOUT = 15
MAX_RETRIES = 2
USER_AGENT = "Mozilla/5.0 (LaLiga-Predictor; fixture discovery)"
FIXTURE_COLUMNS = [
    "fixture_id",
    "provider",
    "provider_event_id",
    "home_team",
    "away_team",
    "kickoff_utc",
    "match_date",
    "status",
    "source",
    "fetched_at",
]
CANCELLED_STATUSES = {"cancelled", "canceled", "abandoned", "finished", "ft", "aet", "pen"}
POSTPONED_STATUSES = {"postponed", "pst", "suspended", "delayed"}


class FixtureProviderError(RuntimeError):
    """Raised when a provider cannot return a usable fixture payload."""


class FixtureProvider(Protocol):
    """Interface implemented by fixture sources."""

    name: str

    def fetch_upcoming(self) -> list["FixtureRecord"]:
        ...


# ---------------------------------------------------------------------------
# Records and identity
# ---------------------------------------------------------------------------

def stable_fixture_id(home_team: str, away_team: str, match_date: str) -> str:
    """Deterministic ID from canonical teams and the calendar date.

    Kickoff *times* can move without changing the fixture identity; a different
    date (a genuinely rescheduled match) yields a different ID. Provider event
    IDs are stored separately and used as the primary reconciliation key.
    """
    home = canonical_team(home_team)
    away = canonical_team(away_team)
    date_part = str(match_date or "").strip() or "TBD"
    digest = hashlib.sha1(f"{home}|{away}|{date_part}".encode("utf-8")).hexdigest()
    return f"fx_{digest[:12]}"


def normalize_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "scheduled"
    if any(token in raw for token in ("postpon", "suspend", "delay")):
        return "postponed"
    if any(token in raw for token in ("cancel", "abandon")):
        return "cancelled"
    if any(token in raw for token in ("finish", "ended", "ft", "aet", "pen")):
        return "finished"
    return "scheduled"


@dataclass(frozen=True)
class FixtureRecord:
    home_team: str
    away_team: str
    kickoff_utc: str = ""
    provider: str = ""
    provider_event_id: str = ""
    status: str = "scheduled"
    source: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def match_date(self) -> str:
        if self.kickoff_utc:
            parsed = pd.to_datetime(self.kickoff_utc, errors="coerce", utc=True)
            if pd.notna(parsed):
                return f"{parsed:%Y-%m-%d}"
        return ""

    @property
    def fixture_id(self) -> str:
        return stable_fixture_id(self.home_team, self.away_team, self.match_date)

    def as_row(self) -> dict:
        row = asdict(self)
        row["home_team"] = canonical_team(self.home_team)
        row["away_team"] = canonical_team(self.away_team)
        row["fixture_id"] = self.fixture_id
        row["match_date"] = self.match_date
        row["status"] = normalize_status(self.status)
        return row


def _iso_utc(date_value: Any, time_value: Any = None, timestamp: Any = None) -> str:
    """Combine provider fields into an ISO-8601 UTC string, or ``""``."""
    for candidate in (timestamp,):
        if candidate:
            parsed = pd.to_datetime(candidate, errors="coerce", utc=True)
            if pd.notna(parsed):
                return f"{parsed:%Y-%m-%dT%H:%M:%SZ}"
    if not date_value:
        return ""
    date_text = str(date_value).strip()
    time_text = str(time_value or "").strip()
    if not time_text:
        parsed = pd.to_datetime(date_text, errors="coerce", utc=True)
        if pd.isna(parsed):
            return ""
        has_clock = "T" in date_text or ":" in date_text
        return f"{parsed:%Y-%m-%dT%H:%M:%SZ}" if has_clock else f"{parsed:%Y-%m-%dT00:00:00Z}"
    parsed = pd.to_datetime(f"{date_text} {time_text}", errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return f"{parsed:%Y-%m-%dT%H:%M:%SZ}"


# ---------------------------------------------------------------------------
# Rate limiting / HTTP
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window limiter shared by every call from one provider."""

    def __init__(
        self,
        max_calls: int = 30,
        period_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_calls = max(1, int(max_calls))
        self.period_seconds = float(period_seconds)
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = self._clock()
        while self._calls and now - self._calls[0] >= self.period_seconds:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            wait = self.period_seconds - (now - self._calls[0]) + 0.05
            if wait > 0:
                self._sleep(wait)
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.period_seconds:
                self._calls.popleft()
        self._calls.append(now)


def successful_status(status: str) -> bool:
    """Whether an attempt produced a usable provider response."""
    return str(status).lower() in {"ok", "success", "fallback_empty", "partial"}


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    limiter: RateLimiter | None = None,
    timeout: float = REQUEST_TIMEOUT,
    retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """GET JSON with rate limiting and bounded retries on 429/5xx."""
    limiter = limiter or RateLimiter()
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})
    last_error = ""
    for attempt in range(retries + 1):
        limiter.acquire()
        try:
            response = requests.get(url, timeout=timeout, headers=request_headers)
        except requests.RequestException as exc:
            last_error = f"network error: {exc}"
        else:
            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise FixtureProviderError(f"Provider returned invalid JSON: {exc}") from exc
                if not isinstance(payload, dict):
                    raise FixtureProviderError("Provider returned a non-object JSON payload.")
                return payload
            last_error = f"HTTP {response.status_code}"
            if response.status_code not in {429, 500, 502, 503, 504}:
                raise FixtureProviderError(f"Provider request failed: {last_error} ({url})")
            if attempt < retries:
                retry_after = 0.0
                try:
                    retry_after = float(response.headers.get("Retry-After", "") or 0)
                except (TypeError, ValueError):
                    retry_after = 0.0
                sleep(min(max(retry_after, 0.0), 30.0) or min(2 ** attempt, 8))
        if attempt < retries:
            sleep(min(2 ** attempt, 8))
    raise FixtureProviderError(f"Provider request failed after {retries + 1} attempts: {last_error}")


# ---------------------------------------------------------------------------
# TheSportsDB
# ---------------------------------------------------------------------------

def _clean_team(value: Any) -> str:
    return canonical_team(str(value or "").strip())


def records_from_thesportsdb(payload: dict) -> list[FixtureRecord]:
    """Normalise a TheSportsDB ``eventsnextleague`` payload."""
    if not isinstance(payload, dict):
        raise FixtureProviderError("TheSportsDB payload is not an object.")
    events = payload.get("events")
    if events is None:
        # The free API returns null when there are no upcoming events.
        return []
    if not isinstance(events, list):
        raise FixtureProviderError("TheSportsDB payload has an unexpected 'events' field.")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[FixtureRecord] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        home = _clean_team(event.get("strHomeTeam"))
        away = _clean_team(event.get("strAwayTeam"))
        if not home or not away:
            logger.warning("Skipping fixture without both team names: %r", event.get("strEvent"))
            continue
        if home.lower() == away.lower():
            logger.warning("Skipping fixture with identical teams: %s vs %s", home, away)
            continue
        kickoff = _iso_utc(event.get("dateEvent"), event.get("strTime"), event.get("strTimestamp"))
        status = normalize_status(event.get("strStatus") or event.get("strPostponed"))
        records.append(FixtureRecord(
            home_team=home,
            away_team=away,
            kickoff_utc=kickoff,
            provider=THESPORTSDB,
            provider_event_id=str(event.get("idEvent") or "").strip(),
            status=status,
            source=f"{THESPORTSDB}:eventsnextleague",
            fetched_at=fetched_at,
        ))
    return records


class TheSportsDBProvider:
    """Free-tier La Liga fixtures from TheSportsDB v1."""

    name = THESPORTSDB

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        league_id: str | None = None,
        limiter: RateLimiter | None = None,
        timeout: float = REQUEST_TIMEOUT,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = (api_key or os.environ.get(FIXTURE_API_KEY_ENV, "") or "123").strip()
        self.base_url = (
            base_url or os.environ.get(FIXTURE_API_BASE_ENV, "") or "https://www.thesportsdb.com/api/v1/json"
        ).rstrip("/")
        self.league_id = (league_id or os.environ.get(FIXTURE_LEAGUE_ID_ENV, "") or "4335").strip()
        self.limiter = limiter or RateLimiter(max_calls=30, period_seconds=60, sleep=sleeper)
        self.timeout = timeout
        self._sleep = sleeper

    def fetch_upcoming(self) -> list[FixtureRecord]:
        url = f"{self.base_url}/{self.api_key}/eventsnextleague.php?id={self.league_id}"
        payload = _request_json(
            url, limiter=self.limiter, timeout=self.timeout, sleep=self._sleep
        )
        records = records_from_thesportsdb(payload)
        logger.info("TheSportsDB returned %d upcoming La Liga event(s).", len(records))
        return records


# ---------------------------------------------------------------------------
# API-Football (optional second provider)
# ---------------------------------------------------------------------------

def records_from_api_football(payload: dict) -> list[FixtureRecord]:
    """Normalise an API-Football ``/fixtures?next=`` payload."""
    if not isinstance(payload, dict):
        raise FixtureProviderError("API-Football payload is not an object.")
    errors = payload.get("errors")
    if errors:
        raise FixtureProviderError(f"API-Football reported errors: {errors}")
    fixtures = payload.get("response")
    if fixtures is None:
        return []
    if not isinstance(fixtures, list):
        raise FixtureProviderError("API-Football payload has an unexpected 'response' field.")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records: list[FixtureRecord] = []
    for item in fixtures:
        if not isinstance(item, dict):
            continue
        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        home = _clean_team((teams.get("home") or {}).get("name"))
        away = _clean_team((teams.get("away") or {}).get("name"))
        if not home or not away or home.lower() == away.lower():
            logger.warning("Skipping incomplete API-Football fixture: %r", item.get("fixture"))
            continue
        status_info = fixture.get("status") or {}
        status = normalize_status(status_info.get("short") or status_info.get("long"))
        records.append(FixtureRecord(
            home_team=home,
            away_team=away,
            kickoff_utc=_iso_utc(fixture.get("date")),
            provider=API_FOOTBALL,
            provider_event_id=str(fixture.get("id") or "").strip(),
            status=status,
            source=f"{API_FOOTBALL}:fixtures",
            fetched_at=fetched_at,
        ))
    return records


class ApiFootballProvider:
    """API-Football free-tier fixtures (requires a registered API key)."""

    name = API_FOOTBALL

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        league_id: str | None = None,
        limiter: RateLimiter | None = None,
        timeout: float = REQUEST_TIMEOUT,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = (api_key or os.environ.get(FIXTURE_API_KEY_ENV, "") or "").strip()
        if not self.api_key:
            raise FixtureProviderError(
                f"{API_FOOTBALL} requires {FIXTURE_API_KEY_ENV} (free plan: 100 requests/day)."
            )
        self.base_url = (
            base_url or os.environ.get(FIXTURE_API_BASE_ENV, "") or "https://v3.football.api-sports.io"
        ).rstrip("/")
        self.league_id = (league_id or os.environ.get(FIXTURE_LEAGUE_ID_ENV, "") or "140").strip()
        self.limiter = limiter or RateLimiter(max_calls=10, period_seconds=60, sleep=sleeper)
        self.timeout = timeout
        self._sleep = sleeper

    def fetch_upcoming(self) -> list[FixtureRecord]:
        url = f"{self.base_url}/fixtures?league={self.league_id}&next=20"
        payload = _request_json(
            url,
            headers={"x-apisports-key": self.api_key},
            limiter=self.limiter,
            timeout=self.timeout,
            sleep=self._sleep,
        )
        records = records_from_api_football(payload)
        logger.info("API-Football returned %d upcoming La Liga event(s).", len(records))
        return records


# ---------------------------------------------------------------------------
# Storage / reconciliation
# ---------------------------------------------------------------------------

def provider_from_environment() -> FixtureProvider | None:
    """Build the configured provider, or ``None`` when discovery is disabled."""
    if os.environ.get(FIXTURE_DISCOVERY_ENV, "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    name = os.environ.get(FIXTURE_PROVIDER_ENV, THESPORTSDB).strip().lower() or THESPORTSDB
    if name in DISABLED_PROVIDERS:
        return None
    if name == THESPORTSDB:
        return TheSportsDBProvider()
    if name == API_FOOTBALL:
        return ApiFootballProvider()
    raise FixtureProviderError(
        f"Unknown fixture provider {name!r}. Use {THESPORTSDB!r} or {API_FOOTBALL!r}."
    )


def load_fixture_rows(path: Path = UPCOMING_FIXTURES_PATH) -> pd.DataFrame:
    """Read stored fixtures, tolerating legacy/minimal CSVs."""
    if not path.exists():
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001 - corrupt file behaves like an empty store
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    for column in FIXTURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    if "match_date" not in frame.columns or frame["match_date"].eq("").all():
        frame["match_date"] = frame["kickoff_utc"].map(
            lambda value: f"{pd.to_datetime(value, errors='coerce', utc=True):%Y-%m-%d}"
            if str(value).strip() else ""
        )
    return frame[FIXTURE_COLUMNS].copy()


def merge_fixtures(
    existing: pd.DataFrame,
    incoming: list[FixtureRecord],
    *,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    """Reconcile a fresh provider batch with the stored fixture list.

    Rules:

    * an event seen before (same provider event ID, else same stable fixture
      ID) is replaced in place, so changed kickoff times/status win;
    * duplicates inside one batch collapse to a single row;
    * stored fixtures the provider no longer returns are kept (the free
      ``next events`` feed is a window, not an authoritative cancellation list);
    * a cancelled/postponed status returned by the provider is stored, never
      dropped or silently turned into a scheduled fixture.
    """
    if existing is None:
        existing = pd.DataFrame(columns=FIXTURE_COLUMNS)
    rows: list[dict] = existing.to_dict(orient="records") if not existing.empty else []
    stamp = fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    def row_key(row: dict) -> tuple[str, str]:
        return (
            str(row.get("provider_event_id", "")).strip(),
            str(row.get("fixture_id", "")).strip(),
        )

    index_by_provider: dict[tuple[str, str], int] = {}
    index_by_pairing: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        provider_id, fixture_id = row_key(row)
        if provider_id:
            index_by_provider[(str(row.get("provider", "")), provider_id)] = i
        if fixture_id:
            index_by_provider[("", fixture_id)] = i
        index_by_pairing.setdefault(
            (str(row.get("home_team", "")), str(row.get("away_team", ""))), []
        ).append(i)

    seen_batch: set[tuple[str, str]] = set()
    seen_fixture_provider: dict[tuple[str, str], str] = {}
    for record in incoming:
        if not isinstance(record, FixtureRecord):
            raise TypeError(f"incoming fixtures must be FixtureRecord, got {type(record)!r}")
        row = record.as_row()
        if not row["kickoff_utc"] and not row["match_date"]:
            logger.warning(
                "Keeping fixture %s vs %s with an unknown kickoff (status=%s).",
                row["home_team"], row["away_team"], row["status"],
            )
        row["fetched_at"] = stamp
        provider_key = (str(row.get("provider", "")), str(row.get("provider_event_id", "")).strip())
        fixture_key = ("", str(row["fixture_id"]))
        if (row["status"] == "finished"):
            # A finished event is history; the fixtures file is for upcoming games.
            continue
        if provider_key[1] and provider_key in seen_batch:
            continue
        if (
            fixture_key in seen_batch
            and seen_fixture_provider.get(fixture_key, row.get("provider", ""))
            == row.get("provider", "")
        ):
            continue

        target = index_by_provider.get(provider_key) if provider_key[1] else None
        if target is None:
            fixture_target = index_by_provider.get(fixture_key)
            if fixture_target is not None:
                previous = rows[fixture_target]
                previous_provider = str(previous.get("provider", "")).strip()
                incoming_provider = str(row.get("provider", "")).strip()
                previous_kickoff = str(previous.get("kickoff_utc", "")).strip()
                incoming_kickoff = str(row.get("kickoff_utc", "")).strip()
                # Two providers disagreeing on kickoff time is preserved as a
                # conflict row for the audit rather than silently resolved.
                if (
                    previous_provider
                    and incoming_provider
                    and previous_provider != incoming_provider
                    and previous_kickoff
                    and incoming_kickoff
                    and previous_kickoff != incoming_kickoff
                ):
                    target = None
                else:
                    target = fixture_target
            elif not str(row.get("kickoff_utc", "")).strip():
                # An incomplete payload with no date can still be the same
                # fixture. Match by pairing only when unambiguous.
                candidates = index_by_pairing.get(
                    (row["home_team"], row["away_team"]), []
                )
                if len(candidates) == 1:
                    target = candidates[0]
        if target is not None:
            # Preserve a stable fixture ID across a time-only kickoff move, and
            # never let a provider with an incomplete payload erase a known
            # kickoff that another provider already supplied.
            previous = rows[target]
            if not str(row.get("kickoff_utc", "")).strip() and str(
                previous.get("kickoff_utc", "")
            ).strip():
                row["kickoff_utc"] = previous["kickoff_utc"]
                row["match_date"] = previous.get("match_date", "")
            if str(previous.get("fixture_id", "")).strip():
                row["fixture_id"] = previous["fixture_id"]
            rows[target] = row
        else:
            rows.append(row)
            target = len(rows) - 1
            index_by_pairing.setdefault(
                (row["home_team"], row["away_team"]), []
            ).append(target)
        if provider_key[1]:
            index_by_provider[provider_key] = target
            seen_batch.add(provider_key)
        index_by_provider[fixture_key] = target
        seen_batch.add(fixture_key)
        seen_fixture_provider[fixture_key] = str(row.get("provider", ""))

    frame = pd.DataFrame(rows)
    for column in FIXTURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[FIXTURE_COLUMNS]
    frame = frame.drop_duplicates(subset=["fixture_id", "kickoff_utc"], keep="last")
    frame["_sort"] = pd.to_datetime(frame["kickoff_utc"], errors="coerce", utc=True)
    frame = frame.sort_values(["_sort", "fixture_id"], na_position="last").drop(columns="_sort")
    return frame.reset_index(drop=True)


def merge_provider_batches(
    existing: pd.DataFrame,
    batches: dict[str, list[FixtureRecord]],
    *,
    fetched_at: str | None = None,
) -> pd.DataFrame:
    """Safely merge batches from several providers into one fixture store.

    Identity order is: provider event ID (same provider), then the stable
    fixture ID (canonical teams + date). Conflicting kickoff times are kept as
    separate rows rather than silently deleted, so the coverage audit can
    report them. A fixture the providers no longer return is preserved.
    """
    combined: list[FixtureRecord] = []
    for provider, records in batches.items():
        for record in records or []:
            if not isinstance(record, FixtureRecord):
                raise TypeError(
                    f"{provider} batch must contain FixtureRecord rows, "
                    f"got {type(record)!r}"
                )
            combined.append(record)
    return merge_fixtures(existing, combined, fetched_at=fetched_at)


def write_fixtures_atomic(rows: pd.DataFrame, path: Path = UPCOMING_FIXTURES_PATH) -> Path:
    """Write the fixture store through a temporary file, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame = rows.copy()
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


@dataclass
class FixtureDiscoveryResult:
    status: str
    provider: str = ""
    fetched: int = 0
    written: int = 0
    kept: int = 0
    errors: list[str] = field(default_factory=list)
    path: str = str(UPCOMING_FIXTURES_PATH)
    failed_providers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def refresh_upcoming_fixtures(
    path: Path = UPCOMING_FIXTURES_PATH,
    *,
    provider: FixtureProvider | None = None,
    providers: list[FixtureProvider] | None = None,
    strict: bool = False,
    status_path: Path | None = None,
) -> FixtureDiscoveryResult:
    """Fetch, merge and atomically persist upcoming fixtures.

    One provider or several can be supplied. With several providers, a failing
    provider does not discard another provider's successful batch; the result
    is ``status="partial"`` and the stored fixtures are merged safely. If every
    provider fails, the stored file is left untouched and the result is
    ``status="fallback"``. ``strict=True`` re-raises instead.

    When ``status_path`` is supplied, each attempt is recorded for provider
    freshness/failure telemetry (never a secret, never an API key).
    """
    existing = load_fixture_rows(path)
    provider_list = list(providers or ([] if provider is None else [provider]))
    if providers is None and provider is None:
        try:
            provider = provider_from_environment()
        except Exception as exc:  # noqa: BLE001 - disabled/misconfigured provider
            result = FixtureDiscoveryResult(
                status="error",
                provider="",
                kept=len(existing),
                errors=[str(exc)],
                path=str(path),
            )
            if strict:
                raise
            logger.warning("Fixture discovery unavailable: %s", exc)
            return result
    if provider is None and not provider_list:
        return FixtureDiscoveryResult(
            status="disabled", kept=len(existing), path=str(path)
        )
    if not provider_list:
        provider_list = [provider]

    batches: dict[str, list[FixtureRecord]] = {}
    errors: list[str] = []
    failed: list[str] = []
    for current in provider_list:
        name = str(getattr(current, "name", "unknown"))
        try:
            batches[name] = current.fetch_upcoming()
        except Exception as exc:  # noqa: BLE001 - other providers may still work
            logger.warning("Fixture provider %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
            failed.append(name)
            if status_path is not None:
                from src.data.fixture_coverage import ProviderTelemetry

                ProviderTelemetry(Path(status_path)).record(
                    name, "failed", errors=[str(exc)]
                )
        else:
            if status_path is not None:
                from src.data.fixture_coverage import ProviderTelemetry

                ProviderTelemetry(Path(status_path)).record(
                    name, "ok", fetched=len(batches[name]), written=0
                )

    if not batches:
        result = FixtureDiscoveryResult(
            status="fallback",
            provider=",".join(failed),
            kept=len(existing),
            errors=errors or ["all fixture providers failed"],
            path=str(path),
            failed_providers=failed,
        )
        if strict and errors:
            raise FixtureProviderError("; ".join(errors))
        return result

    incoming = [record for records in batches.values() for record in records]
    merged = merge_fixtures(existing, incoming)
    write_fixtures_atomic(merged, path)
    status = "ok" if not failed else "partial"
    logger.info(
        "Fixtures refreshed from %s: %d fetched, %d stored (%d kept from before).",
        ", ".join(batches), len(incoming), len(merged), len(existing),
    )
    return FixtureDiscoveryResult(
        status=status,
        provider=",".join(batches),
        fetched=len(incoming),
        written=len(merged),
        kept=len(existing),
        errors=errors,
        path=str(path),
        failed_providers=failed,
    )


def main() -> int:
    """CLI: refresh the upcoming-fixture store once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    from src.utils.config import PROVIDER_STATUS_PATH

    result = refresh_upcoming_fixtures(status_path=PROVIDER_STATUS_PATH)
    print(json.dumps(result.as_dict(), indent=2))
    if result.status == "ok":
        return 0
    if result.status == "disabled":
        print("Fixture discovery is disabled; existing fixtures were left unchanged.")
        return 0
    print(
        "Fixture discovery did not update the store; the previous fixtures "
        "remain in place."
    )
    return 0 if result.status == "fallback" else 1


if __name__ == "__main__":
    raise SystemExit(main())
