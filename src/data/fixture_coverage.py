"""Fixture coverage and provider-reliability reporting.

TheSportsDB's free ``eventsnextleague`` endpoint and API-Football's
``fixtures?next=`` endpoint are *windows* into the schedule, not authoritative
league calendars. This module measures what was actually returned and refuses
to describe an incomplete window as a complete fixture list.

It reports, without inventing data:

* historical team/season coverage (how many matches each club has);
* teams in the most recent full season that are missing from the current one;
* upcoming-fixture counts, teams covered, missing dates and stale snapshots;
* duplicate fixture IDs and conflicting kickoff/status observations;
* provider freshness, last success, request/failure counts and fixture counts;
* a comparison against a cached full-season calendar when one exists, which is
  the only case where completeness is verified rather than described as unknown.

Reports are written as JSON (machine-readable), CSV (team/season coverage) and
Markdown (human-readable).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.data.football_calendar import (
    CALENDAR_HORIZON_DAYS,
    compare_with_calendar,
    load_calendar,
)
from src.data.fixture_provider import (
    FIXTURE_COLUMNS,
    load_fixture_rows,
    successful_status,
)
from src.data.validation import season_from_date
from src.utils.config import (
    FIXTURE_COVERAGE_CSV_PATH,
    FIXTURE_COVERAGE_JSON_PATH,
    FIXTURE_COVERAGE_REPORT_PATH,
    MIN_FULL_SEASON_MATCHES,
    PROVIDER_STATUS_PATH,
    RAW_PATH,
    STALE_DATA_DAYS,
    UPCOMING_FIXTURES_PATH,
)
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

MAX_TELEMETRY_ATTEMPTS = 200
STATUS_SCHEDULED = "scheduled"
TERMINAL_STATUSES = {"finished", "cancelled", "canceled", "abandoned"}


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")


def _parse_ts(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _display_path(path: Path | str) -> str:
    return Path(path).as_posix()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Provider telemetry
# ---------------------------------------------------------------------------

@dataclass
class ProviderTelemetry:
    """Durable record of provider attempts, freshness and failures.

    A provider outage must be visible in operations without failing the data
    pipeline. Every attempt is appended to a bounded history; the aggregate is
    kept per provider.
    """

    path: Path = field(default_factory=lambda: PROVIDER_STATUS_PATH)

    def load(self) -> dict:
        path = Path(self.path)
        if not path.exists():
            return {"generated_at": None, "providers": {}, "attempts": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt log is reported, not fatal
            logger.warning("Could not read provider telemetry %s: %s", path, exc)
            return {"generated_at": None, "providers": {}, "attempts": []}
        if not isinstance(payload, dict):
            return {"generated_at": None, "providers": {}, "attempts": []}
        payload.setdefault("providers", {})
        payload.setdefault("attempts", [])
        payload.setdefault("generated_at", None)
        return payload

    def record(
        self,
        provider: str,
        status: str,
        *,
        fetched: int = 0,
        written: int = 0,
        errors: Iterable[str] = (),
        now: datetime | None = None,
        fixture_count: int | None = None,
    ) -> dict:
        """Record one fetch attempt and return the updated provider entry."""
        name = str(provider or "unknown").strip() or "unknown"
        stamp = _now_iso(now)
        payload = self.load()
        entry = payload["providers"].setdefault(name, {
            "provider": name,
            "last_attempt_at": None,
            "last_success_at": None,
            "last_status": None,
            "last_fixture_count": 0,
            "total_requests": 0,
            "total_failures": 0,
            "consecutive_failures": 0,
        })
        entry["last_attempt_at"] = stamp
        entry["last_status"] = str(status)
        if fixture_count is not None:
            entry["last_fixture_count"] = int(fixture_count)
        elif fetched or successful_status(status):
            entry["last_fixture_count"] = int(fetched)
        entry["total_requests"] = int(entry.get("total_requests", 0)) + 1
        successful = successful_status(status)
        if successful:
            entry["last_success_at"] = stamp
            entry["consecutive_failures"] = 0
        else:
            entry["total_failures"] = int(entry.get("total_failures", 0)) + 1
            entry["consecutive_failures"] = int(entry.get("consecutive_failures", 0)) + 1
        entry["last_error"] = "; ".join(str(e) for e in errors)[:500]

        payload["attempts"].append({
            "at": stamp,
            "provider": name,
            "status": str(status),
            "fetched": int(fetched),
            "written": int(written),
            "errors": [str(e) for e in errors][:5],
        })
        payload["attempts"] = payload["attempts"][-MAX_TELEMETRY_ATTEMPTS:]
        payload["generated_at"] = stamp
        _atomic_json(payload, Path(self.path))
        return entry

    def summary(self) -> dict:
        payload = self.load()
        return {
            "generated_at": payload.get("generated_at"),
            "providers": payload.get("providers", {}),
            "recent_attempts": payload.get("attempts", [])[-10:],
            "path": _display_path(self.path),
        }


# ---------------------------------------------------------------------------
# Conflict / duplicate detection
# ---------------------------------------------------------------------------

def fixture_frame(fixtures: pd.DataFrame | None) -> pd.DataFrame:
    """Return a normalised fixture frame with every canonical column."""
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    frame = fixtures.copy()
    for column in FIXTURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    for column in ("home_team", "away_team"):
        frame[column] = frame[column].astype(str).str.strip().map(canonical_team)
    frame["provider"] = frame["provider"].astype(str).str.strip()
    frame["fixture_id"] = frame["fixture_id"].astype(str).str.strip()
    frame["kickoff_utc"] = frame["kickoff_utc"].astype(str).str.strip()
    frame["match_date"] = frame["match_date"].astype(str).str.strip()
    return frame[FIXTURE_COLUMNS].reset_index(drop=True)


def detect_fixture_conflicts(fixtures: pd.DataFrame | None) -> pd.DataFrame:
    """Find same-fixture rows that disagree on kickoff or status.

    A provider returning the same match twice with different kickoff times is a
    data-quality signal, not a reason to silently pick one. The merged store
    keeps one row, so the audit reads the stored snapshot plus any batch-level
    evidence supplied by callers.
    """
    frame = fixture_frame(fixtures)
    if frame.empty:
        return pd.DataFrame(columns=[
            "fixture_id", "home_team", "away_team", "match_date",
            "conflict", "values", "providers",
        ])
    frame["_date_key"] = frame["match_date"].where(
        frame["match_date"] != "", frame["kickoff_utc"].str[:10]
    )
    frame["_identity"] = (
        frame["home_team"] + "|" + frame["away_team"] + "|" + frame["_date_key"]
    )
    rows: list[dict] = []
    for identity, group in frame.groupby("_identity", sort=True):
        kickoffs = sorted({v for v in group["kickoff_utc"].tolist() if v})
        statuses = sorted({v.lower() for v in group["status"].tolist() if v})
        providers = sorted({v for v in group["provider"].tolist() if v})
        if len(kickoffs) > 1:
            rows.append({
                "fixture_id": str(group.iloc[0]["fixture_id"]),
                "home_team": str(group.iloc[0]["home_team"]),
                "away_team": str(group.iloc[0]["away_team"]),
                "match_date": str(group.iloc[0]["_date_key"]),
                "conflict": "kickoff_time",
                "values": " | ".join(kickoffs),
                "providers": ",".join(providers),
            })
        if len(statuses) > 1 and any(s in TERMINAL_STATUSES for s in statuses):
            rows.append({
                "fixture_id": str(group.iloc[0]["fixture_id"]),
                "home_team": str(group.iloc[0]["home_team"]),
                "away_team": str(group.iloc[0]["away_team"]),
                "match_date": str(group.iloc[0]["_date_key"]),
                "conflict": "status",
                "values": " | ".join(statuses),
                "providers": ",".join(providers),
            })
    return pd.DataFrame(rows)


def duplicate_fixtures(fixtures: pd.DataFrame | None) -> pd.DataFrame:
    """Rows sharing a fixture ID (the store normally de-duplicates these)."""
    frame = fixture_frame(fixtures)
    if frame.empty:
        return frame
    duplicated = frame[frame["fixture_id"].duplicated(keep=False)]
    return duplicated.sort_values(["fixture_id", "kickoff_utc"]).reset_index(drop=True)


def check_fixture_ids_stable(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame | None,
) -> list[str]:
    """Return fixture IDs whose canonical home/away/date identity changed.

    Needing this check is a bug signal: the stable ID is derived only from
    canonical teams and the calendar date, so a changed ID for the same match
    means team canonicalisation or date parsing drifted.
    """
    left = fixture_frame(existing)
    right = fixture_frame(incoming)
    if left.empty or right.empty:
        return []
    left_map = {
        (row.home_team, row.away_team, row.match_date): row.fixture_id
        for row in left.itertuples(index=False)
    }
    problems: list[str] = []
    for row in right.itertuples(index=False):
        key = (row.home_team, row.away_team, row.match_date)
        previous = left_map.get(key)
        if previous and previous != row.fixture_id:
            problems.append(f"{key}: {previous} -> {row.fixture_id}")
    return problems


# ---------------------------------------------------------------------------
# Coverage audit
# ---------------------------------------------------------------------------

def _read_raw(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - report, never crash an audit
        logger.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("home_team", "away_team"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip().map(canonical_team)
    return frame


def _team_season_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or not {"date", "home_team", "away_team"} <= set(raw.columns):
        return pd.DataFrame(columns=["season", "team", "matches"])
    frame = raw.dropna(subset=["date"]).copy()
    if "season" not in frame.columns:
        frame["season"] = frame["date"].map(season_from_date)
    long = pd.concat([
        frame[["season", "home_team"]].rename(columns={"home_team": "team"}),
        frame[["season", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    return (
        long.groupby(["season", "team"], as_index=False)
        .size()
        .rename(columns={"size": "matches"})
        .sort_values(["season", "team"])
        .reset_index(drop=True)
    )


def _fixture_season(row: pd.Series) -> str:
    value = str(row.get("match_date", "")).strip()
    if not value:
        kickoff = _parse_ts(row.get("kickoff_utc"))
        if kickoff is not None:
            value = f"{kickoff:%Y-%m-%d}"
    stamp = _parse_ts(value)
    return season_from_date(stamp) if stamp is not None else ""


def _upcoming_team_counts(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame(columns=["season", "team", "upcoming_fixtures"])
    frame = fixtures.copy()
    frame["season"] = frame.apply(_fixture_season, axis=1)
    long = pd.concat([
        frame[["season", "home_team"]].rename(columns={"home_team": "team"}),
        frame[["season", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    return (
        long.groupby(["season", "team"], as_index=False)
        .size()
        .rename(columns={"size": "upcoming_fixtures"})
    )


def audit_fixture_coverage(
    raw: pd.DataFrame | None = None,
    fixtures: pd.DataFrame | None = None,
    *,
    now: datetime | None = None,
    telemetry: ProviderTelemetry | None = None,
    stale_days: int = STALE_DATA_DAYS,
    full_season_threshold: int = MIN_FULL_SEASON_MATCHES,
    calendar: pd.DataFrame | None = None,
    calendar_meta: dict | None = None,
    horizon_days: int = CALENDAR_HORIZON_DAYS,
) -> tuple[dict, pd.DataFrame]:
    """Measure historical and upcoming fixture coverage.

    Returns ``(report, team_season_coverage)``. Without a ``calendar`` the
    report marks schedule completeness as unknown: a provider window cannot
    prove that all fixtures exist and the audit never fabricates the missing
    ones. Supplying a verified full-season calendar upgrades completeness to
    ``verified_against_calendar`` and lists exactly which fixtures are absent.
    """
    raw = _read_raw(RAW_PATH) if raw is None else raw.copy()
    fixtures = load_fixture_rows() if fixtures is None else fixture_frame(fixtures)
    fixtures = fixture_frame(fixtures)
    now = now or datetime.now(timezone.utc)
    warnings: list[str] = []

    team_season = _team_season_table(raw)
    season_summary: list[dict] = []
    if not team_season.empty:
        if "season" in raw.columns:
            raw_seasons = raw["season"].astype(str)
        elif "date" in raw.columns:
            raw_seasons = raw["date"].map(
                lambda d: season_from_date(d) if pd.notna(d) else ""
            )
        else:
            raw_seasons = pd.Series([""] * len(raw), index=raw.index)
        fixture_counts = raw_seasons.value_counts()
        for season, group in team_season.groupby("season", sort=True):
            teams = sorted(group["team"].unique().tolist())
            fixtures_this_season = raw.loc[raw_seasons.eq(season)]
            dates = (
                fixtures_this_season["date"] if "date" in raw
                else pd.Series(dtype="datetime64[ns]")
            )
            season_summary.append({
                "season": season,
                "matches": int(fixture_counts.get(season, 0)),
                "teams": len(teams),
                "team_list": teams,
                "date_min": str(dates.min().date()) if len(dates) and pd.notna(dates.min()) else None,
                "date_max": str(dates.max().date()) if len(dates) and pd.notna(dates.max()) else None,
                "unexpected_size": False,
            })
    else:
        warnings.append("No historical match data was available for coverage.")

    # Season sizes: a full La Liga season is 380 matches. An in-progress season
    # is expected to be smaller; only flag a *full* season outside the range.
    for entry in season_summary:
        matches = entry["matches"]
        entry["unexpected_size"] = bool(
            matches > 0
            and matches < full_season_threshold
            and entry.get("date_max")
            and pd.Timestamp(entry["date_max"]) < pd.Timestamp(now.date()) - pd.Timedelta(days=60)
        )

    full_seasons = [
        entry for entry in season_summary
        if entry["matches"] >= full_season_threshold
    ]
    reference = full_seasons[-1] if full_seasons else (season_summary[-1] if season_summary else None)
    reference_season = reference["season"] if reference else None
    reference_teams = set(reference["team_list"]) if reference else set()
    latest = season_summary[-1] if season_summary else None
    current_teams = set(latest["team_list"]) if latest else set()
    missing_teams = sorted(reference_teams - current_teams) if latest else []
    promoted_teams = sorted(current_teams - reference_teams) if latest else []

    # Team/season coverage, augmented with upcoming fixture counts.
    coverage = team_season.copy()
    upcoming_counts = _upcoming_team_counts(fixtures)
    if coverage.empty and not upcoming_counts.empty:
        coverage = upcoming_counts.copy()
    elif not coverage.empty:
        coverage = coverage.merge(
            upcoming_counts, on=["season", "team"], how="outer"
        )
    for column in ("matches", "upcoming_fixtures"):
        if column in coverage.columns:
            coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0).astype(int)
    if coverage.empty:
        coverage = pd.DataFrame(columns=["season", "team", "matches", "upcoming_fixtures"])
    coverage["has_upcoming_fixture"] = (
        coverage["upcoming_fixtures"].fillna(0).astype(int) > 0
    )

    # Upcoming window.
    upcoming = {
        "n": int(len(fixtures)),
        "providers": sorted(
            p for p in fixtures["provider"].unique().tolist() if str(p).strip()
        ) if not fixtures.empty else [],
        "status_counts": {},
        "teams_covered": 0,
        "teams_without_fixtures": [],
        "missing_dates": 0,
        "duplicate_fixture_ids": 0,
        "conflicting_fixtures": 0,
        "unresolved_scheduled": 0,
        "oldest_fetched_at": None,
        "newest_fetched_at": None,
        "stale_snapshots": 0,
        "schedule_completeness": "unknown_partial_provider_window",
        "expected_fixtures": None,
    }
    if not fixtures.empty:
        statuses = fixtures["status"].astype(str).str.lower().str.strip()
        upcoming["status_counts"] = {
            str(k): int(v) for k, v in statuses.value_counts().items()
        }
        covered = set(fixtures["home_team"]) | set(fixtures["away_team"])
        upcoming["teams_covered"] = len(covered)
        upcoming["teams_without_fixtures"] = sorted(current_teams - covered) if current_teams else []
        upcoming["missing_dates"] = int(
            (
                fixtures["match_date"].str.strip().eq("")
                & fixtures["kickoff_utc"].str.strip().eq("")
            ).sum()
        )
        duplicates = duplicate_fixtures(fixtures)
        upcoming["duplicate_fixture_ids"] = int(
            duplicates["fixture_id"].nunique() if not duplicates.empty else 0
        )
        conflicts = detect_fixture_conflicts(fixtures)
        upcoming["conflicting_fixtures"] = int(
            conflicts["fixture_id"].nunique() if not conflicts.empty else 0
        )
        upcoming["unresolved_scheduled"] = int(
            statuses.isin({"scheduled", "postponed"}).sum()
        )
        stamps = fixtures["fetched_at"].map(_parse_ts).dropna()
        if not stamps.empty:
            upcoming["oldest_fetched_at"] = f"{stamps.min():%Y-%m-%dT%H:%M:%SZ}"
            upcoming["newest_fetched_at"] = f"{stamps.max():%Y-%m-%dT%H:%M:%SZ}"
            cutoff = pd.Timestamp(now) - pd.Timedelta(days=stale_days)
            upcoming["stale_snapshots"] = int((stamps < cutoff).sum())
        if upcoming["missing_dates"]:
            warnings.append(
                f"{upcoming['missing_dates']} stored fixture(s) have no kickoff date; "
                "they are kept as TBD rather than invented."
            )
        if upcoming["duplicate_fixture_ids"]:
            warnings.append(
                f"{upcoming['duplicate_fixture_ids']} fixture ID(s) appear more than once."
            )
        if upcoming["conflicting_fixtures"]:
            warnings.append(
                f"{upcoming['conflicting_fixtures']} fixture(s) have conflicting kickoff/status rows."
            )
        if upcoming["stale_snapshots"]:
            warnings.append(
                f"{upcoming['stale_snapshots']} upcoming fixture snapshot(s) are older than "
                f"{stale_days} days; refresh before using them."
            )
        if upcoming["teams_without_fixtures"]:
            warnings.append(
                "No upcoming fixture was returned for: "
                + ", ".join(upcoming["teams_without_fixtures"])
                + ". The provider window may be incomplete; this is not proof that "
                  "the fixtures do not exist."
            )
    else:
        warnings.append(
            "No upcoming fixtures are stored. Coverage is unknown, not zero."
        )

    # Calendar verification: only a complete season calendar can prove that a
    # fixture is missing rather than merely absent from a provider window.
    upcoming["calendar_comparison"] = None
    upcoming["calendar_source"] = None
    upcoming["calendar_generated_at"] = None
    has_calendar = calendar is not None and len(calendar) > 0
    if has_calendar:
        comparison = compare_with_calendar(
            fixtures, calendar, now=now, horizon_days=horizon_days
        )
        upcoming["calendar_comparison"] = comparison
        upcoming["calendar_source"] = (calendar_meta or {}).get("source")
        upcoming["calendar_generated_at"] = (calendar_meta or {}).get("fetched_at")
        upcoming["expected_fixtures"] = comparison["calendar_upcoming"]
        upcoming["schedule_completeness"] = "verified_against_calendar"
        if comparison["missing"]:
            warnings.append(
                f"{comparison['missing']} upcoming fixture(s) in the season calendar "
                f"are missing from the store (coverage "
                f"{comparison['coverage_fraction']:.1%})."
                if comparison["coverage_fraction"] is not None
                else f"{comparison['missing']} upcoming fixture(s) are missing from the store."
            )
        if comparison["teams_without_fixtures"]:
            warnings.append(
                "Calendar verification found no stored fixture for: "
                + ", ".join(comparison["teams_without_fixtures"])
                + "."
            )
        if comparison["rescheduled"]:
            warnings.append(
                f"{comparison['rescheduled']} stored fixture(s) disagree with the "
                "calendar date; they are reported as rescheduled, not merged silently."
            )

    if missing_teams:
        warnings.append(
            f"Teams present in reference season {reference_season} but absent from "
            f"the latest season: {', '.join(missing_teams)}."
        )

    provider_summary = (telemetry or ProviderTelemetry()).summary()
    providers = provider_summary.get("providers", {})
    upcoming["provider_coverage_history"] = [
        {
            "provider": attempt.get("provider"),
            "at": attempt.get("at"),
            "status": attempt.get("status"),
            "fetched": attempt.get("fetched"),
        }
        for attempt in provider_summary.get("recent_attempts", [])
    ]
    for name, entry in providers.items():
        consecutive = int(entry.get("consecutive_failures", 0) or 0)
        if consecutive >= 3:
            warnings.append(
                f"Fixture provider {name!r} has {consecutive} consecutive failed attempts."
            )

    unexpected = [entry["season"] for entry in season_summary if entry["unexpected_size"]]
    if unexpected:
        warnings.append(
            "Season size looks incomplete relative to a full La Liga season: "
            + ", ".join(unexpected)
        )

    report = {
        "generated_at": _now_iso(now),
        "status": "warning" if warnings else "ok",
        "data_sources": {
            "historical": _display_path(RAW_PATH),
            "upcoming": _display_path(UPCOMING_FIXTURES_PATH),
        },
        "historical": {
            "rows": int(len(raw)),
            "seasons": season_summary,
            "reference_season": reference_season,
            "reference_teams": sorted(reference_teams),
            "latest_season": latest["season"] if latest else None,
            "latest_season_teams": sorted(current_teams),
            "missing_teams": missing_teams,
            "promoted_teams": promoted_teams,
            "dates": {
                "min": str(raw["date"].min().date()) if "date" in raw and raw["date"].notna().any() else None,
                "max": str(raw["date"].max().date()) if "date" in raw and raw["date"].notna().any() else None,
            },
        },
        "upcoming": upcoming,
        "provider_telemetry": provider_summary,
        "thresholds": {
            "full_season_matches": full_season_threshold,
            "stale_days": stale_days,
        },
        "warnings": warnings,
        "notes": [
            "Provider next-event endpoints are partial windows; a missing fixture "
            "is reported as unknown, never fabricated.",
            "A fixture ID is derived from canonical teams and the calendar date, so "
            "a kickoff-time change preserves identity.",
        ],
    }
    return report, coverage


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No data._"
    columns = [c for c in columns if c in frame.columns]
    if not columns:
        return "_No data._"
    part = frame.head(max_rows) if max_rows else frame
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in part.iterrows():
        values = ["" if pd.isna(row[c]) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_coverage_reports(
    report: dict,
    coverage: pd.DataFrame,
    *,
    json_path: Path = FIXTURE_COVERAGE_JSON_PATH,
    csv_path: Path = FIXTURE_COVERAGE_CSV_PATH,
    markdown_path: Path = FIXTURE_COVERAGE_REPORT_PATH,
) -> list[Path]:
    """Write JSON, CSV and Markdown coverage artifacts."""
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    coverage.to_csv(csv_path, index=False)

    seasons = pd.DataFrame(report["historical"]["seasons"])
    if not seasons.empty:
        seasons = seasons.drop(columns=["team_list"], errors="ignore")
    lines = [
        "# La Liga Predictor — Fixture Coverage",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: **{report['status']}**",
        "",
        "## Historical coverage",
        "",
        _markdown_table(
            seasons,
            ["season", "matches", "teams", "date_min", "date_max", "unexpected_size"],
        ),
        "",
        f"- Reference season: `{report['historical']['reference_season']}` "
        f"({len(report['historical']['reference_teams'])} teams).",
        f"- Latest season: `{report['historical']['latest_season']}`.",
        f"- Teams missing from the latest season: "
        f"{', '.join(report['historical']['missing_teams']) or 'none'}.",
        f"- Teams new relative to the reference season: "
        f"{', '.join(report['historical']['promoted_teams']) or 'none'}.",
        "",
        "## Upcoming fixture window",
        "",
        f"- Stored fixtures: **{report['upcoming']['n']}** from "
        f"{', '.join(report['upcoming']['providers']) or 'no provider'}.",
        f"- Status counts: `{report['upcoming']['status_counts']}`.",
        f"- Teams covered: {report['upcoming']['teams_covered']}.",
        f"- Missing kickoff dates: {report['upcoming']['missing_dates']}.",
        f"- Duplicate fixture IDs: {report['upcoming']['duplicate_fixture_ids']}.",
        f"- Conflicting kickoff/status fixtures: {report['upcoming']['conflicting_fixtures']}.",
        f"- Oldest snapshot: {report['upcoming']['oldest_fetched_at']}.",
        f"- Schedule completeness: **{report['upcoming']['schedule_completeness']}** "
        "(a provider window is not an authoritative league calendar).",
        "",
        *_calendar_markdown(report["upcoming"]),
        "",
        "## Provider telemetry",
        "",
        _markdown_table(
            pd.DataFrame(report["provider_telemetry"]["providers"].values()),
            ["provider", "last_attempt_at", "last_success_at", "last_status",
             "last_fixture_count", "total_requests", "total_failures",
             "consecutive_failures"],
        ),
        "",
        "## Team/season coverage (last rows)",
        "",
        _markdown_table(
            coverage.sort_values(["season", "team"], ascending=[False, True]),
            ["season", "team", "matches", "upcoming_fixtures", "has_upcoming_fixture"],
            max_rows=40,
        ),
        "",
        "## Warnings",
        "",
        "\n".join(f"- {warning}" for warning in report["warnings"]) or "- None.",
        "",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, csv_path, markdown_path]


def _calendar_markdown(upcoming: dict) -> list[str]:
    """Markdown section for the calendar comparison (or its absence)."""
    comparison = upcoming.get("calendar_comparison")
    if not comparison:
        return [
            "## Season calendar comparison",
            "",
            "_No verified season calendar is cached, so completeness stays "
            "`unknown_partial_provider_window`. Run "
            "`python -m src.data.football_calendar --refresh` to enable verification._",
        ]
    lines = [
        "## Season calendar comparison",
        "",
        f"- Calendar source: `{upcoming.get('calendar_source') or 'unknown'}` "
        f"(fetched {upcoming.get('calendar_generated_at') or 'unknown'}).",
        f"- Calendar fixtures still to play: {comparison['calendar_upcoming']} "
        f"(horizon {comparison['horizon_days']} days: "
        f"{comparison['within_horizon']['calendar_upcoming']}).",
        f"- Stored upcoming fixtures: {comparison['stored_upcoming']}.",
        f"- Matched: {comparison['matched']}; missing: {comparison['missing']}; "
        f"unexpected: {comparison['unexpected']}; rescheduled: {comparison['rescheduled']}.",
        f"- Coverage: **{comparison['coverage_fraction']:.1%}** of the calendar "
        if comparison["coverage_fraction"] is not None else "- Coverage: unknown.",
        f"- Teams without a stored fixture: "
        f"{', '.join(comparison['teams_without_fixtures']) or 'none'}.",
        "",
        "### Missing fixtures",
        "",
        _markdown_table(
            pd.DataFrame(comparison["missing_fixtures"]),
            ["home_team", "away_team", "match_date", "kickoff_utc"],
            max_rows=20,
        ),
    ]
    return lines


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    calendar, calendar_meta = load_calendar()
    report, coverage = audit_fixture_coverage(
        calendar=calendar if not calendar.empty else None,
        calendar_meta=calendar_meta,
    )
    paths = write_coverage_reports(report, coverage)
    print(json.dumps({
        "status": report["status"],
        "historical_rows": report["historical"]["rows"],
        "upcoming_fixtures": report["upcoming"]["n"],
        "warnings": report["warnings"],
        "reports": [str(path) for path in paths],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
