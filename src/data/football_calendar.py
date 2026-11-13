"""Season-calendar acquisition and completeness verification.

TheSportsDB's free ``eventsnextleague`` endpoint returns a single event and
API-Football returns a bounded "next N" window, so neither can prove that every
La Liga fixture was discovered. ``openfootball/football.json`` publishes the
complete season (fixtures and results) as CC0 JSON with no key, which turns
coverage into a verifiable question: which fixtures does the store hold, and
which does the calendar say exist?

This module never writes the fixture store. It fetches and caches the calendar,
then compares it with whatever the provider layer stored, so a partial provider
window is still reported as partial and missing fixtures are listed by name
rather than invented or silently accepted.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.fixture_provider import (
    FIXTURE_COLUMNS,
    OPENFOOTBALL,
    OpenFootballProvider,
    records_from_openfootball,
    season_slug,
)
from src.utils.config import SEASON_CALENDAR_PATH
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

CALENDAR_HORIZON_DAYS = 14
TERMINAL_STATUSES = {"finished", "cancelled", "canceled", "abandoned"}
MAX_LISTED_FIXTURES = 50


# ---------------------------------------------------------------------------
# Calendar acquisition and cache
# ---------------------------------------------------------------------------

def calendar_frame(payload: dict) -> pd.DataFrame:
    """Normalise a raw OpenFootball payload into fixture rows."""
    rows = [record.as_row() for record in records_from_openfootball(payload)]
    return pd.DataFrame(rows, columns=FIXTURE_COLUMNS)


def save_calendar(
    payload: dict,
    *,
    path: Path | str = SEASON_CALENDAR_PATH,
    source: str = "",
    season: str = "",
    fetched_at: str | None = None,
) -> Path:
    """Atomically cache the raw calendar with provenance metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "source": source or f"openfootball:{season or season_slug()}:es.1",
        "season": season or season_slug(),
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "payload": payload,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(wrapper, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_calendar(path: Path | str = SEASON_CALENDAR_PATH) -> tuple[pd.DataFrame, dict]:
    """Load the cached calendar.

    Returns ``(frame, meta)``. A missing or unreadable cache yields an empty
    frame and empty metadata, which callers must report as "no calendar
    available" rather than as "no fixtures scheduled".
    """
    path = Path(path)
    empty = pd.DataFrame(columns=FIXTURE_COLUMNS)
    if not path.exists():
        return empty, {}
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt cache is reported, not fatal
        logger.warning("Could not read season calendar %s: %s", path, exc)
        return empty, {}
    if not isinstance(wrapper, dict):
        logger.warning("Season calendar %s has an unexpected shape.", path)
        return empty, {}
    payload = wrapper.get("payload", wrapper)
    meta = {
        key: wrapper.get(key)
        for key in ("source", "season", "fetched_at")
        if wrapper.get(key)
    }
    try:
        frame = calendar_frame(payload)
    except Exception as exc:  # noqa: BLE001 - same contract as above
        logger.warning("Season calendar %s could not be parsed: %s", path, exc)
        return empty, {}
    meta.update({"matches": int(len(frame)), "path": path.as_posix()})
    return frame, meta


def refresh_calendar(
    *,
    season: str | None = None,
    path: Path | str = SEASON_CALENDAR_PATH,
    provider: OpenFootballProvider | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Download the season calendar and cache it."""
    provider = provider or OpenFootballProvider(season=season)
    payload = provider.fetch_payload()
    saved = save_calendar(
        payload,
        path=path,
        source=provider.source_url,
        season=provider.season,
    )
    frame = calendar_frame(payload)
    return frame, {
        "source": provider.source_url,
        "season": provider.season,
        "path": saved.as_posix(),
        "matches": int(len(frame)),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _prepare(frame: pd.DataFrame | list[dict] | None) -> pd.DataFrame:
    """Canonicalise teams, dates and statuses for comparison."""
    if frame is None:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    data = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if data.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)
    for column in FIXTURE_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    data = data[FIXTURE_COLUMNS].copy()
    for column in ("home_team", "away_team"):
        data[column] = data[column].astype(str).str.strip().map(canonical_team)
    data["status"] = data["status"].astype(str).str.lower().str.strip()
    data["match_date"] = data["match_date"].astype(str).str.strip()
    missing = data["match_date"].eq("")
    if missing.any():
        derived = pd.to_datetime(
            data.loc[missing, "kickoff_utc"], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d")
        data.loc[missing, "match_date"] = derived.fillna("")
    data["kickoff_utc"] = data["kickoff_utc"].astype(str).str.strip()
    return data


def compare_with_calendar(
    fixtures: pd.DataFrame | list[dict] | None,
    calendar: pd.DataFrame | list[dict] | None,
    *,
    now: datetime | None = None,
    horizon_days: int = CALENDAR_HORIZON_DAYS,
) -> dict:
    """Compare stored upcoming fixtures with a verified season calendar.

    Identity is ``(canonical home, canonical away, match date)``. A stored
    fixture that matches on the pairing but not the date is reported as
    ``rescheduled`` instead of being counted as both missing and unexpected.
    """
    now = now or datetime.now(timezone.utc)
    today = now.date()
    horizon_end = today + timedelta(days=max(int(horizon_days), 0))

    stored = _prepare(fixtures)
    reference = _prepare(calendar)
    stored = stored[
        ~stored["status"].isin(TERMINAL_STATUSES) & stored["match_date"].ne("")
    ]
    stored = stored[stored["match_date"] >= today.isoformat()]
    reference = reference[
        ~reference["status"].isin(TERMINAL_STATUSES) & reference["match_date"].ne("")
    ]
    reference = reference[reference["match_date"] >= today.isoformat()]

    def index(frame: pd.DataFrame) -> dict[tuple[str, str, str], dict]:
        return {
            (row.home_team, row.away_team, row.match_date): row._asdict()
            for row in frame.itertuples(index=False)
        }

    expected = index(reference)
    actual = index(stored)
    matched_keys = sorted(set(expected) & set(actual))
    missing_keys = sorted(set(expected) - set(actual))
    extra_keys = sorted(set(actual) - set(expected))

    expected_pairs = {(home, away): date for home, away, date in expected}
    rescheduled: list[dict] = []
    remaining_extra: list[tuple[str, str, str]] = []
    for key in extra_keys:
        home, away, date = key
        calendar_date = expected_pairs.get((home, away))
        if calendar_date:
            rescheduled.append({
                "home_team": home,
                "away_team": away,
                "calendar_date": calendar_date,
                "stored_date": date,
            })
        else:
            remaining_extra.append(key)

    def per_team(frame: pd.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in frame.itertuples(index=False):
            counts[row.home_team] = counts.get(row.home_team, 0) + 1
            counts[row.away_team] = counts.get(row.away_team, 0) + 1
        return counts

    expected_teams = per_team(reference)
    stored_teams = per_team(stored)

    def fixture_row(key: tuple[str, str, str]) -> dict:
        home, away, date = key
        source = expected.get(key) or actual.get(key) or {}
        return {
            "home_team": home,
            "away_team": away,
            "match_date": date,
            "kickoff_utc": str(source.get("kickoff_utc", "")),
        }

    horizon_reference = reference[reference["match_date"] <= horizon_end.isoformat()]
    horizon_stored = stored[stored["match_date"] <= horizon_end.isoformat()]
    horizon_matched = set(index(horizon_reference)) & set(index(horizon_stored))

    total_expected = len(expected)
    coverage = round(len(matched_keys) / total_expected, 4) if total_expected else None
    return {
        "verified": True,
        "horizon_days": int(horizon_days),
        "as_of": today.isoformat(),
        "calendar_upcoming": total_expected,
        "stored_upcoming": len(actual),
        "matched": len(matched_keys),
        "missing": len(missing_keys),
        "unexpected": len(remaining_extra),
        "rescheduled": len(rescheduled),
        "coverage_fraction": coverage,
        "teams_expected": len(expected_teams),
        "teams_with_fixtures": len(stored_teams),
        "teams_without_fixtures": sorted(set(expected_teams) - set(stored_teams)),
        "teams_with_partial_coverage": sorted(
            team for team in expected_teams
            if 0 < stored_teams.get(team, 0) < expected_teams[team]
        ),
        "missing_fixtures": [fixture_row(key) for key in missing_keys[:MAX_LISTED_FIXTURES]],
        "unexpected_fixtures": [fixture_row(key) for key in remaining_extra[:MAX_LISTED_FIXTURES]],
        "rescheduled_fixtures": rescheduled[:MAX_LISTED_FIXTURES],
        "within_horizon": {
            "days": int(horizon_days),
            "calendar_upcoming": int(len(horizon_reference)),
            "stored_upcoming": int(len(horizon_stored)),
            "matched": int(len(horizon_matched)),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache the OpenFootball La Liga season calendar and compare it with "
            "the stored upcoming fixtures. No fixture is invented or written."
        )
    )
    parser.add_argument("--refresh", action="store_true", help="download and cache the calendar")
    parser.add_argument("--season", default="", help="season folder, e.g. 2026-27 (default: current)")
    parser.add_argument("--json-path", default=str(SEASON_CALENDAR_PATH), help="cache path")
    parser.add_argument("--horizon-days", type=int, default=CALENDAR_HORIZON_DAYS)
    parser.add_argument("--full", action="store_true", help="print every listed fixture")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if args.refresh:
        frame, meta = refresh_calendar(season=args.season or None, path=args.json_path)
        print(json.dumps({"refreshed": meta}, indent=2))
    else:
        frame, meta = load_calendar(args.json_path)

    if frame.empty:
        print(json.dumps({
            "status": "no_calendar_cached",
            "path": str(args.json_path),
            "detail": "Run with --refresh to download the calendar; no coverage claim is made.",
        }, indent=2))
        return 1

    from src.data.fixture_provider import load_fixture_rows

    comparison = compare_with_calendar(
        load_fixture_rows(), frame, horizon_days=args.horizon_days
    )
    payload: dict[str, Any] = {
        "status": "ok" if comparison["missing"] == 0 else "incomplete",
        "calendar": meta,
        "comparison": comparison,
    }
    if not args.full:
        payload["comparison"] = {
            **comparison,
            "missing_fixtures": comparison["missing_fixtures"][:10],
            "unexpected_fixtures": comparison["unexpected_fixtures"][:10],
        }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
