"""Fixture input parsing.

Preferred source: ``data/raw/upcoming_fixtures.csv`` with columns
``home_team, away_team, match_date`` (and optional ``odds_home``,
``odds_draw``, ``odds_away``).

Fallback: the legacy ``fixtures.txt`` format, one fixture per line::

    Real Madrid vs Valencia
    Barcelona vs Sevilla | 1.55 4.10 6.50 | 2026-11-08

Pipe-separated odds and an ISO date are both optional. No fixture or date is
ever invented: unknown dates stay blank and are shown as TBD.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.utils.config import FIXTURES_PATH, UPCOMING_FIXTURES_PATH

logger = logging.getLogger(__name__)


def parse_fixture_line(line: str) -> dict:
    """Parse ``Home vs Away [| h d a] [| YYYY-MM-DD]`` into a fixture dict."""
    raw = line.strip()
    if not raw or raw.startswith("#"):
        raise ValueError("Empty/comment fixture line.")

    parts = [p.strip() for p in raw.split("|")]
    body, extras = parts[0], parts[1:]

    if not re.search(r"\s+vs\s+", body, flags=re.IGNORECASE):
        raise ValueError(f"Malformed fixture (expected 'Home vs Away'): {line!r}")
    home, away = re.split(r"\s+vs\s+", body, maxsplit=1, flags=re.IGNORECASE)
    home, away = home.strip(), away.strip()
    if not home or not away:
        raise ValueError(f"Malformed fixture (missing team): {line!r}")

    odds = None
    match_date = ""
    for extra in extras:
        if not extra:
            continue
        date_candidate = pd.to_datetime(extra, errors="coerce")
        if pd.notna(date_candidate) and re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", extra):
            match_date = f"{date_candidate:%Y-%m-%d}"
            continue
        values = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", extra)]
        if len(values) == 3 and all(v > 1.0 for v in values):
            odds = (values[0], values[1], values[2])
        elif extra:
            raise ValueError(f"Unrecognised fixture option {extra!r} in {line!r}")

    return {
        "home_team": home,
        "away_team": away,
        "match_date": match_date,
        "odds": odds,
    }


def read_fixture_file(path: Path | str = FIXTURES_PATH) -> list[dict]:
    """Read a text fixture file into fixture dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fixture file not found: {path}")
    fixtures = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        try:
            fixtures.append(parse_fixture_line(line))
        except ValueError as exc:
            logger.warning("Skipping fixture line %d: %s", line_no, exc)
    return fixtures


def load_upcoming_fixtures() -> tuple[list[dict], str]:
    """Load upcoming fixtures from CSV, then fixtures.txt.

    The CSV written by :mod:`src.data.fixture_provider` carries stable fixture
    IDs, kickoff timestamps and a status. ``finished``/``cancelled`` rows are
    excluded; ``postponed`` rows are still returned (the next provider refresh
    updates their date). Duplicate fixture IDs keep the last written row.

    Returns ``(fixtures, source_description)``. An empty list is valid and the
    dashboard shows an empty state instead of fabricating fixtures.
    """
    if UPCOMING_FIXTURES_PATH.exists():
        try:
            df = pd.read_csv(UPCOMING_FIXTURES_PATH, dtype=str, keep_default_na=False)
            required = {"home_team", "away_team"}
            if not required <= set(df.columns):
                raise ValueError(
                    f"upcoming_fixtures.csv needs columns {sorted(required)}"
                )
            if "status" in df.columns:
                df = df[
                    ~df["status"].astype(str).str.strip().str.lower().isin(
                        {"finished", "cancelled", "canceled", "abandoned", "ft", "aet", "pen"}
                    )
                ]
            if "fixture_id" in df.columns:
                populated = df["fixture_id"].astype(str).str.strip() != ""
                df = pd.concat([df[~populated], df[populated].drop_duplicates(
                    subset="fixture_id", keep="last"
                )])
            fixtures = []
            for _, row in df.iterrows():
                home = str(row["home_team"]).strip()
                away = str(row["away_team"]).strip()
                if not home or not away:
                    continue
                odds = None
                try:
                    values = [
                        float(row.get("odds_home", "")),
                        float(row.get("odds_draw", "")),
                        float(row.get("odds_away", "")),
                    ]
                    if all(v > 1.0 for v in values):
                        odds = tuple(values)
                except (TypeError, ValueError):
                    odds = None
                match_date = str(row.get("match_date", "")).strip()
                if not match_date:
                    kickoff = pd.to_datetime(
                        row.get("kickoff_utc", ""), errors="coerce", utc=True
                    )
                    if pd.notna(kickoff):
                        match_date = f"{kickoff:%Y-%m-%d}"
                fixtures.append({
                    "home_team": home,
                    "away_team": away,
                    "match_date": match_date,
                    "odds": odds,
                    "fixture_id": str(row.get("fixture_id", "")).strip(),
                    "kickoff_utc": str(row.get("kickoff_utc", "")).strip(),
                    "status": str(row.get("status", "scheduled")).strip() or "scheduled",
                    "provider_event_id": str(row.get("provider_event_id", "")).strip(),
                    "provider": str(row.get("provider", "")).strip(),
                    "fixture_source": str(row.get("source", "")).strip(),
                })
            return fixtures, str(UPCOMING_FIXTURES_PATH)
        except Exception as exc:  # noqa: BLE001 - fall back to text file
            logger.warning("Could not read %s: %s", UPCOMING_FIXTURES_PATH, exc)

    if FIXTURES_PATH.exists():
        return read_fixture_file(FIXTURES_PATH), str(FIXTURES_PATH)
    return [], "none"
