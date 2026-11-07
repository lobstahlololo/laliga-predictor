"""Fetch, merge and validate La Liga match history from Football-Data.co.uk.

The pipeline is incremental and idempotent:

* Each configured season is downloaded independently.
* If a download fails, the cached rows for that season are retained (the
  pipeline never silently replaces real data with an empty frame).
* New rows are merged with the cached master file, cleaned, de-duplicated by
  stable fixture ID and sorted chronologically.
* A machine-readable quality report is written to
  ``data/processed/data_quality.json``.

Running the script repeatedly therefore refreshes played matches without
duplicating rows or corrupting existing history.

Usage:
    python -m src.data.make_dataset
"""

from __future__ import annotations

import json
import logging
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# Allow direct execution: python src/data/make_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import clean_matches  # noqa: E402
from src.utils.config import (  # noqa: E402
    DATA_QUALITY_PATH,
    RAW_DIR,
    RAW_PATH,
    SEASONS,
    SEASON_CODES,
)

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"
REQUEST_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (LaLiga-Outcome-Predictor; data pipeline)"
PROXY_URL = "https://r.jina.ai/{url}"
CSV_HEADER_TOKEN = "Div,"
USE_PROXY_FALLBACK = True

COLUMN_MAP = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "full_time_home_goals",
    "FTAG": "full_time_away_goals",
    "FTR": "full_time_result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}
REQUIRED_COLUMNS = list(COLUMN_MAP)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str) -> str:
    """Download a URL as text, falling back to the text-extraction proxy."""
    try:
        response = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        logger.warning("Direct download failed for %s: %s", url, exc)
        if not USE_PROXY_FALLBACK:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    proxy = PROXY_URL.format(url=url)
    logger.info("Falling back to proxy fetch for %s", url)
    try:
        response = requests.get(
            proxy, timeout=REQUEST_TIMEOUT * 2, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download {url} (direct and proxy): {exc}") from exc
    return _strip_proxy_metadata(response.text, url)


def _strip_proxy_metadata(text: str, url: str) -> str:
    """Drop the proxy's leading metadata block, keeping the CSV body."""
    for i, line in enumerate(text.splitlines()):
        if line.startswith(CSV_HEADER_TOKEN):
            return "\n".join(text.splitlines()[i:])
    raise RuntimeError(
        f"Proxy response for {url} had no CSV header starting with "
        f"{CSV_HEADER_TOKEN!r}; upstream format may have changed."
    )


def fetch_season(season: str) -> pd.DataFrame:
    """Download and clean one season's CSV.

    ``season`` is a Football-Data code such as ``"2425"``. In-progress
    seasons may not exist upstream yet; callers handle the failure.
    """
    url = BASE_URL.format(season=season)
    logger.info("Fetching season %s from %s", SEASON_CODES.get(season, season), url)
    csv_text = _download(url)
    df = pd.read_csv(StringIO(csv_text), encoding="latin-1", skip_blank_lines=True)

    if "HomeTeam" in df.columns:
        df = df[df["HomeTeam"].notna()].copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Season {season} sheet is missing expected columns: {missing}. "
            "The upstream schema may have changed."
        )

    df = df[REQUIRED_COLUMNS].rename(columns=COLUMN_MAP)
    cleaned, _ = clean_matches(df)
    logger.info("Season %s: %d matches", SEASON_CODES.get(season, season), len(cleaned))
    return cleaned


# ---------------------------------------------------------------------------
# Master dataset
# ---------------------------------------------------------------------------

def load_cached(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the cached master dataset, or an empty frame if absent."""
    if not path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
        logger.warning("Could not read cached dataset %s: %s", path, exc)
        return pd.DataFrame()
    logger.info("Loaded %d cached matches from %s", len(cached), path)
    return cached


def build_master_dataset(
    seasons: list[str] | None = None,
    cached: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch configured seasons and merge them with the cached master.

    Returns the cleaned chronological frame and a quality report. A season
    that fails to download keeps its cached rows.
    """
    seasons = seasons or SEASONS
    cached = load_cached() if cached is None else cached

    frames: list[pd.DataFrame] = []
    fetched_seasons: list[str] = []
    failed_seasons: list[str] = []

    for season in seasons:
        try:
            frames.append(fetch_season(season))
            fetched_seasons.append(season)
        except (RuntimeError, ValueError, KeyError) as exc:
            failed_seasons.append(season)
            logger.warning("Skipping season %s: %s", SEASON_CODES.get(season, season), exc)

    if not frames and cached.empty:
        raise RuntimeError(
            "No season data could be fetched and no cached dataset exists. "
            "Check network access or provide data/raw/laliga_matches.csv."
        )
    if not frames:
        logger.warning(
            "All downloads failed; continuing with the cached dataset "
            "(%d matches). No data was invented.",
            len(cached),
        )

    fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = pd.concat([cached, fresh], ignore_index=True) if not cached.empty else fresh

    # Cached rows for seasons that failed upstream survive the concat; fresh
    # rows win on duplicate fixture IDs because dedupe keeps the last copy.
    if not cached.empty:
        combined = pd.concat([cached, fresh], ignore_index=True)
        if not fresh.empty:
            fresh_ids = set(fresh["match_id"]) if "match_id" in fresh.columns else set()
            if fresh_ids:
                combined = pd.concat(
                    [cached[~cached["match_id"].isin(fresh_ids)], fresh],
                    ignore_index=True,
                )

    cleaned, report = clean_matches(combined)
    report.update(
        {
            "seasons_fetched": [SEASON_CODES.get(s, s) for s in fetched_seasons],
            "seasons_failed": [SEASON_CODES.get(s, s) for s in failed_seasons],
            "seasons_present": sorted(cleaned["season"].dropna().unique().tolist()),
            "date_min": str(cleaned["date"].min().date()) if not cleaned.empty else None,
            "date_max": str(cleaned["date"].max().date()) if not cleaned.empty else None,
        }
    )
    return cleaned, report


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file, then replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_quality_report(report: dict, path: Path = DATA_QUALITY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    logger.info("Building/refreshing the La Liga master dataset...")
    master, report = build_master_dataset()

    write_atomic(master, RAW_PATH)
    write_quality_report(report)

    logger.info("Saved %d matches to %s", len(master), RAW_PATH)
    for season, n in master["season"].value_counts().sort_index().items():
        logger.info("  %s: %d matches", season, n)
    for warning in report.get("warnings", []):
        logger.warning("Data quality: %s", warning)
    if report.get("rows_removed"):
        logger.warning("Removed %d invalid/duplicate rows.", report["rows_removed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
