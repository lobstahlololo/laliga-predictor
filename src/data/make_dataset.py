"""Fetch and merge real La Liga match history from Football-Data.co.uk.

Downloads the raw match sheets for the requested seasons, maps the columns to
recognizable names, converts the full-time result into a numerical target and
saves one master chronological dataset to ``data/raw/laliga_matches.csv``.

Usage:
    python src/data/make_dataset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Project root (two levels up from src/data/make_dataset.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RAW_DATA_DIR / "laliga_matches.csv"

# Football-Data.co.uk uses the code SP1 for La Liga (Spain 1st division).
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/SP1.csv"

# Seasons to download, most recent first (26/27, 25/26, 24/25, 23/24).
SEASONS = ["2627", "2526", "2425", "2324"]

REQUEST_TIMEOUT = 30  # seconds
USER_AGENT = "Mozilla/5.0 (LaLiga-Outcome-Predictor; data pipeline)"

# Some networks block direct access to football-data.co.uk. In that case we
# fall back to fetching through the r.jina.ai text-extraction proxy, which
# retrieves the same upstream CSV from their servers and prepends a small
# metadata block before the raw CSV body.
PROXY_URL = "https://r.jina.ai/{url}"
CSV_HEADER_TOKEN = "Div,"  # first token of a Football-Data CSV header line
USE_PROXY_FALLBACK = True

# Columns we keep, mapped to recognizable names.
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
    # Bet365 decimal odds -> implied market probabilities below.
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}

REQUIRED_COLUMNS = list(COLUMN_MAP.keys())

# Implied market probability = 1 / decimal odds. These wrap up team news,
# market sentiment and historical strength into one very strong baseline.
ODDS_TO_PROB = {
    "odds_home": "implied_prob_home",
    "odds_draw": "implied_prob_draw",
    "odds_away": "implied_prob_away",
}

# Map FTR codes onto the numerical target: 2 = Home Win, 1 = Draw, 0 = Away Win.
FTR_TO_OUTCOME = {"H": 2, "D": 1, "A": 0}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

def _download(url: str) -> str:
    """Download a URL to text, trying direct access then the proxy fallback."""
    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
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
            proxy,
            timeout=REQUEST_TIMEOUT * 2,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download {url} (direct and proxy): {exc}") from exc
    return _strip_proxy_metadata(response.text, url)


def _strip_proxy_metadata(text: str, url: str) -> str:
    """Drop the proxy's leading metadata block, keeping only the CSV body."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(CSV_HEADER_TOKEN):
            return "\n".join(lines[i:])
    raise RuntimeError(
        f"Proxy response for {url} did not contain a CSV header line "
        f"starting with {CSV_HEADER_TOKEN!r}; upstream format may have changed."
    )


def fetch_season(season: str) -> pd.DataFrame:
    """Download one season's CSV from Football-Data.co.uk into a DataFrame."""
    url = BASE_URL.format(season=season)
    logger.info("Fetching season %s/%s from %s", season[:2], season[2:], url)

    csv_text = _download(url)

    # Football-Data serves Latin-1 encoded sheets with occasional blank lines.
    from io import StringIO

    df = pd.read_csv(
        StringIO(csv_text),
        encoding="latin-1",
        skip_blank_lines=True,
    )

    # Drop the trailingTotals/aggregate rows some sheets contain.
    if "Time" in df.columns:
        df = df[df["Time"].notna() | df["HomeTeam"].notna()]
    df = df[df["HomeTeam"].notna()].copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Season {season} sheet is missing expected columns: {missing}. "
            "The upstream schema may have changed."
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["season"] = f"{season[:2]}/{season[2:]}"
    logger.info("Season %s/%s: %d matches", season[:2], season[2:], len(df))
    return df


def build_master_dataset() -> pd.DataFrame:
    """Fetch all configured seasons and merge into one chronological frame."""
    frames = []
    for season in SEASONS:
        try:
            frames.append(fetch_season(season))
        except RuntimeError as exc:
            # A future/in-progress season may 404; degrade gracefully.
            logger.warning("Skipping season %s: %s", season, exc)

    if not frames:
        raise RuntimeError("No season data could be fetched. Aborting.")

    master = pd.concat(frames, ignore_index=True)

    # Rename to recognizable names.
    master = master.rename(columns=COLUMN_MAP)

    # Parse dates (Football-Data uses dd/mm/yy or dd/mm/yyyy).
    master["date"] = pd.to_datetime(
        master["date"], dayfirst=True, errors="coerce"
    )

    # Numerical target: 2 = Home Win, 1 = Draw, 0 = Away Win.
    unknown = set(master["full_time_result"].dropna().unique()) - set(
        FTR_TO_OUTCOME
    )
    if unknown:
        logger.warning("Dropping rows with unknown FTR codes: %s", unknown)
    master = master[master["full_time_result"].isin(FTR_TO_OUTCOME)].copy()
    master["match_outcome"] = (
        master["full_time_result"].map(FTR_TO_OUTCOME).astype(int)
    )

    # Enforce numeric dtypes on statistics columns.
    numeric_cols = [
        "full_time_home_goals",
        "full_time_away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
    ]
    for col in numeric_cols + list(ODDS_TO_PROB):
        master[col] = pd.to_numeric(master[col], errors="coerce")

    # Implied market probabilities from Bet365 decimal odds.
    for odds_col, prob_col in ODDS_TO_PROB.items():
        master[prob_col] = (1.0 / master[odds_col].where(master[odds_col] > 0)).astype(float)
    n_missing_prob = int(master["implied_prob_home"].isna().sum())
    if n_missing_prob:
        logger.warning(
            "%d matches have missing/invalid Bet365 odds; implied probabilities "
            "left as NaN.", n_missing_prob,
        )

    # Sort chronologically (then by season order) for the master frame.
    master = (
        master.dropna(subset=["date"])
        .sort_values(["date", "home_team", "away_team"])
        .reset_index(drop=True)
    )

    # Stable match key for downstream joins.
    master.insert(
        0,
        "match_id",
        [f"{d:%Y%m%d}_{h}_v_{a}" for d, h, a in zip(
            master["date"], master["home_team"], master["away_team"]
        )],
    )

    return master


def main() -> int:
    """Entry point: build and persist the master dataset."""
    logger.info("Building La Liga master dataset...")
    master = build_master_dataset()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_PATH, index=False)

    counts = master["season"].value_counts().sort_index()
    logger.info("Saved %d matches to %s", len(master), OUTPUT_PATH)
    for season, n in counts.items():
        logger.info("  %s: %d matches", season, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
