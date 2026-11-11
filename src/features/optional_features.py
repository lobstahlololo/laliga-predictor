"""Optional historical xG and player-availability feature groups.

Both groups are strictly opt-in and strictly time-aware:

* an xG observation only counts for a fixture when its ``available_at``
  timestamp is *before* that fixture's kickoff, so post-match xG can never
  enter a pre-match feature;
* an availability snapshot (injury/suspension/lineup news) only counts when it
  was published before kickoff, so a future injury report can never rewrite a
  historical prediction;
* when the source file is missing, the group is reported as unavailable rather
  than filled with invented values. Rows with partial coverage keep their NaN
  values for the training-median imputer, and explicit ``*_known`` /
  ``*_observations`` columns tell the model how much evidence exists.

The module never trains or promotes a model by itself; the ablation runner in
``src.experiments.feature_ablation`` decides whether a group is worth keeping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.config import (
    AVAILABILITY_FEATURE_COLUMNS,
    BASELINE_FEATURE_GROUP,
    FEATURE_COLUMNS,
    PLAYER_AVAILABILITY_PATH,
    PROJECT_ROOT,
    ROLLING_WINDOW,
    XG_FEATURE_COLUMNS,
    XG_HISTORY_PATH,
)
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

AVAILABLE_STATUSES = {"available", "fit", "ok", "ready", "in squad"}
UNAVAILABLE_STATUSES = {"injured", "injury", "suspended", "suspension", "out", "doubtful", "ill", "unavailable"}

GROUP_COLUMNS: dict[str, list[str]] = {
    "xg": list(XG_FEATURE_COLUMNS),
    "player_availability": list(AVAILABILITY_FEATURE_COLUMNS),
}


def _display_path(path: Path | str) -> str:
    """Portable path for reports (repo-relative when possible)."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return Path(path).as_posix()


class OptionalFeatureUnavailable(ValueError):
    """Raised when a requested feature group has no reliable source data."""


@dataclass(frozen=True)
class FeatureGroupStatus:
    name: str
    available: bool
    reason: str
    rows: int
    columns: list[str]

    def as_dict(self) -> dict:
        return {
            "group": self.name,
            "available": self.available,
            "reason": self.reason,
            "source_rows": self.rows,
            "feature_count": len(self.columns),
            "columns": self.columns,
        }


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def load_xg_history(path: Path | str | None = None) -> pd.DataFrame:
    """Load timestamped xG records.

    Required columns: ``team``, ``date`` (match date) and ``xg_for`` /
    ``xg_against``. ``available_at`` is strongly preferred; when absent, the
    observation is treated as available one day after the match (the
    conservative assumption for post-match xG).
    """
    path = Path(path or XG_HISTORY_PATH)
    columns = ["match_id", "team", "date", "available_at", "xg_for", "xg_against"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path, dtype={"match_id": str, "team": str})
    except Exception as exc:  # noqa: BLE001 - malformed optional source
        logger.warning("Could not read xG history %s: %s", path, exc)
        return pd.DataFrame(columns=columns)

    required = {"team", "xg_for", "xg_against"}
    if not required <= set(frame.columns):
        logger.warning(
            "xG history %s is missing columns %s; group unavailable.",
            path, sorted(required - set(frame.columns)),
        )
        return pd.DataFrame(columns=columns)
    if "available_at" not in frame.columns:
        if "date" not in frame.columns:
            logger.warning("xG history %s has neither 'available_at' nor 'date'.", path)
            return pd.DataFrame(columns=columns)
        logger.warning(
            "xG history has no 'available_at'; assuming each observation is "
            "available one day after its match date."
        )
        frame["available_at"] = _parse_timestamp(frame["date"]) + pd.Timedelta(days=1)
    frame["available_at"] = _parse_timestamp(frame["available_at"]).dt.tz_convert(None)
    frame["date"] = _parse_timestamp(frame["date"]) if "date" in frame.columns else pd.NaT
    for column in ("xg_for", "xg_against"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["team"] = frame["team"].astype(str).str.strip().map(canonical_team)
    frame = frame.dropna(subset=["team", "available_at", "xg_for", "xg_against"])
    frame = frame[frame["team"] != ""]
    if "match_id" not in frame.columns:
        frame["match_id"] = ""
    # One observation per team/match; keep the latest published value.
    if "match_id" in frame.columns and frame["match_id"].astype(str).str.strip().ne("").any():
        frame = (
            frame.sort_values("available_at")
            .drop_duplicates(subset=["match_id", "team"], keep="last")
        )
    return frame[columns].sort_values("available_at").reset_index(drop=True)


def load_player_availability(path: Path | str | None = None) -> pd.DataFrame:
    """Load timestamped availability snapshots.

    Required columns: ``team``, ``player``, ``status``, ``available_at``. An
    optional ``is_starter`` flag marks expected starters (default: all rows
    are treated as first-team players).
    """
    path = Path(path or PLAYER_AVAILABILITY_PATH)
    columns = ["team", "player", "status", "available_at", "is_starter"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read player availability %s: %s", path, exc)
        return pd.DataFrame(columns=columns)

    stamp_column = next(
        (c for c in ("available_at", "published_at", "timestamp", "date") if c in frame.columns),
        None,
    )
    if not {"team", "player", "status"} <= set(frame.columns) or stamp_column is None:
        logger.warning(
            "Player availability %s lacks team/player/status/timestamp; group unavailable.", path
        )
        return pd.DataFrame(columns=columns)
    frame["available_at"] = _parse_timestamp(frame[stamp_column]).dt.tz_convert(None)
    frame["team"] = frame["team"].astype(str).str.strip().map(canonical_team)
    frame["player"] = frame["player"].astype(str).str.strip()
    frame["status"] = frame["status"].astype(str).str.strip().str.lower()
    if "is_starter" in frame.columns:
        frame["is_starter"] = frame["is_starter"].astype(str).str.strip().str.lower().isin(
            {"1", "true", "yes", "y", "starter"}
        )
    else:
        frame["is_starter"] = True
    frame = frame.dropna(subset=["available_at"])
    frame = frame[(frame["team"] != "") & (frame["player"] != "")]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    # A later snapshot for the same player supersedes earlier ones.
    frame = (
        frame.sort_values("available_at")
        .drop_duplicates(subset=["team", "player"], keep="last")
    )
    return frame[columns].reset_index(drop=True)


def feature_group_status(group: str) -> FeatureGroupStatus:
    """Whether a group has a usable source, without inventing anything."""
    if group == BASELINE_FEATURE_GROUP:
        return FeatureGroupStatus(group, True, "always available", 0, list(FEATURE_COLUMNS))
    if group not in GROUP_COLUMNS:
        raise OptionalFeatureUnavailable(f"Unknown feature group: {group!r}")
    if group == "xg":
        frame = load_xg_history()
        if frame.empty:
            return FeatureGroupStatus(
                group, False,
                f"no timestamped xG history at {_display_path(XG_HISTORY_PATH)}",
                len(frame), GROUP_COLUMNS[group],
            )
        return FeatureGroupStatus(group, True, "timestamped xG history", len(frame), GROUP_COLUMNS[group])
    frame = load_player_availability()
    if frame.empty:
        return FeatureGroupStatus(
            group, False,
            f"no timestamped player-availability data at {_display_path(PLAYER_AVAILABILITY_PATH)}",
            len(frame), GROUP_COLUMNS[group],
        )
    return FeatureGroupStatus(
        group, True, "timestamped availability snapshots", len(frame), GROUP_COLUMNS[group]
    )


def resolve_feature_columns(
    groups: Iterable[str] | None = None,
) -> tuple[list[str], list[FeatureGroupStatus]]:
    """Return the ordered feature columns and per-group statuses.

    Baseline always comes first. Requesting an unavailable group raises
    :class:`OptionalFeatureUnavailable` instead of silently dropping it.
    """
    requested = list(groups or [BASELINE_FEATURE_GROUP])
    if BASELINE_FEATURE_GROUP not in requested:
        requested.insert(0, BASELINE_FEATURE_GROUP)
    columns: list[str] = []
    statuses: list[FeatureGroupStatus] = []
    for group in requested:
        status = feature_group_status(group)
        statuses.append(status)
        if not status.available:
            raise OptionalFeatureUnavailable(
                f"Feature group {group!r} is unavailable: {status.reason}. "
                "Provide the source file or run with the baseline groups only; "
                "no values are fabricated."
            )
        columns.extend(status.columns)
    # A group may legitimately be requested twice; keep one copy of each column.
    return list(dict.fromkeys(columns)), statuses


# ---------------------------------------------------------------------------
# Historical feature construction (pre-match only)
# ---------------------------------------------------------------------------

def _naive_utc(kickoff) -> pd.Timestamp:
    """Naive UTC timestamp so source data and fixture dates compare safely."""
    stamp = pd.Timestamp(kickoff)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp


def _coerce_xg(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise timestamp/numeric dtypes for source or caller-provided data."""
    out = frame.copy()
    out["available_at"] = _parse_timestamp(out["available_at"]).dt.tz_convert(None)
    for column in ("xg_for", "xg_against"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["team"] = out["team"].astype(str).str.strip().map(canonical_team)
    return out


def _coerce_availability(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["available_at"] = _parse_timestamp(out["available_at"]).dt.tz_convert(None)
    out["team"] = out["team"].astype(str).str.strip().map(canonical_team)
    out["player"] = out["player"].astype(str).str.strip()
    out["status"] = out["status"].astype(str).str.strip().str.lower()
    if "is_starter" not in out.columns:
        out["is_starter"] = True
    else:
        out["is_starter"] = out["is_starter"].fillna(True).astype(bool)
    return out


def _history_before(frame: pd.DataFrame, kickoff: pd.Timestamp) -> pd.DataFrame:
    return frame[frame["available_at"] < _naive_utc(kickoff)]


def _xg_form(xg: pd.DataFrame, team: str, kickoff: pd.Timestamp) -> dict[str, float]:
    rows = _history_before(xg[xg["team"] == team], kickoff).sort_values("available_at")
    rows = rows.tail(ROLLING_WINDOW)
    if rows.empty:
        return {
            "xg_for_5": np.nan,
            "xg_against_5": np.nan,
            "xg_diff_5": np.nan,
            "xg_observations": 0.0,
        }
    xg_for = float(rows["xg_for"].mean())
    xg_against = float(rows["xg_against"].mean())
    return {
        "xg_for_5": xg_for,
        "xg_against_5": xg_against,
        "xg_diff_5": xg_for - xg_against,
        "xg_observations": float(len(rows)),
    }


def build_xg_features(
    raw: pd.DataFrame,
    xg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per ``match_id`` with pre-kickoff rolling xG features."""
    xg = load_xg_history() if xg is None else _coerce_xg(xg)
    if xg.empty:
        raise OptionalFeatureUnavailable(
            f"No timestamped xG history available at {_display_path(XG_HISTORY_PATH)}; "
            "the xG feature group cannot be built."
        )
    required = {"match_id", "date", "home_team", "away_team"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"xG feature input missing columns: {sorted(missing)}")

    rows = []
    for match in raw.itertuples(index=False):
        kickoff = pd.Timestamp(getattr(match, "date")).normalize()
        home = canonical_team(getattr(match, "home_team"))
        away = canonical_team(getattr(match, "away_team"))
        home_form = _xg_form(xg, home, kickoff)
        away_form = _xg_form(xg, away, kickoff)
        rows.append({
            "match_id": str(getattr(match, "match_id")),
            "home_xg_for_5": home_form["xg_for_5"],
            "home_xg_against_5": home_form["xg_against_5"],
            "home_xg_diff_5": home_form["xg_diff_5"],
            "away_xg_for_5": away_form["xg_for_5"],
            "away_xg_against_5": away_form["xg_against_5"],
            "away_xg_diff_5": away_form["xg_diff_5"],
            "home_xg_observations": home_form["xg_observations"],
            "away_xg_observations": away_form["xg_observations"],
        })
    return pd.DataFrame(rows).set_index("match_id")[XG_FEATURE_COLUMNS]


def _availability_at(
    availability: pd.DataFrame, team: str, kickoff: pd.Timestamp
) -> dict[str, float]:
    rows = _history_before(availability[availability["team"] == team], kickoff)
    if rows.empty:
        return {
            "availability_score": np.nan,
            "missing_starters": np.nan,
            "availability_known": 0.0,
        }
    latest = rows.sort_values("available_at").groupby("player", as_index=False).tail(1)
    starters = latest[latest["is_starter"].fillna(True)]
    if starters.empty:
        starters = latest
    total = len(starters)
    if total == 0:
        return {
            "availability_score": np.nan,
            "missing_starters": np.nan,
            "availability_known": 0.0,
        }
    statuses = starters["status"].astype(str).str.lower()
    available = statuses.isin(AVAILABLE_STATUSES)
    missing = (~available).sum()
    return {
        "availability_score": float(available.sum() / total),
        "missing_starters": float(missing),
        "availability_known": 1.0,
    }


def build_availability_features(
    raw: pd.DataFrame,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per ``match_id`` with pre-kickoff availability features."""
    availability = (
        load_player_availability() if availability is None
        else _coerce_availability(availability)
    )
    if availability.empty:
        raise OptionalFeatureUnavailable(
            f"No timestamped player-availability data at {_display_path(PLAYER_AVAILABILITY_PATH)}; "
            "the player_availability feature group cannot be built."
        )
    required = {"match_id", "date", "home_team", "away_team"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Availability feature input missing columns: {sorted(missing)}")

    rows = []
    for match in raw.itertuples(index=False):
        kickoff = pd.Timestamp(getattr(match, "date")).normalize()
        home = canonical_team(getattr(match, "home_team"))
        away = canonical_team(getattr(match, "away_team"))
        home_status = _availability_at(availability, home, kickoff)
        away_status = _availability_at(availability, away, kickoff)
        diff = (
            home_status["availability_score"] - away_status["availability_score"]
            if np.isfinite(home_status["availability_score"])
            and np.isfinite(away_status["availability_score"])
            else np.nan
        )
        rows.append({
            "match_id": str(getattr(match, "match_id")),
            "home_availability_score": home_status["availability_score"],
            "away_availability_score": away_status["availability_score"],
            "home_missing_starters": home_status["missing_starters"],
            "away_missing_starters": away_status["missing_starters"],
            "availability_diff": diff,
            "home_availability_known": home_status["availability_known"],
            "away_availability_known": away_status["availability_known"],
        })
    return pd.DataFrame(rows).set_index("match_id")[AVAILABILITY_FEATURE_COLUMNS]


def attach_optional_features(
    frame: pd.DataFrame,
    groups: Iterable[str],
    *,
    xg: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge requested optional group columns onto a match frame."""
    groups = list(groups)
    out = frame.copy()
    if "xg" in groups:
        out = out.merge(
            build_xg_features(out, xg=xg),
            left_on="match_id", right_index=True, how="left",
        )
    if "player_availability" in groups:
        out = out.merge(
            build_availability_features(out, availability=availability),
            left_on="match_id", right_index=True, how="left",
        )
    return out


def fixture_optional_features(
    raw: pd.DataFrame,
    home_team: str,
    away_team: str,
    kickoff: pd.Timestamp,
    columns: Iterable[str],
    *,
    xg: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Feature values for one upcoming fixture, using only pre-kickoff data."""
    columns = list(columns)
    required = set(columns)
    kickoff = _naive_utc(kickoff)
    values: dict[str, float] = {}
    if required & set(XG_FEATURE_COLUMNS):
        source = load_xg_history() if xg is None else _coerce_xg(xg)
        if source.empty:
            raise OptionalFeatureUnavailable(
                "This model expects xG features but no timestamped xG history exists."
            )
        cutoff = pd.Timestamp(kickoff)
        home = _xg_form(source, canonical_team(home_team), cutoff)
        away = _xg_form(source, canonical_team(away_team), cutoff)
        for prefix, form in (("home", home), ("away", away)):
            values[f"{prefix}_xg_for_5"] = form["xg_for_5"]
            values[f"{prefix}_xg_against_5"] = form["xg_against_5"]
            values[f"{prefix}_xg_diff_5"] = form["xg_diff_5"]
            values[f"{prefix}_xg_observations"] = form["xg_observations"]
    if required & set(AVAILABILITY_FEATURE_COLUMNS):
        source = (
            load_player_availability() if availability is None
            else _coerce_availability(availability)
        )
        if source.empty:
            raise OptionalFeatureUnavailable(
                "This model expects availability features but no timestamped "
                "player-availability data exists."
            )
        cutoff = pd.Timestamp(kickoff)
        home = _availability_at(source, canonical_team(home_team), cutoff)
        away = _availability_at(source, canonical_team(away_team), cutoff)
        values["home_availability_score"] = home["availability_score"]
        values["away_availability_score"] = away["availability_score"]
        values["home_missing_starters"] = home["missing_starters"]
        values["away_missing_starters"] = away["missing_starters"]
        values["home_availability_known"] = home["availability_known"]
        values["away_availability_known"] = away["availability_known"]
        values["availability_diff"] = (
            home["availability_score"] - away["availability_score"]
            if np.isfinite(home["availability_score"]) and np.isfinite(away["availability_score"])
            else np.nan
        )
    return {key: values[key] for key in columns if key in values}
