"""Dedicated historical data-quality audit and reports.

This module answers the question "is the cached dataset fit to evaluate and
train on?" without modifying it. It measures, per column and per season:

* missing values (including odds and model features);
* duplicate fixtures and duplicate home/away pairings;
* team-name inconsistencies (raw values that canonicalisation changes);
* invalid scores, invalid dates and result/score contradictions;
* chronological ordering and same-team-same-day anomalies;
* unexpected season sizes (with in-progress seasons treated separately);
* feature availability at prediction time.

Thresholds are explicit configuration. Findings carry a code, severity and
message, so a caller can fail a check or merely warn. Reports are written as
machine-readable JSON plus CSV and Markdown tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.validation import (
    FTR_TO_OUTCOME,
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    parse_match_dates,
    season_from_date,
)
from src.utils.config import (
    DATA_QUALITY_COLUMNS_PATH,
    DATA_QUALITY_MARKDOWN_PATH,
    DATA_QUALITY_REPORT_PATH,
    DATA_QUALITY_SEASONS_PATH,
    FEATURE_COLUMNS,
    MAX_EXPECTED_SEASON_MATCHES,
    MAX_INVALID_SCORE,
    MAX_MISSING_FEATURE_FRACTION,
    MIN_FULL_SEASON_MATCHES,
    RAW_PATH,
    STALE_DATA_DAYS,
)
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"
OK = "ok"


@dataclass(frozen=True)
class QualityThresholds:
    """Declared-before-audit quality limits (never fitted on the data)."""

    max_missing_fraction: float = MAX_MISSING_FEATURE_FRACTION
    max_missing_odds_fraction: float = 0.02
    max_invalid_rows: int = 0
    min_season_matches: int = MIN_FULL_SEASON_MATCHES
    max_season_matches: int = MAX_EXPECTED_SEASON_MATCHES
    max_invalid_score: int = MAX_INVALID_SCORE
    stale_days: int = STALE_DATA_DAYS
    min_feature_coverage: float = 0.95
    future_grace_days: int = 0
    earliest_year: int = 2000

    def as_dict(self) -> dict:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and NaN to strict-JSON values."""
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
    if isinstance(value, (str, bytes)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _num(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _finding(code: str, severity: str, message: str, **details: Any) -> dict:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "details": _json_safe(details),
    }


def _season_series(frame: pd.DataFrame) -> pd.Series:
    if "season" in frame.columns and frame["season"].notna().any():
        return frame["season"].astype(str)
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        return dates.map(lambda d: season_from_date(d) if pd.notna(d) else "")
    return pd.Series([""] * len(frame), index=frame.index)


def _missing_tables(
    frame: pd.DataFrame,
    seasons: pd.Series,
    thresholds: QualityThresholds,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    columns: list[str] = []
    for column in REQUIRED_COLUMNS + ["match_id", "season", "implied_prob_home",
                                      "implied_prob_draw", "implied_prob_away"]:
        if column in frame.columns and column not in columns:
            columns.append(column)
    columns.extend(c for c in frame.columns if c not in columns)

    overall_rows: list[dict] = []
    season_rows: list[dict] = []
    missing_by_column: dict[str, int] = {}
    for column in columns:
        missing = int(frame[column].isna().sum())
        if missing:
            missing_by_column[column] = missing
        overall_rows.append({
            "column": column,
            "rows": int(len(frame)),
            "missing": missing,
            "missing_fraction": missing / len(frame) if len(frame) else 0.0,
        })
        if missing:
            for season, group in frame.groupby(seasons, dropna=False):
                count = int(group[column].isna().sum())
                if count:
                    season_rows.append({
                        "season": str(season),
                        "column": column,
                        "rows": int(len(group)),
                        "missing": count,
                        "missing_fraction": count / len(group) if len(group) else 0.0,
                    })
    return pd.DataFrame(overall_rows), pd.DataFrame(season_rows), missing_by_column


def _invalid_scores(frame: pd.DataFrame, max_score: int) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    summary = {
        "non_numeric": 0,
        "negative": 0,
        "too_large": 0,
        "non_integer": 0,
        "missing": 0,
        "score_result_mismatch": 0,
    }
    if not {"full_time_home_goals", "full_time_away_goals"} <= set(frame.columns):
        return pd.DataFrame(rows), summary
    for column in ("full_time_home_goals", "full_time_away_goals"):
        raw = frame[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        missing = raw.isna() | raw.astype(str).str.strip().isin({"", "nan", "None"})
        summary["missing"] += int(missing.sum())
        non_numeric = numeric.isna() & ~missing
        summary["non_numeric"] += int(non_numeric.sum())
        summary["negative"] += int((numeric < 0).sum())
        summary["too_large"] += int((numeric > max_score).sum())
        summary["non_integer"] += int(
            (numeric.notna() & (numeric % 1 != 0)).sum()
        )
        for idx in frame.index[non_numeric | (numeric < 0) | (numeric > max_score)]:
            rows.append({
                "match_id": str(frame.at[idx, "match_id"]) if "match_id" in frame else str(idx),
                "column": column,
                "value": str(frame.at[idx, column]),
                "problem": (
                    "non_numeric" if non_numeric.loc[idx]
                    else "negative" if numeric.loc[idx] < 0
                    else "too_large"
                ),
            })
    if {"full_time_result", "match_outcome"} <= set(frame.columns):
        home = pd.to_numeric(frame["full_time_home_goals"], errors="coerce")
        away = pd.to_numeric(frame["full_time_away_goals"], errors="coerce")
        expected = np.select([home > away, home == away], [2, 1], default=0)
        mismatch = (
            (expected != pd.to_numeric(frame["match_outcome"], errors="coerce"))
            & home.notna() & away.notna()
        )
        summary["score_result_mismatch"] = int(mismatch.sum())
    return pd.DataFrame(rows), summary


def _chronology(frame: pd.DataFrame) -> dict:
    details: dict = {
        "sorted_ascending": True,
        "ordering_violations": 0,
        "same_team_same_day": 0,
        "match_id_date_mismatches": 0,
    }
    if "date" not in frame.columns:
        return details
    dates = pd.to_datetime(frame["date"], errors="coerce")
    valid = dates.dropna()
    if len(valid) > 1:
        details["sorted_ascending"] = bool(valid.is_monotonic_increasing)
        details["ordering_violations"] = int((valid.diff().dt.days < 0).sum())
    if {"home_team", "away_team"} <= set(frame.columns):
        teams = pd.concat([
            pd.DataFrame({"date": dates, "team": frame["home_team"]}),
            pd.DataFrame({"date": dates, "team": frame["away_team"]}),
        ], ignore_index=True).dropna(subset=["date", "team"])
        duplicate = teams.duplicated(subset=["date", "team"], keep=False)
        details["same_team_same_day"] = int(
            teams.loc[duplicate, ["date", "team"]].drop_duplicates().shape[0]
        )
    if {"match_id", "date"} <= set(frame.columns):
        derived = dates.dt.strftime("%Y%m%d")
        parsed_ids = frame["match_id"].astype(str).str[:8]
        details["match_id_date_mismatches"] = int(
            ((parsed_ids != derived) & dates.notna()).sum()
        )
    return details


def _team_inconsistencies(frame: pd.DataFrame) -> dict:
    rows: list[dict] = []
    for column in ("home_team", "away_team"):
        if column not in frame.columns:
            continue
        raw = frame[column].astype(str).str.strip()
        canonical = raw.map(canonical_team)
        changed = raw.ne(canonical) & raw.ne("")
        counts = raw[changed].value_counts()
        for value, count in counts.items():
            rows.append({
                "column": column,
                "raw_value": value,
                "canonical_value": canonical[raw == value].iloc[0],
                "rows": int(count),
            })
    table = pd.DataFrame(rows)
    return {
        "inconsistent_rows": int(table["rows"].sum()) if not table.empty else 0,
        "distinct_values": int(len(table)),
        "examples": table.head(20).to_dict(orient="records"),
    }


def _feature_audit(
    raw: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame | None,
    meta: pd.DataFrame | None,
    thresholds: QualityThresholds,
) -> tuple[dict, list[dict], pd.DataFrame]:
    findings: list[dict] = []
    report: dict = {
        "audited": False,
        "n_features": 0,
        "rows": 0,
        "rows_fully_available": 0,
        "feature_coverage_fraction": None,
        "max_missing_fraction": None,
        "features_over_threshold": [],
        "missing_by_feature": {},
        "prediction_time_columns_present": [],
        "prediction_time_columns_missing": [],
    }
    table = pd.DataFrame(columns=["feature", "rows", "missing", "missing_fraction"])

    if feature_frame is None:
        try:
            from src.features.build_features import build_model_frame

            feature_frame, _y, built_meta = build_model_frame(raw)
            meta = built_meta if meta is None else meta
        except Exception as exc:  # noqa: BLE001 - a feature failure is a finding
            findings.append(_finding(
                "feature_frame_unavailable", ERROR,
                f"Could not build the model feature frame: {exc}",
            ))
            return report, findings, table

    if feature_frame is None or feature_frame.empty:
        return report, findings, table
    columns = [c for c in FEATURE_COLUMNS if c in feature_frame.columns]
    report["audited"] = True
    report["n_features"] = len(columns)
    report["rows"] = int(len(feature_frame))
    if not columns:
        findings.append(_finding(
            "feature_columns_missing", ERROR,
            "None of the canonical feature columns are present.",
        ))
        return report, findings, table

    missing = feature_frame[columns].isna().mean()
    report["missing_by_feature"] = {
        str(k): float(v) for k, v in missing[missing > 0].items()
    }
    report["max_missing_fraction"] = float(missing.max()) if len(missing) else None
    over = [str(k) for k, v in missing.items() if float(v) > thresholds.max_missing_fraction]
    report["features_over_threshold"] = over
    complete_rows = int((~feature_frame[columns].isna().any(axis=1)).sum())
    report["rows_fully_available"] = complete_rows
    report["feature_coverage_fraction"] = (
        complete_rows / len(feature_frame) if len(feature_frame) else None
    )
    table = pd.DataFrame({
        "feature": columns,
        "rows": len(feature_frame),
        "missing": [int(feature_frame[c].isna().sum()) for c in columns],
        "missing_fraction": [float(feature_frame[c].isna().mean()) for c in columns],
    })

    if over:
        findings.append(_finding(
            "feature_missingness", ERROR if "implied_prob" in " ".join(over) else WARNING,
            "Feature missingness exceeds the configured threshold "
            f"({thresholds.max_missing_fraction:.1%}): " + ", ".join(over),
            features=over,
        ))
    if report["feature_coverage_fraction"] is not None and (
        report["feature_coverage_fraction"] < thresholds.min_feature_coverage
    ):
        findings.append(_finding(
            "feature_coverage_low", WARNING,
            "Only "
            f"{report['feature_coverage_fraction']:.1%} of rows have every feature; "
            f"threshold is {thresholds.min_feature_coverage:.1%}.",
        ))

    # Availability at prediction time: the columns inference needs must exist
    # in the metadata row, and a fixture must not depend on post-kickoff fields.
    required_meta = ["match_id", "date", "home_team", "away_team", "season"]
    if meta is not None:
        present = [c for c in required_meta if c in meta.columns]
        missing_meta = [c for c in required_meta if c not in meta.columns]
        report["prediction_time_columns_present"] = present
        report["prediction_time_columns_missing"] = missing_meta
        if missing_meta:
            findings.append(_finding(
                "prediction_metadata_missing", WARNING,
                "Metadata needed to identify a fixture at prediction time is missing: "
                + ", ".join(missing_meta),
            ))
        if "date" in meta.columns and "match_id" in meta.columns:
            lookup = raw.set_index("match_id", drop=False) if "match_id" in raw else None
            if lookup is not None:
                try:
                    dates = pd.to_datetime(meta["date"], errors="coerce")
                    raw_dates = pd.to_datetime(
                        lookup.loc[meta["match_id"], "date"].to_numpy(), errors="coerce"
                    )
                    mismatch = (
                        dates.dt.normalize().to_numpy() != pd.DatetimeIndex(raw_dates).normalize().to_numpy()
                    ).sum()
                    if mismatch:
                        findings.append(_finding(
                            "feature_row_date_mismatch", ERROR,
                            f"{int(mismatch)} feature rows do not match the raw match date.",
                        ))
                except (KeyError, ValueError):
                    pass
    else:
        report["prediction_time_columns_missing"] = required_meta
    return report, findings, table


def audit_dataset(
    raw: pd.DataFrame | None = None,
    *,
    feature_frame: pd.DataFrame | None = None,
    meta: pd.DataFrame | None = None,
    thresholds: QualityThresholds | None = None,
    now: datetime | None = None,
    raw_path: Path | str = RAW_PATH,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Audit a raw match frame and return ``(report, tables)``.

    The input frame is never modified. ``tables`` contains ``by_column``,
    ``by_season``, ``invalid_scores`` and ``features`` for report writers.
    """
    thresholds = thresholds or QualityThresholds()
    now = now or datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_convert("UTC").tz_localize(None)
    if raw is None:
        path = Path(raw_path)
        if path.exists():
            raw = pd.read_csv(path, parse_dates=["date"])
        else:
            raw = pd.DataFrame()
    frame = raw.copy()
    findings: list[dict] = []
    tables: dict[str, pd.DataFrame] = {}

    report: dict = {
        "generated_at": now.isoformat(timespec="seconds"),
        "status": OK,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "required_columns_present": sorted(set(REQUIRED_COLUMNS) & set(frame.columns)),
        "required_columns_missing": sorted(set(REQUIRED_COLUMNS) - set(frame.columns)),
        "seasons": {},
        "duplicates": {},
        "team_names": {},
        "invalid_scores": {},
        "invalid_dates": {},
        "chronology": {},
        "missing_odds": {},
        "missing_values": {},
        "features": {},
        "thresholds": thresholds.as_dict(),
        "findings": findings,
        "notes": [
            "This audit never deletes or rewrites data.",
            "An in-progress season is allowed to be smaller than a full season.",
        ],
    }

    if frame.empty:
        findings.append(_finding("dataset_empty", ERROR, "The dataset is empty."))
        report["status"] = ERROR
        tables["by_column"] = pd.DataFrame()
        tables["by_season"] = pd.DataFrame()
        tables["invalid_scores"] = pd.DataFrame()
        tables["features"] = pd.DataFrame()
        return _json_safe(report), tables

    missing_required = report["required_columns_missing"]
    if missing_required:
        findings.append(_finding(
            "required_columns_missing", ERROR,
            "Required columns are missing: " + ", ".join(missing_required),
            columns=missing_required,
        ))

    if "date" in frame.columns:
        frame["date"] = parse_match_dates(frame["date"])
    if "season" not in frame.columns and "date" in frame.columns:
        frame["season"] = frame["date"].map(
            lambda d: season_from_date(d) if pd.notna(d) else ""
        )
    seasons = _season_series(frame)

    # Missing values by column/season.
    by_column, by_season, missing_by_column = _missing_tables(frame, seasons, thresholds)
    tables["by_column"] = by_column
    tables["by_season"] = by_season
    report["missing_values"] = missing_by_column
    critical_missing = [
        c for c in missing_by_column
        if c in {"date", "home_team", "away_team", "match_outcome",
                 "full_time_home_goals", "full_time_away_goals"}
    ]
    if critical_missing:
        findings.append(_finding(
            "critical_missing_values", ERROR,
            "Critical columns contain missing values: " + ", ".join(critical_missing),
            columns=critical_missing,
        ))
    sparse = [
        c for c, count in missing_by_column.items()
        if len(frame) and count / len(frame) > thresholds.max_missing_fraction
        and c not in critical_missing
    ]
    if sparse:
        findings.append(_finding(
            "missing_values_threshold", WARNING,
            "Columns exceed the missing-value threshold "
            f"({thresholds.max_missing_fraction:.1%}): " + ", ".join(sparse),
            columns=sparse,
        ))

    # Season sizes. A season is "partial" when it is recent enough to be in
    # progress; a full season outside the expected range is a warning.
    season_rows: list[dict] = []
    for season, group in frame.groupby(seasons, dropna=False):
        dates = pd.to_datetime(group["date"], errors="coerce") if "date" in group else pd.Series(dtype="datetime64[ns]")
        maximum = dates.max() if len(dates.dropna()) else None
        age_days = (
            (now_ts.normalize() - maximum.normalize()).days
            if maximum is not None and pd.notna(maximum) else None
        )
        partial = age_days is not None and age_days <= 60
        status = "ok"
        if partial:
            status = "in_progress"
        elif len(group) > thresholds.max_season_matches:
            status = "too_large"
        elif len(group) < thresholds.min_season_matches:
            status = "too_small"
        season_rows.append({
            "season": str(season),
            "matches": int(len(group)),
            "teams": int(len(set(group.get("home_team", [])) | set(group.get("away_team", [])))),
            "date_min": str(pd.to_datetime(group["date"]).min().date()) if "date" in group and group["date"].notna().any() else None,
            "date_max": str(maximum.date()) if maximum is not None and pd.notna(maximum) else None,
            "age_days": age_days,
            "status": status,
        })
        if status in {"too_large", "too_small"}:
            findings.append(_finding(
                f"season_{status}", WARNING,
                f"Season {season} has {len(group)} matches, outside the expected "
                f"{thresholds.min_season_matches}-{thresholds.max_season_matches}.",
                season=str(season), matches=int(len(group)),
            ))
    season_table = pd.DataFrame(season_rows)
    report["seasons"] = {
        "count": int(len(season_table)),
        "table": season_rows,
        "latest": str(season_table["season"].iloc[-1]) if not season_table.empty else None,
    }

    # Duplicates.
    duplicate_ids = 0
    duplicate_pairs = 0
    exact_duplicates = int(frame.duplicated().sum())
    if "match_id" in frame.columns:
        duplicate_ids = int(frame["match_id"].duplicated(keep=False).sum())
    if {"date", "home_team", "away_team"} <= set(frame.columns):
        duplicate_pairs = int(
            frame.duplicated(subset=["date", "home_team", "away_team"], keep=False).sum()
        )
    report["duplicates"] = {
        "duplicate_fixture_rows": duplicate_ids,
        "duplicate_pairing_rows": duplicate_pairs,
        "exact_duplicate_rows": exact_duplicates,
    }
    if duplicate_ids or duplicate_pairs or exact_duplicates:
        findings.append(_finding(
            "duplicate_fixtures", ERROR,
            f"Duplicate fixtures detected: {duplicate_ids} fixture-ID rows, "
            f"{duplicate_pairs} home/away/date rows, {exact_duplicates} exact rows.",
        ))

    # Team-name inconsistencies.
    report["team_names"] = _team_inconsistencies(frame)
    if report["team_names"]["inconsistent_rows"]:
        findings.append(_finding(
            "team_name_inconsistency", WARNING,
            f"{report['team_names']['inconsistent_rows']} team values are not canonical "
            f"({report['team_names']['distinct_values']} distinct spellings).",
            examples=report["team_names"]["examples"][:5],
        ))

    # Scores and results.
    invalid_scores, score_summary = _invalid_scores(frame, thresholds.max_invalid_score)
    tables["invalid_scores"] = invalid_scores
    report["invalid_scores"] = score_summary
    bad_scores = sum(
        score_summary[key]
        for key in ("non_numeric", "negative", "too_large")
    )
    if bad_scores:
        findings.append(_finding(
            "invalid_scores", ERROR,
            f"{bad_scores} score values are invalid (negative, non-numeric or "
            f"greater than {thresholds.max_invalid_score}).",
            count=bad_scores,
        ))
    if score_summary["score_result_mismatch"]:
        findings.append(_finding(
            "score_result_mismatch", ERROR,
            f"{score_summary['score_result_mismatch']} rows have a scoreline that "
            "contradicts the full-time result code.",
        ))
    if score_summary["missing"]:
        findings.append(_finding(
            "missing_scores", WARNING,
            f"{score_summary['missing']} goal values are missing.",
        ))

    # Dates.
    invalid_dates: dict = {
        "unparseable": 0,
        "future": 0,
        "before_earliest": 0,
    }
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        invalid_dates["unparseable"] = int(dates.isna().sum())
        cutoff = now_ts.normalize() + pd.Timedelta(days=thresholds.future_grace_days)
        invalid_dates["future"] = int((dates > cutoff).sum())
        invalid_dates["before_earliest"] = int(
            (dates < pd.Timestamp(year=thresholds.earliest_year, month=1, day=1)).sum()
        )
    report["invalid_dates"] = invalid_dates
    if sum(invalid_dates.values()):
        findings.append(_finding(
            "invalid_dates", ERROR,
            "Invalid dates detected: "
            + json.dumps(invalid_dates),
            **invalid_dates,
        ))

    # Chronology.
    chronology = _chronology(frame)
    report["chronology"] = chronology
    if chronology["ordering_violations"]:
        findings.append(_finding(
            "chronology_order", ERROR,
            f"{chronology['ordering_violations']} rows are not in ascending date order.",
        ))
    if chronology["same_team_same_day"]:
        findings.append(_finding(
            "same_team_same_day", WARNING,
            f"{chronology['same_team_same_day']} team/date combinations appear more "
            "than once; check for duplicated or mis-dated fixtures.",
            count=chronology["same_team_same_day"],
        ))
    if chronology["match_id_date_mismatches"]:
        findings.append(_finding(
            "match_id_date_mismatch", ERROR,
            f"{chronology['match_id_date_mismatches']} match IDs do not encode their date.",
        ))

    # Odds.
    odds_columns = [c for c in ("odds_home", "odds_draw", "odds_away") if c in frame.columns]
    missing_odds = {}
    if odds_columns:
        numeric = frame[odds_columns].apply(pd.to_numeric, errors="coerce")
        any_missing = numeric.isna().any(axis=1)
        invalid = ((numeric <= 1.0) & numeric.notna()).any(axis=1)
        missing_odds = {
            "rows": int(any_missing.sum()),
            "fraction": float(any_missing.mean()),
            "invalid_odds_rows": int(invalid.sum()),
        }
        report["missing_odds"] = missing_odds
        if missing_odds["rows"] and missing_odds["fraction"] > thresholds.max_missing_odds_fraction:
            findings.append(_finding(
                "missing_odds", WARNING,
                f"{missing_odds['rows']} rows ({missing_odds['fraction']:.1%}) have "
                "missing odds, above the configured threshold "
                f"({thresholds.max_missing_odds_fraction:.1%}).",
            ))
        if missing_odds["invalid_odds_rows"]:
            findings.append(_finding(
                "invalid_odds", ERROR,
                f"{missing_odds['invalid_odds_rows']} rows have decimal odds <= 1.0.",
            ))
    else:
        findings.append(_finding(
            "odds_columns_missing", WARNING, "No odds columns are present.",
        ))

    # Staleness.
    if "date" in frame.columns and frame["date"].notna().any():
        latest = pd.to_datetime(frame["date"], errors="coerce").max()
        age = (now_ts.normalize() - latest.normalize()).days
        report["freshness"] = {
            "latest_match_date": str(latest.date()),
            "age_days": int(age),
            "stale": bool(age > thresholds.stale_days),
        }
        if age > thresholds.stale_days:
            findings.append(_finding(
                "stale_dataset", WARNING,
                f"The latest match is {age} days old (threshold "
                f"{thresholds.stale_days} days).",
            ))
    else:
        report["freshness"] = {"latest_match_date": None, "age_days": None, "stale": None}

    # Feature coverage and prediction-time availability.
    feature_report, feature_findings, feature_table = _feature_audit(
        frame,
        feature_frame=feature_frame,
        meta=meta,
        thresholds=thresholds,
    )
    report["features"] = feature_report
    tables["features"] = feature_table
    findings.extend(feature_findings)

    severity = {finding["severity"] for finding in findings}
    report["status"] = ERROR if ERROR in severity else WARNING if WARNING in severity else OK
    report["summary"] = {
        "errors": sum(1 for f in findings if f["severity"] == ERROR),
        "warnings": sum(1 for f in findings if f["severity"] == WARNING),
        "passed": ERROR not in severity,
    }
    return _json_safe(report), tables


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_No data._"
    columns = [c for c in columns if c in frame.columns]
    if not columns:
        return "_No data._"
    part = frame.head(max_rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in part.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_quality_reports(
    report: dict,
    tables: dict[str, pd.DataFrame],
    *,
    json_path: Path = DATA_QUALITY_REPORT_PATH,
    columns_path: Path = DATA_QUALITY_COLUMNS_PATH,
    seasons_path: Path = DATA_QUALITY_SEASONS_PATH,
    markdown_path: Path = DATA_QUALITY_MARKDOWN_PATH,
) -> list[Path]:
    """Write the data-quality JSON, CSV and Markdown reports."""
    json_path = Path(json_path)
    columns_path = Path(columns_path)
    seasons_path = Path(seasons_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    by_column = tables.get("by_column", pd.DataFrame())
    if by_column.empty:
        # A fresh clone has no cached dataset, so there is no missing-value
        # table. Keep the documented columns so the CSV has a stable header and
        # the Markdown sort below cannot fail on a missing column.
        by_column = pd.DataFrame(
            columns=["column", "rows", "missing", "missing_fraction"]
        )
    by_column.to_csv(columns_path, index=False)
    tables.get("by_season", pd.DataFrame()).to_csv(seasons_path, index=False)

    findings = pd.DataFrame(report.get("findings", []))
    if not findings.empty:
        findings["details"] = findings["details"].map(
            lambda value: json.dumps(value, default=str)
        )
    seasons = pd.DataFrame(report.get("seasons", {}).get("table", []))
    lines = [
        "# La Liga Predictor — Historical Data Quality",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: **{report['status']}** — "
        f"{report.get('summary', {}).get('errors', 0)} error(s), "
        f"{report.get('summary', {}).get('warnings', 0)} warning(s).",
        "",
        "## Dataset",
        "",
        f"- Rows: {report.get('rows', 0)}",
        f"- Seasons: {report.get('seasons', {}).get('count', 0)}",
        f"- Latest match: {report.get('freshness', {}).get('latest_match_date')}",
        f"- Age: {report.get('freshness', {}).get('age_days')} days",
        "",
        "## Season sizes",
        "",
        _markdown_table(
            seasons,
            ["season", "matches", "teams", "date_min", "date_max", "status"],
        ),
        "",
        "## Duplicates and chronology",
        "",
        f"- Duplicate fixture-ID rows: {report['duplicates'].get('duplicate_fixture_rows', 0)}",
        f"- Duplicate pairing rows: {report['duplicates'].get('duplicate_pairing_rows', 0)}",
        f"- Ordering violations: {report['chronology'].get('ordering_violations', 0)}",
        f"- Same team/date occurrences: {report['chronology'].get('same_team_same_day', 0)}",
        "",
        "## Missing odds",
        "",
        f"- Rows with missing odds: {report.get('missing_odds', {}).get('rows', 0)}",
        f"- Fraction: {report.get('missing_odds', {}).get('fraction', 0.0):.2%}",
        "",
        "## Feature availability",
        "",
        f"- Canonical features audited: {report.get('features', {}).get('n_features', 0)}",
        f"- Rows with every feature present: "
        f"{report.get('features', {}).get('rows_fully_available', 0)}",
        f"- Feature coverage: "
        f"{report.get('features', {}).get('feature_coverage_fraction') or 0:.2%}",
        "",
        "## Findings",
        "",
        _markdown_table(findings, ["severity", "code", "message"], max_rows=50),
        "",
        "## Missing values by column",
        "",
        _markdown_table(
            by_column.sort_values("missing_fraction", ascending=False),
            ["column", "missing", "missing_fraction"],
            max_rows=40,
        ),
        "",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, columns_path, seasons_path, markdown_path]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    report, tables = audit_dataset()
    paths = write_quality_reports(report, tables)
    print(json.dumps({
        "status": report["status"],
        "rows": report["rows"],
        "errors": report["summary"]["errors"],
        "warnings": report["summary"]["warnings"],
        "reports": [str(path) for path in paths],
    }, indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
