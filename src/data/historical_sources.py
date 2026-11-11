"""Audit of legitimate historical xG and player-availability sources.

No network request is made here. A source is only marked ``available`` when a
local, timestamped extract actually exists; otherwise it is reported as
``not_configured`` or ``unavailable``. This is deliberate: claiming external
coverage without measuring it is exactly the failure mode Phase 6 forbids.

Candidates are documented with what they do and do not provide. Free fixture
APIs do not supply historical xG, and player-availability archives are mostly
paid or not redistributable. The audited candidate list is therefore a
planning document as much as a runtime check.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.optional_features import (
    load_player_availability,
    load_xg_history,
)
from src.utils.config import (
    HISTORICAL_SOURCE_AUDIT_MARKDOWN_PATH,
    HISTORICAL_SOURCE_AUDIT_PATH,
    PLAYER_AVAILABILITY_PATH,
    RAW_PATH,
    XG_HISTORY_PATH,
)

logger = logging.getLogger(__name__)

SOURCE_AVAILABLE = "available_local_extract"
SOURCE_NOT_CONFIGURED = "not_configured"
SOURCE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HistoricalSource:
    """One audited source candidate and its actual local status."""

    name: str
    kind: str  # "xg" or "player_availability"
    provider_type: str
    historical: str
    credentials: str
    redistribution: str
    status: str
    notes: str
    documentation: str = ""
    local_path: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# The list is intentionally conservative. "available" here means the source
# exists and is legitimate to use, not that this repository holds its data.
SOURCE_CANDIDATES: tuple[HistoricalSource, ...] = (
    HistoricalSource(
        name="Football-Data.co.uk SP1",
        kind="results_and_odds",
        provider_type="published CSV",
        historical="Four cached La Liga season sheets (shots and Bet365 odds)",
        credentials="none",
        redistribution="public download",
        status=SOURCE_AVAILABLE,
        notes=(
            "Provides results, shots and odds but not xG or player availability. "
            "It is the production results source, not an optional feature source."
        ),
        documentation="https://www.football-data.co.uk/spainm.php",
    ),
    HistoricalSource(
        name="Understat",
        kind="xg",
        provider_type="community dataset / scrape",
        historical="Team and player xG by match from 2014/15 onward",
        credentials="none",
        redistribution="no official API; respect site terms and rate limits",
        status=SOURCE_NOT_CONFIGURED,
        notes=(
            "A plausible historical xG source, but no local timestamped extract is "
            "committed. Retrieval requires a scraper, availability timestamps are "
            "not published, and post-match xG must be lagged before kickoff. "
            "Do not treat it as configured until an extract with available_at exists."
        ),
        documentation="https://understat.com/",
    ),
    HistoricalSource(
        name="FBref / StatsBomb data",
        kind="xg",
        provider_type="published tables / open datasets",
        historical="Selected competitions and seasons; not a complete La Liga archive",
        credentials="none for public pages",
        redistribution="site terms; StatsBomb open data has its own licence",
        status=SOURCE_NOT_CONFIGURED,
        notes=(
            "Coverage is incomplete for the cached seasons and access is not a "
            "stable API. Not integrated, so no coverage is claimed."
        ),
        documentation="https://fbref.com/",
    ),
    HistoricalSource(
        name="API-Football (xG)",
        kind="xg",
        provider_type="commercial API",
        historical="xG availability depends on plan/league; free tier has no reliable xG",
        credentials="API key required",
        redistribution="contract terms",
        status=SOURCE_NOT_CONFIGURED,
        notes=(
            "The fixture-discovery integration only uses the fixtures endpoint. "
            "Historical xG on the free plan is not assumed; a paid plan and a "
            "verified extract would be required before enabling the xG group."
        ),
        documentation="https://www.api-football.com/documentation-v3",
    ),
    HistoricalSource(
        name="API-Football (injuries/lineups)",
        kind="player_availability",
        provider_type="commercial API",
        historical="Current injuries and lineups; historical snapshots require paid access",
        credentials="API key required",
        redistribution="contract terms",
        status=SOURCE_NOT_CONFIGURED,
        notes=(
            "Availability must be timestamped before kickoff. The free plan and "
            "the fixture provider do not provide a historical availability archive."
        ),
        documentation="https://www.api-football.com/documentation-v3",
    ),
    HistoricalSource(
        name="Transfermarkt",
        kind="player_availability",
        provider_type="community scrape",
        historical="Injury histories, but publication timestamps are not an API field",
        credentials="none",
        redistribution="site terms; scraping restrictions apply",
        status=SOURCE_NOT_CONFIGURED,
        notes=(
            "Without a reliable publication timestamp, using it would risk "
            "post-kickoff information leakage. Not integrated."
        ),
        documentation="https://www.transfermarkt.com/",
    ),
    HistoricalSource(
        name="Local xG history extract",
        kind="xg",
        provider_type="repository CSV",
        historical="Requires xg_history.csv with team/date/available_at",
        credentials="none",
        redistribution="depends on upstream source",
        status=SOURCE_NOT_CONFIGURED,
        notes=(
            "Place a legitimate, timestamped extract at data/raw/xg_history.csv to "
            "enable the optional xG group. When absent, the group is unavailable."
        ),
        local_path=XG_HISTORY_PATH.as_posix(),
    ),
    HistoricalSource(
        name="Local player-availability extract",
        kind="player_availability",
        provider_type="repository CSV",
        historical="Requires player_availability.csv with team/player/status/available_at",
        credentials="none",
        redistribution="depends on upstream source",
        status=SOURCE_NOT_CONFIGURED,
        notes=(
            "Place a legitimate, timestamped extract at data/raw/player_availability.csv "
            "to enable the availability group. When absent, it is unavailable."
        ),
        local_path=PLAYER_AVAILABILITY_PATH.as_posix(),
    ),
)


def _local_status(source: HistoricalSource) -> HistoricalSource:
    """Return the candidate with status derived from the local file, if any."""
    if not source.local_path:
        return source
    path = Path(source.local_path)
    if not path.exists():
        return source
    status = SOURCE_AVAILABLE if path.stat().st_size > 0 else SOURCE_UNAVAILABLE
    note = source.notes
    if status == SOURCE_AVAILABLE:
        note = "Local timestamped extract found; coverage is measured below."
    return HistoricalSource(**{**source.as_dict(), "status": status, "notes": note})


def _coverage_frames(
    raw: pd.DataFrame,
    xg: pd.DataFrame,
    availability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage by season/team without inventing observations."""
    if raw.empty or "date" not in raw.columns:
        columns = ["season", "team", "matches", "with_source_before_kickoff"]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if "season" not in frame.columns:
        from src.data.validation import season_from_date

        frame["season"] = frame["date"].map(season_from_date)

    def coverage_table(
        source: pd.DataFrame,
        *,
        kind: str,
    ) -> pd.DataFrame:
        if source.empty:
            return pd.DataFrame(columns=[
                "season", "team", "matches", "with_source_before_kickoff",
            ])
        stamp_column = "available_at"
        source = source.copy()
        source["team"] = source["team"].astype(str).str.strip()
        source[stamp_column] = pd.to_datetime(source[stamp_column], errors="coerce")
        source = source.dropna(subset=[stamp_column])
        rows: list[dict] = []
        long = pd.concat([
            frame[["season", "date", "home_team"]].rename(columns={"home_team": "team"}),
            frame[["season", "date", "away_team"]].rename(columns={"away_team": "team"}),
        ], ignore_index=True)
        if kind == "xg" and "match_id" in frame.columns:
            long["match_id"] = pd.concat([frame["match_id"], frame["match_id"]], ignore_index=True)
        for (season, team), group in long.groupby(["season", "team"], sort=True):
            observations = source[source["team"] == team]
            if observations.empty:
                covered = 0
            else:
                covered = 0
                stamps = observations[stamp_column].sort_values().to_numpy()
                for kickoff in pd.to_datetime(group["date"]).dropna():
                    count = int((stamps < pd.Timestamp(kickoff).to_datetime64()).sum())
                    if count:
                        covered += 1
            rows.append({
                "season": str(season),
                "team": str(team),
                "matches": int(len(group)),
                "with_source_before_kickoff": int(covered),
            })
        table = pd.DataFrame(rows)
        table["coverage_fraction"] = (
            table["with_source_before_kickoff"] / table["matches"].replace(0, pd.NA)
        ).astype(float)
        return table

    return coverage_table(xg, kind="xg"), coverage_table(availability, kind="player_availability")


def audit_historical_sources(
    raw: pd.DataFrame | None = None,
    *,
    xg_path: Path | str | None = None,
    availability_path: Path | str | None = None,
    now: datetime | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Audit candidate xG/availability sources and local coverage.

    Returns ``(report, tables)`` where tables contain season/team coverage for
    each source. No network call is performed and no missing value is filled.
    """
    now = now or datetime.now(timezone.utc)
    if raw is None:
        raw_path = Path(RAW_PATH)
        raw = pd.read_csv(raw_path, parse_dates=["date"]) if raw_path.exists() else pd.DataFrame()

    # Read local extracts through the same loaders the feature pipeline uses,
    # so the audit and the model cannot disagree about what is available.
    xg = load_xg_history(Path(xg_path) if xg_path is not None else None)
    availability = load_player_availability(
        Path(availability_path) if availability_path is not None else None
    )

    xg_coverage, availability_coverage = _coverage_frames(raw, xg, availability)

    sources: list[dict] = []
    for candidate in SOURCE_CANDIDATES:
        resolved = _local_status(candidate)
        entry = resolved.as_dict()
        if resolved.kind == "xg" and resolved.local_path:
            entry["local_rows"] = int(len(xg))
            entry["seasons_covered"] = sorted(
                xg_coverage["season"].astype(str).unique().tolist()
            ) if not xg_coverage.empty else []
        elif resolved.kind == "player_availability" and resolved.local_path:
            entry["local_rows"] = int(len(availability))
            entry["seasons_covered"] = sorted(
                availability_coverage["season"].astype(str).unique().tolist()
            ) if not availability_coverage.empty else []
        sources.append(entry)

    xg_available = not xg.empty
    availability_available = not availability.empty
    findings: list[dict] = []
    if not xg_available:
        findings.append({
            "kind": "xg",
            "status": SOURCE_UNAVAILABLE,
            "message": (
                "No timestamped historical xG extract is present. The xG feature "
                "group is unavailable and the production model is unchanged."
            ),
        })
    if not availability_available:
        findings.append({
            "kind": "player_availability",
            "status": SOURCE_UNAVAILABLE,
            "message": (
                "No timestamped player-availability extract is present. The "
                "availability feature group is unavailable and no values are fabricated."
            ),
        })

    coverage_summary = {}
    for kind, table, source in (
        ("xg", xg_coverage, xg),
        ("player_availability", availability_coverage, availability),
    ):
        if table.empty:
            coverage_summary[kind] = {
                "available": False,
                "rows": 0,
                "seasons": [],
                "teams": 0,
                "matches_with_pre_kickoff_source": 0,
                "mean_team_coverage_fraction": None,
            }
            continue
        coverage_summary[kind] = {
            "available": True,
            "source_rows": int(len(source)),
            "seasons": sorted(table["season"].astype(str).unique().tolist()),
            "teams": int(table["team"].nunique()),
            "matches_with_pre_kickoff_source": int(
                table["with_source_before_kickoff"].sum()
            ),
            "mean_team_coverage_fraction": float(
                table["coverage_fraction"].fillna(0).mean()
            ),
        }

    report = {
        "generated_at": now.isoformat(timespec="seconds"),
        "network_requests": 0,
        "production_model_unchanged": True,
        "seasons": sorted(raw["season"].astype(str).unique().tolist())
        if "season" in raw.columns else [],
        "source_candidates": sources,
        "coverage": coverage_summary,
        "findings": findings,
        "local_paths": {
            "xg": Path(XG_HISTORY_PATH).as_posix(),
            "player_availability": Path(PLAYER_AVAILABILITY_PATH).as_posix(),
        },
        "notes": [
            "A source is only 'available' when a local timestamped extract exists.",
            "Coverage is measured from that extract; no external provider coverage is claimed.",
            "Timestamps before kickoff are required; post-match xG is lagged by the loader.",
        ],
    }
    return report, {"xg_coverage": xg_coverage, "availability_coverage": availability_coverage}


def write_historical_source_report(
    report: dict,
    tables: dict[str, pd.DataFrame],
    *,
    json_path: Path = HISTORICAL_SOURCE_AUDIT_PATH,
    markdown_path: Path = HISTORICAL_SOURCE_AUDIT_MARKDOWN_PATH,
) -> list[Path]:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    sources = pd.DataFrame(report["source_candidates"])
    lines = [
        "# La Liga Predictor — Historical xG and Player Availability Source Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "No network requests were made. 'Available' means a local timestamped "
        "extract exists; external provider coverage is not claimed.",
        "",
        "## Candidate sources",
        "",
    ]
    if not sources.empty:
        columns = ["name", "kind", "provider_type", "credentials", "status"]
        part = sources[columns]
        lines += [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---"] * len(columns)) + "|",
        ]
        for _, row in part.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    lines += ["", "## Measured local coverage", ""]
    for kind in ("xg", "player_availability"):
        summary = report["coverage"][kind]
        lines += [
            f"### {kind}",
            "",
            f"- Available: **{summary['available']}**",
            f"- Source rows: {summary.get('source_rows', 0)}",
            f"- Seasons: {', '.join(summary.get('seasons', [])) or 'none'}",
            f"- Teams: {summary.get('teams', 0)}",
            f"- Matches with a pre-kickoff observation: "
            f"{summary.get('matches_with_pre_kickoff_source', 0)}",
            f"- Mean team coverage fraction: "
            f"{summary.get('mean_team_coverage_fraction')}",
            "",
        ]
        table = tables.get(f"{kind}_coverage")
        if table is not None and not table.empty:
            lines.append("| season | team | matches | covered before kickoff | fraction |")
            lines.append("|---|---|---|---|---|")
            for _, row in table.head(40).iterrows():
                fraction = row.get("coverage_fraction")
                lines.append(
                    f"| {row['season']} | {row['team']} | {row['matches']} | "
                    f"{row['with_source_before_kickoff']} | "
                    f"{'' if pd.isna(fraction) else f'{float(fraction):.1%}'} |"
                )
            lines.append("")
    lines += ["## Findings", ""]
    for finding in report["findings"]:
        lines.append(f"- **{finding['kind']}**: {finding['message']}")
    if not report["findings"]:
        lines.append("- None.")
    lines += [
        "",
        "## What remains unavailable",
        "",
        "- No historical xG extract is committed, so xG-based models are not trained "
        "and no xG improvement is claimed.",
        "- No historical player-availability archive is committed, so lineup/injury "
        "features are not trained.",
        "- The fixture APIs used elsewhere are not historical xG or availability "
        "providers and are not described as such.",
        "",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, markdown_path]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    report, tables = audit_historical_sources()
    paths = write_historical_source_report(report, tables)
    print(json.dumps({
        "xg_available": report["coverage"]["xg"]["available"],
        "player_availability_available": report["coverage"]["player_availability"]["available"],
        "reports": [str(path) for path in paths],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
