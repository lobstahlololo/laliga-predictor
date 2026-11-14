"""Append-only prediction history with result reconciliation.

Design rules:

* Every prediction stores a stable ``fixture_id``, the match date (when
  known), the prediction timestamp, the three model probabilities, the
  predicted outcome, the model name/version, market probabilities and the
  eventual actual result.
* Predictions are **never overwritten**. Re-running prediction for the same
  fixture at the same timestamp/model version is de-duplicated; a new
  snapshot (new timestamp or new model version) is appended.
* Reconciliation only fills blank ``actual_outcome``/result fields. Settled
  rows are immutable.
* Legacy ledgers (``date_predicted`` + ``model_prob_*``) are migrated in
  memory and preserved on the next write.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validation import season_from_date
from src.utils.config import LEDGER_PATH
from src.utils.teams import canonical_team

logger = logging.getLogger(__name__)

# Phase 8 lifecycle columns are appended after the Phase 4 schema so old
# ledgers keep loading unchanged. Every column is derived for legacy rows.
CANONICAL_COLUMNS = [
    "prediction_id",
    "fixture_id",
    "prediction_cycle",
    "match_date",
    "kickoff_utc",
    "home_team",
    "away_team",
    "predicted_at",
    "model_name",
    "model_version",
    "model_prob_home",
    "model_prob_draw",
    "model_prob_away",
    "predicted_outcome",
    "confidence",
    "market_prob_home",
    "market_prob_draw",
    "market_prob_away",
    "fixture_source",
    "prediction_status",
    "schedule_revision",
    "supersedes_prediction_id",
    "expected_home_goals",
    "expected_away_goals",
    "most_likely_scoreline",
    "actual_outcome",
    "actual_home_goals",
    "actual_away_goals",
    "correct",
    "result_source",
    "result_status",
    "settled_at",
    "correction_count",
    "corrected_at",
]

STATUS_PENDING = "pending"
STATUS_SETTLED = "settled"
STATUS_VOID = "void"
RESULT_FINAL = "final"
RESULT_CORRECTED = "corrected"

PROB_COLUMNS = ["model_prob_home", "model_prob_draw", "model_prob_away"]
MARKET_COLUMNS = ["market_prob_home", "market_prob_draw", "market_prob_away"]
NUMERIC_COLUMNS = PROB_COLUMNS + MARKET_COLUMNS + ["confidence"]

OUTCOME_LABELS = {2: "Home Win", 1: "Draw", 0: "Away Win"}
LABEL_TO_CODE = {v: k for k, v in OUTCOME_LABELS.items()}
FTR_TO_OUTCOME = {"H": 2, "D": 1, "A": 0}
# Probability columns are [home, draw, away]; labels follow that positional
# order, while OUTCOME_LABELS maps numeric result codes.
PREDICTION_LABELS = ["Home Win", "Draw", "Away Win"]


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS, dtype=object)


def make_fixture_id(home: str, away: str, match_date: str | None = None) -> str:
    """Stable ID matching the raw dataset's ``match_id`` convention.

    When the kickoff date is unknown (a user-provided upcoming fixture), the
    ID is ``TBD_Home_v_Away``. The prediction timestamp still distinguishes
    repeated snapshots.
    """
    home_c, away_c = canonical_team(home), canonical_team(away)
    clean_date = str(match_date or "").strip()
    if clean_date and clean_date.lower() not in {"nan", "none", "nat", "tbd"}:
        parsed = pd.to_datetime(clean_date, errors="coerce")
        if pd.notna(parsed):
            return f"{parsed:%Y%m%d}_{home_c}_v_{away_c}"
    return f"TBD_{home_c}_v_{away_c}"


def _clean_str(value) -> str:
    """Normalise one CSV cell, treating pandas placeholders as absent."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return ""
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "nat", "<na>"} else text


def make_prediction_id(
    fixture_id,
    predicted_at,
    model_version,
    prediction_cycle: str = "",
) -> str:
    """Deterministic identity for one prediction snapshot.

    The ID depends on the fixture, the snapshot timestamp, the model version and
    the prediction cycle, so a re-run of the same cycle cannot create a second
    ID for an existing row.
    """
    payload = "|".join(
        _clean_str(value)
        for value in (fixture_id, predicted_at, model_version, prediction_cycle)
    )
    return "pred_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _to_label(code) -> str:
    try:
        return OUTCOME_LABELS.get(int(float(code)), "")
    except (TypeError, ValueError):
        return ""


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _infer_predicted_outcome(row: pd.Series) -> str:
    """Derive the pick from the stored probabilities.

    Always recomputed when probabilities are available so legacy rows with
    an inconsistent label are corrected on load. Without probabilities the
    existing label is preserved.
    """
    probs = [_safe_float(row.get(c)) for c in PROB_COLUMNS]
    if all(p is not None for p in probs):
        return PREDICTION_LABELS[int(np.argmax(probs))]
    existing = str(row.get("predicted_outcome", "")).strip()
    return existing if existing in LABEL_TO_CODE else ""


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Migrate legacy columns and fill missing canonical fields."""
    if df is None or df.empty:
        return empty_history()

    out = df.copy()
    # Legacy -> canonical renames.
    if "date_predicted" in out.columns and "predicted_at" not in out.columns:
        out = out.rename(columns={"date_predicted": "predicted_at"})

    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[CANONICAL_COLUMNS].copy()

    for col in out.columns:
        out[col] = out[col].astype(object)

    out["home_team"] = out["home_team"].astype(str).str.strip().map(canonical_team)
    out["away_team"] = out["away_team"].astype(str).str.strip().map(canonical_team)

    # Fill IDs and dates for legacy rows.
    for idx in out.index:
        home = str(out.at[idx, "home_team"]).strip()
        away = str(out.at[idx, "away_team"]).strip()
        if not str(out.at[idx, "fixture_id"]).strip() and home and away:
            out.at[idx, "fixture_id"] = make_fixture_id(
                home, away, out.at[idx, "match_date"]
            )
        if not str(out.at[idx, "predicted_at"]).strip():
            out.at[idx, "predicted_at"] = ""

    # ---- Phase 8 lifecycle fields (derived for legacy rows) ----
    for column in (
        "kickoff_utc", "prediction_cycle", "fixture_source",
        "supersedes_prediction_id", "prediction_id", "expected_home_goals",
        "expected_away_goals", "most_likely_scoreline", "result_source",
        "settled_at", "corrected_at",
    ):
        out[column] = out[column].map(_clean_str)
    out["prediction_status"] = out["prediction_status"].map(_clean_str).str.lower()
    out["result_status"] = out["result_status"].map(_clean_str).str.lower()
    out["schedule_revision"] = out["schedule_revision"].map(_clean_str)
    out["correction_count"] = out["correction_count"].map(_clean_str)
    blank_ids = out["prediction_id"] == ""
    if blank_ids.any():
        out.loc[blank_ids, "prediction_id"] = out[blank_ids].apply(
            lambda row: make_prediction_id(
                row["fixture_id"], row["predicted_at"],
                row["model_version"], row["prediction_cycle"],
            ),
            axis=1,
        )
    out.loc[out["prediction_status"] == "", "prediction_status"] = STATUS_PENDING
    out.loc[out["schedule_revision"] == "", "schedule_revision"] = "0"
    out.loc[out["correction_count"] == "", "correction_count"] = "0"

    out["predicted_outcome"] = out.apply(_infer_predicted_outcome, axis=1)
    out["actual_outcome"] = out["actual_outcome"].map(
        lambda v: "" if str(v).strip() == "" else str(int(float(v)))
        if _safe_float(v) is not None else ""
    )

    # Derive confidence from the predicted outcome's probability.
    for idx in out.index:
        label = out.at[idx, "predicted_outcome"]
        if label not in LABEL_TO_CODE:
            out.at[idx, "confidence"] = ""
            continue
        prob_col = PROB_COLUMNS[PREDICTION_LABELS.index(label)]
        value = _safe_float(out.at[idx, prob_col])
        out.at[idx, "confidence"] = "" if value is None else f"{value:.6f}"

    # Derive correctness for settled rows.
    for idx in out.index:
        actual = out.at[idx, "actual_outcome"]
        if str(actual).strip() == "":
            out.at[idx, "correct"] = ""
            continue
        out.at[idx, "correct"] = str(
            int(out.at[idx, "predicted_outcome"] == _to_label(actual))
        )

    # Lifecycle status follows the stored result, never the other way round.
    settled_mask = out["actual_outcome"].astype(str).str.strip() != ""
    void_mask = out["result_status"].isin({"cancelled", "canceled", "abandoned"})
    out.loc[settled_mask, "prediction_status"] = STATUS_SETTLED
    out.loc[
        settled_mask & (out["result_status"] == ""), "result_status"
    ] = RESULT_FINAL
    out.loc[void_mask & ~settled_mask, "prediction_status"] = STATUS_VOID

    # Remove exact duplicate rows only; distinct snapshots are preserved.
    before = len(out)
    out = out.drop_duplicates(subset=CANONICAL_COLUMNS, keep="first")
    if len(out) != before:
        logger.info("Removed %d exact duplicate ledger rows.", before - len(out))
    return out.reset_index(drop=True)


def load_history(
    path: Path = LEDGER_PATH, *, migrate: bool = True
) -> tuple[pd.DataFrame, bool]:
    """Load the ledger. Returns ``(history, needs_write)``.

    ``needs_write`` is True when legacy/blank fields were filled in memory,
    so callers can persist the migration without ever discarding rows.
    """
    if not path.exists():
        return empty_history(), False
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001 - a corrupt ledger must not crash
        logger.warning("Could not read ledger %s: %s", path, exc)
        return empty_history(), False
    normalized = normalize_history(raw)
    aligned = (
        raw.reindex(columns=CANONICAL_COLUMNS)
        .reset_index(drop=True)
        .astype(str)
    )
    changed = (
        migrate
        and len(normalized) > 0
        and not normalized.astype(str).equals(aligned)
    )
    return normalized, changed


def save_history(history: pd.DataFrame, path: Path = LEDGER_PATH) -> None:
    """Atomically write the canonical ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = normalize_history(history)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def build_prediction_record(
    *,
    home_team: str,
    away_team: str,
    match_date: str | None,
    predicted_at: str,
    model_name: str,
    model_version: str,
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    market_probs: dict[str, float] | None = None,
    prediction_cycle: str = "",
    kickoff_utc: str = "",
    fixture_source: str = "",
    prediction_status: str = STATUS_PENDING,
    schedule_revision: int | str = 0,
    supersedes_prediction_id: str = "",
    expected_home_goals: float | None = None,
    expected_away_goals: float | None = None,
    most_likely_scoreline: str = "",
    actual_outcome: str = "",
    actual_home_goals: str = "",
    actual_away_goals: str = "",
    result_source: str = "",
    result_status: str = "",
    settled_at: str = "",
) -> dict:
    """Build one canonical history row, validating probabilities."""
    probs = np.array([prob_home, prob_draw, prob_away], dtype=float)
    if not np.isfinite(probs).all() or (probs < -1e-8).any():
        raise ValueError(f"Invalid probabilities: {probs}")
    total = probs.sum()
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive value.")
    probs = probs / total  # guarantee the display/ledger sums to exactly 1
    label = PREDICTION_LABELS[int(np.argmax(probs))]
    market = market_probs or {}
    fixture_id = make_fixture_id(home_team, away_team, match_date)
    return {
        "prediction_id": make_prediction_id(
            fixture_id, predicted_at, model_version, prediction_cycle
        ),
        "fixture_id": fixture_id,
        "prediction_cycle": _clean_str(prediction_cycle),
        "match_date": str(match_date or "").strip(),
        "kickoff_utc": _clean_str(kickoff_utc),
        "home_team": canonical_team(home_team),
        "away_team": canonical_team(away_team),
        "predicted_at": str(predicted_at),
        "model_name": model_name,
        "model_version": model_version,
        "model_prob_home": f"{probs[0]:.6f}",
        "model_prob_draw": f"{probs[1]:.6f}",
        "model_prob_away": f"{probs[2]:.6f}",
        "predicted_outcome": label,
        "confidence": f"{probs.max():.6f}",
        "market_prob_home": _format_prob(market.get("home")),
        "market_prob_draw": _format_prob(market.get("draw")),
        "market_prob_away": _format_prob(market.get("away")),
        "fixture_source": _clean_str(fixture_source),
        "prediction_status": _clean_str(prediction_status) or STATUS_PENDING,
        "schedule_revision": str(int(schedule_revision or 0)),
        "supersedes_prediction_id": _clean_str(supersedes_prediction_id),
        "expected_home_goals": _format_prob(expected_home_goals),
        "expected_away_goals": _format_prob(expected_away_goals),
        "most_likely_scoreline": _clean_str(most_likely_scoreline),
        "actual_outcome": _clean_str(actual_outcome),
        "actual_home_goals": _clean_str(actual_home_goals),
        "actual_away_goals": _clean_str(actual_away_goals),
        "correct": "",
        "result_source": _clean_str(result_source),
        "result_status": _clean_str(result_status),
        "settled_at": _clean_str(settled_at),
        "correction_count": "0",
        "corrected_at": "",
    }


def _format_prob(value) -> str:
    parsed = _safe_float(value)
    return "" if parsed is None else f"{parsed:.6f}"


def append_predictions(
    records: list[dict],
    path: Path = LEDGER_PATH,
) -> tuple[int, pd.DataFrame]:
    """Append prediction snapshots, skipping exact duplicates.

    Returns ``(n_appended, full_history)``. Existing settled rows are never
    modified by this function.
    """
    if not records:
        return 0, load_history(path)[0]

    history, _ = load_history(path)
    new = normalize_history(pd.DataFrame(records, columns=CANONICAL_COLUMNS))
    if history.empty:
        combined = new
    else:
        combined = pd.concat([history, new], ignore_index=True)

    # One snapshot per fixture per prediction cycle: re-running the same cycle
    # never appends a duplicate. Legacy rows (no cycle) keep the older identity.
    combined = combined.reset_index(drop=True)
    cycle = combined["prediction_cycle"].astype(str).str.strip()
    keep = pd.Series(False, index=combined.index)
    if (cycle != "").any():
        cycled = combined[cycle != ""]
        keep.loc[
            cycled.index[
                ~cycled.duplicated(
                    subset=["fixture_id", "prediction_cycle", "model_version"],
                    keep="first",
                )
            ]
        ] = True
    legacy = combined[cycle == ""]
    if not legacy.empty:
        keep.loc[
            legacy.index[
                ~legacy.duplicated(
                    subset=["fixture_id", "predicted_at", "model_name",
                            "model_version", "model_prob_home",
                            "model_prob_draw", "model_prob_away"],
                    keep="first",
                )
            ]
        ] = True
    combined = combined[keep]
    appended = len(combined) - len(history)
    if appended <= 0:
        logger.info("No new predictions to append (all duplicates).")
        return 0, history
    save_history(combined, path)
    return appended, combined


# ---------------------------------------------------------------------------
# Result reconciliation
# ---------------------------------------------------------------------------

def reconcile_results(
    history: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fill blank actual results from the raw dataset.

    Matching order: exact ``fixture_id``, then the same home/away pairing with
    a match date on/after the prediction date and on/before ``as_of``. A
    fixture is never settled from an earlier meeting or a future fixture.
    Existing settled rows are untouched.
    """
    stats = {"settled": 0, "pending": 0, "unresolved": 0, "already_settled": 0}
    if history.empty:
        return history, stats
    if raw is None or raw.empty:
        stats["pending"] = int((history["actual_outcome"].astype(str).str.strip() == "").sum())
        return history, stats

    out = history.copy()
    today = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw_by_id = {
        str(row["match_id"]): row for _, row in raw.iterrows()
        if "match_id" in raw.columns
    }

    for idx in out.index:
        if str(out.at[idx, "actual_outcome"]).strip() != "":
            stats["already_settled"] += 1
            continue

        home = canonical_team(out.at[idx, "home_team"])
        away = canonical_team(out.at[idx, "away_team"])
        fixture_id = str(out.at[idx, "fixture_id"]).strip()
        match = raw_by_id.get(fixture_id)

        if match is None:
            candidates = raw[
                (raw["home_team"].astype(str) == home)
                & (raw["away_team"].astype(str) == away)
            ]
            if candidates.empty:
                stats["pending"] += 1
                continue
            predicted_at = pd.to_datetime(
                str(out.at[idx, "predicted_at"]).strip(), errors="coerce"
            )
            # Without a prediction timestamp there is no safe way to tell
            # which meeting of the pairing was being predicted. Never settle
            # from an arbitrary earlier season.
            if pd.isna(predicted_at):
                logger.warning(
                    "No valid prediction timestamp for %s vs %s - left pending.",
                    home, away,
                )
                stats["pending"] += 1
                continue
            dates = pd.to_datetime(candidates["date"], errors="coerce")
            played = candidates[(dates <= today)]
            predicted_at = _naive_timestamp(predicted_at)
            played = played[
                dates.loc[played.index] >= predicted_at.normalize()
            ]
            if played.empty:
                stats["pending"] += 1
                continue
            match = played.sort_values("date").iloc[0]

        match_date = pd.to_datetime(match.get("date"), errors="coerce")
        if pd.isna(match_date) or match_date > today:
            stats["pending"] += 1
            continue

        outcome = match.get("match_outcome")
        if pd.isna(outcome):
            outcome = FTR_TO_OUTCOME.get(
                str(match.get("full_time_result", "")).strip().upper()
            )
        if outcome is None or pd.isna(outcome):
            stats["pending"] += 1
            continue

        actual_code = int(outcome)
        actual_label = _to_label(actual_code)
        out.at[idx, "actual_outcome"] = str(actual_code)
        out.at[idx, "actual_home_goals"] = _goal_value(match.get("full_time_home_goals"))
        out.at[idx, "actual_away_goals"] = _goal_value(match.get("full_time_away_goals"))
        out.at[idx, "correct"] = str(int(out.at[idx, "predicted_outcome"] == actual_label))
        out.at[idx, "settled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not str(out.at[idx, "match_date"]).strip():
            out.at[idx, "match_date"] = f"{match_date:%Y-%m-%d}"
        stats["settled"] += 1

    return out, stats


def _goal_value(value) -> str:
    parsed = _safe_float(value)
    return "" if parsed is None else str(int(parsed))


def _naive_timestamp(value) -> pd.Timestamp:
    """Return a timezone-naive timestamp for comparisons with match dates."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def update_results(
    raw: pd.DataFrame,
    path: Path = LEDGER_PATH,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Reconcile and persist only when at least one row was settled."""
    history, _ = load_history(path)
    updated, stats = reconcile_results(history, raw, as_of=as_of)
    if stats["settled"]:
        save_history(updated, path)
    return updated, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def settled_only(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    return history[history["actual_outcome"].astype(str).str.strip() != ""].copy()


def add_derived_columns(history: pd.DataFrame) -> pd.DataFrame:
    """Add numeric probabilities, season, accuracy and model labels."""
    out = normalize_history(history) if not history.empty else empty_history()
    if out.empty:
        return out
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["predicted_at_ts"] = pd.to_datetime(out["predicted_at"], errors="coerce")
    out["match_date_ts"] = pd.to_datetime(out["match_date"], errors="coerce")
    # Season comes from the match date when known, else the prediction date
    # (the upcoming fixture belongs to the current season).
    effective = out["match_date_ts"].fillna(out["predicted_at_ts"])
    out["season"] = effective.map(
        lambda d: season_from_date(d) if pd.notna(d) else ""
    )
    out["actual_label"] = out["actual_outcome"].map(_to_label)
    out["is_correct"] = pd.to_numeric(out["correct"], errors="coerce")
    out["model_label"] = out["model_name"].where(
        out["model_name"].astype(str).str.strip() != "", "legacy/unknown"
    )
    return out


def performance_summary(history: pd.DataFrame) -> dict:
    """Overall settled-prediction metrics from the ledger itself."""
    df = add_derived_columns(history)
    settled = df[df["actual_label"].astype(str) != ""] if not df.empty else df
    summary = {
        "predictions": int(len(df)),
        "settled": int(len(settled)),
        "pending": int(len(df) - len(settled)),
        "accuracy": float("nan"),
        "log_loss": float("nan"),
        "brier_score": float("nan"),
        "draw_recall": float("nan"),
    }
    if settled.empty:
        return summary

    summary["accuracy"] = float(settled["is_correct"].mean())
    probs = settled[PROB_COLUMNS].to_numpy(dtype=float)
    actual = settled["actual_outcome"].astype(int).to_numpy()
    labels = [2, 1, 0]
    # Guard against rows whose probabilities were never stored.
    usable = np.isfinite(probs).all(axis=1)
    if usable.any():
        probs, actual = probs[usable], actual[usable]
        probs = np.clip(probs, 1e-12, 1)
        probs = probs / probs.sum(axis=1, keepdims=True)
        one_hot = (actual[:, None] == np.array(labels)[None, :]).astype(float)
        summary["log_loss"] = float(
            -np.mean(np.log(np.maximum(
                probs[np.arange(len(actual)), [labels.index(a) for a in actual]],
                1e-12,
            )))
        )
        summary["brier_score"] = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        draw_mask = settled["predicted_outcome"].to_numpy()[usable] == "Draw"
        actual_draw = actual == 1
        if draw_mask.any():
            summary["draw_recall"] = float(actual_draw[draw_mask].mean())
    return summary


def performance_by(history: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Grouped settled performance: season, team, outcome or model version."""
    df = add_derived_columns(history)
    settled = df[df["actual_label"].astype(str) != ""].copy()
    if settled.empty:
        return pd.DataFrame(
            columns=[dimension, "predictions", "correct", "accuracy"]
        )

    if dimension == "season":
        grouped = settled.groupby("season", dropna=False)
    elif dimension == "outcome":
        grouped = settled.groupby("predicted_outcome", dropna=False)
    elif dimension == "model":
        grouped = settled.groupby(["model_label", "model_version"], dropna=False)
    elif dimension == "team":
        rows = []
        for side in ("home_team", "away_team"):
            part = settled.copy()
            part["team"] = part[side].map(canonical_team)
            rows.append(part)
        settled = pd.concat(rows, ignore_index=True)
        grouped = settled.groupby("team", dropna=False)
    elif dimension == "time":
        settled["period"] = settled["predicted_at_ts"].dt.to_period("M").astype(str)
        grouped = settled.groupby("period", dropna=False)
    else:
        raise ValueError(f"Unknown report dimension: {dimension}")

    report = grouped.apply(
        lambda g: pd.Series({
            "predictions": len(g),
            "correct": int(g["is_correct"].sum()),
            "accuracy": float(g["is_correct"].mean()),
        }),
        include_groups=False,
    ).reset_index()
    return report.sort_values("accuracy", ascending=False).reset_index(drop=True)


def performance_over_time(history: pd.DataFrame) -> pd.DataFrame:
    """Monthly accuracy and prediction volume."""
    return performance_by(history, "time")


def latest_predictions(history: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    df = add_derived_columns(history)
    if df.empty:
        return df
    return df.sort_values("predicted_at_ts", ascending=False).head(n)
