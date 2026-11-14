"""Fixture operations and prediction lifecycle.

Operational view of the Phase 8 lifecycle: calendar refresh verification,
upcoming slate with prediction status, schedule revisions, settled results and
live performance kept separate from offline evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard import components as ui
from dashboard import data_access as data
from src.predictions import history as hist
from src.utils.config import (
    FIXTURE_REFRESH_STATE_PATH,
    FIXTURE_SCHEDULE_REVISIONS_PATH,
    LEDGER_PATH,
    UPCOMING_FIXTURES_PATH,
)

ui.configure_page("Fixture Operations")
ui.hero(
    "Fixture Operations",
    "Scheduled calendar refresh, full-slate predictions, schedule revisions and "
    "verified result settlement — all append-only.",
    "Operations",
)

state = data.load_fixture_refresh_state(data._mtime(FIXTURE_REFRESH_STATE_PATH))
freshness = data.load_fixture_freshness(data._mtime(FIXTURE_REFRESH_STATE_PATH))
lifecycle = data.fixture_lifecycle_status(
    data._mtime(LEDGER_PATH), data._mtime(UPCOMING_FIXTURES_PATH)
)
live = data.load_live_performance()

status = str(freshness.get("status", "never_refreshed"))
verified = bool(freshness.get("verified"))
ui.metric_row([
    {
        "label": "Calendar status",
        "value": "verified" if verified else status,
        "help": "Only a verified, fresh calendar supports a coverage claim",
        "accent": "green" if verified else "red",
    },
    {
        "label": "Fixtures stored",
        "value": f"{int(state.get('stored', 0) or 0):,}",
        "help": f"Fetched {int(state.get('fetched', 0) or 0)} in the last refresh",
        "accent": "blue",
    },
    {
        "label": "Upcoming slate",
        "value": f"{0 if lifecycle.empty else len(lifecycle):,}",
        "help": "Eligible fixtures with a kickoff in the future",
        "accent": "gold",
    },
    {
        "label": "Schedule revisions",
        "value": f"{int(state.get('schedule_changes', 0) or 0):,}",
        "help": "Recorded in the append-only revision log",
        "accent": "blue",
    },
])

if not verified:
    st.warning(
        "The season calendar has not been refreshed and verified recently. "
        "Coverage completeness is unknown and must not be reported as complete."
    )

st.subheader("Calendar refresh")
if state:
    details = pd.DataFrame([
        {"field": key, "value": value}
        for key, value in {
            "status": state.get("status"),
            "verified": state.get("verified"),
            "provider": state.get("provider"),
            "source": state.get("source"),
            "fetched_at": state.get("fetched_at"),
            "verified_at": state.get("verified_at"),
            "fetched": state.get("fetched"),
            "stored": state.get("stored"),
            "kept": state.get("kept"),
            "schedule_changes": state.get("schedule_changes"),
            "calendar_age_hours": freshness.get("age_hours"),
            "stale": freshness.get("stale"),
        }.items()
    ])
    st.dataframe(details, use_container_width=True, hide_index=True)
    if state.get("errors"):
        st.error("; ".join(str(item) for item in state["errors"]))
else:
    ui.empty_state(
        "No calendar refresh has been recorded yet.",
        "python -m src.pipeline.automation --steps fixtures",
    )

st.subheader("Upcoming fixtures and prediction status")
if lifecycle.empty:
    ui.empty_state(
        "No eligible upcoming fixtures are stored.",
        "python -m src.data.fixture_provider",
    )
else:
    view = lifecycle.copy()
    view["probability home"] = view["probabilities"].map(
        lambda value: f"{value['home']:.1%}" if isinstance(value, dict) else "—"
    )
    columns = [
        column for column in (
            "match_date", "kickoff_utc", "home_team", "away_team",
            "fixture_status", "kickoff_provenance", "predicted",
            "predicted_outcome", "probability home", "model_version",
            "prediction_cycle", "schedule_changed",
        ) if column in view.columns
    ]
    st.dataframe(
        view[columns].sort_values(["match_date", "kickoff_utc"]),
        use_container_width=True,
        hide_index=True,
    )
    missing = int((~view["predicted"].astype(bool)).sum()) if "predicted" in view else 0
    if missing:
        st.info(
            f"{missing} eligible fixture(s) have no prediction yet. "
            "The next cycle generates exactly one prediction each."
        )

revisions = data.load_schedule_revisions(mtime=data._mtime(FIXTURE_SCHEDULE_REVISIONS_PATH))
st.subheader("Schedule revisions")
if revisions.empty:
    st.caption("No kickoff or date changes have been recorded.")
else:
    st.dataframe(revisions.iloc[::-1], use_container_width=True, hide_index=True)

history = data.load_history(data._mtime(LEDGER_PATH))
settled = hist.settled_only(history)
st.subheader("Settled results")
if settled.empty:
    st.caption("No verified results have settled a prediction yet.")
else:
    view = settled.sort_values("settled_at", ascending=False).head(200).copy()
    show = [
        column for column in (
            "match_date", "home_team", "away_team", "predicted_outcome",
            "actual_home_goals", "actual_away_goals", "correct", "model_version",
            "result_source", "result_status", "settled_at", "correction_count",
        ) if column in view.columns
    ]
    st.dataframe(view[show], use_container_width=True, hide_index=True)

corrections = data.load_result_corrections(
    mtime=data._mtime(Path(state.get("store_path") or "")) if state else 0.0
)
if not corrections.empty:
    st.subheader("Result corrections")
    st.caption("Official amendments are appended; the previous record is kept.")
    st.dataframe(corrections.iloc[::-1], use_container_width=True, hide_index=True)

st.subheader("Live performance")
if not live:
    st.caption(
        "No live performance report yet. Run "
        "`python -m src.monitoring.live_performance`."
    )
elif live.get("splits"):
    splits = live["splits"]
    ui.metric_row([
        {
            "label": "Live status",
            "value": str(live.get("status", "unknown")),
            "help": f"Minimum settled sample {live.get('minimum_settled_sample')}",
            "accent": "green" if live.get("status") == "ok" else "gold",
        },
        {
            "label": "Live settled",
            "value": f"{int(live.get('counts', {}).get('settled_rows', 0)):,}",
            "help": "Settled ledger rows",
            "accent": "blue",
        },
        {
            "label": "Live log-loss",
            "value": ui.num(splits.get("live", {}).get("log_loss")),
            "help": "Lower is better; insufficient sample is not a conclusion",
            "accent": "red",
        },
    ])
    rows = []
    for key, label in (
        ("validation", "Validation"),
        ("historical_test", "Historical test"),
        ("live", "Live"),
    ):
        block = splits.get(key, {})
        rows.append({
            "split": label,
            "status": block.get("status"),
            "n": block.get("n", 0),
            "log_loss": block.get("log_loss"),
            "brier_score": block.get("brier_score"),
            "accuracy": block.get("accuracy"),
            "macro_f1": block.get("macro_f1"),
            "ece": block.get("ece"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for key, title in (("by_season", "By season"), ("by_model_version", "By model version")):
        rows = live.get(key) or []
        if rows:
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.caption("The live performance report could not be read.")

ui.footer()
