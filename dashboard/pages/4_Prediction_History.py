"""Prediction history page."""

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
from src.utils.config import LEDGER_PATH

ui.configure_page("Prediction History")
ui.hero(
    "Prediction History",
    "Append-only, versioned predictions with reconciled actual results. "
    "Settled rows are never overwritten.",
    "Audit trail",
)

history = data.load_history(data._mtime(LEDGER_PATH))
if history.empty:
    ui.empty_state(
        "No prediction history yet. Generate predictions first.",
        "python predict_upcoming.py",
    )
    ui.footer()
    st.stop()

df = hist.add_derived_columns(history)
summary = hist.performance_summary(history)

ui.metric_row([
    {"label": "Total predictions", "value": f"{summary['predictions']:,}",
     "help": "Every snapshot is retained", "accent": "green"},
    {"label": "Settled", "value": f"{summary['settled']:,}",
     "help": f"{summary['pending']} awaiting result", "accent": "blue"},
    {"label": "Ledger accuracy", "value": ui.pct(summary["accuracy"]),
     "help": "Settled rows only", "accent": "gold"},
    {"label": "Log-loss", "value": ui.num(summary["log_loss"]),
     "help": "From stored probabilities", "accent": "green"},
    {"label": "Brier score", "value": ui.num(summary["brier_score"]),
     "help": "Multiclass, lower is better", "accent": "red"},
])

with st.sidebar:
    st.divider()
    st.markdown("**Filters**")
    versions = sorted(df["model_version"].astype(str).unique())
    selected_versions = st.multiselect("Model version", versions, default=versions)
    seasons = sorted([s for s in df["season"].astype(str).unique() if s])
    selected_seasons = st.multiselect("Season", seasons, default=seasons)
    outcomes = ["Home Win", "Draw", "Away Win"]
    selected_outcomes = st.multiselect("Predicted outcome", outcomes, default=outcomes)
    status = st.selectbox("Result status", ["All", "Settled", "Pending"])

filtered = df[df["model_version"].astype(str).isin(selected_versions)]
filtered = filtered[filtered["season"].astype(str).isin(selected_seasons)]
filtered = filtered[filtered["predicted_outcome"].isin(selected_outcomes)]
if status == "Settled":
    filtered = filtered[filtered["actual_label"].astype(str) != ""]
elif status == "Pending":
    filtered = filtered[filtered["actual_label"].astype(str) == ""]

ui.section(
    "Prediction records",
    f"{len(filtered)} of {len(df)} rows shown.",
)
if filtered.empty:
    st.info("No rows match the current filters.")
else:
    display = filtered.sort_values("predicted_at_ts", ascending=False).copy()
    display["Probabilities"] = display.apply(
        lambda r: (
            f"{100*r['model_prob_home']:.0f}% / "
            f"{100*r['model_prob_draw']:.0f}% / "
            f"{100*r['model_prob_away']:.0f}%"
        ),
        axis=1,
    )
    display["Actual"] = display["actual_label"].where(
        display["actual_label"].astype(str) != "", "Pending"
    )
    display["✓"] = display["is_correct"].map({1.0: "✓", 0.0: "✗"}).fillna("—")
    view = display[[
        "predicted_at", "match_date", "home_team", "away_team",
        "Probabilities", "predicted_outcome", "Actual", "✓",
        "model_name", "model_version",
    ]].rename(columns={
        "predicted_at": "Predicted at", "match_date": "Match date",
        "home_team": "Home", "away_team": "Away",
        "predicted_outcome": "Pick", "model_name": "Model",
        "model_version": "Version",
    })
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button(
        "Download filtered history (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="laliga_prediction_history.csv",
        mime="text/csv",
    )

ui.section("Performance breakdown", "Settled predictions only.")
tab_time, tab_season, tab_outcome, tab_team, tab_model = st.tabs([
    "Over time", "By season", "By outcome", "By team", "By model version",
])

with tab_time:
    trend = hist.performance_over_time(history)
    if trend.empty:
        st.info("No settled predictions yet.")
    else:
        st.dataframe(trend, width="stretch", hide_index=True)
        chart = trend.set_index("period")["accuracy"]
        st.line_chart(chart.rename("Monthly accuracy"))
    st.caption(
        "Accuracy over time is based on the predictions actually logged. It is "
        "not a substitute for the walk-forward model evaluation."
    )

with tab_season:
    table = hist.performance_by(history, "season")
    if table.empty:
        st.info("No settled predictions yet.")
    else:
        table["accuracy"] = table["accuracy"].map(lambda v: ui.pct(v))
        st.dataframe(table, width="stretch", hide_index=True)

with tab_outcome:
    table = hist.performance_by(history, "outcome")
    if table.empty:
        st.info("No settled predictions yet.")
    else:
        table["accuracy"] = table["accuracy"].map(lambda v: ui.pct(v))
        st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Reading by predicted outcome shows where the model is overconfident. "
        "Low draw accuracy is common in three-way football forecasting."
    )

with tab_team:
    table = hist.performance_by(history, "team")
    if table.empty:
        st.info("No settled predictions yet.")
    else:
        table = table.sort_values("predictions", ascending=False)
        table["accuracy"] = table["accuracy"].map(lambda v: ui.pct(v))
        st.dataframe(table, width="stretch", hide_index=True)

with tab_model:
    table = hist.performance_by(history, "model")
    if table.empty:
        st.info("No settled predictions yet.")
    else:
        table["accuracy"] = table["accuracy"].map(lambda v: ui.pct(v))
        st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "Model version changes are visible here. Legacy rows without version "
        "metadata are grouped as legacy/unknown."
    )

ui.section("Ledger schema")
st.code(", ".join(hist.CANONICAL_COLUMNS), language="text")
st.caption(
    "Reconciliation fills only blank actual-result fields. Re-running prediction "
    "at a new timestamp appends a new snapshot; it never rewrites an older one."
)
ui.footer()
