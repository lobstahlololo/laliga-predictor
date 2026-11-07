"""Team analytics page."""

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
from src.utils.config import RAW_PATH

ui.configure_page("Team Analytics")
ui.hero(
    "Team Analytics",
    "Rolling form, goals, home/away splits and season-by-season performance for "
    "every La Liga club in the dataset.",
    "Club profiles",
)

raw = data.load_raw(data._mtime(RAW_PATH))
history = data.build_team_history(raw)
teams = data.available_teams(raw)
if not teams:
    ui.empty_state("Dataset is empty.", "python -m src.data.make_dataset")
    st.stop()

col_a, col_b = st.columns([1, 1])
with col_a:
    team = st.selectbox("Team", teams, index=0)
with col_b:
    compare_team = st.selectbox(
        "Compare with", ["—"] + [t for t in teams if t != team], index=0
    )

summary = data.team_summary(raw, history, team)
if not summary:
    st.warning("No history for this team.")
    st.stop()

ui.metric_row([
    {"label": "Matches", "value": f"{summary['matches']}", "help": "All competitions in dataset"},
    {"label": "Points per game", "value": ui.num(summary["ppg"], 2),
     "help": f"{summary['wins']}W · {summary['draws']}D · {summary['losses']}L", "accent": "gold"},
    {"label": "Goals for / match", "value": ui.num(summary["gf_per_match"], 2),
     "help": f"{summary['goals_for']} total", "accent": "green"},
    {"label": "Goals against / match", "value": ui.num(summary["ga_per_match"], 2),
     "help": f"{summary['goals_against']} total", "accent": "red"},
    {"label": "Recent form", "value": summary["form"], "help": "Oldest → newest", "accent": "blue"},
])

ui.section("Home vs away", "Points per game and goal output by venue.")
home_rows = history[(history["team"] == team) & (history["side"] == "home")]
away_rows = history[(history["team"] == team) & (history["side"] == "away")]
venue_table = pd.DataFrame({
    "Venue": ["Home", "Away"],
    "Matches": [len(home_rows), len(away_rows)],
    "Goals for": [int(home_rows["goals_scored"].sum()), int(away_rows["goals_scored"].sum())],
    "Goals against": [int(home_rows["goals_conceded"].sum()), int(away_rows["goals_conceded"].sum())],
    "GF / match": [
        round(float(home_rows["goals_scored"].mean()), 2) if len(home_rows) else None,
        round(float(away_rows["goals_scored"].mean()), 2) if len(away_rows) else None,
    ],
    "GA / match": [
        round(float(home_rows["goals_conceded"].mean()), 2) if len(home_rows) else None,
        round(float(away_rows["goals_conceded"].mean()), 2) if len(away_rows) else None,
    ],
})
st.dataframe(venue_table, width="stretch", hide_index=True)
st.caption(f"Home PPG {summary['home_ppg']:.2f} · Away PPG {summary['away_ppg']:.2f}")

ui.section("Rolling form", "5-match rolling points and goals (includes match result).")
rolling = data.team_rolling(history, team, window=5)
chart = rolling.set_index("date")[["rolling_points", "rolling_gf", "rolling_ga"]].rename(
    columns={
        "rolling_points": "Rolling points",
        "rolling_gf": "Rolling goals for",
        "rolling_ga": "Rolling goals against",
    }
)
st.line_chart(chart)

ui.section("Recent matches", "Newest first.")
recent = data.team_recent(history, team, n=10)
opponents = []
for match_id in recent["match_id"]:
    match = raw[raw["match_id"] == match_id]
    if match.empty:
        opponents.append("")
        continue
    row = match.iloc[0]
    opponents.append(
        row["away_team"] if row["home_team"] == team else row["home_team"]
    )
recent = recent.assign(opponent=opponents)
st.dataframe(
    recent[[
        "date", "side", "opponent", "goals_scored", "goals_conceded", "result",
        "total_shots", "shots_on_target",
    ]].rename(columns={
        "date": "Date", "side": "Venue", "opponent": "Opponent",
        "goals_scored": "GF", "goals_conceded": "GA", "result": "Result",
        "total_shots": "Shots", "shots_on_target": "On target",
    }),
    width="stretch", hide_index=True,
)

ui.section("Season history", "Aggregated by season label.")
season_table = data.team_season_table(history, team)
if not season_table.empty:
    display = season_table.copy()
    display["ppg"] = display["ppg"].map(lambda v: ui.num(v, 2))
    st.dataframe(display, width="stretch", hide_index=True)

if compare_team and compare_team != "—":
    ui.section(f"{team} vs {compare_team}", "Same rolling window, side by side.")
    other = data.team_summary(raw, history, compare_team)
    compare = pd.DataFrame({
        "Metric": ["Matches", "Wins", "Draws", "Losses", "Goals for", "Goals against",
                   "GF / match", "GA / match", "PPG", "Home PPG", "Away PPG", "Recent form"],
        team: [summary["matches"], summary["wins"], summary["draws"], summary["losses"],
               summary["goals_for"], summary["goals_against"],
               round(summary["gf_per_match"], 2), round(summary["ga_per_match"], 2),
               round(summary["ppg"], 2), round(summary["home_ppg"], 2),
               round(summary["away_ppg"], 2), summary["form"]],
        compare_team: [other["matches"], other["wins"], other["draws"], other["losses"],
                       other["goals_for"], other["goals_against"],
                       round(other["gf_per_match"], 2), round(other["ga_per_match"], 2),
                       round(other["ppg"], 2), round(other["home_ppg"], 2),
                       round(other["away_ppg"], 2), other["form"]],
    })
    st.dataframe(compare, width="stretch", hide_index=True)
    combined = pd.concat([
        rolling.assign(team=team),
        data.team_rolling(history, compare_team, window=5).assign(team=compare_team),
    ])
    pivot = combined.pivot_table(
        index="date", columns="team", values="rolling_points"
    )
    st.line_chart(pivot.rename(columns={team: f"{team} PPG (5m)",
                                        compare_team: f"{compare_team} PPG (5m)"}))

st.caption(
    "Form and rolling statistics are computed from matches in the dataset only. "
    "They describe historical performance, not a forecast of future results."
)
ui.footer()
