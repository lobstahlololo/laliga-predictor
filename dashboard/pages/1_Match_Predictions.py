"""Match predictions page."""

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
from src.predictions.predictor import load_artifacts, load_raw as load_raw_predictor, predict_fixtures
from src.utils.config import MODEL_METADATA_PATH, RAW_PATH

ui.configure_page("Match Predictions")

if not RAW_PATH.exists():
    ui.hero(
        "Match Predictions",
        "Three-way probabilities for upcoming fixtures, plus recent form and "
        "rest context.",
        "Forecast board",
    )
    ui.empty_state(
        "No dataset found. Build the data and model artifacts first.",
        "python main.py data\npython main.py features\npython main.py train",
    )
    ui.footer()
    st.stop()

raw = data.load_raw(data._mtime(RAW_PATH))
history = data.build_team_history(raw)
teams = data.available_teams(raw)

ui.hero(
    "Match Predictions",
    "Three-way probabilities for upcoming fixtures, plus recent form and rest "
    "context. Probabilities always sum to 100%.",
    "Forecast board",
)

predictions, errors, source = data.get_upcoming_predictions(
    data._mtime(RAW_PATH), data._mtime(ROOT / "models" / "saved_models" / "best_model.pkl")
)

tab_upcoming, tab_custom = st.tabs(["Upcoming fixtures", "Custom matchup"])

with tab_upcoming:
    if predictions:
        st.caption(f"Fixture source: {source}")
        fixture_teams = sorted({
            team for prediction in predictions
            for team in (prediction["home_team"], prediction["away_team"])
        })
        filter_team = st.selectbox(
            "Filter fixtures by team", ["All teams"] + fixture_teams,
            key="prediction_team_filter",
        )
        shown = [
            prediction for prediction in predictions
            if filter_team == "All teams"
            or filter_team in (prediction["home_team"], prediction["away_team"])
        ]
        st.caption(
            "Expected goals and scorelines come from a time-decayed Poisson model "
            "refit on matches before each kickoff; they are supporting estimates, "
            "not a second forecast."
        )
        if not shown:
            st.info("No upcoming fixture involves the selected team.")
        for prediction in shown:
            ui.fixture_card(
                prediction,
                data.form_string(history, prediction["home_team"]),
                data.form_string(history, prediction["away_team"]),
            )
            with st.expander(
                f"Form detail · {prediction['home_team']} vs {prediction['away_team']}"
            ):
                home_recent = data.team_recent(history, prediction["home_team"])
                away_recent = data.team_recent(history, prediction["away_team"])
                col1, col2 = st.columns(2)
                for column, team, recent in (
                    (col1, prediction["home_team"], home_recent),
                    (col2, prediction["away_team"], away_recent),
                ):
                    with column:
                        st.markdown(f"**{team}**")
                        st.caption(
                            f"Rest: {prediction['home_rest_days' if team == prediction['home_team'] else 'away_rest_days']} days"
                        )
                        st.dataframe(
                            recent[["date", "side", "goals_scored", "goals_conceded", "result"]]
                            .rename(columns={
                                "date": "Date", "side": "Venue",
                                "goals_scored": "GF", "goals_conceded": "GA",
                                "result": "Result",
                            }),
                            width="stretch", hide_index=True,
                        )
                st.markdown("**Model inputs**")
                st.json({
                    "home_form": prediction["home_form"],
                    "away_form": prediction["away_form"],
                    "market": prediction["market"],
                    "short_rest_advantage": prediction["short_rest_advantage"],
                })
    else:
        ui.empty_state(
            "No upcoming fixtures could be predicted. Add fixtures (no dates are "
            "invented) and ensure model artifacts exist.",
            "Real Madrid vs Valencia\nBarcelona vs Sevilla",
        )
    for error in errors:
        st.warning(f"{error['fixture']}: {error['error']}")

with tab_custom:
    st.caption(
        "Build any matchup. Select a date to avoid using matches played after "
        "kickoff; otherwise today's date is used."
    )
    with st.form("custom_fixture"):
        col1, col2 = st.columns(2)
        with col1:
            home = st.selectbox("Home team", teams, index=0)
        with col2:
            away = st.selectbox("Away team", teams, index=min(1, len(teams) - 1))
        date_value = st.date_input("Match date (optional)", value=None)
        use_odds = st.checkbox("Supply decimal odds")
        odds = (None, None, None)
        if use_odds:
            cols = st.columns(3)
            odds = (
                cols[0].number_input("Home odds", min_value=1.01, value=2.20, step=0.01),
                cols[1].number_input("Draw odds", min_value=1.01, value=3.40, step=0.01),
                cols[2].number_input("Away odds", min_value=1.01, value=3.80, step=0.01),
            )
        submitted = st.form_submit_button("Predict matchup", type="primary")

    if submitted:
        if home == away:
            st.error("Home and away teams must differ.")
        else:
            try:
                model, preprocessor, metadata = load_artifacts()
                fixture = {
                    "home_team": home,
                    "away_team": away,
                    "match_date": date_value.isoformat() if date_value else "",
                    "odds": odds if use_odds else None,
                }
                custom, custom_errors = predict_fixtures(
                    [fixture], model=model, preprocessor=preprocessor,
                    metadata=metadata, raw=load_raw_predictor(),
                )
                if custom:
                    ui.fixture_card(
                        custom[0],
                        data.form_string(history, custom[0]["home_team"]),
                        data.form_string(history, custom[0]["away_team"]),
                    )
                for error in custom_errors:
                    st.error(error["error"])
            except Exception as exc:  # noqa: BLE001 - show actionable error
                st.error(f"Prediction failed: {exc}")

st.caption(
    "Reminder: these are probabilistic estimates. Value-bet flags compare model "
    "probabilities with supplied or historical average market probabilities; they "
    "are not betting advice."
)
ui.footer()
