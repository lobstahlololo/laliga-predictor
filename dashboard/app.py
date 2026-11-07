"""La Liga Predictor — dashboard home page."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from dashboard import components as ui
from dashboard import data_access as data
from src.utils.config import (
    EVAL_DIR,
    LEDGER_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    RAW_PATH,
    REGISTRY_PATH,
    TEST_SEASON,
    VALIDATION_SEASON,
)

ui.configure_page("Home")

with st.sidebar:
    st.markdown("## ⚽ La Liga Predictor")
    st.caption("Probabilistic match analytics")
    st.divider()
    st.markdown("**Pipeline status**")
    status = data.artifact_status()
    st.write(f"{'✅' if status['raw'] else '❌'} Raw dataset")
    st.write(f"{'✅' if status['model'] else '❌'} Trained model")
    st.write(f"{'✅' if status['evaluation'] else '❌'} Evaluation artifacts")
    st.write(f"{'✅' if status['ledger'] else '❌'} Prediction ledger")
    registry = data.load_registry_state(data._mtime(REGISTRY_PATH))
    st.write(f"{'✅' if registry.get('production') else '⚪'} Model registry")
    if registry.get("production_version"):
        st.caption(f"Production: {registry['production_version']}")
    st.divider()
    st.caption(
        "Use the pages above to inspect predictions, model metrics, team form, "
        "history and monitoring signals."
    )

ui.hero(
    "La Liga Predictor",
    "Leakage-safe match outcome probabilities, walk-forward evaluation and "
    "transparent prediction history for Spanish top-flight football.",
    "Football analytics platform",
)

if not RAW_PATH.exists():
    ui.empty_state(
        "No dataset found. Build the data and model artifacts first.",
        "python -m src.data.make_dataset\npython -m src.features.build_features\npython -m src.models.train_model",
    )
    ui.footer()
    st.stop()

raw = data.load_raw(data._mtime(RAW_PATH))
metadata = data.load_metadata(data._mtime(MODEL_METADATA_PATH))
history = data.load_history(data._mtime(LEDGER_PATH))
summary = data.history_summary(data._mtime(LEDGER_PATH))
predictions, errors, source = data.get_upcoming_predictions(
    data._mtime(RAW_PATH), data._mtime(ROOT / "models" / "saved_models" / "best_model.pkl")
)

test = metadata.get("test_metrics", {})
validation = metadata.get("validation_metrics", {})
model_name = metadata.get("model_name", "No trained model")
model_version = metadata.get("model_version", "—")

ui.metric_row([
    {
        "label": "Selected model",
        "value": model_name,
        "help": model_version,
        "accent": "green",
    },
    {
        "label": "Test log-loss",
        "value": ui.num(test.get("log_loss"), 3),
        "help": f"Untouched {TEST_SEASON} · lower is better",
        "accent": "blue",
    },
    {
        "label": "Test accuracy",
        "value": ui.pct(test.get("accuracy")),
        "help": f"Validation accuracy {ui.pct(validation.get('accuracy'))}",
        "accent": "gold",
    },
    {
        "label": "Test Brier score",
        "value": ui.num(test.get("brier_score"), 3),
        "help": f"ECE {ui.num(test.get('ece'), 3)} · calibration gap",
        "accent": "green",
    },
])

left, right = st.columns([1.55, 1])

with left:
    ui.section(
        "Upcoming fixtures",
        "Probabilities are generated from the latest model and pre-kickoff features.",
    )
    if predictions:
        rows = []
        for prediction in predictions:
            probs = prediction["probs"]
            rows.append({
                "Home": prediction["home_team"],
                "Away": prediction["away_team"],
                "Date": prediction["match_date"] or "TBD",
                "Home %": round(100 * probs["home"], 1),
                "Draw %": round(100 * probs["draw"], 1),
                "Away %": round(100 * probs["away"], 1),
                "Pick": prediction["predicted_outcome"],
            })
        ui.dataframe(pd.DataFrame(rows))
        ui.fixture_card(
            predictions[0],
            data.form_string(data.build_team_history(raw), predictions[0]["home_team"]),
            data.form_string(data.build_team_history(raw), predictions[0]["away_team"]),
        )
    else:
        st.info(
            "No parseable upcoming fixtures. Add fixtures to `fixtures.txt` or "
            "`data/raw/upcoming_fixtures.csv` (no dates are invented)."
        )
        for error in errors:
            st.caption(f"⚠️ {error['error']}")

with right:
    ui.section("Prediction ledger")
    ui.metric_row([
        {
            "label": "Predictions",
            "value": f"{summary['predictions']:,}",
            "help": f"{summary['settled']} settled · {summary['pending']} pending",
            "accent": "green",
        },
        {
            "label": "Ledger accuracy",
            "value": ui.pct(summary["accuracy"]),
            "help": "Settled predictions only",
            "accent": "blue",
        },
    ])
    recent = data.history_derived(data._mtime(LEDGER_PATH))
    if not recent.empty:
        display = recent.sort_values("predicted_at_ts", ascending=False).head(8).copy()
        display["Status"] = display["actual_label"].where(
            display["actual_label"].astype(str) != "", "Pending"
        )
        display["Correct"] = display["is_correct"].map(
            {1.0: "✓", 0.0: "✗"}
        ).fillna("—")
        ui.dataframe(display[[
            "predicted_at", "home_team", "away_team", "predicted_outcome",
            "Status", "Correct",
        ]].rename(columns={
            "predicted_at": "Predicted", "home_team": "Home", "away_team": "Away",
            "predicted_outcome": "Pick",
        }))
    else:
        st.caption("No prediction history yet. Run `python predict_upcoming.py`.")

ui.section(
    "Latest model performance",
    "Selection uses validation log-loss only; the test season is scored once.",
)
if MODEL_COMPARISON_PATH.exists():
    image = EVAL_DIR / "model_comparison.png"
    if image.exists():
        st.image(str(image), width="stretch")
    comparison = data.load_comparison(data._mtime(MODEL_COMPARISON_PATH))
    if not comparison.empty:
        show = comparison.copy()
        for col in ("accuracy", "draw_recall", "draw_precision", "ece"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: ui.pct(v))
        ui.dataframe(show)
else:
    st.info("No evaluation artifacts yet. Train a model to populate this section.")

with st.expander("How the prediction system works"):
    st.markdown(
        f"""
        1. **Data** — real La Liga results, shots, shots on target and Bet365 odds
           are refreshed from Football-Data.co.uk and validated for duplicates and
           score/result consistency.
        2. **19 pre-kickoff features** — 5-match and 3-match rolling form, goal
           differentials, market-implied probabilities, rest advantage and home
           advantage. Rolling windows use `shift(1)`, so a match never sees itself
           or future matches.
        3. **Chronological evaluation** — train on {', '.join(metadata.get('split', {}).get('train_seasons', ['23/24', '24/25']))},
           validate on {metadata.get('split', {}).get('validation_season', VALIDATION_SEASON)},
           test once on {metadata.get('split', {}).get('test_season', TEST_SEASON)}.
           Hyperparameter search uses `TimeSeriesSplit`, not shuffled CV.
        4. **Model selection** — lowest validation multiclass log-loss (Brier breaks
           ties). Accuracy alone is never the selection criterion.
        5. **History** — every prediction is appended with fixture ID, timestamp,
           model name/version and probabilities; results are reconciled without
           overwriting prior records.
        """
    )

ui.footer()
