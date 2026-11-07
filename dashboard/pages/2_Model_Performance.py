"""Model performance page."""

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
from src.utils.config import (
    EVAL_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    SEASON_METRICS_PATH,
    TEST_SEASON,
    VALIDATION_SEASON,
    WALK_FORWARD_PATH,
)

ui.configure_page("Model Performance")
ui.hero(
    "Model Performance",
    "Random Forest vs XGBoost on identical chronological windows. Selection is "
    "based on validation log-loss, then a single untouched test score.",
    "Evaluation",
)

metadata = data.load_metadata(data._mtime(MODEL_METADATA_PATH))
if not metadata:
    ui.empty_state(
        "No model metadata found. Train the models first.",
        "python -m src.models.train_model",
    )
    ui.footer()
    st.stop()

test = metadata.get("test_metrics", {})
validation = metadata.get("validation_metrics", {})
split = metadata.get("split", {})

ui.metric_row([
    {"label": "Selected model", "value": metadata.get("model_name", "—"),
     "help": metadata.get("model_version", ""), "accent": "green"},
    {"label": "Validation log-loss", "value": ui.num(validation.get("log_loss")),
     "help": f"{VALIDATION_SEASON} · selection metric", "accent": "blue"},
    {"label": "Test log-loss", "value": ui.num(test.get("log_loss")),
     "help": f"{TEST_SEASON} · scored once", "accent": "gold"},
    {"label": "Test accuracy", "value": ui.pct(test.get("accuracy")),
     "help": "Not used for model selection", "accent": "green"},
    {"label": "Draw recall", "value": ui.pct(test.get("draw_recall")),
     "help": "Hardest outcome class", "accent": "red"},
])

st.caption(
    f"Train {', '.join(split.get('train_seasons', []))} "
    f"({split.get('train_rows', '—')} rows) · "
    f"Validation {split.get('validation_season', VALIDATION_SEASON)} "
    f"({split.get('validation_rows', '—')} rows) · "
    f"Test {split.get('test_season', TEST_SEASON)} ({split.get('test_rows', '—')} rows)"
)

tab_overview, tab_diagnostics, tab_season, tab_walk, tab_importance = st.tabs([
    "Overview", "Confusion & calibration", "By season", "Walk-forward", "Feature importance",
])

with tab_overview:
    ui.section(
        "Model comparison",
        "Both models use the same preprocessing, splits and evaluation code.",
    )
    rows = data.model_rows(metadata)
    if not rows.empty:
        display = rows.copy()
        for col in ("accuracy", "draw_recall", "draw_precision", "ece"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.pct(v))
        for col in ("log_loss", "brier_score", "macro_f1"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.num(v))
        st.dataframe(display, width="stretch", hide_index=True)

    comparison = data.load_comparison(data._mtime(MODEL_COMPARISON_PATH))
    if comparison.empty:
        st.info("No test comparison CSV found.")
    else:
        st.markdown("**Final test-season comparison**")
        show = comparison.copy()
        for col in ("accuracy", "draw_recall", "draw_precision", "ece"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: ui.pct(v))
        for col in ("log_loss", "brier_score", "macro_f1"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: ui.num(v))
        st.dataframe(show, width="stretch", hide_index=True)

    image = EVAL_DIR / "model_comparison.png"
    if image.exists():
        st.image(str(image), width="stretch")

    st.markdown(
        """
        > **Why log-loss?** Three-way football probabilities must be honest about
        > uncertainty. Accuracy rewards a model that guesses the majority outcome;
        > log-loss and Brier score punish overconfident wrong probabilities and
        > reward calibrated draws. Model selection therefore uses validation
        > log-loss, and the test season is never consulted until after selection.
        """
    )

with tab_diagnostics:
    ui.section("Confusion matrices", "Counts and row-normalised percentages.")
    cols = st.columns(2)
    for column, slug in zip(cols, ("random_forest", "xgboost")):
        path = EVAL_DIR / f"confusion_matrix_{slug}_test.png"
        if path.exists():
            with column:
                st.image(str(path), width="stretch")
        else:
            with column:
                st.caption(f"No confusion matrix for {slug}.")
    ui.section("Calibration curves", "Diagonal = perfect calibration.")
    cols = st.columns(2)
    for column, slug in zip(cols, ("random_forest", "xgboost")):
        path = EVAL_DIR / f"calibration_{slug}_test.png"
        if path.exists():
            with column:
                st.image(str(path), width="stretch")
        else:
            with column:
                st.caption(f"No calibration curve for {slug}.")
    ui.section("Per-class metrics", "Precision, recall and F1 for each outcome.")
    cols = st.columns(2)
    for column, slug in zip(cols, ("random_forest", "xgboost")):
        path = EVAL_DIR / f"per_class_metrics_{slug}_test.png"
        if path.exists():
            with column:
                st.image(str(path), width="stretch")

    st.caption(
        "Draw recall is reported separately because draws are the minority class "
        "and are easy for a model to ignore. Calibration uses one-vs-rest bins."
    )

with tab_season:
    ui.section(
        "Performance by season",
        "Walk-forward folds plus the final test season, same models and features.",
    )
    seasons = data.load_season_metrics(data._mtime(SEASON_METRICS_PATH))
    if seasons.empty:
        st.info("No season metrics available.")
    else:
        display = seasons.copy()
        for col in ("accuracy", "draw_recall"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.pct(v))
        for col in ("log_loss", "brier_score", "macro_f1", "ece"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.num(v))
        st.dataframe(display, width="stretch", hide_index=True)
        image = EVAL_DIR / "season_performance.png"
        if image.exists():
            st.image(str(image), width="stretch")
    st.caption(
        "Season-level scores vary with squad turnover, mid-season managerial "
        "changes and the in-progress test season. A single strong season is not "
        "evidence of a stable edge."
    )

with tab_walk:
    ui.section(
        "Walk-forward backtest",
        "Fit on all earlier seasons, predict the next season, roll forward.",
    )
    walk = data.load_walk_forward(data._mtime(WALK_FORWARD_PATH))
    if walk.empty:
        st.info("No walk-forward results available.")
    else:
        display = walk.copy()
        for col in ("accuracy", "draw_recall"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.pct(v))
        for col in ("log_loss", "brier_score", "ece"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.num(v))
        keep = [c for c in (
            "fold", "test_season", "train_seasons", "model", "n_samples",
            "accuracy", "log_loss", "brier_score", "draw_recall", "ece",
        ) if c in display.columns]
        st.dataframe(display[keep], width="stretch", hide_index=True)
        image = EVAL_DIR / "walk_forward_log_loss.png"
        if image.exists():
            st.image(str(image), width="stretch")
    st.caption(
        "The walk-forward folds end before the final test season, preserving the "
        "test set as a single untouched holdout."
    )

with tab_importance:
    ui.section(
        "Feature importance",
        "Impurity/gain-based importance from the final fitted tree models.",
    )
    path = EVAL_DIR / "feature_importance.csv"
    if path.exists():
        importance = pd.read_csv(path)
        pivot = importance.pivot(index="feature", columns="model", values="importance")
        st.dataframe(pivot.sort_values(pivot.columns[0], ascending=False),
                     width="stretch")
        st.caption(
            "Importance is descriptive, not causal. Correlated features share "
            "credit and tree importance can mask substitutable signals."
        )
    else:
        st.info("No feature importance file found.")

ui.footer()
