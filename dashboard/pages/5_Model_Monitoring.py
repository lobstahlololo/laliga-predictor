"""Model monitoring page."""

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
    DATA_QUALITY_PATH,
    EVAL_DIR,
    LEDGER_PATH,
    MODEL_METADATA_PATH,
    RAW_PATH,
    WALK_FORWARD_PATH,
)

ui.configure_page("Model Monitoring")
ui.hero(
    "Model Monitoring",
    "Staleness, calibration, prediction distribution and feature drift signals. "
    "This page describes risk; it does not automatically retrain anything.",
    "Operations",
)

raw = data.load_raw(data._mtime(RAW_PATH))
metadata = data.load_metadata(data._mtime(MODEL_METADATA_PATH))
history = data.load_history(data._mtime(LEDGER_PATH))
warnings = data.quality_warnings(raw)
drift = data.drift_table(raw)
diagnostics = data.monitoring_diagnostics(
    data._mtime(RAW_PATH), data._mtime(LEDGER_PATH), data._mtime(MODEL_METADATA_PATH)
)
alert_count = int((drift["status"] == "Alert").sum()) if not drift.empty else 0
watch_count = int((drift["status"] == "Watch").sum()) if not drift.empty else 0
error_count = sum(1 for severity, _ in warnings if severity == "error")
warn_count = sum(1 for severity, _ in warnings if severity == "warning")

ui.metric_row([
    {"label": "Data quality", "value": "Attention" if error_count or warn_count else "Healthy",
     "help": f"{error_count} errors · {warn_count} warnings",
     "accent": "red" if error_count else ("gold" if warn_count else "green")},
    {"label": "Drift alerts", "value": f"{alert_count}",
     "help": f"{watch_count} watch · 19 features", "accent": "red" if alert_count else "green"},
    {"label": "Diagnostic state", "value": diagnostics.get("overall_status", "not run"),
     "help": "Confirmed requires enough settled evidence",
     "accent": "red" if diagnostics.get("overall_status") == "confirmed" else ("gold" if diagnostics.get("overall_status") == "warning" else "green")},
    {"label": "Model version", "value": metadata.get("model_name", "—"),
     "help": metadata.get("model_version", "no model metadata"), "accent": "blue"},
    {"label": "Raw matches", "value": f"{len(raw):,}",
     "help": f"Latest {pd.to_datetime(raw['date']).max().date()}", "accent": "green"},
])

if alert_count or error_count:
    st.warning(
        "Monitoring signals suggest a refresh and retrain are worth reviewing. "
        "The model is not retrained automatically."
    )

tab_quality, tab_perf, tab_calibration, tab_distribution, tab_drift, tab_verdicts = st.tabs([
    "Data quality", "Performance over time", "Calibration changes",
    "Prediction distribution", "Feature drift", "Diagnostic verdicts",
])

with tab_quality:
    ui.section("Live checks", "Run against the current raw dataset.")
    for severity, message in warnings:
        st.markdown(
            f"{ui.severity_chip(severity)} &nbsp; {message}",
            unsafe_allow_html=True,
        )
    report = data.load_quality_report(data._mtime(DATA_QUALITY_PATH))
    if report:
        st.markdown("**Ingestion quality report**")
        rows = {
            "Rows": report.get("row_count"),
            "Rows removed in cleaning": report.get("rows_removed"),
            "Duplicate fixture IDs": report.get("duplicate_fixture_ids"),
            "Invalid dates": report.get("invalid_dates"),
            "Inconsistent score/result": report.get("inconsistent_results"),
            "Missing odds": report.get("missing_odds"),
            "Seasons fetched": ", ".join(report.get("seasons_fetched", [])),
            "Seasons failed": ", ".join(report.get("seasons_failed", [])) or "none",
            "Date range": f"{report.get('date_min')} → {report.get('date_max')}",
        }
        st.dataframe(
            pd.DataFrame({
                "Check": list(rows),
                "Value": [str(value) for value in rows.values()],
            }),
            width="stretch", hide_index=True,
        )
        for warning in report.get("warnings", []):
            st.caption(f"⚠️ {warning}")
    else:
        st.caption(
            "No data_quality.json yet. Run `python -m src.data.make_dataset`."
        )

    ui.section("Staleness and provenance")
    st.markdown(
        f"""
        <div class="ll-note">
        • Data source: Football-Data.co.uk SP1 (La Liga), Bet365 odds.<br>
        • Refresh command: <code>python -m src.data.make_dataset</code><br>
        • The pipeline keeps cached rows when a download fails; it never fills
          missing results with invented data.<br>
        • Test season {metadata.get('split', {}).get('test_season', '26/27')} is
          in progress, so test metrics are provisional.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_perf:
    ui.section(
        "Model performance over time",
        "Walk-forward folds measure seasonal stability; the final test season is separate.",
    )
    walk = data.load_walk_forward(data._mtime(WALK_FORWARD_PATH))
    if walk.empty:
        st.info("No walk-forward results. Train the models to populate this chart.")
    else:
        image = EVAL_DIR / "walk_forward_log_loss.png"
        if image.exists():
            st.image(str(image), width="stretch")
        display = walk.copy()
        for col in ("accuracy", "draw_recall"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.pct(v))
        for col in ("log_loss", "brier_score", "ece"):
            if col in display.columns:
                display[col] = display[col].map(lambda v: ui.num(v))
        keep = [c for c in (
            "test_season", "model", "accuracy", "log_loss", "brier_score",
            "draw_recall", "ece",
        ) if c in display.columns]
        st.dataframe(display[keep], width="stretch", hide_index=True)
    st.caption(
        "If log-loss rises in later seasons, review feature drift, squad changes "
        "and market-odds coverage before trusting the model."
    )

with tab_calibration:
    ui.section(
        "Calibration changes",
        "Expected calibration error (ECE) of settled ledger predictions by month.",
    )
    calibrated = data.calibration_by_period(history)
    if calibrated.empty:
        st.info(
            "Not enough settled predictions to measure calibration over time. "
            "The static test-season calibration curves are on Model Performance."
        )
    else:
        display = calibrated.copy()
        display["ece"] = display["ece"].map(lambda v: ui.num(v, 4))
        display["log_loss"] = display["log_loss"].map(lambda v: ui.num(v, 3))
        display["accuracy"] = display["accuracy"].map(lambda v: ui.pct(v))
        st.dataframe(display, width="stretch", hide_index=True)
        st.line_chart(calibrated.set_index("period")["ece"].rename("ECE"))
    st.caption(
        "ECE is a summary, not proof of calibration. Inspect the reliability "
        "diagrams before acting on small monthly samples."
    )

with tab_distribution:
    ui.section(
        "Prediction distribution",
        "A sudden shift in home/draw/away pick shares can signal drift or a data change.",
    )
    derived = data.history_derived(data._mtime(LEDGER_PATH))
    if derived.empty:
        st.info("No predictions logged yet.")
    else:
        counts = derived["predicted_outcome"].value_counts().rename_axis(
            "Predicted outcome"
        ).reset_index(name="Count")
        counts["Share"] = counts["Count"] / counts["Count"].sum()
        counts["Share"] = counts["Share"].map(lambda v: ui.pct(v))
        st.dataframe(counts, width="stretch", hide_index=True)
        image = EVAL_DIR / "prediction_distribution.png"
        if image.exists():
            st.image(str(image), width="stretch")
    st.caption(
        "The distribution is over logged predictions, which may include repeated "
        "snapshots of the same fixture at different model versions."
    )

with tab_drift:
    ui.section(
        "Feature drift",
        f"Recent {60} matches compared with the training reference (standardized mean shift).",
    )
    if drift.empty:
        st.info(
            "No feature reference found. Run `python -m src.features.build_features` "
            "after training to create data/processed/feature_reference.csv."
        )
    else:
        display = drift.copy()
        display["train_mean"] = display["train_mean"].map(lambda v: ui.num(v, 3))
        display["recent_mean"] = display["recent_mean"].map(lambda v: ui.num(v, 3))
        display["shift_sd"] = display["shift_sd"].map(lambda v: ui.num(v, 2))
        st.dataframe(
            display[["feature", "train_mean", "recent_mean", "shift_sd", "status"]],
            width="stretch", hide_index=True,
        )
        st.caption(
            "Watch ≥ 1 SD, Alert > 2 SD. This is a descriptive drift indicator, "
            "not a statistical test; correlated football features can move together."
        )

with tab_verdicts:
    ui.section(
        "Diagnostic verdicts",
        "Insufficient data is reported separately from a warning or confirmed degradation.",
    )
    checks = diagnostics.get("checks", [])
    if not checks:
        st.info("No monitoring report yet. Run `python main.py experiments`.")
    else:
        for check in checks:
            status = check.get("status", "insufficient_data")
            chip = {
                "confirmed": ui.severity_chip("error"),
                "warning": ui.severity_chip("warning"),
                "ok": ui.severity_chip("ok"),
            }.get(status, ui.severity_chip("info"))
            st.markdown(
                f"{chip} &nbsp; **{check.get('name', '')}** — {check.get('message', '')}",
                unsafe_allow_html=True,
            )
        st.caption(diagnostics.get("note", ""))
        with st.expander("Raw monitoring JSON"):
            st.json(diagnostics)

ui.footer()
