"""Model registry, pipeline runs and monitoring status."""

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
from src.features.preprocessing import DEFAULT_PREPROCESSOR_PATH
from src.utils.config import (
    BEST_MODEL_PATH,
    LEDGER_PATH,
    MODEL_METADATA_PATH,
    MONITORING_REPORT_PATH,
    PIPELINE_RUNS_PATH,
    RAW_PATH,
    REGISTRY_PATH,
)

ui.configure_page("Model Operations")

with st.sidebar:
    st.markdown("## ⚽ La Liga Predictor")
    st.caption("Production operations")
    st.divider()
    status = data.artifact_status()
    st.write(f"{'✅' if status['raw'] else '❌'} Raw dataset")
    st.write(f"{'✅' if status['model'] else '❌'} Production artifact")
    st.write(f"{'✅' if REGISTRY_PATH.exists() else '⚪'} Model registry")
    st.write(f"{'✅' if LEDGER_PATH.exists() else '⚪'} Prediction ledger")
    st.caption(
        "Promotion and rollback are CLI actions so they stay auditable: "
        "`python -m src.models.registry`."
    )

ui.hero(
    "Model Operations",
    "Registered model versions, validation evidence, update cycles and "
    "monitoring verdicts. No experiment is promoted automatically.",
    "Production readiness",
)

registry_state = data.load_registry_state(data._mtime(REGISTRY_PATH))
versions = data.load_registry_versions(data._mtime(REGISTRY_PATH))
production = registry_state.get("production") or {}
monitoring = data.load_monitoring_report(data._mtime(MONITORING_REPORT_PATH))
runs = data.load_pipeline_runs(20, data._mtime(PIPELINE_RUNS_PATH))

validation = production.get("validation_metrics", {}) or {}
test = production.get("test_metrics", {}) or {}

ui.metric_row([
    {
        "label": "Production version",
        "value": production.get("version", "not registered"),
        "help": production.get("model_family", "Run `python -m src.models.registry register`"),
        "accent": "green",
    },
    {
        "label": "Validation log-loss",
        "value": ui.num(validation.get("log_loss"), 4),
        "help": "Evidence used for promotion",
        "accent": "blue",
    },
    {
        "label": "Test log-loss",
        "value": ui.num(test.get("log_loss"), 4),
        "help": "Confirmatory only, never used to promote",
        "accent": "gold",
    },
    {
        "label": "Monitoring",
        "value": monitoring.get("overall_status", "no report"),
        "help": (monitoring.get("generated_at") or "Run `python main.py update`"),
        "accent": "red" if monitoring.get("overall_status") == "confirmed" else "green",
    },
])

ui.section(
    "Production model",
    "Exactly one version is designated production. Promotion requires measured "
    "validation evidence; the test season is never a promotion signal.",
)
if production:
    st.markdown(
        f"**{production.get('version')}** · {production.get('model_family')} · "
        f"since {registry_state.get('production_since', 'unknown')}"
    )
    detail = pd.DataFrame([
        {"field": "Features", "value": str(production.get("feature_count"))},
        {"field": "Feature-set hash", "value": str(production.get("feature_set_hash"))},
        {"field": "Artifact", "value": str(production.get("artifact"))},
        {"field": "Artifact sha256", "value": str(production.get("artifact_sha256") or "")[:16] + "…"},
        {"field": "Training window", "value": str((production.get("training_period") or {}).get("seasons"))},
        {"field": "Validation window", "value": str((production.get("validation_period") or {}).get("seasons"))},
        {"field": "Test window", "value": str((production.get("test_period") or {}).get("seasons"))},
    ])
    ui.dataframe(detail)
    st.caption(
        "Promotion rule: challenger validation log-loss strictly lower AND "
        "validation Brier not worse. Missing metrics block promotion."
    )
else:
    ui.empty_state(
        "No production version is registered yet.",
        "python -m src.models.registry register\npython main.py registry",
    )

ui.section("Version history", "Every registered snapshot is retained for rollback.")
if not versions.empty:
    show = versions.copy()
    for column in ("validation_log_loss", "test_log_loss", "validation_brier", "test_brier"):
        if column in show.columns:
            show[column] = pd.to_numeric(show[column], errors="coerce").round(4)
    ui.dataframe(show.rename(columns={
        "version": "Version", "family": "Family", "status": "Status",
        "registered_at": "Registered", "validation_log_loss": "Val log-loss",
        "validation_brier": "Val Brier", "test_log_loss": "Test log-loss",
        "test_brier": "Test Brier",
    }))
else:
    st.info("The registry is empty. Register the current training artifacts first.")

ui.section(
    "Promotion evidence",
    "Candidates are compared with the production model on the same validation window.",
)
if not versions.empty and production:
    prod_loss = validation.get("log_loss")
    prod_brier = validation.get("brier_score")
    candidates = versions[versions["status"] == "candidate"].copy()
    if candidates.empty:
        st.caption("No candidate versions awaiting review.")
    else:
        def verdict(row) -> str:
            loss, brier = row.get("validation_log_loss"), row.get("validation_brier")
            if pd.isna(loss) or pd.isna(brier) or prod_loss is None or prod_brier is None:
                return "insufficient evidence"
            if loss < prod_loss and brier <= prod_brier:
                return "eligible"
            return "not eligible"

        candidates["verdict"] = candidates.apply(verdict, axis=1)
        ui.dataframe(candidates[[
            "version", "family", "validation_log_loss", "validation_brier", "verdict",
        ]].rename(columns={
            "version": "Version", "family": "Family",
            "validation_log_loss": "Val log-loss", "validation_brier": "Val Brier",
            "verdict": "Promotion verdict",
        }))
        st.code(
            "python -m src.models.registry promote <version> --reason \"validation evidence\"",
            language="bash",
        )

ui.section("Update cycles", "Each cycle is idempotent and recorded as one JSON line.")
if runs:
    rows = []
    for run in reversed(runs):
        steps = run.get("steps", [])
        rows.append({
            "Run": run.get("run_id"),
            "Finished": run.get("finished_at"),
            "Status": run.get("status"),
            "Steps": ", ".join(
                f"{step['name']}:{step['status']}" for step in steps
            ),
            "Notified": run.get("notified", False),
        })
    ui.dataframe(pd.DataFrame(rows))
    latest = runs[-1]
    with st.expander("Latest run detail"):
        st.json(latest)
else:
    ui.empty_state(
        "No update cycle has run yet. Manual runs are safe and idempotent.",
        "python main.py update",
    )

ui.section(
    "Monitoring verdicts",
    "Insufficient data is reported honestly; a warning is not proof of degradation.",
)
checks = monitoring.get("checks", [])
if checks:
    ui.dataframe(pd.DataFrame([
        {"Check": check.get("name"), "Status": check.get("status"),
         "Message": check.get("message")}
        for check in checks
    ]))
    st.caption(monitoring.get("note", ""))
else:
    st.info("No monitoring report yet. Run `python main.py update`.")

ui.section("Artifact health")
health = pd.DataFrame([
    {"Artifact": "Raw dataset", "Present": RAW_PATH.exists()},
    {"Artifact": "Production model", "Present": BEST_MODEL_PATH.exists()},
    {"Artifact": "Preprocessor", "Present": DEFAULT_PREPROCESSOR_PATH.exists()},
    {"Artifact": "Model metadata", "Present": MODEL_METADATA_PATH.exists()},
    {"Artifact": "Model registry", "Present": REGISTRY_PATH.exists()},
    {"Artifact": "Prediction ledger", "Present": LEDGER_PATH.exists()},
])
ui.dataframe(health)
if registry_state.get("error"):
    st.warning(f"Registry issue: {registry_state['error']}")

st.caption(
    "Rollback restores the previous production snapshot and artifact hashes are "
    "re-verified before any deployment action. Use "
    "`python -m src.models.registry rollback --reason \"...\"`."
)
ui.footer()
