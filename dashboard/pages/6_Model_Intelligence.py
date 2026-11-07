"""Phase 2 model-intelligence page: baselines, experiments and calibration."""

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
    BASELINE_TEST_PATH,
    BASELINE_VALIDATION_PATH,
    CALIBRATION_COMPARISON_PATH,
    CALIBRATION_SELECTION_PATH,
    ELO_HISTORY_PATH,
    ELO_RATINGS_PATH,
    ENSEMBLE_COMPARISON_PATH,
    ENSEMBLE_WEIGHTS_PATH,
    EVAL_DIR,
    EXPERIMENTS_INDEX_PATH,
    FEATURE_EXPERIMENTS_PATH,
    FEATURE_RECOMMENDATION_PATH,
    FEATURE_STABILITY_PATH,
    MARKET_DISAGREEMENT_PATH,
    MARKET_MODEL_COMPARISON_PATH,
    MARKET_OVERROUND_PATH,
    MARKET_PROBABILITY_DIFFERENCES_PATH,
    MARKET_STACK_PATH,
    MODEL_METADATA_PATH,
    MONITORING_REPORT_PATH,
    PERMUTATION_IMPORTANCE_PATH,
    PHASE2_SUMMARY_PATH,
    PHASE3_SUMMARY_PATH,
    PHASE3_TEST_PATH,
    PHASE3_VALIDATION_PATH,
    PHASE3_WALK_FORWARD_PATH,
    POISSON_RATINGS_PATH,
    POISSON_SCORELINES_PATH,
    POISSON_SUMMARY_PATH,
    POISSON_TEST_PREDICTIONS_PATH,
    RAW_PATH,
    UNCERTAINTY_PATH,
    UNCERTAINTY_SUMMARY_PATH,
)

ui.configure_page("Model Intelligence", "🔬")
ui.hero(
    "Model Intelligence",
    "Baselines, validation-only calibration, dynamic Elo, Poisson/Dixon-Coles "
    "goal models, constrained ensembles, market comparisons and prediction "
    "uncertainty — without changing the production model by default.",
    "Phase 2 + Phase 3 · Diagnostics",
)

metadata = data.load_metadata(data._mtime(MODEL_METADATA_PATH))
baseline_validation = data.load_baseline_validation(data._mtime(BASELINE_VALIDATION_PATH))
baseline_test = data.load_baseline_test(data._mtime(BASELINE_TEST_PATH))
experiments = data.load_feature_experiments(data._mtime(FEATURE_EXPERIMENTS_PATH))
recommendation = data.load_feature_recommendation(data._mtime(FEATURE_RECOMMENDATION_PATH))
calibration = data.load_calibration_comparison(data._mtime(CALIBRATION_COMPARISON_PATH))
calibration_selection = data.load_calibration_selection(data._mtime(CALIBRATION_SELECTION_PATH))
calibration_note = data.load_calibration_recommendation(data._mtime(CALIBRATION_COMPARISON_PATH))
monitoring = data.load_monitoring_report(data._mtime(MONITORING_REPORT_PATH))
phase3_summary = data.load_phase3_summary(data._mtime(PHASE3_SUMMARY_PATH))
phase3_validation = data.load_phase3_validation(data._mtime(PHASE3_VALIDATION_PATH))
phase3_test = data.load_phase3_test(data._mtime(PHASE3_TEST_PATH))
phase3_walk_forward = data.load_phase3_walk_forward(data._mtime(PHASE3_WALK_FORWARD_PATH))
elo_history = data.load_elo_history(data._mtime(ELO_HISTORY_PATH))
elo_ratings = data.load_elo_ratings(data._mtime(ELO_RATINGS_PATH))
poisson_ratings = data.load_poisson_ratings(data._mtime(POISSON_RATINGS_PATH))
poisson_predictions = data.load_poisson_predictions(data._mtime(POISSON_TEST_PREDICTIONS_PATH))
poisson_scorelines = data.load_poisson_scorelines(data._mtime(POISSON_SCORELINES_PATH))
poisson_note = data.load_poisson_summary(data._mtime(POISSON_SUMMARY_PATH))
ensemble_weights = data.load_ensemble_weights(data._mtime(ENSEMBLE_WEIGHTS_PATH))
ensemble_comparison = data.load_ensemble_comparison(data._mtime(ENSEMBLE_COMPARISON_PATH))
market_comparison = data.load_market_comparison(data._mtime(MARKET_MODEL_COMPARISON_PATH))
market_differences = data.load_market_differences(data._mtime(MARKET_PROBABILITY_DIFFERENCES_PATH))
market_disagreement = data.load_market_disagreement(data._mtime(MARKET_DISAGREEMENT_PATH))
market_overround = data.load_market_overround(data._mtime(MARKET_OVERROUND_PATH))
market_stack = data.load_market_stack(data._mtime(MARKET_STACK_PATH))
uncertainty = data.load_uncertainty(data._mtime(UNCERTAINTY_PATH))
uncertainty_summary = data.load_uncertainty_summary(data._mtime(UNCERTAINTY_SUMMARY_PATH))

best_baseline = "n/a"
if not baseline_test.empty and "log_loss" in baseline_test:
    best = baseline_test.sort_values("log_loss").iloc[0]
    best_baseline = f"{best['model']} · {float(best['log_loss']):.3f}"

selected_cal = ", ".join(
    f"{model}: {method}" for model, method in (calibration_note.get("selected_by_model") or {}).items()
) or "n/a"

ui.metric_row([
    {
        "label": "Feature set",
        "value": recommendation.get("recommended_feature_set", "n/a"),
        "help": recommendation.get("reason", "No recommendation artifact yet."),
        "accent": "green",
    },
    {
        "label": "Best test baseline",
        "value": best_baseline.split(" · ")[0] if " · " in best_baseline else "n/a",
        "help": f"Log-loss {best_baseline.split(' · ')[1]}" if " · " in best_baseline else "Run experiments",
        "accent": "blue",
    },
    {
        "label": "Calibration (validation)",
        "value": selected_cal,
        "help": "Selected on a chronological validation holdout, not test.",
        "accent": "gold",
    },
    {
        "label": "Monitoring",
        "value": monitoring.get("overall_status", "not run"),
        "help": "Insufficient data is not the same as healthy.",
        "accent": "red" if monitoring.get("overall_status") == "confirmed" else "green",
    },
])

tabs = st.tabs([
    "Baselines", "Feature experiments", "Calibration",
    "Importance & stability", "Experiment log",
    "Elo ratings", "Poisson & scorelines", "Phase 3 models",
    "Ensemble & market", "Uncertainty",
])

with tabs[0]:
    ui.section(
        "Fixed baselines vs learned models",
        "All rows use the same chronological windows. Baselines are fixed before test scoring.",
    )
    if baseline_validation.empty and baseline_test.empty:
        st.info(
            "No baseline artifacts. Run `python main.py experiments` after training the model."
        )
    else:
        for label, frame, path in (
            ("Validation (25/26)", baseline_validation, BASELINE_VALIDATION_PATH),
            ("Test (26/27)", baseline_test, BASELINE_TEST_PATH),
        ):
            if frame.empty:
                continue
            st.markdown(f"**{label}**")
            show = frame.copy()
            for col in ("accuracy", "draw_recall"):
                if col in show:
                    show[col] = show[col].map(lambda v: ui.pct(v))
            for col in ("log_loss", "brier_score", "ece"):
                if col in show:
                    show[col] = show[col].map(lambda v: ui.num(v, 4))
            keep = [c for c in (
                "model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece",
            ) if c in show.columns]
            ui.dataframe(show[keep])
        image = EVAL_DIR / "baseline_comparison.png"
        if image.exists():
            st.image(str(image), width="stretch")
        st.caption(
            "A learned model that does not beat the market baseline is reported as a "
            "negative result. Market odds are also an input feature, so this is not an "
            "independent edge."
        )

with tabs[1]:
    ui.section(
        "Feature-group experiments",
        "Each candidate adds one group to the canonical 19 features. Selection used validation and walk-forward only.",
    )
    if experiments.empty:
        st.info("No feature experiments yet.")
    else:
        show = experiments.copy()
        for col in (
            "validation_log_loss", "walk_forward_log_loss",
            "delta_validation_log_loss", "delta_walk_forward_log_loss",
            "delta_validation_ci_low", "delta_validation_ci_high",
        ):
            if col in show:
                show[col] = show[col].map(lambda v: ui.num(v, 4))
        keep = [c for c in (
            "model", "feature_set", "n_features", "validation_log_loss",
            "walk_forward_log_loss", "delta_validation_log_loss",
            "delta_validation_ci_low", "delta_validation_ci_high",
            "delta_walk_forward_log_loss",
        ) if c in show.columns]
        ui.dataframe(show[keep])
        if recommendation:
            st.success(
                f"Recommendation: **{recommendation.get('recommended_feature_set')}** — "
                f"{recommendation.get('reason', '')}"
            )
            st.caption(recommendation.get("rule", ""))
            evidence = recommendation.get("head_to_head_evidence") or []
            if evidence:
                st.markdown("**Head-to-head evidence**")
                frame = pd.DataFrame(evidence)
                for col in (
                    "delta_validation_log_loss", "delta_validation_ci_low",
                    "delta_validation_ci_high", "delta_walk_forward_log_loss",
                ):
                    if col in frame:
                        frame[col] = frame[col].map(lambda v: ui.num(v, 4))
                ui.dataframe(frame)
                st.caption(
                    "The interval includes zero for both models, so head-to-head is "
                    "not retained despite its slightly better point estimates."
                )
        image = EVAL_DIR / "feature_experiments.png"
        if image.exists():
            st.image(str(image), width="stretch")

with tabs[2]:
    ui.section(
        "Calibration comparison",
        "Methods were chosen on a validation holdout, then scored once on test. No calibrator is deployed automatically.",
    )
    if calibration.empty:
        st.info("No calibration artifacts yet.")
    else:
        show = calibration.copy()
        for col in ("accuracy", "draw_recall"):
            if col in show:
                show[col] = show[col].map(lambda v: ui.pct(v))
        for col in ("log_loss", "brier_score", "ece"):
            if col in show:
                show[col] = show[col].map(lambda v: ui.num(v, 4))
        keep = [c for c in (
            "model", "calibration", "selected_on_validation", "accuracy",
            "log_loss", "brier_score", "draw_recall", "ece",
        ) if c in show.columns]
        ui.dataframe(show[keep])
        note = calibration_note.get("deployment_note", "")
        if note:
            st.warning(note)
        if not calibration_selection.empty:
            st.markdown("**Validation-holdout selection**")
            sel = calibration_selection.copy()
            for col in ("accuracy", "draw_recall"):
                if col in sel:
                    sel[col] = sel[col].map(lambda v: ui.pct(v))
            for col in ("log_loss", "brier_score", "ece"):
                if col in sel:
                    sel[col] = sel[col].map(lambda v: ui.num(v, 4))
            ui.dataframe(sel)
        for model in ("random_forest", "xgboost"):
            image = EVAL_DIR / f"calibration_comparison_{model}.png"
            if image.exists():
                st.image(str(image), width="stretch")

with tabs[3]:
    ui.section(
        "Feature importance and stability",
        "Descriptive only. Correlated football features share credit; importance is not causal.",
    )
    permutation = data.load_permutation_importance(data._mtime(PERMUTATION_IMPORTANCE_PATH))
    stability = data.load_feature_stability(data._mtime(FEATURE_STABILITY_PATH))
    if permutation.empty and stability.empty:
        st.info("No explainability artifacts yet.")
    else:
        if not permutation.empty:
            st.markdown("**Permutation importance (validation, top 10 per model)**")
            top = permutation.sort_values(
                ["model", "importance_mean"], ascending=[True, False]
            ).groupby("model").head(10)
            show = top[["model", "feature", "importance_mean", "importance_std"]].copy()
            show["importance_mean"] = show["importance_mean"].map(lambda v: ui.num(v, 5))
            show["importance_std"] = show["importance_std"].map(lambda v: ui.num(v, 5))
            ui.dataframe(show)
        if not stability.empty:
            st.markdown("**Stability across walk-forward folds**")
            show = stability.sort_values(["model", "method", "mean_rank"]).groupby(
                ["model", "method"]
            ).head(8)
            keep = [c for c in (
                "model", "method", "feature", "folds", "mean_importance",
                "std_importance", "mean_rank",
            ) if c in show.columns]
            ui.dataframe(show[keep])
        for name in ("feature_importance_comparison.png", "feature_stability.png"):
            image = EVAL_DIR / name
            if image.exists():
                st.image(str(image), width="stretch")

with tabs[4]:
    ui.section(
        "Experiment ledger",
        "Each row is reproducible from its JSON record: data fingerprint, feature hash, split and hyperparameters.",
    )
    experiment_index = data.load_experiments(data._mtime(EXPERIMENTS_INDEX_PATH))
    if experiment_index.empty:
        st.info("No experiment records yet.")
    else:
        keep = [c for c in (
            "experiment_id", "model_type", "feature_set", "variant",
            "validation_log_loss", "test_log_loss", "test_evaluated",
            "selection_status",
        ) if c in experiment_index.columns]
        show = experiment_index.sort_values("timestamp", ascending=False)[keep].copy()
        for col in ("validation_log_loss", "test_log_loss"):
            if col in show:
                show[col] = show[col].map(lambda v: ui.num(v, 4))
        ui.dataframe(show)
        st.caption(
            "Test log-loss is blank for experiments where test was deliberately "
            "withheld during selection."
        )
    summary = data.load_phase2_summary(data._mtime(PHASE2_SUMMARY_PATH))
    if summary:
        with st.expander("Phase 2 summary JSON"):
            st.json(summary)
    image = EVAL_DIR / "experiment_leaderboard.png"
    if image.exists():
        st.image(str(image), width="stretch")

with tabs[5]:
    ui.section(
        "Dynamic Elo ratings",
        "Chronological pre-match ratings with margin-of-victory updates and rating uncertainty. "
        "Higher rating deviation means less certainty, not a worse team.",
    )
    if elo_ratings.empty:
        st.info("No Elo artifacts. Run `python main.py phase3` after training the model.")
    else:
        top = elo_ratings.iloc[0]
        ui.metric_row([
            {"label": "Top rating", "value": f"{top['team']} · {ui.num(top['rating'], 1)}",
             "help": "Latest snapshot; descriptive only.", "accent": "green"},
            {"label": "Mean rating deviation", "value": ui.num(elo_ratings['rating_deviation'].mean(), 1),
             "help": "Lower is more established; first-season clubs start at 350.", "accent": "blue"},
            {"label": "Clubs rated", "value": str(len(elo_ratings)),
             "help": "All clubs with at least one cached match.", "accent": "gold"},
        ])
        show = elo_ratings.head(15).copy()
        show["rating"] = show["rating"].map(lambda v: ui.num(v, 2))
        show["rating_deviation"] = show["rating_deviation"].map(lambda v: ui.num(v, 1))
        ui.dataframe(show[["rank", "team", "rating", "rating_deviation", "matches"]])

        options = elo_ratings["team"].tolist()
        selected = st.multiselect(
            "Rating history", options, default=options[:3], key="elo_teams"
        )
        if selected and not elo_history.empty:
            history = elo_history.copy()
            history["date"] = pd.to_datetime(history["date"], errors="coerce")
            part = history[history["team"].isin(selected)]
            pivot = part.pivot_table(
                index="date", columns="team", values="rating", aggfunc="last"
            ).sort_index()
            st.line_chart(pivot)
        for name in ("elo_rating_history.png", "elo_rating_uncertainty.png"):
            image = EVAL_DIR / name
            if image.exists():
                st.image(str(image), width="stretch")

with tabs[6]:
    ui.section(
        "Poisson and Dixon-Coles goal model",
        "Expected goals come from time-decayed attack/defence estimates. Scorelines are "
        "a normalized Poisson matrix; the Dixon-Coles correction adjusts low-score cells.",
    )
    if poisson_predictions.empty:
        st.info("No Poisson artifacts yet.")
    else:
        config = poisson_note.get("config", {})
        ui.metric_row([
            {"label": "Selected half-life", "value": f"{ui.num(config.get('half_life_days'), 0)} d",
             "help": "Selected on validation only.", "accent": "blue"},
            {"label": "Prior strength", "value": ui.num(config.get('prior_strength'), 1),
             "help": "Shrinkage toward the league average.", "accent": "gold"},
            {"label": "Fitted rho", "value": ui.num(poisson_note.get('rho'), 4),
             "help": "Dixon-Coles low-score correction.", "accent": "green"},
        ])
        labels = {
            str(row.match_id): f"{row.home_team} vs {row.away_team} · {str(row.date)[:10]}"
            for row in poisson_predictions.itertuples(index=False)
        }
        match_id = st.selectbox(
            "Fixture", list(labels), format_func=lambda value: labels.get(value, value),
            key="poisson_fixture",
        )
        row = poisson_predictions[poisson_predictions["match_id"] == match_id].iloc[0]
        ui.metric_row([
            {"label": "Expected goals", "value": f"{ui.num(row['lambda_home'], 2)} – {ui.num(row['lambda_away'], 2)}",
             "help": "Home – away expected goals.", "accent": "green"},
            {"label": "Most likely score", "value": str(row["most_likely_score"]),
             "help": f"Probability {ui.pct(row['most_likely_score_probability'])}", "accent": "gold"},
            {"label": "Home known / away known", "value": f"{bool(row['home_known'])} / {bool(row['away_known'])}",
             "help": "False means a sparse/unknown team fell back to league average.", "accent": "blue"},
        ])
        ui.probability_bar(
            {"home": row["prob_home"], "draw": row["prob_draw"], "away": row["prob_away"]},
            row["home_team"], row["away_team"],
        )

        col_a, col_b = st.columns([2, 3])
        with col_a:
            st.markdown("**Most likely scorelines**")
            scores = poisson_scorelines[poisson_scorelines["match_id"] == match_id].copy()
            scores = scores.sort_values("probability", ascending=False).head(10)
            scores["scoreline"] = scores["home_goals"].astype(int).astype(str) + "-" + scores["away_goals"].astype(int).astype(str)
            scores["probability"] = scores["probability"].map(lambda v: ui.pct(v, 2))
            ui.dataframe(scores[["scoreline", "probability"]])
        with col_b:
            st.markdown("**Scoreline probability heatmap**")
            matrix = poisson_scorelines[poisson_scorelines["match_id"] == match_id].pivot(
                index="home_goals", columns="away_goals", values="probability"
            ).fillna(0.0)
            st.dataframe(
                matrix.style.background_gradient(cmap="Greens", axis=None).format("{:.3f}"),
                width="stretch",
            )
            st.caption("Rows = home goals, columns = away goals. The matrix sums to 100%.")

        st.markdown("**Team attack and defence estimates**")
        ratings = poisson_ratings.head(15).copy()
        for column in ("attack", "defence"):
            ratings[column] = ratings[column].map(lambda v: ui.num(v, 3))
        if "effective_matches" in ratings:
            ratings["effective_matches"] = ratings["effective_matches"].map(lambda v: ui.num(v, 1))
        ui.dataframe(ratings[["team", "attack", "defence", "matches", "effective_matches", "known"]])
        image = EVAL_DIR / "poisson_attack_defence.png"
        if image.exists():
            st.image(str(image), width="stretch")

with tabs[7]:
    ui.section(
        "Phase 3 model comparison",
        "Validation and one-shot test results on identical chronological windows. "
        "No model is promoted automatically; the production model is unchanged.",
    )
    if phase3_test.empty and phase3_validation.empty:
        st.info("No Phase 3 comparison artifacts. Run `python main.py phase3`.")
    else:
        best_test = phase3_test.sort_values("log_loss").iloc[0] if not phase3_test.empty else None
        best_val = phase3_validation.sort_values("log_loss").iloc[0] if not phase3_validation.empty else None
        ui.metric_row([
            {"label": "Best test log-loss", "value": ui.num(best_test["log_loss"], 4) if best_test is not None else "n/a",
             "help": str(best_test["model"]) if best_test is not None else "No data", "accent": "blue"},
            {"label": "Best validation log-loss", "value": ui.num(best_val["log_loss"], 4) if best_val is not None else "n/a",
             "help": str(best_val["model"]) if best_val is not None else "No data", "accent": "green"},
            {"label": "Market test benchmark", "value": ui.num(phase3_summary.get("market_test_log_loss"), 4),
             "help": "Lower is better; reported without spin.", "accent": "gold"},
            {"label": "Deployment", "value": "Not promoted",
             "help": phase3_summary.get("deployment_decision", "Current production preserved."), "accent": "red"},
        ])
        for label, frame in (("Validation (25/26)", phase3_validation), ("Test (26/27)", phase3_test)):
            if frame.empty:
                continue
            st.markdown(f"**{label}**")
            show = frame.copy()
            for column in ("accuracy", "draw_recall"):
                if column in show:
                    show[column] = show[column].map(lambda v: ui.pct(v))
            for column in ("log_loss", "brier_score", "ece"):
                if column in show:
                    show[column] = show[column].map(lambda v: ui.num(v, 4))
            ui.dataframe(show[["model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"]])
        image = EVAL_DIR / "phase3_model_comparison.png"
        if image.exists():
            st.image(str(image), width="stretch")
        if not phase3_walk_forward.empty:
            with st.expander("Walk-forward folds"):
                show = phase3_walk_forward.copy()
                for column in ("accuracy", "draw_recall"):
                    if column in show:
                        show[column] = show[column].map(lambda v: ui.pct(v))
                for column in ("log_loss", "brier_score", "ece"):
                    if column in show:
                        show[column] = show[column].map(lambda v: ui.num(v, 4))
                ui.dataframe(show[["fold", "test_season", "model", "log_loss", "brier_score", "accuracy", "draw_recall", "ece"]])
        st.caption(
            "Dynamic Elo is the closest independent challenger on this window, but it "
            "still does not beat the market benchmark. The test season has only 109 matches."
        )

with tabs[8]:
    ui.section(
        "Constrained ensemble and market analysis",
        "Ensemble weights are non-negative, sum to one and are selected on validation only. "
        "Market odds are a benchmark, not a claim of a betting edge.",
    )
    if ensemble_weights.empty:
        st.info("No ensemble artifacts yet.")
    else:
        show = ensemble_weights.copy()
        show["weight"] = show["weight"].map(lambda v: ui.num(v, 2))
        show["equal_weight"] = show["equal_weight"].map(lambda v: ui.num(v, 2))
        ui.dataframe(show[["component", "weight", "equal_weight"]])
        st.bar_chart(ensemble_weights.set_index("component")["weight"])
        image = EVAL_DIR / "ensemble_weights.png"
        if image.exists():
            st.image(str(image), width="stretch")
        if not ensemble_comparison.empty:
            st.markdown("**Ensemble comparison**")
            show = ensemble_comparison.copy()
            for column in ("accuracy", "draw_recall"):
                if column in show:
                    show[column] = show[column].map(lambda v: ui.pct(v))
            for column in ("log_loss", "brier_score", "ece"):
                if column in show:
                    show[column] = show[column].map(lambda v: ui.num(v, 4))
            ui.dataframe(show[["split", "model", "accuracy", "log_loss", "brier_score", "draw_recall", "ece"]])

    if not market_comparison.empty:
        st.markdown("**Model versus market**")
        show = market_comparison.copy()
        for column in ("delta_log_loss", "delta_log_loss_ci_low", "delta_log_loss_ci_high", "mean_abs_probability_difference"):
            if column in show:
                show[column] = show[column].map(lambda v: ui.num(v, 4))
        ui.dataframe(show[["model", "model_log_loss", "market_log_loss", "delta_log_loss", "delta_log_loss_ci_low", "delta_log_loss_ci_high", "beats_market_log_loss"]])
        st.caption("Positive delta means the model is worse than the market on test log-loss. The bootstrap interval is descriptive.")
    if not market_overround.empty:
        with st.expander("Overround removal by season"):
            show = market_overround.copy()
            for column in ("mean_overround", "median_overround", "min_overround", "max_overround"):
                if column in show:
                    show[column] = show[column].map(lambda v: ui.num(v, 4))
            ui.dataframe(show)
    if not market_disagreement.empty:
        with st.expander("Where model and market disagree"):
            part = market_disagreement[market_disagreement["subset"] == "disagreement"].copy()
            for column in ("model_log_loss", "market_log_loss", "model_accuracy", "market_accuracy"):
                if column in part:
                    part[column] = part[column].map(lambda v: ui.num(v, 4) if "loss" in column else ui.pct(v))
            ui.dataframe(part[["model", "n", "model_log_loss", "market_log_loss", "model_accuracy", "market_accuracy", "model_beats_market"]])
    if not market_stack.empty:
        st.markdown("**Validation-fitted market + model stack**")
        show = market_stack.copy()
        for column in ("log_loss", "brier_score", "market_log_loss", "delta_log_loss_vs_market", "delta_log_loss_ci_low", "delta_log_loss_ci_high"):
            if column in show:
                show[column] = show[column].map(lambda v: ui.num(v, 4))
        if "draw_recall" in show:
            show["draw_recall"] = show["draw_recall"].map(lambda v: ui.pct(v))
        ui.dataframe(show[["model", "log_loss", "brier_score", "draw_recall", "delta_log_loss_vs_market", "delta_log_loss_ci_low", "delta_log_loss_ci_high"]])
        st.caption("The stack is diagnostic only. Its interval includes zero, so incremental value beyond the market is not established.")
    if not market_differences.empty:
        st.markdown("**Per-fixture market versus model differences**")
        models = sorted(market_differences["model"].dropna().unique().tolist())
        selected_model = st.selectbox("Model", models, key="market_model_select")
        part = market_differences[market_differences["model"] == selected_model].copy()
        part["match"] = part["home_team"] + " vs " + part["away_team"]
        for column in ("market_home", "market_draw", "market_away", "model_home", "model_draw", "model_away"):
            if column in part:
                part[column] = part[column].map(lambda v: ui.num(v, 3))
        ui.dataframe(part[["date", "match", "market_home", "model_home", "diff_home", "market_draw", "model_draw", "diff_draw"]].head(30))
        image = EVAL_DIR / "market_probability_scatter.png"
        if image.exists():
            st.image(str(image), width="stretch")

with tabs[9]:
    ui.section(
        "Prediction uncertainty",
        "Confidence, entropy, model agreement and historical reliability are kept separate. "
        "A high probability is not certainty.",
    )
    if uncertainty.empty:
        st.info("No uncertainty artifacts yet.")
    else:
        ui.metric_row([
            {"label": "Fixtures", "value": str(len(uncertainty)),
             "help": "One row per test fixture.", "accent": "blue"},
            {"label": "Mean confidence", "value": ui.pct(uncertainty_summary.get("mean_confidence")),
             "help": "Largest class probability.", "accent": "green"},
            {"label": "Mean entropy", "value": ui.num(uncertainty_summary.get("mean_entropy"), 3),
             "help": "Normalized 0–1 spread across outcomes.", "accent": "gold"},
            {"label": "High uncertainty", "value": ui.pct(uncertainty_summary.get("high_uncertainty_share")),
             "help": "Descriptive diagnostic, not a guarantee.", "accent": "red"},
        ])
        levels = ["all"] + sorted(uncertainty["uncertainty_level"].dropna().unique().tolist())
        selected_level = st.selectbox("Uncertainty level", levels, key="uncertainty_level")
        part = uncertainty if selected_level == "all" else uncertainty[uncertainty["uncertainty_level"] == selected_level]
        show = part.sort_values("entropy", ascending=False).head(40).copy()
        for column in ("prob_home", "prob_draw", "prob_away", "confidence"):
            if column in show:
                show[column] = show[column].map(lambda v: ui.pct(v))
        for column in ("entropy", "margin", "model_agreement", "component_std", "component_deviation"):
            if column in show:
                show[column] = show[column].map(lambda v: ui.num(v, 3))
        ui.dataframe(show[["date", "home_team", "away_team", "prob_home", "prob_draw", "prob_away", "confidence", "entropy", "model_agreement", "historical_reliability", "uncertainty_level"]])
        if "correlation_entropy_loss" in uncertainty_summary:
            st.caption(
                f"Spearman correlation between entropy and realised log-loss: "
                f"{ui.num(uncertainty_summary.get('correlation_entropy_loss'), 3)}. "
                "Higher entropy tended to mean worse realised loss in this window, but this is descriptive."
            )
        image = EVAL_DIR / "prediction_uncertainty.png"
        if image.exists():
            st.image(str(image), width="stretch")

ui.footer()
