"""Phase 2 tests: candidate features, baselines, calibration, tracking, monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import (
    BASELINE_LABELS,
    class_prior,
    fixed_baseline_probabilities,
)
from src.evaluation.calibration import (
    compare_calibrated_test,
    fit_calibrator,
    select_calibration_method,
)
from src.evaluation.explainability import permutation_table
from src.experiments.tracking import (
    load_experiments,
    make_experiment_record,
    save_experiment,
)
from src.features.experimental_features import (
    CANDIDATE_FEATURE_SETS,
    EXTENDED_COLUMNS,
    build_extended_frame,
)
from src.monitoring.diagnostics import (
    check_missing_inputs,
    check_performance,
    check_prediction_distribution,
    run_diagnostics,
)
from src.predictions.history import build_prediction_record


def test_extended_features_exclude_future_matches(raw_matches):
    full_X, _full_y, full_meta = build_extended_frame(raw_matches)
    cutoff = pd.Timestamp("2025-01-01")
    truncated = raw_matches[pd.to_datetime(raw_matches["date"]) < cutoff]
    cut_X, _cut_y, cut_meta = build_extended_frame(truncated)

    common = sorted(set(full_meta["match_id"]) & set(cut_meta["match_id"]))
    assert common
    full_lookup = full_X.set_index(full_meta["match_id"]).loc[common]
    cut_lookup = cut_X.set_index(cut_meta["match_id"]).loc[common]
    pd.testing.assert_frame_equal(full_lookup, cut_lookup, check_dtype=False)


def test_candidate_feature_sets_have_expected_columns(raw_matches):
    X, _y, _meta = build_extended_frame(raw_matches)
    assert all(column in X.columns for column in EXTENDED_COLUMNS)
    for name, columns in CANDIDATE_FEATURE_SETS.items():
        assert len(columns) == len(set(columns))
        assert set(columns).issubset(X.columns), name


def test_fixed_baselines_are_valid_probabilities(model_frame):
    X, y, meta = model_frame
    for name in ("Majority class", "Home always", "Historical frequencies", "Market odds"):
        probs = fixed_baseline_probabilities(name, X.head(20), meta.head(20), y.head(80))
        assert probs.shape == (20, 3)
        assert np.isfinite(probs).all()
        assert (probs >= 0).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    assert class_prior(y.head(80)).sum() == pytest.approx(1.0)


def test_calibrators_return_valid_probabilities():
    rng = np.random.default_rng(7)
    y = rng.choice(BASELINE_LABELS, size=160, p=[0.48, 0.24, 0.28])
    raw = np.full((160, 3), 1 / 3)
    raw[:, 0] += 0.05
    raw = raw / raw.sum(axis=1, keepdims=True)
    dates = pd.date_range("2025-08-01", periods=160, freq="D")

    best, table = select_calibration_method(raw, y, dates)
    assert best in {"none", "sigmoid", "isotonic"}
    assert {"none", "sigmoid", "isotonic"}.issubset(set(table["method"]))
    assert np.isfinite(table["log_loss"]).all()

    for method in ("none", "sigmoid", "isotonic"):
        calibrator = fit_calibrator(method, raw[:100], y[:100])
        out = calibrator.transform(raw[100:])
        assert out.shape == (60, 3)
        assert np.isfinite(out).all()
        assert (out >= 0).all()
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-8)

    comparison = compare_calibrated_test(
        y[100:], raw[100:],
        {
            "sigmoid": fit_calibrator("sigmoid", raw[:100], y[:100]),
            "isotonic": fit_calibrator("isotonic", raw[:100], y[:100]),
        },
    )
    assert set(comparison["calibration"]) == {"none", "sigmoid", "isotonic"}


def test_experiment_ids_are_unique_per_variant_and_reload(tmp_path):
    common = dict(
        model_type="Random Forest",
        feature_set="base_19",
        features=["a"],
        training_period=["23/24"],
        validation_period="25/26",
        test_period="26/27",
        hyperparameters={"n_estimators": 10},
        validation_metrics={"log_loss": 1.0},
        data_fingerprint="abc",
    )
    one = make_experiment_record(**common, variant="none")
    two = make_experiment_record(**common, variant="sigmoid")
    assert one["experiment_id"] != two["experiment_id"]

    experiments_dir = tmp_path / "experiments"
    index_path = tmp_path / "index.csv"
    save_experiment(one, experiments_dir=experiments_dir, index_path=index_path)
    save_experiment(two, experiments_dir=experiments_dir, index_path=index_path)
    index = load_experiments(index_path)
    assert len(index) == 2
    assert set(index["variant"]) == {"none", "sigmoid"}
    assert len(list(experiments_dir.glob("*.json"))) == 2


def test_monitoring_distinguishes_insufficient_from_confirmed(raw_matches):
    match = raw_matches.iloc[10]
    record = build_prediction_record(
        home_team=match["home_team"],
        away_team=match["away_team"],
        match_date=f"{match['date']:%Y-%m-%d}",
        predicted_at=f"{match['date']:%Y-%m-%d}",
        model_name="Random Forest",
        model_version="test",
        prob_home=0.45, prob_draw=0.30, prob_away=0.25,
    )
    report = run_diagnostics(raw_matches, pd.DataFrame([record]))
    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses["performance"] == "insufficient_data"
    assert statuses["calibration_change"] == "insufficient_data"
    assert report["overall_status"] in {"ok", "warning", "insufficient_data"}

    # A stable 40-row ledger is still below the confirmation threshold.
    rows = []
    for i, row in raw_matches.head(40).iterrows():
        rec = build_prediction_record(
            home_team=row["home_team"],
            away_team=row["away_team"],
            match_date=f"{row['date']:%Y-%m-%d}",
            predicted_at=f"{row['date']:%Y-%m-%d}",
            model_name="Random Forest",
            model_version="test",
            prob_home=0.45, prob_draw=0.30, prob_away=0.25,
        )
        rec["actual_outcome"] = str(int(row["match_outcome"]))
        rec["actual_home_goals"] = str(int(row["full_time_home_goals"]))
        rec["actual_away_goals"] = str(int(row["full_time_away_goals"]))
        rec["correct"] = "1" if int(row["match_outcome"]) == 2 else "0"
        rows.append(rec)
    history = pd.DataFrame(rows)
    perf = check_performance(history)
    assert perf["status"] in {"insufficient_data", "ok", "warning"}
    assert int(perf["details"].get("recent_n", 0)) == 0 or perf["status"] != "confirmed"


def test_monitoring_confirms_large_sustained_degradation(raw_matches):
    labels = [2, 1, 0]
    rows = []
    for i, row in raw_matches.head(120).iterrows():
        actual = int(row["match_outcome"])
        correct_index = labels.index(actual)
        chosen = correct_index if i < 60 else (correct_index + 1) % 3
        probs = [0.05, 0.05, 0.05]
        probs[chosen] = 0.90
        total = sum(probs)
        probs = [value / total for value in probs]
        rec = build_prediction_record(
            home_team=row["home_team"],
            away_team=row["away_team"],
            match_date=f"{row['date']:%Y-%m-%d}",
            predicted_at=f"{row['date']:%Y-%m-%d}",
            model_name="Random Forest",
            model_version="test",
            prob_home=probs[0], prob_draw=probs[1], prob_away=probs[2],
        )
        rec["actual_outcome"] = str(actual)
        rec["actual_home_goals"] = str(int(row["full_time_home_goals"]))
        rec["actual_away_goals"] = str(int(row["full_time_away_goals"]))
        rec["correct"] = "1" if chosen == correct_index else "0"
        rows.append(rec)

    result = check_performance(pd.DataFrame(rows))
    assert result["status"] == "confirmed"
    assert result["details"]["log_loss_delta"] > 0.5
    assert result["details"]["brier_delta_ci"][0] > 0


def test_prediction_distribution_and_missing_inputs(raw_matches):
    rows = []
    for i, row in raw_matches.head(80).iterrows():
        rec = build_prediction_record(
            home_team=row["home_team"],
            away_team=row["away_team"],
            match_date=f"{row['date']:%Y-%m-%d}",
            predicted_at=f"{row['date']:%Y-%m-%d}",
            model_name="Random Forest",
            model_version="test",
            prob_home=0.45, prob_draw=0.30, prob_away=0.25,
        )
        rows.append(rec)
    result = check_prediction_distribution(pd.DataFrame(rows))
    assert result["status"] in {"ok", "warning", "confirmed", "insufficient_data"}
    assert "message" in result

    missing = check_missing_inputs(raw_matches)
    # The synthetic season is historical, so staleness may legitimately warn.
    assert missing["status"] in {"ok", "warning"}
    assert "message" in missing


def test_permutation_importance_is_finite_and_complete(model_frame):
    from sklearn.ensemble import RandomForestClassifier

    X, y, _meta = model_frame
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X.iloc[:80], y.iloc[:80])
    table = permutation_table(
        model, X.iloc[80:100], y.iloc[80:100], "test", n_repeats=2
    )
    assert len(table) == X.shape[1]
    assert np.isfinite(table["importance_mean"]).all()
    assert set(table["feature"]) == set(X.columns)
