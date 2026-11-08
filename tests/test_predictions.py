"""Prediction, ledger and reconciliation tests."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.build_features import build_model_frame, build_team_history
from src.features.preprocessing import FeaturePreprocessor
from src.models import registry
from src.predictions import predictor as predictor_module
from src.predictions.fixtures import parse_fixture_line
from src.predictions.history import (
    append_predictions,
    build_prediction_record,
    load_history,
    normalize_history,
    performance_by,
    performance_summary,
    reconcile_results,
)
from src.predictions.predictor import (
    PREDICTION_LABELS,
    predict_fixtures,
    team_form,
)
from src.utils.config import FEATURE_COLUMNS


def make_record(**overrides):
    base = dict(
        home_team="Real Madrid",
        away_team="Barcelona",
        match_date="2026-11-08",
        predicted_at="2026-11-05T10:00:00+00:00",
        model_name="Random Forest",
        model_version="test-v1",
        prob_home=0.5,
        prob_draw=0.3,
        prob_away=0.2,
        market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
    )
    base.update(overrides)
    return build_prediction_record(**base)


def test_prediction_record_is_normalized_and_labeled():
    record = make_record(prob_home=0.6, prob_draw=0.3, prob_away=0.3)
    total = sum(
        float(record[f"model_prob_{side}"]) for side in ("home", "draw", "away")
    )
    assert total == pytest.approx(1.0)
    assert record["predicted_outcome"] == "Home Win"
    assert record["fixture_id"] == "20261108_Real Madrid_v_Barcelona"
    assert float(record["confidence"]) == pytest.approx(0.5)


def test_inference_loads_registry_snapshot_after_live_artifacts_are_gone(
    tmp_path, monkeypatch, raw_matches
):
    """A clean checkout must predict from the committed production snapshot."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(registry, "PROJECT_ROOT", checkout)

    X, y, _ = build_model_frame(raw_matches)
    preprocessor = FeaturePreprocessor(feature_names=list(FEATURE_COLUMNS)).fit(X)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        preprocessor.transform(X), y
    )

    live_model = checkout / "models" / "saved_models" / "best_model.pkl"
    live_model.parent.mkdir(parents=True)
    joblib.dump(model, live_model)
    live_pre = checkout / "data" / "processed" / "preprocessor.joblib"
    live_pre.parent.mkdir(parents=True)
    preprocessor.save(live_pre)

    entry, created = registry.register_model(
        {
            "model_name": "Random Forest",
            "model_version": "prod-snapshot-v1",
            "feature_names": list(FEATURE_COLUMNS),
        },
        model_path=live_model,
        preprocessor_path=live_pre,
        path=checkout / "registry.json",
        versions_dir=checkout / "models" / "registry" / "versions",
    )
    assert created and entry["status"] == "production"

    # Simulate a fresh checkout: only registry snapshots are present.
    live_model.unlink()
    live_pre.unlink()

    real_production_paths = registry.production_paths
    monkeypatch.setattr(
        registry,
        "production_paths",
        lambda: real_production_paths(checkout / "registry.json"),
    )
    raw_path = tmp_path / "laliga_matches.csv"
    raw_matches.to_csv(raw_path, index=False)
    monkeypatch.setattr(predictor_module, "RAW_PATH", raw_path)

    loaded_model, loaded_pre, metadata = predictor_module.load_artifacts()
    assert metadata["model_version"] == "prod-snapshot-v1"
    assert loaded_pre.feature_names == list(FEATURE_COLUMNS)
    assert hasattr(loaded_model, "predict_proba")

    predictions, errors = predictor_module.predict_fixtures(
        [{"home_team": "Real Madrid", "away_team": "Barcelona", "match_date": "2026-12-01"}],
        raw=raw_matches,
    )
    assert errors == []
    assert len(predictions) == 1
    assert predictions[0]["model_version"] == "prod-snapshot-v1"
    assert sum(predictions[0]["probs"].values()) == pytest.approx(1.0)


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        make_record(prob_home=np.nan)
    with pytest.raises(ValueError):
        make_record(prob_home=0, prob_draw=0, prob_away=0)


def test_legacy_ledger_is_migrated_and_label_corrected():
    legacy = pd.DataFrame([{
        "date_predicted": "2026-11-01",
        "home_team": "Real Madrid",
        "away_team": "Valencia",
        "model_prob_home": "0.381489",
        "model_prob_draw": "0.333286",
        "model_prob_away": "0.285225",
        "market_prob_home": "0.47",
        "market_prob_draw": "0.27",
        "market_prob_away": "0.31",
        "predicted_outcome": "Away Win",  # deliberately inconsistent
        "actual_outcome": "",
    }])
    normalized = normalize_history(legacy)
    assert normalized.loc[0, "predicted_outcome"] == "Home Win"
    assert normalized.loc[0, "fixture_id"] == "TBD_Real Madrid_v_Valencia"
    assert float(normalized.loc[0, "confidence"]) == pytest.approx(0.381489)
    assert set(normalized.columns) >= set([
        "fixture_id", "match_date", "predicted_at", "model_version",
    ])


def test_append_predictions_deduplicates_and_preserves_snapshots(tmp_path):
    path = tmp_path / "history.csv"
    record = make_record()

    appended, history = append_predictions([record], path)
    assert appended == 1 and len(history) == 1

    appended, history = append_predictions([record], path)
    assert appended == 0 and len(history) == 1

    newer = make_record(predicted_at="2026-11-06T10:00:00+00:00", prob_home=0.4,
                        prob_draw=0.35, prob_away=0.25)
    appended, history = append_predictions([newer], path)
    assert appended == 1 and len(history) == 2

    loaded, _ = load_history(path)
    assert len(loaded) == 2
    assert loaded["actual_outcome"].astype(str).str.strip().eq("").all()


def test_reconcile_fills_result_and_never_overwrites(tmp_path, tiny_history, raw_matches):
    history = normalize_history(pd.DataFrame([tiny_history]))
    updated, stats = reconcile_results(history, raw_matches, as_of=pd.Timestamp("2027-06-01"))
    assert stats["settled"] == 1
    assert updated.loc[0, "actual_outcome"] in {"0", "1", "2"}
    assert updated.loc[0, "correct"] in {"0", "1"}
    assert updated.loc[0, "settled_at"]

    # A settled row is immutable, even if the raw result changes.
    settled_value = updated.loc[0, "actual_outcome"]
    changed_raw = raw_matches.copy()
    changed_raw.loc[
        changed_raw["match_id"] == updated.loc[0, "fixture_id"], "match_outcome"
    ] = 9
    again, stats2 = reconcile_results(
        updated, changed_raw, as_of=pd.Timestamp("2027-06-01")
    )
    assert stats2["already_settled"] == 1
    assert again.loc[0, "actual_outcome"] == settled_value


def test_reconcile_handles_timezone_and_tbd_fixture(raw_matches):
    record = make_record(
        home_team=raw_matches.iloc[0]["home_team"],
        away_team=raw_matches.iloc[0]["away_team"],
        match_date="",
        predicted_at="2026-08-01T10:00:00+00:00",
    )
    assert record["fixture_id"].startswith("TBD_")
    updated, stats = reconcile_results(
        normalize_history(pd.DataFrame([record])),
        raw_matches,
        as_of=pd.Timestamp("2026-08-10"),
    )
    # The pairing's 2026 meeting (25 Aug) is after the cutoff, so it must NOT
    # settle yet. The tz-aware prediction timestamp must not crash the check.
    assert stats["settled"] == 0
    assert stats["pending"] == 1


def test_missing_prediction_timestamp_never_settles_from_old_season(raw_matches):
    record = make_record(
        home_team=raw_matches.iloc[0]["home_team"],
        away_team=raw_matches.iloc[0]["away_team"],
        match_date="",
        predicted_at="",
    )
    updated, stats = reconcile_results(
        normalize_history(pd.DataFrame([record])),
        raw_matches,
        as_of=pd.Timestamp("2027-06-01"),
    )
    assert stats["settled"] == 0
    assert stats["pending"] == 1
    assert str(updated.loc[0, "actual_outcome"]).strip() == ""


def test_future_fixture_stays_pending(raw_matches):
    record = make_record(match_date="2030-01-01", predicted_at="2026-11-05")
    updated, stats = reconcile_results(
        normalize_history(pd.DataFrame([record])),
        raw_matches,
        as_of=pd.Timestamp("2027-06-01"),
    )
    assert stats["pending"] == 1
    assert str(updated.loc[0, "actual_outcome"]).strip() == ""


def test_fixture_parser_supports_odds_and_date():
    fixture = parse_fixture_line(
        "Girona vs Betis | 2.10 3.40 3.60 | 2026-11-08"
    )
    assert fixture["home_team"] == "Girona"
    assert fixture["away_team"] == "Betis"
    assert fixture["odds"] == (2.10, 3.40, 3.60)
    assert fixture["match_date"] == "2026-11-08"

    with pytest.raises(ValueError):
        parse_fixture_line("Girona - Betis")


def test_predict_fixtures_probabilities_are_valid_and_labels_consistent(model_frame, raw_matches):
    X, y, _meta = model_frame
    pre = FeaturePreprocessor(FEATURE_COLUMNS)
    X_scaled = pre.fit_transform(X)
    model = RandomForestClassifier(
        n_estimators=50, max_depth=4, random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X_scaled, y)

    predictions, errors = predict_fixtures(
        [{"home_team": "atletico madrid", "away_team": "Real Betis",
          "match_date": "", "odds": None}],
        model=model, preprocessor=pre, metadata={}, raw=raw_matches,
        as_of=pd.Timestamp("2027-06-01"),
    )
    assert not errors
    assert len(predictions) == 1
    prediction = predictions[0]
    probs = prediction["probs"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in probs.values())
    expected_label = PREDICTION_LABELS[
        int(np.argmax([probs["home"], probs["draw"], probs["away"]]))
    ]
    assert prediction["predicted_outcome"] == expected_label
    assert prediction["home_team"] == "Ath Madrid"
    assert prediction["away_team"] == "Betis"
    assert prediction["fixture_id"].startswith("TBD_")


def test_predict_fixtures_reports_unknown_team(model_frame, raw_matches):
    X, y, _meta = model_frame
    pre = FeaturePreprocessor(FEATURE_COLUMNS)
    X_scaled = pre.fit_transform(X)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_scaled, y)
    predictions, errors = predict_fixtures(
        [{"home_team": "Fake FC", "away_team": "Real Madrid"}],
        model=model, preprocessor=pre, metadata={}, raw=raw_matches,
        as_of=pd.Timestamp("2027-06-01"),
    )
    assert predictions == []
    assert errors and "Unknown team" in errors[0]["error"]


def test_team_form_excludes_future_matches(raw_matches):
    history = build_team_history(raw_matches)
    team = "Real Madrid"
    cutoff = pd.Timestamp("2025-01-01")
    form = team_form(history, team, as_of=cutoff)
    rows = history[
        (history["team"] == team) & (history["date"] < cutoff)
    ].sort_values("date").tail(5)
    assert form["form_goals_scored"] == pytest.approx(rows["goals_scored"].mean())


def test_financial_audit_counts_value_bets(raw_matches):
    from backfill_results import STAKE, audit_ledger

    record = make_record(
        prob_home=0.60, prob_draw=0.20, prob_away=0.20,
        market_probs={"home": 0.45, "draw": 0.28, "away": 0.27},
    )
    settled = normalize_history(pd.DataFrame([record]))
    settled.loc[0, "actual_outcome"] = "2"
    settled.loc[0, "correct"] = "1"
    result = audit_ledger(settled, raw_matches)
    assert result["bets"] == 1
    assert result["wins"] == 1
    assert result["staked"] == STAKE
    assert result["pnl"] > 0


def test_history_performance_reports(raw_matches, tiny_history):
    record = normalize_history(pd.DataFrame([tiny_history]))
    settled, stats = reconcile_results(
        record, raw_matches, as_of=pd.Timestamp("2027-06-01")
    )
    assert stats["settled"] == 1
    summary = performance_summary(settled)
    assert summary["settled"] == 1
    assert 0 <= summary["accuracy"] <= 1

    for dimension in ("season", "outcome", "team", "model", "time"):
        report = performance_by(settled, dimension)
        assert not report.empty
