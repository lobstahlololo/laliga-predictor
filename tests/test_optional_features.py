"""Optional xG/availability groups: leakage guards and honest missing data."""

from __future__ import annotations

import pytest
import pandas as pd

from src.experiments import feature_ablation
from src.features import optional_features as opt
from src.features.build_features import build_model_frame
from src.features.preprocessing import FeaturePreprocessor
from src.utils.config import FEATURE_COLUMNS


def _xg_frame() -> pd.DataFrame:
    return pd.DataFrame([
        # Available before the 2024-01-10 fixture.
        {"match_id": "m0", "team": "Real Madrid", "date": "2024-01-05",
         "available_at": "2024-01-05T23:00:00Z", "xg_for": 2.0, "xg_against": 0.5},
        {"match_id": "m0", "team": "Barcelona", "date": "2024-01-05",
         "available_at": "2024-01-05T23:00:00Z", "xg_for": 1.0, "xg_against": 1.0},
        # Same-day xG must never leak into that day's pre-match features.
        {"match_id": "m1", "team": "Real Madrid", "date": "2024-01-10",
         "available_at": "2024-01-10T23:00:00Z", "xg_for": 9.0, "xg_against": 9.0},
        # Extreme future observation.
        {"match_id": "m2", "team": "Real Madrid", "date": "2024-06-01",
         "available_at": "2024-06-01T23:00:00Z", "xg_for": 100.0, "xg_against": 100.0},
    ])


def _row(match_id, date, home, away, outcome) -> dict:
    return {
        "match_id": match_id, "date": pd.Timestamp(date),
        "home_team": home, "away_team": away,
        "match_outcome": outcome,
        "full_time_home_goals": 2, "full_time_away_goals": 1,
        "home_shots": 12, "away_shots": 9,
        "home_shots_on_target": 5, "away_shots_on_target": 3,
        "odds_home": 2.0, "odds_draw": 3.4, "odds_away": 3.8,
        "implied_prob_home": 0.5, "implied_prob_draw": 0.294,
        "implied_prob_away": 0.263,
        "season": "23/24",
    }


def _raw_matches() -> pd.DataFrame:
    return pd.DataFrame([
        _row("m1", "2024-01-10", "Real Madrid", "Barcelona", 2),
        _row("m2", "2024-01-20", "Barcelona", "Real Madrid", 1),
    ])


def _availability_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"team": "Real Madrid", "player": "A", "status": "available",
         "available_at": "2024-01-08T12:00:00Z", "is_starter": True},
        {"team": "Real Madrid", "player": "B", "status": "injured",
         "available_at": "2024-01-08T12:00:00Z", "is_starter": True},
        {"team": "Real Madrid", "player": "C", "status": "available",
         "available_at": "2024-01-08T12:00:00Z", "is_starter": False},
        {"team": "Barcelona", "player": "D", "status": "available",
         "available_at": "2024-01-08T12:00:00Z", "is_starter": True},
        # Published after kickoff: must be ignored for the 2024-01-10 fixture.
        {"team": "Real Madrid", "player": "A", "status": "suspended",
         "available_at": "2024-01-10T19:00:00Z", "is_starter": True},
    ])


def test_xg_features_use_only_pre_kickoff_observations():
    raw = _raw_matches()
    features = opt.build_xg_features(raw, xg=_xg_frame())
    first = features.loc["m1"]
    assert first["home_xg_for_5"] == pytest.approx(2.0)
    assert first["home_xg_against_5"] == pytest.approx(0.5)
    assert first["home_xg_observations"] == pytest.approx(1.0)
    assert first["away_xg_for_5"] == pytest.approx(1.0)

    changed = pd.concat([_xg_frame(), pd.DataFrame([{
        "match_id": "m3", "team": "Real Madrid", "date": "2024-02-01",
        "available_at": "2024-02-01T23:00:00Z", "xg_for": 50.0, "xg_against": 50.0,
    }])], ignore_index=True)
    again = opt.build_xg_features(raw, xg=changed)
    pd.testing.assert_series_equal(again.loc["m1"], features.loc["m1"])


def test_xg_history_without_available_at_is_treated_as_next_day(tmp_path):
    path = tmp_path / "xg.csv"
    pd.DataFrame([{
        "match_id": "m0", "team": "Real Madrid", "date": "2024-01-05",
        "xg_for": 1.5, "xg_against": 0.5,
    }]).to_csv(path, index=False)

    loaded = opt.load_xg_history(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["available_at"] == pd.Timestamp("2024-01-06")

    features = opt.build_xg_features(_raw_matches(), xg=loaded)
    assert features.loc["m1", "home_xg_for_5"] == pytest.approx(1.5)


def test_availability_ignores_post_kickoff_snapshots():
    features = opt.build_availability_features(
        _raw_matches(), availability=_availability_frame()
    )
    first = features.loc["m1"]
    assert first["home_availability_score"] == pytest.approx(0.5)
    assert first["home_missing_starters"] == pytest.approx(1.0)
    assert first["away_availability_score"] == pytest.approx(1.0)
    assert first["availability_diff"] == pytest.approx(-0.5)
    assert first["home_availability_known"] == pytest.approx(1.0)


def test_availability_without_snapshot_is_nan_and_flagged_unknown():
    before_any_snapshot = pd.DataFrame([
        _row("m1", "2024-01-01", "Real Madrid", "Barcelona", 2)
    ])
    features = opt.build_availability_features(
        before_any_snapshot, availability=_availability_frame()
    )
    row = features.loc["m1"]
    assert pd.isna(row["home_availability_score"])
    assert row["home_availability_known"] == 0.0
    assert pd.isna(row["availability_diff"])


def test_fixture_optional_features_are_cut_at_kickoff():
    values = opt.fixture_optional_features(
        _raw_matches(), "Real Madrid", "Barcelona", pd.Timestamp("2024-01-10"),
        opt.AVAILABILITY_FEATURE_COLUMNS,
        availability=_availability_frame(),
    )
    assert values["home_missing_starters"] == pytest.approx(1.0)
    assert values["home_availability_score"] == pytest.approx(0.5)


def test_missing_optional_sources_are_reported_not_fabricated(monkeypatch, tmp_path):
    monkeypatch.setattr(opt, "XG_HISTORY_PATH", tmp_path / "missing_xg.csv")
    monkeypatch.setattr(opt, "PLAYER_AVAILABILITY_PATH", tmp_path / "missing_availability.csv")

    for group in ("xg", "player_availability"):
        status = opt.feature_group_status(group)
        assert status.available is False
        assert "no timestamped" in status.reason

    with pytest.raises(opt.OptionalFeatureUnavailable):
        opt.resolve_feature_columns(["baseline", "xg"])

    with pytest.raises(opt.OptionalFeatureUnavailable):
        build_model_frame(_raw_matches(), feature_groups=["baseline", "player_availability"])


def test_baseline_frame_keeps_the_19_feature_contract(raw_matches):
    X, y, meta = build_model_frame(raw_matches)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[1] == 19
    assert len(X) == len(y) == len(meta)


def test_optional_groups_are_fitted_on_training_only(monkeypatch, tmp_path):
    path = tmp_path / "xg.csv"
    _xg_frame().to_csv(path, index=False)
    monkeypatch.setattr(opt, "XG_HISTORY_PATH", path)
    raw = pd.DataFrame([
        _row("a", "2024-01-10", "Real Madrid", "Barcelona", 2),
        _row("b", "2024-01-20", "Barcelona", "Real Madrid", 0),
    ])
    X, _y, _meta = build_model_frame(raw, feature_groups=["baseline", "xg"])
    columns, statuses = opt.resolve_feature_columns(["baseline", "xg"])
    assert list(X.columns) == columns
    assert [s.name for s in statuses] == ["baseline", "xg"]
    assert X.shape[1] == 19 + len(opt.XG_FEATURE_COLUMNS)

    pre = FeaturePreprocessor(feature_names=list(X.columns)).fit(X.iloc[:1])
    transformed = pre.transform(X.iloc[1:])
    assert transformed.shape[1] == X.shape[1]
    assert transformed.notna().all().all()


def test_ablation_reports_unavailable_groups_without_training_them(monkeypatch):
    calls = {"n": 0}

    def fake_evaluate(raw, groups, *, label):
        calls["n"] += 1
        return {
            "group": label, "status": "evaluated", "n_features": 19,
            "n_train": 76, "n_validation": 36,
            "accuracy": 0.4, "log_loss": 1.05, "brier_score": 0.62,
            "macro_f1": 0.35,
        }

    monkeypatch.setattr(feature_ablation, "_evaluate", fake_evaluate)
    table, summary = feature_ablation.run_ablation(pd.DataFrame())
    assert calls["n"] == 1
    assert summary["test_season_used"] is False
    assert summary["production_model_unchanged"] is True
    assert summary["retained_groups"] == []
    assert set(table["status"]) <= {"evaluated", "unavailable"}
