"""Feature engineering and leakage-prevention tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_engineered_columns,
    build_feature_matrix,
    build_model_frame,
    build_team_history,
    compute_team_form,
    short_rest_flag,
)
from src.features.preprocessing import FeaturePreprocessor
from src.utils.config import FEATURE_COLUMNS, ROLLING_WINDOW


def test_feature_count_and_order(model_frame):
    X, y, meta = model_frame
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape[1] == 19
    assert len(X) == len(y) == len(meta)


def test_rolling_form_matches_manual_prior_matches(raw_matches):
    prepared = add_engineered_columns(compute_team_form(raw_matches))
    long = build_team_history(raw_matches).sort_values(["team", "date"]).reset_index(drop=True)
    team = "Real Madrid"
    team_rows = long[long["team"] == team].reset_index(drop=True)
    assert len(team_rows) > ROLLING_WINDOW + 1

    position = 8
    row = team_rows.iloc[position]
    prior = team_rows.iloc[max(0, position - ROLLING_WINDOW):position]
    expected_goals = prior["goals_scored"].mean()
    expected_conceded = prior["goals_conceded"].mean()

    match = prepared[prepared["match_id"] == row["match_id"]].iloc[0]
    side = row["side"]
    assert match[f"{side}_form_goals_scored"] == pytest.approx(expected_goals)
    assert match[f"{side}_form_goals_conceded"] == pytest.approx(expected_conceded)


def test_future_matches_do_not_change_past_features(raw_matches):
    cutoff = pd.Timestamp("2025-01-01")
    truncated = raw_matches[pd.to_datetime(raw_matches["date"]) < cutoff]

    full_X, _full_y, full_meta = build_model_frame(raw_matches)
    cut_X, _cut_y, cut_meta = build_model_frame(truncated)
    common = sorted(set(full_meta["match_id"]) & set(cut_meta["match_id"]))
    assert common, "Synthetic data should share matches before the cutoff."

    full_lookup = full_X.set_index(full_meta["match_id"]).loc[common]
    cut_lookup = cut_X.set_index(cut_meta["match_id"]).loc[common]
    pd.testing.assert_frame_equal(full_lookup, cut_lookup, check_dtype=False)


def test_goal_differential_derived(raw_matches):
    prepared = add_engineered_columns(compute_team_form(raw_matches))
    expected = prepared["home_form_goals_scored"] - prepared["home_form_goals_conceded"]
    pd.testing.assert_series_equal(
        prepared["home_goal_diff_5m"], expected, check_names=False
    )


def test_short_rest_flag_scalar_and_series():
    assert short_rest_flag(5, 3) == 1
    assert short_rest_flag(4, 3) == 0
    assert short_rest_flag(6, 4) == 0
    assert short_rest_flag(np.nan, 3) == 0
    series = short_rest_flag(
        pd.Series([5, 4, 6]), pd.Series([3, 3, 2])
    )
    assert series.tolist() == [1, 0, 1]


def test_preprocessing_imputes_from_training_only(raw_matches):
    X, _y, _meta = build_model_frame(raw_matches)
    train = X.iloc[:60].copy()
    train.loc[train.index[:3], "home_form_goals_scored"] = np.nan
    held_out = X.iloc[60:70].copy()
    held_out.loc[held_out.index[0], "home_form_goals_scored"] = np.nan

    pre = FeaturePreprocessor(FEATURE_COLUMNS)
    pre.fit(train)
    train_median = train["home_form_goals_scored"].median()
    assert pre.medians["home_form_goals_scored"] == pytest.approx(train_median)

    transformed = pre.transform(held_out)
    assert np.isfinite(transformed.to_numpy()).all()

    # Changing held-out values cannot move the learned median or scaler.
    held_out.loc[held_out.index[1], "home_form_goals_scored"] = 999.0
    pre.transform(held_out)
    assert pre.medians["home_form_goals_scored"] == pytest.approx(train_median)

    # Training features are standardized with ~zero mean by construction.
    fitted = pre.transform(train)
    assert abs(float(fitted["home_form_goals_scored"].mean())) < 1e-8


def test_preprocessor_requires_columns(raw_matches):
    X, _y, _meta = build_model_frame(raw_matches)
    pre = FeaturePreprocessor(FEATURE_COLUMNS)
    pre.fit(X.iloc[:50])
    with pytest.raises(ValueError, match="Missing required feature"):
        pre.transform(X.iloc[50:60].drop(columns=["home_advantage"]))
