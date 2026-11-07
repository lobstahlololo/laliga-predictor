"""Chronological splitting and walk-forward tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.backtesting import balanced_sample_weights
from src.evaluation.splits import (
    chronological_split,
    season_train_test,
    walk_forward_folds,
)
from src.features.preprocessing import FeaturePreprocessor
from src.utils.config import FEATURE_COLUMNS


def test_chronological_split_is_disjoint_and_ordered(model_frame):
    _X, _y, meta = model_frame
    split = chronological_split(
        meta,
        train_seasons=["23/24", "24/25"],
        validation_season="25/26",
        test_season="26/27",
    )
    train, val, test = split["train"], split["validation"], split["test"]

    assert set(train["season"]) == {"23/24", "24/25"}
    assert set(val["season"]) == {"25/26"}
    assert set(test["season"]) == {"26/27"}
    assert set(train.index) & set(val.index) == set()
    assert set(train.index) & set(test.index) == set()
    assert set(val.index) & set(test.index) == set()

    assert pd.to_datetime(train["date"]).max() < pd.to_datetime(val["date"]).min()
    assert pd.to_datetime(val["date"]).max() < pd.to_datetime(test["date"]).min()


def test_chronological_split_rejects_non_chronological_config(model_frame):
    _X, _y, meta = model_frame
    with pytest.raises(ValueError, match="precede"):
        chronological_split(
            meta,
            train_seasons=["25/26"],
            validation_season="24/25",
            test_season="26/27",
        )
    with pytest.raises(ValueError, match="not present"):
        chronological_split(
            meta,
            train_seasons=["23/24"],
            validation_season="24/25",
            test_season="99/00",
        )


def test_walk_forward_folds_only_use_earlier_seasons(model_frame):
    _X, _y, meta = model_frame
    folds = walk_forward_folds(meta, min_train_seasons=2)
    assert folds
    for fold in folds:
        train = meta.loc[fold.train_index]
        test = meta.loc[fold.test_index]
        assert all(season < fold.test_season for season in fold.train_seasons)
        assert pd.to_datetime(train["date"]).max() < pd.to_datetime(test["date"]).min()
        assert set(train["season"]) & set(test["season"]) == set()


def test_season_train_test_rejects_future_training(model_frame):
    _X, _y, meta = model_frame
    with pytest.raises(ValueError, match="precede"):
        season_train_test(meta, ["24/25"], "23/24")

    train, test = season_train_test(meta, ["23/24"], "24/25")
    assert train["date"].max() < test["date"].min()


def test_preprocessor_fit_on_fold_train_only(model_frame):
    X, _y, _meta = model_frame
    train = X.iloc[:60]
    test = X.iloc[60:80]
    pre = FeaturePreprocessor(FEATURE_COLUMNS)
    pre.fit(train)
    medians = dict(pre.medians)

    # Transforming test data never changes learned statistics.
    pre.transform(test)
    assert pre.medians == medians
    assert pre.scaler.mean_ is not None


def test_balanced_sample_weights_are_inverse_frequency():
    y = pd.Series([0, 0, 0, 1, 1, 2])
    weights = balanced_sample_weights(y)
    assert len(weights) == len(y)
    # Rare class 2 gets a larger weight than majority class 0.
    assert weights[5] > weights[0]
    assert weights[3] == pytest.approx(weights[4])
