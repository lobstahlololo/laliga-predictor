"""Metric correctness tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    calibration_table,
    compare_models,
    evaluate_probabilities,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
)


def test_log_loss_matches_manual_calculation():
    y = [2, 1, 0, 2]
    probs = np.array([
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],
        [0.5, 0.3, 0.2],
    ])
    expected = -np.mean(np.log([0.6, 0.6, 0.6, 0.5]))
    assert multiclass_log_loss(y, probs, [2, 1, 0]) == pytest.approx(expected)


def test_log_loss_accepts_sorted_class_order():
    y = [2, 1, 0]
    # Columns ordered [away, draw, home].
    probs = np.array([
        [0.2, 0.2, 0.6],
        [0.2, 0.6, 0.2],
        [0.6, 0.2, 0.2],
    ])
    assert multiclass_log_loss(y, probs, [0, 1, 2]) == pytest.approx(
        -np.mean(np.log([0.6, 0.6, 0.6]))
    )


def test_multiclass_brier_score_perfect_forecast():
    y = [0, 1, 2]
    probs = np.eye(3)
    assert multiclass_brier_score(y, probs, [0, 1, 2]) == pytest.approx(0.0)


def test_evaluate_probabilities_metrics_and_confusion():
    y = [2, 2, 1, 0]
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ])
    metrics = evaluate_probabilities(y, probs, [2, 1, 0], "test")
    assert metrics["accuracy"] == 1.0
    assert metrics["draw_recall"] == 1.0
    assert metrics["confusion_matrix"].shape == (3, 3)
    assert metrics["confusion_matrix"].values.trace() == 4
    assert metrics["log_loss"] < 0.6
    assert set(metrics["prediction_distribution"]) == {"home", "draw", "away"}


def test_draw_recall_measures_missed_draws():
    y = [1, 1, 2, 0]
    probs = np.array([
        [0.5, 0.2, 0.3],
        [0.5, 0.2, 0.3],
        [0.6, 0.2, 0.2],
        [0.2, 0.2, 0.6],
    ])
    metrics = evaluate_probabilities(y, probs, [2, 1, 0], "no-draws")
    assert metrics["draw_recall"] == 0.0


def test_calibration_is_perfect_for_one_hot_forecasts():
    y = [0, 1, 2, 0]
    probs = np.eye(3)[[0, 1, 2, 0]]
    ece = expected_calibration_error(y, probs, [0, 1, 2])
    assert ece == pytest.approx(0.0, abs=1e-9)
    table = calibration_table(y, probs, [0, 1, 2], n_bins=5)
    # One-vs-rest table: each row contributes once per class.
    assert table["count"].sum() == len(y) * 3


def test_compare_models_sorted_by_log_loss_then_brier():
    y = [2, 1, 0]
    good = evaluate_probabilities(y, np.eye(3), [2, 1, 0], "good")
    bad = evaluate_probabilities(y, np.ones((3, 3)) / 3, [2, 1, 0], "bad")
    table = compare_models([bad, good])
    assert table.iloc[0]["model"] == "good"
    assert table.iloc[0]["log_loss"] <= table.iloc[1]["log_loss"]


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError):
        evaluate_probabilities([0, 1], np.array([[1.0, 0.0], [0.0, 1.0]]), [0, 1, 2], "x")
    with pytest.raises(ValueError):
        multiclass_log_loss([0], np.array([[np.nan, 0.0, 1.0]]), [0, 1, 2])
