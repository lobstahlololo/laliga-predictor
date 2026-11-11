"""Full-season evaluation: chronological protocol, intervals and breakdowns."""

from __future__ import annotations

import numpy as np

from src.evaluation.full_season import evaluate_full_season
from src.utils.config import TEST_SEASON


EXPECTED_MODELS = {
    "Market odds", "Random Forest", "XGBoost", "Dynamic Elo",
    "Independent Poisson", "Dixon-Coles", "Ensemble (equal)",
    "Ensemble (validation)",
}


def test_full_season_evaluation_compares_every_model(raw_matches):
    result = evaluate_full_season(raw_matches, n_boot=40, save=False)

    assert EXPECTED_MODELS <= set(result.validation["model"])
    assert EXPECTED_MODELS <= set(result.test["model"])
    assert result.summary["protocol"]["test_used_for_selection"] is False
    assert result.summary["protocol"]["ensemble_weights_selected_on"] == "validation only"

    test = result.test.set_index("model")
    expected_rows = int((raw_matches["season"] == TEST_SEASON).sum())
    assert int(test.loc["Market odds", "n_samples"]) == expected_rows
    for model in EXPECTED_MODELS:
        low = float(test.loc[model, "log_loss_ci_low"])
        high = float(test.loc[model, "log_loss_ci_high"])
        point = float(test.loc[model, "log_loss"])
        assert low <= point <= high
        assert 0 <= float(test.loc[model, "accuracy_ci_low"]) <= 1
        assert 0 <= float(test.loc[model, "accuracy_ci_high"]) <= 1

    # Selection is validation-only, and market odds are never selected as the
    # learned model.
    selected = result.summary["validation_best_learned_model"]
    assert selected is not None
    assert selected["model"] != "Market odds"
    assert selected["model"] in EXPECTED_MODELS

    weights = result.summary["ensemble"]["weights"]
    assert set(weights) == {
        "Random Forest", "XGBoost", "Dynamic Elo", "Independent Poisson"
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert min(weights.values()) >= 0


def test_full_season_walk_forward_excludes_the_final_test_season(raw_matches):
    result = evaluate_full_season(raw_matches, n_boot=20, save=False)
    assert not result.walk_forward.empty
    assert TEST_SEASON not in set(result.walk_forward["test_season"])
    assert "Ensemble (equal)" in set(result.walk_forward["model"])
    assert result.walk_forward["fold"].nunique() >= 1


def test_breakdowns_refuse_small_samples(raw_matches):
    result = evaluate_full_season(raw_matches, n_boot=10, save=False)
    assert not result.breakdowns.empty
    assert {"season", "team", "predicted_side", "confidence"} <= set(
        result.breakdowns["dimension"]
    )
    small = result.breakdowns[result.breakdowns["status"] == "insufficient_sample"]
    assert not small.empty
    assert small["log_loss"].isna().all()
    # Team breakdowns are smaller than the declared minimum on this synthetic
    # one-round-robin data, so they must not carry metric values.
    team_rows = result.breakdowns[result.breakdowns["dimension"] == "team"]
    assert not team_rows.empty
    assert (team_rows["status"] == "insufficient_sample").all()
    # Season breakdowns do meet the minimum and carry metrics.
    eligible = result.breakdowns[
        (result.breakdowns["status"] == "ok")
        & (result.breakdowns["dimension"] == "season")
    ]
    assert not eligible.empty
    assert eligible["n"].min() >= result.summary["protocol"]["min_breakdown_samples"]
