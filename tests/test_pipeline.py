"""End-to-end training/selection pipeline tests (small synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.train_model import build_estimator, train_and_select
from src.utils.config import FEATURE_COLUMNS, PRIMARY_METRIC, TIEBREAK_METRIC


@pytest.fixture(scope="module")
def trained(model_frame):
    X, y, meta = model_frame
    return train_and_select(
        X, y, meta, n_search_iter=1, cv_folds=2, save=False
    )


def test_training_returns_both_models(trained):
    assert set(trained["models"]) == {"Random Forest", "XGBoost"}
    assert len(trained["validation_results"]) == 2
    assert len(trained["test_results"]) == 2
    assert trained["preprocessor"].fitted


def test_selection_uses_validation_only(trained):
    metadata = trained["metadata"]
    validation = trained["validation_results"]
    expected = min(
        validation,
        key=lambda m: (m[PRIMARY_METRIC], m[TIEBREAK_METRIC]),
    )["model"]
    assert metadata["model_name"] == expected
    assert metadata["selection"]["test_used_for_selection"] is False
    assert metadata["selection"]["metric"] == PRIMARY_METRIC


def test_test_predictions_are_valid_probabilities(trained):
    for metrics in trained["test_results"]:
        assert metrics["n_samples"] > 0
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["log_loss"] >= 0
        assert metrics["brier_score"] >= 0
        assert 0 <= metrics["draw_recall"] <= 1
        assert np.isfinite(metrics["ece"])
        assert metrics["confusion_matrix"].shape == (3, 3)


def test_metadata_contains_reproducibility_and_split_info(trained):
    metadata = trained["metadata"]
    assert metadata["random_state"] == 42
    assert metadata["feature_names"] == FEATURE_COLUMNS
    assert metadata["split"]["train_seasons"] == ["23/24", "24/25"]
    assert metadata["split"]["validation_season"] == "25/26"
    assert metadata["split"]["test_season"] == "26/27"
    assert metadata["split"]["train_rows"] > 0
    assert metadata["model_version"]


def test_random_forest_is_reproducible(model_frame):
    X, y, _meta = model_frame
    params = {"n_estimators": 30, "max_depth": 4}
    first = build_estimator("Random Forest", params).fit(X, y)
    second = build_estimator("Random Forest", params).fit(X, y)
    np.testing.assert_allclose(
        first.predict_proba(X), second.predict_proba(X)
    )


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_estimator("Magic Net")


def test_training_writes_a_candidate_not_production(monkeypatch, tmp_path, model_frame):
    """A new training run must not replace the live production model."""
    import src.models.train_model as tm
    from src.models import registry

    saved_dir = tmp_path / "saved_models"
    eval_dir = tmp_path / "evaluation"
    monkeypatch.setattr(tm, "SAVED_MODEL_DIR", saved_dir)
    monkeypatch.setattr(tm, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(tm, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tm, "MODEL_COMPARISON_PATH", eval_dir / "comparison.csv")
    monkeypatch.setattr(tm, "SEASON_METRICS_PATH", eval_dir / "seasons.csv")
    monkeypatch.setattr(tm, "WALK_FORWARD_PATH", eval_dir / "wf.csv")
    monkeypatch.setattr(tm, "EVALUATION_REPORT_PATH", eval_dir / "summary.json")
    monkeypatch.setattr(tm, "MODEL_METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(tm, "BEST_MODEL_PATH", tmp_path / "best_model.pkl")
    monkeypatch.setattr(tm, "save_feature_reference", lambda *a, **k: None)
    for plot_name in (
        "plot_calibration_curves", "plot_confusion_matrix", "plot_model_comparison",
        "plot_per_class_metrics", "plot_season_performance", "plot_walk_forward",
    ):
        monkeypatch.setattr(tm, plot_name, lambda *a, **k: None)

    # Keep the regression test fast: this test checks the save/promotion path,
    # not model quality, so a tiny search space is sufficient.
    monkeypatch.setattr(tm, "build_search_spaces", lambda: {
        "Random Forest": (
            tm.build_estimator("Random Forest"),
            {"n_estimators": [20], "max_depth": [4]},
        ),
        "XGBoost": (
            tm.build_estimator("XGBoost"),
            {"n_estimators": [20], "max_depth": [3]},
        ),
    })

    calls = {}

    def fake_register(metadata, **kwargs):
        calls["metadata"] = metadata
        calls["model_path"] = kwargs.get("model_path")
        entry = {"version": metadata["model_version"], "status": "candidate"}
        return entry, True

    monkeypatch.setattr(registry, "register_model", fake_register)
    monkeypatch.setattr(registry, "current_production", lambda: {"version": "prod-v0"})
    monkeypatch.setattr(
        registry, "restore_artifacts",
        lambda *a, **k: pytest.fail("training must not deploy over production"),
    )

    X, y, meta = model_frame
    result = tm.train_and_select(X, y, meta, n_search_iter=1, cv_folds=2, save=True)

    version = result["metadata"]["model_version"]
    assert calls["metadata"]["model_version"] == version
    candidate = saved_dir / f"candidate_{version}.pkl"
    assert candidate.exists()
    assert calls["model_path"] == candidate
    assert not (tmp_path / "best_model.pkl").exists()
    assert result["metadata"]["promotion_status"] == "candidate"
    assert result["metadata"]["production_version"] == "prod-v0"
