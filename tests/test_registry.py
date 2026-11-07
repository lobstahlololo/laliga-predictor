"""Model registry: snapshots, evidence-gated promotion and rollback."""

from __future__ import annotations

import json

import pytest

from src.models import registry
from src.utils.config import FEATURE_COLUMNS


def _metadata(version: str, loss: float, brier: float) -> dict:
    return {
        "model_name": "Random Forest",
        "model_version": version,
        "trained_at": "2026-01-01T00:00:00+00:00",
        "feature_names": FEATURE_COLUMNS,
        "parameters": {"n_estimators": 10},
        "split": {
            "train_seasons": ["23/24", "24/25"],
            "validation_season": "25/26",
            "test_season": "26/27",
            "train_rows": 760,
            "validation_rows": 380,
            "test_rows": 109,
            "train_date_min": "2023-08-11",
            "train_date_max": "2025-05-25",
            "validation_date_max": "2026-05-24",
            "test_date_min": "2026-08-15",
            "test_date_max": "2026-11-02",
        },
        "validation_metrics": {"log_loss": loss, "brier_score": brier},
        "test_metrics": {"log_loss": loss + 0.05, "brier_score": brier + 0.01},
    }


def _setup(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model_v1 = artifacts / "model_v1.pkl"
    model_v2 = artifacts / "model_v2.pkl"
    model_v1.write_bytes(b"model-one")
    model_v2.write_bytes(b"model-two")
    pre = artifacts / "preprocessor.joblib"
    pre.write_bytes(b"preprocessor")
    return {
        "path": tmp_path / "registry.json",
        "versions": tmp_path / "versions",
        "model_v1": model_v1,
        "model_v2": model_v2,
        "preprocessor": pre,
        "live_model": tmp_path / "live" / "model.pkl",
        "live_meta": tmp_path / "live" / "model_metadata.json",
        "live_pre": tmp_path / "live" / "preprocessor.joblib",
    }


def _register(env, metadata, model_path):
    return registry.register_model(
        metadata,
        model_path=model_path,
        preprocessor_path=env["preprocessor"],
        path=env["path"],
        versions_dir=env["versions"],
    )


def _promote(env, version, reason="validation evidence"):
    return registry.promote(
        version,
        reason=reason,
        path=env["path"],
        model_target=env["live_model"],
        metadata_target=env["live_meta"],
        preprocessor_target=env["live_pre"],
    )


def test_first_registered_version_becomes_production_and_is_idempotent(tmp_path):
    env = _setup(tmp_path)
    entry, created = _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    assert created and entry["status"] == "production"
    assert env["live_model"].exists() is False  # registration never deploys

    snapshot = env["versions"] / "v1" / "model.pkl"
    assert snapshot.read_bytes() == b"model-one"
    assert entry["artifact_sha256"] == registry._sha256(snapshot)
    assert entry["training_period"]["seasons"] == ["23/24", "24/25"]

    again, created_again = _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    assert not created_again
    assert again["registered_at"] == entry["registered_at"]
    assert len(registry.load_registry(env["path"])["versions"]) == 1


def test_promotion_requires_better_validation_evidence(tmp_path):
    env = _setup(tmp_path)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    _register(env, _metadata("v2", 1.1, 0.55), env["model_v2"])

    with pytest.raises(registry.PromotionError, match="log-loss"):
        _promote(env, "v2")
    # A refused promotion must not touch the live artifact.
    assert not env["live_model"].exists()
    assert registry.status(env["path"])["production_version"] == "v1"


def test_promotion_requires_validation_metrics(tmp_path):
    env = _setup(tmp_path)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    metadata = _metadata("v2", 0.9, 0.55)
    metadata["validation_metrics"] = {}
    _register(env, metadata, env["model_v2"])

    with pytest.raises(registry.PromotionError, match="no validation"):
        _promote(env, "v2")


def test_promotion_swaps_artifacts_and_rollback_restores_them(tmp_path):
    env = _setup(tmp_path)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    _register(env, _metadata("v2", 0.9, 0.58), env["model_v2"])

    _promote(env, "v2")
    assert env["live_model"].read_bytes() == b"model-two"
    assert env["live_pre"].read_bytes() == b"preprocessor"
    live_meta = json.loads(env["live_meta"].read_text(encoding="utf-8"))
    assert live_meta["model_version"] == "v2"

    state = registry.status(env["path"])
    assert state["production_version"] == "v2"
    assert state["production"]["promotion"]["evidence"]["delta_validation_log_loss"] < 0

    restored = registry.rollback(
        reason="regression observed",
        path=env["path"],
        model_target=env["live_model"],
        metadata_target=env["live_meta"],
        preprocessor_target=env["live_pre"],
    )
    assert restored["version"] == "v1"
    assert env["live_model"].read_bytes() == b"model-one"
    assert registry.status(env["path"])["production_version"] == "v1"


def test_rollback_without_history_is_rejected(tmp_path):
    env = _setup(tmp_path)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    with pytest.raises(registry.RegistryError, match="No previous"):
        registry.rollback(
            reason="nothing to roll back",
            path=env["path"],
            model_target=env["live_model"],
            metadata_target=env["live_meta"],
            preprocessor_target=env["live_pre"],
        )


def test_verify_detects_tampered_snapshot(tmp_path):
    env = _setup(tmp_path)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    assert registry.verify("v1", path=env["path"])[0]["ok"]

    (env["versions"] / "v1" / "model.pkl").write_bytes(b"tampered")
    result = registry.verify("v1", path=env["path"])[0]
    assert not result["ok"]
    assert "hash mismatch" in " ".join(result["problems"])


def test_feature_set_hash_is_order_sensitive():
    assert registry.feature_set_hash(FEATURE_COLUMNS) == registry.feature_set_hash()
    assert registry.feature_set_hash(FEATURE_COLUMNS) != registry.feature_set_hash(
        list(reversed(FEATURE_COLUMNS))
    )
