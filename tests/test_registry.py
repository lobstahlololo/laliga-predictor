"""Model registry: snapshots, evidence-gated promotion and rollback."""

from __future__ import annotations

import json
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Cross-platform paths
# ---------------------------------------------------------------------------


def _legacy_windows_registry(path: Path) -> dict:
    """Rewrite a registry on disk the way an old Windows run stored it."""
    registry_data = json.loads(path.read_text(encoding="utf-8"))
    for entry in registry_data["versions"].values():
        entry.pop("preprocessor_sha256", None)
        for field in registry.REGISTRY_PATH_FIELDS:
            entry[field] = entry[field].replace("/", "\\")
    path.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")
    return registry_data


def _root_relative_env(tmp_path, monkeypatch):
    """Treat ``tmp_path`` as the repository root so paths stay relative."""
    monkeypatch.setattr(registry, "PROJECT_ROOT", tmp_path)
    return _setup(tmp_path)


def test_windows_style_paths_normalise_to_portable_form():
    stored = "models\\registry\\versions\\v1\\model.pkl"
    assert registry.normalize_stored_path(stored) == "models/registry/versions/v1/model.pkl"
    assert registry.normalize_stored_path("models/registry/./versions//v1/model.pkl") == (
        "models/registry/versions/v1/model.pkl"
    )
    assert registry.normalize_stored_path("C:\\models\\registry") == "C:/models/registry"


def test_relative_paths_resolve_from_project_root_not_cwd(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    monkeypatch.setattr(registry, "PROJECT_ROOT", repo_root)
    monkeypatch.chdir(tmp_path)  # an unrelated working directory

    resolved = registry.resolve_stored_path("models\\registry\\versions\\v1\\model.pkl")
    assert resolved == repo_root / "models" / "registry" / "versions" / "v1" / "model.pkl"
    # The stored portion must stay POSIX: no component may carry a Windows
    # separator, whichever host resolves it.
    assert resolved.relative_to(repo_root).parts == (
        "models",
        "registry",
        "versions",
        "v1",
        "model.pkl",
    )
    assert resolved.as_posix().endswith("models/registry/versions/v1/model.pkl")


def test_registered_paths_are_portable_and_root_relative(tmp_path, monkeypatch):
    env = _root_relative_env(tmp_path, monkeypatch)
    entry, _ = _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])

    assert entry["artifact"] == "versions/v1/model.pkl"
    assert entry["preprocessor"] == "versions/v1/preprocessor.joblib"
    assert entry["metadata_snapshot"] == "versions/v1/metadata.json"
    for field in registry.REGISTRY_PATH_FIELDS:
        assert "\\" not in entry[field]
    assert entry["preprocessor_sha256"] == registry._sha256(
        env["versions"] / "v1" / "preprocessor.joblib"
    )


def test_legacy_windows_entries_still_verify_and_resolve(tmp_path, monkeypatch):
    env = _root_relative_env(tmp_path, monkeypatch)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    _legacy_windows_registry(env["path"])

    # Loading normalises the in-memory view, so every consumer sees POSIX paths.
    loaded = registry.load_registry(env["path"])
    assert loaded["versions"]["v1"]["artifact"] == "versions/v1/model.pkl"

    result = registry.verify("v1", path=env["path"])[0]
    assert result["ok"], result["problems"]

    resolved = registry.production_paths(env["path"])
    assert resolved["model"] == env["versions"] / "v1" / "model.pkl"
    assert resolved["preprocessor"] == env["versions"] / "v1" / "preprocessor.joblib"
    assert resolved["metadata"] == env["versions"] / "v1" / "metadata.json"


def test_migrate_normalises_windows_registry_and_backfills_hash(tmp_path, monkeypatch):
    env = _root_relative_env(tmp_path, monkeypatch)
    entry, _ = _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    _legacy_windows_registry(env["path"])

    report = registry.migrate_registry(env["path"])
    assert report["migrated_paths"] == ["v1"]
    assert report["backfilled_hashes"] == ["v1"]

    on_disk = json.loads(env["path"].read_text(encoding="utf-8"))["versions"]["v1"]
    assert on_disk["artifact"] == "versions/v1/model.pkl"
    assert on_disk["preprocessor"] == "versions/v1/preprocessor.joblib"
    assert on_disk["metadata_snapshot"] == "versions/v1/metadata.json"
    assert "\\" not in "".join(on_disk[field] for field in registry.REGISTRY_PATH_FIELDS)
    # Migration must never rewrite or re-hash the immutable model snapshot.
    assert on_disk["artifact_sha256"] == entry["artifact_sha256"]
    assert on_disk["preprocessor_sha256"] == registry._sha256(
        env["versions"] / "v1" / "preprocessor.joblib"
    )
    assert (env["versions"] / "v1" / "model.pkl").read_bytes() == b"model-one"

    again = registry.migrate_registry(env["path"])
    assert again["migrated_paths"] == []
    assert again["backfilled_hashes"] == []


def test_save_registry_writes_portable_paths(tmp_path, monkeypatch):
    env = _root_relative_env(tmp_path, monkeypatch)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    legacy = _legacy_windows_registry(env["path"])

    registry.save_registry(legacy, env["path"])

    assert "\\" not in env["path"].read_text(encoding="utf-8")
    assert registry.load_registry(env["path"])["versions"]["v1"]["artifact"] == (
        "versions/v1/model.pkl"
    )


def test_verify_detects_tampered_preprocessor_snapshot(tmp_path):
    env = _setup(tmp_path)
    entry, _ = _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])
    assert entry["preprocessor_sha256"]

    (env["versions"] / "v1" / "preprocessor.joblib").write_bytes(b"tampered")
    result = registry.verify("v1", path=env["path"])[0]
    assert not result["ok"]
    assert "preprocessor snapshot hash mismatch" in result["problems"]


def test_verify_detects_missing_model_artifact(tmp_path):
    env = _setup(tmp_path)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])

    (env["versions"] / "v1" / "model.pkl").unlink()
    result = registry.verify("v1", path=env["path"])[0]
    assert not result["ok"]
    assert any(problem.startswith("artifact missing:") for problem in result["problems"])
    assert "\\" not in " ".join(result["problems"])


def test_verify_passes_from_relocated_clean_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(registry, "PROJECT_ROOT", checkout)
    env = _setup(checkout)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])

    # A clean checkout has only the committed registry tree; the live training
    # artifacts it was built from are gone.
    env["model_v1"].unlink()
    env["preprocessor"].unlink()
    assert not env["model_v1"].exists()

    # The same tree checked out at a different absolute path (e.g. a runner).
    moved = tmp_path / "runner" / "work" / "laliga-predictor"
    moved.parent.mkdir(parents=True)
    checkout.rename(moved)
    monkeypatch.setattr(registry, "PROJECT_ROOT", moved)

    registry_path = moved / "registry.json"
    result = registry.verify("v1", path=registry_path)[0]
    assert result["ok"], result["problems"]

    resolved = registry.production_paths(registry_path)
    assert resolved["model"] == moved / "versions" / "v1" / "model.pkl"
    assert resolved["model"].is_file()
    assert resolved["preprocessor"].is_file()
    assert resolved["metadata"].is_file()


def test_verify_detects_missing_preprocessor_snapshot(tmp_path):
    env = _setup(tmp_path)
    _register(env, _metadata("v1", 1.0, 0.60), env["model_v1"])

    (env["versions"] / "v1" / "preprocessor.joblib").unlink()
    result = registry.verify("v1", path=env["path"])[0]
    assert not result["ok"]
    assert any("preprocessor snapshot missing" in p for p in result["problems"])
    # The reported path must be portable, not a host-specific string.
    assert "\\" not in " ".join(result["problems"])
