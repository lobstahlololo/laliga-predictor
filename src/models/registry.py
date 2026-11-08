"""Model registry with evidence-gated promotion and rollback.

Every trained model is registered as an immutable version: metadata, metrics,
training/validation/test periods, feature set hash and a snapshot of the model
artifact. Exactly one version is designated production.

Rules enforced here:

* Registering never changes the production model.
* Promotion requires measured validation evidence: the challenger's validation
  log-loss must be strictly better than the current production model's, and its
  validation Brier score must not be worse. Missing metrics means no promotion.
* Rollback restores a previous production snapshot and is always allowed; it is
  recorded in the registry history.
* Artifacts are copied, never moved, so a promotion/rollback can be undone from
  the registry snapshots.

Usage:
    python -m src.models.registry status
    python -m src.models.registry register
    python -m src.models.registry promote <version> --reason "validation evidence"
    python -m src.models.registry rollback --reason "regression observed"
    python -m src.models.registry verify
    python -m src.models.registry migrate

Registry files are committed to Git and read on Linux CI as well as Windows
machines, so every stored path is portable (forward slashes) and every relative
path is resolved against the repository root, never against the working
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocessing import DEFAULT_PREPROCESSOR_PATH  # noqa: E402
from src.utils.config import (  # noqa: E402
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    MODEL_METADATA_PATH,
    REGISTRY_PATH,
    REGISTRY_VERSIONS_DIR,
    SAVED_MODEL_DIR,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
STATUS_PRODUCTION = "production"
STATUS_CANDIDATE = "candidate"
STATUS_ARCHIVED = "archived"
STATUS_ROLLED_BACK = "rolled_back"


class RegistryError(RuntimeError):
    """Raised for invalid registry operations."""


class PromotionError(RegistryError):
    """Raised when validation evidence does not support a promotion."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Path fields persisted in the registry. They are stored in portable form.
REGISTRY_PATH_FIELDS = ("artifact", "preprocessor", "metadata_snapshot")
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:/")


def normalize_stored_path(value: str | Path) -> str:
    """Return a registry path in portable, forward-slash form.

    Windows separators are rewritten so an entry written on a developer machine
    still resolves on Linux CI (and vice versa). Drive letters and leading
    slashes are preserved; ``.`` and empty segments are dropped.
    """
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    if _DRIVE_PREFIX.match(text):
        prefix, remainder = f"{text[0].upper()}:/", text[3:]
    elif text.startswith("//"):  # UNC share
        prefix, remainder = "//", text[2:]
    elif text.startswith("/"):
        prefix, remainder = "/", text[1:]
    else:
        prefix, remainder = "", text
    parts = [part for part in remainder.split("/") if part not in ("", ".")]
    return prefix + "/".join(parts)


def resolve_stored_path(value: str | Path) -> Path:
    """Resolve a stored registry path against the repository root.

    Relative entries are anchored at ``PROJECT_ROOT`` rather than the current
    working directory, so ``python -m src.models.registry verify`` behaves the
    same whether it runs from the repo root, CI, or another directory.
    """
    text = normalize_stored_path(value)
    if not text:
        return Path()
    path = Path(text)
    if path.is_absolute() or _DRIVE_PREFIX.match(text):
        return path
    return PROJECT_ROOT / path


def normalize_registry_paths(registry: dict) -> list[str]:
    """Rewrite every registry path field to portable form in place.

    Returns the versions whose stored paths changed. Artifact bytes are never
    touched here, so immutable snapshots remain immutable.
    """
    changed_versions: list[str] = []
    versions = registry.get("versions")
    if not isinstance(versions, dict):
        return changed_versions
    for version, entry in versions.items():
        if not isinstance(entry, dict):
            continue
        changed = False
        for field in REGISTRY_PATH_FIELDS:
            value = entry.get(field)
            if isinstance(value, str) and value.strip():
                portable = normalize_stored_path(value)
                if portable != value:
                    entry[field] = portable
                    changed = True
        if changed:
            changed_versions.append(version)
    return changed_versions


def empty_registry() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "production": None,
        "production_since": None,
        "production_history": [],
        "versions": {},
    }


def _read_registry(path: Path) -> dict:
    """Read the registry exactly as stored, without normalising it."""
    if not Path(path).exists():
        return empty_registry()
    try:
        registry = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - corruption must be explicit
        raise RegistryError(f"Registry at {path} is unreadable: {exc}") from exc
    if not isinstance(registry, dict) or "versions" not in registry:
        raise RegistryError(f"Registry at {path} has an unexpected schema.")
    registry.setdefault("schema_version", SCHEMA_VERSION)
    registry.setdefault("production", None)
    registry.setdefault("production_since", None)
    registry.setdefault("production_history", [])
    return registry


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    """Load the registry, returning an empty structure when absent.

    Legacy entries written with Windows separators are normalised in memory so
    they resolve on every platform. A corrupt registry is an error: silently
    discarding version history would make rollback impossible.
    """
    registry = _read_registry(path)
    normalize_registry_paths(registry)
    return registry


def save_registry(registry: dict, path: Path = REGISTRY_PATH) -> None:
    """Atomically persist the registry with portable paths."""
    normalize_registry_paths(registry)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = _now()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backfill_preprocessor_hashes(registry: dict) -> list[str]:
    """Record a hash for snapshots registered before preprocessor hashing.

    Only the manifest changes; the snapshot file is never rewritten.
    """
    backfilled: list[str] = []
    versions = registry.get("versions")
    if not isinstance(versions, dict):
        return backfilled
    for version, entry in versions.items():
        if not isinstance(entry, dict) or entry.get("preprocessor_sha256"):
            continue
        stored = entry.get("preprocessor")
        if not stored:
            continue
        snapshot = resolve_stored_path(stored)
        if snapshot.is_file():
            entry["preprocessor_sha256"] = _sha256(snapshot)
            backfilled.append(version)
    return backfilled


def feature_set_hash(feature_names: list[str] | None = None) -> str:
    names = list(feature_names or FEATURE_COLUMNS)
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()[:16]


def _version_dir(version: str, base: Path = REGISTRY_VERSIONS_DIR) -> Path:
    safe = str(version).replace("/", "_").replace("\\", "_")
    return Path(base) / safe


def _stored_path(path: Path) -> str:
    """Store project-relative paths when possible, absolute otherwise.

    The result always uses forward slashes so the registry stays readable on
    every platform.
    """
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return normalize_stored_path(resolved)
    return normalize_stored_path(relative)


def _period_from_split(split: dict, key: str) -> dict:
    split = split or {}
    seasons_key = {
        "train": "train_seasons",
        "validation": "validation_season",
        "test": "test_season",
    }[key]
    return {
        "seasons": split.get(seasons_key),
        "rows": split.get(f"{key}_rows"),
        "date_min": split.get(f"{key}_date_min"),
        "date_max": split.get(f"{key}_date_max"),
    }


def register_model(
    metadata: dict,
    *,
    model_path: Path = BEST_MODEL_PATH,
    preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    path: Path = REGISTRY_PATH,
    versions_dir: Path = REGISTRY_VERSIONS_DIR,
) -> tuple[dict, bool]:
    """Register one model version. Returns ``(entry, created)``.

    Registration is idempotent: re-registering an existing version returns the
    stored entry unchanged and never overwrites its snapshot.
    """
    version = str(metadata.get("model_version") or "").strip()
    if not version:
        raise RegistryError("Metadata has no model_version; cannot register.")
    registry = load_registry(path)
    if version in registry["versions"]:
        return registry["versions"][version], False
    if not model_path.exists():
        raise RegistryError(f"Model artifact not found: {model_path}")

    snapshot = _version_dir(version, versions_dir)
    snapshot.mkdir(parents=True, exist_ok=True)
    artifact = snapshot / "model.pkl"
    shutil.copy2(model_path, artifact)
    preprocessor_snapshot = ""
    preprocessor_sha256 = ""
    if Path(preprocessor_path).exists():
        preprocessor_target = snapshot / "preprocessor.joblib"
        shutil.copy2(preprocessor_path, preprocessor_target)
        preprocessor_snapshot = _stored_path(preprocessor_target)
        preprocessor_sha256 = _sha256(preprocessor_target)
    (snapshot / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    split = metadata.get("split", {}) or {}
    feature_names = metadata.get("feature_names") or FEATURE_COLUMNS
    entry = {
        "version": version,
        "model_family": metadata.get("model_name", "unknown"),
        "created_at": metadata.get("trained_at") or _now(),
        "registered_at": _now(),
        "status": STATUS_CANDIDATE,
        "artifact": _stored_path(artifact),
        "artifact_sha256": _sha256(artifact),
        "preprocessor": preprocessor_snapshot,
        "preprocessor_sha256": preprocessor_sha256,
        "metadata_snapshot": _stored_path(snapshot / "metadata.json"),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "feature_set_hash": feature_set_hash(feature_names),
        "parameters": metadata.get("parameters", {}),
        "training_period": _period_from_split(split, "train"),
        "validation_period": _period_from_split(split, "validation"),
        "test_period": _period_from_split(split, "test"),
        "validation_metrics": metadata.get("validation_metrics", {}),
        "test_metrics": metadata.get("test_metrics", {}),
        "source": "training_artifacts",
        "promotion": None,
        "notes": [],
    }

    if not registry["production"]:
        entry["status"] = STATUS_PRODUCTION
        entry["promotion"] = {
            "promoted_at": _now(),
            "actor": "auto-bootstrap",
            "reason": "first registered version becomes the initial production model",
            "evidence": None,
        }
        registry["production"] = version
        registry["production_since"] = entry["promotion"]["promoted_at"]

    registry["versions"][version] = entry
    save_registry(registry, path)
    logger.info("Registered model version %s (%s)", version, entry["status"])
    return entry, True


def register_current_artifacts(path: Path = REGISTRY_PATH, **kwargs) -> tuple[dict, bool]:
    """Register the metadata written by the most recent training run."""
    """Register whatever ``train_model`` last wrote, without promoting it."""
    if not MODEL_METADATA_PATH.exists():
        raise RegistryError(
            f"Missing {MODEL_METADATA_PATH}. Run `python -m src.models.train_model`."
        )
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return register_model(metadata, path=path, **kwargs)


def get_model(version: str, path: Path = REGISTRY_PATH) -> dict:
    registry = load_registry(path)
    if version not in registry["versions"]:
        raise RegistryError(f"Unknown model version: {version}")
    return registry["versions"][version]


def list_models(path: Path = REGISTRY_PATH) -> pd.DataFrame:
    registry = load_registry(path)
    rows = []
    for version, entry in registry["versions"].items():
        validation = entry.get("validation_metrics", {}) or {}
        test = entry.get("test_metrics", {}) or {}
        rows.append({
            "version": version,
            "family": entry.get("model_family"),
            "status": entry.get("status"),
            "registered_at": entry.get("registered_at"),
            "validation_log_loss": validation.get("log_loss"),
            "validation_brier": validation.get("brier_score"),
            "test_log_loss": test.get("log_loss"),
            "test_brier": test.get("brier_score"),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("registered_at", ascending=False).reset_index(drop=True)
    return frame


def current_production(path: Path = REGISTRY_PATH) -> dict | None:
    registry = load_registry(path)
    version = registry.get("production")
    return registry["versions"].get(version) if version else None


def _validation_evidence(challenger: dict, production: dict | None) -> dict:
    """Check the promotion rule and return the evidence record."""
    challenger_val = challenger.get("validation_metrics") or {}
    challenger_loss = challenger_val.get("log_loss")
    challenger_brier = challenger_val.get("brier_score")
    if challenger_loss is None or challenger_brier is None:
        raise PromotionError(
            f"{challenger['version']} has no validation log-loss/Brier metrics; "
            "promotion requires measured validation evidence."
        )
    evidence = {
        "challenger_validation_log_loss": float(challenger_loss),
        "challenger_validation_brier": float(challenger_brier),
        "production_version": production.get("version") if production else None,
        "checked_at": _now(),
    }
    if production is None:
        evidence["rule"] = "no current production model; first version promoted"
        return evidence

    production_val = production.get("validation_metrics") or {}
    production_loss = production_val.get("log_loss")
    production_brier = production_val.get("brier_score")
    if production_loss is None or production_brier is None:
        raise PromotionError(
            "The current production model has no recorded validation metrics, so "
            "the challenger cannot be compared. Re-register the production model "
            "with its training metadata first."
        )
    evidence.update({
        "production_validation_log_loss": float(production_loss),
        "production_validation_brier": float(production_brier),
        "delta_validation_log_loss": float(challenger_loss - production_loss),
        "delta_validation_brier": float(challenger_brier - production_brier),
    })
    if not float(challenger_loss) < float(production_loss):
        raise PromotionError(
            "Challenger validation log-loss "
            f"({float(challenger_loss):.4f}) is not better than production "
            f"({float(production_loss):.4f})."
        )
    if float(challenger_brier) > float(production_brier):
        raise PromotionError(
            "Challenger validation Brier score "
            f"({float(challenger_brier):.4f}) is worse than production "
            f"({float(production_brier):.4f})."
        )
    evidence["rule"] = (
        "challenger validation log-loss strictly better AND Brier not worse; "
        "test metrics are reported but never used to decide promotion"
    )
    return evidence


def production_paths(path: Path = REGISTRY_PATH) -> dict:
    """Resolve the production snapshot paths for inference.

    Returns an empty dict when no production version is registered. Callers fall
    back to the legacy live paths so a fresh clone still works before the first
    registration.
    """
    entry = current_production(path)
    if entry is None:
        return {}
    resolved = {
        "version": entry["version"],
        "entry": entry,
        "model": resolve_stored_path(entry["artifact"]),
        "metadata": resolve_stored_path(entry["metadata_snapshot"]),
        "preprocessor": None,
    }
    if entry.get("preprocessor"):
        resolved["preprocessor"] = resolve_stored_path(entry["preprocessor"])
    return resolved


def restore_artifacts(
    version: str,
    *,
    path: Path = REGISTRY_PATH,
    model_target: Path = BEST_MODEL_PATH,
    metadata_target: Path = MODEL_METADATA_PATH,
    preprocessor_target: Path = DEFAULT_PREPROCESSOR_PATH,
) -> dict:
    """Copy a registered version's snapshot to the live inference paths."""
    entry = get_model(version, path)
    _restore_snapshot(
        entry,
        model_target=model_target,
        metadata_target=metadata_target,
        preprocessor_target=preprocessor_target,
    )
    return entry


def _restore_snapshot(
    entry: dict,
    *,
    model_target: Path = BEST_MODEL_PATH,
    metadata_target: Path = MODEL_METADATA_PATH,
    preprocessor_target: Path = DEFAULT_PREPROCESSOR_PATH,
) -> None:
    """Copy a version snapshot back to the live inference paths."""
    artifact = resolve_stored_path(entry["artifact"])
    if not artifact.exists():
        raise RegistryError(f"Snapshot artifact is missing: {artifact}")
    expected = entry.get("artifact_sha256")
    if expected and _sha256(artifact) != expected:
        raise RegistryError(
            f"Snapshot artifact hash mismatch for {entry['version']}; refusing to deploy."
        )
    Path(model_target).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, model_target)

    snapshot_meta = resolve_stored_path(entry["metadata_snapshot"])
    if snapshot_meta.exists():
        Path(metadata_target).parent.mkdir(parents=True, exist_ok=True)
        deployed = json.loads(snapshot_meta.read_text(encoding="utf-8"))
        # Keep the metadata pointing at the live paths it is deployed to.
        deployed["selected_model_path"] = _stored_path(model_target)
        deployed["preprocessor_path"] = _stored_path(preprocessor_target)
        Path(metadata_target).write_text(
            json.dumps(deployed, indent=2, default=str), encoding="utf-8"
        )

    preprocessor = entry.get("preprocessor")
    if preprocessor:
        source = resolve_stored_path(preprocessor)
        if source.exists():
            Path(preprocessor_target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, preprocessor_target)
        else:
            logger.warning("Preprocessor snapshot missing for %s", entry["version"])


def promote(
    version: str,
    *,
    reason: str,
    actor: str = "cli",
    path: Path = REGISTRY_PATH,
    model_target: Path = BEST_MODEL_PATH,
    metadata_target: Path = MODEL_METADATA_PATH,
    preprocessor_target: Path = DEFAULT_PREPROCESSOR_PATH,
) -> dict:
    """Promote a registered version after checking validation evidence."""
    registry = load_registry(path)
    if version not in registry["versions"]:
        raise RegistryError(f"Unknown model version: {version}")
    if registry.get("production") == version:
        raise RegistryError(f"{version} is already the production model.")
    challenger = registry["versions"][version]
    production = registry["versions"].get(registry.get("production") or "")
    evidence = _validation_evidence(challenger, production)

    _restore_snapshot(
        challenger,
        model_target=model_target,
        metadata_target=metadata_target,
        preprocessor_target=preprocessor_target,
    )

    previous_version = registry.get("production")
    if production is not None:
        production["status"] = STATUS_ARCHIVED
    challenger["status"] = STATUS_PRODUCTION
    challenger["promotion"] = {
        "promoted_at": _now(),
        "actor": actor,
        "reason": reason,
        "evidence": evidence,
    }
    registry["production"] = version
    registry["production_since"] = challenger["promotion"]["promoted_at"]
    if previous_version:
        registry["production_history"].append({
            "version": previous_version,
            "replaced_at": challenger["promotion"]["promoted_at"],
            "replaced_by": version,
            "reason": reason,
        })
    save_registry(registry, path)
    logger.info("Promoted %s to production (previous: %s)", version, previous_version)
    return challenger


def rollback(
    *,
    reason: str,
    actor: str = "cli",
    path: Path = REGISTRY_PATH,
    model_target: Path = BEST_MODEL_PATH,
    metadata_target: Path = MODEL_METADATA_PATH,
    preprocessor_target: Path = DEFAULT_PREPROCESSOR_PATH,
) -> dict:
    """Restore the previous production version from its snapshot."""
    registry = load_registry(path)
    history = registry.get("production_history") or []
    if not history:
        raise RegistryError("No previous production version is recorded; cannot roll back.")
    previous_version = history[-1]["version"]
    previous = registry["versions"].get(previous_version)
    if previous is None:
        raise RegistryError(f"Previous production version {previous_version} is missing.")

    _restore_snapshot(
        previous,
        model_target=model_target,
        metadata_target=metadata_target,
        preprocessor_target=preprocessor_target,
    )

    current_version = registry.get("production")
    if current_version and current_version in registry["versions"]:
        current = registry["versions"][current_version]
        current["status"] = STATUS_ROLLED_BACK
        current.setdefault("notes", []).append({
            "at": _now(),
            "note": f"rolled back to {previous_version}: {reason}",
        })
    previous["status"] = STATUS_PRODUCTION
    previous["promotion"] = {
        "promoted_at": _now(),
        "actor": actor,
        "reason": f"rollback: {reason}",
        "evidence": {"restored_from_history": True},
    }
    registry["production"] = previous_version
    registry["production_since"] = previous["promotion"]["promoted_at"]
    registry["production_history"].pop()
    save_registry(registry, path)
    logger.info("Rolled back production to %s", previous_version)
    return previous


def verify(version: str | None = None, path: Path = REGISTRY_PATH) -> list[dict]:
    """Check artifact/preprocessor hashes and feature order for versions."""
    registry = load_registry(path)
    versions = [version] if version else list(registry["versions"])
    results = []
    for name in versions:
        entry = registry["versions"].get(name)
        if entry is None:
            results.append({"version": name, "ok": False, "problems": ["unknown version"]})
            continue
        problems = []
        stored_artifact = entry.get("artifact")
        if not stored_artifact:
            problems.append("artifact path is not recorded")
        else:
            artifact = resolve_stored_path(stored_artifact)
            if not artifact.exists():
                problems.append(f"artifact missing: {_stored_path(artifact)}")
            elif entry.get("artifact_sha256") and _sha256(artifact) != entry["artifact_sha256"]:
                problems.append("artifact hash mismatch")
        if entry.get("feature_names") != list(FEATURE_COLUMNS):
            problems.append("feature order differs from FEATURE_COLUMNS")
        elif feature_set_hash() != entry.get("feature_set_hash"):
            problems.append("feature set hash mismatch")
        stored_pre = entry.get("preprocessor")
        if stored_pre:
            preprocessor = resolve_stored_path(stored_pre)
            if not preprocessor.exists():
                problems.append(f"preprocessor snapshot missing: {_stored_path(preprocessor)}")
            elif (
                entry.get("preprocessor_sha256")
                and _sha256(preprocessor) != entry["preprocessor_sha256"]
            ):
                problems.append("preprocessor snapshot hash mismatch")
        stored_meta = entry.get("metadata_snapshot")
        if stored_meta:
            meta_path = resolve_stored_path(stored_meta)
            if not meta_path.exists():
                problems.append(f"metadata snapshot missing: {_stored_path(meta_path)}")
        results.append({"version": name, "ok": not problems, "problems": problems})
    return results


def migrate_registry(path: Path = REGISTRY_PATH) -> dict:
    """Migrate legacy registry entries in place, safely.

    Rewrites Windows-style path fields to portable forward-slash form and, for
    snapshots registered before preprocessor hashing existed, records the
    SHA-256 of the snapshot on disk. Model/preprocessor files are never
    modified, so immutable snapshots and recorded model hashes are preserved.
    """
    path = Path(path)
    report = {"path": _stored_path(path), "migrated_paths": [], "backfilled_hashes": []}
    if not path.exists():
        return report
    registry = _read_registry(path)
    report["migrated_paths"] = normalize_registry_paths(registry)
    report["backfilled_hashes"] = _backfill_preprocessor_hashes(registry)
    if report["migrated_paths"] or report["backfilled_hashes"]:
        save_registry(registry, path)
    return report


def status(path: Path = REGISTRY_PATH) -> dict:
    """Compact production status for the dashboard and API."""
    registry = load_registry(path)
    production = registry["versions"].get(registry.get("production") or "")
    return {
        "production": production,
        "production_version": registry.get("production"),
        "production_since": registry.get("production_since"),
        "n_versions": len(registry["versions"]),
        "candidates": [
            entry for entry in registry["versions"].values()
            if entry.get("status") == STATUS_CANDIDATE
        ],
        "updated_at": registry.get("updated_at"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="La Liga model registry")
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("status", help="Show the production model and candidate count")
    sub.add_parser("list", help="List every registered version")
    sub.add_parser("register", help="Register the latest training artifacts")
    sub.add_parser("verify", help="Verify artifact hashes")
    sub.add_parser("migrate", help="Normalise legacy registry paths and backfill hashes")

    promote_parser = sub.add_parser("promote", help="Promote a version to production")
    promote_parser.add_argument("version")
    promote_parser.add_argument("--reason", required=True)

    rollback_parser = sub.add_parser("rollback", help="Restore the previous production version")
    rollback_parser.add_argument("--reason", required=True)

    args = parser.parse_args(argv)
    # ``python main.py registry`` forwards no subcommand; status is the safe default.
    if not args.command:
        args.command = "status"

    try:
        if args.command == "status":
            state = status()
            production = state["production"]
            if production is None:
                print("No production model registered.")
            else:
                validation = production.get("validation_metrics", {})
                print(
                    f"Production: {production['version']} "
                    f"({production['model_family']}) since {state['production_since']}"
                )
                print(
                    f"  validation log-loss={validation.get('log_loss')} "
                    f"brier={validation.get('brier_score')}"
                )
            print(f"Registered versions: {state['n_versions']} | "
                  f"candidates: {len(state['candidates'])}")
            return 0

        if args.command == "list":
            table = list_models()
            print(table.to_string(index=False) if not table.empty else "No versions registered.")
            return 0

        if args.command == "register":
            entry, created = register_current_artifacts()
            action = "Registered" if created else "Already registered"
            print(f"{action}: {entry['version']} ({entry['status']})")
            return 0

        if args.command == "verify":
            results = verify()
            for result in results:
                state = "OK" if result["ok"] else "FAILED"
                print(f"{result['version']}: {state}")
                for problem in result["problems"]:
                    print(f"  - {problem}")
            return 0 if all(r["ok"] for r in results) else 1

        if args.command == "migrate":
            report = migrate_registry()
            if report["migrated_paths"]:
                print(f"Normalised registry paths: {', '.join(report['migrated_paths'])}")
            if report["backfilled_hashes"]:
                print(f"Recorded preprocessor hashes: {', '.join(report['backfilled_hashes'])}")
            if not report["migrated_paths"] and not report["backfilled_hashes"]:
                print("Registry paths are already portable; nothing to migrate.")
            return 0

        if args.command == "promote":
            entry = promote(args.version, reason=args.reason)
            print(f"Promoted {entry['version']} to production.")
            return 0

        if args.command == "rollback":
            entry = rollback(reason=args.reason)
            print(f"Rolled back to {entry['version']}.")
            return 0
    except (RegistryError, PromotionError) as exc:
        print(f"Registry error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
