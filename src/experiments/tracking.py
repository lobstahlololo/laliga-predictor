"""Append-only experiment index with enough context to reproduce a run.

Each experiment is one JSON document plus a row in ``index.csv``. Records include
the data fingerprint, feature hash, split, hyperparameters and validation
metrics. Test metrics are optional and are only recorded for the pre-declared
final/baseline evaluations, never used to choose features or calibrators.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import EXPERIMENTS_DIR, EXPERIMENTS_INDEX_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)

INDEX_COLUMNS = [
    "experiment_id",
    "timestamp",
    "model_family",
    "model_type",
    "model_version",
    "feature_set",
    "variant",
    "feature_count",
    "feature_hash",
    "data_fingerprint",
    "random_state",
    "training_period",
    "validation_period",
    "test_period",
    "validation_log_loss",
    "validation_brier_score",
    "validation_accuracy",
    "test_log_loss",
    "test_brier_score",
    "test_accuracy",
    "test_evaluated",
    "selection_status",
    "notes",
]


def _jsonable(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def fingerprint_file(path: Path | str) -> str:
    path = Path(path)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def fingerprint_features(features: list[str]) -> str:
    payload = "|".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def experiment_id(
    model_type: str,
    feature_set: str,
    *,
    timestamp: datetime | None = None,
    config: dict | None = None,
) -> str:
    stamp = timestamp or datetime.now(timezone.utc)
    model_slug = "".join(
        c.lower() if c.isalnum() else "_" for c in model_type
    ).strip("_")
    feature_slug = "".join(
        c.lower() if c.isalnum() else "_" for c in feature_set
    ).strip("_")
    config_hash = hashlib.sha256(
        json.dumps(_jsonable(config or {}), sort_keys=True).encode("utf-8")
    ).hexdigest()[:6]
    return f"{model_slug}__{feature_slug}__{stamp:%Y%m%dT%H%M%SZ}__{config_hash}"


def make_experiment_record(
    *,
    model_type: str,
    feature_set: str,
    features: list[str],
    training_period: list[str],
    validation_period: str,
    test_period: str,
    hyperparameters: dict,
    validation_metrics: dict,
    model_version: str = "",
    model_family: str = "",
    test_metrics: dict | None = None,
    data_fingerprint: str = "unknown",
    selection_status: str = "",
    notes: str = "",
    timestamp: datetime | None = None,
    extra: dict | None = None,
    variant: str = "",
) -> dict:
    stamp = timestamp or datetime.now(timezone.utc)
    record = {
        "experiment_id": experiment_id(
            model_type, feature_set, timestamp=stamp,
            config={
                "params": hyperparameters,
                "features": features,
                "variant": variant,
            },
        ),
        "timestamp": stamp.isoformat(),
        "model_type": model_type,
        "model_family": model_family,
        "model_version": model_version,
        "feature_set": feature_set,
        "variant": variant,
        "features": list(features),
        "feature_count": len(features),
        "feature_hash": fingerprint_features(list(features)),
        "hyperparameters": dict(hyperparameters or {}),
        "training_period": list(training_period),
        "validation_period": validation_period,
        "test_period": test_period,
        "validation_metrics": _jsonable(validation_metrics or {}),
        "test_metrics": _jsonable(test_metrics) if test_metrics else None,
        "test_evaluated": bool(test_metrics),
        "data_fingerprint": data_fingerprint,
        "random_state": RANDOM_STATE,
        "selection_status": selection_status,
        "notes": notes,
    }
    if extra:
        record.update(_jsonable(extra))
    return record


def _index_row(record: dict) -> dict:
    validation = record.get("validation_metrics") or {}
    test = record.get("test_metrics") or {}
    return {
        "experiment_id": record["experiment_id"],
        "timestamp": record["timestamp"],
        "model_family": record.get("model_family", ""),
        "model_type": record["model_type"],
        "model_version": record.get("model_version", ""),
        "feature_set": record["feature_set"],
        "variant": record.get("variant", ""),
        "feature_count": record.get("feature_count"),
        "feature_hash": record.get("feature_hash", ""),
        "data_fingerprint": record.get("data_fingerprint", ""),
        "random_state": record.get("random_state"),
        "training_period": "+".join(record.get("training_period", [])),
        "validation_period": record.get("validation_period", ""),
        "test_period": record.get("test_period", ""),
        "validation_log_loss": validation.get("log_loss"),
        "validation_brier_score": validation.get("brier_score"),
        "validation_accuracy": validation.get("accuracy"),
        "test_log_loss": test.get("log_loss") if test else None,
        "test_brier_score": test.get("brier_score") if test else None,
        "test_accuracy": test.get("accuracy") if test else None,
        "test_evaluated": bool(record.get("test_evaluated", False)),
        "selection_status": record.get("selection_status", ""),
        "notes": record.get("notes", ""),
    }


def save_experiment(
    record: dict,
    *,
    experiments_dir: Path = EXPERIMENTS_DIR,
    index_path: Path = EXPERIMENTS_INDEX_PATH,
) -> Path:
    """Write one experiment JSON and upsert its row in the index."""
    experiments_dir = Path(experiments_dir)
    index_path = Path(index_path)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    exp_id = str(record["experiment_id"])
    path = experiments_dir / f"{exp_id}.json"
    path.write_text(json.dumps(_jsonable(record), indent=2), encoding="utf-8")

    if index_path.exists():
        index = pd.read_csv(index_path)
    else:
        index = pd.DataFrame(columns=INDEX_COLUMNS)
    row = _index_row(record)
    index = index[index["experiment_id"].astype(str) != exp_id] if not index.empty else index
    index = pd.concat([index, pd.DataFrame([row])], ignore_index=True)
    index = index.reindex(columns=INDEX_COLUMNS)
    index = index.sort_values("timestamp").reset_index(drop=True)
    tmp = index_path.with_suffix(".csv.tmp")
    index.to_csv(tmp, index=False)
    tmp.replace(index_path)
    logger.info("Tracked experiment %s", exp_id)
    return path


def load_experiments(
    index_path: Path = EXPERIMENTS_INDEX_PATH,
) -> pd.DataFrame:
    if not Path(index_path).exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    try:
        frame = pd.read_csv(index_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read experiment index: %s", exc)
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_experiment(experiment_id_value: str, experiments_dir: Path = EXPERIMENTS_DIR) -> dict:
    path = Path(experiments_dir) / f"{experiment_id_value}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
