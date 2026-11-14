"""Live prediction performance, kept separate from offline evaluation.

Three different things are reported side by side and never mixed:

* **validation** — the model-selection split from the production metadata;
* **historical test** — the untouched test season from the production metadata;
* **live** — settled predictions in the append-only ledger.

Every live statement carries its sample size and a ``status`` that stays
``insufficient_data`` below ``MONITORING_MIN_SETTLED``. Live results are grouped
by season and by model version so a model change cannot hide behind pooled
numbers. Nothing here trains, promotes or rewrites a model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.baselines import BASELINE_LABELS
from src.evaluation.metrics import evaluate_probabilities
from src.predictions import history as hist
from src.utils.config import (
    LIVE_PERFORMANCE_MARKDOWN_PATH,
    LIVE_PERFORMANCE_PATH,
    MODEL_METADATA_PATH,
    MONITORING_MIN_SETTLED,
)

logger = logging.getLogger(__name__)

PROB_COLUMNS = ["model_prob_home", "model_prob_draw", "model_prob_away"]
INSUFFICIENT = "insufficient_data"
OK = "ok"


def _now(now: datetime | None = None) -> datetime:
    moment = now or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    if not isinstance(value, (str, bytes, dict, list)) and pd.isna(value):
        return None
    return value


def metrics_block(
    frame: pd.DataFrame | None,
    name: str,
    *,
    min_settled: int = MONITORING_MIN_SETTLED,
) -> dict:
    """Log-loss, Brier, accuracy, macro-F1, per-class metrics and calibration."""
    empty = {
        "name": name,
        "n": 0,
        "status": INSUFFICIENT,
        "min_settled": int(min_settled),
        "accuracy": None,
        "log_loss": None,
        "brier_score": None,
        "macro_f1": None,
        "draw_recall": None,
        "ece": None,
        "per_class": [],
    }
    if frame is None or frame.empty:
        return empty

    probabilities = frame[PROB_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    actual = pd.to_numeric(frame["actual_outcome"], errors="coerce")
    usable = (
        np.isfinite(probabilities).all(axis=1)
        & (probabilities >= 0).all(axis=1)
        & (probabilities.sum(axis=1) > 0)
        & actual.notna().to_numpy()
    )
    n = int(usable.sum())
    if n < 2:
        empty["n"] = n
        return empty

    result = evaluate_probabilities(
        actual.to_numpy()[usable].astype(int),
        probabilities[usable],
        BASELINE_LABELS,
        name,
    )
    per_class = result.get("per_class")
    return {
        "name": name,
        "n": n,
        "status": OK if n >= min_settled else INSUFFICIENT,
        "min_settled": int(min_settled),
        "accuracy": float(result["accuracy"]),
        "log_loss": float(result["log_loss"]),
        "brier_score": float(result["brier_score"]),
        "macro_f1": float(result["macro_f1"]),
        "draw_recall": float(result["draw_recall"]),
        "ece": float(result["ece"]),
        "per_class": (
            per_class.to_dict(orient="records")
            if isinstance(per_class, pd.DataFrame) else []
        ),
    }


def _split_block(metadata: dict | None, key: str, label: str) -> dict:
    metrics = (metadata or {}).get(key) or {}
    if not metrics:
        return {
            "name": label,
            "status": "not_available",
            "source": "model metadata",
            "note": f"No {label} metrics are recorded for this model version.",
        }
    block = {
        "name": label,
        "status": OK,
        "source": "model metadata",
        "n": int(metrics.get("n_samples") or 0),
        "note": f"{label.capitalize()} metrics, not live performance.",
    }
    for field in (
        "accuracy", "log_loss", "brier_score", "macro_f1", "draw_recall", "ece",
    ):
        if metrics.get(field) is not None:
            block[field] = float(metrics[field])
    return block


def load_production_metadata() -> dict:
    """Production snapshot metadata, falling back to the live metadata file."""
    try:
        from src.models.registry import production_paths

        paths = production_paths()
        path = Path(paths.get("metadata") or MODEL_METADATA_PATH)
    except Exception:  # noqa: BLE001 - registry is optional for reporting
        path = MODEL_METADATA_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - reporting must not crash
        logger.warning("Could not read model metadata for live performance: %s", exc)
        return {}


def build_live_performance(
    history: pd.DataFrame | None = None,
    *,
    metadata: dict | None = None,
    now: datetime | None = None,
    min_settled: int = MONITORING_MIN_SETTLED,
) -> dict:
    """Build the live performance report without modifying any artifact."""
    moment = _now(now)
    if history is None:
        history, _ = hist.load_history()
    if metadata is None:
        metadata = load_production_metadata()

    frame = (
        hist.add_derived_columns(history)
        if history is not None and not history.empty
        else pd.DataFrame()
    )
    settled = (
        frame[frame["actual_label"].astype(str) != ""].copy()
        if not frame.empty else pd.DataFrame()
    )

    overall = metrics_block(settled, "live settled predictions", min_settled=min_settled)

    by_season: list[dict] = []
    if not settled.empty and "season" in settled:
        for season, group in settled.groupby("season", dropna=False):
            block = metrics_block(
                group, f"season {season}", min_settled=min_settled
            )
            block["season"] = str(season)
            by_season.append(block)

    by_model: list[dict] = []
    if not settled.empty:
        grouped = settled.groupby(
            ["model_name", "model_version"], dropna=False
        )
        for (model_name, model_version), group in grouped:
            block = metrics_block(
                group, f"{model_name} {model_version}", min_settled=min_settled
            )
            block["model_name"] = str(model_name)
            block["model_version"] = str(model_version)
            by_model.append(block)

    report = {
        "generated_at": moment.isoformat(timespec="seconds"),
        "minimum_settled_sample": int(min_settled),
        "status": overall["status"],
        "counts": {
            "prediction_rows": int(len(frame)),
            "settled_rows": int(len(settled)),
            "pending_rows": int(len(frame) - len(settled)),
            "void_rows": int(
                frame["prediction_status"].astype(str).eq(hist.STATUS_VOID).sum()
            ) if not frame.empty else 0,
            "distinct_fixtures": int(frame["fixture_id"].nunique()) if not frame.empty else 0,
        },
        "splits": {
            "validation": _split_block(metadata, "validation_metrics", "validation"),
            "historical_test": _split_block(metadata, "test_metrics", "historical test"),
            "live": overall,
        },
        "by_season": by_season,
        "by_model_version": by_model,
        "notes": [
            "Validation, historical test and live metrics come from different samples "
            "and are never pooled.",
            "Live metrics below the minimum settled sample are arithmetic only and are "
            "marked insufficient_data.",
            "Settled rows include corrected official results; the prior record is kept "
            "in the correction log.",
            "Void fixtures (cancelled/abandoned) are excluded from all performance metrics.",
        ],
    }
    return _jsonable(report)


def write_live_performance(
    report: dict,
    *,
    json_path: Path = LIVE_PERFORMANCE_PATH,
    markdown_path: Path = LIVE_PERFORMANCE_MARKDOWN_PATH,
) -> list[Path]:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )
    tmp.replace(json_path)

    def fmt(value, digits=4, pct=False):
        if value is None:
            return "n/a"
        return f"{100 * float(value):.1f}%" if pct else f"{float(value):.{digits}f}"

    splits = report.get("splits", {})
    live = splits.get("live", {})
    lines = [
        "# La Liga Predictor — Live Prediction Performance",
        "",
        f"Generated: {report['generated_at']}",
        f"Overall live status: **{report['status']}** "
        f"(minimum settled sample: {report['minimum_settled_sample']})",
        "",
        "Validation, historical test and live performance are reported separately "
        "and never pooled.",
        "",
        "## Splits",
        "",
        "| split | status | n | log-loss | Brier | accuracy | macro-F1 | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in (
        ("validation", "Validation"),
        ("historical_test", "Historical test"),
        ("live", "Live"),
    ):
        block = splits.get(key, {})
        lines.append(
            f"| {label} | {block.get('status')} | {block.get('n', 0)} | "
            f"{fmt(block.get('log_loss'))} | {fmt(block.get('brier_score'))} | "
            f"{fmt(block.get('accuracy'), pct=True)} | "
            f"{fmt(block.get('macro_f1'))} | {fmt(block.get('ece'))} |"
        )
    counts = report.get("counts", {})
    lines += [
        "",
        "## Ledger",
        "",
        f"- Prediction rows: {counts.get('prediction_rows', 0)}",
        f"- Settled rows: {counts.get('settled_rows', 0)}",
        f"- Pending rows: {counts.get('pending_rows', 0)}",
        f"- Void rows (postponed/cancelled/abandoned): {counts.get('void_rows', 0)}",
        f"- Distinct fixtures: {counts.get('distinct_fixtures', 0)}",
        "",
        "## Per class (live)",
        "",
        "| class | precision | recall |",
        "|---|---|---|",
    ]
    for row in live.get("per_class", []) or []:
        label = row.get("label", row.get("class", ""))
        lines.append(
            f"| {label} | {fmt(row.get('precision'), pct=True)} | "
            f"{fmt(row.get('recall'), pct=True)} |"
        )
    if not live.get("per_class"):
        lines.append("| n/a | n/a | n/a |")

    for key, title, label_field in (
        ("by_season", "By season", "season"),
        ("by_model_version", "By model version", "model_version"),
    ):
        rows = report.get(key, [])
        lines += ["", f"## {title}", ""]
        if not rows:
            lines.append("No settled predictions.")
            continue
        lines += [
            "| group | status | n | log-loss | Brier | accuracy | macro-F1 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row.get(label_field, '')} | {row.get('status')} | "
                f"{row.get('n', 0)} | {fmt(row.get('log_loss'))} | "
                f"{fmt(row.get('brier_score'))} | "
                f"{fmt(row.get('accuracy'), pct=True)} | {fmt(row.get('macro_f1'))} |"
            )
    lines += ["", "## Notes", ""]
    lines += [f"- {note}" for note in report.get("notes", [])]
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return [json_path, markdown_path]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )
    report = build_live_performance()
    paths = write_live_performance(report)
    print(json.dumps({
        "status": report["status"],
        "counts": report["counts"],
        "reports": [str(path) for path in paths],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
