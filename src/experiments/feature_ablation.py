"""Optional feature-group ablation vs the 19-feature baseline.

The runner answers one narrow question: does adding the xG or player-
availability group improve *validation* log-loss without worsening Brier
score? Selection uses the validation season only; the untouched test season is
never read by this experiment.

If a group's source file does not exist, the row says ``unavailable`` and no
values are simulated. The production model is never retrained, registered or
promoted here.

Usage:
    python -m src.experiments.feature_ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_probabilities  # noqa: E402
from src.evaluation.splits import chronological_split  # noqa: E402
from src.features.build_features import build_model_frame, load_raw  # noqa: E402
from src.features.optional_features import (  # noqa: E402
    OptionalFeatureUnavailable,
    feature_group_status,
)
from src.features.preprocessing import FeaturePreprocessor  # noqa: E402
from src.utils.config import (  # noqa: E402
    BASELINE_FEATURE_GROUP,
    FEATURE_ABLATION_PATH,
    FEATURE_ABLATION_SUMMARY_PATH,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TEST_SEASON,
    TRAIN_SEASONS,
    VALIDATION_SEASON,
)

logger = logging.getLogger(__name__)

CANDIDATE_GROUPS = ("xg", "player_availability")
# Declared before seeing results: keep a group only for a meaningful
# validation improvement, and never at the cost of calibration (Brier score).
MIN_LOG_LOSS_GAIN = 0.002
MAX_BRIER_REGRESSION = 0.002
RF_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "max_depth": 5,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
}


def _evaluate(
    raw: pd.DataFrame,
    groups: list[str],
    *,
    label: str,
) -> dict:
    """Fit preprocessing/RF on the training window and score validation."""
    X, y, meta = build_model_frame(raw, feature_groups=groups)
    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx = splits["train"].index
    val_idx = splits["validation"].index
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(f"Split is empty for {label}: check configured seasons.")

    preprocessor = FeaturePreprocessor(feature_names=list(X.columns))
    X_train = preprocessor.fit_transform(X.loc[train_idx])
    X_val = preprocessor.transform(X.loc[val_idx])
    model = RandomForestClassifier(random_state=RANDOM_STATE, **RF_PARAMS)
    model.fit(X_train, y.loc[train_idx])

    proba = model.predict_proba(X_val)
    class_index = {int(c): i for i, c in enumerate(model.classes_)}
    ordered = proba[:, [class_index[2], class_index[1], class_index[0]]]
    metrics = evaluate_probabilities(
        y.loc[val_idx].to_numpy(), ordered, [2, 1, 0], label
    )
    return {
        "group": label,
        "status": "evaluated",
        "n_features": int(X.shape[1]),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "accuracy": float(metrics["accuracy"]),
        "log_loss": float(metrics["log_loss"]),
        "brier_score": float(metrics["brier_score"]),
        "macro_f1": float(metrics.get("macro_f1", float("nan"))),
    }


def run_ablation(raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Evaluate baseline and each optional group on validation data only."""
    raw = load_raw() if raw is None else raw
    group_statuses = {
        group: feature_group_status(group).as_dict() for group in CANDIDATE_GROUPS
    }

    rows: list[dict] = []
    baseline = _evaluate(raw, [BASELINE_FEATURE_GROUP], label="baseline_19_features")
    rows.append(baseline)
    baseline_loss = baseline["log_loss"]
    baseline_brier = baseline["brier_score"]

    for group in CANDIDATE_GROUPS:
        status = group_statuses[group]
        if not status["available"]:
            rows.append({
                "group": group,
                "status": "unavailable",
                "n_features": 0,
                "n_train": 0,
                "n_validation": 0,
                "accuracy": float("nan"),
                "log_loss": float("nan"),
                "brier_score": float("nan"),
                "macro_f1": float("nan"),
                "reason": status["reason"],
            })
            continue
        try:
            row = _evaluate(raw, [BASELINE_FEATURE_GROUP, group], label=group)
        except OptionalFeatureUnavailable as exc:
            rows.append({
                "group": group,
                "status": "unavailable",
                "n_features": 0,
                "n_train": 0,
                "n_validation": 0,
                "accuracy": float("nan"),
                "log_loss": float("nan"),
                "brier_score": float("nan"),
                "macro_f1": float("nan"),
                "reason": str(exc),
            })
            continue
        row["delta_log_loss_vs_baseline"] = row["log_loss"] - baseline_loss
        row["delta_brier_vs_baseline"] = row["brier_score"] - baseline_brier
        improved = (
            row["log_loss"] <= baseline_loss - MIN_LOG_LOSS_GAIN
            and row["brier_score"] <= baseline_brier + MAX_BRIER_REGRESSION
        )
        row["status"] = "retained" if improved else "rejected"
        row["reason"] = (
            "validation log-loss improved materially without a Brier regression"
            if improved else
            "did not beat the baseline by the declared margin on validation data"
        )
        rows.append(row)

    for row in rows:
        row.setdefault("delta_log_loss_vs_baseline", 0.0 if row["group"].endswith("19_features") else float("nan"))
        row.setdefault("delta_brier_vs_baseline", 0.0 if row["group"].endswith("19_features") else float("nan"))
        row.setdefault("reason", "")

    table = pd.DataFrame(rows)
    retained = [r["group"] for r in rows if r["status"] == "retained"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_split": f"train={TRAIN_SEASONS}, validation={VALIDATION_SEASON}",
        "test_season_used": False,
        "baseline": baseline,
        "groups": group_statuses,
        "retained_groups": retained,
        "decision": (
            "keep baseline only"
            if not retained
            else f"candidate groups passed validation: {', '.join(retained)}; "
                 "training/promotion remains a separate, explicit step"
        ),
        "min_log_loss_gain": MIN_LOG_LOSS_GAIN,
        "max_brier_regression": MAX_BRIER_REGRESSION,
        "production_model_unchanged": True,
    }
    return table, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional feature-group ablation")
    parser.add_argument(
        "--output", type=Path, default=FEATURE_ABLATION_PATH, help="CSV output path"
    )
    parser.add_argument(
        "--summary", type=Path, default=FEATURE_ABLATION_SUMMARY_PATH,
        help="JSON summary path",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    table, summary = run_ablation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    args.summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s and %s", args.output, args.summary)
    print(table.to_string(index=False))
    print(f"\nDecision: {summary['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
