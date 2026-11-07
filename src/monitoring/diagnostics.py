"""Evidence-aware monitoring for the La Liga predictor.

The checks intentionally separate three states:

* ``insufficient_data`` — too few settled predictions to say anything useful.
* ``warning`` — an indicator moved, but the sample or effect is not strong
  enough to call it degradation.
* ``confirmed`` — a sufficiently large, statistically supportive change.

None of these checks retrains or changes the deployed model. A confirmed
signal is a reason to investigate, not proof that the model is broken.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_probabilities
from src.features.build_features import build_model_frame
from src.predictions.history import add_derived_columns
from src.utils.config import (
    FEATURE_COLUMNS,
    FEATURE_REFERENCE_PATH,
    MONITORING_CONFIRM_SETTLED,
    MONITORING_MIN_SETTLED,
    MONITORING_REPORT_PATH,
    STALE_DATA_DAYS,
)

logger = logging.getLogger(__name__)

LABELS = [2, 1, 0]
PREDICTION_LABELS = ["Home Win", "Draw", "Away Win"]
WARNING = "warning"
CONFIRMED = "confirmed"
INSUFFICIENT = "insufficient_data"
OK = "ok"


def _status(rank: int) -> str:
    return {0: OK, 1: WARNING, 2: CONFIRMED}.get(rank, INSUFFICIENT)


def _bootstrap_mean_diff_ci(
    baseline: np.ndarray,
    recent: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired-free bootstrap interval for ``mean(recent) - mean(baseline)``."""
    baseline = np.asarray(baseline, dtype=float)
    recent = np.asarray(recent, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    recent = recent[np.isfinite(recent)]
    if len(baseline) < 2 or len(recent) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draws[i] = (
            rng.choice(recent, size=len(recent), replace=True).mean()
            - rng.choice(baseline, size=len(baseline), replace=True).mean()
        )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _history_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = add_derived_columns(history)
    if "predicted_at_ts" in frame:
        frame = frame.sort_values("predicted_at_ts").reset_index(drop=True)
    return frame


def _settled_with_probabilities(history: pd.DataFrame) -> pd.DataFrame:
    frame = _history_frame(history)
    if frame.empty:
        return frame
    settled = frame[frame["actual_label"].astype(str) != ""].copy()
    if settled.empty:
        return settled
    probs = settled[["model_prob_home", "model_prob_draw", "model_prob_away"]].to_numpy(
        dtype=float
    )
    settled = settled[np.isfinite(probs).all(axis=1)].copy()
    return settled


def _per_row_brier(probs: np.ndarray, actual: Iterable[int]) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    actual = np.asarray(list(actual), dtype=int)
    one_hot = (actual[:, None] == np.asarray(LABELS)[None, :]).astype(float)
    return np.sum((probs - one_hot) ** 2, axis=1)


def check_prediction_distribution(
    history: pd.DataFrame,
    *,
    recent_n: int = 40,
    warning_shift: float = 0.15,
    confirmed_shift: float = 0.30,
) -> dict:
    """Compare recent predicted-outcome shares with the earlier ledger."""
    frame = _history_frame(history)
    if frame.empty or "predicted_outcome" not in frame:
        return {
            "name": "prediction_distribution",
            "status": INSUFFICIENT,
            "message": "No logged predictions yet.",
            "details": {},
        }
    valid = frame[frame["predicted_outcome"].isin(PREDICTION_LABELS)].copy()
    if len(valid) < recent_n + 10:
        return {
            "name": "prediction_distribution",
            "status": INSUFFICIENT,
            "message": (
                f"Only {len(valid)} valid logged predictions; need at least "
                f"{recent_n + 10} for a before/after comparison."
            ),
            "details": {"n": int(len(valid))},
        }
    recent = valid.tail(recent_n)
    baseline = valid.iloc[:-recent_n]
    recent_share = recent["predicted_outcome"].value_counts(normalize=True).reindex(
        PREDICTION_LABELS, fill_value=0.0
    )
    baseline_share = baseline["predicted_outcome"].value_counts(normalize=True).reindex(
        PREDICTION_LABELS, fill_value=0.0
    )
    shifts = (recent_share - baseline_share).abs()
    max_shift = float(shifts.max())
    rank = 1 if max_shift >= warning_shift else 0
    if len(baseline) >= 50 and len(recent) >= 30 and max_shift >= confirmed_shift:
        rank = 2
    status = _status(rank)
    changed = str(shifts.idxmax()) if max_shift > 0 else "none"
    return {
        "name": "prediction_distribution",
        "status": status,
        "message": (
            f"Largest predicted-share shift is {max_shift:.1%} for {changed} "
            f"(recent {len(recent)} vs earlier {len(baseline)})."
        ),
        "details": {
            "recent_n": int(len(recent)),
            "baseline_n": int(len(baseline)),
            "max_shift": max_shift,
            "recent_shares": {k: float(v) for k, v in recent_share.items()},
            "baseline_shares": {k: float(v) for k, v in baseline_share.items()},
        },
    }


def check_team_distribution(
    raw: pd.DataFrame,
    *,
    recent_n: int = 60,
    warning_shift: float = 0.10,
    confirmed_shift: float = 0.20,
) -> dict:
    """Compare recent team appearance shares with the earlier dataset."""
    if raw is None or raw.empty or not {"home_team", "away_team", "date"}.issubset(raw):
        return {
            "name": "team_distribution",
            "status": INSUFFICIENT,
            "message": "Raw team/date columns are unavailable.",
            "details": {},
        }
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(frame) < recent_n + 60:
        return {
            "name": "team_distribution",
            "status": INSUFFICIENT,
            "message": f"Only {len(frame)} dated matches; need more history.",
            "details": {"n": int(len(frame))},
        }
    recent = frame.tail(recent_n)
    baseline = frame.iloc[:-recent_n]

    def shares(part: pd.DataFrame) -> pd.Series:
        teams = pd.concat([part["home_team"], part["away_team"]]).astype(str)
        return teams.value_counts(normalize=True)

    recent_share = shares(recent)
    baseline_share = shares(baseline)
    all_teams = sorted(set(recent_share.index) | set(baseline_share.index))
    recent_share = recent_share.reindex(all_teams, fill_value=0.0)
    baseline_share = baseline_share.reindex(all_teams, fill_value=0.0)
    shifts = (recent_share - baseline_share).abs()
    max_shift = float(shifts.max()) if len(shifts) else 0.0
    rank = 1 if max_shift >= warning_shift else 0
    if len(baseline) >= 100 and len(recent) >= 30 and max_shift >= confirmed_shift:
        rank = 2
    team = str(shifts.idxmax()) if max_shift > 0 else "none"
    return {
        "name": "team_distribution",
        "status": _status(rank),
        "message": (
            f"Largest team-appearance share shift is {max_shift:.1%} for {team} "
            f"(recent {len(recent)} matches)."
        ),
        "details": {
            "recent_n": int(len(recent)),
            "baseline_n": int(len(baseline)),
            "max_shift": max_shift,
            "team": team,
        },
    }


def check_missing_inputs(
    raw: pd.DataFrame,
    *,
    feature_frame: pd.DataFrame | None = None,
) -> dict:
    """Check required raw columns, odds coverage and model-feature missingness."""
    problems: list[str] = []
    details: dict = {}
    required = {"date", "home_team", "away_team", "season"}
    if raw is None or raw.empty:
        return {
            "name": "missing_inputs",
            "status": WARNING,
            "message": "Raw dataset is empty.",
            "details": {},
        }
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        problems.append(f"Missing raw columns: {', '.join(missing_columns)}")
    if "implied_prob_home" in raw:
        missing_odds = int(pd.to_numeric(raw["implied_prob_home"], errors="coerce").isna().sum())
        details["missing_odds"] = missing_odds
        if missing_odds:
            problems.append(f"{missing_odds} matches have no home implied probability")
    if feature_frame is not None and not feature_frame.empty:
        missing = feature_frame[list(FEATURE_COLUMNS)].isna().mean()
        details["max_feature_missing_fraction"] = float(missing.max())
        details["feature_missing"] = {
            str(k): float(v) for k, v in missing[missing > 0].items()
        }
        if float(missing.max()) > 0.05:
            problems.append(
                f"Feature missing fraction reaches {float(missing.max()):.1%}"
            )
    if "date" in raw.columns and raw["date"].notna().any():
        latest = pd.to_datetime(raw["date"], errors="coerce").max()
        if pd.notna(latest):
            age = (pd.Timestamp.today().normalize() - latest.normalize()).days
            details["latest_match_age_days"] = int(age)
            if age > STALE_DATA_DAYS:
                problems.append(f"Latest match is {age} days old")
    status = WARNING if problems else OK
    return {
        "name": "missing_inputs",
        "status": status,
        "message": "; ".join(problems) if problems else "Required inputs are present.",
        "details": details,
    }


def _feature_reference() -> pd.DataFrame:
    if not FEATURE_REFERENCE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(FEATURE_REFERENCE_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read feature reference: %s", exc)
        return pd.DataFrame()


def check_feature_drift(
    raw: pd.DataFrame,
    *,
    recent_n: int = 60,
    warning_sd: float = 2.0,
    confirmed_sd: float = 3.5,
) -> dict:
    """Standardized mean shift of recent features against the training reference."""
    reference = _feature_reference()
    if reference.empty or raw is None or raw.empty:
        return {
            "name": "feature_drift",
            "status": INSUFFICIENT,
            "message": "Feature reference or raw data is unavailable.",
            "details": {},
        }
    try:
        from src.features.preprocessing import FeaturePreprocessor

        X, _y, meta = build_model_frame(raw)
        recent_idx = meta.sort_values("date").tail(recent_n).index
        recent = X.loc[recent_idx]
        # The reference stores training-standardized moments (mean 0, std 1).
        recent = FeaturePreprocessor.load().transform(recent)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "feature_drift",
            "status": INSUFFICIENT,
            "message": f"Could not build features for drift: {exc}",
            "details": {},
        }
    shifts = []
    for _, ref in reference.iterrows():
        feature = ref.get("feature")
        if feature not in recent:
            continue
        std = float(ref.get("std", 0.0) or 0.0)
        if std <= 0:
            continue
        shift = (float(recent[feature].mean()) - float(ref.get("mean", 0.0))) / std
        shifts.append((str(feature), float(shift)))
    if not shifts:
        return {
            "name": "feature_drift",
            "status": INSUFFICIENT,
            "message": "No comparable training features were found.",
            "details": {},
        }
    shifts.sort(key=lambda item: abs(item[1]), reverse=True)
    feature, shift = shifts[0]
    rank = 1 if abs(shift) >= warning_sd else 0
    if len(recent) >= 30 and abs(shift) >= confirmed_sd:
        rank = 2
    return {
        "name": "feature_drift",
        "status": _status(rank),
        "message": (
            f"Largest feature shift is {shift:+.2f} SD for {feature} "
            f"over {len(recent)} recent matches."
        ),
        "details": {
            "recent_n": int(len(recent)),
            "max_abs_shift_sd": abs(shift),
            "feature": feature,
            "top_shifts": [
                {"feature": name, "shift_sd": value} for name, value in shifts[:8]
            ],
        },
    }


def check_calibration_change(
    history: pd.DataFrame,
    *,
    recent_n: int = 60,
    warning_gap: float = 0.03,
    confirmed_gap: float = 0.06,
) -> dict:
    """Compare recent and earlier ECE for settled ledger probabilities."""
    settled = _settled_with_probabilities(history)
    if len(settled) < MONITORING_MIN_SETTLED:
        return {
            "name": "calibration_change",
            "status": INSUFFICIENT,
            "message": (
                f"Only {len(settled)} settled predictions with probabilities; "
                f"need {MONITORING_MIN_SETTLED}."
            ),
            "details": {"n": int(len(settled))},
        }
    probs = settled[["model_prob_home", "model_prob_draw", "model_prob_away"]].to_numpy(float)
    y = settled["actual_outcome"].astype(int).to_numpy()
    n_recent = min(recent_n, max(MONITORING_MIN_SETTLED, len(settled) // 3))
    recent = slice(len(settled) - n_recent, len(settled))
    earlier = slice(0, len(settled) - n_recent)
    earlier_metrics = evaluate_probabilities(
        y[earlier], probs[earlier], LABELS, "earlier"
    )
    recent_metrics = evaluate_probabilities(
        y[recent], probs[recent], LABELS, "recent"
    )
    gap = float(recent_metrics["ece"] - earlier_metrics["ece"])
    rank = 1 if gap >= warning_gap else 0
    if len(settled) >= MONITORING_CONFIRM_SETTLED and n_recent >= 30 and gap >= confirmed_gap:
        rank = 2
    return {
        "name": "calibration_change",
        "status": _status(rank),
        "message": (
            f"Recent ECE is {recent_metrics['ece']:.3f} vs earlier "
            f"{earlier_metrics['ece']:.3f} (gap {gap:+.3f}, "
            f"{n_recent} recent / {len(settled) - n_recent} earlier)."
        ),
        "details": {
            "recent_n": int(n_recent),
            "earlier_n": int(len(settled) - n_recent),
            "recent_ece": float(recent_metrics["ece"]),
            "earlier_ece": float(earlier_metrics["ece"]),
            "gap": gap,
        },
    }


def check_performance(
    history: pd.DataFrame,
    *,
    recent_n: int = 60,
    warning_log_loss: float = 0.03,
    confirmed_log_loss: float = 0.05,
) -> dict:
    """Compare recent settled performance with the earlier ledger period."""
    settled = _settled_with_probabilities(history)
    n = len(settled)
    if n < MONITORING_MIN_SETTLED:
        return {
            "name": "performance",
            "status": INSUFFICIENT,
            "message": (
                f"Only {n} settled predictions with probabilities; need "
                f"{MONITORING_MIN_SETTLED} before judging degradation."
            ),
            "details": {"n": n},
        }
    n_recent = min(recent_n, max(MONITORING_MIN_SETTLED, n // 3))
    if n - n_recent < 10:
        return {
            "name": "performance",
            "status": INSUFFICIENT,
            "message": "Not enough earlier settled predictions for a comparison.",
            "details": {"n": n, "recent_n": n_recent},
        }
    recent = settled.tail(n_recent)
    earlier = settled.iloc[:-n_recent]
    recent_probs = recent[["model_prob_home", "model_prob_draw", "model_prob_away"]].to_numpy(float)
    earlier_probs = earlier[["model_prob_home", "model_prob_draw", "model_prob_away"]].to_numpy(float)
    recent_y = recent["actual_outcome"].astype(int).to_numpy()
    earlier_y = earlier["actual_outcome"].astype(int).to_numpy()
    recent_metrics = evaluate_probabilities(recent_y, recent_probs, LABELS, "recent")
    earlier_metrics = evaluate_probabilities(earlier_y, earlier_probs, LABELS, "earlier")
    loss_delta = float(recent_metrics["log_loss"] - earlier_metrics["log_loss"])
    acc_delta = float(recent_metrics["accuracy"] - earlier_metrics["accuracy"])
    recent_brier = _per_row_brier(recent_probs, recent_y)
    earlier_brier = _per_row_brier(earlier_probs, earlier_y)
    ci_low, ci_high = _bootstrap_mean_diff_ci(earlier_brier, recent_brier)

    rank = 0
    if loss_delta >= warning_log_loss or acc_delta <= -0.03:
        rank = 1
    confirmed = (
        n >= MONITORING_CONFIRM_SETTLED
        and n_recent >= 30
        and loss_delta >= confirmed_log_loss
        and acc_delta <= -0.03
        and np.isfinite(ci_low)
        and ci_low > 0
    )
    if confirmed:
        rank = 2
    direction = "worse" if loss_delta > 0 else "better"
    return {
        "name": "performance",
        "status": _status(rank),
        "message": (
            f"Recent log-loss is {recent_metrics['log_loss']:.4f} vs earlier "
            f"{earlier_metrics['log_loss']:.4f} ({direction} by {abs(loss_delta):.4f}); "
            f"accuracy {recent_metrics['accuracy']:.1%} vs "
            f"{earlier_metrics['accuracy']:.1%}."
        ),
        "details": {
            "recent_n": int(n_recent),
            "earlier_n": int(len(earlier)),
            "recent_log_loss": float(recent_metrics["log_loss"]),
            "earlier_log_loss": float(earlier_metrics["log_loss"]),
            "log_loss_delta": loss_delta,
            "recent_accuracy": float(recent_metrics["accuracy"]),
            "earlier_accuracy": float(earlier_metrics["accuracy"]),
            "accuracy_delta": acc_delta,
            "recent_draw_recall": float(recent_metrics.get("draw_recall", float("nan"))),
            "earlier_draw_recall": float(earlier_metrics.get("draw_recall", float("nan"))),
            "brier_delta": float(recent_brier.mean() - earlier_brier.mean()),
            "brier_delta_ci": [ci_low, ci_high],
        },
    }


def run_diagnostics(
    raw: pd.DataFrame,
    history: pd.DataFrame,
    metadata: dict | None = None,
    *,
    feature_frame: pd.DataFrame | None = None,
) -> dict:
    """Run all monitoring checks and return a JSON-ready report."""
    checks = [
        check_missing_inputs(raw, feature_frame=feature_frame),
        check_feature_drift(raw),
        check_team_distribution(raw),
        check_prediction_distribution(history),
        check_calibration_change(history),
        check_performance(history),
    ]
    severities = [check["status"] for check in checks]
    if CONFIRMED in severities:
        overall = CONFIRMED
    elif WARNING in severities:
        overall = WARNING
    elif INSUFFICIENT in severities:
        # Do not call the system healthy when a required evidence window is
        # missing (for example, no settled predictions yet).
        overall = INSUFFICIENT
    else:
        overall = OK
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": overall,
        "model_version": (metadata or {}).get("model_version"),
        "checks": checks,
        "note": (
            "Warnings indicate movement worth reviewing. Confirmed degradation "
            "requires a sufficient settled sample and a supportive interval; it "
            "still does not prove the model is broken."
        ),
    }


def save_monitoring_report(report: dict, path: Path = MONITORING_REPORT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote monitoring report to %s", path)
    return path
