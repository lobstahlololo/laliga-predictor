"""Full-season out-of-sample evaluation of every supported model family.

This is the Phase 6 evaluation contract:

* **Chronological only.** Earlier seasons train, the penultimate season
  validates, the newest season is the untouched test season.
* **Validation selects; test confirms.** Hyperparameters are fixed inputs
  (recorded with the model metadata or declared defaults), ensemble weights
  are selected on validation only, and the test season is scored once after
  every choice is fixed.
* **Market benchmark included.** Bookmaker probabilities are row-normalised to
  remove overround and never described as a learned model.
* **Uncertainty is reported.** Log-loss, Brier score and accuracy carry
  bootstrap confidence intervals; paired deltas versus market odds are
  reported for the test window.
* **Breakdowns need sample size.** Per-season, per-team, predicted-side and
  confidence-bin rows below the minimum sample are marked
  ``insufficient_sample`` instead of being presented as conclusions.
* **Walk-forward is rolling-origin.** Each fold trains on seasons before its
  target and evaluates the next season. The final test season is excluded.

No function here tunes a threshold, feature, weight or model on the test
season, and no improvement is claimed from a point estimate alone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.backtesting import balanced_sample_weights
from src.evaluation.baselines import BASELINE_LABELS, class_prior
from src.evaluation.ensemble import blend_probabilities, select_ensemble_weights
from src.evaluation.metrics import (
    align_proba_to_labels,
    evaluate_probabilities,
    metrics_to_row,
)
from src.evaluation.splits import chronological_split, walk_forward_folds
from src.features.build_features import build_model_frame, load_raw
from src.features.preprocessing import FeaturePreprocessor
from src.models.elo import (
    compute_elo,
    elo_probabilities,
    fit_elo_probability_params,
)
from src.models.poisson import (
    PoissonConfig,
    PoissonModel,
)
from src.models.train_model import build_estimator
from src.utils.config import (
    EVAL_DIR,
    FEATURE_COLUMNS,
    FULL_SEASON_BOOTSTRAP_DRAWS,
    FULL_SEASON_BREAKDOWNS_PATH,
    FULL_SEASON_METRICS_PATH,
    FULL_SEASON_REPORT_PATH,
    FULL_SEASON_SUMMARY_PATH,
    FULL_SEASON_WALK_FORWARD_PATH,
    MIN_BREAKDOWN_SAMPLES,
    MODEL_METADATA_PATH,
    RANDOM_STATE,
    TEST_SEASON,
    TRAIN_SEASONS,
    VALIDATION_SEASON,
)

logger = logging.getLogger(__name__)

TREE_MODELS = ("Random Forest", "XGBoost")
ENSEMBLE_COMPONENTS = (
    "Random Forest",
    "XGBoost",
    "Dynamic Elo",
    "Independent Poisson",
)
MODEL_FAMILIES = {
    "Market odds": "market benchmark",
    "Random Forest": "tree ensemble",
    "XGBoost": "tree ensemble",
    "Dynamic Elo": "elo",
    "Independent Poisson": "poisson",
    "Dixon-Coles": "poisson",
    "Ensemble (equal)": "ensemble",
    "Ensemble (validation)": "ensemble",
}
DEFAULT_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}
DEFAULT_XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "max_delta_step": 1,
}
MARKET_COLUMNS = ["market_home", "market_draw", "market_away"]
EPS = 1e-12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _params(metadata: dict | None) -> dict[str, dict]:
    metadata = metadata or {}
    result: dict[str, dict] = {}
    for name in TREE_MODELS:
        info = (metadata.get("all_models") or {}).get(name, {})
        result[name] = dict(
            info.get("parameters")
            or (DEFAULT_RF_PARAMS if name == "Random Forest" else DEFAULT_XGB_PARAMS)
        )
    return result


def _align(proba: np.ndarray, classes) -> np.ndarray:
    out = align_proba_to_labels(proba, classes, BASELINE_LABELS)
    out = np.clip(out, EPS, None)
    return out / out.sum(axis=1, keepdims=True)


def _market_probabilities(meta: pd.DataFrame, fallback: np.ndarray) -> np.ndarray:
    columns = ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]
    if not all(c in meta.columns for c in columns):
        return np.tile(fallback, (len(meta), 1))
    values = meta[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float, copy=True)
    missing = ~np.isfinite(values).all(axis=1) | (values.sum(axis=1) <= 0)
    values[missing] = fallback
    values = np.clip(values, EPS, None)
    return values / values.sum(axis=1, keepdims=True)


def _fit_tree(
    name: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    preprocessor = FeaturePreprocessor(feature_names=list(X_train.columns))
    X_fit = preprocessor.fit_transform(X_train)
    X_score = preprocessor.transform(X_eval)
    model = build_estimator(name, params)
    if name == "XGBoost":
        model.fit(X_fit, y_train, sample_weight=balanced_sample_weights(y_train))
    else:
        model.fit(X_fit, y_train)
    return _align(model.predict_proba(X_score), model.classes_)


def _poisson_probabilities(
    train_raw: pd.DataFrame,
    eval_meta: pd.DataFrame,
    *,
    dixon_coles: bool,
    config: PoissonConfig | None = None,
) -> np.ndarray:
    base = config or PoissonConfig()
    fitted = PoissonModel(
        PoissonConfig(**{**base.__dict__, "dixon_coles": dixon_coles})
    ).fit(train_raw, as_of=pd.to_datetime(train_raw["date"]).max())
    prediction = fitted.predict_frame(eval_meta[["home_team", "away_team"]].copy())
    return prediction[["prob_home", "prob_draw", "prob_away"]].to_numpy(float)


def _window_predictions(
    *,
    raw_train: pd.DataFrame,
    train_meta: pd.DataFrame,
    eval_meta: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    elo_train: pd.DataFrame,
    elo_eval: pd.DataFrame,
    params: dict[str, dict],
    poisson_config: PoissonConfig,
) -> dict[str, np.ndarray]:
    """Fit every model on the training window and score the evaluation window."""
    predictions: dict[str, np.ndarray] = {
        "Market odds": _market_probabilities(eval_meta, class_prior(y_train)),
    }
    for name in TREE_MODELS:
        predictions[name] = _fit_tree(name, params[name], X_train, y_train, X_eval)

    elo_params, _grid = fit_elo_probability_params(elo_train, y_train)
    predictions["Dynamic Elo"] = elo_probabilities(
        elo_eval, elo_params["draw_base"], elo_params["draw_scale"]
    )
    predictions["Independent Poisson"] = _poisson_probabilities(
        raw_train, eval_meta, dixon_coles=False, config=poisson_config
    )
    predictions["Dixon-Coles"] = _poisson_probabilities(
        raw_train, eval_meta, dixon_coles=True, config=poisson_config
    )
    return predictions


def _per_row_log_loss(y, probabilities: np.ndarray) -> np.ndarray:
    probs = _align(probabilities, BASELINE_LABELS)
    index = {label: j for j, label in enumerate(BASELINE_LABELS)}
    labels = np.asarray(y, dtype=int)
    true_prob = probs[np.arange(len(labels)), [index[int(v)] for v in labels]]
    return -np.log(np.clip(true_prob, EPS, 1.0))


def _per_row_brier(y, probabilities: np.ndarray) -> np.ndarray:
    probs = _align(probabilities, BASELINE_LABELS)
    labels = np.asarray(y, dtype=int)
    one_hot = (labels[:, None] == np.asarray(BASELINE_LABELS)[None, :]).astype(float)
    return np.sum((probs - one_hot) ** 2, axis=1)


def bootstrap_intervals(
    y,
    probabilities: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = RANDOM_STATE,
) -> dict:
    """Percentile bootstrap intervals for log-loss, Brier and accuracy."""
    labels = np.asarray(y, dtype=int)
    probs = _align(probabilities, BASELINE_LABELS)
    n = len(labels)
    result = {
        "log_loss_ci_low": None,
        "log_loss_ci_high": None,
        "brier_ci_low": None,
        "brier_ci_high": None,
        "accuracy_ci_low": None,
        "accuracy_ci_high": None,
        "bootstrap_draws": int(n_boot) if n > 1 else 0,
    }
    if n < 2 or n_boot <= 0:
        return result
    log_loss = _per_row_log_loss(labels, probs)
    brier = _per_row_brier(labels, probs)
    predicted = np.asarray(BASELINE_LABELS)[np.argmax(probs, axis=1)]
    correct = (predicted == labels).astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(n_boot), n))
    log_loss_draws = log_loss[indices].mean(axis=1)
    brier_draws = brier[indices].mean(axis=1)
    accuracy_draws = correct[indices].mean(axis=1)
    result.update({
        "log_loss_ci_low": float(np.quantile(log_loss_draws, 0.025)),
        "log_loss_ci_high": float(np.quantile(log_loss_draws, 0.975)),
        "brier_ci_low": float(np.quantile(brier_draws, 0.025)),
        "brier_ci_high": float(np.quantile(brier_draws, 0.975)),
        "accuracy_ci_low": float(np.quantile(accuracy_draws, 0.025)),
        "accuracy_ci_high": float(np.quantile(accuracy_draws, 0.975)),
    })
    return result


def paired_bootstrap_delta(
    y,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    n_boot: int = 500,
    seed: int = RANDOM_STATE,
) -> dict:
    """Paired bootstrap interval for candidate minus baseline log-loss."""
    labels = np.asarray(y, dtype=int)
    if len(labels) < 2 or len(candidate) != len(baseline):
        return {"delta_ci_low": None, "delta_ci_high": None, "delta_ci_excludes_zero": None}
    candidate_loss = _per_row_log_loss(labels, candidate)
    baseline_loss = _per_row_log_loss(labels, baseline)
    deltas = candidate_loss - baseline_loss
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(int(n_boot), len(deltas)))
    draws = deltas[indices].mean(axis=1)
    low = float(np.quantile(draws, 0.025))
    high = float(np.quantile(draws, 0.975))
    return {
        "delta_log_loss_vs_market": float(deltas.mean()),
        "delta_ci_low": low,
        "delta_ci_high": high,
        "delta_ci_excludes_zero": bool(low > 0 or high < 0),
    }


def _metrics_row(
    name: str,
    split: str,
    y,
    probabilities: np.ndarray,
    *,
    n_boot: int,
    market_probabilities: np.ndarray | None = None,
    extra: dict | None = None,
) -> dict:
    metrics = evaluate_probabilities(y, probabilities, BASELINE_LABELS, name)
    row = metrics_to_row(metrics)
    row.update(bootstrap_intervals(y, probabilities, n_boot=n_boot))
    row["model_family"] = MODEL_FAMILIES.get(name, "other")
    row["split"] = split
    row["provenance"] = "holdout" if split in {"validation", "test"} else split
    if market_probabilities is not None:
        row.update(paired_bootstrap_delta(y, probabilities, market_probabilities, n_boot=n_boot))
    row.update(extra or {})
    return row


def _confidence_bin(confidence: float) -> str:
    if confidence < 0.40:
        return "<40%"
    if confidence < 0.50:
        return "40-50%"
    if confidence < 0.60:
        return "50-60%"
    if confidence < 0.70:
        return "60-70%"
    return ">=70%"


def breakdown_rows(
    name: str,
    split: str,
    y,
    probabilities: np.ndarray,
    meta: pd.DataFrame,
    *,
    min_samples: int = MIN_BREAKDOWN_SAMPLES,
) -> list[dict]:
    """Metrics by season, team, predicted side and confidence bin.

    A row below ``min_samples`` is marked ``insufficient_sample`` and carries
    no metric values, so it cannot be read as a conclusion.
    """
    labels = np.asarray(y, dtype=int)
    probs = _align(probabilities, BASELINE_LABELS)
    predicted = np.asarray(BASELINE_LABELS)[np.argmax(probs, axis=1)]
    confidence = probs.max(axis=1)
    frame = pd.DataFrame({
        "y": labels,
        "predicted": predicted,
        "confidence": confidence,
    })
    frame["season"] = meta["season"].astype(str).to_numpy() if "season" in meta else ""
    frame["home_team"] = meta["home_team"].astype(str).to_numpy() if "home_team" in meta else ""
    frame["away_team"] = meta["away_team"].astype(str).to_numpy() if "away_team" in meta else ""
    frame["confidence_bin"] = frame["confidence"].map(_confidence_bin)

    groups: list[tuple[str, str, pd.Series]] = []
    for season in sorted(frame["season"].unique()):
        groups.append(("season", str(season), frame["season"] == season))
    if {"home_team", "away_team"} <= set(frame.columns):
        for team in sorted(set(frame["home_team"]) | set(frame["away_team"])):
            mask = (frame["home_team"] == team) | (frame["away_team"] == team)
            groups.append(("team", team, mask))
    for side, label in (("predicted_side", "Home Win"), ("predicted_side", "Away Win")):
        groups.append((side, label, frame["predicted"] == BASELINE_LABELS[0 if label == "Home Win" else 2]))
    for bucket in ["<40%", "40-50%", "50-60%", "60-70%", ">=70%"]:
        groups.append(("confidence", bucket, frame["confidence_bin"] == bucket))

    rows: list[dict] = []
    for dimension, value, mask in groups:
        part = frame[mask]
        n = int(len(part))
        row: dict[str, Any] = {
            "model": name,
            "split": split,
            "dimension": dimension,
            "value": value,
            "n": n,
            "status": "ok" if n >= min_samples else "insufficient_sample",
            "min_samples": int(min_samples),
        }
        if n >= min_samples and part["y"].nunique() >= 2:
            metrics = evaluate_probabilities(
                part["y"].to_numpy(),
                probs[mask.to_numpy()],
                BASELINE_LABELS,
                f"{name}:{dimension}:{value}",
            )
            row.update(metrics_to_row(metrics))
        else:
            row.update({
                "accuracy": None, "log_loss": None, "brier_score": None,
                "macro_f1": None, "draw_recall": None, "ece": None,
            })
        rows.append(row)
    return rows


def _walk_forward_rows(
    *,
    raw: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    elo: pd.DataFrame,
    params: dict[str, dict],
    poisson_config: PoissonConfig,
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    folds = [
        fold for fold in walk_forward_folds(meta, min_train_seasons=1)
        if fold.test_season != TEST_SEASON
    ]
    if not folds:
        logger.warning("No walk-forward folds available outside the test season.")
        return pd.DataFrame()
    equal_weights = {name: 1.0 / len(ENSEMBLE_COMPONENTS) for name in ENSEMBLE_COMPONENTS}
    for fold in folds:
        train_idx, test_idx = fold.train_index, fold.test_index
        raw_train = raw.set_index("match_id", drop=False).loc[
            meta.loc[train_idx, "match_id"].to_numpy()
        ].reset_index(drop=True)
        predictions = _window_predictions(
            raw_train=raw_train,
            train_meta=meta.loc[train_idx],
            eval_meta=meta.loc[test_idx],
            X_train=X.loc[train_idx],
            y_train=y.loc[train_idx],
            X_eval=X.loc[test_idx],
            elo_train=elo.loc[train_idx],
            elo_eval=elo.loc[test_idx],
            params=params,
            poisson_config=poisson_config,
        )
        predictions["Ensemble (equal)"] = blend_probabilities(
            {name: predictions[name] for name in ENSEMBLE_COMPONENTS}, equal_weights
        )
        for name, probabilities in predictions.items():
            rows.append(_metrics_row(
                name, "walk_forward", y.loc[test_idx], probabilities,
                n_boot=n_boot,
                extra={
                    "fold": int(fold.fold),
                    "test_season": fold.test_season,
                    "train_seasons": "+".join(fold.train_seasons),
                    "ensemble_weights": (
                        "equal (no future-season weights)"
                        if name == "Ensemble (equal)" else ""
                    ),
                },
            ))
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No data._"
    columns = [c for c in columns if c in frame.columns]
    if not columns:
        return "_No data._"
    part = frame.head(max_rows) if max_rows else frame
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in part.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                cells.append("")
            elif isinstance(value, float):
                if column.startswith(("accuracy", "draw_recall")):
                    cells.append(f"{value:.1%}")
                elif column in {"log_loss", "brier_score", "ece", "log_loss_ci_low",
                                "log_loss_ci_high", "brier_ci_low", "brier_ci_high",
                                "delta_log_loss_vs_market", "delta_ci_low", "delta_ci_high"}:
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(f"{value:.3g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_full_season_report(
    *,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    walk_forward: pd.DataFrame,
    summary: dict,
    path: Path = FULL_SEASON_REPORT_PATH,
) -> Path:
    best_val = summary.get("validation_best_learned_model")
    best_test = summary.get("test_best_log_loss_model")
    production = summary.get("production_reference", {})
    lines = [
        "# La Liga Predictor — Full-Season Out-of-Sample Evaluation",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Protocol",
        "",
        f"- Train seasons: {', '.join(summary['protocol']['train_seasons'])}",
        f"- Validation season: {summary['protocol']['validation_season']}",
        f"- Untouched test season: {summary['protocol']['test_season']}",
        "- Model selection and ensemble weights use validation only. "
        f"Test used for selection: **{summary['protocol']['test_used_for_selection']}**.",
        "- Metrics carry bootstrap confidence intervals; paired deltas are against "
        "row-normalised market probabilities.",
        "- A test point estimate is not a claim of improvement on its own.",
        "",
        "## Headline",
        "",
    ]
    if best_val:
        lines.append(
            f"- Best **validation** learned model: **{best_val['model']}** "
            f"(log-loss {best_val['log_loss']:.4f}, Brier {best_val['brier_score']:.4f}, "
            f"accuracy {best_val['accuracy']:.1%})."
        )
    if best_test:
        lines.append(
            f"- Best **test** log-loss: **{best_test['model']}** "
            f"({best_test['log_loss']:.4f}). This is confirmatory, not a selection signal."
        )
    market_test = test[test["model"] == "Market odds"]
    if not market_test.empty:
        row = market_test.iloc[0]
        lines.append(
            f"- Market test log-loss: **{row['log_loss']:.4f}** "
            f"(Brier {row['brier_score']:.4f}, accuracy {row['accuracy']:.1%})."
        )
    if production:
        lines.append(
            f"- Production reference (unchanged): `{production.get('version', 'unknown')}` "
            f"({production.get('model_name', 'unknown')})."
        )
    lines += [
        "",
        "## Validation",
        "",
        _markdown_table(
            validation,
            ["model", "n_samples", "accuracy", "log_loss", "log_loss_ci_low",
             "log_loss_ci_high", "brier_score", "macro_f1", "draw_recall", "ece"],
        ),
        "",
        "## Test (single scoring pass)",
        "",
        _markdown_table(
            test,
            ["model", "n_samples", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
             "log_loss", "log_loss_ci_low", "log_loss_ci_high", "brier_score",
             "macro_f1", "draw_recall", "ece", "delta_log_loss_vs_market",
             "delta_ci_low", "delta_ci_high"],
        ),
        "",
        "## Walk-forward (test season excluded)",
        "",
        _markdown_table(
            walk_forward.sort_values(["fold", "log_loss"]) if not walk_forward.empty else walk_forward,
            ["fold", "test_season", "model", "n_samples", "accuracy", "log_loss",
             "brier_score", "macro_f1", "draw_recall", "ece"],
            max_rows=80,
        ),
        "",
        "## Breakdowns",
        "",
        "Per-season, per-team, predicted-side and confidence-bin rows are in "
        f"`{summary['artifacts']['breakdowns']}`. Rows with fewer than "
        f"{summary['protocol']['min_breakdown_samples']} matches are marked "
        "`insufficient_sample` and contain no metrics.",
        "",
        "## Limitations",
        "",
        "- The test season is in progress, so its sample is small and test metrics are provisional.",
        "- Football outcomes are noisy; confidence intervals overlap for close models.",
        "- Market probabilities are inputs to the tree models, so learned models are not independent forecasts.",
        "- No betting edge is claimed: there is no closing-line or flat-stake out-of-sample audit here.",
        "- Historical xG and player availability are unavailable in this repository; no improvement from them is claimed.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class FullSeasonResult:
    validation: pd.DataFrame
    test: pd.DataFrame
    walk_forward: pd.DataFrame
    breakdowns: pd.DataFrame
    summary: dict


def evaluate_full_season(
    raw: pd.DataFrame | None = None,
    *,
    metadata: dict | None = None,
    n_boot: int = FULL_SEASON_BOOTSTRAP_DRAWS,
    save: bool = True,
    paths: dict[str, Path] | None = None,
) -> FullSeasonResult:
    """Run the full chronological comparison and optionally persist artifacts."""
    raw = load_raw() if raw is None else raw.copy()
    if metadata is None and MODEL_METADATA_PATH.exists():
        try:
            metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - metadata is an input, not a requirement
            logger.warning("Could not read model metadata: %s", exc)
            metadata = {}
    metadata = metadata or {}
    params = _params(metadata)

    X, y, meta = build_model_frame(raw)
    elo, _elo_history = compute_elo(raw)
    elo = elo.reindex(meta["match_id"].to_numpy())
    elo.index = X.index

    splits = chronological_split(
        meta,
        train_seasons=TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        test_season=TEST_SEASON,
    )
    train_idx = splits["train"].index
    val_idx = splits["validation"].index
    test_idx = splits["test"].index
    hist_idx = train_idx.append(val_idx)
    raw_lookup = raw.set_index("match_id", drop=False)

    def raw_for(index) -> pd.DataFrame:
        return raw_lookup.loc[meta.loc[index, "match_id"].to_numpy()].reset_index(drop=True)

    poisson_config = PoissonConfig()
    val_predictions = _window_predictions(
        raw_train=raw_for(train_idx),
        train_meta=meta.loc[train_idx],
        eval_meta=meta.loc[val_idx],
        X_train=X.loc[train_idx],
        y_train=y.loc[train_idx],
        X_eval=X.loc[val_idx],
        elo_train=elo.loc[train_idx],
        elo_eval=elo.loc[val_idx],
        params=params,
        poisson_config=poisson_config,
    )
    selection = select_ensemble_weights(
        {name: val_predictions[name] for name in ENSEMBLE_COMPONENTS},
        y.loc[val_idx].to_numpy(),
        step=0.05,
    )
    equal_weights = {name: float(selection["equal_weights"][name]) for name in ENSEMBLE_COMPONENTS}
    # ``select_ensemble_weights`` may omit zero components; report every
    # component explicitly so the stored weights are a complete simplex.
    raw_weights = selection.get("all_weights") or selection["weights"]
    validation_weights = {
        name: float(raw_weights.get(name, 0.0)) for name in ENSEMBLE_COMPONENTS
    }
    val_predictions["Ensemble (equal)"] = blend_probabilities(
        {name: val_predictions[name] for name in ENSEMBLE_COMPONENTS}, equal_weights
    )
    val_predictions["Ensemble (validation)"] = blend_probabilities(
        {name: val_predictions[name] for name in ENSEMBLE_COMPONENTS}, validation_weights
    )

    test_predictions = _window_predictions(
        raw_train=raw_for(hist_idx),
        train_meta=meta.loc[hist_idx],
        eval_meta=meta.loc[test_idx],
        X_train=X.loc[hist_idx],
        y_train=y.loc[hist_idx],
        X_eval=X.loc[test_idx],
        elo_train=elo.loc[hist_idx],
        elo_eval=elo.loc[test_idx],
        params=params,
        poisson_config=poisson_config,
    )
    # The validation-selected weights are applied unchanged to test.
    test_predictions["Ensemble (equal)"] = blend_probabilities(
        {name: test_predictions[name] for name in ENSEMBLE_COMPONENTS}, equal_weights
    )
    test_predictions["Ensemble (validation)"] = blend_probabilities(
        {name: test_predictions[name] for name in ENSEMBLE_COMPONENTS}, validation_weights
    )

    market_val = val_predictions["Market odds"]
    market_test = test_predictions["Market odds"]
    validation_rows = [
        _metrics_row(
            name, "validation", y.loc[val_idx], probabilities,
            n_boot=n_boot,
            market_probabilities=market_val,
            extra={"ensemble_weights": (
                "validation-selected" if name == "Ensemble (validation)"
                else "equal" if name == "Ensemble (equal)" else ""
            )},
        )
        for name, probabilities in val_predictions.items()
    ]
    test_rows = [
        _metrics_row(
            name, "test", y.loc[test_idx], probabilities,
            n_boot=n_boot,
            market_probabilities=market_test,
            extra={"ensemble_weights": (
                "validation-selected" if name == "Ensemble (validation)"
                else "equal" if name == "Ensemble (equal)" else ""
            )},
        )
        for name, probabilities in test_predictions.items()
    ]
    validation = pd.DataFrame(validation_rows).sort_values(
        ["log_loss", "brier_score"]
    ).reset_index(drop=True)
    test = pd.DataFrame(test_rows).sort_values(
        ["log_loss", "brier_score"]
    ).reset_index(drop=True)

    breakdown_frames = []
    for split_name, predictions, index in (
        ("validation", val_predictions, val_idx),
        ("test", test_predictions, test_idx),
    ):
        split_meta = meta.loc[index].reset_index(drop=True)
        for name, probabilities in predictions.items():
            breakdown_frames.append(pd.DataFrame(breakdown_rows(
                name, split_name, y.loc[index], probabilities, split_meta
            )))
    breakdowns = pd.concat(breakdown_frames, ignore_index=True, sort=False) if breakdown_frames else pd.DataFrame()

    walk_forward = _walk_forward_rows(
        raw=raw, X=X, y=y, meta=meta, elo=elo, params=params,
        poisson_config=poisson_config, n_boot=n_boot,
    )

    learned_validation = validation[validation["model"] != "Market odds"]
    best_validation = learned_validation.iloc[0].to_dict() if not learned_validation.empty else None
    best_test = test.iloc[0].to_dict() if not test.empty else None
    artifacts = {
        "metrics": str(FULL_SEASON_METRICS_PATH),
        "breakdowns": str(FULL_SEASON_BREAKDOWNS_PATH),
        "walk_forward": str(FULL_SEASON_WALK_FORWARD_PATH),
        "summary": str(FULL_SEASON_SUMMARY_PATH),
        "report": str(FULL_SEASON_REPORT_PATH),
    }
    summary = {
        "generated_at": _now(),
        "protocol": {
            "train_seasons": list(TRAIN_SEASONS),
            "validation_season": VALIDATION_SEASON,
            "test_season": TEST_SEASON,
            "test_used_for_selection": False,
            "selection_rule": "lowest validation log-loss; Brier breaks ties",
            "ensemble_weights_selected_on": "validation only",
            "min_breakdown_samples": MIN_BREAKDOWN_SAMPLES,
            "bootstrap_draws": int(n_boot),
        },
        "data": {
            "rows": int(len(raw)),
            "seasons": sorted(meta["season"].astype(str).unique().tolist()),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "date_min": str(pd.to_datetime(raw["date"]).min().date()),
            "date_max": str(pd.to_datetime(raw["date"]).max().date()),
        },
        "ensemble": {
            "weights": {k: float(v) for k, v in validation_weights.items()},
            "equal_weights": {k: float(v) for k, v in equal_weights.items()},
            "validation_log_loss": float(selection["validation_log_loss"]),
            "equal_validation_log_loss": float(selection["equal_validation_log_loss"]),
        },
        "validation_best_learned_model": {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in (best_validation or {}).items()
            if k in {"model", "log_loss", "brier_score", "accuracy", "macro_f1", "draw_recall", "ece"}
        } if best_validation else None,
        "test_best_log_loss_model": {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in (best_test or {}).items()
            if k in {"model", "log_loss", "brier_score", "accuracy", "macro_f1", "draw_recall", "ece"}
        } if best_test else None,
        "production_reference": {
            "version": metadata.get("model_version"),
            "model_name": metadata.get("model_name"),
            "status": "unchanged",
        },
        "provenance": {
            "validation": "holdout season, used for selection",
            "test": "untouched final season, scored once",
            "walk_forward": "rolling-origin folds, final test excluded",
            "live": "not measured in this report",
        },
        "artifacts": artifacts,
        "notes": [
            "Test metrics confirm the validation decision; they are not used to choose a model.",
            "Intervals overlap for close models; no improvement is claimed from a point estimate.",
            "Market odds are a benchmark and also a tree-model input.",
            "Historical xG and player availability were unavailable; no such improvement is claimed.",
        ],
    }

    if save:
        target = dict(artifacts)
        target.update({k: str(v) for k, v in (paths or {}).items()})
        metrics_path = Path(target["metrics"])
        breakdown_path = Path(target["breakdowns"])
        walk_path = Path(target["walk_forward"])
        summary_path = Path(target["summary"])
        report_path = Path(target["report"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        validation.assign(split="validation").pipe(
            lambda frame: pd.concat([
                frame, test.assign(split="test")
            ], ignore_index=True, sort=False)
        ).to_csv(metrics_path, index=False)
        breakdowns.to_csv(breakdown_path, index=False)
        walk_forward.to_csv(walk_path, index=False)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        write_full_season_report(
            validation=validation, test=test, walk_forward=walk_forward,
            summary=summary, path=report_path,
        )
        summary["artifacts"] = {k: str(Path(v)) for k, v in target.items()}

    return FullSeasonResult(
        validation=validation,
        test=test,
        walk_forward=walk_forward,
        breakdowns=breakdowns,
        summary=summary,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Full-season out-of-sample evaluation")
    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts")
    parser.add_argument("--bootstrap", type=int, default=FULL_SEASON_BOOTSTRAP_DRAWS)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    result = evaluate_full_season(n_boot=max(args.bootstrap, 0), save=not args.no_save)
    print(result.test[[
        "model", "n_samples", "accuracy", "log_loss", "log_loss_ci_low",
        "log_loss_ci_high", "brier_score", "macro_f1", "draw_recall", "ece",
    ]].to_string(index=False))
    print(f"\nValidation-selected model: {result.summary['validation_best_learned_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
