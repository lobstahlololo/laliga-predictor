"""Shared inference engine used by the CLI and the Streamlit dashboard."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.uncertainty import model_agreement, normalized_entropy
from src.features.build_features import build_team_history, rest_days_as_of, short_rest_flag
from src.features.preprocessing import DEFAULT_PREPROCESSOR_PATH, FeaturePreprocessor
from src.models.poisson import PoissonConfig, PoissonModel
from src.utils.config import (
    AVAILABILITY_FEATURE_COLUMNS,
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    MODEL_METADATA_PATH,
    PHASE3_SUMMARY_PATH,
    RAW_PATH,
    ROLLING_WINDOW,
    SHORT_WINDOW,
    VALUE_BET_THRESHOLD,
    XG_FEATURE_COLUMNS,
)
from src.utils.teams import TeamNotFoundError, canonical_team, resolve_team

logger = logging.getLogger(__name__)

OUTCOME_LABELS = {2: "Home Win", 1: "Draw", 0: "Away Win"}
# Probability columns are ordered [home, draw, away]; labels must follow the
# same positional order (not the numeric outcome codes).
PREDICTION_LABELS = ["Home Win", "Draw", "Away Win"]

# Neutral fallback market when neither supplied odds nor historical means
# exist. Implied probabilities are 1 / typical odds.
FALLBACK_MARKET = {
    "implied_prob_home": 1 / 2.20,
    "implied_prob_draw": 1 / 3.40,
    "implied_prob_away": 1 / 3.80,
}


class ArtifactError(RuntimeError):
    """Raised when model/scaler/data artifacts are missing or inconsistent."""


def load_artifacts():
    """Load the production model, preprocessor and metadata.

    When the registry has a production version it is the source of truth, so a
    newly trained candidate cannot silently change live predictions. Before the
    first registration the legacy live paths are used.
    """
    model_path = BEST_MODEL_PATH
    pre_path = DEFAULT_PREPROCESSOR_PATH
    metadata_path = MODEL_METADATA_PATH
    try:
        from src.models.registry import production_paths

        resolved = production_paths()
    except Exception as exc:  # noqa: BLE001 - registry is optional for inference
        logger.warning("Model registry unavailable, using live artifacts: %s", exc)
        resolved = {}
    if resolved:
        model_path = resolved["model"]
        metadata_path = resolved["metadata"]
        if resolved.get("preprocessor"):
            pre_path = resolved["preprocessor"]

    for path, hint in (
        (model_path, "Register a model with `python -m src.models.registry` or run training."),
        (pre_path, "Run `python -m src.features.build_features` first."),
        (RAW_PATH, "Run `python -m src.data.make_dataset` first."),
    ):
        if not path.exists():
            raise ArtifactError(f"Missing {path}. {hint}")
    model = joblib.load(model_path)
    preprocessor = FeaturePreprocessor.load(pre_path)
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - metadata is optional
            logger.warning("Could not read model metadata: %s", exc)
    return model, preprocessor, metadata


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}.")
    return pd.read_csv(path, parse_dates=["date"])


def team_form(history: pd.DataFrame, team: str, as_of=None) -> dict[str, float]:
    """Rolling form using only matches played before ``as_of``."""
    rows = history[history["team"] == team].sort_values("date")
    if as_of is not None:
        rows = rows[rows["date"] < pd.Timestamp(as_of)]
    if rows.empty:
        raise TeamNotFoundError(f"No historical matches found for '{team}'.")

    last5 = rows.tail(ROLLING_WINDOW)
    last3 = rows.tail(SHORT_WINDOW)
    form = {
        "form_goals_scored": float(last5["goals_scored"].mean()),
        "form_goals_conceded": float(last5["goals_conceded"].mean()),
        "form_total_shots": float(last5["total_shots"].mean()),
        "form_shots_on_target": float(last5["shots_on_target"].mean()),
        "short_goals_scored": float(last3["goals_scored"].mean()),
        "short_goals_conceded": float(last3["goals_conceded"].mean()),
    }
    form["goal_diff_5m"] = form["form_goals_scored"] - form["form_goals_conceded"]
    if any(not np.isfinite(v) for v in form.values()):
        raise TeamNotFoundError(f"Incomplete statistics for '{team}'.")
    return form


def neutral_market(raw: pd.DataFrame) -> dict[str, float]:
    """Average historical implied probabilities, or a neutral fallback."""
    probs: dict[str, float] = {}
    for col in FALLBACK_MARKET:
        if col in raw.columns and raw[col].notna().any():
            value = float(raw[col].mean())
            if np.isfinite(value):
                probs[col] = value
    for col, fallback in FALLBACK_MARKET.items():
        probs.setdefault(col, fallback)
    return probs


def value_bet_outcomes(
    probs: dict[str, float], market: dict[str, float]
) -> list[tuple[str, float, float, float]]:
    """Outcomes beating the market by more than ``VALUE_BET_THRESHOLD``."""
    hits = []
    for outcome in ("home", "draw", "away"):
        model_p = probs.get(outcome, 0.0)
        market_p = market.get(f"implied_prob_{outcome}", 0.0)
        gap = model_p - market_p
        if gap > VALUE_BET_THRESHOLD:
            hits.append((outcome, model_p, market_p, gap))
    return hits


OPTIONAL_FEATURE_COLUMNS = list(XG_FEATURE_COLUMNS) + list(AVAILABILITY_FEATURE_COLUMNS)


def build_feature_row(
    home_form: dict[str, float],
    away_form: dict[str, float],
    market: dict[str, float],
    headers: list[str],
    short_rest: int = 0,
    extras: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assemble one feature row in the exact preprocessing column order."""
    extras = extras or {}
    row: dict[str, float] = {}
    for col in headers:
        if col in extras:
            row[col] = float(extras[col])
        elif col == "home_advantage":
            row[col] = 1.0
        elif col == "short_rest_advantage":
            row[col] = float(short_rest)
        elif col.startswith("implied_prob_"):
            row[col] = market[col]
        elif col.startswith("home_"):
            row[col] = home_form[col[len("home_"):]]
        elif col.startswith("away_"):
            row[col] = away_form[col[len("away_"):]]
        else:
            raise KeyError(f"Unrecognized feature column: {col!r}")
    return pd.DataFrame([row], columns=headers)


def _market_from_odds(odds) -> dict[str, float]:
    if odds is None:
        return {}
    return {
        "implied_prob_home": 1.0 / float(odds[0]),
        "implied_prob_draw": 1.0 / float(odds[1]),
        "implied_prob_away": 1.0 / float(odds[2]),
    }


def goal_model_config() -> PoissonConfig:
    """Reuse the validation-selected Phase 3 Poisson settings when available.

The configuration was chosen on validation data only; falling back to the
    dataclass defaults keeps the engine usable before Phase 3 has run.
    """
    fields = set(PoissonConfig.__dataclass_fields__)
    if PHASE3_SUMMARY_PATH.exists():
        try:
            summary = json.loads(PHASE3_SUMMARY_PATH.read_text(encoding="utf-8"))
            config = summary.get("poisson_selected_config") or {}
            filtered = {k: v for k, v in config.items() if k in fields}
            if filtered:
                return PoissonConfig(**filtered)
        except Exception as exc:  # noqa: BLE001 - config is an optimisation
            logger.warning("Could not read Phase 3 Poisson config: %s", exc)
    return PoissonConfig()


def fit_goal_model(raw: pd.DataFrame, as_of) -> PoissonModel:
    """Fit the goal model strictly on matches before ``as_of``.

    Date-only data cannot distinguish kickoff times, so matches on the same
    calendar date are excluded rather than assumed to be known.
    """
    cutoff = pd.Timestamp(as_of).normalize()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["date"] < cutoff]
    return PoissonModel(goal_model_config()).fit(frame)


def _uncertainty_level(entropy: float, agreement: float, confidence: float) -> str:
    """Descriptive level, mirroring the Phase 3 uncertainty rule."""
    if entropy >= 0.90 or agreement <= 0.65 or confidence < 0.45:
        return "high"
    if entropy >= 0.80 or agreement <= 0.80 or confidence < 0.55:
        return "medium"
    return "low"


def attach_goal_context(
    predictions: list[dict],
    raw: pd.DataFrame,
    as_of: pd.Timestamp,
) -> None:
    """Attach expected goals, scorelines and uncertainty in place.

    The goal model is refit per distinct fixture cutoff from earlier matches
    only. A fitting failure is reported per prediction instead of fabricating
    goal estimates.
    """
    if not predictions:
        return
    cuts: dict[pd.Timestamp, PoissonModel] = {}
    for prediction in predictions:
        cutoff = pd.Timestamp(as_of).normalize()
        if prediction.get("match_date"):
            parsed = pd.to_datetime(prediction["match_date"], errors="coerce")
            if pd.notna(parsed):
                cutoff = parsed.normalize()
        try:
            if cutoff not in cuts:
                cuts[cutoff] = fit_goal_model(raw, cutoff)
            goal = cuts[cutoff].predict_fixture(
                prediction["home_team"], prediction["away_team"]
            )
            probs = np.array(
                [goal["prob_home"], goal["prob_draw"], goal["prob_away"]],
                dtype=float,
            )
            rf_probs = np.array([
                prediction["probs"]["home"],
                prediction["probs"]["draw"],
                prediction["probs"]["away"],
            ], dtype=float)
            entropy = float(normalized_entropy(rf_probs[None, :])[0])
            agreement = float(
                model_agreement([rf_probs[None, :], probs[None, :]])[0]
            )
            scorelines = cuts[cutoff].scoreline_table(
                prediction["home_team"], prediction["away_team"], top_n=5
            )
            prediction.update({
                "expected_home_goals": float(goal["lambda_home"]),
                "expected_away_goals": float(goal["lambda_away"]),
                "most_likely_scoreline": goal["most_likely_score"],
                "scoreline_probability": float(goal["most_likely_score_probability"]),
                "top_scorelines": [
                    {
                        "scoreline": str(row["scoreline"]),
                        "probability": float(row["probability"]),
                    }
                    for _, row in scorelines.iterrows()
                ],
                "goal_probabilities": {
                    "home": float(probs[0]),
                    "draw": float(probs[1]),
                    "away": float(probs[2]),
                },
                "goal_model": {
                    "name": "Dixon-Coles" if abs(goal["rho"]) > 0 else "Independent Poisson",
                    "rho": float(goal["rho"]),
                    "as_of": f"{cutoff:%Y-%m-%d}",
                    "home_known": bool(goal["home_known"]),
                    "away_known": bool(goal["away_known"]),
                    "home_matches": int(goal["home_matches"]),
                    "away_matches": int(goal["away_matches"]),
                },
                "uncertainty": {
                    "entropy": entropy,
                    "agreement": agreement,
                    "level": _uncertainty_level(
                        entropy, agreement, float(max(prediction["probs"].values()))
                    ),
                    "note": (
                        "Descriptive confidence/entropy/agreement diagnostics, "
                        "not a calibrated probability interval."
                    ),
                },
            })
        except Exception as exc:  # noqa: BLE001 - never fake goal estimates
            logger.warning(
                "Goal model unavailable for %s vs %s: %s",
                prediction.get("home_team"), prediction.get("away_team"), exc,
            )
            prediction["goal_model_error"] = str(exc)


def predictions_to_records(predictions: list[dict]) -> list[dict]:
    """Convert engine output into append-only ledger records.

    Shared by the CLI, the API and the automation pipeline so all three write
    identical history rows.
    """
    from src.predictions.history import build_prediction_record

    records = []
    for prediction in predictions:
        probs = prediction["probs"]
        market = prediction["market"]
        records.append(build_prediction_record(
            home_team=prediction["home_team"],
            away_team=prediction["away_team"],
            match_date=prediction["match_date"],
            predicted_at=prediction["predicted_at"],
            model_name=prediction["model_name"],
            model_version=prediction["model_version"],
            prob_home=probs["home"],
            prob_draw=probs["draw"],
            prob_away=probs["away"],
            market_probs={
                "home": market["implied_prob_home"],
                "draw": market["implied_prob_draw"],
                "away": market["implied_prob_away"],
            },
            prediction_cycle=prediction.get("prediction_cycle", ""),
            kickoff_utc=prediction.get("kickoff_utc", ""),
            fixture_source=prediction.get("fixture_source", ""),
            expected_home_goals=prediction.get("expected_home_goals"),
            expected_away_goals=prediction.get("expected_away_goals"),
            most_likely_scoreline=prediction.get("most_likely_scoreline", ""),
        ))
    return records


def predict_fixtures(
    fixtures: list[dict],
    *,
    model=None,
    preprocessor: FeaturePreprocessor | None = None,
    metadata: dict | None = None,
    raw: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    predicted_at: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Predict a batch of fixtures.

    Returns ``(predictions, errors)``. Each prediction includes probabilities,
    market comparison, value bets and the form inputs used, so the dashboard
    and CLI show exactly what the model saw. Fixtures that cannot be resolved
    are returned as errors, never fabricated.
    """
    if model is None or preprocessor is None:
        model, preprocessor, loaded_meta = load_artifacts()
        metadata = metadata or loaded_meta
    metadata = metadata or {}
    raw = load_raw() if raw is None else raw
    history = build_team_history(raw)
    available = sorted(
        set(raw["home_team"].astype(str)) | set(raw["away_team"].astype(str))
    )
    default_market = neutral_market(raw)
    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    timestamp = predicted_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    headers = list(preprocessor.feature_names)
    missing_baseline = [c for c in FEATURE_COLUMNS if c not in headers]
    unknown_features = [
        c for c in headers
        if c not in FEATURE_COLUMNS and c not in OPTIONAL_FEATURE_COLUMNS
    ]
    if missing_baseline or unknown_features:
        raise ArtifactError(
            "Preprocessor features do not match the supported contract "
            f"(missing={missing_baseline}, unknown={unknown_features}). "
            "Retrain the model."
        )

    optional_columns = [c for c in headers if c not in FEATURE_COLUMNS]
    xg_source = availability_source = None
    if set(optional_columns) & set(XG_FEATURE_COLUMNS):
        from src.features.optional_features import load_xg_history

        xg_source = load_xg_history()
        if xg_source.empty:
            raise ArtifactError(
                "This model expects xG features but no timestamped xG history "
                "is available. Supply data/raw/xg_history.csv or retrain on the "
                "baseline feature set."
            )
    if set(optional_columns) & set(AVAILABILITY_FEATURE_COLUMNS):
        from src.features.optional_features import load_player_availability

        availability_source = load_player_availability()
        if availability_source.empty:
            raise ArtifactError(
                "This model expects player-availability features but no "
                "timestamped availability data is available. Supply "
                "data/raw/player_availability.csv or retrain on the baseline "
                "feature set."
            )

    rows, prepared, errors = [], [], []
    for fixture in fixtures:
        home_q = str(fixture.get("home_team", "")).strip()
        away_q = str(fixture.get("away_team", "")).strip()
        try:
            home = resolve_team(home_q, available)
            away = resolve_team(away_q, available)
            if home == away:
                raise TeamNotFoundError("Home and away resolve to the same team.")

            match_date = str(fixture.get("match_date", "") or "").strip()
            market = dict(default_market)
            market.update(_market_from_odds(fixture.get("odds")))
            market_note = (
                "odds supplied by user" if fixture.get("odds")
                else "historical average market"
            )

            form_as_of = (
                pd.Timestamp(match_date) if match_date else as_of
            )
            home_form = team_form(history, home, as_of=form_as_of)
            away_form = team_form(history, away, as_of=form_as_of)
            home_rest = rest_days_as_of(history, home, form_as_of)
            away_rest = rest_days_as_of(history, away, form_as_of)

            extras: dict[str, float] = {}
            if optional_columns:
                from src.features.optional_features import fixture_optional_features

                extras = fixture_optional_features(
                    raw, home, away, form_as_of, optional_columns,
                    xg=xg_source, availability=availability_source,
                )

            rows.append(build_feature_row(
                home_form, away_form, market, headers,
                short_rest=short_rest_flag(home_rest, away_rest),
                extras=extras,
            ))
            prepared.append({
                "home_team": home,
                "away_team": away,
                "match_date": match_date,
                "market": market,
                "market_note": market_note,
                "home_form": home_form,
                "away_form": away_form,
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "short_rest_advantage": short_rest_flag(home_rest, away_rest),
            })
        except (ValueError, KeyError, TeamNotFoundError) as exc:
            errors.append({"fixture": fixture, "error": str(exc)})

    if not rows:
        return [], errors

    matrix = pd.concat(rows, ignore_index=True)
    scaled = preprocessor.transform(matrix)
    proba = model.predict_proba(scaled)
    class_index = {int(c): i for i, c in enumerate(model.classes_)}

    predictions = []
    for meta, probs_row in zip(prepared, proba):
        probs_array = np.array([
            probs_row[class_index.get(2, 0)],
            probs_row[class_index.get(1, 0)],
            probs_row[class_index.get(0, 0)],
        ], dtype=float)
        probs_array = np.clip(probs_array, 0, None)
        probs_array = probs_array / probs_array.sum()
        probs = {
            "home": float(probs_array[0]),
            "draw": float(probs_array[1]),
            "away": float(probs_array[2]),
        }
        predicted = PREDICTION_LABELS[int(np.argmax(probs_array))]
        predictions.append({
            **meta,
            "probs": probs,
            "predicted_outcome": predicted,
            "confidence": float(max(probs.values())),
            "value_bets": value_bet_outcomes(probs, meta["market"]),
            "fixture_id": (
                f"{pd.Timestamp(meta['match_date']):%Y%m%d}_{meta['home_team']}_v_{meta['away_team']}"
                if meta["match_date"] and pd.notna(pd.to_datetime(meta["match_date"], errors="coerce"))
                else f"TBD_{meta['home_team']}_v_{meta['away_team']}"
            ),
            "model_name": metadata.get("model_name", type(model).__name__),
            "model_version": metadata.get("model_version", "unversioned"),
            "predicted_at": timestamp,
        })
    attach_goal_context(predictions, raw, as_of)
    return predictions, errors
