"""FastAPI service for the La Liga predictor.

The API is a thin transport layer: probabilities, expected goals, ledger
metrics and team statistics all come from the same modules the CLI and the
dashboard use, so the three interfaces cannot disagree.

Run it with:
    python main.py api
    # or
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000

Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.cache import Cache, build_cache, cache_key, cached_call  # noqa: E402
from src.features.build_features import build_team_history  # noqa: E402
from src.models.elo import compute_elo, latest_elo_ratings  # noqa: E402
from src.models.registry import RegistryError, current_production, list_models, status  # noqa: E402
from src.predictions import history as hist  # noqa: E402
from src.predictions.fixtures import load_upcoming_fixtures  # noqa: E402
from src.predictions.predictor import (  # noqa: E402
    ArtifactError,
    fit_goal_model,
    load_artifacts,
    load_raw,
    predict_fixtures,
)
from src.pipeline.automation import load_runs  # noqa: E402
from src.utils.config import (  # noqa: E402
    FEATURE_COLUMNS,
    LEDGER_PATH,
    MODEL_METADATA_PATH,
    RAW_PATH,
)
from src.utils.teams import TeamNotFoundError, canonical_team  # noqa: E402

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class FixtureRequest(BaseModel):
    home_team: str = Field(min_length=1, max_length=64, examples=["Real Madrid"])
    away_team: str = Field(min_length=1, max_length=64, examples=["Valencia"])
    match_date: str | None = Field(
        default=None, description="ISO kickoff date; omitted means TBD.", examples=["2026-11-08"]
    )
    odds_home: float | None = Field(default=None, gt=1.0, le=1000.0)
    odds_draw: float | None = Field(default=None, gt=1.0, le=1000.0)
    odds_away: float | None = Field(default=None, gt=1.0, le=1000.0)

    @model_validator(mode="after")
    def _odds_must_be_complete(self) -> "FixtureRequest":
        provided = [self.odds_home, self.odds_draw, self.odds_away]
        if any(v is not None for v in provided) and not all(v is not None for v in provided):
            raise ValueError("Provide all three odds or none of them.")
        if self.home_team.strip().lower() == self.away_team.strip().lower():
            raise ValueError("Home and away teams must differ.")
        return self

    def as_fixture(self) -> dict:
        odds = None
        if self.odds_home is not None:
            odds = (self.odds_home, self.odds_draw, self.odds_away)
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "match_date": self.match_date or "",
            "odds": odds,
        }


class Scoreline(BaseModel):
    scoreline: str
    probability: float


class Uncertainty(BaseModel):
    entropy: float
    agreement: float
    level: str
    note: str


class GoalModelInfo(BaseModel):
    name: str
    rho: float
    as_of: str
    home_known: bool
    away_known: bool
    home_matches: int
    away_matches: int


class Prediction(BaseModel):
    fixture_id: str
    home_team: str
    away_team: str
    match_date: str
    probabilities: dict[str, float]
    predicted_outcome: str
    confidence: float
    expected_home_goals: float | None = None
    expected_away_goals: float | None = None
    most_likely_scoreline: str | None = None
    scoreline_probability: float | None = None
    top_scorelines: list[Scoreline] = Field(default_factory=list)
    goal_model: GoalModelInfo | None = None
    goal_model_error: str | None = None
    uncertainty: Uncertainty | None = None
    model_name: str
    model_version: str
    predicted_at: str
    market_note: str


class HealthResponse(BaseModel):
    status: str
    api_version: str
    generated_at: str
    model_version: str | None
    artifacts: dict[str, bool]


class ModelResponse(BaseModel):
    production: dict | None
    metadata: dict
    feature_count: int
    feature_names: list[str]


class TeamStatsResponse(BaseModel):
    team: str
    matches: int | None = None
    wins: int | None = None
    draws: int | None = None
    losses: int | None = None
    goals_for: int | None = None
    goals_against: int | None = None
    goals_for_per_match: float | None = None
    goals_against_per_match: float | None = None
    points_per_game: float | None = None
    home_points_per_game: float | None = None
    away_points_per_game: float | None = None
    form: str | None = None
    elo_rating: float | None = None
    elo_rating_deviation: float | None = None
    attack: float | None = None
    defence: float | None = None
    poisson_matches: int | None = None


class PerformanceResponse(BaseModel):
    summary: dict
    by_season: list[dict]
    by_model: list[dict]


class RegistryResponse(BaseModel):
    production_version: str | None
    production_since: str | None
    n_versions: int
    versions: list[dict]


# ---------------------------------------------------------------------------
# Service layer (overridable in tests)
# ---------------------------------------------------------------------------

@dataclass
class Services:
    model: Any
    preprocessor: Any
    metadata: dict
    raw: pd.DataFrame
    history: pd.DataFrame


_SERVICES: Services | None = None
_SERVICE_STAMP: tuple[float, float, float] | None = None


def _stamp() -> tuple[float, float, float]:
    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return (mtime(RAW_PATH), mtime(MODEL_METADATA_PATH), mtime(LEDGER_PATH))


def load_services() -> Services:
    """Load model/raw/ledger once, reloading when artifacts change on disk."""
    global _SERVICES, _SERVICE_STAMP
    stamp = _stamp()
    if _SERVICES is not None and stamp == _SERVICE_STAMP:
        return _SERVICES
    try:
        model, preprocessor, metadata = load_artifacts()
    except ArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    history, _ = hist.load_history()
    _SERVICES = Services(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata or {},
        raw=load_raw(),
        history=history,
    )
    _SERVICE_STAMP = stamp
    return _SERVICES


def get_services() -> Services:
    return load_services()


def _clean(value: Any) -> Any:
    """Recursively convert NaN/inf/NumPy types into strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and (pd.isna(value) or value in (float("inf"), float("-inf"))):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty:
        return []
    return [_clean(row) for row in frame.to_dict(orient="records")]


def get_cache(request: Request) -> Cache:
    """FastAPI dependency returning the app's cache backend."""
    cache = getattr(request.app.state, "cache", None)
    return cache if cache is not None else build_cache("none")


def _path_signature(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return "missing"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def _elo_ratings_payload(raw: pd.DataFrame) -> list[dict]:
    """JSON-safe Elo tables (cached; refits are the expensive part)."""
    return _clean(latest_elo_ratings(compute_elo(raw)[1]).to_dict(orient="records"))


def _goal_model_payload(raw: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """JSON-safe Poisson/Dixon-Coles strength tables."""
    model = fit_goal_model(raw, as_of)
    return _clean({
        "attack": model.attack,
        "defence": model.defence,
        "matches": model.matches,
    })


def _data_fingerprint(services: Services) -> str:
    """Fingerprint of everything that can make a cached prediction stale.

    The production model version invalidates every entry on promotion or
    rollback; raw/ledger/metadata file signatures invalidate on a data refresh
    or retrain. Fixture identity and kickoff are added per request.
    """
    version = str(services.metadata.get("model_version") or "unversioned")
    try:
        production = current_production()
    except RegistryError:
        production = None
    production_version = str((production or {}).get("version") or "")
    payload = "|".join([
        version,
        production_version,
        _path_signature(RAW_PATH),
        _path_signature(LEDGER_PATH),
        _path_signature(MODEL_METADATA_PATH),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _prediction_payload(prediction: dict) -> Prediction:
    goal_model = prediction.get("goal_model")
    uncertainty = prediction.get("uncertainty")
    return Prediction(
        fixture_id=prediction["fixture_id"],
        home_team=prediction["home_team"],
        away_team=prediction["away_team"],
        match_date=prediction.get("match_date") or "",
        probabilities={k: float(v) for k, v in prediction["probs"].items()},
        predicted_outcome=prediction["predicted_outcome"],
        confidence=float(prediction["confidence"]),
        expected_home_goals=prediction.get("expected_home_goals"),
        expected_away_goals=prediction.get("expected_away_goals"),
        most_likely_scoreline=prediction.get("most_likely_scoreline"),
        scoreline_probability=prediction.get("scoreline_probability"),
        top_scorelines=[
            Scoreline(**item) for item in prediction.get("top_scorelines", [])
        ],
        goal_model=GoalModelInfo(**goal_model) if goal_model else None,
        goal_model_error=prediction.get("goal_model_error"),
        uncertainty=Uncertainty(**uncertainty) if uncertainty else None,
        model_name=prediction.get("model_name", "unknown"),
        model_version=prediction.get("model_version", "unversioned"),
        predicted_at=prediction["predicted_at"],
        market_note=prediction.get("market_note", ""),
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def create_app(cache: Cache | None = None) -> FastAPI:
    app = FastAPI(
        title="La Liga Predictor API",
        version=API_VERSION,
        description=(
            "Probabilistic La Liga match analytics. Probabilities include market "
            "information, are not betting advice, and are only as good as the "
            "documented validation/test windows."
        ),
    )
    # One backend per application. Tests pass their own cache to isolate state.
    app.state.cache = cache if cache is not None else build_cache()

    @app.exception_handler(ArtifactError)
    async def _artifact_error(_request, exc: ArtifactError):  # noqa: ANN001
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        from src.features.preprocessing import DEFAULT_PREPROCESSOR_PATH
        from src.utils.config import BEST_MODEL_PATH

        production = None
        try:
            production = current_production()
        except RegistryError:
            production = None
        version = production.get("version") if production else None
        if version is None and MODEL_METADATA_PATH.exists():
            try:
                version = json.loads(
                    MODEL_METADATA_PATH.read_text(encoding="utf-8")
                ).get("model_version")
            except Exception:  # noqa: BLE001 - health must not raise
                version = None
        artifacts = {
            "raw": RAW_PATH.exists(),
            "model": BEST_MODEL_PATH.exists(),
            "preprocessor": DEFAULT_PREPROCESSOR_PATH.exists(),
            "metadata": MODEL_METADATA_PATH.exists(),
        }
        return HealthResponse(
            status="ok" if all(artifacts.values()) else "degraded",
            api_version=API_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_version=version,
            artifacts=artifacts,
        )

    @app.get("/model", response_model=ModelResponse, tags=["model"])
    def model_info(services: Services = Depends(get_services)) -> ModelResponse:
        production = None
        try:
            production = current_production()
        except RegistryError as exc:
            logger.warning("Registry unavailable: %s", exc)
        metadata = dict(services.metadata)
        return ModelResponse(
            production=production,
            metadata=_clean({
                "model_name": metadata.get("model_name"),
                "model_version": metadata.get("model_version"),
                "trained_at": metadata.get("trained_at"),
                "split": metadata.get("split"),
                "selection": metadata.get("selection"),
                "validation_metrics": metadata.get("validation_metrics"),
                "test_metrics": metadata.get("test_metrics"),
            }),
            feature_count=len(services.preprocessor.feature_names),
            feature_names=list(services.preprocessor.feature_names),
        )

    @app.get("/fixtures", response_model=list[Prediction], tags=["predictions"])
    def fixtures(
        limit: int = Query(default=50, ge=1, le=500),
        services: Services = Depends(get_services),
        cache: Cache = Depends(get_cache),
    ) -> list[Prediction]:
        fixtures, _source = load_upcoming_fixtures()
        if not fixtures:
            return []
        selected = fixtures[:limit]
        signature = cache_key(*[
            (f.get("home_team"), f.get("away_team"),
             f.get("match_date", ""), f.get("kickoff_utc", ""), f.get("odds"))
            for f in selected
        ])
        key = cache_key("fixtures", _data_fingerprint(services), limit, signature)

        def compute() -> list[dict]:
            predictions, errors = predict_fixtures(
                selected,
                model=services.model,
                preprocessor=services.preprocessor,
                metadata=services.metadata,
                raw=services.raw,
            )
            if not predictions and errors:
                raise HTTPException(
                    status_code=422,
                    detail={"message": "No fixture could be resolved.", "errors": errors},
                )
            return [_prediction_payload(p).model_dump() for p in predictions]

        return [Prediction(**item) for item in cached_call(cache, key, compute)]

    @app.post("/predict", response_model=Prediction, tags=["predictions"])
    def predict(
        payload: FixtureRequest,
        services: Services = Depends(get_services),
        cache: Cache = Depends(get_cache),
    ) -> Prediction:
        fixture = payload.as_fixture()
        key = cache_key(
            "predict",
            _data_fingerprint(services),
            fixture["home_team"],
            fixture["away_team"],
            fixture["match_date"],
            fixture["odds"],
        )

        def compute() -> dict:
            predictions, errors = predict_fixtures(
                [fixture],
                model=services.model,
                preprocessor=services.preprocessor,
                metadata=services.metadata,
                raw=services.raw,
            )
            if not predictions:
                detail = errors[0]["error"] if errors else "Prediction failed."
                raise HTTPException(status_code=422, detail=detail)
            return _prediction_payload(predictions[0]).model_dump()

        return Prediction(**cached_call(cache, key, compute))

    @app.get("/teams", response_model=list[str], tags=["teams"])
    def teams(services: Services = Depends(get_services)) -> list[str]:
        raw = services.raw
        return sorted(set(raw["home_team"].astype(str)) | set(raw["away_team"].astype(str)))

    @app.get("/teams/{team}/stats", response_model=TeamStatsResponse, tags=["teams"])
    def team_stats(
        team: str,
        services: Services = Depends(get_services),
        cache: Cache = Depends(get_cache),
    ) -> TeamStatsResponse:
        available = sorted(
            set(services.raw["home_team"].astype(str))
            | set(services.raw["away_team"].astype(str))
        )
        try:
            canonical = canonical_team(team)
            if canonical not in available:
                matches = [t for t in available if canonical and canonical in t]
                if len(matches) != 1:
                    raise TeamNotFoundError(f"Unknown team: {team}")
                canonical = matches[0]
        except TeamNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        from src.predictions.predictor import team_form  # local: avoids cycles

        history = build_team_history(services.raw)
        rows = history[history["team"] == canonical].sort_values("date")
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"No matches for {canonical}.")

        points = (
            (rows["goals_scored"] > rows["goals_conceded"]).astype(int) * 3
            + (rows["goals_scored"] == rows["goals_conceded"]).astype(int)
        )
        home_mask = rows["side"] == "home"
        recent = rows.tail(5)

        fingerprint = _data_fingerprint(services)
        today = pd.Timestamp.today().normalize()
        elo_records = cached_call(
            cache, cache_key("elo-ratings", fingerprint),
            lambda: _elo_ratings_payload(services.raw),
        )
        elo_row = next((r for r in elo_records if r.get("team") == canonical), None)
        goal_payload = cached_call(
            cache, cache_key("goal-model", fingerprint, f"{today:%Y-%m-%d}"),
            lambda: _goal_model_payload(services.raw, today),
        )
        form = team_form(history, canonical) if not rows.empty else {}

        return TeamStatsResponse(
            team=canonical,
            matches=int(len(rows)),
            wins=int((rows["goals_scored"] > rows["goals_conceded"]).sum()),
            draws=int((rows["goals_scored"] == rows["goals_conceded"]).sum()),
            losses=int((rows["goals_scored"] < rows["goals_conceded"]).sum()),
            goals_for=int(rows["goals_scored"].sum()),
            goals_against=int(rows["goals_conceded"].sum()),
            goals_for_per_match=float(rows["goals_scored"].mean()),
            goals_against_per_match=float(rows["goals_conceded"].mean()),
            points_per_game=float(points.mean()),
            home_points_per_game=float(points[home_mask].mean()) if home_mask.any() else None,
            away_points_per_game=float(points[~home_mask].mean()) if (~home_mask).any() else None,
            form=" ".join(
                "W" if gf > ga else ("D" if gf == ga else "L")
                for gf, ga in zip(recent["goals_scored"], recent["goals_conceded"])
            ),
            elo_rating=float(elo_row["rating"]) if elo_row else None,
            elo_rating_deviation=(
                float(elo_row["rating_deviation"]) if elo_row else None
            ),
            attack=float(goal_payload["attack"].get(canonical, 1.0)),
            defence=float(goal_payload["defence"].get(canonical, 1.0)),
            poisson_matches=int(goal_payload["matches"].get(canonical, 0)),
        )

    @app.get("/performance", response_model=PerformanceResponse, tags=["performance"])
    def performance(
        services: Services = Depends(get_services),
    ) -> PerformanceResponse:
        history = services.history
        return PerformanceResponse(
            summary=_clean(hist.performance_summary(history)),
            by_season=_records(hist.performance_by(history, "season")),
            by_model=_records(hist.performance_by(history, "model")),
        )

    @app.get("/registry", response_model=RegistryResponse, tags=["model"])
    def registry() -> RegistryResponse:
        try:
            state = status()
            versions = _records(list_models())
        except RegistryError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return RegistryResponse(
            production_version=state.get("production_version"),
            production_since=state.get("production_since"),
            n_versions=state.get("n_versions", 0),
            versions=versions,
        )

    @app.get("/runs", tags=["operations"])
    def runs(limit: int = Query(default=10, ge=1, le=100)) -> list[dict]:
        return _clean(load_runs(limit=limit))

    # ------------------------------------------------------------------
    # Phase 8 lifecycle and operations endpoints. Additive only: the existing
    # /fixtures, /predict and /performance contracts are unchanged.
    # ------------------------------------------------------------------

    @app.get("/fixtures/upcoming", tags=["operations"])
    def upcoming_fixtures(
        limit: int = Query(default=200, ge=1, le=500),
        services: Services = Depends(get_services),
    ) -> list[dict]:
        from src.predictions.lifecycle import upcoming_fixture_status

        return _clean(upcoming_fixture_status(services.history, limit=limit))

    @app.get("/fixtures/schedule-changes", tags=["operations"])
    def fixture_schedule_changes(
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict]:
        from src.predictions.lifecycle import load_schedule_revisions

        return _clean(list(reversed(load_schedule_revisions(limit=limit))))

    @app.get("/fixtures/{fixture_id}/predictions", tags=["operations"])
    def fixture_predictions(
        fixture_id: str,
        limit: int = Query(default=50, ge=1, le=500),
        services: Services = Depends(get_services),
    ) -> list[dict]:
        from src.predictions.lifecycle import prediction_history

        return _clean(
            prediction_history(services.history, fixture_id=fixture_id, limit=limit)
        )

    @app.get("/predictions/history", tags=["operations"])
    def prediction_history_endpoint(
        limit: int = Query(default=200, ge=1, le=1000),
        services: Services = Depends(get_services),
    ) -> list[dict]:
        from src.predictions.lifecycle import prediction_history

        return _clean(prediction_history(services.history, limit=limit))

    @app.get("/results/settled", tags=["operations"])
    def settled_results_endpoint(
        limit: int = Query(default=200, ge=1, le=1000),
        services: Services = Depends(get_services),
    ) -> list[dict]:
        from src.predictions.lifecycle import settled_results

        return _clean(settled_results(services.history, limit=limit))

    @app.get("/performance/live", tags=["operations"])
    def live_performance(
        services: Services = Depends(get_services),
    ) -> dict:
        from src.monitoring.live_performance import build_live_performance

        return _clean(build_live_performance(services.history))

    return app


app = create_app()
