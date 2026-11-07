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

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

def create_app() -> FastAPI:
    app = FastAPI(
        title="La Liga Predictor API",
        version=API_VERSION,
        description=(
            "Probabilistic La Liga match analytics. Probabilities include market "
            "information, are not betting advice, and are only as good as the "
            "documented validation/test windows."
        ),
    )

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
    ) -> list[Prediction]:
        fixtures, _source = load_upcoming_fixtures()
        if not fixtures:
            return []
        predictions, errors = predict_fixtures(
            fixtures[:limit],
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
        return [_prediction_payload(p) for p in predictions]

    @app.post("/predict", response_model=Prediction, tags=["predictions"])
    def predict(
        request: FixtureRequest,
        services: Services = Depends(get_services),
    ) -> Prediction:
        predictions, errors = predict_fixtures(
            [request.as_fixture()],
            model=services.model,
            preprocessor=services.preprocessor,
            metadata=services.metadata,
            raw=services.raw,
        )
        if not predictions:
            detail = errors[0]["error"] if errors else "Prediction failed."
            raise HTTPException(status_code=422, detail=detail)
        return _prediction_payload(predictions[0])

    @app.get("/teams", response_model=list[str], tags=["teams"])
    def teams(services: Services = Depends(get_services)) -> list[str]:
        raw = services.raw
        return sorted(set(raw["home_team"].astype(str)) | set(raw["away_team"].astype(str)))

    @app.get("/teams/{team}/stats", response_model=TeamStatsResponse, tags=["teams"])
    def team_stats(
        team: str,
        services: Services = Depends(get_services),
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

        elo = latest_elo_ratings(compute_elo(services.raw)[1])
        elo_row = elo[elo["team"] == canonical]
        goal_model = fit_goal_model(services.raw, pd.Timestamp.today().normalize())
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
            elo_rating=float(elo_row["rating"].iloc[0]) if not elo_row.empty else None,
            elo_rating_deviation=(
                float(elo_row["rating_deviation"].iloc[0]) if not elo_row.empty else None
            ),
            attack=float(goal_model.attack.get(canonical, 1.0)),
            defence=float(goal_model.defence.get(canonical, 1.0)),
            poisson_matches=int(goal_model.matches.get(canonical, 0)),
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

    return app


app = create_app()
