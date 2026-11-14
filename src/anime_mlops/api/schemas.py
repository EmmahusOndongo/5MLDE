"""Schémas d'entrée et de sortie de l'API."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PREDICTION_FEATURES = [
    "num_episodes",
    "average_episode_duration",
    "start_year",
    "prequel_score",
    "media_type",
    "rating",
    "source",
    "start_season",
    "start_weekday",
    "is_sequel",
    "genres",
    "studios",
    "tags",
]

PositiveEpisodes = Annotated[int, Field(gt=0, le=10_000)]
PositiveDuration = Annotated[float, Field(gt=0, le=604_800)]
StartYear = Annotated[int, Field(ge=1900, le=2100)]
Score = Annotated[float, Field(ge=0, le=10)]
Category = Annotated[str, Field(min_length=1, max_length=100)]
Label = Annotated[str, Field(min_length=1, max_length=200)]
Labels = Annotated[list[Label], Field(max_length=200)]


class PredictionInput(BaseModel):
    """Les champs bruts attendus par le pipeline scikit-learn."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    num_episodes: PositiveEpisodes | None
    average_episode_duration: PositiveDuration | None
    start_year: StartYear | None
    prequel_score: Score | None
    media_type: Category | None
    rating: Category | None
    source: Category | None
    start_season: Literal["winter", "spring", "summer", "fall"] | None
    start_weekday: (
        Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"] | None
    )
    is_sequel: bool
    genres: Labels
    studios: Labels
    tags: Labels

    def to_record(self) -> dict[str, object]:
        """Sérialise les features dans l'ordre du contrat d'entraînement."""
        values = self.model_dump()
        return {feature: values[feature] for feature in PREDICTION_FEATURES}


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: Annotated[list[PredictionInput], Field(min_length=1, max_length=1000)]


class ModelMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    alias: str
    version: str
    run_id: str | None
    model_uri: str
    loaded_at: datetime


class PredictionResponse(BaseModel):
    prediction: float
    model: ModelMetadata


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    count: int
    model: ModelMetadata


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: ModelMetadata | None
    reason: str | None = None


class ModelResponse(BaseModel):
    model: ModelMetadata


class ReloadResponse(BaseModel):
    status: Literal["reloaded"] = "reloaded"
    model: ModelMetadata
