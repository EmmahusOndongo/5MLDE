"""Application FastAPI de prédiction en ligne et par lot."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from anime_mlops.api.model_service import (
    MlflowRegistryLoader,
    ModelLoader,
    ModelLoadError,
    ModelRuntime,
    ModelUnavailableError,
    PredictionError,
)
from anime_mlops.api.schemas import (
    PREDICTION_FEATURES,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelResponse,
    PredictionInput,
    PredictionResponse,
    ReloadResponse,
)
from anime_mlops.config import Settings, get_settings

LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    settings: Settings | None = None,
    loader: ModelLoader | None = None,
    load_model_on_startup: bool = True,
) -> FastAPI:
    """Construit l'application et permet d'injecter un chargeur isolé dans les tests."""
    resolved_settings = settings or get_settings()
    resolved_loader = loader or MlflowRegistryLoader(resolved_settings.mlflow.tracking_uri)
    runtime = ModelRuntime(
        model_name=resolved_settings.mlflow.registered_model_name,
        alias=resolved_settings.api.model_alias,
        loader=resolved_loader,
        max_batch_size=resolved_settings.api.max_batch_size,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        if load_model_on_startup:
            try:
                runtime.reload()
            except ModelLoadError:
                # Le processus reste vivant pour exposer /health et permettre /admin/reload.
                LOGGER.warning("L'API démarre sans modèle disponible.", exc_info=True)
        yield

    application = FastAPI(
        title="Anime Score API",
        version="0.1.0",
        description="Prédiction du score MyAnimeList d'un anime avant sa diffusion.",
        lifespan=lifespan,
    )
    application.state.model_runtime = runtime

    @application.exception_handler(ModelUnavailableError)
    async def handle_model_unavailable(
        _request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="model_unavailable",
            message=str(exc),
        )

    @application.exception_handler(ModelLoadError)
    async def handle_model_load_error(_request: Request, exc: ModelLoadError) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="model_reload_failed",
            message=str(exc),
        )

    @application.exception_handler(PredictionError)
    async def handle_prediction_error(_request: Request, exc: PredictionError) -> JSONResponse:
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="prediction_failed",
            message=str(exc),
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
        tags=["exploitation"],
    )
    def health() -> HealthResponse | JSONResponse:
        if not runtime.is_ready():
            response = HealthResponse(
                status="degraded",
                model=None,
                reason=runtime.last_error() or "Aucun modèle n'est chargé.",
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(mode="json"),
            )

        reason = runtime.last_error()
        return HealthResponse(
            status="degraded" if reason else "ok",
            model=runtime.metadata(),
            reason=reason,
        )

    @application.get("/model", response_model=ModelResponse, tags=["exploitation"])
    def model() -> ModelResponse:
        return ModelResponse(model=runtime.metadata())

    @application.post("/predict", response_model=PredictionResponse, tags=["prédiction"])
    def predict_one(payload: PredictionInput) -> PredictionResponse:
        model_input = pd.DataFrame.from_records([payload.to_record()], columns=PREDICTION_FEATURES)
        predictions, metadata = runtime.predict(model_input)
        return PredictionResponse(prediction=predictions[0], model=metadata)

    @application.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prédiction"])
    def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
        if len(payload.instances) > runtime.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "batch_too_large",
                    "message": (
                        f"Le lot contient {len(payload.instances)} entrées ; "
                        f"la limite est {runtime.max_batch_size}."
                    ),
                },
            )
        records = [instance.to_record() for instance in payload.instances]
        model_input = pd.DataFrame.from_records(records, columns=PREDICTION_FEATURES)
        predictions, metadata = runtime.predict(model_input)
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            model=metadata,
        )

    @application.post("/admin/reload", response_model=ReloadResponse, tags=["administration"])
    def reload_model() -> ReloadResponse:
        return ReloadResponse(model=runtime.reload())

    return application


def _error_response(status_code: int, *, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": code, "message": message}},
    )


app = create_app()
