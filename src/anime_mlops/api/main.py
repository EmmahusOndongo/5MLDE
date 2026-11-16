"""Application FastAPI de prédiction en ligne et par lot."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import UUID, uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from anime_mlops.api.cache import PredictionCache
from anime_mlops.api.metrics import PROMETHEUS_CONTENT_TYPE, ApiMetrics
from anime_mlops.api.model_service import (
    MlflowRegistryLoader,
    ModelLoader,
    ModelLoadError,
    ModelRuntime,
    ModelUnavailableError,
    PredictionError,
)
from anime_mlops.api.persistence import (
    DatabaseStore,
    DatabaseUnavailableError,
    FeedbackConflictError,
    Persistence,
    PredictionNotFoundError,
    PredictionTrace,
)
from anime_mlops.api.schemas import (
    NUMERIC_PREDICTION_FEATURES,
    PREDICTION_FEATURES,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ComponentHealth,
    FeedbackInput,
    FeedbackResponse,
    HealthResponse,
    ModelMetadata,
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
    persistence: Persistence | None = None,
    metrics: ApiMetrics | None = None,
    cache: PredictionCache | None = None,
    load_model_on_startup: bool = True,
) -> FastAPI:
    """Construit l'application et permet d'injecter un chargeur isolé dans les tests."""
    resolved_settings = settings or get_settings()
    resolved_loader = loader or MlflowRegistryLoader(resolved_settings.mlflow.tracking_uri)
    resolved_persistence = persistence or DatabaseStore.from_url(
        resolved_settings.database.url,
        connect_timeout_seconds=resolved_settings.database.connect_timeout_seconds,
        retry_cooldown_seconds=resolved_settings.database.retry_cooldown_seconds,
    )
    resolved_metrics = metrics or ApiMetrics.create()
    resolved_cache = cache or PredictionCache(
        enabled=resolved_settings.api.cache_enabled,
        max_entries=resolved_settings.api.cache_max_entries,
        ttl_seconds=resolved_settings.api.cache_ttl_seconds,
    )
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
                resolved_metrics.set_model(runtime.reload())
            except ModelLoadError:
                # Le processus reste vivant pour exposer /health et permettre /admin/reload.
                LOGGER.warning("L'API démarre sans modèle disponible.", exc_info=True)
        if not resolved_persistence.initialize():
            LOGGER.warning("L'API démarre sans persistance disponible.")
        try:
            yield
        finally:
            resolved_persistence.close()

    application = FastAPI(
        title="Anime Score API",
        version="0.1.0",
        description="Prédiction du score MyAnimeList d'un anime avant sa diffusion.",
        lifespan=lifespan,
    )
    application.state.model_runtime = runtime
    application.state.persistence = resolved_persistence
    application.state.metrics = resolved_metrics
    application.state.cache = resolved_cache

    @application.middleware("http")
    async def observe_http_requests(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        resolved_metrics.in_flight.inc()
        started_at = perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            resolved_metrics.in_flight.dec()
            resolved_metrics.observe_http(
                method=request.method,
                route=_route_template(request),
                status_code=status_code,
                duration=perf_counter() - started_at,
            )

    def build_trace(
        *,
        prediction_id: UUID,
        endpoint: str,
        input_data: dict[str, object],
        duration_ms: float,
        prediction: float | None = None,
        metadata: ModelMetadata | None = None,
        error: ModelUnavailableError | PredictionError | None = None,
        batch_id: UUID | None = None,
    ) -> PredictionTrace:
        if metadata is None and runtime.is_ready():
            metadata = runtime.metadata()
        return PredictionTrace(
            id=prediction_id,
            endpoint=endpoint,
            input_data=input_data,
            duration_ms=duration_ms,
            prediction=prediction,
            model_name=metadata.name if metadata else runtime.model_name,
            model_alias=metadata.alias if metadata else runtime.alias,
            model_version=metadata.version if metadata else None,
            model_run_id=metadata.run_id if metadata else None,
            error_code=(
                "model_unavailable"
                if isinstance(error, ModelUnavailableError)
                else "prediction_failed"
                if error
                else None
            ),
            error_message=str(error)[:1000] if error else None,
            batch_id=batch_id,
        )

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

    @application.exception_handler(DatabaseUnavailableError)
    async def handle_database_unavailable(
        _request: Request, exc: DatabaseUnavailableError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="database_unavailable",
            message=str(exc),
        )

    @application.exception_handler(PredictionNotFoundError)
    async def handle_prediction_not_found(
        _request: Request, exc: PredictionNotFoundError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_404_NOT_FOUND,
            code="prediction_not_found",
            message=str(exc),
        )

    @application.exception_handler(FeedbackConflictError)
    async def handle_feedback_conflict(
        _request: Request, exc: FeedbackConflictError
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_409_CONFLICT,
            code="feedback_conflict",
            message=str(exc),
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
        tags=["exploitation"],
    )
    def health() -> HealthResponse | JSONResponse:
        database_status = resolved_persistence.status()
        database_health = ComponentHealth(
            status="ok" if database_status.available else "degraded",
            reason=database_status.reason,
        )
        if not runtime.is_ready():
            response = HealthResponse(
                status="degraded",
                model=None,
                database=database_health,
                reason=runtime.last_error() or "Aucun modèle n'est chargé.",
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump(mode="json"),
            )

        model_reason = runtime.last_error()
        reasons = [reason for reason in (model_reason, database_status.reason) if reason]
        return HealthResponse(
            status="degraded" if reasons else "ok",
            model=runtime.metadata(),
            database=database_health,
            reason=" ".join(reasons) or None,
        )

    @application.get("/model", response_model=ModelResponse, tags=["exploitation"])
    def model() -> ModelResponse:
        return ModelResponse(model=runtime.metadata())

    @application.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(
            content=resolved_metrics.render(),
            headers={"Content-Type": PROMETHEUS_CONTENT_TYPE},
        )

    @application.post("/predict", response_model=PredictionResponse, tags=["prédiction"])
    def predict_one(payload: PredictionInput) -> PredictionResponse:
        prediction_id = uuid4()
        record = payload.to_record()
        model_input = _model_frame([record])
        cache_key = resolved_cache.key([record]) if resolved_cache.enabled else ""
        started_at = perf_counter()
        try:
            predictions, metadata, cache_hit = runtime.predict_cached(
                model_input, cache=resolved_cache, cache_key=cache_key
            )
        except (ModelUnavailableError, PredictionError) as exc:
            duration_ms = (perf_counter() - started_at) * 1000
            resolved_persistence.record_predictions(
                [
                    build_trace(
                        prediction_id=prediction_id,
                        endpoint="/predict",
                        input_data=record,
                        duration_ms=duration_ms,
                        error=exc,
                    )
                ]
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        resolved_metrics.observe_predictions(
            endpoint="/predict", count=len(predictions), metadata=metadata
        )
        persisted = resolved_persistence.record_predictions(
            [
                build_trace(
                    prediction_id=prediction_id,
                    endpoint="/predict",
                    input_data=record,
                    duration_ms=duration_ms,
                    prediction=predictions[0],
                    metadata=metadata,
                )
            ]
        )
        return PredictionResponse(
            prediction_id=prediction_id,
            prediction=predictions[0],
            persisted=persisted,
            cached=cache_hit,
            model=metadata,
        )

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
        model_input = _model_frame(records)
        cache_key = resolved_cache.key(records) if resolved_cache.enabled else ""
        batch_id = uuid4()
        prediction_ids = [uuid4() for _instance in payload.instances]
        started_at = perf_counter()
        try:
            predictions, metadata, cache_hit = runtime.predict_cached(
                model_input, cache=resolved_cache, cache_key=cache_key
            )
        except (ModelUnavailableError, PredictionError) as exc:
            duration_ms = (perf_counter() - started_at) * 1000
            resolved_persistence.record_predictions(
                [
                    build_trace(
                        prediction_id=prediction_id,
                        endpoint="/predict/batch",
                        input_data=record,
                        duration_ms=duration_ms,
                        error=exc,
                        batch_id=batch_id,
                    )
                    for prediction_id, record in zip(prediction_ids, records, strict=True)
                ]
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        resolved_metrics.observe_predictions(
            endpoint="/predict/batch", count=len(predictions), metadata=metadata
        )
        persisted = resolved_persistence.record_predictions(
            [
                build_trace(
                    prediction_id=prediction_id,
                    endpoint="/predict/batch",
                    input_data=record,
                    duration_ms=duration_ms,
                    prediction=prediction,
                    metadata=metadata,
                    batch_id=batch_id,
                )
                for prediction_id, record, prediction in zip(
                    prediction_ids, records, predictions, strict=True
                )
            ]
        )
        return BatchPredictionResponse(
            prediction_ids=prediction_ids,
            predictions=predictions,
            count=len(predictions),
            persisted=persisted,
            cached=cache_hit,
            model=metadata,
        )

    @application.post(
        "/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["feedback"],
    )
    def add_feedback(payload: FeedbackInput) -> FeedbackResponse:
        feedback = resolved_persistence.add_feedback(
            prediction_id=payload.prediction_id,
            actual_score=payload.actual_score,
            source="manual",
            comment=payload.comment,
        )
        return FeedbackResponse(
            feedback_id=feedback.id,
            prediction_id=feedback.prediction_id,
            actual_score=feedback.actual_score,
            source=feedback.source,
            created_at=feedback.created_at,
        )

    @application.post("/admin/reload", response_model=ReloadResponse, tags=["administration"])
    def reload_model() -> ReloadResponse:
        metadata = runtime.reload()
        resolved_metrics.set_model(metadata)
        resolved_cache.clear()
        return ReloadResponse(model=metadata)

    return application


def _model_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Reproduit les dtypes numériques float64 produits par le pipeline de features."""
    frame = pd.DataFrame.from_records(records, columns=PREDICTION_FEATURES)
    # MLflow applique strictement la signature enregistrée et refuse notamment int64 -> float64.
    # astype convertit aussi les valeurs absentes en NaN, que les imputers du modèle gèrent.
    frame[NUMERIC_PREDICTION_FEATURES] = frame[NUMERIC_PREDICTION_FEATURES].astype("float64")
    return frame


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def _error_response(status_code: int, *, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": {"code": code, "message": message}},
    )


app = create_app()
