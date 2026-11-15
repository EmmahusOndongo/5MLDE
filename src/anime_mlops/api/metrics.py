"""Métriques Prometheus de l'API, compatibles avec plusieurs workers Uvicorn."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import Lock

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from anime_mlops.api.schemas import ModelMetadata

PROMETHEUS_MULTIPROC_DIR_ENV = "PROMETHEUS_MULTIPROC_DIR"


@dataclass
class ApiMetrics:
    """Regroupe les séries et le registre exposé sur ``/metrics``."""

    registry: CollectorRegistry
    requests: Counter
    request_duration: Histogram
    errors: Counter
    predictions: Counter
    in_flight: Gauge
    model_info: Gauge
    _served_model: tuple[str, str, str] | None = None
    _model_lock: Lock = field(default_factory=Lock)

    @classmethod
    def create(cls) -> ApiMetrics:
        multiprocess_enabled = bool(os.environ.get(PROMETHEUS_MULTIPROC_DIR_ENV))
        exposition_registry = CollectorRegistry()
        metric_registry = None if multiprocess_enabled else exposition_registry
        if multiprocess_enabled:
            multiprocess.MultiProcessCollector(exposition_registry)

        return cls(
            registry=exposition_registry,
            requests=Counter(
                "anime_api_http_requests_total",
                "Nombre de requêtes HTTP traitées par l'API.",
                ("method", "route", "status_code"),
                registry=metric_registry,
            ),
            request_duration=Histogram(
                "anime_api_http_request_duration_seconds",
                "Durée des requêtes HTTP en secondes.",
                ("method", "route"),
                registry=metric_registry,
            ),
            errors=Counter(
                "anime_api_http_errors_total",
                "Nombre de réponses HTTP en erreur (statut supérieur ou égal à 400).",
                ("method", "route", "status_code"),
                registry=metric_registry,
            ),
            predictions=Counter(
                "anime_api_predictions_total",
                "Nombre de prédictions produites, par endpoint et version du modèle.",
                ("endpoint", "model_name", "model_version"),
                registry=metric_registry,
            ),
            in_flight=Gauge(
                "anime_api_http_requests_in_flight",
                "Nombre de requêtes HTTP actuellement en cours.",
                multiprocess_mode="livesum",
                registry=metric_registry,
            ),
            model_info=Gauge(
                "anime_api_model_info",
                "Version du modèle actuellement servie (1 pour la version active).",
                ("model_name", "model_alias", "model_version"),
                multiprocess_mode="livemostrecent",
                registry=metric_registry,
            ),
        )

    def observe_http(self, *, method: str, route: str, status_code: int, duration: float) -> None:
        labels = {"method": method, "route": route, "status_code": str(status_code)}
        self.requests.labels(**labels).inc()
        self.request_duration.labels(method=method, route=route).observe(max(duration, 0.0))
        if status_code >= 400:
            self.errors.labels(**labels).inc()

    def observe_predictions(self, *, endpoint: str, count: int, metadata: ModelMetadata) -> None:
        self.predictions.labels(
            endpoint=endpoint,
            model_name=metadata.name,
            model_version=metadata.version,
        ).inc(count)

    def set_model(self, metadata: ModelMetadata) -> None:
        current = (metadata.name, metadata.alias, metadata.version)
        with self._model_lock:
            if self._served_model is not None and self._served_model != current:
                self.model_info.labels(
                    model_name=self._served_model[0],
                    model_alias=self._served_model[1],
                    model_version=self._served_model[2],
                ).set(0)
            self.model_info.labels(
                model_name=current[0], model_alias=current[1], model_version=current[2]
            ).set(1)
            self._served_model = current

    def render(self) -> bytes:
        return generate_latest(self.registry)


PROMETHEUS_CONTENT_TYPE = CONTENT_TYPE_LATEST
