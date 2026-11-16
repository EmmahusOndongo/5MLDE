from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

from anime_mlops.api.main import create_app
from anime_mlops.api.model_service import LoadedModel
from anime_mlops.api.persistence import (
    DatabaseStatus,
    DatabaseStore,
    DatabaseUnavailableError,
    FeedbackConflictError,
    FeedbackRecord,
    FeedbackRow,
    PredictionNotFoundError,
    PredictionRow,
)
from anime_mlops.api.schemas import (
    NUMERIC_PREDICTION_FEATURES,
    PREDICTION_FEATURES,
    ModelMetadata,
)
from anime_mlops.config import Settings


class StubPredictor:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.fail = False

    def predict(self, model_input: pd.DataFrame):
        self.frames.append(model_input.copy())
        if self.fail:
            raise ValueError("échec simulé")
        return [7.25] * len(model_input)


class SequenceLoader:
    def __init__(self, predictor: StubPredictor | None = None) -> None:
        self.predictor = predictor or StubPredictor()
        self.load_count = 0
        self.fail = False

    def load(self, model_name: str, alias: str) -> LoadedModel:
        if self.fail:
            raise ConnectionError("registre indisponible")
        self.load_count += 1
        version = str(self.load_count)
        return LoadedModel(
            predictor=self.predictor,
            metadata=ModelMetadata(
                name=model_name,
                alias=alias,
                version=version,
                run_id=f"run-{version}",
                model_uri=f"models:/{model_name}/{version}",
                loaded_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )


class MemoryPersistence:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.traces = []
        self.feedbacks: dict[tuple, FeedbackRecord] = {}
        self.closed = False

    def initialize(self) -> bool:
        return self.available

    def status(self) -> DatabaseStatus:
        return DatabaseStatus(
            available=self.available,
            reason=None if self.available else "PostgreSQL indisponible.",
        )

    def record_predictions(self, traces) -> bool:
        if not self.available:
            return False
        self.traces.extend(traces)
        return True

    def add_feedback(self, *, prediction_id, actual_score, source, comment=None):
        if not self.available:
            raise DatabaseUnavailableError("PostgreSQL indisponible.")
        if prediction_id not in {trace.id for trace in self.traces}:
            raise PredictionNotFoundError(f"Prédiction inconnue : {prediction_id}")
        key = (prediction_id, source)
        if key in self.feedbacks:
            raise FeedbackConflictError("Feedback déjà enregistré.")
        feedback = FeedbackRecord(
            id=uuid4(),
            prediction_id=prediction_id,
            actual_score=actual_score,
            source=source,
            comment=comment,
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        self.feedbacks[key] = feedback
        return feedback

    def close(self) -> None:
        self.closed = True


def build_app(loader, *, settings=None, persistence=None, cache=None):
    return create_app(
        settings=settings or Settings(),
        loader=loader,
        persistence=persistence or MemoryPersistence(),
        cache=cache,
    )


def prediction_payload() -> dict:
    return {
        "num_episodes": 12,
        "average_episode_duration": 1_440,
        "start_year": 2027,
        "prequel_score": None,
        "media_type": "tv",
        "rating": "pg_13",
        "source": "MANGA",
        "start_season": "spring",
        "start_weekday": "monday",
        "is_sequel": False,
        "genres": ["Action", "Drama"],
        "studios": ["Bones"],
        "tags": ["Shounen"],
    }


def test_nominal_endpoints_expose_the_served_version():
    loader = SequenceLoader()
    persistence = MemoryPersistence()
    application = build_app(loader, persistence=persistence)

    with TestClient(application) as client:
        health = client.get("/health")
        model = client.get("/model")
        prediction = client.post("/predict", json=prediction_payload())
        batch = client.post(
            "/predict/batch", json={"instances": [prediction_payload(), prediction_payload()]}
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["database"]["status"] == "ok"
    assert health.json()["model"]["version"] == "1"
    assert model.status_code == 200
    assert model.json()["model"]["version"] == "1"
    assert prediction.status_code == 200
    assert prediction.json()["persisted"] is True
    assert prediction.json()["cached"] is False
    assert prediction.json()["prediction"] == 7.25
    assert prediction.json()["model"]["version"] == "1"
    assert batch.status_code == 200
    assert batch.json()["predictions"] == [7.25, 7.25]
    assert batch.json()["count"] == 2
    assert batch.json()["persisted"] is True
    assert batch.json()["cached"] is False
    assert batch.json()["model"]["version"] == "1"
    assert loader.predictor.frames[0].columns.tolist() == PREDICTION_FEATURES
    assert all(
        frame[NUMERIC_PREDICTION_FEATURES].dtypes.eq("float64").all()
        for frame in loader.predictor.frames
    )
    assert pd.isna(loader.predictor.frames[0].loc[0, "prequel_score"])
    assert persistence.traces[0].input_data["prequel_score"] is None
    assert len(persistence.traces) == 3
    assert {trace.model_version for trace in persistence.traces} == {"1"}


def test_reload_atomically_changes_the_served_version():
    loader = SequenceLoader()
    application = build_app(loader)

    with TestClient(application) as client:
        response = client.post("/admin/reload")
        prediction = client.post("/predict", json=prediction_payload())
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"
    assert response.json()["model"]["version"] == "2"
    assert prediction.json()["model"]["version"] == "2"
    assert (
        'anime_api_model_info{model_alias="champion",model_name="anime-score-regressor",'
        'model_version="1"} 0.0'
    ) in metrics.text
    assert (
        'anime_api_model_info{model_alias="champion",model_name="anime-score-regressor",'
        'model_version="2"} 1.0'
    ) in metrics.text


def test_metrics_expose_throughput_latency_errors_and_served_version():
    application = build_app(SequenceLoader())
    invalid_payload = prediction_payload()
    invalid_payload["start_year"] = 1800

    with TestClient(application) as client:
        prediction = client.post("/predict", json=prediction_payload())
        invalid = client.post("/predict", json=invalid_payload)
        metrics = client.get("/metrics")

    assert prediction.status_code == 200
    assert invalid.status_code == 422
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert (
        'anime_api_http_requests_total{method="POST",route="/predict",status_code="200"} 1.0'
    ) in metrics.text
    assert (
        'anime_api_http_errors_total{method="POST",route="/predict",status_code="422"} 1.0'
    ) in metrics.text
    assert (
        'anime_api_http_request_duration_seconds_count{method="POST",route="/predict"} 2.0'
    ) in metrics.text
    assert (
        'anime_api_predictions_total{endpoint="/predict",model_name="anime-score-regressor",'
        'model_version="1"} 1.0'
    ) in metrics.text
    assert (
        'anime_api_model_info{model_alias="champion",model_name="anime-score-regressor",'
        'model_version="1"} 1.0'
    ) in metrics.text
    assert 'route="/metrics"' not in metrics.text


def test_cache_avoids_recomputation_but_keeps_each_trace():
    settings = Settings()
    settings.api.cache_enabled = True
    settings.api.cache_max_entries = 10
    settings.api.cache_ttl_seconds = 60
    loader = SequenceLoader()
    persistence = MemoryPersistence()
    application = build_app(loader, settings=settings, persistence=persistence)

    with TestClient(application) as client:
        first = client.post("/predict", json=prediction_payload())
        second = client.post("/predict", json=prediction_payload())
        reload_response = client.post("/admin/reload")
        after_reload = client.post("/predict", json=prediction_payload())

    assert first.status_code == second.status_code == after_reload.status_code == 200
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True
    assert after_reload.json()["cached"] is False
    assert first.json()["prediction_id"] != second.json()["prediction_id"]
    assert reload_response.json()["model"]["version"] == "2"
    assert after_reload.json()["model"]["version"] == "2"
    assert len(loader.predictor.frames) == 2
    assert len(persistence.traces) == 3


def test_invalid_inputs_are_rejected_before_the_model():
    loader = SequenceLoader()
    application = build_app(loader)
    missing_feature = prediction_payload()
    missing_feature.pop("genres")
    invalid_score = prediction_payload()
    invalid_score["prequel_score"] = 11
    extra_feature = prediction_payload()
    extra_feature["popularity"] = 1

    with TestClient(application) as client:
        responses = [
            client.post("/predict", json=missing_feature),
            client.post("/predict", json=invalid_score),
            client.post("/predict", json=extra_feature),
            client.post("/predict/batch", json={"instances": []}),
        ]

    assert all(response.status_code == 422 for response in responses)
    assert loader.predictor.frames == []


def test_configured_batch_limit_is_enforced():
    settings = Settings()
    settings.api.max_batch_size = 1
    loader = SequenceLoader()
    application = build_app(loader, settings=settings)

    with TestClient(application) as client:
        response = client.post(
            "/predict/batch", json={"instances": [prediction_payload(), prediction_payload()]}
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "batch_too_large"
    assert loader.predictor.frames == []


def test_api_starts_degraded_when_registry_is_unavailable():
    loader = SequenceLoader()
    loader.fail = True
    application = build_app(loader)

    with TestClient(application) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=prediction_payload())
        reload_response = client.post("/admin/reload")

    assert health.status_code == 503
    assert health.json()["status"] == "degraded"
    assert health.json()["model"] is None
    assert prediction.status_code == 503
    assert prediction.json()["detail"]["code"] == "model_unavailable"
    assert reload_response.status_code == 503
    assert reload_response.json()["detail"]["code"] == "model_reload_failed"


def test_prediction_errors_return_a_controlled_response():
    predictor = StubPredictor()
    predictor.fail = True
    persistence = MemoryPersistence()
    application = build_app(SequenceLoader(predictor), persistence=persistence)

    with TestClient(application) as client:
        response = client.post("/predict", json=prediction_payload())

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "prediction_failed"
    assert persistence.traces[0].error_code == "prediction_failed"


def test_predictions_continue_when_database_is_unavailable():
    persistence = MemoryPersistence(available=False)
    application = build_app(SequenceLoader(), persistence=persistence)

    with TestClient(application) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=prediction_payload())
        feedback = client.post(
            "/feedback",
            json={"prediction_id": str(uuid4()), "actual_score": 8.0},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["database"]["status"] == "degraded"
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] == 7.25
    assert prediction.json()["persisted"] is False
    assert feedback.status_code == 503
    assert feedback.json()["detail"]["code"] == "database_unavailable"


def test_feedback_is_linked_to_a_persisted_prediction():
    persistence = MemoryPersistence()
    application = build_app(SequenceLoader(), persistence=persistence)

    with TestClient(application) as client:
        prediction = client.post("/predict", json=prediction_payload())
        prediction_id = prediction.json()["prediction_id"]
        feedback = client.post(
            "/feedback",
            json={
                "prediction_id": prediction_id,
                "actual_score": 8.5,
                "comment": "Score observé après diffusion",
            },
        )
        duplicate = client.post(
            "/feedback",
            json={"prediction_id": prediction_id, "actual_score": 8.0},
        )
        unknown = client.post(
            "/feedback",
            json={"prediction_id": str(uuid4()), "actual_score": 8.0},
        )

    assert feedback.status_code == 201
    assert feedback.json()["prediction_id"] == prediction_id
    assert feedback.json()["source"] == "manual"
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "feedback_conflict"
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "prediction_not_found"


def test_api_persists_prediction_and_feedback_with_sqlalchemy():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    persistence = DatabaseStore(engine, retry_cooldown_seconds=0)
    application = build_app(SequenceLoader(), persistence=persistence)

    with TestClient(application) as client:
        prediction = client.post("/predict", json=prediction_payload())
        feedback = client.post(
            "/feedback",
            json={
                "prediction_id": prediction.json()["prediction_id"],
                "actual_score": 8.5,
            },
        )
        with engine.connect() as connection:
            prediction_count = connection.scalar(select(func.count()).select_from(PredictionRow))
            feedback_count = connection.scalar(select(func.count()).select_from(FeedbackRow))

    assert prediction.status_code == 200
    assert prediction.json()["persisted"] is True
    assert feedback.status_code == 201
    assert prediction_count == 1
    assert feedback_count == 1
