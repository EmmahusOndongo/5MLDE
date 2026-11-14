from datetime import UTC, datetime

import pandas as pd
from fastapi.testclient import TestClient

from anime_mlops.api.main import create_app
from anime_mlops.api.model_service import LoadedModel
from anime_mlops.api.schemas import PREDICTION_FEATURES, ModelMetadata
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
    application = create_app(settings=Settings(), loader=loader)

    with TestClient(application) as client:
        health = client.get("/health")
        model = client.get("/model")
        prediction = client.post("/predict", json=prediction_payload())
        batch = client.post(
            "/predict/batch", json={"instances": [prediction_payload(), prediction_payload()]}
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model"]["version"] == "1"
    assert model.status_code == 200
    assert model.json()["model"]["version"] == "1"
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] == 7.25
    assert prediction.json()["model"]["version"] == "1"
    assert batch.status_code == 200
    assert batch.json()["predictions"] == [7.25, 7.25]
    assert batch.json()["count"] == 2
    assert batch.json()["model"]["version"] == "1"
    assert loader.predictor.frames[0].columns.tolist() == PREDICTION_FEATURES


def test_reload_atomically_changes_the_served_version():
    loader = SequenceLoader()
    application = create_app(settings=Settings(), loader=loader)

    with TestClient(application) as client:
        response = client.post("/admin/reload")
        prediction = client.post("/predict", json=prediction_payload())

    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"
    assert response.json()["model"]["version"] == "2"
    assert prediction.json()["model"]["version"] == "2"


def test_invalid_inputs_are_rejected_before_the_model():
    loader = SequenceLoader()
    application = create_app(settings=Settings(), loader=loader)
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
    application = create_app(settings=settings, loader=loader)

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
    application = create_app(settings=Settings(), loader=loader)

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
    application = create_app(settings=Settings(), loader=SequenceLoader(predictor))

    with TestClient(application) as client:
        response = client.post("/predict", json=prediction_payload())

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "prediction_failed"
