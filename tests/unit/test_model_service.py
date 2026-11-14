from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from anime_mlops.api import model_service
from anime_mlops.api.model_service import (
    LoadedModel,
    MlflowRegistryLoader,
    ModelLoadError,
    ModelRuntime,
    PredictionError,
)
from anime_mlops.api.schemas import ModelMetadata


class ConstantPredictor:
    def predict(self, model_input: pd.DataFrame):
        return [7.0] * len(model_input)


def metadata(version: str = "1") -> ModelMetadata:
    return ModelMetadata(
        name="anime-score-regressor",
        alias="champion",
        version=version,
        run_id=f"run-{version}",
        model_uri=f"models:/anime-score-regressor/{version}",
        loaded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_mlflow_loader_resolves_alias_then_loads_immutable_version(monkeypatch):
    calls = {}
    predictor = ConstantPredictor()

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str):
            calls["alias"] = (name, alias)
            return SimpleNamespace(version="12", run_id="run-12")

    def fake_set_tracking_uri(uri: str) -> None:
        calls["set_tracking_uri"] = uri

    def fake_client(tracking_uri: str) -> FakeClient:
        calls["tracking_uri"] = tracking_uri
        return FakeClient()

    def fake_load_model(uri: str) -> ConstantPredictor:
        calls["model_uri"] = uri
        return predictor

    monkeypatch.setattr(model_service.mlflow, "set_tracking_uri", fake_set_tracking_uri)
    monkeypatch.setattr(model_service, "MlflowClient", fake_client)
    monkeypatch.setattr(model_service, "load_model", fake_load_model)

    loaded = MlflowRegistryLoader("http://mlflow:5000").load("anime-score-regressor", "champion")

    assert calls["tracking_uri"] == "http://mlflow:5000"
    assert calls["set_tracking_uri"] == "http://mlflow:5000"
    assert calls["alias"] == ("anime-score-regressor", "champion")
    assert calls["model_uri"] == "models:/anime-score-regressor/12"
    assert loaded.predictor is predictor
    assert loaded.metadata.version == "12"


def test_failed_reload_preserves_the_previously_loaded_model():
    class Loader:
        def __init__(self) -> None:
            self.fail = False

        def load(self, _model_name: str, _alias: str) -> LoadedModel:
            if self.fail:
                raise ConnectionError("registre indisponible")
            return LoadedModel(predictor=ConstantPredictor(), metadata=metadata())

    loader = Loader()
    runtime = ModelRuntime(
        model_name="anime-score-regressor",
        alias="champion",
        loader=loader,
        max_batch_size=100,
    )
    runtime.reload()
    loader.fail = True

    with pytest.raises(ModelLoadError):
        runtime.reload()

    predictions, served_metadata = runtime.predict(pd.DataFrame({"x": [1]}))
    assert predictions == [7.0]
    assert served_metadata.version == "1"
    assert runtime.last_error() is not None


@pytest.mark.parametrize("invalid_prediction", [[float("nan")], [], [1.0, 2.0]])
def test_invalid_model_outputs_are_rejected(invalid_prediction):
    class InvalidPredictor:
        def predict(self, _model_input: pd.DataFrame):
            return invalid_prediction

    class Loader:
        def load(self, _model_name: str, _alias: str) -> LoadedModel:
            return LoadedModel(predictor=InvalidPredictor(), metadata=metadata())

    runtime = ModelRuntime(
        model_name="anime-score-regressor",
        alias="champion",
        loader=Loader(),
        max_batch_size=100,
    )
    runtime.reload()

    with pytest.raises(PredictionError):
        runtime.predict(pd.DataFrame({"x": [1]}))
