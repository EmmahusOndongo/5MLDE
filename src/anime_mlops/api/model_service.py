"""Chargement atomique et exécution du modèle servi."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock, RLock
from typing import Protocol

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.pyfunc import load_model

from anime_mlops.api.schemas import ModelMetadata


class ModelUnavailableError(RuntimeError):
    """Aucun modèle n'est actuellement disponible pour servir une prédiction."""


class ModelLoadError(RuntimeError):
    """Le chargement d'une version du modèle a échoué."""


class PredictionError(RuntimeError):
    """Le modèle chargé n'a pas produit des prédictions valides."""


class Predictor(Protocol):
    def predict(self, model_input: pd.DataFrame): ...


@dataclass(frozen=True)
class LoadedModel:
    predictor: Predictor
    metadata: ModelMetadata


class ModelLoader(Protocol):
    def load(self, model_name: str, alias: str) -> LoadedModel: ...


class PredictionCacheProtocol(Protocol):
    def get(self, key: str, *, model_version: str) -> tuple[float, ...] | None: ...

    def put(self, key: str, predictions: list[float], *, model_version: str) -> None: ...


class MlflowRegistryLoader:
    """Résout un alias MLflow puis charge explicitement la version correspondante."""

    def __init__(self, tracking_uri: str) -> None:
        self._tracking_uri = tracking_uri

    def load(self, model_name: str, alias: str) -> LoadedModel:
        mlflow.set_tracking_uri(self._tracking_uri)
        client = MlflowClient(tracking_uri=self._tracking_uri)
        model_version = client.get_model_version_by_alias(model_name, alias)

        # Charger l'URI immuable évite qu'un changement d'alias entre les deux appels
        # rende les métadonnées incohérentes avec le modèle réellement en mémoire.
        model_uri = f"models:/{model_name}/{model_version.version}"
        predictor = load_model(model_uri)
        metadata = ModelMetadata(
            name=model_name,
            alias=alias,
            version=str(model_version.version),
            run_id=getattr(model_version, "run_id", None),
            model_uri=model_uri,
            loaded_at=datetime.now(UTC),
        )
        return LoadedModel(predictor=predictor, metadata=metadata)


class ModelRuntime:
    """Conserve le modèle courant et permet son remplacement sans couper le service."""

    def __init__(
        self,
        *,
        model_name: str,
        alias: str,
        loader: ModelLoader,
        max_batch_size: int,
    ) -> None:
        self.model_name = model_name
        self.alias = alias
        self.max_batch_size = max_batch_size
        self._loader = loader
        self._loaded: LoadedModel | None = None
        self._last_error: str | None = None
        self._state_lock = RLock()
        self._reload_lock = Lock()

    def reload(self) -> ModelMetadata:
        """Charge une nouvelle version puis la substitue atomiquement à l'ancienne."""
        with self._reload_lock:
            try:
                candidate = self._loader.load(self.model_name, self.alias)
            except Exception as exc:
                with self._state_lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                raise ModelLoadError(
                    f"Impossible de charger {self.model_name}@{self.alias}."
                ) from exc

            with self._state_lock:
                self._loaded = candidate
                self._last_error = None
                return candidate.metadata

    def metadata(self) -> ModelMetadata:
        with self._state_lock:
            if self._loaded is None:
                raise ModelUnavailableError("Aucun modèle n'est chargé.")
            return self._loaded.metadata

    def is_ready(self) -> bool:
        with self._state_lock:
            return self._loaded is not None

    def last_error(self) -> str | None:
        with self._state_lock:
            return self._last_error

    def predict(self, model_input: pd.DataFrame) -> tuple[list[float], ModelMetadata]:
        loaded = self._loaded_model()
        return self._predict_loaded(loaded, model_input), loaded.metadata

    def predict_cached(
        self,
        model_input: pd.DataFrame,
        *,
        cache: PredictionCacheProtocol,
        cache_key: str,
    ) -> tuple[list[float], ModelMetadata, bool]:
        """Utilise un seul instantané du modèle pour la clé, le résultat et les métadonnées."""
        loaded = self._loaded_model()
        cached = cache.get(cache_key, model_version=loaded.metadata.version)
        if cached is not None:
            predictions = _coerce_predictions(cached, expected_count=len(model_input))
            return predictions, loaded.metadata, True

        predictions = self._predict_loaded(loaded, model_input)
        cache.put(cache_key, predictions, model_version=loaded.metadata.version)
        return predictions, loaded.metadata, False

    def _loaded_model(self) -> LoadedModel:
        with self._state_lock:
            loaded = self._loaded
        if loaded is None:
            raise ModelUnavailableError("Aucun modèle n'est chargé.")
        return loaded

    @staticmethod
    def _predict_loaded(loaded: LoadedModel, model_input: pd.DataFrame) -> list[float]:
        try:
            raw_predictions = loaded.predictor.predict(model_input)
            predictions = _coerce_predictions(raw_predictions, expected_count=len(model_input))
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError("Le modèle n'a pas pu calculer la prédiction.") from exc
        return predictions


def _coerce_predictions(raw_predictions, *, expected_count: int) -> list[float]:
    try:
        raw_values = list(raw_predictions)
    except TypeError:
        raw_values = [raw_predictions]

    if len(raw_values) != expected_count:
        raise PredictionError(
            f"Le modèle a renvoyé {len(raw_values)} valeur(s) pour {expected_count} entrée(s)."
        )

    predictions = []
    for raw_value in raw_values:
        if hasattr(raw_value, "item"):
            raw_value = raw_value.item()
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise PredictionError("Le modèle a renvoyé une valeur non numérique.") from exc
        if not math.isfinite(value):
            raise PredictionError("Le modèle a renvoyé une valeur non finie.")
        predictions.append(value)
    return predictions
