"""Charge fermée : 90 % de requêtes récurrentes, 10 % de requêtes uniques."""

import math
from itertools import count

import gevent
from locust import HttpUser, constant, events, task

from load_tests.workload import cold_payload, hot_payload

_hot_sequence = count()
_cold_sequence = count()


@events.init.add_listener
def refresh_gevent_clock(**_kwargs: object) -> None:
    """Évite qu'un import à froid réduise la durée effective, notamment sous Windows."""
    gevent.get_hub().loop.update_now()


class PredictionUser(HttpUser):
    """Sature volontairement l'API pour mesurer sa capacité maximale."""

    wait_time = constant(0)

    @task(9)
    def predict_hot(self) -> None:
        self._predict(hot_payload(next(_hot_sequence)), name="POST /predict [hot]")

    @task(1)
    def predict_cold(self) -> None:
        self._predict(cold_payload(next(_cold_sequence)), name="POST /predict [cold]")

    def _predict(self, payload: dict[str, object], *, name: str) -> None:
        with self.client.post("/predict", json=payload, name=name, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return

            try:
                body = response.json()
                prediction = float(body["prediction"])
                cached = body["cached"]
                persisted = body["persisted"]
                version = body["model"]["version"]
            except (KeyError, TypeError, ValueError) as exc:
                response.failure(f"Réponse invalide: {exc}")
                return

            if not math.isfinite(prediction):
                response.failure("Prédiction non finie")
            elif not isinstance(cached, bool):
                response.failure("Indicateur cached invalide")
            elif persisted is not True:
                response.failure("Trace PostgreSQL non persistée")
            elif not isinstance(version, str) or not version:
                response.failure("Version du modèle absente")
            else:
                response.success()
