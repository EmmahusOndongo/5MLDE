"""Démonstrations nominale et dégradée du service de prédiction.

Le scénario PostgreSQL ne manipule qu'un service nommé ``postgres`` dans le
projet Docker Compose explicitement confirmé. Le redémarrage est exécuté dans
un bloc ``finally`` pour réduire le risque de laisser la stack en panne.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Protocol

import httpx

_PROJECT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{1,62}")


class DemoError(RuntimeError):
    """Le comportement observé ne correspond pas au contrat démontré."""


class HttpClient(Protocol):
    def get(self, url: str) -> httpx.Response: ...

    def post(self, url: str, *, json: dict[str, object] | None = None) -> httpx.Response: ...


class PostgresController(Protocol):
    def verify(self) -> None: ...

    def stop(self) -> None: ...

    def start(self) -> None: ...


@dataclass(frozen=True)
class NominalResult:
    model_version: str
    prediction: float
    first_cached: bool
    second_cached: bool
    feedback_source: str
    metrics_exposed: bool


@dataclass(frozen=True)
class PostgresIncidentResult:
    project_name: str
    degraded_status: str
    prediction_while_down: float
    persisted_while_down: bool
    feedback_error: str
    recovered_status: str
    persisted_after_recovery: bool


class ComposePostgresController:
    """Arrête/redémarre PostgreSQL dans un projet Compose borné et vérifié."""

    def __init__(
        self,
        project_name: str,
        *,
        confirm_project: str,
        repository: Path,
        compose_file: Path | None = None,
    ) -> None:
        validate_project_confirmation(project_name, confirm_project)
        docker = shutil.which("docker")
        if docker is None:
            raise DemoError("Docker est introuvable dans PATH.")
        self.project_name = project_name
        self._repository = repository.resolve()
        selected_compose_file = compose_file or Path("docker-compose.yml")
        if not selected_compose_file.is_absolute():
            selected_compose_file = self._repository / selected_compose_file
        self._compose_file = selected_compose_file.resolve()
        try:
            self._compose_file.relative_to(self._repository)
        except ValueError as exc:
            raise DemoError("Le fichier Compose doit rester dans le dépôt.") from exc
        if not self._compose_file.is_file():
            raise DemoError(f"Fichier Compose introuvable : {self._compose_file}")
        self._base_command = [
            docker,
            "compose",
            "--project-name",
            project_name,
            "--file",
            str(self._compose_file),
        ]

    def verify(self) -> None:
        result = self._run("ps", "--quiet", "postgres")
        container_ids = [line for line in result.stdout.splitlines() if line.strip()]
        if len(container_ids) != 1:
            raise DemoError(
                f"Le projet {self.project_name!r} doit contenir exactement un service "
                "postgres déjà créé."
            )

    def stop(self) -> None:
        self._run("stop", "postgres", capture_output=False)

    def start(self) -> None:
        self._run("start", "postgres", capture_output=False)

    def _run(
        self, *arguments: str, capture_output: bool = True
    ) -> subprocess.CompletedProcess[str]:
        command = [*self._base_command, *arguments]
        return subprocess.run(  # noqa: S603 - arguments bornés, jamais de shell
            command,
            cwd=self._repository,
            check=True,
            capture_output=capture_output,
            text=True,
        )


def prediction_payload() -> dict[str, object]:
    """Entrée déterministe conforme au contrat du modèle."""
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


def run_nominal(client: HttpClient) -> NominalResult:
    """Prouve santé, version, persistance, feedback, cache et métriques."""
    health = _json_response(client.get("/health"), expected_status=200, step="santé")
    if health.get("status") != "ok" or health.get("database", {}).get("status") != "ok":
        raise DemoError(f"La stack n'est pas nominale : {health}")

    first = _prediction(client)
    second = _prediction(client)
    if not first.get("persisted") or not second.get("persisted"):
        raise DemoError("Une prédiction nominale n'a pas été persistée.")
    if first["model"]["version"] != second["model"]["version"]:
        raise DemoError("La version du modèle a changé entre deux appels consécutifs.")
    if float(first["prediction"]) != float(second["prediction"]):
        raise DemoError("La prédiction a changé pour deux entrées identiques.")
    if not second.get("cached", False):
        raise DemoError(
            "Le second appel n'a pas utilisé le cache. "
            "Activer ANIME_API__CACHE_ENABLED puis recréer l'API."
        )

    feedback = _json_response(
        client.post(
            "/feedback",
            json={
                "prediction_id": first["prediction_id"],
                "actual_score": 8.0,
                "comment": "Démonstration nominale reproductible.",
            },
        ),
        expected_status=201,
        step="feedback",
    )
    feedback_source = str(feedback.get("source", ""))
    if feedback_source != "manual":
        raise DemoError(f"Source de feedback inattendue : {feedback_source!r}.")
    metrics_response = client.get("/metrics")
    if metrics_response.status_code != 200:
        raise DemoError(f"Les métriques répondent HTTP {metrics_response.status_code}.")
    expected_metrics = ("anime_api_http_requests_total", "anime_api_model_info")
    metrics_exposed = all(metric in metrics_response.text for metric in expected_metrics)
    if not metrics_exposed:
        raise DemoError("Les métriques attendues ne sont pas toutes exposées.")

    return NominalResult(
        model_version=str(first["model"]["version"]),
        prediction=float(first["prediction"]),
        first_cached=bool(first.get("cached", False)),
        second_cached=bool(second.get("cached", False)),
        feedback_source=feedback_source,
        metrics_exposed=metrics_exposed,
    )


def run_postgres_incident(
    client: HttpClient,
    controller: PostgresController,
    *,
    project_name: str,
    timeout_seconds: float = 60,
    sleep_fn: Callable[[float], None] = sleep,
    clock: Callable[[], float] = monotonic,
) -> PostgresIncidentResult:
    """Prouve le service dégradé puis son rétablissement après une panne SQL."""
    controller.verify()
    stopped = False
    degraded_health: dict[str, Any] | None = None
    prediction: dict[str, Any] | None = None
    feedback_error = ""
    try:
        controller.stop()
        stopped = True
        degraded_health = _wait_for_database(
            client,
            "degraded",
            timeout_seconds=timeout_seconds,
            sleep_fn=sleep_fn,
            clock=clock,
        )
        prediction = _prediction(client)
        if prediction.get("persisted") is not False:
            raise DemoError("La prédiction devait signaler persisted=false pendant la panne.")
        feedback_response = client.post(
            "/feedback",
            json={"prediction_id": prediction["prediction_id"], "actual_score": 8.0},
        )
        feedback_body = _json_response(
            feedback_response,
            expected_status=503,
            step="feedback pendant la panne",
        )
        feedback_error = str(feedback_body.get("detail", {}).get("code", ""))
        if feedback_error != "database_unavailable":
            raise DemoError(f"Erreur feedback inattendue : {feedback_body}")
    finally:
        if stopped:
            controller.start()

    recovered_health = _wait_for_database(
        client,
        "ok",
        timeout_seconds=timeout_seconds,
        sleep_fn=sleep_fn,
        clock=clock,
    )
    recovered_prediction = _prediction(client)
    if not recovered_prediction.get("persisted"):
        raise DemoError("La persistance n'est pas revenue après le redémarrage PostgreSQL.")

    if degraded_health is None or prediction is None:
        raise DemoError("Le scénario dégradé n'a pas produit toutes ses observations.")
    return PostgresIncidentResult(
        project_name=project_name,
        degraded_status=str(degraded_health["database"]["status"]),
        prediction_while_down=float(prediction["prediction"]),
        persisted_while_down=bool(prediction["persisted"]),
        feedback_error=feedback_error,
        recovered_status=str(recovered_health["database"]["status"]),
        persisted_after_recovery=bool(recovered_prediction["persisted"]),
    )


def validate_project_confirmation(project_name: str, confirmation: str) -> None:
    if not _PROJECT_PATTERN.fullmatch(project_name):
        raise DemoError(
            "Nom de projet Compose refusé : utiliser 2 à 63 caractères minuscules, "
            "chiffres, tirets ou underscores."
        )
    if confirmation != project_name:
        raise DemoError(
            "Refus d'arrêter PostgreSQL : --confirm-project doit être exactement égal "
            f"à {project_name!r}."
        )


def _prediction(client: HttpClient) -> dict[str, Any]:
    return _json_response(
        client.post("/predict", json=prediction_payload()),
        expected_status=200,
        step="prédiction",
    )


def _wait_for_database(
    client: HttpClient,
    expected_status: str,
    *,
    timeout_seconds: float,
    sleep_fn: Callable[[float], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    deadline = clock() + timeout_seconds
    last_observation: object = None
    while clock() <= deadline:
        try:
            response = client.get("/health")
            last_observation = response.text
            if response.status_code in {200, 503}:
                body = response.json()
                if body.get("database", {}).get("status") == expected_status:
                    return body
        except (httpx.HTTPError, ValueError) as exc:
            last_observation = str(exc)
        sleep_fn(1)
    raise DemoError(
        f"PostgreSQL n'a pas atteint l'état {expected_status!r} dans le délai. "
        f"Dernière observation : {last_observation}"
    )


def _json_response(response: httpx.Response, *, expected_status: int, step: str) -> dict[str, Any]:
    if response.status_code != expected_status:
        raise DemoError(
            f"Étape {step!r} : HTTP {response.status_code}, attendu {expected_status}. "
            f"Réponse : {response.text[:500]}"
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise DemoError(f"Étape {step!r} : réponse JSON invalide.") from exc
    if not isinstance(body, dict):
        raise DemoError(f"Étape {step!r} : objet JSON attendu.")
    return body


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    nominal = commands.add_parser("nominal", help="prédiction, cache, feedback et métriques")
    nominal.add_argument("--api-url", default="http://127.0.0.1:8000")

    outage = commands.add_parser(
        "postgres-outage", help="arrête PostgreSQL, vérifie le mode dégradé, puis le redémarre"
    )
    outage.add_argument("--api-url", default="http://127.0.0.1:8000")
    outage.add_argument("--project-name", required=True)
    outage.add_argument("--confirm-project", required=True)
    outage.add_argument("--compose-file", type=Path)
    outage.add_argument("--timeout", type=float, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        with httpx.Client(base_url=arguments.api_url, timeout=10) as client:
            if arguments.command == "nominal":
                result: NominalResult | PostgresIncidentResult = run_nominal(client)
            else:
                repository = _repository_root()
                controller = ComposePostgresController(
                    arguments.project_name,
                    confirm_project=arguments.confirm_project,
                    repository=repository,
                    compose_file=arguments.compose_file,
                )
                result = run_postgres_incident(
                    client,
                    controller,
                    project_name=arguments.project_name,
                    timeout_seconds=arguments.timeout,
                )
    except (DemoError, httpx.HTTPError, subprocess.CalledProcessError) as exc:
        print(f"DÉMONSTRATION EN ÉCHEC : {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
