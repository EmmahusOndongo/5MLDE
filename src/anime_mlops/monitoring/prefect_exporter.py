"""Expose un résumé des exécutions Prefect au format Prometheus.

L'exporteur interroge uniquement l'API publique du serveur Prefect. Il reste
disponible quand Prefect est arrêté : ``anime_prefect_up`` passe alors à zéro et
les séries décrivant les anciennes exécutions sont retirées.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import sleep
from typing import Any
from urllib.request import Request, urlopen

from prometheus_client import CollectorRegistry, Gauge, start_http_server
from prometheus_client import Counter as PrometheusCounter

LOGGER = logging.getLogger(__name__)

JsonObject = Mapping[str, Any]
JsonRequester = Callable[[str, Mapping[str, object], float], object]


@dataclass(frozen=True)
class LatestFlowRun:
    """Dernière exécution connue d'un flow."""

    flow_name: str
    state: str
    started_at: float
    duration_seconds: float


@dataclass(frozen=True)
class PrefectSnapshot:
    """Vue bornée des exécutions renvoyées par Prefect."""

    recent_counts: Mapping[str, int]
    latest_by_flow: Mapping[str, LatestFlowRun]


@dataclass(frozen=True)
class PrefectMetrics:
    """Séries Prometheus possédées par l'exporteur."""

    up: Gauge
    scrape_errors: PrometheusCounter
    recent_runs: Gauge
    last_state: Gauge
    last_start: Gauge
    last_duration: Gauge

    @classmethod
    def create(cls, registry: CollectorRegistry) -> PrefectMetrics:
        return cls(
            up=Gauge(
                "anime_prefect_up",
                "1 si l'API Prefect a répondu au dernier relevé, sinon 0.",
                registry=registry,
            ),
            scrape_errors=PrometheusCounter(
                "anime_prefect_scrape_errors_total",
                "Nombre de relevés Prefect ayant échoué.",
                registry=registry,
            ),
            recent_runs=Gauge(
                "anime_prefect_flow_runs_recent",
                "Nombre d'exécutions dans la fenêtre bornée interrogée, par état.",
                ("state",),
                registry=registry,
            ),
            last_state=Gauge(
                "anime_prefect_flow_last_state_info",
                "État de la dernière exécution connue de chaque flow (série active à 1).",
                ("flow_name", "state"),
                registry=registry,
            ),
            last_start=Gauge(
                "anime_prefect_flow_last_start_timestamp_seconds",
                "Date de démarrage Unix de la dernière exécution connue de chaque flow.",
                ("flow_name",),
                registry=registry,
            ),
            last_duration=Gauge(
                "anime_prefect_flow_last_duration_seconds",
                "Durée de la dernière exécution connue de chaque flow.",
                ("flow_name",),
                registry=registry,
            ),
        )

    def clear_flow_series(self) -> None:
        self.recent_runs.clear()
        self.last_state.clear()
        self.last_start.clear()
        self.last_duration.clear()


class PrefectExporter:
    """Actualise les métriques à partir de snapshots indépendants et testables."""

    def __init__(
        self,
        metrics: PrefectMetrics,
        snapshot_loader: Callable[[], PrefectSnapshot],
    ) -> None:
        self._metrics = metrics
        self._snapshot_loader = snapshot_loader

    def collect_once(self) -> bool:
        try:
            snapshot = self._snapshot_loader()
        except Exception:  # noqa: BLE001 - une panne réseau ne doit pas tuer l'exporteur
            LOGGER.warning("Impossible d'interroger Prefect.", exc_info=True)
            self._metrics.up.set(0)
            self._metrics.scrape_errors.inc()
            self._metrics.clear_flow_series()
            return False

        self._metrics.clear_flow_series()
        for state, count in snapshot.recent_counts.items():
            self._metrics.recent_runs.labels(state=state).set(count)
        for flow_name, run in snapshot.latest_by_flow.items():
            self._metrics.last_state.labels(flow_name=flow_name, state=run.state).set(1)
            self._metrics.last_start.labels(flow_name=flow_name).set(run.started_at)
            self._metrics.last_duration.labels(flow_name=flow_name).set(run.duration_seconds)
        self._metrics.up.set(1)
        return True


def fetch_prefect_snapshot(
    api_url: str,
    *,
    limit: int = 200,
    timeout_seconds: float = 3,
    requester: JsonRequester | None = None,
) -> PrefectSnapshot:
    """Lit les flows et leurs dernières exécutions via l'API REST Prefect."""
    if limit < 1:
        raise ValueError("La limite Prefect doit être positive.")
    request_json = requester or _post_json
    base_url = api_url.rstrip("/")
    flows_payload = request_json(
        f"{base_url}/flows/filter",
        {"limit": limit, "offset": 0},
        timeout_seconds,
    )
    runs_payload = request_json(
        f"{base_url}/flow_runs/filter",
        {"limit": limit, "offset": 0, "sort": "START_TIME_DESC"},
        timeout_seconds,
    )
    flows = _object_sequence(flows_payload, "flows")
    runs = _object_sequence(runs_payload, "flow_runs")
    flow_names = {
        str(flow["id"]): str(flow["name"]) for flow in flows if flow.get("id") and flow.get("name")
    }

    recent_counts: Counter[str] = Counter()
    latest_by_flow: dict[str, LatestFlowRun] = {}
    for run in runs:
        state = _state_name(run)
        recent_counts[state] += 1
        flow_id = str(run.get("flow_id") or "unknown")
        flow_name = flow_names.get(flow_id, f"flow-{flow_id}")
        started_at = _parse_timestamp(run.get("start_time") or run.get("expected_start_time"))
        if started_at is None:
            continue
        candidate = LatestFlowRun(
            flow_name=flow_name,
            state=state,
            started_at=started_at,
            duration_seconds=_duration_seconds(run, started_at),
        )
        current = latest_by_flow.get(flow_name)
        if current is None or candidate.started_at > current.started_at:
            latest_by_flow[flow_name] = candidate

    return PrefectSnapshot(dict(recent_counts), latest_by_flow)


def _post_json(url: str, payload: Mapping[str, object], timeout_seconds: float) -> object:
    request = Request(  # noqa: S310 - URL HTTP(S) configurée par l'exploitant
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - URL configurée
        return json.load(response)


def _object_sequence(payload: object, endpoint: str) -> Sequence[JsonObject]:
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Réponse Prefect inattendue pour {endpoint}.")
    return payload


def _state_name(run: JsonObject) -> str:
    raw_state = run.get("state_type")
    if not raw_state and isinstance(run.get("state"), dict):
        raw_state = run["state"].get("type")
    return str(raw_state or "UNKNOWN").lower()


def _parse_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _duration_seconds(run: JsonObject, started_at: float) -> float:
    raw_duration = run.get("total_run_time")
    if isinstance(raw_duration, (int, float)):
        return max(float(raw_duration), 0.0)
    ended_at = _parse_timestamp(run.get("end_time"))
    return max((ended_at or datetime.now(UTC).timestamp()) - started_at, 0.0)


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    api_url = os.environ.get("PREFECT_API_URL", "http://prefect:4200/api")
    port = int(os.environ.get("PREFECT_EXPORTER_PORT", "9105"))
    poll_seconds = max(float(os.environ.get("PREFECT_EXPORTER_POLL_SECONDS", "15")), 1.0)
    timeout_seconds = max(float(os.environ.get("PREFECT_EXPORTER_TIMEOUT_SECONDS", "3")), 0.1)
    limit = max(int(os.environ.get("PREFECT_EXPORTER_RUN_LIMIT", "200")), 1)

    registry = CollectorRegistry()
    metrics = PrefectMetrics.create(registry)
    exporter = PrefectExporter(
        metrics,
        lambda: fetch_prefect_snapshot(
            api_url,
            limit=limit,
            timeout_seconds=timeout_seconds,
        ),
    )
    start_http_server(port, addr="0.0.0.0", registry=registry)  # noqa: S104 - conteneur
    LOGGER.info("Exporteur Prefect disponible sur le port %s.", port)
    while True:
        exporter.collect_once()
        sleep(poll_seconds)


if __name__ == "__main__":
    main()
