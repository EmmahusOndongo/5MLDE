"""Exécute les six configurations B8 et consolide leurs mesures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import secrets
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
LOCUST_FILE = PROJECT_ROOT / "load_tests" / "locustfile.py"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "load_tests" / "results"
PROJECT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class Scenario:
    api_workers: int
    cache_enabled: bool

    @property
    def label(self) -> str:
        cache = "on" if self.cache_enabled else "off"
        return f"workers-{self.api_workers}_cache-{cache}"


@dataclass(frozen=True)
class ResourceSample:
    cpu_percent: float
    memory_mib: float


def build_scenarios(workers: list[int]) -> list[Scenario]:
    return [
        Scenario(api_workers=worker_count, cache_enabled=cache_enabled)
        for worker_count in workers
        for cache_enabled in (False, True)
    ]


def parse_cpu_percent(raw_value: str) -> float:
    return float(raw_value.strip().removesuffix("%"))


def parse_memory_mib(raw_value: str) -> float:
    used = raw_value.split("/", maxsplit=1)[0].strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)", used)
    if match is None:
        raise ValueError(f"Mesure mémoire Docker inconnue : {raw_value}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    factors = {
        "b": 1 / (1024 * 1024),
        "kb": 1000 / (1024 * 1024),
        "kib": 1 / 1024,
        "mb": 1_000_000 / (1024 * 1024),
        "mib": 1,
        "gb": 1_000_000_000 / (1024 * 1024),
        "gib": 1024,
        "tb": 1_000_000_000_000 / (1024 * 1024),
        "tib": 1024 * 1024,
    }
    if unit not in factors:
        raise ValueError(f"Unité mémoire Docker inconnue : {unit}")
    return value * factors[unit]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Une série vide n'a pas de percentile.")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def summarize_resources(samples: list[ResourceSample]) -> dict[str, float | int | None]:
    if not samples:
        return {
            "samples": 0,
            "cpu_percent_mean": None,
            "cpu_percent_p95": None,
            "cpu_percent_max": None,
            "memory_mib_mean": None,
            "memory_mib_max": None,
        }
    cpu_values = [sample.cpu_percent for sample in samples]
    memory_values = [sample.memory_mib for sample in samples]
    return {
        "samples": len(samples),
        "cpu_percent_mean": round(statistics.fmean(cpu_values), 3),
        "cpu_percent_p95": round(_percentile(cpu_values, 95), 3),
        "cpu_percent_max": round(max(cpu_values), 3),
        "memory_mib_mean": round(statistics.fmean(memory_values), 3),
        "memory_mib_max": round(max(memory_values), 3),
    }


def parse_locust_stats(path: Path) -> dict[str, int | float]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    aggregate = next((row for row in rows if row.get("Name") == "Aggregated"), None)
    if aggregate is None:
        raise ValueError(f"Ligne Aggregated absente de {path}")

    def integer(key: str) -> int:
        raw_value = aggregate.get(key)
        return 0 if not raw_value or raw_value == "N/A" else int(float(raw_value))

    def number(key: str) -> float:
        raw_value = aggregate.get(key)
        return 0.0 if not raw_value or raw_value == "N/A" else round(float(raw_value), 3)

    return {
        "requests": integer("Request Count"),
        "failures": integer("Failure Count"),
        "requests_per_second": number("Requests/s"),
        "failures_per_second": number("Failures/s"),
        "latency_mean_ms": number("Average Response Time"),
        "latency_p50_ms": number("50%"),
        "latency_p95_ms": number("95%"),
        "latency_p99_ms": number("99%"),
        "latency_max_ms": number("Max Response Time"),
    }


class ResourceSampler:
    def __init__(self, docker_binary: str, container_id: str, *, interval_seconds: float) -> None:
        self._docker_binary = docker_binary
        self._container_id = container_id
        self._interval_seconds = interval_seconds
        self._stop = Event()
        self._thread = Thread(target=self._collect, name="docker-stats", daemon=True)
        self.samples: list[ResourceSample] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval_seconds + 15)

    def _collect(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        self._docker_binary,
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{json .}}",
                        self._container_id,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                data = json.loads(completed.stdout.strip())
                self.samples.append(
                    ResourceSample(
                        cpu_percent=parse_cpu_percent(data["CPUPerc"]),
                        memory_mib=parse_memory_mib(data["MemUsage"]),
                    )
                )
            except (KeyError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
                # Une mesure manquante ne doit pas invalider les statistiques Locust.
                pass
            self._stop.wait(self._interval_seconds)


def validate_project_name(project_name: str, *, owns_stack: bool) -> None:
    if PROJECT_NAME_PATTERN.fullmatch(project_name) is None:
        raise ValueError("Le nom de projet Compose doit contenir a-z, 0-9, _ ou -.")
    if owns_stack and project_name in {"anime-mlops", "default"}:
        raise ValueError("Utiliser un nom de projet dédié aux tests de charge.")


def _docker_binary() -> str:
    binary = shutil.which("docker")
    if binary is None:
        raise RuntimeError("Docker est introuvable dans PATH.")
    return binary


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    capture_output: bool = False,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {subprocess.list2cmdline(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=check,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


def _compose_command(docker_binary: str, project_name: str, *arguments: str) -> list[str]:
    return [
        docker_binary,
        "compose",
        "--project-name",
        project_name,
        "--file",
        str(COMPOSE_FILE),
        *arguments,
    ]


def _compose(
    docker_binary: str,
    project_name: str,
    *arguments: str,
    environment: dict[str, str],
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        _compose_command(docker_binary, project_name, *arguments),
        environment=environment,
        capture_output=capture_output,
        check=check,
    )


def _wait_for_api(host: str, *, timeout_seconds: float = 120) -> dict[str, Any]:
    deadline = monotonic() + timeout_seconds
    last_reason = "aucune réponse"
    while monotonic() < deadline:
        try:
            with urlopen(f"{host}/health", timeout=3) as response:  # noqa: S310
                body = json.load(response)
            if response.status == 200 and body.get("status") in {"ok", "degraded"}:
                return body
            last_reason = f"statut inattendu : {response.status}"
        except HTTPError as exc:
            last_reason = f"HTTP {exc.code}"
        except (TimeoutError, URLError, json.JSONDecodeError) as exc:
            last_reason = str(exc)
        sleep(1)
    raise TimeoutError(f"L'API n'est pas prête après {timeout_seconds}s ({last_reason}).")


def _docker_metadata(docker_binary: str, environment: dict[str, str]) -> dict[str, Any]:
    try:
        completed = _run(
            [docker_binary, "info", "--format", "{{json .}}"],
            environment=environment,
            capture_output=True,
            timeout=30,
        )
        info = json.loads(completed.stdout)
    except (json.JSONDecodeError, subprocess.SubprocessError):
        return {}
    memory_bytes = info.get("MemTotal")
    return {
        "server_version": info.get("ServerVersion"),
        "operating_system": info.get("OperatingSystem"),
        "architecture": info.get("Architecture"),
        "cpus": info.get("NCPU"),
        "memory_gib": round(memory_bytes / 1024**3, 3) if memory_bytes else None,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bootstrap_stack(
    *,
    docker_binary: str,
    project_name: str,
    environment: dict[str, str],
    skip_build: bool,
) -> None:
    if not skip_build:
        _compose(docker_binary, project_name, "build", "api", environment=environment)
    _compose(
        docker_binary,
        project_name,
        "up",
        "--detach",
        "--wait",
        "postgres",
        "mlflow",
        environment=environment,
    )
    _compose(
        docker_binary,
        project_name,
        "run",
        "--rm",
        "--no-deps",
        "api",
        "python",
        "-m",
        "anime_mlops.api.register_mock_model",
        environment=environment,
    )


def _reset_application_tables(
    docker_binary: str,
    project_name: str,
    environment: dict[str, str],
) -> None:
    _compose(
        docker_binary,
        project_name,
        "exec",
        "--no-TTY",
        "postgres",
        "psql",
        "--username",
        environment["POSTGRES_USER"],
        "--dbname",
        "anime",
        "--set",
        "ON_ERROR_STOP=1",
        "--command",
        "TRUNCATE TABLE feedbacks, predictions RESTART IDENTITY CASCADE;",
        environment=environment,
    )


def _run_scenario(
    scenario: Scenario,
    *,
    args: argparse.Namespace,
    docker_binary: str,
    environment: dict[str, str],
    run_directory: Path,
) -> dict[str, Any]:
    scenario_environment = environment.copy()
    scenario_environment["API_WORKERS"] = str(scenario.api_workers)
    scenario_environment["ANIME_API__CACHE_ENABLED"] = str(scenario.cache_enabled).lower()
    _compose(
        docker_binary,
        args.project_name,
        "up",
        "--detach",
        "--no-deps",
        "--force-recreate",
        "--wait",
        "api",
        environment=scenario_environment,
    )
    health = _wait_for_api(args.host)
    _reset_application_tables(docker_binary, args.project_name, scenario_environment)

    container = _compose(
        docker_binary,
        args.project_name,
        "ps",
        "--quiet",
        "api",
        environment=scenario_environment,
        capture_output=True,
    ).stdout.strip()
    if not container:
        raise RuntimeError("Conteneur API introuvable après son démarrage.")

    output_prefix = run_directory / scenario.label
    sampler = ResourceSampler(
        docker_binary,
        container,
        interval_seconds=args.resource_sample_interval,
    )
    locust_command = [
        sys.executable,
        "-m",
        "locust",
        "--locustfile",
        str(LOCUST_FILE),
        "--headless",
        "--only-summary",
        "--host",
        args.host,
        "--users",
        str(args.users),
        "--spawn-rate",
        str(args.spawn_rate),
        "--run-time",
        args.run_time,
        "--csv",
        str(output_prefix),
        "--csv-full-history",
        "--html",
        f"{output_prefix}.html",
        "--exit-code-on-error",
        "1",
    ]
    sampler.start()
    started_at = datetime.now(UTC)
    try:
        _run(locust_command, environment=scenario_environment)
    finally:
        sampler.stop()
    finished_at = datetime.now(UTC)

    locust_stats = parse_locust_stats(Path(f"{output_prefix}_stats.csv"))
    model = health.get("model") or {}
    return {
        **asdict(scenario),
        "label": scenario.label,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "model_name": model.get("name"),
        "model_version": model.get("version"),
        "locust": locust_stats,
        "resources": summarize_resources(sampler.samples),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--spawn-rate", type=float, default=50)
    parser.add_argument("--run-time", default="60s")
    parser.add_argument("--resource-sample-interval", type=float, default=1)
    parser.add_argument("--project-name", default="anime-mlops-load")
    parser.add_argument("--api-port", type=int, default=8800)
    parser.add_argument("--postgres-port", type=int, default=55432)
    parser.add_argument("--mlflow-port", type=int, default=5500)
    parser.add_argument("--host", default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="réutilise PostgreSQL, MLflow et le champion d'un projet Compose existant",
    )
    parser.add_argument("--keep-stack", action="store_true")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.workers or any(worker < 1 or worker > 32 for worker in args.workers):
        raise ValueError("Les workers doivent être compris entre 1 et 32.")
    if args.users < 1 or args.spawn_rate <= 0:
        raise ValueError("users et spawn-rate doivent être strictement positifs.")
    if args.resource_sample_interval <= 0:
        raise ValueError("L'intervalle de mesure doit être strictement positif.")
    for port in (args.api_port, args.postgres_port, args.mlflow_port):
        if not 1 <= port <= 65_535:
            raise ValueError("Les ports doivent être compris entre 1 et 65535.")
    validate_project_name(args.project_name, owns_stack=not args.skip_bootstrap)


def run_matrix(args: argparse.Namespace) -> Path:
    _validate_arguments(args)
    docker_binary = _docker_binary()
    args.host = args.host or f"http://127.0.0.1:{args.api_port}"
    environment = os.environ.copy()
    if args.skip_bootstrap:
        missing = [
            key for key in ("POSTGRES_USER", "POSTGRES_PASSWORD") if not environment.get(key)
        ]
        if missing:
            raise ValueError(f"Variables requises avec --skip-bootstrap : {', '.join(missing)}")
    else:
        environment["POSTGRES_USER"] = "anime"
        environment["POSTGRES_PASSWORD"] = secrets.token_urlsafe(24)
    environment.update(
        {
            "API_PORT": str(args.api_port),
            "POSTGRES_PORT": str(args.postgres_port),
            "MLFLOW_PORT": str(args.mlflow_port),
        }
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = args.results_dir.resolve() / timestamp
    run_directory.mkdir(parents=True, exist_ok=False)
    scenarios: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "metadata": {
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
            "host_platform": platform.platform(),
            "host_logical_cpus": os.cpu_count(),
            "python_version": platform.python_version(),
            "locust_version": importlib_metadata.version("locust"),
            "docker": _docker_metadata(docker_binary, environment),
            "project_name": args.project_name,
            "model_source": "existing" if args.skip_bootstrap else "mock",
            "users": args.users,
            "spawn_rate": args.spawn_rate,
            "run_time": args.run_time,
            "workload": "90 % récurrente / 10 % unique, sans temps d'attente",
        },
        "scenarios": scenarios,
    }
    report_path = run_directory / "summary.json"
    _write_json(report_path, report)

    owns_stack = not args.skip_bootstrap
    try:
        if owns_stack:
            _bootstrap_stack(
                docker_binary=docker_binary,
                project_name=args.project_name,
                environment=environment,
                skip_build=args.skip_build,
            )
        elif not args.skip_build:
            _compose(docker_binary, args.project_name, "build", "api", environment=environment)

        for scenario in build_scenarios(args.workers):
            print(f"\n=== {scenario.label} ===", flush=True)
            result = _run_scenario(
                scenario,
                args=args,
                docker_binary=docker_binary,
                environment=environment,
                run_directory=run_directory,
            )
            scenarios.append(result)
            _write_json(report_path, report)
            stats = result["locust"]
            print(
                f"{stats['requests_per_second']} req/s, "
                f"p95={stats['latency_p95_ms']} ms, échecs={stats['failures']}",
                flush=True,
            )
        report["metadata"]["status"] = "completed"
    except Exception as exc:
        report["metadata"]["status"] = "failed"
        report["metadata"]["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["metadata"]["finished_at"] = datetime.now(UTC).isoformat()
        _write_json(report_path, report)
        if owns_stack and not args.keep_stack:
            _compose(
                docker_binary,
                args.project_name,
                "down",
                "--volumes",
                "--remove-orphans",
                environment=environment,
                check=False,
            )
    return report_path


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        report_path = run_matrix(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Échec de la matrice : {exc}", file=sys.stderr)
        return 1
    print(f"Rapport consolidé : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
