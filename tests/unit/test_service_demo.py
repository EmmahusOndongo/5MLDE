from collections.abc import Callable

import httpx
import pytest

from anime_mlops.demos.service import (
    ComposePostgresController,
    DemoError,
    run_nominal,
    run_postgres_incident,
    validate_project_confirmation,
)


def _response(status: int, body: dict | None = None, text: str | None = None) -> httpx.Response:
    request = httpx.Request("GET", "http://testserver")
    if body is not None:
        return httpx.Response(status, json=body, request=request)
    return httpx.Response(status, text=text or "", request=request)


class FakeClient:
    def __init__(self, handler: Callable[[str, str, dict | None], httpx.Response]) -> None:
        self._handler = handler

    def get(self, url: str) -> httpx.Response:
        return self._handler("GET", url, None)

    def post(self, url: str, *, json: dict | None = None) -> httpx.Response:
        return self._handler("POST", url, json)


class FakeController:
    def __init__(self) -> None:
        self.running = True
        self.events: list[str] = []

    def verify(self) -> None:
        self.events.append("verify")

    def stop(self) -> None:
        self.running = False
        self.events.append("stop")

    def start(self) -> None:
        self.running = True
        self.events.append("start")


def test_nominal_demo_proves_prediction_cache_feedback_and_metrics():
    prediction_count = 0

    def handler(method, path, payload):
        nonlocal prediction_count
        if path == "/health":
            return _response(200, {"status": "ok", "database": {"status": "ok"}})
        if path == "/predict":
            prediction_count += 1
            return _response(
                200,
                {
                    "prediction_id": f"00000000-0000-0000-0000-{prediction_count:012d}",
                    "prediction": 7.25,
                    "persisted": True,
                    "cached": prediction_count > 1,
                    "model": {"version": "4"},
                },
            )
        if path == "/feedback":
            assert method == "POST"
            assert payload["actual_score"] == 8.0
            return _response(201, {"source": "manual"})
        if path == "/metrics":
            return _response(200, text="anime_api_http_requests_total\nanime_api_model_info")
        raise AssertionError((method, path))

    result = run_nominal(FakeClient(handler))

    assert result.model_version == "4"
    assert result.first_cached is False
    assert result.second_cached is True
    assert result.feedback_source == "manual"
    assert result.metrics_exposed is True


def test_nominal_demo_rejects_a_missing_cache_hit():
    def handler(_method, path, _payload):
        if path == "/health":
            return _response(200, {"status": "ok", "database": {"status": "ok"}})
        if path == "/predict":
            return _response(
                200,
                {
                    "prediction_id": "00000000-0000-0000-0000-000000000001",
                    "prediction": 7.25,
                    "persisted": True,
                    "cached": False,
                    "model": {"version": "4"},
                },
            )
        raise AssertionError(path)

    with pytest.raises(DemoError, match="second appel.*cache"):
        run_nominal(FakeClient(handler))


def test_postgres_incident_restores_the_service_after_degraded_mode():
    controller = FakeController()
    prediction_count = 0

    def handler(_method, path, _payload):
        nonlocal prediction_count
        if path == "/health":
            status = "ok" if controller.running else "degraded"
            return _response(200, {"status": status, "database": {"status": status}})
        if path == "/predict":
            prediction_count += 1
            return _response(
                200,
                {
                    "prediction_id": f"00000000-0000-0000-0000-{prediction_count:012d}",
                    "prediction": 7.25,
                    "persisted": controller.running,
                    "model": {"version": "4"},
                },
            )
        if path == "/feedback" and not controller.running:
            return _response(503, {"detail": {"code": "database_unavailable"}})
        raise AssertionError(path)

    result = run_postgres_incident(
        FakeClient(handler),
        controller,
        project_name="anime-mlops-demo",
        sleep_fn=lambda _seconds: None,
    )

    assert controller.events == ["verify", "stop", "start"]
    assert result.degraded_status == "degraded"
    assert result.persisted_while_down is False
    assert result.feedback_error == "database_unavailable"
    assert result.recovered_status == "ok"
    assert result.persisted_after_recovery is True


def test_postgres_is_restarted_when_an_assertion_fails():
    controller = FakeController()

    def handler(_method, path, _payload):
        if path == "/health":
            status = "ok" if controller.running else "degraded"
            return _response(200, {"database": {"status": status}})
        if path == "/predict":
            return _response(
                200,
                {
                    "prediction_id": "00000000-0000-0000-0000-000000000001",
                    "prediction": 7.25,
                    "persisted": True,
                    "model": {"version": "4"},
                },
            )
        raise AssertionError(path)

    with pytest.raises(DemoError, match="persisted=false"):
        run_postgres_incident(
            FakeClient(handler),
            controller,
            project_name="anime-mlops-demo",
            sleep_fn=lambda _seconds: None,
        )

    assert controller.running is True
    assert controller.events[-1] == "start"


@pytest.mark.parametrize(
    ("project", "confirmation"),
    [
        ("anime-mlops", ""),
        ("anime-mlops", "un-autre-projet"),
        ("../anime-mlops", "../anime-mlops"),
        ("A", "A"),
    ],
)
def test_postgres_incident_requires_an_exact_safe_project_name(project, confirmation):
    with pytest.raises(DemoError):
        validate_project_confirmation(project, confirmation)


def test_postgres_incident_accepts_an_exact_safe_confirmation():
    validate_project_confirmation("anime-mlops", "anime-mlops")


def test_postgres_controller_rejects_a_compose_file_outside_repository(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    external_compose = tmp_path / "docker-compose.yml"
    external_compose.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setattr("anime_mlops.demos.service.shutil.which", lambda _name: "docker")

    with pytest.raises(DemoError, match="doit rester dans le dépôt"):
        ComposePostgresController(
            "anime-mlops",
            confirm_project="anime-mlops",
            repository=repository,
            compose_file=external_compose,
        )
