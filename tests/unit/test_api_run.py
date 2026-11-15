from types import SimpleNamespace

from anime_mlops.api import run


def test_api_runner_uses_configured_network_and_worker_settings(monkeypatch):
    settings = SimpleNamespace(api=SimpleNamespace(host="api.internal", port=9000, workers=4))
    captured = {}

    monkeypatch.setattr(run, "get_settings", lambda: settings)
    monkeypatch.setattr(
        run.uvicorn,
        "run",
        lambda application, **options: captured.update({"application": application, **options}),
    )

    run.main()

    assert captured == {
        "application": "anime_mlops.api.main:app",
        "host": "api.internal",
        "port": 9000,
        "workers": 4,
        "access_log": True,
    }
