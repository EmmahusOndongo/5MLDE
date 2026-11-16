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


def test_api_runner_clears_stale_multiprocess_metrics(monkeypatch, tmp_path):
    metrics_directory = tmp_path / "prometheus"
    metrics_directory.mkdir()
    stale_metrics = metrics_directory / "counter_123.db"
    unrelated_file = metrics_directory / "keep.txt"
    stale_metrics.write_text("stale", encoding="utf-8")
    unrelated_file.write_text("keep", encoding="utf-8")

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(metrics_directory))
    run._prepare_prometheus_multiprocess_directory()

    assert not stale_metrics.exists()
    assert unrelated_file.read_text(encoding="utf-8") == "keep"
