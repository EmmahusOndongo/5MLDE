import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_grafana_dashboard_covers_service_feedback_and_prefect():
    dashboard = json.loads(
        (ROOT / "monitoring/grafana/dashboards/anime-mlops.json").read_text(encoding="utf-8")
    )
    panels = dashboard["panels"]
    titles = {panel["title"] for panel in panels}
    queries = "\n".join(
        target.get("expr", "") + target.get("rawSql", "")
        for panel in panels
        for target in panel.get("targets", [])
    )

    assert dashboard["uid"] == "anime-mlops-overview"
    assert len(panels) >= 15
    assert "Taux d'erreurs (5 min)" in titles
    assert "Prédiction vs score réel" in titles
    assert "Orchestrateur Prefect" in titles
    assert "feedbacks f JOIN predictions p" in queries
    assert "anime_prefect_up" in queries


def test_grafana_provisioning_keeps_database_password_out_of_git():
    datasource = (ROOT / "monitoring/grafana/provisioning/datasources/datasources.yml").read_text(
        encoding="utf-8"
    )
    alerts = (ROOT / "monitoring/grafana/provisioning/alerting/alert-rules.yml").read_text(
        encoding="utf-8"
    )

    assert "password: $__env{GRAFANA_DATABASE_PASSWORD}" in datasource
    assert "change-me" not in datasource
    assert "anime-api-error-rate" in alerts
    assert "anime-api-latency-p95" in alerts
    assert "anime-prefect-unavailable" in alerts


def test_prometheus_scrapes_api_and_prefect_exporter():
    config = (ROOT / "monitoring/prometheus/prometheus.yml").read_text(encoding="utf-8")

    assert "api:8000" in config
    assert "prefect-exporter:9105" in config
