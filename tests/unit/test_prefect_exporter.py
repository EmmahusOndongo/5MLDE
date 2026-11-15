import pytest
from prometheus_client import CollectorRegistry, generate_latest

from anime_mlops.monitoring.prefect_exporter import (
    PrefectExporter,
    PrefectMetrics,
    PrefectSnapshot,
    fetch_prefect_snapshot,
)


def test_snapshot_maps_flow_names_states_and_latest_runs():
    responses = {
        "/flows/filter": [
            {"id": "flow-1", "name": "entrainement"},
            {"id": "flow-2", "name": "ingestion-quotidienne"},
        ],
        "/flow_runs/filter": [
            {
                "flow_id": "flow-1",
                "state_type": "COMPLETED",
                "start_time": "2026-11-15T10:00:00Z",
                "total_run_time": 12.5,
            },
            {
                "flow_id": "flow-1",
                "state_type": "FAILED",
                "start_time": "2026-11-14T10:00:00+00:00",
                "end_time": "2026-11-14T10:00:03+00:00",
            },
            {
                "flow_id": "flow-2",
                "state": {"type": "RUNNING"},
                "expected_start_time": "2026-11-15T10:01:00Z",
            },
        ],
    }

    def requester(url, _payload, _timeout):
        return next(value for suffix, value in responses.items() if url.endswith(suffix))

    snapshot = fetch_prefect_snapshot(
        "http://prefect:4200/api/",
        requester=requester,
    )

    assert snapshot.recent_counts == {"completed": 1, "failed": 1, "running": 1}
    assert snapshot.latest_by_flow["entrainement"].state == "completed"
    assert snapshot.latest_by_flow["entrainement"].duration_seconds == 12.5
    assert snapshot.latest_by_flow["ingestion-quotidienne"].state == "running"


def test_exporter_exposes_snapshot_and_clears_stale_labels():
    registry = CollectorRegistry()
    metrics = PrefectMetrics.create(registry)
    snapshots = iter(
        [
            PrefectSnapshot(
                recent_counts={"completed": 2},
                latest_by_flow={},
            ),
            PrefectSnapshot(
                recent_counts={"failed": 1},
                latest_by_flow={},
            ),
        ]
    )
    exporter = PrefectExporter(metrics, lambda: next(snapshots))

    assert exporter.collect_once() is True
    assert exporter.collect_once() is True
    exposition = generate_latest(registry).decode()

    assert "anime_prefect_up 1.0" in exposition
    assert 'anime_prefect_flow_runs_recent{state="failed"} 1.0' in exposition
    assert 'state="completed"' not in exposition


def test_exporter_survives_prefect_outage_and_reports_it():
    registry = CollectorRegistry()
    metrics = PrefectMetrics.create(registry)

    def unavailable():
        raise ConnectionError("Prefect arrêté")

    exporter = PrefectExporter(metrics, unavailable)

    assert exporter.collect_once() is False
    exposition = generate_latest(registry).decode()

    assert "anime_prefect_up 0.0" in exposition
    assert "anime_prefect_scrape_errors_total 1.0" in exposition


def test_snapshot_rejects_unexpected_prefect_response():
    def requester(url, _payload, _timeout):
        return {} if url.endswith("/flows/filter") else []

    with pytest.raises(ValueError, match="flows"):
        fetch_prefect_snapshot("http://prefect:4200/api", requester=requester)
