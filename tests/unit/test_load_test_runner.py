import csv

import pytest
from load_tests.run_matrix import (
    ResourceSample,
    Scenario,
    build_scenarios,
    parse_cpu_percent,
    parse_locust_stats,
    parse_memory_mib,
    summarize_resources,
    validate_project_name,
)


def test_scenario_matrix_covers_workers_and_cache_modes():
    scenarios = build_scenarios([1, 2, 4])

    assert scenarios == [
        Scenario(1, False),
        Scenario(1, True),
        Scenario(2, False),
        Scenario(2, True),
        Scenario(4, False),
        Scenario(4, True),
    ]
    assert scenarios[-1].label == "workers-4_cache-on"


def test_scenario_matrix_can_select_cache_only():
    assert build_scenarios([1, 4], (True,)) == [Scenario(1, True), Scenario(4, True)]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("12.50%", 12.5), ("512MiB / 8GiB", 512.0), ("1.5GiB / 8GiB", 1536.0)],
)
def test_docker_resource_values_are_parsed(raw_value, expected):
    if raw_value.endswith("%"):
        result = parse_cpu_percent(raw_value)
    else:
        result = parse_memory_mib(raw_value)

    assert result == pytest.approx(expected)


def test_resource_summary_contains_capacity_indicators():
    summary = summarize_resources(
        [
            ResourceSample(cpu_percent=50, memory_mib=100),
            ResourceSample(cpu_percent=150, memory_mib=120),
        ]
    )

    assert summary == {
        "samples": 2,
        "cpu_percent_mean": 100.0,
        "cpu_percent_p95": 150,
        "cpu_percent_max": 150,
        "memory_mib_mean": 110.0,
        "memory_mib_max": 120,
    }


def test_locust_aggregate_row_is_converted(tmp_path):
    path = tmp_path / "stats.csv"
    columns = [
        "Type",
        "Name",
        "Request Count",
        "Failure Count",
        "Average Response Time",
        "Max Response Time",
        "Requests/s",
        "Failures/s",
        "50%",
        "95%",
        "99%",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "Type": "",
                "Name": "Aggregated",
                "Request Count": "120",
                "Failure Count": "0",
                "Average Response Time": "12.3456",
                "Max Response Time": "50",
                "Requests/s": "24.1234",
                "Failures/s": "0",
                "50%": "10",
                "95%": "25",
                "99%": "40",
            }
        )

    assert parse_locust_stats(path) == {
        "requests": 120,
        "failures": 0,
        "requests_per_second": 24.123,
        "failures_per_second": 0.0,
        "latency_mean_ms": 12.346,
        "latency_p50_ms": 10.0,
        "latency_p95_ms": 25.0,
        "latency_p99_ms": 40.0,
        "latency_max_ms": 50.0,
    }


def test_owned_stack_cannot_reuse_the_development_project_name():
    with pytest.raises(ValueError, match="dédié"):
        validate_project_name("anime-mlops", owns_stack=True)

    validate_project_name("anime-mlops", owns_stack=False)
