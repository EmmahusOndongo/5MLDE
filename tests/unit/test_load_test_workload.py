import json

from load_tests.workload import HOT_PAYLOADS, cold_payload, hot_payload

from anime_mlops.api.schemas import PredictionInput


def test_hot_payloads_are_valid_and_distinct():
    serialized = set()
    for payload in HOT_PAYLOADS:
        validated = PredictionInput.model_validate(payload)
        serialized.add(json.dumps(validated.model_dump(), sort_keys=True))

    assert len(HOT_PAYLOADS) == 8
    assert len(serialized) == len(HOT_PAYLOADS)


def test_payload_helpers_return_independent_copies():
    first = hot_payload(0)
    second = hot_payload(0)
    first["genres"].append("Comedy")

    assert second["genres"] == ["Action", "Drama"]


def test_cold_payloads_stay_valid_and_unique():
    payloads = [cold_payload(index) for index in range(100)]

    assert len({payload["prequel_score"] for payload in payloads}) == len(payloads)
    for payload in payloads:
        PredictionInput.model_validate(payload)
