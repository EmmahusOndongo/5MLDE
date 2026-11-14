import json
from datetime import date

import httpx
import pandas as pd
import pytest

from anime_mlops.config import AnilistSourceConfig
from anime_mlops.data.sources import anilist

TODAY = date(2026, 11, 11)


@pytest.fixture
def config():
    return AnilistSourceConfig(
        url="https://anilist.test/graphql",
        page_size=2,
        max_retries=2,
        retry_delay_seconds=5,
        request_interval_seconds=2,
    )


@pytest.fixture
def sleeps():
    """Remplace time.sleep : les tests n'attendent pas et vérifient les pauses demandées."""
    return []


def media(anilist_id, id_mal, relations=()):
    return {
        "id": anilist_id,
        "idMal": id_mal,
        "format": "TV",
        "source": "MANGA",
        "episodes": 12,
        "tags": [{"name": "Space"}, {"name": "Crime"}],
        "relations": {
            "edges": [
                {"relationType": kind, "node": {"idMal": related_id, "type": node_type}}
                for kind, related_id, node_type in relations
            ]
        },
    }


def page(items, has_next):
    return httpx.Response(
        200, json={"data": {"Page": {"pageInfo": {"hasNextPage": has_next}, "media": items}}}
    )


def make_client(responses):
    """Client simulé qui renvoie les réponses dans l'ordre et mémorise les lots demandés."""
    queue = list(responses)
    requested = []

    def handler(request):
        variables = json.loads(request.content)["variables"]
        requested.append((variables["ids"], variables["page"]))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = httpx.Client(transport=httpx.MockTransport(handler))
    client.requested = requested
    return client


def test_parse_media_detects_sequel():
    row = anilist.parse_media(media(10, 100, relations=[("PREQUEL", 99, "ANIME")]))

    assert row == {
        "id_mal": 100,
        "anilist_id": 10,
        "format": "TV",
        "source": "MANGA",
        "episodes": 12,
        "tags": ["Space", "Crime"],
        "is_sequel": True,
        "prequel_id_mal": 99,
    }


@pytest.mark.parametrize(
    "relations",
    [
        [],
        [("SEQUEL", 101, "ANIME")],
        [("PREQUEL", 55, "MANGA")],  # un manga d'origine n'est pas une saison précédente
    ],
)
def test_parse_media_without_anime_prequel_is_not_a_sequel(relations):
    row = anilist.parse_media(media(10, 100, relations=relations))

    assert row["is_sequel"] is False
    assert row["prequel_id_mal"] is None


def test_parse_media_tolerates_missing_optional_fields():
    row = anilist.parse_media({"id": 10, "idMal": 100, "tags": None, "relations": None})

    assert row["tags"] == []
    assert row["is_sequel"] is False
    assert row["source"] is None


def test_fetch_queries_by_batches_of_mal_ids_and_paces_requests(config, sleeps):
    client = make_client(
        [
            page([media(1, 100), media(2, 200)], has_next=False),
            page([media(3, 300)], has_next=False),
        ]
    )

    data = anilist.fetch_by_mal_ids(
        [300, 100, 200, 100], config, client=client, sleep=sleeps.append
    )

    # Identifiants dédoublonnés, triés, découpés par lots de page_size (2).
    assert client.requested == [([100, 200], 1), ([300], 1)]
    assert data["id_mal"].tolist() == [100, 200, 300]
    assert list(data.columns) == anilist.COLUMNS
    assert sleeps == [config.request_interval_seconds]


def test_fetch_follows_pagination_inside_a_batch(config, sleeps):
    client = make_client(
        [
            page([media(1, 100), media(2, 100)], has_next=True),
            page([media(3, 200)], has_next=False),
        ]
    )

    data = anilist.fetch_by_mal_ids([100, 200], config, client=client, sleep=sleeps.append)

    assert client.requested == [([100, 200], 1), ([100, 200], 2)]
    assert data["id_mal"].tolist() == [100, 200]


def test_duplicate_entries_for_one_mal_id_keep_the_oldest(config, sleeps, caplog):
    client = make_client([page([media(9, 100), media(2, 100), media(5, 200)], has_next=False)])

    data = anilist.fetch_by_mal_ids([100, 200], config, client=client, sleep=sleeps.append)

    assert data[["id_mal", "anilist_id"]].values.tolist() == [[100, 2], [200, 5]]
    assert "en double" in caplog.text


def test_fetch_drops_entries_without_mal_id(config, sleeps):
    client = make_client([page([media(1, 100), media(2, None)], has_next=False)])

    data = anilist.fetch_by_mal_ids([100], config, client=client, sleep=sleeps.append)

    assert data["id_mal"].tolist() == [100]
    assert data["id_mal"].dtype == "int64"


def test_fetch_stops_at_max_batches(config, sleeps):
    limited = config.model_copy(update={"max_batches": 1})
    client = make_client([page([media(1, 100)], has_next=False)])

    data = anilist.fetch_by_mal_ids([100, 200, 300], limited, client=client, sleep=sleeps.append)

    assert client.requested == [([100, 200], 1)]
    assert len(data) == 1


def test_fetch_without_ids_makes_no_request(config, sleeps):
    client = make_client([])

    data = anilist.fetch_by_mal_ids([], config, client=client, sleep=sleeps.append)

    assert client.requested == []
    assert data.empty
    assert list(data.columns) == anilist.COLUMNS


def test_server_error_is_retried(config, sleeps):
    client = make_client(
        [
            httpx.Response(503),
            httpx.ConnectError("réseau coupé"),
            page([media(1, 100)], has_next=False),
        ]
    )

    data = anilist.fetch_by_mal_ids([100], config, client=client, sleep=sleeps.append)

    assert len(data) == 1
    assert sleeps == [config.retry_delay_seconds, config.retry_delay_seconds]


def test_rate_limit_waits_for_retry_after(config, sleeps):
    client = make_client(
        [
            httpx.Response(429, headers={"Retry-After": "42"}),
            page([media(1, 100)], has_next=False),
        ]
    )

    anilist.fetch_by_mal_ids([100], config, client=client, sleep=sleeps.append)

    assert 42.0 in sleeps


def test_persistent_failure_raises_after_all_attempts(config, sleeps):
    client = make_client([httpx.Response(500)] * 3)

    with pytest.raises(anilist.AnilistUnavailableError, match="3 tentative"):
        anilist.fetch_by_mal_ids([100], config, client=client, sleep=sleeps.append)

    assert len(client.requested) == 3


def test_client_error_is_not_retried(config, sleeps):
    client = make_client([httpx.Response(400)])

    with pytest.raises(anilist.AnilistUnavailableError, match="HTTP 400"):
        anilist.fetch_by_mal_ids([100], config, client=client, sleep=sleeps.append)

    assert len(client.requested) == 1


def test_unexpected_payload_raises(config, sleeps):
    client = make_client([httpx.Response(200, json={"errors": [{"message": "boom"}]})])

    with pytest.raises(anilist.AnilistUnavailableError, match="format inattendu"):
        anilist.fetch_by_mal_ids([100], config, client=client, sleep=sleeps.append)


def test_get_extract_caches_then_reuses(config, sleeps, tmp_path):
    client = make_client([page([media(1, 100)], has_next=False)])

    first = anilist.get_extract([100], config, tmp_path, TODAY, client, sleeps.append)
    second = anilist.get_extract([100], config, tmp_path, TODAY, client, sleeps.append)

    assert first.path.name == "anilist_2026-11-11.parquet"
    assert first.extraction_date == TODAY
    assert not first.is_fallback
    assert len(client.requested) == 1  # le second appel n'interroge pas l'API
    pd.testing.assert_frame_equal(first.data, second.data, check_dtype=False)
    assert second.data["tags"].iloc[0].tolist() == ["Space", "Crime"]


def test_get_extract_falls_back_to_latest_cache_when_api_is_down(config, sleeps, tmp_path):
    for day in (date(2026, 11, 3), date(2026, 11, 9)):
        client = make_client([page([media(day.day, day.day * 10)], has_next=False)])
        anilist.get_extract([100], config, tmp_path, day, client, sleeps.append)

    failing = make_client([httpx.Response(500)] * 3)
    extract = anilist.get_extract([100], config, tmp_path, TODAY, failing, sleeps.append)

    assert extract.is_fallback
    assert extract.extraction_date == date(2026, 11, 9)
    assert extract.data["id_mal"].tolist() == [90]
    assert not (tmp_path / "anilist" / "anilist_2026-11-11.parquet").exists()


def test_get_extract_without_cache_propagates_failure(config, sleeps, tmp_path):
    failing = make_client([httpx.Response(500)] * 3)

    with pytest.raises(anilist.AnilistUnavailableError, match="aucun cache"):
        anilist.get_extract([100], config, tmp_path, TODAY, failing, sleeps.append)
