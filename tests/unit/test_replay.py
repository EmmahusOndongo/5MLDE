from datetime import date

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from anime_mlops.ingestion import client as ingestion_client
from anime_mlops.ingestion import landing
from anime_mlops.ingestion.receiver import create_app as create_receiver
from anime_mlops.ingestion.schemas import DailyBatch
from anime_mlops.replay.app import create_app as create_replay
from anime_mlops.replay.batches import (
    HIDDEN_COLUMNS,
    DayOutOfRangeError,
    ReplayCatalogue,
    release_day,
)

START = date(2025, 1, 1)
COMMIT = "a" * 40
PUSH_URL = "http://receiver.test/batches"


@pytest.fixture
def catalogue():
    rows = [
        # id, date de début, score
        (1, "1998-04-03", 8.75),  # historique
        (2, "2024-12-31", 7.0),  # historique (veille du départ)
        (3, None, 6.0),  # non daté : historique
        (10, "2025-01-01", 7.5),  # jour 1
        (11, "2025-01-01", None),  # jour 1, jamais noté
        (12, "2025-01-02", 6.5),  # jour 2
        (13, "2025-03", 8.0),  # date partielle → 1er mars
    ]
    data = pd.DataFrame(rows, columns=["id", "start_date", "mean"])
    data["title"] = "Anime " + data["id"].astype(str)
    data["media_type"] = "tv"
    data["rating"] = ["pg_13", None, "g", "r", "pg_13", None, "g"]
    data["num_episodes"] = 12
    data["average_episode_duration"] = 1440
    data["genres"] = "Action, Drama"
    data["studios"] = ["Sunrise", None, "Bones", "MAPPA", None, "Trigger", "Bones"]
    data["status"] = "finished_airing"
    data["end_date"] = "2025-06-30"
    data["rank"] = 100
    data["num_list_users"] = 50_000
    data["num_scoring_users"] = 20_000
    return ReplayCatalogue(data, COMMIT, START)


@pytest.fixture
def landing_dir(tmp_path):
    return tmp_path / "landing"


@pytest.fixture
def receiver(landing_dir):
    return TestClient(create_receiver(landing_dir))


@pytest.fixture
def replay(catalogue, receiver):
    """Source simulée dont le mode push livre au point de réception de test."""
    return TestClient(create_replay(catalogue, "http://testserver/batches", http_client=receiver))


# --- Découpage en lots -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025-01-15", date(2025, 1, 15)),
        ("2025-03", date(2025, 3, 1)),
        ("1999", date(1999, 1, 1)),
        (None, None),
        (float("nan"), None),
        ("pas-une-date", None),
        ("2025-13-01", None),
    ],
)
def test_release_day(value, expected):
    assert release_day(value) == expected


def test_history_is_everything_before_the_start_day(catalogue):
    history = catalogue.history()

    assert history["id"].tolist() == [1, 2, 3]
    assert "mean" in history.columns  # l'historique garde les scores : c'est le jeu d'entraînement
    assert "release_day" not in history.columns


def test_first_day_announces_without_revealing_anything(catalogue):
    batch = catalogue.batch(START)

    assert batch.day == START
    assert batch.source_commit == COMMIT
    assert [anime.id for anime in batch.announced] == [10, 11]
    assert batch.scores == []


def test_scores_are_revealed_the_next_day(catalogue):
    batch = catalogue.batch(date(2025, 1, 2))

    assert [anime.id for anime in batch.announced] == [12]
    # L'anime 11, jamais noté, n'a pas de score à révéler.
    assert [(score.id, score.mean) for score in batch.scores] == [(10, 7.5)]


def test_announcements_never_carry_audience_dependent_fields(catalogue):
    announced = catalogue.batch(START).announced[0].model_dump()

    assert not set(HIDDEN_COLUMNS) & set(announced)
    assert announced["status"] == "not_yet_aired"
    assert announced["studios"] == "MAPPA"


def test_missing_values_become_null(catalogue):
    second = catalogue.batch(START).announced[1]

    assert second.studios is None


def test_partial_date_is_released_on_the_first_day_of_the_month(catalogue):
    assert [anime.id for anime in catalogue.batch(date(2025, 3, 1)).announced] == [13]


def test_window_ends_the_day_after_the_last_release(catalogue):
    assert catalogue.end_day == date(2025, 3, 2)
    assert [score.id for score in catalogue.batch(catalogue.end_day).scores] == [13]


@pytest.mark.parametrize("day", [date(2024, 12, 31), date(2025, 3, 3)])
def test_day_outside_the_window_is_refused(catalogue, day):
    with pytest.raises(DayOutOfRangeError):
        catalogue.batch(day)


def test_days_lists_only_days_that_deliver_something(catalogue):
    summaries = catalogue.days()

    assert [(s.day.isoformat(), s.announced, s.scores) for s in summaries] == [
        ("2025-01-01", 2, 0),
        ("2025-01-02", 1, 1),
        ("2025-01-03", 0, 1),
        ("2025-03-01", 1, 0),
        ("2025-03-02", 0, 1),
    ]


def test_corrupt_batch_is_well_formed_but_has_invalid_content(catalogue):
    batch = catalogue.batch(date(2025, 1, 2), corrupt=True)

    assert all(score.mean == 15.0 for score in batch.scores)
    assert all(anime.num_episodes < 0 for anime in batch.announced)
    assert len(batch.announced) == 2  # un identifiant en double


def test_catalogue_without_anything_to_replay_is_refused(catalogue):
    with pytest.raises(ValueError, match="Aucun anime à rejouer"):
        ReplayCatalogue(catalogue.history(), COMMIT, START)


# --- Source simulée : mode pull ----------------------------------------------------


def test_health_describes_the_simulation(replay):
    body = replay.get("/health").json()

    assert body["status"] == "ok"
    assert body["source_commit"] == COMMIT
    assert body["start_day"] == "2025-01-01"


def test_pull_returns_the_daily_batch(replay):
    response = replay.get("/days/2025-01-02")

    assert response.status_code == 200
    batch = DailyBatch.model_validate(response.json())
    assert [anime.id for anime in batch.announced] == [12]


def test_pull_outside_the_window_is_a_404(replay):
    response = replay.get("/days/2030-01-01")

    assert response.status_code == 404
    assert "hors de la fenêtre" in response.json()["detail"]


def test_pull_with_an_invalid_date_is_a_422(replay):
    assert replay.get("/days/demain").status_code == 422


def test_days_can_be_filtered(replay):
    days = replay.get("/days", params={"first": "2025-01-02", "last": "2025-01-31"}).json()

    assert [day["day"] for day in days] == ["2025-01-02", "2025-01-03"]


# --- Zone de dépôt -----------------------------------------------------------------


def test_store_is_idempotent(catalogue, landing_dir):
    batch = catalogue.batch(START)

    first = landing.store_batch(batch, landing_dir)
    second = landing.store_batch(batch, landing_dir)

    assert first.status is landing.StoreStatus.CREATED
    assert second.status is landing.StoreStatus.UNCHANGED
    assert first.sha256 == second.sha256
    assert first.path.name == "batch_2025-01-01.json"
    assert landing.load_batch(START, landing_dir) == batch


def test_store_never_overwrites_a_different_batch(catalogue, landing_dir):
    original = catalogue.batch(date(2025, 1, 2))
    landing.store_batch(original, landing_dir)

    result = landing.store_batch(catalogue.batch(date(2025, 1, 2), corrupt=True), landing_dir)

    assert result.status is landing.StoreStatus.CONFLICT
    assert landing.load_batch(date(2025, 1, 2), landing_dir) == original


def test_landed_days_are_sorted_and_ignore_other_files(catalogue, landing_dir):
    for day in (date(2025, 1, 2), START):
        landing.store_batch(catalogue.batch(day), landing_dir)
    (landing_dir / "batch_brouillon.json").write_text("{}", "utf-8")

    assert landing.landed_days(landing_dir) == [START, date(2025, 1, 2)]


def test_loading_a_missing_batch_fails_explicitly(landing_dir):
    with pytest.raises(FileNotFoundError, match="Aucun lot déposé"):
        landing.load_batch(START, landing_dir)


# --- Réception : mode push ---------------------------------------------------------


def test_receiver_stores_a_pushed_batch(catalogue, receiver, landing_dir):
    payload = catalogue.batch(START).model_dump(mode="json")

    response = receiver.post("/batches", json=payload)

    assert response.status_code == 201
    assert response.json()["status"] == "created"
    assert response.json()["announced"] == 2
    assert receiver.get("/batches").json() == {"days": ["2025-01-01"]}
    assert landing.landed_days(landing_dir) == [START]


def test_receiver_accepts_a_retry_of_the_same_batch(catalogue, receiver):
    payload = catalogue.batch(START).model_dump(mode="json")
    receiver.post("/batches", json=payload)

    response = receiver.post("/batches", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "unchanged"


def test_receiver_refuses_a_different_batch_for_the_same_day(catalogue, receiver):
    receiver.post("/batches", json=catalogue.batch(START).model_dump(mode="json"))
    other = catalogue.batch(START, corrupt=True).model_dump(mode="json")

    response = receiver.post("/batches", json=other)

    assert response.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"day": "2025-01-01"},  # champs manquants
        {
            "day": "2025-01-01",
            "source_commit": "x",
            "announced": [{"title": "sans id"}],
            "scores": [],
        },
        {"day": "2025-01-01", "source_commit": "x", "announced": [], "scores": [], "inattendu": 1},
    ],
)
def test_receiver_rejects_a_malformed_batch(receiver, landing_dir, payload):
    response = receiver.post("/batches", json=payload)

    assert response.status_code == 422
    assert landing.landed_days(landing_dir) == []


def test_receiver_accepts_a_well_formed_batch_with_invalid_values(catalogue, receiver):
    # Le contenu aberrant est reçu et tracé ; c'est le contrôle qualité du pipeline qui bloque.
    payload = catalogue.batch(date(2025, 1, 2), corrupt=True).model_dump(mode="json")

    assert receiver.post("/batches", json=payload).status_code == 201


# --- Bout en bout ------------------------------------------------------------------


def test_push_delivers_the_batch_to_the_receiver(replay, landing_dir, catalogue):
    response = replay.post("/days/2025-01-02/push")

    assert response.status_code == 200
    assert response.json()["receiver_status"] == 201
    assert landing.load_batch(date(2025, 1, 2), landing_dir) == catalogue.batch(date(2025, 1, 2))


def test_push_reports_an_unreachable_receiver(catalogue):
    def refuse(request):
        raise httpx.ConnectError("connexion refusée")

    unreachable = httpx.Client(transport=httpx.MockTransport(refuse))
    replay = TestClient(create_replay(catalogue, PUSH_URL, http_client=unreachable))

    response = replay.post("/days/2025-01-01/push")

    assert response.status_code == 502
    assert "injoignable" in response.json()["detail"]


def test_push_reports_a_batch_refused_by_the_receiver(catalogue):
    refusing = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(409)))
    replay = TestClient(create_replay(catalogue, PUSH_URL, http_client=refusing))

    response = replay.post("/days/2025-01-01/push")

    assert response.status_code == 502
    assert "409" in response.json()["detail"]


def test_pull_and_store_lands_the_batch(replay, landing_dir, catalogue):
    result = ingestion_client.pull_and_store(START, "http://testserver", landing_dir, client=replay)

    assert result.status is landing.StoreStatus.CREATED
    assert landing.load_batch(START, landing_dir) == catalogue.batch(START)


def test_pull_and_store_refuses_to_replace_a_different_batch(replay, landing_dir):
    ingestion_client.pull_and_store(START, "http://testserver", landing_dir, client=replay)

    with pytest.raises(ingestion_client.BatchConflictError):
        ingestion_client.pull_and_store(
            START, "http://testserver", landing_dir, client=replay, corrupt=True
        )


def test_pull_from_an_unavailable_source_fails_explicitly(replay):
    with pytest.raises(ingestion_client.SourceUnavailableError, match="HTTP 404"):
        ingestion_client.pull_batch(date(2030, 1, 1), "http://testserver", client=replay)

    def refuse(request):
        raise httpx.ConnectError("connexion refusée")

    down = httpx.Client(transport=httpx.MockTransport(refuse))
    with pytest.raises(ingestion_client.SourceUnavailableError, match="injoignable"):
        ingestion_client.pull_batch(START, "http://source.test", client=down)


def test_pull_rejects_a_malformed_answer():
    garbage = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"day": "?"}))
    )

    with pytest.raises(ingestion_client.SourceUnavailableError, match="format inattendu"):
        ingestion_client.pull_batch(START, "http://source.test", client=garbage)
