"""Étapes du pipeline, alertes et flows Prefect (serveur Prefect temporaire, sans réseau)."""

import json
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from prefect.testing.utilities import prefect_test_harness

from anime_mlops import alerts
from anime_mlops.config import Settings, get_settings
from anime_mlops.data.sources import anilist
from anime_mlops.flows.daily import daily_ingestion_flow
from anime_mlops.flows.training import training_flow
from anime_mlops.ingestion import landing
from anime_mlops.pipeline import steps
from anime_mlops.replay.app import create_app as create_replay
from anime_mlops.replay.app import load_catalogue

COMMIT = "c" * 40
DAY_1 = date(2025, 1, 1)
DAY_2 = date(2025, 1, 2)
DAY_3 = date(2025, 1, 3)
HISTORY_ROWS = 150


@pytest.fixture(scope="module", autouse=True)
def prefect_server():
    """Un serveur Prefect temporaire pour tout le module : aucun flow ne touche le vrai."""
    with prefect_test_harness():
        yield


def make_snapshot() -> pd.DataFrame:
    """Catalogue synthétique : un historique noté, puis quelques animes rejoués en 2025."""
    rng = np.random.default_rng(1)
    is_action = rng.random(HISTORY_ROWS) < 0.5
    history = pd.DataFrame(
        {
            "id": range(1, HISTORY_ROWS + 1),
            "start_date": "2020-04-03",
            "mean": np.round(6.0 + is_action * 1.5 + rng.normal(0, 0.1, HISTORY_ROWS), 2),
            "genres": np.where(is_action, "Action, Drama", "Comedy"),
        }
    )
    replayed = pd.DataFrame(
        {
            "id": [1001, 1002, 1003],
            "start_date": ["2025-01-01", "2025-01-01", "2025-01-02"],
            "mean": [7.5, None, 6.5],
            "genres": ["Action, Drama", "Comedy", "Comedy"],
        }
    )
    data = pd.concat([history, replayed], ignore_index=True)
    data["title"] = "Anime " + data["id"].astype(str)
    data["media_type"] = "tv"
    data["rating"] = "pg_13"
    data["num_episodes"] = 12
    data["average_episode_duration"] = 1440
    data["studios"] = "Sunrise"
    data["status"] = "finished_airing"
    data["end_date"] = "2025-06-30"
    data["rank"] = 100
    data["num_list_users"] = 50_000
    data["num_scoring_users"] = 20_000
    return data


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """Projet miniature dans un dossier temporaire : données, MLflow local, seuils réduits."""
    data_dir = tmp_path / "data"
    raw = data_dir / "raw" / "mal"
    raw.mkdir(parents=True)
    make_snapshot().to_csv(raw / f"anime-standalone_{COMMIT}.csv", index=False)

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name, value in {
        "ANIME_PATHS__RAW_DIR": data_dir / "raw",
        "ANIME_PATHS__CACHE_DIR": data_dir / "cache",
        "ANIME_PATHS__REPORTS_DIR": data_dir / "reports",
        "ANIME_PATHS__LANDING_DIR": data_dir / "landing",
        "ANIME_PATHS__ACCEPTED_DIR": data_dir / "accepted",
        "ANIME_PATHS__QUARANTINE_DIR": data_dir / "quarantine",
        "ANIME_PATHS__PREDICTIONS_DIR": data_dir / "predictions",
        # Le feedback automatique a ses propres tests (test_feedback.py).
        "ANIME_FEEDBACK__ENABLED": "false",
        "ANIME_ALERTS__FILE": data_dir / "reports" / "alerts.jsonl",
        "ANIME_SOURCES__MAL__COMMIT": COMMIT,
        "ANIME_QUALITY__MIN_ROWS": 20,
        "ANIME_MODEL__N_ESTIMATORS": 10,
        "ANIME_MODEL__N_JOBS": 1,
        "ANIME_MLFLOW__TRACKING_URI": f"sqlite:///{tmp_path.as_posix()}/mlflow.db",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def source(settings):
    """La source simulée, bâtie sur le catalogue synthétique."""
    return load_catalogue(settings)


def land(source, settings, day, corrupt=False):
    """Simule la réception du lot en push : il attend dans la zone de dépôt."""
    landing.store_batch(source.batch(day, corrupt=corrupt), settings.paths.landing_dir)


def read_alerts(settings) -> list[dict]:
    if not settings.alerts.file.is_file():
        return []
    return [json.loads(line) for line in settings.alerts.file.read_text("utf-8").splitlines()]


# --- Étapes ----------------------------------------------------------------------


def test_next_day_starts_at_the_replay_start_then_follows_processed_days(settings, source):
    assert steps.next_day(settings) == DAY_1

    landing.store_batch(source.batch(DAY_1), settings.paths.accepted_dir)
    landing.store_batch(source.batch(DAY_2), settings.paths.quarantine_dir)

    # Un lot rejeté compte comme traité : on ne reste pas bloqué dessus.
    assert steps.next_day(settings) == DAY_3


def test_obtain_batch_pulls_when_nothing_was_pushed(settings, source):
    replay = TestClient(create_replay(source, "http://unused.test"))
    settings.replay.source_url = "http://testserver"

    batch = steps.obtain_batch(DAY_1, settings, client=replay)

    assert [anime.id for anime in batch.announced] == [1001, 1002]
    assert landing.landed_days(settings.paths.landing_dir) == [DAY_1]


def test_obtain_batch_uses_the_pushed_batch_without_calling_the_source(settings, source):
    land(source, settings, DAY_1)

    def must_not_be_called(request):
        raise AssertionError("la source ne doit pas être appelée")

    client = httpx.Client(transport=httpx.MockTransport(must_not_be_called))

    assert steps.obtain_batch(DAY_1, settings, client=client) == source.batch(DAY_1)


def test_valid_batch_passes_both_checks(settings, source):
    batch = source.batch(DAY_2)
    known = set(source.history()["id"]) | {1001, 1002}

    assert steps.check_announcements(batch, known, settings).success
    assert steps.check_scores(batch, settings).success


def test_corrupt_batch_fails_the_checks(settings, source):
    batch = source.batch(DAY_2, corrupt=True)

    announcements = steps.check_announcements(batch, set(), settings)
    scores = steps.check_scores(batch, settings)

    assert {failure.column for failure in announcements.failures} == {"id", "num_episodes"}
    assert {failure.column for failure in scores.failures} == {"mean"}


def test_announcing_an_already_known_anime_is_blocking(settings, source):
    report = steps.check_announcements(source.batch(DAY_1), {1001}, settings)

    assert [failure.column for failure in report.failures] == ["already_known"]


def test_empty_batch_is_valid(settings, source):
    batch = source.batch(DAY_3).model_copy(update={"scores": []})

    assert steps.check_announcements(batch, set(), settings).success
    assert steps.check_scores(batch, settings).success


def test_catalogue_grows_with_accepted_batches_and_revealed_scores(source):
    history = source.history()

    after_day_1 = steps.build_catalogue(history, [source.batch(DAY_1)])
    after_day_2 = steps.build_catalogue(history, [source.batch(DAY_1), source.batch(DAY_2)])

    assert len(after_day_1) == HISTORY_ROWS + 2
    announced = after_day_1.set_index("id").loc[1001]
    assert pd.isna(announced["mean"])
    assert announced["status"] == "not_yet_aired"

    revealed = after_day_2.set_index("id").loc[1001]
    assert revealed["mean"] == 7.5
    assert revealed["status"] == "finished_airing"
    # 1002 n'est jamais noté : il reste hors du jeu d'entraînement.
    assert after_day_2.set_index("id").loc[1002, "status"] == "not_yet_aired"
    assert after_day_2["id"].is_unique


def test_catalogue_without_batches_is_the_history(source):
    assert len(steps.build_catalogue(source.history(), [])) == HISTORY_ROWS


def test_orphan_scores_are_counted(source):
    batch = source.batch(DAY_2)

    assert steps.count_orphan_scores(batch, {1001, 1002}) == 0
    assert steps.count_orphan_scores(batch, set()) == 1


# --- Alertes ---------------------------------------------------------------------


def test_alert_is_appended_to_the_alert_file(settings):
    alerts.send_alert(
        alerts.Alert(alerts.Level.WARNING, "Titre", "Message", {"k": 1}), settings.alerts
    )
    alerts.send_alert(alerts.Alert(alerts.Level.CRITICAL, "Autre", "Message"), settings.alerts)

    recorded = read_alerts(settings)
    assert [(a["level"], a["title"]) for a in recorded] == [
        ("warning", "Titre"),
        ("critical", "Autre"),
    ]
    assert recorded[0]["context"] == {"k": 1}
    assert recorded[0]["raised_at"]


def test_alert_is_posted_to_the_webhook_when_configured(settings):
    received = []

    def handler(request):
        received.append(json.loads(request.content))
        return httpx.Response(200)

    config = settings.alerts.model_copy(update={"webhook_url": "https://hooks.test/secret"})
    config = type(config).model_validate(config.model_dump())
    client = httpx.Client(transport=httpx.MockTransport(handler))

    alerts.send_alert(alerts.Alert(alerts.Level.CRITICAL, "Pipeline", "en échec"), config, client)

    assert "[CRITICAL] Pipeline" in received[0]["text"]


def test_failing_webhook_never_breaks_the_caller(settings, caplog):
    config = type(settings.alerts).model_validate(
        {**settings.alerts.model_dump(), "webhook_url": "https://hooks.test/secret"}
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))

    alerts.send_alert(alerts.Alert(alerts.Level.CRITICAL, "Pipeline", "en échec"), config, client)

    assert len(read_alerts(settings)) == 1
    assert "hooks.test" not in caplog.text  # l'adresse du webhook est un secret


# --- Flow quotidien --------------------------------------------------------------


def test_daily_flow_accepts_a_valid_batch(settings, source):
    land(source, settings, DAY_1)

    result = daily_ingestion_flow()

    assert result == {
        "day": "2025-01-01",
        "status": "accepted",
        "announced": 2,
        "scores": 0,
        "orphan_scores": 0,
        "feedback": None,
    }
    assert landing.landed_days(settings.paths.accepted_dir) == [DAY_1]
    assert landing.landed_days(settings.paths.quarantine_dir) == []
    assert read_alerts(settings) == []


def test_daily_flow_is_idempotent(settings, source):
    land(source, settings, DAY_1)
    daily_ingestion_flow(DAY_1)

    assert daily_ingestion_flow(DAY_1)["status"] == "already_processed"


def test_daily_flow_blocks_and_quarantines_an_invalid_batch(settings, source):
    land(source, settings, DAY_1)
    daily_ingestion_flow()
    land(source, settings, DAY_2, corrupt=True)

    state = daily_ingestion_flow(return_state=True)

    assert state.is_failed()
    assert landing.landed_days(settings.paths.accepted_dir) == [DAY_1]
    assert landing.landed_days(settings.paths.quarantine_dir) == [DAY_2]
    # Un rapport de validation en échec est archivé.
    assert list((settings.paths.reports_dir / "validation").glob("*_echec.json"))
    # Une alerte critique est émise par le crochet d'échec du flow.
    [alert] = read_alerts(settings)
    assert alert["level"] == "critical"
    assert "ingestion-journaliere" in alert["title"]
    assert "non conformes" in alert["message"]


def test_daily_flow_moves_on_after_a_rejected_day_and_flags_orphan_scores(settings, source):
    land(source, settings, DAY_1)
    daily_ingestion_flow()
    landing.store_batch(source.batch(DAY_2, corrupt=True), settings.paths.quarantine_dir)
    land(source, settings, DAY_3)

    result = daily_ingestion_flow()

    # Le jour 3 révèle le score d'un anime annoncé le jour 2, qui a été rejeté.
    assert result["day"] == "2025-01-03"
    assert result["status"] == "accepted"
    assert result["orphan_scores"] == 1
    assert [alert["level"] for alert in read_alerts(settings)] == ["warning"]


# --- Flow d'entraînement ---------------------------------------------------------


@pytest.fixture
def anilist_stub(settings, monkeypatch):
    """Remplace l'appel à AniList par une extraction locale couvrant tout le catalogue."""
    calls = {"fallback": False}

    def fake_get_extract(mal_ids, config, cache_dir, **kwargs):
        ids = sorted(mal_ids)
        data = pd.DataFrame(
            {
                "id_mal": ids,
                "anilist_id": [i + 100_000 for i in ids],
                "format": "TV",
                "source": "MANGA",
                "episodes": 12.0,
                "tags": [["Space"]] * len(ids),
                "is_sequel": False,
                "prequel_id_mal": None,
            }
        )
        path = cache_dir / "anilist" / "anilist_2026-11-14.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path, index=False)
        return anilist.AnilistExtract(data, date(2026, 11, 14), path, is_fallback=calls["fallback"])

    monkeypatch.setattr(anilist, "get_extract", fake_get_extract)
    return calls


def test_training_flow_trains_registers_and_promotes(settings, source, anilist_stub):
    for day in (DAY_1, DAY_2):
        land(source, settings, day)
        daily_ingestion_flow()

    result = training_flow()

    # L'historique, plus l'anime 1001 dont le score a été révélé le jour 2.
    assert result["training_rows"] == HISTORY_ROWS + 1
    assert result["batches"] == 2
    assert result["status"] == "champion"

    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    champion = client.get_model_version_by_alias(settings.mlflow.registered_model_name, "champion")
    run = client.get_run(champion.run_id)
    assert run.data.tags["data.mal_commit"] == COMMIT
    assert run.data.tags["data.replay_batches"] == "2"
    assert run.data.tags["data.validation"] == "passed"
    validation_artifacts = [a.path for a in client.list_artifacts(champion.run_id, "validation")]
    assert len(validation_artifacts) == 3


def test_training_flow_alerts_when_anilist_falls_back_to_cache(settings, source, anilist_stub):
    anilist_stub["fallback"] = True

    result = training_flow()

    assert result["status"] == "champion"
    [alert] = read_alerts(settings)
    assert alert["level"] == "warning"
    assert "AniList" in alert["title"]


def test_training_flow_is_blocked_by_invalid_data(settings, source, anilist_stub, monkeypatch):
    monkeypatch.setenv("ANIME_QUALITY__MIN_ROWS", "100000")  # seuil inatteignable
    get_settings.cache_clear()

    state = training_flow(return_state=True)

    assert state.is_failed()
    client = MlflowClient(tracking_uri=get_settings().mlflow.tracking_uri)
    assert client.search_registered_models() == []  # rien n'a été entraîné
    assert read_alerts(get_settings())[-1]["level"] == "critical"
