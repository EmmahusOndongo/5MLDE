"""Feedback automatique : vraie API (modèle factice) et vraie persistance, sur SQLite."""

import json
from contextlib import contextmanager
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from prefect.testing.utilities import prefect_test_harness
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from anime_mlops.api.main import create_app
from anime_mlops.api.persistence import DatabaseStore, FeedbackRow, PredictionRow
from anime_mlops.api.schemas import PredictionInput
from anime_mlops.config import Settings, get_settings
from anime_mlops.feedback import ledger, service
from anime_mlops.flows import feedback as feedback_flows
from anime_mlops.flows.daily import daily_ingestion_flow
from anime_mlops.flows.feedback import feedback_flow
from anime_mlops.ingestion import landing
from anime_mlops.pipeline import steps
from anime_mlops.replay.app import load_catalogue
from tests.unit.test_api import MemoryPersistence, SequenceLoader
from tests.unit.test_pipeline import COMMIT, DAY_1, DAY_2, DAY_3, make_snapshot

# Le modèle factice de l'API prédit toujours cette valeur.
PREDICTED = 7.25


@pytest.fixture(scope="module", autouse=True)
def prefect_server():
    with prefect_test_harness():
        yield


@pytest.fixture
def settings(monkeypatch, tmp_path):
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
        "ANIME_ALERTS__FILE": data_dir / "reports" / "alerts.jsonl",
        "ANIME_SOURCES__MAL__COMMIT": COMMIT,
        "ANIME_QUALITY__MIN_ROWS": 20,
        "ANIME_FEEDBACK__API_URL": "http://testserver",
        # Un seul anime par appel : le découpage en lots est exercé dès deux annonces.
        "ANIME_FEEDBACK__BATCH_SIZE": 1,
    }.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def source(settings):
    return load_catalogue(settings)


@pytest.fixture
def store():
    """La persistance de l'API, sur une base SQLite en mémoire."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    database = DatabaseStore(engine, retry_cooldown_seconds=0)
    assert database.initialize()
    yield database
    database.close()


@pytest.fixture
def api(settings, store):
    """L'API de prédiction du projet, avec un modèle factice."""
    application = create_app(settings=settings, loader=SequenceLoader(), persistence=store)
    application.state.model_runtime.reload()
    return TestClient(application)


@pytest.fixture
def write(store):
    def write_feedback(prediction_id, actual_score):
        store.add_feedback(
            prediction_id=prediction_id, actual_score=actual_score, source="automatic"
        )

    return write_feedback


@pytest.fixture
def wired(monkeypatch, api, write):
    """Branche les flows sur l'API et la base de test, sans réseau ni PostgreSQL."""

    @contextmanager
    def writer(_settings):
        yield write

    # Un client neuf à chaque passage : le service ferme celui qu'il a ouvert.
    monkeypatch.setattr(service, "api_client", lambda _config: TestClient(api.app))
    monkeypatch.setattr(service, "database_writer", writer)


def accept(source, settings, *days):
    for day in days:
        landing.store_batch(source.batch(day), settings.paths.accepted_dir)


def rows(store, model):
    with store._sessions() as session:
        return list(session.scalars(select(model)))


def read_alerts(settings) -> list[dict]:
    if not settings.alerts.file.is_file():
        return []
    return [json.loads(line) for line in settings.alerts.file.read_text("utf-8").splitlines()]


# --- Jour J : prédire les annonces ------------------------------------------------


def test_prediction_records_match_the_api_contract(settings, source):
    batch = source.batch(DAY_1)
    catalogue = steps.build_catalogue(source.history(), [batch])
    catalogue.loc[catalogue["id"] == 1002, "num_episodes"] = 0  # « inconnu » chez MAL

    records = service.prediction_records(
        [1001, 1002], catalogue, service._anilist_enrichment(settings)
    )

    assert set(records) == {1001, 1002}
    for record in records.values():
        PredictionInput.model_validate(record)  # refuse tout champ manquant ou en trop
    assert records[1001]["num_episodes"] == 12
    assert records[1001]["start_year"] == 2025
    assert records[1001]["start_season"] == "winter"
    assert records[1001]["start_weekday"] == "wednesday"
    assert records[1001]["genres"] == ["Action", "Drama"]
    assert records[1001]["prequel_score"] is None
    assert records[1002]["num_episodes"] is None


def test_predict_day_traces_predictions_and_keeps_their_ids(settings, source, api, store):
    accept(source, settings, DAY_1)

    day_ledger = service.predict_day(DAY_1, settings, source.history(), api)

    assert [entry.anime_id for entry in day_ledger.predictions] == [1001, 1002]
    assert all(entry.prediction == PREDICTED for entry in day_ledger.predictions)
    assert all(entry.model_version == "1" for entry in day_ledger.predictions)
    traced = {row.id for row in rows(store, PredictionRow)}
    assert {entry.prediction_id for entry in day_ledger.predictions} == traced
    assert ledger.load(DAY_1, settings.paths.predictions_dir) == day_ledger


def test_predict_day_is_idempotent(settings, source, api, store):
    accept(source, settings, DAY_1)
    first = service.predict_day(DAY_1, settings, source.history(), api)

    again = service.predict_day(DAY_1, settings, source.history(), api)

    assert again == first
    assert len(rows(store, PredictionRow)) == 2
    assert service.pending_prediction_days(settings) == []


def test_api_down_leaves_the_day_to_be_predicted_later(settings, source):
    accept(source, settings, DAY_1)

    def refuse(request):
        raise httpx.ConnectError("connexion refusée")

    down = httpx.Client(transport=httpx.MockTransport(refuse))

    with pytest.raises(service.ApiUnavailableError, match="injoignable"):
        service.predict_day(DAY_1, settings, source.history(), down)
    assert service.pending_prediction_days(settings) == [DAY_1]


def test_untraced_predictions_are_not_kept(settings, source):
    """Base indisponible côté API : la prédiction existe mais ne pourra pas être notée."""
    accept(source, settings, DAY_1)
    application = create_app(
        settings=settings, loader=SequenceLoader(), persistence=MemoryPersistence(available=False)
    )
    application.state.model_runtime.reload()

    with pytest.raises(service.ApiUnavailableError, match="tracer"):
        service.predict_day(DAY_1, settings, source.history(), TestClient(application))
    assert service.pending_prediction_days(settings) == [DAY_1]


def test_a_refused_anime_does_not_block_the_others(settings, source, api):
    catalogue = steps.build_catalogue(source.history(), [source.batch(DAY_1)])
    records = service.prediction_records(
        [1001, 1002], catalogue, service._anilist_enrichment(settings)
    )
    invalid = {**records[1002], "start_year": 1500}
    config = settings.feedback.model_copy(update={"batch_size": 2})

    refused, accepted = service.request_predictions([invalid, records[1001]], config, api)

    assert refused is None
    assert accepted.prediction == PREDICTED


# --- Jour J+1 : rattacher les scores ----------------------------------------------


def test_revealed_score_becomes_an_automatic_feedback(settings, source, api, store, write):
    accept(source, settings, DAY_1)
    service.predict_day(DAY_1, settings, source.history(), api)

    before = service.record_feedback(settings, write)
    accept(source, settings, DAY_2)  # révèle le score de 1001 (7.5) ; 1002 n'a pas de score
    after = service.record_feedback(settings, write)

    assert (before.recorded, before.awaiting_score) == (0, 2)
    assert (after.recorded, after.awaiting_score) == (1, 1)
    assert after.mean_absolute_error == pytest.approx(abs(PREDICTED - 7.5))

    [feedback] = rows(store, FeedbackRow)
    tracked = {
        e.anime_id: e for e in ledger.load(DAY_1, settings.paths.predictions_dir).predictions
    }
    assert feedback.prediction_id == tracked[1001].prediction_id
    assert (feedback.actual_score, feedback.source) == (7.5, "automatic")
    assert tracked[1001].actual_score == 7.5
    assert tracked[1001].absolute_error == pytest.approx(0.25)
    assert tracked[1002].is_pending


def test_feedback_is_recorded_only_once(settings, source, api, store, write):
    accept(source, settings, DAY_1, DAY_2)
    service.predict_day(DAY_1, settings, source.history(), api)
    service.record_feedback(settings, write)

    again = service.record_feedback(settings, write)

    assert (again.recorded, again.already_recorded) == (0, 0)
    assert len(rows(store, FeedbackRow)) == 1


def test_interrupted_run_is_resumed_without_duplicate(settings, source, api, store, write):
    """Feedback écrit en base mais pas encore pointé dans le registre (arrêt brutal)."""
    accept(source, settings, DAY_1, DAY_2)
    day_ledger = service.predict_day(DAY_1, settings, source.history(), api)
    write(day_ledger.predictions[0].prediction_id, 7.5)

    summary = service.record_feedback(settings, write)

    assert (summary.recorded, summary.already_recorded) == (0, 1)
    assert len(rows(store, FeedbackRow)) == 1
    assert not ledger.load(DAY_1, settings.paths.predictions_dir).predictions[0].is_pending


def test_unknown_prediction_is_reported_and_stays_pending(settings, source, write):
    accept(source, settings, DAY_1, DAY_2)
    orphan = ledger.TrackedPrediction(
        anime_id=1001, prediction_id=uuid4(), prediction=PREDICTED, model_version="1"
    )
    ledger.save(ledger.DayLedger(day=DAY_1, predictions=[orphan]), settings.paths.predictions_dir)

    summary = service.record_feedback(settings, write)

    assert (summary.recorded, summary.missing_predictions) == (0, 1)
    assert ledger.load(DAY_1, settings.paths.predictions_dir).predictions[0].is_pending


# --- Flows ------------------------------------------------------------------------


def test_feedback_flow_predicts_then_scores(settings, source, store, wired):
    accept(source, settings, DAY_1)
    first = feedback_flow()
    accept(source, settings, DAY_2)
    second = feedback_flow()

    assert (first["predicted_days"], first["predictions"], first["feedbacks"]) == (1, 2, 0)
    assert (second["predicted_days"], second["predictions"], second["feedbacks"]) == (1, 1, 1)
    assert second["mean_absolute_error"] == pytest.approx(0.25)
    assert len(rows(store, PredictionRow)) == 3
    assert len(rows(store, FeedbackRow)) == 1
    assert read_alerts(settings) == []


def test_daily_flow_runs_the_feedback_after_accepting_a_batch(settings, source, store, wired):
    for day in (DAY_1, DAY_2, DAY_3):
        landing.store_batch(source.batch(day), settings.paths.landing_dir)

    daily_ingestion_flow()
    result = daily_ingestion_flow()

    assert result["status"] == "accepted"
    assert result["feedback"]["feedbacks"] == 1
    assert [row.source for row in rows(store, FeedbackRow)] == ["automatic"]


def test_daily_flow_keeps_the_batch_when_the_api_is_down(settings, source, monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connexion refusée")

    monkeypatch.setattr(
        service, "api_client", lambda _c: httpx.Client(transport=httpx.MockTransport(refuse))
    )
    monkeypatch.setattr(feedback_flows.predict_announcements, "retry_delay_seconds", 0)
    landing.store_batch(source.batch(DAY_1), settings.paths.landing_dir)

    result = daily_ingestion_flow()

    # Le lot est accepté ; le feedback, lui, a échoué en émettant une alerte.
    assert (result["status"], result["feedback"]) == ("accepted", None)
    assert landing.landed_days(settings.paths.accepted_dir) == [DAY_1]
    [alert] = read_alerts(settings)
    assert alert["level"] == "critical"
    assert "feedback-automatique" in alert["title"]
    assert service.pending_prediction_days(settings) == [DAY_1]
