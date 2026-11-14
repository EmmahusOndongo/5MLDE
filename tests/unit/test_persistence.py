from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, create_mock_engine, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import StaticPool

from anime_mlops.api import save_feedback
from anime_mlops.api.persistence import (
    Base,
    DatabaseStore,
    FeedbackConflictError,
    FeedbackRow,
    PredictionNotFoundError,
    PredictionRow,
    PredictionTrace,
)


@pytest.fixture
def sqlite_store():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    store = DatabaseStore(engine, retry_cooldown_seconds=60)
    assert store.initialize()
    yield store, engine
    store.close()


def prediction_trace(*, error_code: str | None = None) -> PredictionTrace:
    return PredictionTrace(
        id=uuid4(),
        endpoint="/predict",
        input_data={"media_type": "tv"},
        prediction=None if error_code else 7.25,
        model_name="anime-score-regressor",
        model_alias="champion",
        model_version="3",
        model_run_id="run-3",
        duration_ms=12.5,
        error_code=error_code,
        error_message="échec simulé" if error_code else None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_prediction_and_feedback_are_persisted(sqlite_store):
    store, engine = sqlite_store
    trace = prediction_trace()

    assert store.record_predictions([trace])
    feedback = store.add_feedback(
        prediction_id=trace.id,
        actual_score=8.5,
        source="manual",
        comment="score observé",
    )

    assert feedback.prediction_id == trace.id
    assert feedback.actual_score == 8.5
    with engine.connect() as connection:
        prediction_count = connection.scalar(select(func.count()).select_from(PredictionRow))
        feedback_count = connection.scalar(select(func.count()).select_from(FeedbackRow))
        outcome = connection.scalar(
            select(PredictionRow.outcome).where(PredictionRow.id == trace.id)
        )
    assert prediction_count == 1
    assert feedback_count == 1
    assert outcome == "success"


def test_duplicate_and_orphan_feedbacks_are_rejected(sqlite_store):
    store, _engine = sqlite_store
    trace = prediction_trace()
    store.record_predictions([trace])
    store.add_feedback(prediction_id=trace.id, actual_score=7.0, source="automatic")

    with pytest.raises(FeedbackConflictError):
        store.add_feedback(prediction_id=trace.id, actual_score=7.1, source="automatic")
    with pytest.raises(PredictionNotFoundError):
        store.add_feedback(prediction_id=uuid4(), actual_score=7.0, source="manual")


def test_error_trace_is_kept_for_operational_analysis(sqlite_store):
    store, engine = sqlite_store
    trace = prediction_trace(error_code="prediction_failed")

    assert store.record_predictions([trace])

    with engine.connect() as connection:
        row = connection.execute(
            select(PredictionRow.outcome, PredictionRow.error_code).where(
                PredictionRow.id == trace.id
            )
        ).one()
    assert row.outcome == "error"
    assert row.error_code == "prediction_failed"


def test_database_failure_opens_the_circuit(sqlite_store, monkeypatch):
    store, _engine = sqlite_store
    original_ping = store._ping

    def fail_ping() -> None:
        raise OperationalError("SELECT 1", {}, RuntimeError("panne simulée"))

    monkeypatch.setattr(store, "_ping", fail_ping)

    assert store.status().available is False
    assert store.record_predictions([prediction_trace()]) is False
    assert store.status().reason is not None

    monkeypatch.setattr(store, "_ping", original_ping)
    store._retry_after = 0
    assert store.status().available is True
    assert store.record_predictions([prediction_trace()]) is True


def test_save_feedback_is_the_stable_contract_for_the_training_flow(tmp_path):
    database_path = (tmp_path / "anime.db").as_posix()
    database_url = f"sqlite+pysqlite:///{database_path}"
    store = DatabaseStore.from_url(database_url, retry_cooldown_seconds=0)
    trace = prediction_trace()
    assert store.initialize()
    assert store.record_predictions([trace])
    store.close()

    feedback = save_feedback(
        database_url,
        prediction_id=trace.id,
        actual_score=8.75,
    )

    assert feedback.prediction_id == trace.id
    assert feedback.source == "automatic"


def test_schema_compiles_for_postgresql():
    statements = []

    def capture(sql, *_multiparams, **_params) -> None:
        statements.append(str(sql.compile(dialect=postgresql.dialect())))

    engine = create_mock_engine("postgresql+psycopg://", capture)
    Base.metadata.create_all(engine)
    ddl = "\n".join(statements)

    assert "CREATE TABLE predictions" in ddl
    assert "CREATE TABLE feedbacks" in ddl
    assert "JSONB" in ddl
