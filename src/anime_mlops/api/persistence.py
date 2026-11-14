"""Persistance résiliente des prédictions et des retours utilisateurs."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock, RLock
from time import monotonic
from typing import Literal, Protocol
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class DatabaseUnavailableError(RuntimeError):
    """La base ne peut pas accepter l'opération demandée."""


class PredictionNotFoundError(RuntimeError):
    """Le feedback désigne une prédiction inconnue."""


class FeedbackConflictError(RuntimeError):
    """Un feedback de même source existe déjà pour cette prédiction."""


@dataclass(frozen=True)
class DatabaseStatus:
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class PredictionTrace:
    id: UUID
    endpoint: str
    input_data: dict[str, object]
    duration_ms: float
    prediction: float | None = None
    model_name: str | None = None
    model_alias: str | None = None
    model_version: str | None = None
    model_run_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    batch_id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class FeedbackRecord:
    id: UUID
    prediction_id: UUID
    actual_score: float
    source: Literal["manual", "automatic"]
    comment: str | None
    created_at: datetime


class Persistence(Protocol):
    def initialize(self) -> bool: ...

    def status(self) -> DatabaseStatus: ...

    def record_predictions(self, traces: list[PredictionTrace]) -> bool: ...

    def add_feedback(
        self,
        *,
        prediction_id: UUID,
        actual_score: float,
        source: Literal["manual", "automatic"],
        comment: str | None = None,
    ) -> FeedbackRecord: ...

    def close(self) -> None: ...


class Base(DeclarativeBase):
    pass


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    endpoint: Mapped[str] = mapped_column(String(50))
    batch_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), index=True)
    input_data: Mapped[dict] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    prediction: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str | None] = mapped_column(String(250))
    model_alias: Mapped[str | None] = mapped_column(String(100))
    model_version: Mapped[str | None] = mapped_column(String(100), index=True)
    model_run_id: Mapped[str | None] = mapped_column(String(250))
    duration_ms: Mapped[float] = mapped_column(Float)
    outcome: Mapped[str] = mapped_column(String(20), index=True)
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)


class FeedbackRow(Base):
    __tablename__ = "feedbacks"
    __table_args__ = (
        UniqueConstraint("prediction_id", "source", name="uq_feedback_prediction_source"),
        CheckConstraint("actual_score >= 0 AND actual_score <= 10", name="ck_feedback_score"),
        CheckConstraint("source IN ('manual', 'automatic')", name="ck_feedback_source"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    prediction_id: Mapped[UUID] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), index=True
    )
    actual_score: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), index=True)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class DatabaseStore:
    """Accès SQLAlchemy avec un court circuit après une panne de PostgreSQL."""

    def __init__(self, engine: Engine, *, retry_cooldown_seconds: float = 5) -> None:
        self._engine = engine
        self._sessions = sessionmaker(bind=engine, expire_on_commit=False)
        self._retry_cooldown_seconds = retry_cooldown_seconds
        self._state_lock = RLock()
        self._recovery_lock = Lock()
        self._available = False
        self._last_error: str | None = "Base de données non initialisée."
        self._retry_after = 0.0

    @classmethod
    def from_url(
        cls,
        database_url: str,
        *,
        connect_timeout_seconds: int = 2,
        retry_cooldown_seconds: float = 5,
    ) -> "DatabaseStore":
        connect_args = {}
        if database_url.startswith("postgresql"):
            connect_args["connect_timeout"] = connect_timeout_seconds
        engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        return cls(engine, retry_cooldown_seconds=retry_cooldown_seconds)

    def initialize(self) -> bool:
        return self._prepare_operation()

    def status(self) -> DatabaseStatus:
        if not self._prepare_operation():
            with self._state_lock:
                return DatabaseStatus(available=False, reason=self._last_error)
        try:
            self._ping()
        except SQLAlchemyError as exc:
            self._mark_failure(exc)
            with self._state_lock:
                return DatabaseStatus(available=False, reason=self._last_error)
        self._mark_success()
        return DatabaseStatus(available=True)

    def record_predictions(self, traces: list[PredictionTrace]) -> bool:
        if not traces:
            return True
        if not self._prepare_operation():
            return False

        rows = [
            PredictionRow(
                id=trace.id,
                created_at=trace.created_at,
                endpoint=trace.endpoint,
                batch_id=trace.batch_id,
                input_data=trace.input_data,
                prediction=trace.prediction,
                model_name=trace.model_name,
                model_alias=trace.model_alias,
                model_version=trace.model_version,
                model_run_id=trace.model_run_id,
                duration_ms=trace.duration_ms,
                outcome="error" if trace.error_code else "success",
                error_code=trace.error_code,
                error_message=trace.error_message,
            )
            for trace in traces
        ]
        try:
            with self._sessions.begin() as session:
                session.add_all(rows)
        except SQLAlchemyError as exc:
            self._mark_failure(exc)
            return False
        self._mark_success()
        return True

    def add_feedback(
        self,
        *,
        prediction_id: UUID,
        actual_score: float,
        source: Literal["manual", "automatic"],
        comment: str | None = None,
    ) -> FeedbackRecord:
        if not 0 <= actual_score <= 10:
            raise ValueError("Le score réel doit être compris entre 0 et 10.")
        if source not in {"manual", "automatic"}:
            raise ValueError("La source du feedback doit être manual ou automatic.")
        if not self._prepare_operation():
            raise DatabaseUnavailableError(self._current_error())

        created_at = datetime.now(UTC)
        feedback_id = uuid4()
        try:
            with self._sessions.begin() as session:
                prediction_exists = session.scalar(
                    select(PredictionRow.id).where(PredictionRow.id == prediction_id)
                )
                if prediction_exists is None:
                    raise PredictionNotFoundError(f"Prédiction inconnue : {prediction_id}")
                session.add(
                    FeedbackRow(
                        id=feedback_id,
                        prediction_id=prediction_id,
                        actual_score=actual_score,
                        source=source,
                        comment=comment,
                        created_at=created_at,
                    )
                )
        except PredictionNotFoundError:
            raise
        except IntegrityError as exc:
            raise FeedbackConflictError(
                f"Un feedback {source} existe déjà pour la prédiction {prediction_id}."
            ) from exc
        except SQLAlchemyError as exc:
            self._mark_failure(exc)
            raise DatabaseUnavailableError(self._current_error()) from exc

        self._mark_success()
        return FeedbackRecord(
            id=feedback_id,
            prediction_id=prediction_id,
            actual_score=actual_score,
            source=source,
            comment=comment,
            created_at=created_at,
        )

    def close(self) -> None:
        self._engine.dispose()

    def _ping(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def _prepare_operation(self) -> bool:
        with self._state_lock:
            if self._available:
                return True
            if monotonic() < self._retry_after:
                return False
        with self._recovery_lock:
            with self._state_lock:
                if self._available:
                    return True
                if monotonic() < self._retry_after:
                    return False
            try:
                Base.metadata.create_all(self._engine)
                self._ping()
            except SQLAlchemyError as exc:
                self._mark_failure(exc)
                return False
            self._mark_success()
            return True

    def _mark_success(self) -> None:
        with self._state_lock:
            self._available = True
            self._last_error = None
            self._retry_after = 0.0

    def _mark_failure(self, exc: SQLAlchemyError) -> None:
        with self._state_lock:
            self._available = False
            self._last_error = f"Base de données indisponible ({type(exc).__name__})."
            self._retry_after = monotonic() + self._retry_cooldown_seconds

    def _current_error(self) -> str:
        with self._state_lock:
            return self._last_error or "Base de données indisponible."


def save_feedback(
    database_url: str,
    *,
    prediction_id: UUID,
    actual_score: float,
    source: Literal["manual", "automatic"] = "automatic",
    comment: str | None = None,
    connect_timeout_seconds: int = 2,
) -> FeedbackRecord:
    """Fonction stable appelée par le flow A9 pour enregistrer le score observé."""
    if not 0 <= actual_score <= 10:
        raise ValueError("Le score réel doit être compris entre 0 et 10.")
    store = DatabaseStore.from_url(
        database_url,
        connect_timeout_seconds=connect_timeout_seconds,
        retry_cooldown_seconds=0,
    )
    try:
        if not store.initialize():
            raise DatabaseUnavailableError(store.status().reason or "Base indisponible.")
        return store.add_feedback(
            prediction_id=prediction_id,
            actual_score=actual_score,
            source=source,
            comment=comment,
        )
    finally:
        store.close()
