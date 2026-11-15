"""Registre des prédictions faites à l'annonce, un fichier par jour.

L'API ne connaît pas l'identifiant MAL d'un anime : elle reçoit des features et renvoie
un ``prediction_id``. Ce registre garde le lien « anime → prédiction », indispensable
pour rattacher le score réel à la bonne prédiction quand il est révélé.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

_PREFIX = "predictions_"


class TrackedPrediction(BaseModel):
    """Une prédiction faite à l'annonce, et son score réel une fois connu."""

    model_config = ConfigDict(extra="forbid")

    anime_id: int
    prediction_id: UUID
    prediction: float
    model_version: str
    # Renseignés quand le score réel a été enregistré comme feedback.
    actual_score: float | None = None
    feedback_at: datetime | None = None

    @property
    def is_pending(self) -> bool:
        return self.feedback_at is None

    @property
    def absolute_error(self) -> float | None:
        if self.actual_score is None:
            return None
        return abs(self.prediction - self.actual_score)


class DayLedger(BaseModel):
    """Les prédictions des animes annoncés un jour donné."""

    model_config = ConfigDict(extra="forbid")

    day: date
    predictions: list[TrackedPrediction]


def save(ledger: DayLedger, directory: Path) -> Path:
    """Écrit le registre du jour (écriture atomique : jamais de fichier à moitié écrit)."""
    path = _path(directory, ledger.day)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load(day: date, directory: Path) -> DayLedger | None:
    path = _path(directory, day)
    if not path.is_file():
        return None
    return DayLedger.model_validate_json(path.read_bytes())


def ledger_days(directory: Path) -> list[date]:
    """Jours disposant d'un registre, du plus ancien au plus récent."""
    days = []
    for path in directory.glob(f"{_PREFIX}*.json"):
        try:
            days.append(date.fromisoformat(path.stem.removeprefix(_PREFIX)))
        except ValueError:
            logger.warning("Fichier ignoré dans le registre (nom inattendu) : %s", path.name)
    return sorted(days)


def load_all(directory: Path) -> list[DayLedger]:
    return [ledger for day in ledger_days(directory) if (ledger := load(day, directory))]


def _path(directory: Path, day: date) -> Path:
    return directory / f"{_PREFIX}{day.isoformat()}.json"
