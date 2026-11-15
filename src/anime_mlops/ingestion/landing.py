"""Zone de dépôt : les lots journaliers reçus, tels quels, avant tout traitement.

Un lot y est écrit une seule fois. Le recevoir à nouveau à l'identique ne change rien
(idempotence) ; recevoir un contenu différent pour le même jour est un conflit, signalé
plutôt qu'écrasé en silence.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

from anime_mlops.ingestion.schemas import DailyBatch

logger = logging.getLogger(__name__)

_PREFIX = "batch_"


class StoreStatus(StrEnum):
    CREATED = "created"
    UNCHANGED = "unchanged"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class StoreResult:
    status: StoreStatus
    path: Path
    sha256: str


def store_batch(batch: DailyBatch, landing_dir: Path) -> StoreResult:
    """Dépose le lot. N'écrase jamais un lot existant."""
    content = _serialize(batch)
    digest = hashlib.sha256(content).hexdigest()
    path = _path(landing_dir, batch.day)

    if path.is_file():
        existing = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing == digest:
            return StoreResult(StoreStatus.UNCHANGED, path, digest)
        logger.error("Lot du %s déjà déposé avec un contenu différent.", batch.day)
        return StoreResult(StoreStatus.CONFLICT, path, existing)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(content)
    temporary.replace(path)
    logger.info(
        "Lot du %s déposé : %d annonce(s), %d score(s).",
        batch.day,
        len(batch.announced),
        len(batch.scores),
    )
    return StoreResult(StoreStatus.CREATED, path, digest)


def load_batch(day: date, landing_dir: Path) -> DailyBatch:
    path = _path(landing_dir, day)
    if not path.is_file():
        raise FileNotFoundError(f"Aucun lot déposé pour le {day} dans {landing_dir}.")
    return DailyBatch.model_validate_json(path.read_bytes())


def landed_days(landing_dir: Path) -> list[date]:
    """Jours pour lesquels un lot a été déposé, du plus ancien au plus récent."""
    days = []
    for path in landing_dir.glob(f"{_PREFIX}*.json"):
        try:
            days.append(date.fromisoformat(path.stem.removeprefix(_PREFIX)))
        except ValueError:
            logger.warning("Fichier ignoré dans la zone de dépôt (nom inattendu) : %s", path.name)
    return sorted(days)


def _path(landing_dir: Path, day: date) -> Path:
    return landing_dir / f"{_PREFIX}{day.isoformat()}.json"


def _serialize(batch: DailyBatch) -> bytes:
    """Forme canonique : le même lot donne toujours les mêmes octets, donc la même empreinte."""
    payload = batch.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=1).encode("utf-8")
