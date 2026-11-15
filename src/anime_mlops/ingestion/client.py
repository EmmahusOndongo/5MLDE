"""Consommation de la source en mode pull : le pipeline va chercher le lot du jour."""

import logging
from datetime import date
from pathlib import Path

import httpx
from pydantic import ValidationError

from anime_mlops.ingestion.landing import StoreResult, StoreStatus, store_batch
from anime_mlops.ingestion.schemas import DailyBatch

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30.0


class SourceUnavailableError(RuntimeError):
    """La source ne répond pas, ou pas correctement."""


class BatchConflictError(RuntimeError):
    """Un lot différent a déjà été déposé pour ce jour."""


def pull_batch(
    day: date, source_url: str, client: httpx.Client | None = None, corrupt: bool = False
) -> DailyBatch:
    """Récupère le lot d'un jour auprès de la source et vérifie sa structure."""
    url = f"{source_url.rstrip('/')}/days/{day.isoformat()}"
    params = {"corrupt": "true"} if corrupt else None
    try:
        if client is not None:
            response = client.get(url, params=params)
        else:
            response = httpx.get(url, params=params, timeout=_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise SourceUnavailableError(f"Source injoignable ({url}) : {exc}") from exc
    # Code vérifié à la main : l'exception de ``raise_for_status`` dépend du client fourni.
    if response.status_code >= 400:
        raise SourceUnavailableError(
            f"La source a refusé le lot du {day} (HTTP {response.status_code})."
        )
    try:
        return DailyBatch.model_validate(response.json())
    except (ValidationError, ValueError) as exc:
        raise SourceUnavailableError(f"Lot du {day} au format inattendu : {exc}") from exc


def pull_and_store(
    day: date,
    source_url: str,
    landing_dir: Path,
    client: httpx.Client | None = None,
    corrupt: bool = False,
) -> StoreResult:
    """Pull puis dépôt dans la zone de dépôt (même traitement qu'un lot reçu en push)."""
    batch = pull_batch(day, source_url, client, corrupt)
    result = store_batch(batch, landing_dir)
    if result.status is StoreStatus.CONFLICT:
        raise BatchConflictError(f"Un lot différent existe déjà pour le {day} : {result.path}")
    return result
