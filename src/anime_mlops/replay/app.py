"""Source de données simulée : rejoue le catalogue MAL jour par jour.

Lancement : ``uvicorn anime_mlops.replay.app:app --port 8100``

Deux modes de consommation :

- **pull** : le pipeline vient chercher le lot du jour (``GET /days/{day}``) ;
- **push** : la source envoie elle-même le lot au point de réception du pipeline
  (``POST /days/{day}/push``), comme le ferait un webhook.
"""

import logging
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from anime_mlops.config import Settings, get_settings
from anime_mlops.ingestion.schemas import DailyBatch, DaySummary
from anime_mlops.replay.batches import DayOutOfRangeError, ReplayCatalogue

logger = logging.getLogger(__name__)

_PUSH_TIMEOUT_SECONDS = 10.0


def create_app(
    catalogue: ReplayCatalogue,
    push_target_url: str,
    http_client: httpx.Client | None = None,
) -> FastAPI:
    """Construit l'application autour d'un catalogue (injectable pour les tests)."""
    app = FastAPI(
        title="Source simulée — rejeu journalier du catalogue MAL",
        description="Simule l'arrivée quotidienne de données à partir d'un instantané figé.",
    )

    def get_batch(day: date, corrupt: bool) -> DailyBatch:
        try:
            return catalogue.batch(day, corrupt=corrupt)
        except DayOutOfRangeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "source_commit": catalogue.source_commit,
            "start_day": catalogue.start_day,
            "end_day": catalogue.end_day,
            "push_target_url": push_target_url,
        }

    @app.get("/days", summary="Jours disponibles et volumes")
    def list_days(
        first: date | None = Query(default=None, description="Premier jour (inclus)"),
        last: date | None = Query(default=None, description="Dernier jour (inclus)"),
    ) -> list[DaySummary]:
        return catalogue.days(first, last)

    @app.get("/days/{day}", summary="Lot du jour (mode pull)")
    def pull_day(
        day: date,
        corrupt: bool = Query(
            default=False, description="Livre un lot invalide (incident contrôlé)"
        ),
    ) -> DailyBatch:
        return get_batch(day, corrupt)

    @app.post("/days/{day}/push", summary="Envoie le lot du jour au pipeline (mode push)")
    def push_day(
        day: date,
        corrupt: bool = Query(
            default=False, description="Envoie un lot invalide (incident contrôlé)"
        ),
    ) -> dict:
        batch = get_batch(day, corrupt)
        payload = batch.model_dump(mode="json")
        try:
            if http_client is not None:
                response = http_client.post(push_target_url, json=payload)
            else:
                response = httpx.post(push_target_url, json=payload, timeout=_PUSH_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            logger.error("Envoi du lot %s impossible vers %s : %s", day, push_target_url, exc)
            raise HTTPException(
                status_code=502, detail=f"Point de réception injoignable ({push_target_url})."
            ) from exc

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise HTTPException(
                status_code=502,
                detail=f"Lot refusé par le point de réception (HTTP {response.status_code}).",
            )
        return {
            "day": day,
            "delivered_to": push_target_url,
            "receiver_status": response.status_code,
            "announced": len(batch.announced),
            "scores": len(batch.scores),
        }

    return app


def load_catalogue(settings: Settings) -> ReplayCatalogue:
    """Charge l'instantané MAL à rejouer : commit configuré, sinon le plus récent sur disque."""
    directory = settings.paths.raw_dir / "mal"
    commit = settings.sources.mal.commit
    if commit:
        candidates = [directory / f"anime-standalone_{commit}.csv"]
    else:
        candidates = sorted(directory.glob("anime-standalone_*.csv"), key=_modified_at)
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"Aucun instantané MAL dans {directory}. Lancer d'abord l'ingestion : "
            "python -m anime_mlops.data.ingest"
        )
    path = existing[-1]
    source_commit = path.stem.removeprefix("anime-standalone_")
    return ReplayCatalogue(pd.read_csv(path), source_commit, settings.replay.start_day)


def _modified_at(path: Path) -> float:
    return path.stat().st_mtime


def build_default_app() -> FastAPI:
    settings = get_settings()
    return create_app(load_catalogue(settings), settings.replay.push_target_url)


def __getattr__(name: str):
    # ``app`` n'est construite qu'à la demande d'uvicorn : importer ce module
    # (tests, outils) ne lit donc aucun fichier de données.
    if name == "app":
        return build_default_app()
    raise AttributeError(name)
