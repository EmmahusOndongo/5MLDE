"""Point de réception des lots en mode push.

Lancement : ``uvicorn anime_mlops.ingestion.receiver:app --port 8101``

La source appelle ``POST /batches`` ; le lot est vérifié (structure), puis déposé dans la
zone de dépôt, où le pipeline le reprendra. Les contrôles de contenu, bloquants, sont
faits par le pipeline : la réception ne fait qu'accuser réception et tracer.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status

from anime_mlops.config import get_settings
from anime_mlops.ingestion.landing import StoreStatus, landed_days, store_batch
from anime_mlops.ingestion.schemas import DailyBatch


def create_app(landing_dir: Path) -> FastAPI:
    app = FastAPI(
        title="Réception des lots journaliers",
        description="Reçoit les lots poussés par la source et les dépose pour le pipeline.",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "landing_dir": str(landing_dir),
            "batches": len(landed_days(landing_dir)),
        }

    @app.get("/batches", summary="Jours déjà reçus")
    def list_batches() -> dict:
        return {"days": landed_days(landing_dir)}

    @app.post("/batches", status_code=status.HTTP_201_CREATED, summary="Reçoit un lot (mode push)")
    def receive_batch(batch: DailyBatch, response: Response) -> dict:
        result = store_batch(batch, landing_dir)
        if result.status is StoreStatus.CONFLICT:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Un lot différent a déjà été reçu pour le {batch.day}.",
            )
        if result.status is StoreStatus.UNCHANGED:
            # Renvoi du même lot (nouvelle tentative de la source) : accepté sans effet.
            response.status_code = status.HTTP_200_OK
        return {
            "day": batch.day,
            "status": result.status.value,
            "sha256": result.sha256,
            "announced": len(batch.announced),
            "scores": len(batch.scores),
        }

    return app


def __getattr__(name: str):
    # ``app`` n'est construite qu'à la demande d'uvicorn (voir replay/app.py).
    if name == "app":
        return create_app(get_settings().paths.landing_dir)
    raise AttributeError(name)
