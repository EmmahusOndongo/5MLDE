"""Point d'entrée de l'API pour les conteneurs et la production locale."""

import os
from pathlib import Path

import uvicorn

from anime_mlops.api.metrics import PROMETHEUS_MULTIPROC_DIR_ENV
from anime_mlops.config import get_settings


def main() -> None:
    settings = get_settings()
    _prepare_prometheus_multiprocess_directory()
    uvicorn.run(
        "anime_mlops.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        access_log=True,
    )


def _prepare_prometheus_multiprocess_directory() -> None:
    """Supprime les anciennes métriques mmap avant de lancer les workers."""
    configured_directory = os.environ.get(PROMETHEUS_MULTIPROC_DIR_ENV)
    if not configured_directory:
        return
    directory = Path(configured_directory)
    directory.mkdir(parents=True, exist_ok=True)
    for metrics_file in directory.glob("*.db"):
        metrics_file.unlink()


if __name__ == "__main__":
    main()
