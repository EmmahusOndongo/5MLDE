"""Point d'entrée de l'API pour les conteneurs et la production locale."""

import uvicorn

from anime_mlops.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "anime_mlops.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        access_log=True,
    )


if __name__ == "__main__":
    main()
