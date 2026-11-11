"""Ingestion des deux sources. Usage : ``python -m anime_mlops.data.ingest``."""

import logging
from dataclasses import dataclass

from anime_mlops.config import Settings, get_settings
from anime_mlops.data.sources import anilist, mal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    mal: mal.MalSnapshot
    anilist: anilist.AnilistExtract


def ingest(settings: Settings, mal_commit: str | None = None) -> IngestionResult:
    """Récupère MAL (commit figé) et AniList (extraction datée), versions tracées."""
    commit = mal_commit or settings.sources.mal.commit
    if commit is None:
        commit = mal.resolve_latest_commit(settings.sources.mal)
        logger.info("Aucun commit MAL configuré : dernier commit retenu (%s).", commit)

    snapshot = mal.download_snapshot(commit, settings.sources.mal, settings.paths.raw_dir)
    extract = anilist.get_extract(settings.sources.anilist, settings.paths.cache_dir)
    return IngestionResult(mal=snapshot, anilist=extract)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = ingest(get_settings())
    logger.info(
        "MAL : commit=%s sha256=%s fichier=%s",
        result.mal.commit,
        result.mal.sha256[:12],
        result.mal.path,
    )
    logger.info(
        "AniList : date=%s animes=%d repli=%s fichier=%s",
        result.anilist.extraction_date,
        len(result.anilist.data),
        result.anilist.is_fallback,
        result.anilist.path,
    )


if __name__ == "__main__":
    main()
