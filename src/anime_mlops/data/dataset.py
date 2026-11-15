"""Préparation du jeu d'entraînement : ingestion, contrôles bloquants, jointure.

Point d'entrée unique, réutilisé par la commande d'entraînement et par l'orchestrateur.
"""

from dataclasses import dataclass

import pandas as pd

from anime_mlops.config import Settings
from anime_mlops.data import validate
from anime_mlops.data.ingest import ingest
from anime_mlops.data.merge import merge_sources, training_rows
from anime_mlops.data.sources import mal
from anime_mlops.versioning import file_sha256


@dataclass(frozen=True)
class DataVersions:
    """Tout ce qu'il faut pour retrouver exactement les données d'un entraînement."""

    mal_commit: str
    mal_sha256: str
    anilist_extraction_date: str
    anilist_sha256: str
    # True si AniList était indisponible et qu'un cache antérieur a servi.
    anilist_is_fallback: bool
    # Rejeu journalier : dernier jour pris en compte et nombre de lots acceptés intégrés.
    replay_as_of: str | None = None
    replay_batches: int = 0

    @property
    def label(self) -> str:
        return f"{self.mal_commit[:12]}_{self.anilist_extraction_date}"


@dataclass(frozen=True)
class PreparedData:
    training: pd.DataFrame
    versions: DataVersions
    reports: tuple[validate.ValidationReport, ...]


def prepare_training_data(settings: Settings, mal_commit: str | None = None) -> PreparedData:
    """Renvoie le jeu d'entraînement validé.

    Lève ``DataQualityError`` au premier lot non conforme : rien n'est entraîné ensuite.
    """
    ingestion = ingest(settings, mal_commit)
    quality, reports_dir = settings.quality, settings.paths.reports_dir
    versions = DataVersions(
        mal_commit=ingestion.mal.commit,
        mal_sha256=ingestion.mal.sha256,
        anilist_extraction_date=ingestion.anilist.extraction_date.isoformat(),
        anilist_sha256=file_sha256(ingestion.anilist.path),
        anilist_is_fallback=ingestion.anilist.is_fallback,
    )

    mal_data = mal.load_snapshot(ingestion.mal)
    mal_report = validate.validate_mal(mal_data, versions.mal_commit, quality, reports_dir)
    anilist_report = validate.validate_anilist(
        ingestion.anilist.data, versions.anilist_extraction_date, quality, reports_dir
    )

    training = training_rows(merge_sources(mal_data, ingestion.anilist.data))
    merged_report = validate.validate_merged(training, versions.label, quality, reports_dir)

    return PreparedData(training, versions, (mal_report, anilist_report, merged_report))
