"""Étapes du pipeline, sans dépendance à l'orchestrateur (donc testables seules).

Parcours d'un lot journalier :

    source ──pull──┐
                   ├──► zone de dépôt ──► contrôles qualité ──┬─► lots acceptés ──► entraînement
    source ──push──┘                                          └─► quarantaine (+ alerte)
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import pandas as pd

from anime_mlops.config import Settings
from anime_mlops.data import validate
from anime_mlops.ingestion import landing
from anime_mlops.ingestion.client import pull_and_store
from anime_mlops.ingestion.schemas import DailyBatch

logger = logging.getLogger(__name__)

ONE_DAY = timedelta(days=1)


@dataclass(frozen=True)
class BatchCheck:
    """Résultat des contrôles d'un lot."""

    batch: DailyBatch
    reports: tuple[validate.ValidationReport, ...]
    # Scores reçus pour des animes jamais annoncés : ignorés, mais signalés.
    orphan_scores: int

    @property
    def success(self) -> bool:
        return all(report.success for report in self.reports)


def next_day(settings: Settings) -> date:
    """Prochain jour à traiter : le lendemain du dernier lot traité (accepté ou rejeté)."""
    processed = [
        *landing.landed_days(settings.paths.accepted_dir),
        *landing.landed_days(settings.paths.quarantine_dir),
    ]
    return max(processed) + ONE_DAY if processed else settings.replay.start_day


def is_processed(day: date, settings: Settings) -> bool:
    return day in landing.landed_days(settings.paths.accepted_dir)


def obtain_batch(
    day: date, settings: Settings, client: httpx.Client | None = None, corrupt: bool = False
) -> DailyBatch:
    """Lot du jour : déjà reçu en push s'il est dans la zone de dépôt, sinon récupéré en pull."""
    if day in landing.landed_days(settings.paths.landing_dir):
        logger.info("Lot du %s déjà reçu (push) : pas de pull.", day)
    else:
        pull_and_store(day, settings.replay.source_url, settings.paths.landing_dir, client, corrupt)
    return landing.load_batch(day, settings.paths.landing_dir)


def accepted_batches(settings: Settings, as_of: date | None = None) -> list[DailyBatch]:
    """Lots acceptés, dans l'ordre chronologique, jusqu'à ``as_of`` inclus."""
    days = landing.landed_days(settings.paths.accepted_dir)
    return [
        landing.load_batch(day, settings.paths.accepted_dir)
        for day in days
        if as_of is None or day <= as_of
    ]


def announced_ids(batches: list[DailyBatch]) -> set[int]:
    return {anime.id for batch in batches for anime in batch.announced}


def announcements_frame(batch: DailyBatch, known_ids: set[int]) -> pd.DataFrame:
    """Annonces du lot sous forme tabulaire, avec l'indicateur « déjà connu »."""
    frame = pd.DataFrame(
        [anime.model_dump() for anime in batch.announced],
        columns=[c for c in validate.BATCH_ANNOUNCED_COLUMNS if c != "already_known"],
    )
    frame["already_known"] = frame["id"].isin(known_ids)
    return frame


def scores_frame(batch: DailyBatch) -> pd.DataFrame:
    return pd.DataFrame(
        [score.model_dump() for score in batch.scores], columns=validate.BATCH_SCORE_COLUMNS
    )


def check_announcements(
    batch: DailyBatch, known_ids: set[int], settings: Settings
) -> validate.ValidationReport:
    return validate.validate(
        announcements_frame(batch, known_ids),
        validate.announcements_suite(settings.quality),
        batch.day.isoformat(),
        settings.paths.reports_dir,
    )


def check_scores(batch: DailyBatch, settings: Settings) -> validate.ValidationReport:
    return validate.validate(
        scores_frame(batch),
        validate.scores_suite(settings.quality),
        batch.day.isoformat(),
        settings.paths.reports_dir,
    )


def count_orphan_scores(batch: DailyBatch, previously_announced: set[int]) -> int:
    return sum(1 for score in batch.scores if score.id not in previously_announced)


def route_batch(check: BatchCheck, settings: Settings) -> landing.StoreResult:
    """Range le lot selon le verdict : accepté (alimente l'entraînement) ou en quarantaine."""
    target = settings.paths.accepted_dir if check.success else settings.paths.quarantine_dir
    return landing.store_batch(check.batch, target)


def build_catalogue(history: pd.DataFrame, batches: list[DailyBatch]) -> pd.DataFrame:
    """Catalogue connu à ce jour : l'historique, plus les annonces et scores des lots acceptés.

    Un anime annoncé reste « pas encore diffusé » et sans score tant qu'un lot ultérieur
    n'a pas révélé son score ; il devient alors utilisable pour l'entraînement.
    """
    announced = pd.DataFrame([anime.model_dump() for batch in batches for anime in batch.announced])
    if announced.empty:
        return history.reset_index(drop=True)

    # Un identifiant déjà présent dans l'historique n'est pas ajouté une seconde fois.
    announced = announced.loc[~announced["id"].isin(history["id"])].drop_duplicates("id")
    scores = {score.id: score.mean for batch in batches for score in batch.scores}
    announced["mean"] = announced["id"].map(scores)
    announced.loc[announced["mean"].notna(), "status"] = "finished_airing"

    return pd.concat([history, announced], ignore_index=True)
