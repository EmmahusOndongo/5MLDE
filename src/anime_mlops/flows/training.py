"""Flow hebdomadaire : reconstruire le jeu de données, le contrôler, entraîner, promouvoir.

Le jeu d'entraînement est l'historique plus tous les lots journaliers **acceptés**.
Chaque contrôle qualité est bloquant : s'il échoue, rien n'est entraîné et le champion
en place continue de servir.
"""

from datetime import date

import pandas as pd
from prefect import flow, get_run_logger, task

from anime_mlops.alerts import Alert, Level, send_alert
from anime_mlops.config import get_settings, load_promotion_rules
from anime_mlops.data import validate
from anime_mlops.data.dataset import DataVersions
from anime_mlops.data.merge import merge_sources, training_rows
from anime_mlops.data.sources import anilist
from anime_mlops.features.build import split_features_target
from anime_mlops.flows.common import alert_on_failure, publish_validation_reports
from anime_mlops.pipeline import steps
from anime_mlops.replay.app import load_catalogue
from anime_mlops.training.promote import Decision, promote
from anime_mlops.training.train import TrainingResult, train_and_log
from anime_mlops.versioning import file_sha256


@task(name="construire-le-catalogue")
def build_catalogue(as_of: date | None) -> tuple[pd.DataFrame, str, int]:
    """Historique + lots acceptés. Renvoie (catalogue, commit source, nombre de lots)."""
    settings = get_settings()
    source = load_catalogue(settings)
    batches = steps.accepted_batches(settings, as_of)
    return steps.build_catalogue(source.history(), batches), source.source_commit, len(batches)


@task(name="controler-le-catalogue")
def check_catalogue(catalogue: pd.DataFrame, version: str) -> validate.ValidationReport:
    settings = get_settings()
    return validate.validate_mal(catalogue, version, settings.quality, settings.paths.reports_dir)


@task(name="extraire-anilist", retries=1, retry_delay_seconds=30)
def extract_anilist(mal_ids: list[int]) -> anilist.AnilistExtract:
    """Extraction AniList (cache récent réutilisé, repli sur le dernier cache si l'API tombe)."""
    settings = get_settings()
    extract = anilist.get_extract(mal_ids, settings.sources.anilist, settings.paths.cache_dir)
    if extract.is_fallback:
        send_alert(
            Alert(
                level=Level.WARNING,
                title="AniList indisponible : repli sur le cache",
                message=(
                    f"L'extraction a échoué ; le cache du {extract.extraction_date} est utilisé. "
                    "L'entraînement continue avec des données d'enrichissement plus anciennes."
                ),
                context={"cache": extract.path.as_posix()},
            ),
            settings.alerts,
        )
    return extract


@task(name="controler-anilist")
def check_anilist(extract: anilist.AnilistExtract) -> validate.ValidationReport:
    settings = get_settings()
    return validate.validate_anilist(
        extract.data,
        extract.extraction_date.isoformat(),
        settings.quality,
        settings.paths.reports_dir,
    )


@task(name="fusionner-et-controler")
def merge_and_check(
    catalogue: pd.DataFrame, extract: anilist.AnilistExtract, version: str
) -> tuple[pd.DataFrame, validate.ValidationReport]:
    settings = get_settings()
    training = training_rows(merge_sources(catalogue, extract.data))
    report = validate.validate_merged(
        training, version, settings.quality, settings.paths.reports_dir
    )
    return training, report


@task(name="entrainer")
def train(
    training: pd.DataFrame, versions: DataVersions, reports: list[validate.ValidationReport]
) -> TrainingResult:
    features, target = split_features_target(training)
    return train_and_log(features, target, versions, get_settings(), tuple(reports))


@task(name="promouvoir", retries=2, retry_delay_seconds=5)
def promote_model(model_version: str) -> Decision:
    return promote(get_settings(), load_promotion_rules(), model_version)


@flow(name="entrainement", on_failure=[alert_on_failure], on_crashed=[alert_on_failure])
def training_flow(as_of: date | None = None) -> dict:
    """Réentraîne sur les données connues au jour ``as_of`` (défaut : tous les lots acceptés)."""
    logger = get_run_logger()
    settings = get_settings()

    catalogue, source_commit, batch_count = build_catalogue(as_of)
    as_of_label = as_of.isoformat() if as_of else "tous-les-lots"
    logger.info("Catalogue : %d animes, dont %d lot(s) journalier(s).", len(catalogue), batch_count)

    # Deux branches indépendantes, lancées en parallèle : le contrôle du catalogue d'un
    # côté, l'extraction AniList (réseau) puis son contrôle de l'autre.
    catalogue_report = check_catalogue.submit(catalogue, source_commit)
    extract_future = extract_anilist.submit(catalogue["id"].tolist())
    anilist_report = check_anilist.submit(extract_future)

    extract = extract_future.result()
    reports = [catalogue_report.result(), anilist_report.result()]

    versions = DataVersions(
        mal_commit=source_commit,
        mal_sha256=file_sha256(
            settings.paths.raw_dir / "mal" / f"anime-standalone_{source_commit}.csv"
        ),
        anilist_extraction_date=extract.extraction_date.isoformat(),
        anilist_sha256=file_sha256(extract.path),
        anilist_is_fallback=extract.is_fallback,
        replay_as_of=as_of_label,
        replay_batches=batch_count,
    )
    training, merged_report = merge_and_check(catalogue, extract, versions.label)
    reports.append(merged_report)
    publish_validation_reports(
        key=f"controles-entrainement-{as_of_label}",
        title=f"Contrôles qualité avant entraînement ({as_of_label})",
        reports=reports,
    )

    result = train(training, versions, reports)
    decision = promote_model(result.model_version)
    logger.info(
        "Version %s : %s (%s).", result.model_version, decision.status.value, decision.reason
    )
    return {
        "model_version": result.model_version,
        "run_id": result.run_id,
        "test_mae": result.metrics["test_mae"],
        "status": decision.status.value,
        "reason": decision.reason,
        "training_rows": len(training),
        "batches": batch_count,
    }
