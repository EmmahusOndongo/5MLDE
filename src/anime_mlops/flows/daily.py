"""Flow quotidien : récupérer le lot du jour, le contrôler, puis l'accepter ou le rejeter.

Un lot rejeté part en quarantaine, le flow échoue et une alerte est émise : il
n'alimentera jamais l'entraînement.
"""

from datetime import date

from prefect import flow, get_run_logger, task

from anime_mlops.alerts import Alert, Level, send_alert
from anime_mlops.config import get_settings
from anime_mlops.data.validate import DataQualityError, ValidationReport
from anime_mlops.flows.common import alert_on_failure, publish_validation_reports
from anime_mlops.ingestion.schemas import DailyBatch
from anime_mlops.pipeline import steps
from anime_mlops.replay.app import load_catalogue


@task(name="recuperer-le-lot", retries=3, retry_delay_seconds=10)
def obtain_batch(day: date, corrupt: bool) -> DailyBatch:
    """Push déjà reçu, sinon pull. Réessayé si la source est momentanément indisponible."""
    return steps.obtain_batch(day, get_settings(), corrupt=corrupt)


@task(name="identifiants-deja-connus")
def known_ids() -> tuple[set[int], set[int]]:
    """(tous les animes déjà connus, ceux annoncés par les lots acceptés)."""
    settings = get_settings()
    announced = steps.announced_ids(steps.accepted_batches(settings))
    history = set(load_catalogue(settings).history()["id"])
    return history | announced, announced


@task(name="controler-les-annonces")
def check_announcements(batch: DailyBatch, known: set[int]) -> ValidationReport:
    return steps.check_announcements(batch, known, get_settings())


@task(name="controler-les-scores")
def check_scores(batch: DailyBatch) -> ValidationReport:
    return steps.check_scores(batch, get_settings())


@task(name="ranger-le-lot")
def route_batch(check: steps.BatchCheck) -> str:
    return steps.route_batch(check, get_settings()).path.as_posix()


@flow(
    name="ingestion-journaliere",
    on_failure=[alert_on_failure],
    on_crashed=[alert_on_failure],
)
def daily_ingestion_flow(day: date | None = None, corrupt: bool = False) -> dict:
    """Traite le lot d'un jour (par défaut : le lendemain du dernier jour traité).

    ``corrupt=True`` demande à la source un lot invalide, pour démontrer le blocage.
    """
    logger = get_run_logger()
    settings = get_settings()
    day = day or steps.next_day(settings)

    if steps.is_processed(day, settings):
        logger.info("Lot du %s déjà accepté : rien à faire.", day)
        return {"day": day.isoformat(), "status": "already_processed"}

    batch = obtain_batch(day, corrupt)
    known, previously_announced = known_ids()

    # Les deux contrôles sont indépendants : ils sont soumis en parallèle.
    announcements_report = check_announcements.submit(batch, known)
    scores_report = check_scores.submit(batch)
    reports = [announcements_report.result(), scores_report.result()]

    check = steps.BatchCheck(
        batch=batch,
        reports=tuple(reports),
        orphan_scores=steps.count_orphan_scores(batch, previously_announced),
    )
    publish_validation_reports(
        key=f"controle-lot-{day.isoformat()}", title=f"Contrôle du lot du {day}", reports=reports
    )
    path = route_batch(check)

    if not check.success:
        logger.error("Lot du %s rejeté et placé en quarantaine : %s", day, path)
        raise DataQualityError(next(report for report in reports if not report.success))

    if check.orphan_scores:
        send_alert(
            Alert(
                level=Level.WARNING,
                title=f"Lot du {day} : scores sans annonce",
                message=(
                    f"{check.orphan_scores} score(s) concernent des animes jamais annoncés "
                    "(lot précédent manquant ou rejeté) : ils seront ignorés."
                ),
                context={"day": day.isoformat()},
            ),
            settings.alerts,
        )

    logger.info(
        "Lot du %s accepté : %d annonce(s), %d score(s).",
        day,
        len(batch.announced),
        len(batch.scores),
    )
    return {
        "day": day.isoformat(),
        "status": "accepted",
        "announced": len(batch.announced),
        "scores": len(batch.scores),
        "orphan_scores": check.orphan_scores,
    }
