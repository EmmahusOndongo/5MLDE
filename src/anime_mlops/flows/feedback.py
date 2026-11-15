"""Flow de feedback automatique : prédire à l'annonce, comparer au score réel ensuite.

Il tourne après chaque lot accepté. Il n'est jamais bloquant pour l'ingestion : si l'API
ou la base est indisponible, le flow échoue avec une alerte, et le passage suivant
rattrape tout ce qui manque.
"""

from datetime import date

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact

from anime_mlops.config import get_settings
from anime_mlops.feedback import service
from anime_mlops.flows.common import alert_on_failure
from anime_mlops.replay.app import load_catalogue


@task(name="predire-les-annonces", retries=2, retry_delay_seconds=10)
def predict_announcements(days: list[date]) -> int:
    """Prédit, via l'API, les animes annoncés des jours pas encore traités."""
    settings = get_settings()
    history = load_catalogue(settings).history()
    return sum(len(service.predict_day(day, settings, history).predictions) for day in days)


@task(name="rattacher-les-scores", retries=2, retry_delay_seconds=10)
def attach_scores() -> service.FeedbackSummary:
    """Enregistre comme feedback « automatic » chaque score réel désormais connu."""
    settings = get_settings()
    with service.database_writer(settings) as write:
        return service.record_feedback(settings, write)


@flow(name="feedback-automatique", on_failure=[alert_on_failure], on_crashed=[alert_on_failure])
def feedback_flow() -> dict:
    logger = get_run_logger()
    days = service.pending_prediction_days(get_settings())

    predicted = predict_announcements(days) if days else 0
    summary = attach_scores()

    error = summary.mean_absolute_error
    error_label = f"{error:.3f}" if error is not None else "—"
    logger.info(
        "%d prédiction(s) faite(s), %d feedback(s) enregistré(s), erreur absolue moyenne : %s.",
        predicted,
        summary.recorded,
        error_label,
    )
    create_markdown_artifact(
        key="feedback-automatique",
        description="Bilan du feedback automatique",
        markdown="\n".join(
            [
                "# Feedback automatique",
                "",
                "| Indicateur | Valeur |",
                "|---|---|",
                f"| Jours prédits pendant ce passage | {len(days)} |",
                f"| Prédictions faites à l'annonce | {predicted} |",
                f"| Feedbacks enregistrés | {summary.recorded} |",
                f"| Feedbacks déjà en base | {summary.already_recorded} |",
                f"| Prédictions absentes de la base | {summary.missing_predictions} |",
                f"| Prédictions en attente de leur score | {summary.awaiting_score} |",
                f"| Erreur absolue moyenne (ce passage) | {error_label} |",
            ]
        ),
    )
    return {
        "predicted_days": len(days),
        "predictions": predicted,
        "feedbacks": summary.recorded,
        "already_recorded": summary.already_recorded,
        "missing_predictions": summary.missing_predictions,
        "awaiting_score": summary.awaiting_score,
        "mean_absolute_error": error,
    }
