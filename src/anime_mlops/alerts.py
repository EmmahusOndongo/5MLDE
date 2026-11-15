"""Alertes d'exploitation.

Une alerte est toujours journalisée et ajoutée à un fichier local (une ligne JSON par
alerte, exploitable par la supervision). Si un webhook est configuré, elle y est aussi
envoyée. L'envoi ne doit jamais faire échouer le traitement qui alerte.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import httpx

from anime_mlops.config import AlertsConfig

logger = logging.getLogger(__name__)

_WEBHOOK_TIMEOUT_SECONDS = 5.0


class Level(StrEnum):
    WARNING = "warning"  # dégradation gérée (ex. repli sur un cache)
    CRITICAL = "critical"  # traitement interrompu (ex. lot invalide, pipeline en échec)


@dataclass(frozen=True)
class Alert:
    level: Level
    title: str
    message: str
    context: dict = field(default_factory=dict)
    raised_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def send_alert(alert: Alert, config: AlertsConfig, client: httpx.Client | None = None) -> Alert:
    """Émet l'alerte sur tous les canaux disponibles."""
    log = logger.error if alert.level is Level.CRITICAL else logger.warning
    log("ALERTE [%s] %s — %s", alert.level.value, alert.title, alert.message)

    try:
        config.file.parent.mkdir(parents=True, exist_ok=True)
        with config.file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(alert), ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.error("Alerte non écrite dans %s : %s", config.file, exc)

    if config.webhook_url is not None:
        _post_webhook(alert, config.webhook_url.get_secret_value(), client)
    return alert


def _post_webhook(alert: Alert, url: str, client: httpx.Client | None) -> None:
    text = f"[{alert.level.value.upper()}] {alert.title}\n{alert.message}"
    # "text" est lu par Slack et Teams, "content" par Discord.
    payload = {"text": text, "content": text}
    try:
        if client is not None:
            response = client.post(url, json=payload)
        else:
            response = httpx.post(url, json=payload, timeout=_WEBHOOK_TIMEOUT_SECONDS)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # L'adresse du webhook est un secret : on ne la journalise pas.
        logger.error("Envoi de l'alerte au webhook impossible (%s).", exc.__class__.__name__)
