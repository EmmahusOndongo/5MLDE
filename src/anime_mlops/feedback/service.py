"""Boucle de feedback, sans dépendance à l'orchestrateur (donc testable seule).

    jour J   : lot accepté ──► features des animes annoncés ──► API /predict/batch
                                                                  │ prediction_id
                                                                  ▼
                                                        registre « anime → prédiction »
    jour J+1 : lot accepté ──► scores révélés ──► feedback « automatic » en base

Les deux moitiés sont idempotentes et se rattrapent seules : un jour sans registre est
prédit au passage suivant, une prédiction sans feedback est complétée dès que son score
est connu. Une panne de l'API ou de la base ne fait donc rien perdre.
"""

import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from uuid import UUID

import httpx
import pandas as pd

from anime_mlops.api.persistence import (
    DatabaseStore,
    FeedbackConflictError,
    PredictionNotFoundError,
)
from anime_mlops.config import FeedbackConfig, Settings
from anime_mlops.data.merge import merge_sources
from anime_mlops.data.sources import anilist
from anime_mlops.features.build import FEATURES, build_features
from anime_mlops.feedback import ledger
from anime_mlops.feedback.ledger import DayLedger, TrackedPrediction
from anime_mlops.ingestion import landing
from anime_mlops.pipeline import steps

logger = logging.getLogger(__name__)

_INTEGER_FEATURES = ("num_episodes", "start_year")
_UNPROCESSABLE = 422

# Enregistre le score réel d'une prédiction (prediction_id, score).
FeedbackWriter = Callable[[UUID, float], None]


class ApiUnavailableError(RuntimeError):
    """L'API de prédiction n'a pas pu prédire, ou n'a pas pu tracer ses prédictions."""


@dataclass(frozen=True)
class ApiPrediction:
    prediction_id: UUID
    prediction: float
    model_version: str


@dataclass(frozen=True)
class FeedbackSummary:
    """Bilan d'un passage de rattachement des scores réels."""

    # Feedbacks enregistrés pendant ce passage.
    recorded: int
    # Déjà présents en base (passage précédent interrompu) : simplement pointés.
    already_recorded: int
    # Prédictions absentes de la base : le score ne peut pas être rattaché.
    missing_predictions: int
    # Prédictions dont le score réel n'est pas encore connu.
    awaiting_score: int
    # Erreur absolue moyenne des feedbacks de ce passage (None s'il n'y en a aucun).
    mean_absolute_error: float | None


# --- Jour J : prédire les animes annoncés -------------------------------------------


def prediction_records(
    announced_ids: list[int], catalogue: pd.DataFrame, anilist_data: pd.DataFrame
) -> dict[int, dict[str, object]]:
    """Features brutes de chaque anime annoncé, au format attendu par l'API.

    Elles passent par la même jointure et le même ``build_features`` que l'entraînement :
    le modèle voit en service exactement ce qu'il a vu à l'apprentissage.
    """
    merged = merge_sources(catalogue, anilist_data)
    features = build_features(merged.loc[merged["id"].isin(announced_ids)])
    return {
        int(anime_id): {name: _json_value(name, row[name]) for name in FEATURES}
        for anime_id, row in features.iterrows()
    }


def request_predictions(
    records: list[dict[str, object]], config: FeedbackConfig, client: httpx.Client | None = None
) -> list[ApiPrediction | None]:
    """Prédit par lots via l'API. ``None`` pour un anime que l'API refuse (entrée invalide)."""
    own_client = client is None
    client = client or api_client(config)
    try:
        results: list[ApiPrediction | None] = []
        for start in range(0, len(records), config.batch_size):
            results += _predict_chunk(records[start : start + config.batch_size], config, client)
        return results
    finally:
        if own_client:
            client.close()


def api_client(config: FeedbackConfig) -> httpx.Client:
    return httpx.Client(timeout=config.timeout_seconds)


def pending_prediction_days(settings: Settings) -> list[date]:
    """Jours acceptés dont les annonces n'ont pas encore été prédites."""
    done = set(ledger.ledger_days(settings.paths.predictions_dir))
    return [day for day in landing.landed_days(settings.paths.accepted_dir) if day not in done]


def predict_day(
    day: date, settings: Settings, history: pd.DataFrame, client: httpx.Client | None = None
) -> DayLedger:
    """Prédit les animes annoncés le jour ``day`` et consigne les ``prediction_id``."""
    existing = ledger.load(day, settings.paths.predictions_dir)
    if existing is not None:
        return existing

    batch = landing.load_batch(day, settings.paths.accepted_dir)
    announced_ids = [anime.id for anime in batch.announced]
    tracked: list[TrackedPrediction] = []
    if announced_ids:
        # Catalogue connu ce jour-là : aucune information postérieure n'est utilisée.
        catalogue = steps.build_catalogue(history, steps.accepted_batches(settings, day))
        records = prediction_records(announced_ids, catalogue, _anilist_enrichment(settings))
        ordered_ids = [anime_id for anime_id in announced_ids if anime_id in records]
        answers = request_predictions(
            [records[anime_id] for anime_id in ordered_ids], settings.feedback, client
        )
        for anime_id, answer in zip(ordered_ids, answers, strict=True):
            if answer is None:
                logger.warning("Anime %d : entrée refusée par l'API, pas de prédiction.", anime_id)
                continue
            tracked.append(
                TrackedPrediction(
                    anime_id=anime_id,
                    prediction_id=answer.prediction_id,
                    prediction=answer.prediction,
                    model_version=answer.model_version,
                )
            )

    day_ledger = DayLedger(day=day, predictions=tracked)
    ledger.save(day_ledger, settings.paths.predictions_dir)
    logger.info("Jour %s : %d annonce(s), %d prédiction(s).", day, len(announced_ids), len(tracked))
    return day_ledger


# --- Jour J+1 : rattacher les scores réels ------------------------------------------


@contextmanager
def database_writer(settings: Settings) -> Iterator[FeedbackWriter]:
    """Écriture des feedbacks dans la table ``feedbacks`` de PostgreSQL (source « automatic »)."""
    store = DatabaseStore.from_url(
        settings.database.url,
        connect_timeout_seconds=settings.database.connect_timeout_seconds,
        retry_cooldown_seconds=0,
    )

    def write(prediction_id: UUID, actual_score: float) -> None:
        store.add_feedback(
            prediction_id=prediction_id,
            actual_score=actual_score,
            source="automatic",
            comment="Score réel révélé par le rejeu journalier.",
        )

    try:
        yield write
    finally:
        store.close()


def revealed_scores(settings: Settings) -> dict[int, float]:
    """Score réel de chaque anime, d'après les lots acceptés."""
    batches = steps.accepted_batches(settings)
    return {score.id: score.mean for batch in batches for score in batch.scores}


def record_feedback(settings: Settings, write: FeedbackWriter) -> FeedbackSummary:
    """Rattache chaque score révélé à la prédiction faite à l'annonce.

    La progression est sauvegardée registre par registre : si la base tombe en cours de
    route, l'erreur remonte et le passage suivant reprend là où celui-ci s'est arrêté.
    """
    scores = revealed_scores(settings)
    recorded = already_recorded = missing = awaiting = 0
    errors: list[float] = []

    for day_ledger in ledger.load_all(settings.paths.predictions_dir):
        changed = False
        try:
            for entry in day_ledger.predictions:
                if not entry.is_pending:
                    continue
                if entry.anime_id not in scores:
                    awaiting += 1
                    continue
                actual_score = scores[entry.anime_id]
                try:
                    write(entry.prediction_id, actual_score)
                    recorded += 1
                    errors.append(abs(entry.prediction - actual_score))
                except FeedbackConflictError:
                    already_recorded += 1
                except PredictionNotFoundError:
                    logger.warning(
                        "Anime %d : prédiction %s absente de la base, feedback impossible.",
                        entry.anime_id,
                        entry.prediction_id,
                    )
                    missing += 1
                    continue
                entry.actual_score = actual_score
                entry.feedback_at = datetime.now(UTC)
                changed = True
        finally:
            if changed:
                ledger.save(day_ledger, settings.paths.predictions_dir)

    return FeedbackSummary(
        recorded=recorded,
        already_recorded=already_recorded,
        missing_predictions=missing,
        awaiting_score=awaiting,
        mean_absolute_error=sum(errors) / len(errors) if errors else None,
    )


# --- Détails ------------------------------------------------------------------------


def _anilist_enrichment(settings: Settings) -> pd.DataFrame:
    """Dernière extraction AniList en cache (aucun appel réseau dans le flow quotidien)."""
    recent = anilist.latest_cache(settings.paths.cache_dir / "anilist")
    if recent is None:
        logger.warning("Aucun cache AniList : prédictions faites sans enrichissement.")
        return pd.DataFrame(columns=anilist.COLUMNS).astype({"id_mal": "int64"})
    return pd.read_parquet(recent[1])


def _json_value(name: str, value: object) -> object:
    """Valeur sérialisable : NaN → null, entiers sans décimale, listes Python."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if name in _INTEGER_FEATURES:
        return int(value)
    if isinstance(value, bool | str):
        return value
    return float(value) if isinstance(value, int | float) else value.item()


def _predict_chunk(
    chunk: list[dict[str, object]], config: FeedbackConfig, client: httpx.Client
) -> list[ApiPrediction | None]:
    body = _post(client, f"{config.api_url.rstrip('/')}/predict/batch", {"instances": chunk})
    if body is None:
        # Une entrée au moins est refusée : on isole la fautive au lieu de perdre tout le lot.
        return [_predict_one(record, config, client) for record in chunk]
    version = body["model"]["version"]
    return [
        ApiPrediction(UUID(prediction_id), float(prediction), version)
        for prediction_id, prediction in zip(
            body["prediction_ids"], body["predictions"], strict=True
        )
    ]


def _predict_one(
    record: dict[str, object], config: FeedbackConfig, client: httpx.Client
) -> ApiPrediction | None:
    body = _post(client, f"{config.api_url.rstrip('/')}/predict", record)
    if body is None:
        return None
    return ApiPrediction(
        UUID(body["prediction_id"]), float(body["prediction"]), body["model"]["version"]
    )


def _post(client: httpx.Client, url: str, payload: dict) -> dict | None:
    """Réponse de l'API, ou ``None`` si elle juge l'entrée invalide (422)."""
    try:
        response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise ApiUnavailableError(f"API de prédiction injoignable ({url}) : {exc}") from exc
    if response.status_code == _UNPROCESSABLE:
        return None
    if response.status_code >= 400:
        raise ApiUnavailableError(
            f"L'API de prédiction a répondu HTTP {response.status_code} ({url})."
        )
    body = response.json()
    if not body.get("persisted", False):
        # Sans trace en base, le score réel ne pourrait jamais être rattaché.
        raise ApiUnavailableError(
            "L'API a prédit sans pouvoir tracer en base : prédictions à refaire plus tard."
        )
    return body
