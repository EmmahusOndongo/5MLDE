"""Contrat d'un lot journalier, partagé par la source simulée et par la réception.

La validation faite ici est **structurelle** (champs présents, types). Les contrôles de
valeurs (score entre 0 et 10, etc.) sont faits ensuite par Great Expectations, pour
qu'un lot au contenu aberrant soit reçu, tracé, puis bloqué avec un rapport.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict


class AnnouncedAnime(BaseModel):
    """Un anime annoncé : uniquement ce qui est connu avant sa diffusion."""

    model_config = ConfigDict(extra="forbid")

    id: int
    title: str
    start_date: str
    media_type: str | None = None
    rating: str | None = None
    num_episodes: int = 0
    average_episode_duration: int = 0
    genres: str | None = None
    studios: str | None = None
    status: str = "not_yet_aired"


class RevealedScore(BaseModel):
    """Score réel d'un anime annoncé la veille : la vérité terrain, différée d'un jour."""

    model_config = ConfigDict(extra="forbid")

    id: int
    mean: float


class DailyBatch(BaseModel):
    """Ce que la source livre pour un jour donné."""

    model_config = ConfigDict(extra="forbid")

    day: date
    # Version du catalogue d'origine (hash du commit MAL) : traçabilité du lot.
    source_commit: str
    announced: list[AnnouncedAnime]
    scores: list[RevealedScore]


class DaySummary(BaseModel):
    day: date
    announced: int
    scores: int
