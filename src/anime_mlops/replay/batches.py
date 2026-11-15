"""Découpage du catalogue historique en lots journaliers.

Le jeu de données est un instantané figé : pour simuler l'arrivée de nouvelles données,
on le **rejoue** jour par jour à partir d'une date de départ.

Règles de la simulation (à préciser dans la documentation, car c'est une simplification) :

- un anime est « annoncé » le jour de sa date de début, sans aucune information
  dépendant du public (ni score, ni classement, ni nombre de membres) ;
- son score réel est « révélé » le lendemain ;
- tout ce qui précède la date de départ forme l'historique, connu d'emblée.
"""

from datetime import date, timedelta

import pandas as pd

from anime_mlops.features.build import LEAKAGE_COLUMNS
from anime_mlops.ingestion.schemas import AnnouncedAnime, DailyBatch, DaySummary, RevealedScore

ANNOUNCED_COLUMNS = [
    "id",
    "title",
    "start_date",
    "media_type",
    "rating",
    "num_episodes",
    "average_episode_duration",
    "genres",
    "studios",
]
# Jamais dans une annonce : la cible et tout ce qui dépend de la réception du public.
HIDDEN_COLUMNS = ["mean", "end_date", *LEAKAGE_COLUMNS]
ONE_DAY = timedelta(days=1)


class DayOutOfRangeError(LookupError):
    """Le jour demandé est hors de la fenêtre de simulation."""


def release_day(start_date: object) -> date | None:
    """Jour d'annonce d'un anime. Une date partielle est ramenée au premier jour connu."""
    if not isinstance(start_date, str):
        return None
    parts = start_date.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1  # noqa: PLR2004
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


class ReplayCatalogue:
    """Vue « jour par jour » d'un instantané du catalogue MAL."""

    def __init__(self, catalogue: pd.DataFrame, source_commit: str, start_day: date):
        self.source_commit = source_commit
        self.start_day = start_day
        self._catalogue = catalogue.assign(release_day=catalogue["start_date"].map(release_day))

        released = self._catalogue["release_day"].dropna()
        replayed = released[released >= start_day]
        if replayed.empty:
            raise ValueError(f"Aucun anime à rejouer à partir du {start_day}.")
        # Le dernier jour sert à révéler les scores des animes annoncés la veille.
        self.end_day = max(replayed) + ONE_DAY

    def history(self) -> pd.DataFrame:
        """Animes antérieurs à la date de départ (ou non datés) : connus dès le début."""
        day = self._catalogue["release_day"]
        mask = day.isna() | (day < self.start_day)
        return self._catalogue.loc[mask].drop(columns=["release_day"]).reset_index(drop=True)

    def batch(self, day: date, corrupt: bool = False) -> DailyBatch:
        """Lot du jour : animes annoncés ce jour-là, et scores des animes de la veille.

        ``corrupt=True`` livre volontairement un lot invalide (incident contrôlé).
        """
        if not self.start_day <= day <= self.end_day:
            raise DayOutOfRangeError(
                f"{day} est hors de la fenêtre de simulation ({self.start_day} → {self.end_day})."
            )
        announced = self._released_on(day)
        # La veille du premier jour appartient à l'historique : rien à révéler.
        revealed = self._released_on(day - ONE_DAY) if day > self.start_day else announced.head(0)
        revealed = revealed.dropna(subset=["mean"])

        announced_records = [
            {key: _clean(value) for key, value in row.items()}
            for row in announced[ANNOUNCED_COLUMNS].to_dict("records")
        ]
        scores = [
            {"id": int(row["id"]), "mean": float(row["mean"])}
            for row in revealed[["id", "mean"]].to_dict("records")
        ]
        if corrupt:
            announced_records, scores = _corrupt(announced_records, scores)

        return DailyBatch(
            day=day,
            source_commit=self.source_commit,
            announced=[AnnouncedAnime(**record) for record in announced_records],
            scores=[RevealedScore(**score) for score in scores],
        )

    def days(self, first: date | None = None, last: date | None = None) -> list[DaySummary]:
        """Jours de la fenêtre qui livrent quelque chose, avec leurs volumes."""
        first = max(first or self.start_day, self.start_day)
        last = min(last or self.end_day, self.end_day)
        summaries = []
        day = first
        while day <= last:
            batch = self.batch(day)
            if batch.announced or batch.scores:
                summaries.append(
                    DaySummary(day=day, announced=len(batch.announced), scores=len(batch.scores))
                )
            day += ONE_DAY
        return summaries

    def _released_on(self, day: date) -> pd.DataFrame:
        return self._catalogue.loc[self._catalogue["release_day"] == day]


def _clean(value: object) -> object:
    """NaN → None, entiers NumPy → int : le lot doit être du JSON standard."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _corrupt(announced: list[dict], scores: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fabrique un lot au format valide mais au contenu aberrant, pour la démonstration."""
    broken_announced = [{**record, "num_episodes": -5} for record in announced]
    broken_scores = [{**score, "mean": 15.0} for score in scores]
    # Un identifiant en double, si le lot s'y prête.
    return broken_announced + broken_announced[:1], broken_scores
