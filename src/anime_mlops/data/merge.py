"""Jointure MyAnimeList ↔ AniList sur l'identifiant MAL.

Tous les animes MAL sont conservés : AniList est un enrichissement, et environ un
quart du catalogue MAL (clips, publicités…) n'y figure pas. Les colonnes AniList
restent vides dans ce cas, et ``has_anilist`` le signale.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

ANILIST_COLUMNS = {
    "anilist_id": "anilist_id",
    "format": "anilist_format",
    "source": "source",
    "episodes": "anilist_episodes",
    "tags": "tags",
    "is_sequel": "is_sequel",
    "prequel_id_mal": "prequel_id_mal",
}


def merge_sources(mal: pd.DataFrame, anilist: pd.DataFrame) -> pd.DataFrame:
    """Enrichit le catalogue MAL avec AniList (jointure gauche, une ligne par anime MAL)."""
    enrichment = anilist[["id_mal", *ANILIST_COLUMNS]].rename(columns=ANILIST_COLUMNS)
    merged = mal.merge(
        enrichment,
        how="left",
        left_on="id",
        right_on="id_mal",
        # Garantit qu'aucune ligne n'est dupliquée, même si un doublon passait la validation.
        validate="one_to_one",
        indicator=True,
    )
    merged["has_anilist"] = merged.pop("_merge") == "both"
    merged = merged.drop(columns=["id_mal"])

    # Sans fiche AniList, on ne sait pas si l'anime est une suite : on suppose que non.
    merged["is_sequel"] = merged["is_sequel"].astype("boolean").fillna(False).astype(bool)
    merged["prequel_score"] = _prequel_score(merged)
    merged["episodes_agree"] = _episodes_agree(merged)

    logger.info(
        "Jointure : %d animes MAL, %d enrichis par AniList (%.1f %%).",
        len(merged),
        int(merged["has_anilist"].sum()),
        100 * merged["has_anilist"].mean() if len(merged) else 0.0,
    )
    return merged


def training_rows(merged: pd.DataFrame) -> pd.DataFrame:
    """Animes utilisables pour l'entraînement : diffusion terminée et score connu."""
    mask = (merged["status"] == "finished_airing") & merged["mean"].notna()
    return merged.loc[mask].reset_index(drop=True)


def _prequel_score(merged: pd.DataFrame) -> pd.Series:
    """Score MAL de la saison précédente : connu avant la diffusion de la suite."""
    scores = merged.set_index("id")["mean"]
    return merged["prequel_id_mal"].map(scores)


def _episodes_agree(merged: pd.DataFrame) -> pd.Series:
    """Accord des deux sources sur le nombre d'épisodes (vide si l'une des deux l'ignore)."""
    mal_episodes = merged["num_episodes"].where(merged["num_episodes"] > 0)
    anilist_episodes = merged["anilist_episodes"]
    agree = (mal_episodes == anilist_episodes).astype("boolean")
    return agree.where(mal_episodes.notna() & anilist_episodes.notna())
