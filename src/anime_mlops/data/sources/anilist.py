"""Source AniList : API GraphQL publique, extraite par pages et mise en cache Parquet daté.

La date d'extraction est la version des données. Si l'API est indisponible,
on se replie sur le cache le plus récent plutôt que de casser la chaîne.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from anime_mlops.config import AnilistSourceConfig

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30.0
_DEFAULT_RATE_LIMIT_WAIT_SECONDS = 60.0
_CACHE_PREFIX = "anilist_"

QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: ANIME, sort: ID) {
      id
      idMal
      format
      source
      episodes
      tags { name }
      relations { edges { relationType node { idMal type } } }
    }
  }
}
"""

COLUMNS = [
    "id_mal",
    "anilist_id",
    "format",
    "source",
    "episodes",
    "tags",
    "is_sequel",
    "prequel_id_mal",
]


class AnilistUnavailableError(RuntimeError):
    """L'API AniList ne répond pas correctement malgré les nouvelles tentatives."""


@dataclass(frozen=True)
class AnilistExtract:
    """Une extraction AniList et sa version (date d'extraction)."""

    data: pd.DataFrame
    extraction_date: date
    path: Path
    # True si l'API était indisponible et qu'un cache antérieur a été utilisé.
    is_fallback: bool


def get_extract(
    config: AnilistSourceConfig,
    cache_dir: Path,
    today: date | None = None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AnilistExtract:
    """Renvoie l'extraction du jour : cache du jour, sinon API, sinon dernier cache connu."""
    today = today or date.today()
    directory = cache_dir / "anilist"
    todays_cache = _cache_path(directory, today)

    if todays_cache.is_file():
        logger.info("Cache AniList du %s réutilisé.", today)
        return AnilistExtract(pd.read_parquet(todays_cache), today, todays_cache, is_fallback=False)

    try:
        data = fetch_all(config, client=client, sleep=sleep)
    except AnilistUnavailableError as exc:
        fallback = latest_cache(directory)
        if fallback is None:
            raise AnilistUnavailableError(
                f"AniList indisponible et aucun cache local pour se replier : {exc}"
            ) from exc
        fallback_date, fallback_path = fallback
        logger.warning(
            "AniList indisponible (%s) : repli sur le cache du %s.", exc, fallback_date
        )
        return AnilistExtract(
            pd.read_parquet(fallback_path), fallback_date, fallback_path, is_fallback=True
        )

    directory.mkdir(parents=True, exist_ok=True)
    temporary = todays_cache.with_suffix(".part")
    data.to_parquet(temporary, index=False)
    temporary.replace(todays_cache)
    logger.info("Extraction AniList du %s mise en cache (%d animes).", today, len(data))
    return AnilistExtract(data, today, todays_cache, is_fallback=False)


def fetch_all(
    config: AnilistSourceConfig,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Parcourt toutes les pages de l'API en respectant la limite de requêtes."""
    owns_client = client is None
    client = client or httpx.Client(timeout=_TIMEOUT_SECONDS)
    records: list[dict] = []
    try:
        page = 1
        while True:
            payload = _fetch_page(client, config, page, sleep)
            records.extend(parse_media(media) for media in payload["media"])
            reached_limit = config.max_pages is not None and page >= config.max_pages
            if not payload["pageInfo"]["hasNextPage"] or reached_limit:
                break
            page += 1
            sleep(config.request_interval_seconds)
    finally:
        if owns_client:
            client.close()

    data = pd.DataFrame(records, columns=COLUMNS)
    # Un anime AniList sans identifiant MAL ne peut pas être joint : il est écarté ici.
    data = data.dropna(subset=["id_mal"]).astype({"id_mal": "int64"})
    return data.reset_index(drop=True)


def parse_media(media: dict) -> dict:
    """Aplatit un objet Media de l'API en une ligne tabulaire."""
    edges = (media.get("relations") or {}).get("edges") or []
    prequel_ids = [
        edge["node"].get("idMal")
        for edge in edges
        if edge.get("relationType") == "PREQUEL" and (edge.get("node") or {}).get("type") == "ANIME"
    ]
    known_prequel_ids = [prequel_id for prequel_id in prequel_ids if prequel_id is not None]
    return {
        "id_mal": media.get("idMal"),
        "anilist_id": media["id"],
        "format": media.get("format"),
        "source": media.get("source"),
        "episodes": media.get("episodes"),
        "tags": [tag["name"] for tag in media.get("tags") or []],
        "is_sequel": bool(prequel_ids),
        "prequel_id_mal": known_prequel_ids[0] if known_prequel_ids else None,
    }


def latest_cache(directory: Path) -> tuple[date, Path] | None:
    """Cache le plus récent disponible, ou None s'il n'y en a aucun."""
    candidates = []
    for path in directory.glob(f"{_CACHE_PREFIX}*.parquet"):
        try:
            candidates.append((date.fromisoformat(path.stem.removeprefix(_CACHE_PREFIX)), path))
        except ValueError:
            logger.warning("Fichier de cache AniList ignoré (nom inattendu) : %s", path.name)
    return max(candidates, default=None)


def _cache_path(directory: Path, day: date) -> Path:
    return directory / f"{_CACHE_PREFIX}{day.isoformat()}.parquet"


def _fetch_page(
    client: httpx.Client,
    config: AnilistSourceConfig,
    page: int,
    sleep: Callable[[float], None],
) -> dict:
    """Récupère une page, avec nouvelles tentatives sur erreur réseau, 5xx ou limite atteinte."""
    body = {"query": QUERY, "variables": {"page": page, "perPage": config.page_size}}
    last_error = "aucune tentative effectuée"

    for attempt in range(config.max_retries + 1):
        if attempt > 0:
            sleep(config.retry_delay_seconds)
        try:
            response = client.post(config.url, json=body)
        except httpx.TransportError as exc:
            last_error = f"erreur réseau ({exc.__class__.__name__})"
            logger.warning("AniList page %d, tentative %d : %s", page, attempt + 1, last_error)
            continue

        if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
            wait = _retry_after_seconds(response)
            last_error = "limite de requêtes atteinte (HTTP 429)"
            logger.warning("AniList page %d : %s, attente de %.0f s.", page, last_error, wait)
            sleep(wait)
            continue
        if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            last_error = f"erreur serveur (HTTP {response.status_code})"
            logger.warning("AniList page %d, tentative %d : %s", page, attempt + 1, last_error)
            continue
        if response.status_code != httpx.codes.OK:
            # Erreur côté client (requête refusée) : réessayer ne changerait rien.
            raise AnilistUnavailableError(
                f"page {page} refusée par AniList (HTTP {response.status_code})"
            )

        try:
            return response.json()["data"]["Page"]
        except (ValueError, KeyError, TypeError) as exc:
            raise AnilistUnavailableError(f"page {page} : réponse au format inattendu") from exc

    raise AnilistUnavailableError(
        f"page {page} : {last_error} après {config.max_retries + 1} tentative(s)"
    )


def _retry_after_seconds(response: httpx.Response) -> float:
    try:
        return max(float(response.headers["Retry-After"]), 0.0)
    except (KeyError, ValueError):
        return _DEFAULT_RATE_LIMIT_WAIT_SECONDS
