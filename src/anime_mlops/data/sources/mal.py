"""Source MyAnimeList : CSV du dépôt meesvandongen/anime-dataset, figé par hash de commit.

Le hash du commit est la version des données : un même hash redonne toujours
exactement le même fichier, ce qui rend chaque entraînement rejouable.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from anime_mlops.config import MalSourceConfig

logger = logging.getLogger(__name__)

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_TIMEOUT_SECONDS = 60.0


class MalSourceError(RuntimeError):
    """La source MAL est indisponible ou la version demandée est invalide."""


@dataclass(frozen=True)
class MalSnapshot:
    """Un instantané du catalogue MAL, identifié par son commit."""

    path: Path
    commit: str
    sha256: str
    size_bytes: int


def resolve_latest_commit(config: MalSourceConfig, client: httpx.Client | None = None) -> str:
    """Hash du dernier commit ayant modifié le CSV (pour figer une nouvelle version)."""
    url = f"https://api.github.com/repos/{config.repo}/commits"
    params = {"path": config.file_path, "per_page": 1}
    try:
        response = _get(url, client, params=params)
        commits = response.json()
        return commits[0]["sha"]
    except (httpx.HTTPError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise MalSourceError(f"Impossible de résoudre le dernier commit MAL : {exc}") from exc


def download_snapshot(
    commit: str,
    config: MalSourceConfig,
    raw_dir: Path,
    client: httpx.Client | None = None,
) -> MalSnapshot:
    """Télécharge le CSV du commit donné. Idempotent : un fichier déjà présent est réutilisé."""
    if not _COMMIT_SHA.fullmatch(commit):
        raise MalSourceError(
            f"Version MAL invalide : {commit!r}. Un hash de commit complet (40 caractères "
            "hexadécimaux) est exigé ; une branche comme 'main' n'est pas reproductible."
        )

    target = raw_dir / "mal" / f"anime-standalone_{commit}.csv"
    if target.is_file():
        logger.info("Instantané MAL %s déjà présent, téléchargement ignoré.", commit[:8])
    else:
        try:
            content = _get(config.raw_url(commit), client).content
        except httpx.HTTPError as exc:
            raise MalSourceError(f"Téléchargement MAL impossible ({commit[:8]}) : {exc}") from exc
        if not content.strip():
            raise MalSourceError(f"Le fichier MAL du commit {commit[:8]} est vide.")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Écriture atomique : un téléchargement interrompu ne laisse pas de fichier tronqué.
        temporary = target.with_suffix(".part")
        temporary.write_bytes(content)
        temporary.replace(target)
        logger.info("Instantané MAL %s téléchargé (%d octets).", commit[:8], len(content))

    data = target.read_bytes()
    return MalSnapshot(
        path=target,
        commit=commit,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def load_snapshot(snapshot: MalSnapshot) -> pd.DataFrame:
    """Charge le CSV brut, sans transformation (la validation se fait en aval)."""
    return pd.read_csv(snapshot.path)


def _get(url: str, client: httpx.Client | None, **kwargs) -> httpx.Response:
    if client is not None:
        response = client.get(url, **kwargs)
    else:
        response = httpx.get(url, timeout=_TIMEOUT_SECONDS, follow_redirects=True, **kwargs)
    response.raise_for_status()
    return response
