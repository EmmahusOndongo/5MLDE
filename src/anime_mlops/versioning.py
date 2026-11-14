"""Version du code, pour relier chaque entraînement au commit qui l'a produit."""

import hashlib
import subprocess
from pathlib import Path

from anime_mlops.config import PROJECT_ROOT

UNKNOWN = "inconnu"


def code_version(repository: Path = PROJECT_ROOT) -> tuple[str, bool]:
    """Renvoie (hash du commit, présence de modifications non commitées)."""
    try:
        commit = _git(repository, "rev-parse", "HEAD")
        dirty = bool(_git(repository, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        # Hors dépôt Git (image Docker, archive) : on ne bloque pas l'entraînement.
        return UNKNOWN, False
    return commit, dirty


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", *arguments],  # noqa: S607
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return completed.stdout.strip()
