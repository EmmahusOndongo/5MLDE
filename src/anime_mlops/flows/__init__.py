"""Flows Prefect : orchestration de l'ingestion journalière et de l'entraînement."""

import os

# Fait remonter les journaux du projet dans l'interface Prefect (à fixer avant son import).
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "anime_mlops")
