"""Ligne de commande des flows.

    python -m anime_mlops.flows.cli serve                      # planification automatique
    python -m anime_mlops.flows.cli daily [--day 2025-01-05] [--corrupt]
    python -m anime_mlops.flows.cli training [--as-of 2025-01-31]
    python -m anime_mlops.flows.cli feedback                   # rattrapage du feedback

``serve`` reste au premier plan : il enregistre les deux déploiements auprès du serveur
Prefect et exécute les flows aux heures prévues (ou à la demande depuis l'interface).
"""

import argparse
import sys
from datetime import date

from prefect import serve
from prefect.schedules import Cron

from anime_mlops.config import get_settings
from anime_mlops.flows.daily import daily_ingestion_flow
from anime_mlops.flows.feedback import feedback_flow
from anime_mlops.flows.training import training_flow


def serve_flows() -> None:
    schedules = get_settings().schedules
    serve(
        daily_ingestion_flow.to_deployment(
            name="quotidien",
            schedules=[Cron(schedules.daily_ingestion, timezone=schedules.timezone)],
            tags=["ingestion"],
            description="Récupère et contrôle le lot du jour.",
        ),
        training_flow.to_deployment(
            name="hebdomadaire",
            schedules=[Cron(schedules.weekly_training, timezone=schedules.timezone)],
            tags=["entrainement"],
            description="Réentraîne le modèle et applique les règles de promotion.",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Flows Prefect du projet.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="planifie et exécute les flows")
    daily = commands.add_parser("daily", help="traite le lot d'un jour")
    daily.add_argument("--day", type=date.fromisoformat, help="jour à traiter (AAAA-MM-JJ)")
    daily.add_argument("--corrupt", action="store_true", help="demande un lot invalide")
    training = commands.add_parser("training", help="réentraîne le modèle")
    training.add_argument("--as-of", type=date.fromisoformat, help="dernier jour pris en compte")
    commands.add_parser("feedback", help="prédit les annonces et rattache les scores réels")
    arguments = parser.parse_args(argv)

    if arguments.command == "serve":
        serve_flows()
        return 0
    # Un flow en échec renvoie un état « Failed » : on le traduit en code de retour.
    if arguments.command == "daily":
        state = daily_ingestion_flow(arguments.day, arguments.corrupt, return_state=True)
    elif arguments.command == "training":
        state = training_flow(arguments.as_of, return_state=True)
    else:
        state = feedback_flow(return_state=True)
    return 0 if state.is_completed() else 1


if __name__ == "__main__":
    sys.exit(main())
