.DEFAULT_GOAL := help
.PHONY: help setup lint format test ingest train promote rollback status replay receiver flows daily pipeline feedback demo-incident demo-nominal demo-postgres-incident api register-mock-model up down logs monitoring-logs load-test

LOAD_TEST_ARGS ?=
API_URL ?= http://127.0.0.1:8000
PROJECT_NAME ?= anime-mlops
CONFIRM_PROJECT ?=

help:  ## Liste les commandes disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Installe les dépendances (versions figées par uv.lock)
	uv sync --locked

lint:  ## Analyse statique et vérification du formatage (ruff)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Corrige automatiquement le style
	uv run ruff check --fix .
	uv run ruff format .

test:  ## Tests unitaires (sans la stack Docker)
	uv run pytest -m "not integration"

ingest:  ## Télécharge MAL (commit figé) et AniList (cache daté)
	uv run python -m anime_mlops.data.ingest

train:  ## Ingestion, contrôles qualité, entraînement, traçabilité MLflow et promotion
	uv run python -m anime_mlops.training.train

status:  ## Versions du modèle, alias et tags dans le registre
	uv run python -m anime_mlops.training.promote status

promote:  ## Applique les règles de promotion (dernière version, ou VERSION=N)
	uv run python -m anime_mlops.training.promote promote $(if $(VERSION),--version $(VERSION))

rollback:  ## Retour arrière du champion (version remplacée, ou VERSION=N)
	uv run python -m anime_mlops.training.promote rollback $(if $(VERSION),--version $(VERSION))

replay:  ## Source simulée : rejoue le catalogue jour par jour (port 8100)
	uv run uvicorn anime_mlops.replay.app:app --port 8100

receiver:  ## Point de réception des lots poussés par la source (port 8101)
	uv run uvicorn anime_mlops.ingestion.receiver:app --port 8101

flows:  ## Planifie et exécute les flows Prefect (reste au premier plan)
	uv run python -m anime_mlops.flows.cli serve

daily:  ## Traite le lot du prochain jour (ou DAY=AAAA-MM-JJ)
	uv run python -m anime_mlops.flows.cli daily $(if $(DAY),--day $(DAY))

pipeline:  ## Flow d'entraînement complet : contrôles, entraînement, promotion
	uv run python -m anime_mlops.flows.cli training

feedback:  ## Feedback automatique : prédit les annonces, rattache les scores réels
	uv run python -m anime_mlops.flows.cli feedback

demo-incident:  ## Incident contrôlé : lot invalide, rejeté et mis en quarantaine
	uv run python -m anime_mlops.flows.cli daily --corrupt $(if $(DAY),--day $(DAY))

demo-nominal:  ## Démonstration API : modèle, cache, persistance, feedback et métriques
	uv run python -m anime_mlops.demos.service nominal --api-url $(API_URL)

demo-postgres-incident:  ## Arrête/rétablit PostgreSQL (CONFIRM_PROJECT requis)
	uv run python -m anime_mlops.demos.service postgres-outage --api-url $(API_URL) --project-name $(PROJECT_NAME) --confirm-project $(CONFIRM_PROJECT)

api:  ## Lance l'API de prédiction en développement
	uv run uvicorn anime_mlops.api.main:app --reload

register-mock-model:  ## Enregistre un modèle factice sous l'alias champion
	uv run python -m anime_mlops.api.register_mock_model

up:  ## Démarre la stack Docker (PostgreSQL, MLflow)
	docker compose up -d --build --wait

down:  ## Arrête la stack Docker (les volumes sont conservés)
	docker compose down

logs:  ## Suit les journaux de la stack
	docker compose logs -f

monitoring-logs:  ## Suit uniquement Prometheus, Grafana et l'exporteur Prefect
	docker compose logs -f prometheus grafana prefect-exporter

load-test:  ## Lance la matrice Locust 1/2/4 workers, cache désactivé/activé
	uv run --group load python -m load_tests.run_matrix $(LOAD_TEST_ARGS)
