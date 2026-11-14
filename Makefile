.DEFAULT_GOAL := help
.PHONY: help setup lint format test ingest train promote rollback status up down logs

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

up:  ## Démarre la stack Docker (PostgreSQL, MLflow)
	docker compose up -d --build --wait

down:  ## Arrête la stack Docker (les volumes sont conservés)
	docker compose down

logs:  ## Suit les journaux de la stack
	docker compose logs -f
