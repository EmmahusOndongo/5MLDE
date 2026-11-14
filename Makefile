.DEFAULT_GOAL := help
.PHONY: help setup lint format test api register-mock-model up down logs

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
