-- Exécuté une seule fois, à la première initialisation du volume PostgreSQL.
-- La base "anime" (applicative) est créée par POSTGRES_DB ; MLflow a sa propre base.
CREATE DATABASE mlflow;
