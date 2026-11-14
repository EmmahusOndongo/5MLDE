# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.12-slim AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim AS runtime

ENV ANIME_CONFIG_FILE=/app/configs/config.yaml \
    PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system anime \
    && useradd --system --gid anime --create-home anime

WORKDIR /app

COPY --from=builder --chown=anime:anime /app/.venv /app/.venv
COPY --chown=anime:anime configs ./configs

USER anime

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=2)"]

CMD ["anime-api"]
