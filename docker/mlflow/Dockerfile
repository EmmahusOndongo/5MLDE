FROM python:3.12-slim

# Versions figées : le serveur de suivi doit être reproductible.
RUN pip install --no-cache-dir mlflow==3.18.0 psycopg2-binary==2.9.10

EXPOSE 5000
