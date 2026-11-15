#!/bin/sh
set -eu

: "${APP_DATABASE_USER:?APP_DATABASE_USER manquant}"
: "${GRAFANA_DATABASE_USER:?GRAFANA_DATABASE_USER manquant}"
: "${GRAFANA_DATABASE_PASSWORD:?GRAFANA_DATABASE_PASSWORD manquant}"

psql \
  --set=ON_ERROR_STOP=1 \
  --set=app_user="$APP_DATABASE_USER" \
  --set=grafana_user="$GRAFANA_DATABASE_USER" \
  --set=grafana_password="$GRAFANA_DATABASE_PASSWORD" <<'SQL'
SELECT format(
  'CREATE ROLE %I LOGIN PASSWORD %L',
  :'grafana_user',
  :'grafana_password'
)
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'grafana_user')
\gexec

SELECT format(
  'ALTER ROLE %I WITH LOGIN PASSWORD %L',
  :'grafana_user',
  :'grafana_password'
)
\gexec

SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'grafana_user')
\gexec
SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'grafana_user')
\gexec
SELECT format('GRANT SELECT ON ALL TABLES IN SCHEMA public TO %I', :'grafana_user')
\gexec
SELECT format(
  'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public GRANT SELECT ON TABLES TO %I',
  :'app_user',
  :'grafana_user'
)
\gexec
SQL
