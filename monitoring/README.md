# Supervision locale

La supervision est livrée **as code** : Docker Compose démarre Prometheus,
Grafana et un exporteur léger pour Prefect. Les sources de données, le dashboard
et les règles d'alerte sont provisionnés au démarrage ; aucune création manuelle
dans l'interface n'est nécessaire.

## Démarrage

1. Copier `.env.example` vers `.env` et remplacer tous les mots de passe
   `change-me`.
2. Vérifier qu'une version du modèle `anime-score-regressor` porte l'alias
   `champion` dans MLflow.
3. Démarrer la stack :

   ```bash
   docker compose up -d --build --wait
   ```

4. Ouvrir :

   - Grafana : <http://localhost:3000> ;
   - Prometheus : <http://localhost:9090> ;
   - Prefect, après intégration de A8 : <http://localhost:4200>.

L'identifiant Grafana vient de `GRAFANA_ADMIN_USER` et le mot de passe de
`GRAFANA_ADMIN_PASSWORD`. Le service `grafana-db-init` crée ou remet à jour un
compte PostgreSQL limité à la lecture. Le mot de passe de ce compte n'est jamais
stocké dans Git.

## Ce que montre le dashboard

Le dashboard **Anime MLOps — Exploitation** couvre :

- disponibilité, débit et requêtes en vol de l'API ;
- latences p50/p95, erreurs par endpoint et codes HTTP ;
- version du modèle effectivement servie et débit par version ;
- prédictions comparées aux scores réels, MAE observée et couverture des
  feedbacks ;
- état, fraîcheur et durée des dernières exécutions Prefect.

Les graphiques de qualité interrogent directement les tables `predictions` et
`feedbacks`. Ils fonctionnent déjà avec `/feedback`. Quand A9 écrit des retours
de source `automatic` via `save_feedback`, ils apparaissent sans modification du
dashboard.

L'exporteur Prefect interroge `PREFECT_API_URL` toutes les 15 secondes. Si A8
n'est pas encore présent ou si Prefect tombe, l'exporteur continue à répondre et
positionne `anime_prefect_up` à `0` : la panne reste donc visible dans Prometheus
et Grafana.

## Alertes provisionnées

Trois règles sont visibles dans **Alerting > Alert rules** :

| Règle | Seuil | Temporisation |
|---|---:|---:|
| Taux d'erreurs API | supérieur à 5 % sur 5 min | 2 min |
| Latence API p95 | supérieure à 1 s sur 5 min | 2 min |
| Prefect indisponible | `anime_prefect_up < 1` | 2 min |

Elles sont volontairement livrées sans destinataire externe : une adresse mail
ou un webhook est un secret d'exploitation propre à l'environnement. Pour une
démo locale, leur changement d'état est visible dans Grafana et annoté sur le
dashboard. En environnement partagé, ajouter un point de contact Grafana et une
politique de notification sans versionner le secret.

## Incident contrôlé

Pour démontrer la supervision de l'orchestrateur, arrêter uniquement le service
Prefect de cette stack :

```bash
docker compose stop prefect
```

Après environ deux minutes, la tuile Prefect devient rouge et la règle
`anime-prefect-unavailable` passe en alerte. Le rétablissement se démontre avec :

```bash
docker compose start prefect
```

Pour l'API, envoyer pendant deux minutes une majorité de requêtes invalides sur
`/predict` déclenche la règle de taux d'erreurs. Les réponses `422` restent aussi
ventilées par endpoint dans le dashboard.

## Diagnostic

```bash
docker compose ps
docker compose logs prometheus grafana prefect-exporter
curl http://localhost:9090/-/ready
curl http://localhost:3000/api/health
```

Une absence de séries Prefect avec `anime_prefect_up 0` indique que l'exporteur
fonctionne mais ne joint pas encore l'orchestrateur. Une erreur de requête sur les
panels SQL indique généralement que l'API n'a pas encore créé ses tables ou que
les identifiants `GRAFANA_DATABASE_*` ne correspondent pas au compte initialisé.
