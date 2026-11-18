# Synthèse des choix et résultats OPS

Cette synthèse complète le [README](README.md) : elle résume les décisions techniques,
ce qu'on a vérifié, les résultats des tests de charge, le dimensionnement recommandé et les
limites du projet.

## 1. Décisions techniques

| Sujet | Choix | Pourquoi | Compromis |
|---|---|---|---|
| Environnement | uv + `uv.lock`, Python 3.12, configuration YAML surchargeable par variables d'environnement | installation identique en local, dans Docker et en CI | le lockfile et les images doivent être mis à jour ensemble |
| Données | MAL figé par commit et SHA-256, AniList extrait par lots et mis en cache daté | savoir exactement sur quelles données un modèle a été entraîné, pouvoir continuer si AniList tombe | le cache peut être un peu ancien |
| Arrivée des données | API de rejeu qui découpe le dataset en lots journaliers, récupérés en pull ou reçus en push | simuler un flux réel à partir d'un dataset figé | les dates sont simulées |
| Qualité | Great Expectations, contrôles bloquants par source, après fusion et sur chaque lot journalier, avec des règles métier ajoutées | une donnée invalide n'atteint jamais l'entraînement | les seuils sont à recalibrer si les données changent |
| Orchestration | Prefect 3 : flow quotidien, flow d'entraînement, flow de feedback | retries par tâche, parallélisme, planification, historique, alertes en cas d'échec, un seul conteneur | un service de plus à faire tourner |
| Modèle | pipeline scikit-learn (prétraitement + Random Forest), graine fixe | le prétraitement voyage avec le modèle, l'API envoie les champs bruts | modèle simple, ce n'est pas l'objet du projet |
| Suivi et registre | MLflow : paramètres, versions des données et du code, métriques, artefacts ; alias `champion` / `challenger` | l'API ne dépend pas d'un numéro de version, le retour arrière est un déplacement d'alias | la promotion dépend de métriques bien journalisées |
| Mise en service | règles écrites dans `configs/promotion.yaml` : données validées, meilleur que la moyenne, pas moins bon que le champion | décision automatique, explicable et tracée en tags | pas de seuil absolu de performance |
| API | FastAPI, schémas Pydantic stricts, prédiction en ligne et par lot | erreurs contrôlées (422, 503), contrat OpenAPI | pas d'authentification (périmètre local) |
| Persistance | PostgreSQL pour MLflow, les prédictions et les feedbacks ; écriture « best effort » pour `/predict` | une panne de base ne coupe pas les prédictions | une trace peut être perdue pendant la panne |
| Cache | LRU/TTL en mémoire, clé incluant la version du modèle, vidé au rechargement | débit ×5 sans nouvelle infrastructure, pas de réponse d'un ancien modèle | un cache par worker, pas partagé |
| Monitoring | Prometheus + Grafana provisionnés depuis le dépôt, exporteur pour l'état de Prefect, compte PostgreSQL en lecture seule | rien à configurer à la main, alertes versionnées | rétention locale seulement |
| Déploiement | Docker Compose, images en versions fixes | toute la stack se lance en une commande | pas de haute disponibilité |

## 2. Gestion des échecs

**Données et entraînement**

- Un contrôle qualité en échec arrête l'entraînement ; le rapport est archivé et le
  champion en place continue de servir.
- Un lot journalier invalide part en quarantaine, le flow échoue et une alerte critique est
  émise. Seuls les lots acceptés servent à l'entraînement.
- La récupération du lot est réessayée 3 fois. AniList est réessayé puis remplacé par le
  dernier cache, avec une alerte `warning` et le tag `anilist_is_fallback` dans MLflow.
- Tout flow en échec ou planté déclenche une alerte (fichier `alerts.jsonl` et webhook
  optionnel).

**Service**

- Entrée invalide : 422 avant l'appel au modèle.
- Modèle ou registre indisponible : 503 contrôlé, `/health` indique la cause, l'API reste
  en vie et peut recharger le modèle avec `/admin/reload`.
- PostgreSQL arrêté : `/predict` répond avec `persisted=false`, `/feedback` renvoie 503 car
  le lien avec la prédiction ne peut pas être garanti.
- Feedback automatique : si l'API ou la base est indisponible, le lot reste accepté, une
  alerte est émise et le passage suivant rattrape ce qui manque, sans doublon.
- Retour arrière : `make rollback` déplace l'alias `champion`, puis `/admin/reload`.

**Monitoring** : alertes Grafana sur le taux d'erreur (> 5 % sur 5 min), la latence p95
(> 1 s) et l'indisponibilité de Prefect. L'exporteur reste en ligne quand Prefect tombe et
publie `anime_prefect_up = 0`, ce qui distingue une panne de Prefect d'une panne du
monitoring.

## 3. Preuves d'intégration

Vérifications faites le 17/11/2026 sur la branche `main` (toutes les parties fusionnées).

**Qualité du code** : `ruff check` et `ruff format --check` sans erreur, 241 tests
unitaires réussis. La CI GitHub Actions lance les mêmes vérifications à chaque push.

**Stack complète** : `docker compose up -d --build --wait` dans un projet Compose isolé, les
8 services démarrent et sont sains (postgres, mlflow, prefect, api, prefect-exporter,
prometheus, grafana, et grafana-db-init qui se termine normalement). Prometheus voit ses
deux cibles (`anime-api`, `prefect-exporter`) en `up`. Grafana a ses deux sources de
données saines, le dashboard « Anime MLOps — Exploitation » et ses trois règles d'alerte.

**Scénario nominal** (`make demo-nominal`), sortie obtenue :

```json
{
  "model_version": "1",
  "prediction": 7.0,
  "first_cached": false,
  "second_cached": true,
  "feedback_source": "manual",
  "metrics_exposed": true
}
```

**Incident PostgreSQL** (`make demo-postgres-incident`), sortie obtenue :

```json
{
  "project_name": "anime-verif",
  "degraded_status": "degraded",
  "prediction_while_down": 7.0,
  "persisted_while_down": false,
  "feedback_error": "database_unavailable",
  "recovered_status": "ok",
  "persisted_after_recovery": true
}
```

Ces deux scénarios ont été joués avec un modèle factice enregistré dans la stack isolée.

**Entraînement sur les vraies données** :

- MAL : 25 752 animes (commit `136f9de5ee2d9b4863330d506d3439d9925f800a`) ; AniList :
  19 344 animes après dédoublonnage ;
- 19 340 animes notés pour l'entraînement ;
- MAE de test 0,381 contre 0,700 pour la prédiction constante, R² 0,65 ;
- registre MLflow : une version `champion`, une version `challenger` (MAE 0,396, entraînée
  par le flow Prefect sur l'historique et les lots acceptés, donc non promue) et une version
  `archived`.

**Flows Prefect** sur le serveur Prefect réel : lots journaliers acceptés, lot corrompu mis
en quarantaine avec alerte critique et code de sortie 1, relance idempotente, déploiements
quotidien et hebdomadaire enregistrés.

**Feedback automatique**, trois jours rejoués avec l'API et le vrai champion :

| Jour | Annonces | Scores révélés | Prédictions | Feedbacks |
|---|---:|---:|---:|---:|
| 01/01/2025 | 6 | 0 | 6 | 0 |
| 02/01/2025 | 5 | 4 | 5 | 4 |
| 03/01/2025 | 5 | 5 | 5 | 5 |

Résultat en base : 16 prédictions tracées, 9 feedbacks `automatic`, aucune alerte. L'erreur
moyenne observée (0,175) n'est pas représentative : ce champion a été entraîné sur tout le
catalogue, animes de 2025 compris.

## 4. Résultats des tests de charge

Protocole (`load_tests/`) : Locust en mode headless sur `POST /predict`, sans temps
d'attente, 90 % d'entrées récurrentes et 10 % d'entrées uniques, persistance PostgreSQL
vérifiée à chaque réponse, API recréée et tables vidées entre deux scénarios. Machine :
12 CPU logiques, Docker Desktop. Mesures brutes :
[`load_tests/mesures-2026-11-15.json`](load_tests/mesures-2026-11-15.json).

**50 utilisateurs simultanés, 60 secondes**

| Workers | Cache | req/s | p95 | p99 | CPU max | RAM max | Échecs |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | non | 14,3 | 3 900 ms | 4 900 ms | 1,48 cœur | 310 Mio | 0 |
| 1 | oui | 81,6 | 990 ms | 3 100 ms | 1,55 cœur | 292 Mio | 0 |
| 2 | non | 27,7 | 2 300 ms | 2 400 ms | 2,53 cœurs | 614 Mio | 0 |
| 2 | oui | 146,2 | 710 ms | 1 100 ms | 2,68 cœurs | 620 Mio | 0 |
| 4 | non | 51,8 | 1 600 ms | 1 800 ms | 4,76 cœurs | 1 111 Mio | 0 |
| 4 | oui | 260,9 | 520 ms | 780 ms | 4,88 cœurs | 1 104 Mio | 0 |

Le cache multiplie le débit par 5 à 5,7 et réduit le p95 de 67 à 75 %. Même en cas de hit,
la requête est validée et tracée en base : le gain est volontairement plus faible qu'un
cache pur, pour garder l'audit complet.

**Avec cache, selon le nombre d'utilisateurs (30 secondes)**

| Utilisateurs | 1 worker | 2 workers | 4 workers |
|---:|---|---|---|
| 5 | 89,6 req/s, p95 180 ms | 146,0 req/s, p95 130 ms | 164,6 req/s, p95 110 ms |
| 10 | 89,2 req/s, p95 300 ms | 156,4 req/s, p95 200 ms | 200,7 req/s, p95 140 ms |
| 25 | 79,1 req/s, p95 630 ms | 156,3 req/s, p95 390 ms | 276,3 req/s, p95 290 ms |

La mémoire augmente presque linéairement avec le nombre de workers, car chacun charge son
modèle et son cache.

## 5. Dimensionnement recommandé

Objectif retenu (aucun n'était imposé) : zéro échec, persistance confirmée, p95 sous 1 s
et 30 % de marge sur le débit mesuré. Les ressources indiquées concernent uniquement le
conteneur de l'API, avec au moins 25 % de marge sur les maxima observés.

| Utilisateurs simultanés | Configuration de l'API | Allocation | Mesure de référence |
|---:|---|---|---|
| 1 à 5 | 1 worker, cache activé | 2 vCPU, 512 Mo | 89,6 req/s, p95 180 ms |
| 6 à 25 | 2 workers, cache activé | 4 vCPU, 1 Go | 156,3 req/s, p95 390 ms |
| 26 à 50 | 4 workers, cache activé | 8 vCPU, 2 Go | 260,9 req/s, p95 520 ms |

Le cache doit rester activé : sans lui, 4 workers à 50 utilisateurs ne tiennent que
51,8 req/s avec un p95 de 1,6 s. Au-delà de 50 utilisateurs, il faudrait plusieurs
réplicas derrière un répartiteur de charge, avec un cache partagé, puis refaire une
campagne de mesures.

## 6. Limites connues

1. **Modèle des tests de charge** : il avait le pipeline et les hyperparamètres finaux mais
   était entraîné sur 4 000 lignes synthétiques. Les temps de réponse devraient être
   proches, mais la campagne est à rejouer avec le champion final.
2. **Durée des tests** : des scénarios de 30 à 60 s mesurent la saturation, pas les fuites
   lentes ; un test d'endurance de 15 à 30 min serait nécessaire avant une vraie mise en
   service.
3. **Une seule machine** : Locust, Docker et l'API partageaient le même hôte Windows. Les
   chiffres ne sont pas extrapolés à un autre environnement.
4. **Comparaison des modèles** : chaque version est évaluée sur son propre jeu de test ; un
   jeu de test commun et figé rendrait la comparaison plus juste.
5. **Fuite possible côté AniList** : certains tags sont ajoutés par la communauté après la
   diffusion.
6. **Couverture AniList** : environ un quart du catalogue MAL (surtout clips et pubs) n'est
   pas sur AniList ; le seuil de jointure est fixé à 80 %.
7. **Cache local** : un cache par worker ; plusieurs réplicas demanderaient un cache
   partagé.
8. **Flows hors Docker** : l'API de rejeu et l'exécution des flows Prefect tournent sur
   l'hôte.
9. **Sécurité** : prévu pour un usage local, sans authentification ni TLS. Une exposition
   réelle demanderait un reverse proxy TLS, une authentification (surtout pour
   `/admin/reload`), la rotation des secrets, des sauvegardes et des destinataires
   d'alerte configurés.
