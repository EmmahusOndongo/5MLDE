

# Projet MLOps — Prédiction du score MyAnimeList d'un anime avant diffusion

> Document de cadrage : sujet, objectifs, architecture et plan de réalisation.

---

## 1. Contexte

Chaque saison (hiver, printemps, été, automne), plusieurs dizaines de nouveaux animes sont annoncés. Les plateformes de streaming et les distributeurs doivent choisir lesquels acquérir, licencier ou mettre en avant, souvent **avant leur diffusion**. À ce moment-là, aucune note ni aucun retour du public n'existe : seules les caractéristiques de l'œuvre sont connues (studio, genres, tags, source, format, nombre d'épisodes, statut de suite…).

Le projet consiste à construire un **service de prédiction du score MyAnimeList** à partir de ces seules informations, et surtout à démontrer une **chaîne MLOps complète, reproductible et supervisée** autour de ce service.

> La performance prédictive n'est pas l'objectif : un modèle simple suffit. L'enjeu est la qualité opérationnelle de la chaîne.

---

## 2. Problématique

> **Comment estimer, de manière fiable et maintenable dans le temps, la note MyAnimeList d'un anime à partir des seules informations connues avant sa diffusion, en croisant plusieurs sources de données hétérogènes qui évoluent en continu, tout en garantissant un service de prédiction traçable, supervisé et résilient aux défaillances de ces sources ?**

Défis opérationnels associés :

- **Deux sources indépendantes** (MyAnimeList et AniList), avec leurs propres formats, rythmes de mise à jour et pannes possibles.
- **Une jointure à fiabiliser** : un anime peut exister d'un côté et pas de l'autre, ou présenter des informations contradictoires.
- **Des données qui évoluent** : nouveaux animes chaque saison et tendances qui changent, donc dérive du modèle.
- **Une vérité terrain différée** : le score réel n'apparaît qu'après diffusion, d'où la nécessité d'une boucle de feedback.
- **Des données imparfaites** : valeurs manquantes, champs inconnus, formats qui changent.

---

## 3. Sources de données

| Source | Rôle | Accès | Apport |
|---|---|---|---|
| **MyAnimeList** via [meesvandongen/anime-dataset](https://github.com/meesvandongen/anime-dataset) | Source principale | Fichier CSV sur GitHub (`data/anime-standalone.csv`) | Catalogue complet, features, **score `mean` (cible)**, statut de diffusion |
| **AniList** via l'API GraphQL (`https://graphql.anilist.co`) | Enrichissement | API publique gratuite, sans clé | Tags détaillés, relations (suite / préquel), format de source |

### Provenance et conditions d'utilisation

- **MyAnimeList** : le dépôt `meesvandongen/anime-dataset` est un projet tiers non officiel, alimenté via l'API MyAnimeList et mis à jour par incréments chaque jour (rafraîchissement complet en environ une semaine). Aucune licence n'est indiquée sur le dépôt : les données restent la propriété de MyAnimeList et sont utilisées ici dans un cadre strictement pédagogique.
- **AniList** : API publique soumise à ses conditions d'utilisation et à une limite de requêtes par minute (à vérifier dans la documentation officielle au moment de l'implémentation).
- **Aucun dataset n'est versionné dans le dépôt Git** : les données sont téléchargées par script.

### Stratégie de versionnement des données

- **MAL** : téléchargement via l'URL `raw.githubusercontent.com/meesvandongen/anime-dataset/<hash_commit>/data/anime-standalone.csv`. On ne pointe **jamais** sur `main` pour un entraînement ; le hash du commit est la version des données.
- **AniList** : chaque extraction est stockée en cache local (Parquet daté). La date d'extraction est la version des données.
- Les deux versions sont enregistrées dans **MLflow** pour chaque entraînement.

### Clé de jointure

AniList expose le champ `idMal` (identifiant MyAnimeList) pour chaque anime. La jointure se fait directement sur cet identifiant, sans rapprochement fragile par titre.

---

## 4. Objectifs

### Objectif général

Mettre en place une chaîne MLOps complète, de l'ingestion multi-sources à la supervision du service, démontrée de bout en bout en fonctionnement nominal et lors d'incidents contrôlés.

### Objectifs spécifiques

1. **Ingestion multi-sources reproductible** : récupérer MAL (commit figé) et AniList (extraction datée, en cache), chaque version étant traçable.
2. **Qualité contrôlée à chaque étape** : valider chaque source séparément, puis le dataset fusionné, et bloquer l'entraînement en cas de non-conformité.
3. **Résilience aux sources** : gérer l'indisponibilité d'AniList par retry, fallback sur le cache et alerte, sans casser la chaîne.
4. **Pipeline automatisé** : enchaîner ingestion, validation, jointure, entraînement et promotion via un flow planifié, avec suivi des exécutions.
5. **Traçabilité totale** : relier chaque prédiction à une version de modèle, elle-même liée à ses versions de données, ses paramètres et ses métriques.
6. **Mise en service contrôlée** : promouvoir un modèle uniquement s'il fait au moins aussi bien que le modèle en production, avec retour arrière en une commande.
7. **Service exploitable** : API de prédiction unitaire et par lot, avec validation des entrées, gestion des erreurs et identification de la version utilisée.
8. **Boucle de feedback réelle** : collecter automatiquement les scores réels des animes une fois diffusés, et mesurer l'erreur en conditions réelles.
9. **Supervision** : dashboard couvrant les traitements, l'usage du service, la latence, les erreurs et la performance réelle du modèle.
10. **Dimensionnement justifié** : tests de charge, comparaison de configurations et recommandation chiffrée des ressources.

### Hors périmètre

- Rechercher le meilleur score prédictif possible.
- Remplacer le jugement humain : l'outil est une **aide à la décision**.
- Utiliser des informations connues uniquement après diffusion (popularité, nombre de membres, rang) : elles fausseraient le cas d'usage réel (fuite de données).

---

## 5. Architecture

```
                    ┌──────────────── Pipeline Prefect (planifié chaque semaine) ────────────────┐
                    │                                                                            │
  MyAnimeList ──► Validation MAL ─────┐                                                         │
  (commit figé)     (Pandera)          ├──► Jointure idMal ──► Validation ──► Entraînement ──► Registre MLflow
  AniList ──────► Validation AniList ─┘    + contrôles        fusionnée       + évaluation      (alias champion)
  (cache daté)      (Pandera)                                                                    │
                    └────────────────────────────────────────────────────────────────────────────┘
                                                                                                 │
                                                                                                 ▼
  Client / Locust ──► API FastAPI ──► PostgreSQL (prédictions, versions, durées, erreurs, feedbacks)
                         │                         │
                         └──► Prometheus ──────────┴──► Grafana (dashboard)
```

L'API charge toujours le modèle portant l'alias **`champion`** dans le registre MLflow.

---

## 6. Stack technique

| Besoin | Outil | Justification |
|---|---|---|
| Langage | Python 3.11+ | Écosystème ML et MLOps |
| Dépendances | `uv` + `pyproject.toml` + lockfile | Installation reproductible et rapide |
| Configuration | YAML + Pydantic Settings | Configuration séparée du code, surcharge par variables d'environnement |
| Contrôles qualité | Pandera | Schémas déclaratifs, exceptions bloquantes |
| Orchestration | Prefect | UI locale gratuite, planification, retries, historique des runs, notifications |
| Suivi et registre | MLflow | Paramètres, métriques, artefacts, versions, alias, retour arrière |
| Modèle | scikit-learn (`Pipeline` + Gradient Boosting ou Random Forest) | Simple, sérialisable, prétraitement inclus dans l'artefact |
| API | FastAPI + Uvicorn | Validation Pydantic native, documentation Swagger automatique |
| Persistance | PostgreSQL | Traces d'utilisation et feedbacks |
| Métriques | Prometheus (`prometheus-fastapi-instrumentator`) | Latence, débit, erreurs de l'API |
| Dashboard | Grafana | Sources Prometheus et PostgreSQL dans un même tableau de bord |
| Tests de charge | Locust | Scénarios en Python, export CSV |
| Tests | pytest | Tests unitaires et d'intégration |
| CI | GitHub Actions | Lint et tests à chaque push |
| Déploiement local | Docker Compose | Toute la stack en une commande |

Chaque choix est à justifier brièvement dans la synthèse finale.

---

## 7. Modèle et données

### Tâche

**Régression** : prédire le score `mean` (0 à 10) d'un anime.

### Jeu d'entraînement

Animes au statut `finished_airing` disposant d'un `mean` renseigné.

### Features autorisées (connues avant diffusion)

| Source | Features |
|---|---|
| MAL | `media_type`, `source`, `num_episodes`, `average_episode_duration`, `rating`, `nsfw`, `genres` (multi-hot), `studios` (top N + « autre »), `start_season` (année et saison), jour de diffusion |
| AniList | Tags (top N en multi-hot), `is_sequel` |
| Dérivée | `prequel_score` : score MAL de la saison précédente si l'anime est une suite |

### Features exclues (fuite de données)

`rank`, `popularity`, `num_list_users`, `num_scoring_users`, et plus généralement tout indicateur qui dépend de la réception du public.

### Métriques suivies

- MAE (métrique de décision pour la promotion)
- RMSE
- MAE réelle calculée sur les feedbacks collectés en production

---

## 8. Contrôles qualité

| Étape | Contrôle | Bloquant si… |
|---|---|---|
| Source MAL | Colonnes attendues, types, `mean` entre 0 et 10, IDs uniques | Colonne manquante, type incorrect, ID dupliqué |
| Source AniList | Schéma de réponse, `idMal` présent, tags bien formés | Réponse vide ou format inattendu |
| Jointure | Taux de correspondance MAL ↔ AniList | Moins de 90 % des animes MAL retrouvés (seuil configurable) |
| Dataset fusionné | Cohérence du nombre d'épisodes entre les sources, taux de valeurs manquantes par colonne, taille minimale du lot | Seuils dépassés (définis dans `configs/`) |

En cas d'échec : **arrêt du flow, aucun entraînement, alerte émise et rapport de validation archivé**.

---

## 9. Règles de mise en service et retour arrière

- Chaque entraînement produit une **nouvelle version** du modèle dans le registre MLflow, avec ses tags : hash du commit MAL, date d'extraction AniList, taille du jeu, métriques.
- **Règle de promotion** : le nouveau modèle reçoit l'alias `champion` uniquement si sa MAE sur le jeu de test de référence est **inférieure ou égale** à celle du champion actuel. Sinon, il reçoit l'alias `challenger`.
- **Retour arrière** : une commande (`make rollback VERSION=N`) remet l'alias `champion` sur une version précédente. L'API recharge le modèle via `POST /admin/reload` ou au redémarrage.

---

## 10. API de prédiction

| Endpoint | Rôle |
|---|---|
| `POST /predict` | Prédiction pour un anime |
| `POST /predict/batch` | Prédiction pour une liste d'animes |
| `POST /feedback` | Enregistrement du score réel associé à un `request_id` |
| `GET /health` | État du service et de ses dépendances (modèle, base de données) |
| `GET /model` | Version et métadonnées du modèle servi |
| `GET /metrics` | Métriques Prometheus |
| `POST /admin/reload` | Rechargement du modèle `champion` |

Chaque réponse de prédiction contient : `prediction`, `model_version`, `request_id`.

Les entrées sont validées par Pydantic (types, bornes, valeurs autorisées). Les erreurs renvoient des codes HTTP explicites (422 pour une entrée invalide, 503 si le modèle n'est pas disponible).

### Persistance (PostgreSQL)

- **`predictions`** : `request_id`, horodatage, entrées, prédiction, version du modèle, durée de traitement, statut, message d'erreur éventuel.
- **`feedbacks`** : `request_id` ou `anime_id`, score réel, source du feedback (manuel ou automatique), horodatage.

### Feedback automatique

Le flow hebdomadaire repère les animes prédits alors qu'ils étaient `not_yet_aired`, qui disposent désormais d'un `mean` dans la dernière version de MAL, et enregistre ce score réel comme feedback.

---

## 11. Supervision

Dashboard Grafana avec au minimum :

- **Traitements** : statut et durée des derniers runs du pipeline, lots rejetés.
- **Usage** : requêtes par minute, répartition unitaire / lot.
- **Performance du service** : latences p50 et p95, taux d'erreur par endpoint.
- **Modèle** : version servie, répartition des prédictions par version, MAE réelle sur les feedbacks dans le temps.

Alertes : échec de validation, échec du pipeline, taux d'erreur de l'API au-delà d'un seuil.

---

## 12. Tests de charge et dimensionnement

**Protocole :**

1. Scénario Locust réaliste : 80 % de `/predict`, 15 % de `/predict/batch` (lots de 20), 5 % de `/feedback`.
2. Montée en charge progressive (par exemple 10, 50, 100, 200 utilisateurs simultanés).
3. Configurations comparées : 1, 2 et 4 workers Uvicorn, avec des limites CPU et mémoire Docker fixées.
4. Mesures : débit (req/s), latences p50 et p95, taux d'erreur, utilisation CPU et mémoire.

**Livrable :** un tableau comparatif et une recommandation chiffrée, par exemple :
« 2 workers sur 1 vCPU / 512 Mo tiennent X req/s avec une latence p95 inférieure à Y ms ; au-delà, passer à 4 workers sur 2 vCPU. »

---

## 13. Démonstrations à préparer

| Scénario | Déroulé attendu |
|---|---|
| **Fonctionnement nominal** | Nouveau commit MAL → validation OK → jointure → entraînement → promotion → prédiction avec la nouvelle version → feedback → visible dans Grafana |
| **Lot de données invalide** | Commit MAL modifié (scores à 15, colonne renommée) → validation en échec → entraînement bloqué → alerte visible dans Prefect |
| **Dépendance indisponible (AniList)** | API AniList inaccessible → retries → fallback sur le cache → avertissement et alerte → pipeline terminé |
| **Base de données indisponible** | PostgreSQL arrêté → l'API continue de prédire, logue l'erreur, `/health` signale l'état dégradé |
| **Retour arrière** | Promotion d'un mauvais modèle → `make rollback` → l'API sert immédiatement la version précédente |

---

## 14. Structure du dépôt

```
anime-score-mlops/
├── configs/
│   ├── config.yaml            # chemins, URLs, seuils, hyperparamètres
│   └── promotion.yaml         # règles de mise en service
├── src/anime_mlops/
│   ├── data/
│   │   ├── sources/
│   │   │   ├── mal.py         # téléchargement par hash de commit
│   │   │   └── anilist.py     # requêtes GraphQL par lots + cache Parquet
│   │   ├── validate.py        # schémas Pandera par source et fusionné
│   │   └── merge.py           # jointure sur idMal + contrôles inter-sources
│   ├── features/
│   │   └── build.py           # construction des features
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── promote.py         # promotion et rollback
│   ├── flows/
│   │   └── pipeline.py        # flow Prefect principal
│   └── api/
│       ├── main.py
│       ├── schemas.py
│       ├── db.py
│       └── metrics.py
├── tests/
│   ├── unit/
│   └── integration/
├── load_tests/
│   ├── locustfile.py
│   └── results/
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── .env.example               # jamais de secrets dans le dépôt
└── README.md
```

---

## 15. Plan de réalisation

### Étape 1 — Fondations
- [ ] Initialiser le dépôt, `pyproject.toml`, `uv`, linter (ruff) et pytest.
- [ ] Créer la structure de dossiers et la configuration YAML.
- [ ] Écrire le `docker-compose.yml` de base (PostgreSQL, MLflow).

### Étape 2 — Données
- [ ] Script d'ingestion MAL par hash de commit.
- [ ] Client AniList GraphQL par lots, avec cache Parquet et gestion des limites de requêtes.
- [ ] Schémas Pandera par source.
- [ ] Jointure sur `idMal` et contrôles inter-sources.
- [ ] Tests unitaires de la validation (données valides et invalides).

### Étape 3 — Modèle et traçabilité
- [ ] Construction des features (hors features à fuite).
- [ ] Entraînement scikit-learn et évaluation.
- [ ] Journalisation MLflow : paramètres, métriques, versions des données, artefacts.
- [ ] Promotion par alias et commande de rollback.

### Étape 4 — Orchestration
- [ ] Flow Prefect enchaînant toutes les étapes.
- [ ] Planification hebdomadaire.
- [ ] Retries, fallback AniList, gestion des échecs et notifications.

### Étape 5 — API et persistance
- [ ] Endpoints FastAPI, validation et gestion des erreurs.
- [ ] Persistance des prédictions et des feedbacks dans PostgreSQL.
- [ ] Collecte automatique des feedbacks dans le flow.
- [ ] Tests d'intégration de l'API.

### Étape 6 — Supervision
- [ ] Instrumentation Prometheus de l'API.
- [ ] Dashboard Grafana (traitements, usage, performances, erreurs, MAE réelle).
- [ ] Alertes.

### Étape 7 — Charge et dimensionnement
- [ ] Scénario Locust.
- [ ] Campagne de tests sur plusieurs configurations.
- [ ] Analyse et recommandation chiffrée.

### Étape 8 — Finalisation
- [ ] CI GitHub Actions.
- [ ] README complet (sujet, architecture, récupération des données, commandes).
- [ ] Synthèse des choix et résultats OPS.
- [ ] Répétition des démonstrations (nominal + incidents).

---

## 16. Livrables

1. **Dépôt Git** : code, tests, configurations, sans secrets ni dataset volumineux.
2. **README** : sujet, architecture, récupération des données, commandes de reproduction.
3. **Synthèse OPS** : décisions techniques, preuves d'intégration (captures, logs), résultats des tests de charge, dimensionnement recommandé, limites connues.

### Commandes cibles (Makefile)

```bash
make setup          # installation des dépendances
make up             # lancement de la stack Docker
make ingest         # ingestion et validation des sources
make train          # entraînement + promotion éventuelle
make pipeline       # exécution complète du flow Prefect
make serve          # lancement de l'API
make test           # tests unitaires et d'intégration
make loadtest       # campagne Locust
make rollback VERSION=N
make demo-incident  # injection d'un lot invalide
```

---

## 17. Correspondance avec la grille d'évaluation

| Critère | Poids | Éléments du projet |
|---|---|---|
| Qualité du code | 20 % | Code modulaire, configuration séparée, tests pytest, CI, lint |
| Reproductibilité | 10 % | `uv` + lockfile, Docker Compose, données versionnées par commit et date, Makefile |
| Intégration du cycle MLOps | 20 % | Flow Prefect de bout en bout, artefacts MLflow consommés par l'API |
| Fiabilité et traçabilité | 20 % | Contrôles Pandera bloquants, fallback AniList, versions et tags MLflow, rollback |
| Exploitation du service | 20 % | API FastAPI, persistance PostgreSQL, feedbacks automatiques, Grafana |
| Dimensionnement | 10 % | Protocole Locust, comparaison de configurations, recommandation chiffrée |

---

## 18. Limites connues

- **Dépendance à un dépôt tiers** : si `meesvandongen/anime-dataset` cesse d'être mis à jour, le flux de nouvelles données s'arrête (un instantané figé reste utilisable pour la démonstration).
- **Données non officielles** : pas de licence explicite, usage pédagogique uniquement.
- **Feedback lent** : le score réel d'un anime n'est stable que plusieurs semaines après sa diffusion.
- **Modèle volontairement simple** : les performances ne sont pas optimisées.
- **Exécution locale** : le dimensionnement recommandé est mesuré sur une machine de développement et doit être extrapolé avec prudence.
