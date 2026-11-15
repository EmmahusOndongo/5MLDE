# Synthèse des choix et résultats OPS

## Résumé exécutif

Le projet industrialise la prédiction du score MyAnimeList d'un anime avant sa diffusion.
Le modèle est volontairement simple : la valeur du projet réside dans une chaîne complète,
reproductible et observable, depuis l'arrivée des données jusqu'au feedback sur une
prédiction servie.

La solution retenue repose sur :

- `uv`, une configuration YAML et du code Python modulaire pour la reproductibilité ;
- Great Expectations pour des contrôles qualité bloquants ;
- Prefect pour l'orchestration, les reprises et le suivi des exécutions ;
- MLflow pour les paramètres, métriques, artefacts, versions et alias du modèle ;
- FastAPI et Pydantic pour le service de prédiction ;
- PostgreSQL pour les traces de prédiction et les scores observés ;
- Prometheus et Grafana pour les métriques, dashboards et alertes ;
- Locust et `docker stats` pour les mesures de charge et de ressources.

Le profil recommandé pour la démonstration est **2 workers Uvicorn, cache activé,
4 vCPU et 1 Gio de RAM pour l'API**. Il a tenu 25 utilisateurs simultanés à
156,3 requêtes/s, avec un p95 de 390 ms et zéro échec. Le détail est conservé dans
[`dimensionnement.md`](dimensionnement.md).

## Architecture et circulation des artefacts

```text
MAL (commit figé) ─┐
                   ├─ contrôles ─ jointure/features ─ entraînement ─ MLflow Registry
AniList (cache) ───┘        │                                  │          │
                            └─ rapports archivés               │     alias champion
                                                               │          │
Rejeu journalier ─ Prefect ─ lots acceptés/quarantaine         │          ▼
                                                               │      FastAPI
                                                               │       │   │
                                                  prédictions ─┘       │   └─ Prometheus
                                                                       ▼          │
                                                                 PostgreSQL       ▼
                                                              prédictions/feedbacks
                                                                       │          │
                                                                       └── Grafana
```

Deux identifiants relient l'exploitation au modèle : chaque réponse expose la version
MLflow servie, et chaque trace PostgreSQL conserve la version, le run MLflow, la durée,
l'entrée brute et l'éventuelle erreur. Le `prediction_id` relie ensuite la prédiction à
un feedback manuel ou automatique.

## Décisions techniques

| Sujet | Décision | Justification OPS | Compromis assumé |
|---|---|---|---|
| Environnement | `uv.lock`, Python 3.11/3.12, configuration séparée | installation déterministe, mêmes commandes en local et en CI | les images et le lockfile doivent être mis à jour ensemble |
| Données | MAL figé par commit, AniList daté et mis en cache | provenance et répétabilité, repli possible si AniList tombe | le cache peut rendre l'enrichissement temporairement plus ancien |
| Qualité | contrôles bloquants par source puis après fusion | un lot invalide n'atteint jamais l'entraînement | les seuils devront être recalibrés si la distribution change |
| Orchestration | Prefect 3, tâches indépendantes en parallèle, retries ciblés | suivi natif des runs, reprise des pannes réseau, planification locale | ajoute un service et une base d'état à exploiter |
| Registre | MLflow, alias `champion` et `challenger` | découple l'API du numéro de version et permet le retour arrière | la promotion reste dépendante de métriques correctement journalisées |
| Mise en service | validation des données, meilleur que la baseline et MAE non dégradée | règle explicite, testable et auditable | aucun seuil absolu arbitraire de performance n'est imposé |
| API | FastAPI + schémas Pydantic stricts | erreurs HTTP contrôlées et contrat OpenAPI | aucune authentification, conformément au périmètre local |
| Persistance | PostgreSQL en mode *best effort* pour `/predict` | une panne de base ne coupe pas la prédiction ; `persisted=false` l'indique | une trace peut être perdue pendant l'incident |
| Cache | LRU/TTL mémoire, activable, clé déterministe et isolée par version | gain important sans infrastructure supplémentaire, invalidation sûre après promotion | un cache distinct par worker et aucun partage entre réplicas |
| Métriques | Prometheus, compatible avec plusieurs workers | débit, latence, erreurs, version active et saturation visibles | la rétention locale n'est pas une solution d'archivage long terme |
| Dashboard | Grafana provisionné depuis Git, PostgreSQL en lecture seule | installation sans clic, panels et règles d'alerte reproductibles | les destinataires externes d'alerte restent propres à l'environnement |
| Déploiement | Docker Compose et images versionnées | démonstration locale de bout en bout sans service payant | ce n'est pas une plateforme haute disponibilité |

## Chaîne nominale

1. Le flow quotidien récupère un lot par push s'il est déjà arrivé, sinon par pull.
2. Les contrôles des annonces et des scores s'exécutent en parallèle.
3. Un lot conforme rejoint les lots acceptés ; un lot invalide est mis en quarantaine et
   fait échouer le flow.
4. Le flow d'entraînement reconstruit le catalogue avec uniquement les lots acceptés,
   contrôle MAL, AniList et le jeu fusionné, puis entraîne le pipeline complet.
5. MLflow reçoit les versions de données, paramètres, métriques et artefacts. La règle de
   promotion choisit `champion`, `challenger` ou le rejet.
6. L'API charge `models:/anime-score-regressor@champion`, valide les entrées et renvoie la
   prédiction avec la version utilisée.
7. La prédiction est tracée dans PostgreSQL. Le score réel peut ensuite être rattaché par
   `/feedback` ou par le flow automatique A9.
8. Prometheus collecte les métriques API et l'état Prefect ; Grafana combine ces séries avec
   les prédictions et feedbacks PostgreSQL.

## Fiabilité, gestion des échecs et retour arrière

### Données et entraînement

- Les contrôles qualité produisent des rapports archivés et lèvent une erreur bloquante.
- Le rejeu n'entraîne que sur des lots acceptés ; les lots rejetés restent en quarantaine.
- La récupération du lot quotidien est retentée trois fois. AniList est retenté puis utilise
  le dernier cache disponible, avec une alerte explicite.
- L'échec ou le crash d'un flow déclenche le mécanisme d'alerte Prefect du projet.
- Le champion en place continue de servir si un entraînement, une validation ou une
  promotion échoue.

### Service de prédiction

- Une entrée invalide est rejetée en `422` avant l'appel au modèle.
- Un registre ou modèle indisponible produit une réponse `503` contrôlée ; `/health`
  explique le composant dégradé.
- Une panne PostgreSQL laisse `/predict` répondre avec `persisted=false`, tandis que
  `/feedback` retourne `503` car l'association ne peut pas être garantie.
- Les erreurs du modèle, durées, entrées et identifiants de version sont conservés quand la
  base est disponible.
- `/admin/reload` recharge atomiquement le modèle. Le retour arrière déplace l'alias
  `champion` sur une version précédente, puis l'API recharge cet alias.

### Supervision

Grafana livre trois règles provisionnées : taux d'erreurs supérieur à 5 % sur cinq minutes,
latence p95 supérieure à une seconde et Prefect indisponible. L'exporteur Prefect reste en
ligne quand l'orchestrateur tombe et publie alors `anime_prefect_up = 0`, ce qui distingue
une panne de Prefect d'une panne de la supervision elle-même.

Le compte PostgreSQL de Grafana est créé automatiquement avec `CONNECT`, `USAGE` et
`SELECT` uniquement. Un smoke test a confirmé qu'il lit les données mais qu'un `INSERT`
est refusé.

## Incidents contrôlés démontrables

| Incident | Injection | Résultat attendu et preuve |
|---|---|---|
| Lot journalier invalide | flow quotidien avec `corrupt=true` | rapport en échec, lot en quarantaine, entraînement non alimenté, alerte |
| AniList indisponible | endpoint inaccessible après présence d'un cache | repli sur le cache daté, entraînement poursuivi, alerte de niveau warning |
| PostgreSQL arrêté | arrêt du service pendant `/predict` | prédiction `200`, `persisted=false`, santé dégradée ; feedback refusé en `503` |
| Prefect arrêté | `docker compose stop prefect` | tuile rouge et règle Grafana en alerte après deux minutes |
| Modèle indisponible | alias supprimé ou MLflow inaccessible au démarrage | API vivante, `/health` et `/predict` en `503`, rechargement possible après correction |
| Retour arrière | commande de rollback vers l'ancien champion puis reload | nouvelle réponse identifiant la version restaurée |

Ces scénarios doivent être exécutés dans la stack du projet uniquement. Les scripts de
charge et de démonstration utilisent des noms de projet Compose isolés afin de ne pas
arrêter une autre stack locale.

## Preuves d'intégration

| Preuve | Résultat observé |
|---|---|
| Tests du service B6 | 40 tests unitaires réussis, ruff et formatage sans erreur |
| Intégration A6 + API/charge | 123 tests réussis avec le vrai pipeline scikit-learn et le contrat de features final |
| Contrat MLflow | version 1 du modèle réel chargée par alias `champion`, run `d9a8dbebbee449cab3439ceb696997b6` |
| Persistance | chaque scénario de charge vérifie `persisted=true` et la présence des lignes PostgreSQL |
| Matrice de charge | 15 scénarios consolidés, aucune requête en échec |
| Grafana | version 13.3.0 saine, deux datasources saines, dashboard et trois alertes provisionnés |
| Prefect | un vrai flow Prefect 3.9.3 a été lu comme `completed` avec sa durée par l'exporteur |
| Isolation | les stacks de test et leurs volumes ont été supprimés après chaque campagne |

Les mesures brutes consolidées et le contexte matériel sont versionnés dans
[`mesures-charge-2026-11-15.json`](mesures-charge-2026-11-15.json). Les rapports Locust
HTML/CSV générés restent hors Git car ils sont reproductibles et volumineux.

## Résultats de charge et dimensionnement

### Matrice principale à 50 utilisateurs

| Workers | Cache | Débit | p95 | CPU max | RAM max | Échecs |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | non | 14,3 req/s | 3 900 ms | 1,48 cœur | 310 Mio | 0 |
| 1 | oui | 81,6 req/s | 990 ms | 1,55 cœur | 292 Mio | 0 |
| 2 | non | 27,7 req/s | 2 300 ms | 2,53 cœurs | 614 Mio | 0 |
| 2 | oui | 146,2 req/s | 710 ms | 2,68 cœurs | 620 Mio | 0 |
| 4 | non | 51,8 req/s | 1 600 ms | 4,76 cœurs | 1 111 Mio | 0 |
| 4 | oui | 260,9 req/s | 520 ms | 4,88 cœurs | 1 104 Mio | 0 |

Le cache multiplie le débit par 5,0 à 5,7 selon le nombre de workers. Il ne supprime pas le
coût de validation et de persistance : chaque hit reste tracé, ce qui préserve l'audit.

### Recommandation

| Utilisateurs simultanés | Configuration API | Allocation recommandée | Référence mesurée |
|---:|---|---:|---|
| 1 à 5 | 1 worker + cache | 2 vCPU, 512 Mio | p95 180 ms, 89,6 req/s |
| 6 à 25 | 2 workers + cache | 4 vCPU, 1 Gio | p95 390 ms, 156,3 req/s |
| 26 à 50 | 4 workers + cache | 8 vCPU, 2 Gio | p95 520 ms, 260,9 req/s |

La capacité planifiée conserve 30 % de marge sur le débit mesuré. Les allocations CPU/RAM
incluent au moins 25 % de marge et concernent uniquement l'API. PostgreSQL, MLflow,
Prometheus, Prefect et Grafana nécessitent leurs propres réserves.

## Sécurité et exploitation locale

- Aucun mot de passe, `.env`, dataset ou cache n'est versionné.
- Les variables sensibles sont injectées à l'exécution par Docker Compose.
- Grafana utilise un compte SQL en lecture seule, distinct du compte applicatif.
- Les entrées API sont validées et les messages d'erreur internes sont bornés avant
  persistance.
- Les dépendances et images principales sont figées pour rendre les démonstrations
  reproductibles.

Le périmètre demandé est local. Pour une exposition réseau réelle, il faudra ajouter un
reverse proxy TLS, une authentification de l'API et de `/admin/reload`, une rotation des
secrets, des sauvegardes PostgreSQL/MLflow et un contact d'alerte externe.

## Limites connues et travaux de clôture

1. **Données de benchmark** — le pipeline et les hyperparamètres sont réels, mais le modèle
   utilisé pour la campagne de charge a été entraîné sur 4 000 lignes synthétiques
   déterministes. La campagne B8 doit être rejouée après entraînement du champion définitif
   sur les données MAL/AniList.
2. **Feedback automatique** — le dashboard et `save_feedback` respectent déjà le contrat A9,
   mais la branche A9 doit encore être intégrée et testée de bout en bout.
3. **Durée des tests** — les scénarios de 30 à 60 secondes mesurent la saturation, pas les
   fuites lentes. Ajouter un test de tenue de 15 à 30 minutes avant un service durable.
4. **Plateforme unique** — Locust, Docker, PostgreSQL et le modèle partageaient un hôte
   Windows. Les chiffres ne sont pas extrapolés à une plateforme cloud non mesurée.
5. **Cache local** — chaque worker possède son propre LRU/TTL. Plusieurs réplicas justifieront
   un cache partagé et une nouvelle campagne de charge.
6. **Observabilité** — ajouter le taux de hits du cache, la saturation du pool PostgreSQL et
   la rétention/archivage des séries pour approfondir le diagnostic.
7. **Notifications Grafana** — les règles sont provisionnées et évaluées, mais les adresses
   mail ou webhooks restent hors Git et doivent être configurées dans l'environnement cible.
8. **Intégration finale** — les PR A et B sont volontairement séparées. Après leur fusion,
   rejouer tous les tests, le scénario nominal, les incidents et la matrice de charge courte.

## Correspondance avec la grille

| Critère | Éléments apportés |
|---|---|
| Qualité du code (20 %) | modules séparés, typage, configuration, ruff, tests unitaires et CI |
| Reproductibilité (10 %) | `uv.lock`, images figées, Compose, données versionnées par empreinte, commandes documentées |
| Cycle MLOps (20 %) | ingestion/rejeu → qualité → features → entraînement → promotion → API → feedback |
| Fiabilité et traçabilité (20 %) | contrôles bloquants, quarantaine, retries, alertes, MLflow, versions dans les réponses, rollback |
| Exploitation (20 %) | API, batch, PostgreSQL résilient, feedbacks, Prometheus, Grafana et incidents contrôlés |
| Dimensionnement (10 %) | matrice 1/2/4 workers avec/sans cache, CPU/RAM, marges et recommandations par palier |

## Commandes de vérification

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest
docker compose up -d --build --wait
```

Pour reproduire le protocole de charge complet :

```bash
uv sync --locked --group load
make load-test LOAD_TEST_ARGS="--users 50 --spawn-rate 50 --run-time 60s"
```

Le protocole détaillé, les paliers complémentaires et les réserves de ressources sont dans
[`dimensionnement.md`](dimensionnement.md).
