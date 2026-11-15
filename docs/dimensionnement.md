# Note de dimensionnement de l'API

## Décision

Le profil recommandé pour la démonstration et un petit déploiement local est **2 workers
Uvicorn, cache activé, 4 vCPU et 1 Gio de RAM pour le conteneur API**. Il tient le palier
mesuré de 25 utilisateurs simultanés à 156,3 requêtes/s, avec un p95 de 390 ms et zéro
échec.

Pour 50 utilisateurs simultanés en charge continue, passer à **4 workers, 8 vCPU et
2 Gio de RAM**. Cette configuration atteint 260,9 requêtes/s, un p95 de 520 ms et zéro
échec. Les allocations incluent une marge de 25 % sur les maxima CPU/RAM observés, puis
sont arrondies à une taille exploitable.

Le cache doit rester activé pour cette recommandation. Avec 50 utilisateurs, 4 workers
sans cache ne fournissent que 51,8 requêtes/s et un p95 de 1,6 s.

## Objectif de service retenu

Aucun SLO n'étant imposé dans le sujet, la décision du projet est :

- aucune réponse en échec pendant la campagne ;
- persistance PostgreSQL confirmée pour chaque prédiction ;
- p95 inférieur à 1 seconde ;
- 30 % de capacité de débit conservée pour absorber les pointes.

Ce seuil sert au dimensionnement ; ce n'est pas une propriété générale du modèle.

## Environnement mesuré

| Élément | Valeur |
|---|---|
| Date | 15 novembre 2026 |
| Hôte | Windows 10, 12 CPU logiques |
| Docker | Docker Desktop 28.5.1, 7,614 Gio disponibles |
| Python / Locust | Python 3.12.16 / Locust 2.46.7 |
| API | FastAPI + Uvicorn, persistance PostgreSQL active |
| Modèle | `anime-score-regressor@champion`, version MLflow 1 |
| Pipeline | A6 complet : prétraitement + Random Forest |
| Configuration | 200 arbres, profondeur 12, `min_samples_leaf=2`, `n_jobs=-1` |
| Données du modèle de mesure | 4 000 lignes synthétiques déterministes, usage opérationnel uniquement |

Le dataset de benchmark ne sert pas à conclure sur la qualité prédictive. Il permet de
charger le vrai type de pipeline et sa configuration finale alors que le dataset historique
final n'est pas versionné dans le dépôt. La campagne doit être rejouée une dernière fois
après entraînement du champion définitif sur les données MAL/AniList.

## Protocole

La matrice est exécutée par `load_tests/run_matrix.py`, en mode headless documenté par
[Locust](https://docs.locust.io/en/stable/running-without-web-ui.html). Locust exporte les
[statistiques CSV](https://docs.locust.io/en/stable/configuration.html#csv-options) et le
runner ajoute les mesures `docker stats`.

- endpoint testé : `POST /predict` ;
- modèle de charge fermé, sans temps d'attente, pour rechercher la capacité maximale ;
- 90 % des appels utilisent huit entrées récurrentes, 10 % une entrée numérique unique ;
- cache vide au début de chaque scénario, local à chaque worker ;
- validation du statut HTTP, de la prédiction, de la version et de `persisted=true` ;
- recréation de l'API et vidage des tables de la stack de test entre deux scénarios ;
- matrice principale : 50 utilisateurs, montée à 50 utilisateurs/s, 60 secondes ;
- paliers complémentaires : 5, 10 et 25 utilisateurs, 30 secondes, cache activé ;
- CPU Docker : 100 % correspond à un cœur utilisé ;
- aucun échec sur les 15 scénarios consolidés.

Les mesures consolidées, sans identifiant secret ni dataset, sont conservées dans
[`mesures-charge-2026-11-15.json`](mesures-charge-2026-11-15.json).

## Matrice principale — 50 utilisateurs

| Workers | Cache | Requêtes | req/s | Moyenne | p95 | p99 | CPU moyen / max | RAM max | Échecs |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | non | 847 | 14,3 | 3 394 ms | 3 900 ms | 4 900 ms | 1,15 / 1,48 cœur | 310 Mio | 0 |
| 1 | oui | 4 845 | 81,6 | 610 ms | 990 ms | 3 100 ms | 1,22 / 1,55 cœur | 292 Mio | 0 |
| 2 | non | 1 642 | 27,7 | 1 777 ms | 2 300 ms | 2 400 ms | 2,17 / 2,53 cœurs | 614 Mio | 0 |
| 2 | oui | 8 671 | 146,2 | 340 ms | 710 ms | 1 100 ms | 2,27 / 2,68 cœurs | 620 Mio | 0 |
| 4 | non | 3 072 | 51,8 | 955 ms | 1 600 ms | 1 800 ms | 4,41 / 4,76 cœurs | 1 111 Mio | 0 |
| 4 | oui | 15 457 | 260,9 | 191 ms | 520 ms | 780 ms | 4,34 / 4,88 cœurs | 1 104 Mio | 0 |

### Effet du cache

| Workers | Gain de débit | Réduction du p95 |
|---:|---:|---:|
| 1 | ×5,71 | −74,6 % |
| 2 | ×5,27 | −69,1 % |
| 4 | ×5,04 | −67,5 % |

Le gain reste inférieur à un cache « pur » : même lors d'un hit, l'API valide la requête,
calcule la clé, construit la réponse et écrit une trace PostgreSQL. Ce comportement est
volontaire pour conserver l'audit complet.

## Montée en charge avec cache

| Utilisateurs | Workers | req/s | p95 | CPU max | RAM max |
|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 89,6 | 180 ms | 1,35 cœur | 284 Mio |
| 5 | 2 | 146,0 | 130 ms | 2,08 cœurs | 580 Mio |
| 5 | 4 | 164,6 | 110 ms | 2,31 cœurs | 1 067 Mio |
| 10 | 1 | 89,2 | 300 ms | 1,35 cœur | 279 Mio |
| 10 | 2 | 156,4 | 200 ms | 2,51 cœurs | 579 Mio |
| 10 | 4 | 200,7 | 140 ms | 2,84 cœurs | 1 074 Mio |
| 25 | 1 | 79,1 | 630 ms | 1,46 cœur | 297 Mio |
| 25 | 2 | 156,3 | 390 ms | 2,61 cœurs | 598 Mio |
| 25 | 4 | 276,3 | 290 ms | 4,58 cœurs | 1 095 Mio |
| 50 | 1 | 81,6 | 990 ms | 1,55 cœur | 292 Mio |
| 50 | 2 | 146,2 | 710 ms | 2,68 cœurs | 620 Mio |
| 50 | 4 | 260,9 | 520 ms | 4,88 cœurs | 1 104 Mio |

La mémoire croît presque linéairement avec les workers, car chacun charge son propre modèle
et son propre cache. À 5 utilisateurs, passer de 2 à 4 workers n'apporte que 12,8 % de débit
supplémentaire pour environ 84 % de RAM en plus : 4 workers ne sont donc justifiés que pour
les paliers supérieurs.

## Recommandation par palier

La capacité planifiée correspond à 70 % du débit mesuré, afin de garder 30 % de marge.
Les ressources concernent uniquement le conteneur API ; PostgreSQL et MLflow doivent avoir
leurs propres réserves.

| Utilisateurs simultanés | Configuration | Mesure de référence | Capacité planifiée | Allocation API |
|---:|---|---|---:|---:|
| 1 à 5 | 1 worker + cache | p95 180 ms, max 1,35 CPU / 284 Mio | 62 req/s | 2 vCPU, 512 Mio |
| 6 à 25 | 2 workers + cache | p95 390 ms, max 2,61 CPU / 598 Mio | 109 req/s | 4 vCPU, 1 Gio |
| 26 à 50 | 4 workers + cache | p95 520 ms, max 4,88 CPU / 1 104 Mio | 183 req/s | 8 vCPU, 2 Gio |

Au-delà de 50 utilisateurs simultanés, aucune extrapolation n'est présentée comme une mesure.
Il faut ajouter des réplicas API derrière un répartiteur, vérifier la capacité PostgreSQL,
et relancer la campagne. Un cache partagé peut être envisagé si plusieurs conteneurs doivent
réutiliser les mêmes prédictions.

## Commandes de reproduction

Installation et matrice complète sur une stack isolée avec modèle factice :

```bash
uv sync --locked --group load
make load-test LOAD_TEST_ARGS="--users 50 --spawn-rate 50 --run-time 60s"
```

Sur une stack existante qui contient le vrai alias `champion` :

```bash
uv run --group load python -m load_tests.run_matrix \
  --skip-bootstrap --skip-build \
  --project-name <projet-compose> \
  --api-port <port-api> --mlflow-port <port-mlflow> --postgres-port <port-postgres> \
  --users 50 --spawn-rate 50 --run-time 60s
```

Palier cache seul :

```bash
uv run --group load python -m load_tests.run_matrix \
  --skip-bootstrap --skip-build --project-name <projet-compose> \
  --cache-modes on --workers 1 2 4 \
  --users 25 --spawn-rate 25 --run-time 30s
```

## Limites connues

- Le ratio de réutilisation est de 90 %. Une charge à faible taux de hits se rapprochera des
  résultats « cache non » ; il faut ajouter une métrique de hit ratio au dashboard.
- Le client Locust, PostgreSQL, MLflow et Docker tournent sur la même machine. Les chiffres
  sont valables pour cette plateforme locale, pas pour une instance cloud non mesurée.
- Le test vise `/predict` unitaire. Les lots et les feedbacks nécessitent un protocole dédié.
- Soixante secondes détectent la saturation mais pas une fuite mémoire ; ajouter un test de
  tenue de 15 à 30 minutes avant un déploiement durable.
- `n_jobs=-1` parallélise chaque Random Forest tandis qu'Uvicorn multiplie les processus.
  Modifier `n_jobs` impose de refaire la matrice pour éviter une surallocation CPU.
- Le modèle a la configuration finale mais a été entraîné sur des données synthétiques de
  benchmark. Rejouer B8 après production du champion définitif reste une condition de clôture.
