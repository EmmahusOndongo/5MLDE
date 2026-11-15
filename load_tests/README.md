# Tests de charge B8

Cette matrice compare **les processus Uvicorn de l'API** (et non les workers Locust) :
1, 2 puis 4 workers, avec le cache désactivé et activé. Les six scénarios utilisent
exactement la même charge fermée : 50 utilisateurs simultanés, sans temps d'attente,
90 % de requêtes choisies dans huit entrées récurrentes et 10 % d'entrées uniques.

Chaque réponse doit être valide, identifier la version du modèle et confirmer sa
persistance PostgreSQL. Une seule erreur rend le scénario en échec. Entre deux scénarios,
l'API est recréée et les tables de la stack de test sont vidées. Les métriques exportées
sont le débit, les latences moyenne/p50/p95/p99/max, les erreurs ainsi que CPU et mémoire
du conteneur API. Un CPU Docker supérieur à 100 % signifie que plusieurs cœurs sont utilisés.

## Exécution reproductible

Prérequis : Docker Desktop démarré, `uv` installé et les ports 8800, 5500 et 55432 libres.

```bash
uv sync --locked --group load
make load-test
```

La commande crée le projet Compose isolé `anime-mlops-load`, génère un mot de passe
éphémère, enregistre le modèle factice `champion`, exécute la matrice puis supprime cette
stack et ses volumes. Elle ne touche pas à la stack de développement `anime-mlops`.
Les résultats sont écrits sous `load_tests/results/<horodatage>/` et ne sont pas versionnés.

Pour une vérification rapide du protocole :

```bash
uv run --group load python -m load_tests.run_matrix --users 5 --run-time 10s
```

Les paramètres peuvent être transmis à la cible Make :

```bash
make load-test LOAD_TEST_ARGS="--users 100 --spawn-rate 50 --run-time 2m"
```

`--keep-stack` conserve la stack isolée après la campagne. Pour mesurer un vrai modèle déjà
enregistré dans un autre projet Compose, fournir les mêmes `POSTGRES_USER` et
`POSTGRES_PASSWORD`, puis utiliser `--skip-bootstrap --project-name <projet>` et les ports
de ce projet. Dans ce mode, le script ne supprime jamais la stack existante.

## Lecture des sorties

- `summary.json` : tableau consolidé Locust + CPU/RAM et contexte matériel ;
- `*_stats.csv` : statistiques détaillées Locust ;
- `*_stats_history.csv` : évolution temporelle ;
- `*.html` : rapport autonome par scénario ;
- `*_failures.csv` et `*_exceptions.csv` : diagnostic des incidents.

Le modèle factice sert uniquement à valider la reproductibilité de B8. La note B9 doit
retenir la campagne exécutée avec le vrai pipeline `champion`, car son coût d'inférence
est représentatif du service final.
