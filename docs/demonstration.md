# Démonstration de bout en bout

Ce déroulé couvre le fonctionnement nominal et les incidents demandés sans modifier une
stack extérieure au projet. Les commandes sont prévues pour PowerShell, à la racine du
dépôt.

## 1. Préparation

Copier `.env.example` vers `.env`, remplacer tous les mots de passe `change-me`, puis
installer l'environnement :

```powershell
uv sync --locked --group load
```

Démarrer PostgreSQL et MLflow avant l'API :

```powershell
docker compose up -d --build --wait postgres mlflow prefect
```

Pour une démonstration purement opérationnelle, enregistrer le modèle factice. Pour la
démonstration finale, préférer le champion entraîné par le pipeline complet.

```powershell
make register-mock-model
docker compose up -d --build --wait
```

Interfaces utiles :

- API : <http://localhost:8000/docs> ;
- MLflow : <http://localhost:5000> ;
- Prefect : <http://localhost:4200> ;
- Prometheus : <http://localhost:9090> ;
- Grafana : <http://localhost:3000>.

## 2. Scénario nominal du service

Activer le cache dans `.env` avec `ANIME_API__CACHE_ENABLED=true`, recréer l'API, puis
lancer :

```powershell
docker compose up -d --build --wait api prometheus grafana
make demo-nominal
```

Le script vérifie automatiquement :

1. santé du modèle et de PostgreSQL ;
2. deux prédictions identifiant la même version du modèle ;
3. persistance des deux traces ;
4. hit du cache au second appel si le cache est activé ;
5. feedback manuel rattaché à la première prédiction ;
6. exposition des métriques HTTP et de la version servie.

Une sortie JSON et un code retour `0` constituent la preuve de réussite. Les nouvelles
lignes sont immédiatement visibles dans le dashboard Grafana.

## 3. Feedback automatique, jour J puis J+1

Dans un premier terminal, démarrer la source de rejeu :

```powershell
make replay
```

Dans un second terminal, traiter deux lots successifs :

```powershell
make daily
make daily
make feedback
```

Au premier passage, les animes annoncés sont prédits et leurs `prediction_id` sont inscrits
dans `data/predictions/`. Au passage suivant, les scores révélés sont écrits dans
`feedbacks` avec la source `automatic`. Relancer `make feedback` ne crée aucun doublon.

Preuves à montrer : artefact **feedback-automatique** dans Prefect, table des derniers
feedbacks dans Grafana et courbe « Prédiction vs score réel ».

## 4. Incident : lot invalide

```powershell
make demo-incident
```

Le contrôle échoue, le lot rejoint `data/quarantine/`, le flow passe en échec et une alerte
est ajoutée à `data/reports/alerts.jsonl`. Aucun entraînement ni feedback n'utilise ce lot.

## 5. Incident : AniList indisponible

Ce scénario nécessite qu'un cache AniList valide existe déjà. Dans PowerShell :

```powershell
$env:ANIME_SOURCES__ANILIST__URL = "http://127.0.0.1:9/graphql"
make pipeline
Remove-Item Env:ANIME_SOURCES__ANILIST__URL
```

Le flow retente l'appel, utilise ensuite le dernier cache et publie une alerte `warning`.
L'entraînement reste reproductible avec la date et l'empreinte de ce cache dans MLflow.

## 6. Incident : PostgreSQL arrêté

La commande suivante arrête **uniquement** le service `postgres` du projet Compose confirmé,
teste le mode dégradé, puis le redémarre dans un bloc `finally` :

```powershell
make demo-postgres-incident PROJECT_NAME=anime-mlops CONFIRM_PROJECT=anime-mlops
```

Le script refuse de démarrer si les deux noms diffèrent ou si le projet ne contient pas
exactement un service PostgreSQL créé. Il vérifie successivement :

- `/predict` reste en `200` avec `persisted=false` ;
- `/feedback` répond `503 database_unavailable` ;
- après redémarrage, PostgreSQL revient à `ok` et une nouvelle prédiction est persistée.

## 7. Retour arrière du modèle

Afficher d'abord les versions et noter le champion courant :

```powershell
make status
make rollback VERSION=<version-précédente>
Invoke-RestMethod -Method Post http://localhost:8000/admin/reload
Invoke-RestMethod http://localhost:8000/model
```

La dernière commande doit afficher la version restaurée. Une nouvelle prédiction porte la
même version et le cache précédent a été vidé lors du rechargement.

## 8. Supervision et conclusion

Dans Grafana, montrer :

- débit, p50/p95 et erreurs de l'API ;
- modèle servi et prédictions par version ;
- état et durée des flows Prefect ;
- feedbacks, MAE et prédiction vs score réel ;
- règles d'alerte API/Prefect.

Terminer par la recommandation mesurée : 2 workers avec cache, 4 vCPU et 1 Gio pour 6 à
25 utilisateurs simultanés. Les chiffres détaillés et leurs limites sont dans
[`dimensionnement.md`](dimensionnement.md).

## 9. Nettoyage

Le nettoyage conserve les volumes pour une prochaine répétition :

```powershell
docker compose down
```

Ne lancer `docker compose down --volumes` que si la suppression définitive des bases,
artefacts MLflow et données Grafana est réellement voulue.
