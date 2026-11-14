# Mise en service du modèle : alias, tags et retour arrière

Le modèle vit dans le registre MLflow sous le nom `anime-score-regressor`.
**L'API sert toujours la version qui porte l'alias `champion`** : mettre un modèle en
service ou revenir en arrière revient à déplacer cet alias, sans redéployer de code.

Les règles sont dans [`configs/promotion.yaml`](../configs/promotion.yaml), le code dans
[`src/anime_mlops/training/promote.py`](../src/anime_mlops/training/promote.py).

## 1. Cycle de vie d'une version

```
entraînement ──► nouvelle version (tags d'origine)
                      │
                      ▼
            critères d'acceptation ──── non ──► status = rejected   (aucun alias)
                      │ oui
                      ▼
          fait au moins aussi bien ──── non ──► status = challenger (alias challenger)
            que le champion ?
                      │ oui
                      ▼
            status = champion (alias champion)
            l'ancien champion passe à status = archived
```

## 2. Critères d'acceptation

Une version n'est mise en service que si **toutes** ces conditions sont vraies :

| # | Critère | Pourquoi | Sinon |
|---|---|---|---|
| 1 | Les contrôles qualité des données sont passés (`data.validation = passed` sur le run) | Un modèle entraîné sur des données non contrôlées n'est pas fiable, même avec un bon score | `rejected` |
| 2 | La MAE de test est strictement meilleure que la référence naïve (prédire la moyenne) | Garde-fou minimal : le modèle doit apporter quelque chose | `rejected` |
| 3 | La MAE de test est inférieure ou égale à celle du champion (tolérance configurable, 0 par défaut) | Ne jamais dégrader le service | `challenger` |

S'il n'y a encore aucun champion, les critères 1 et 2 suffisent.

## 3. Alias

| Alias | Signification | Posé quand |
|---|---|---|
| `champion` | Version servie par l'API | Promotion réussie, ou retour arrière |
| `challenger` | Dernière version valide qui n'a pas battu le champion | Critère 3 non rempli |

Un alias ne pointe que sur une version à la fois : le déplacer est atomique.

## 4. Tags

### Sur le run d'entraînement (posés au moment de l'entraînement)

| Tag | Contenu |
|---|---|
| `data.mal_commit`, `data.mal_sha256` | Version exacte du catalogue MyAnimeList |
| `data.anilist_extraction_date`, `data.anilist_sha256` | Version exacte de l'extraction AniList |
| `data.anilist_is_fallback` | `True` si AniList était en panne et qu'un cache antérieur a servi |
| `data.validation` | `passed` si les trois contrôles qualité ont réussi |
| `code.commit`, `code.dirty` | Commit Git du code, et présence de modifications non commitées |
| `sklearn.version` | Version de la bibliothèque d'entraînement |

### Sur la version du modèle

| Tag | Posé quand | Contenu |
|---|---|---|
| `mal_commit`, `anilist_extraction_date`, `anilist_is_fallback` | À l'enregistrement | Origine des données |
| `code_commit`, `code_dirty` | À l'enregistrement | Origine du code |
| `test_mae`, `test_rmse`, `train_rows` | À l'enregistrement | Performance et taille du jeu |
| `status` | À chaque décision | `champion`, `challenger`, `rejected`, `archived` ou `rolled_back` |
| `decision_reason` | À chaque décision | Phrase expliquant la décision, chiffres à l'appui |
| `decided_at` | À chaque décision | Date et heure (UTC) |
| `replaced_version` | À la promotion | Version que ce champion a remplacée (cible du retour arrière) |
| `rollback_from_version` | Au retour arrière | Version qui a été retirée |

Les tags d'origine répondent à « d'où vient ce modèle ? », les tags de décision à
« pourquoi est-il (ou n'est-il pas) en service ? ».

## 5. Commandes

```bash
make status                # versions, alias, statut et raison de chaque décision
make train                 # entraîne puis applique les règles de promotion
make promote               # réévalue la dernière version (ou VERSION=N)
make rollback              # revient à la version que le champion a remplacée
make rollback VERSION=3    # revient à une version précise
```

Sans `make` (Windows) :

```bash
uv run python -m anime_mlops.training.promote status
uv run python -m anime_mlops.training.promote rollback --version 3
```

## 6. Retour arrière

`rollback` remet l'alias `champion` sur la version précédente (lue dans le tag
`replaced_version`) ou sur la version demandée. La version retirée reçoit
`status = rolled_back`, la version restaurée `rollback_from_version`.

Le retour arrière ne supprime rien : la version retirée reste dans le registre avec son
historique, et peut être remise en service par `make promote VERSION=N`.

L'API prend en compte le changement au rechargement du modèle (`POST /admin/reload`)
ou à son redémarrage.

## 7. Limite connue

Le critère 3 compare la MAE que chaque version a obtenue sur **son propre** jeu de test
(même graine, même proportion). Quand les données changent entre deux entraînements,
les deux jeux de test ne sont pas strictement identiques. Une comparaison plus rigoureuse
réévaluerait le champion sur le jeu de test du candidat.
