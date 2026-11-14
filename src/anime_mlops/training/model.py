"""Modèle : un ``Pipeline`` scikit-learn qui embarque tout le prétraitement.

L'artefact enregistré accepte directement les champs bruts du contrat
(``anime_mlops.features.build.FEATURES``) : l'API n'a aucun encodage à refaire.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from anime_mlops.config import ModelConfig
from anime_mlops.features.build import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

OTHER_LABEL = "autre"


class TopNMultiHot(BaseEstimator, TransformerMixin):
    """Encode une colonne de listes de libellés en colonnes binaires.

    Seuls les ``top_n`` libellés les plus fréquents à l'entraînement ont leur colonne ;
    une colonne « autre » signale la présence d'au moins un libellé hors de cette liste
    (rare ou jamais vu), ce qui rend la prédiction robuste aux nouveautés.
    """

    def __init__(self, top_n: int | None = None):
        self.top_n = top_n

    def fit(self, X, y=None):
        counts = Counter(label for labels in _as_label_lists(X) for label in set(labels))
        # Tri par fréquence puis par nom : l'ordre des colonnes est déterministe.
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        self.labels_ = [label for label, _ in ranked[: self.top_n]]
        return self

    def transform(self, X):
        rows = _as_label_lists(X)
        position = {label: index for index, label in enumerate(self.labels_)}
        encoded = np.zeros((len(rows), len(self.labels_) + 1), dtype=np.float32)
        for row, labels in enumerate(rows):
            for label in labels:
                encoded[row, position.get(label, len(self.labels_))] = 1.0
        return encoded

    def get_feature_names_out(self, input_features=None):
        return np.array([*self.labels_, OTHER_LABEL], dtype=object)


def build_model(config: ModelConfig, seed: int) -> Pipeline:
    """Construit le pipeline non entraîné à partir de la configuration."""
    if config.type != "random_forest":
        raise ValueError(f"Type de modèle non pris en charge : {config.type!r}")

    preprocessing = ColumnTransformer(
        transformers=[
            # Médiane + indicateur de valeur manquante (ex. prequel_score absent = pas une suite).
            (
                "numerique",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                NUMERIC_FEATURES,
            ),
            # Une valeur manquante ou jamais vue donne une ligne de zéros, sans erreur.
            ("categorie", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("booleen", "passthrough", BOOLEAN_FEATURES),
            ("genres", TopNMultiHot(), "genres"),
            ("studios", TopNMultiHot(top_n=config.top_n_studios), "studios"),
            ("tags", TopNMultiHot(top_n=config.top_n_tags), "tags"),
        ],
        sparse_threshold=0.0,
    )
    regressor = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        # Les arbres sont entraînés en parallèle sur les cœurs disponibles.
        n_jobs=config.n_jobs,
        random_state=seed,
    )
    return Pipeline([("pretraitement", preprocessing), ("regresseur", regressor)])


def feature_importances(model: Pipeline) -> pd.DataFrame:
    """Importance de chaque colonne encodée, de la plus à la moins influente."""
    names = model.named_steps["pretraitement"].get_feature_names_out()
    importances = model.named_steps["regresseur"].feature_importances_
    table = pd.DataFrame({"feature": names, "importance": importances})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def _as_label_lists(X) -> list[list[str]]:
    """Accepte une Series, un DataFrame à une colonne ou un tableau ; renvoie des listes."""
    values = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
    values = (
        np.asarray(values, dtype=object).ravel() if not isinstance(values, pd.Series) else values
    )
    return [
        list(labels) if isinstance(labels, (list, tuple, np.ndarray)) else [] for labels in values
    ]
