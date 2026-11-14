"""Construction des features à partir du jeu fusionné.

Seules les informations connues **avant la diffusion** sont retenues. Ce module fixe
le contrat d'entrée du modèle : l'encodage (multi-hot, top N…) est fait ensuite dans
le ``Pipeline`` scikit-learn, pour que l'API puisse lui passer ces champs bruts.
"""

import pandas as pd

TARGET = "mean"

NUMERIC_FEATURES = ["num_episodes", "average_episode_duration", "start_year", "prequel_score"]
CATEGORICAL_FEATURES = ["media_type", "rating", "source", "start_season", "start_weekday"]
BOOLEAN_FEATURES = ["is_sequel"]
# Listes de libellés, encodées en multi-hot par le modèle.
MULTI_LABEL_FEATURES = ["genres", "studios", "tags"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *BOOLEAN_FEATURES, *MULTI_LABEL_FEATURES]

# Indicateurs qui dépendent de la réception du public : inconnus avant diffusion,
# les utiliser fausserait l'évaluation (fuite de données).
LEAKAGE_COLUMNS = ["rank", "popularity", "num_list_users", "num_scoring_users"]

_SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "winter",
    4: "spring",
    5: "spring",
    6: "spring",
    7: "summer",
    8: "summer",
    9: "summer",
    10: "fall",
    11: "fall",
    12: "fall",
}
_FULL_DATE_LENGTH = len("2020-04-03")


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Renvoie les features (indexées par identifiant MAL), sans la cible ni colonne à fuite."""
    features = pd.DataFrame(index=pd.Index(merged["id"], name="id"))
    source = merged.set_index(features.index)

    # Chez MAL, 0 signifie « inconnu » pour ces deux compteurs.
    for column in ("num_episodes", "average_episode_duration"):
        features[column] = source[column].where(source[column] > 0).astype("float64")

    start_date = source["start_date"].astype("string")
    features["start_year"] = pd.to_numeric(start_date.str[:4], errors="coerce").astype("float64")
    features["prequel_score"] = source["prequel_score"].astype("float64")

    for column in ("media_type", "rating", "source"):
        features[column] = source[column].astype("object").where(source[column].notna(), None)

    month = pd.to_numeric(start_date.str[5:7], errors="coerce")
    features["start_season"] = month.map(_SEASON_BY_MONTH).astype("object")
    # Le jour de diffusion n'a de sens que si la date est complète.
    full_date = pd.to_datetime(
        start_date.where(start_date.str.len() == _FULL_DATE_LENGTH),
        format="%Y-%m-%d",
        errors="coerce",
    )
    features["start_weekday"] = full_date.dt.day_name().str.lower().astype("object")
    for column in ("start_season", "start_weekday"):
        features[column] = features[column].where(features[column].notna(), None)

    features["is_sequel"] = source["is_sequel"].astype(bool)

    features["genres"] = source["genres"].map(_split_labels)
    features["studios"] = source["studios"].map(_split_labels)
    features["tags"] = source["tags"].map(_as_list)

    return features[FEATURES]


def split_features_target(training: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare features et cible pour l'entraînement."""
    if training[TARGET].isna().any():
        raise ValueError("Le jeu d'entraînement contient des animes sans score.")
    features = build_features(training)
    target = pd.Series(training[TARGET].to_numpy(), index=features.index, name=TARGET)
    return features, target


def _split_labels(value) -> list[str]:
    """« Action, Drama » → ['Action', 'Drama'] ; valeur manquante → liste vide."""
    if not isinstance(value, str):
        return []
    return [label.strip() for label in value.split(",") if label.strip()]


def _as_list(value) -> list[str]:
    """Normalise les tags (liste, tableau NumPy après lecture Parquet, ou manquant)."""
    if value is None or isinstance(value, float):
        return []
    return [str(label) for label in value]
