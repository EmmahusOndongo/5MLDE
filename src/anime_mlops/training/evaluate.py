"""Évaluation du modèle. La MAE est la métrique de décision pour la mise en service."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def baseline_mae(y_train, y_test) -> float:
    """MAE d'un modèle naïf qui prédit toujours la moyenne : point de repère minimal."""
    return float(mean_absolute_error(y_test, np.full(len(y_test), np.mean(y_train))))


def error_by_group(groups: pd.Series, y_true, y_pred) -> pd.DataFrame:
    """MAE et effectif par catégorie (ex. par type de média), pour repérer les points faibles."""
    errors = pd.DataFrame(
        {
            "groupe": pd.Series(groups).fillna("inconnu").to_numpy(),
            "erreur_absolue": np.abs(np.asarray(y_true) - np.asarray(y_pred)),
        }
    )
    table = errors.groupby("groupe")["erreur_absolue"].agg(mae="mean", effectif="size")
    return table.sort_values("effectif", ascending=False).reset_index()
