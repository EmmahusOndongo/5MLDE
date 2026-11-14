"""Enregistre un régresseur factice pour développer l'API avant le vrai modèle."""

import argparse
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyRegressor

from anime_mlops.config import get_settings

MOCK_MODEL_TAG = "model_kind"
MOCK_MODEL_TAG_VALUE = "mock"


def register_mock_model(*, force: bool = False) -> str:
    """Crée une version factice et lui attribue l'alias de service."""
    settings = get_settings()
    tracking_uri = settings.mlflow.tracking_uri
    model_name = settings.mlflow.registered_model_name
    alias = settings.api.model_alias

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    _refuse_to_replace_real_model(client, model_name, alias, force=force)

    model = DummyRegressor(strategy="constant", constant=7.0)
    model.fit(pd.DataFrame({"placeholder": [0]}), [7.0])

    with mlflow.start_run(run_name="mock-api-model"):
        mlflow.set_tag(MOCK_MODEL_TAG, MOCK_MODEL_TAG_VALUE)
        model_info = mlflow.sklearn.log_model(sk_model=model, name="model")

    version = mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
    client.set_model_version_tag(
        name=model_name,
        version=version.version,
        key=MOCK_MODEL_TAG,
        value=MOCK_MODEL_TAG_VALUE,
    )
    client.set_registered_model_alias(model_name, alias, version.version)
    return str(version.version)


def _refuse_to_replace_real_model(
    client: MlflowClient, model_name: str, alias: str, *, force: bool
) -> None:
    try:
        current = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return

    if current.tags.get(MOCK_MODEL_TAG) != MOCK_MODEL_TAG_VALUE and not force:
        raise RuntimeError(
            f"L'alias {model_name}@{alias} désigne déjà un modèle non factice "
            f"(version {current.version}). Utiliser --force pour le remplacer explicitement."
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        # MLflow affiche des emojis dans ses liens de run ; certaines consoles Windows
        # utilisent encore CP-1252 et lèvent sinon UnicodeEncodeError en fin de run.
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="autorise le remplacement d'un alias qui désigne un modèle non factice",
    )
    args = parser.parse_args()
    version = register_mock_model(force=args.force)
    alias = get_settings().api.model_alias
    print(f"Modèle factice enregistré : version {version}, alias {alias}.")


if __name__ == "__main__":
    main()
