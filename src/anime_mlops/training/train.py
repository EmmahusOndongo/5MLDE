"""Entraînement, évaluation et traçabilité MLflow.

Usage : ``python -m anime_mlops.training.train``

Chaque exécution enregistre de quoi la rejouer à l'identique : hyperparamètres, graine,
versions des données, version du code, ainsi que les métriques et artefacts produits.
Le modèle est ajouté au registre comme nouvelle version ; sa mise en service (alias)
est décidée ensuite par les règles de promotion (``anime_mlops.training.promote``).
"""

import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import pandas as pd
import sklearn
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from anime_mlops.config import Settings, get_settings, load_promotion_rules
from anime_mlops.data.dataset import DataVersions, prepare_training_data
from anime_mlops.data.validate import ValidationReport
from anime_mlops.features.build import FEATURES, split_features_target
from anime_mlops.training.evaluate import baseline_mae, error_by_group, regression_metrics
from anime_mlops.training.model import build_model, feature_importances
from anime_mlops.training.promote import promote
from anime_mlops.versioning import code_version

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    model_name: str
    model_version: str
    metrics: dict[str, float]


def train_and_log(
    features: pd.DataFrame,
    target: pd.Series,
    versions: DataVersions,
    settings: Settings,
    reports: tuple[ValidationReport, ...] = (),
) -> TrainingResult:
    """Entraîne le modèle, l'évalue et trace l'ensemble dans MLflow."""
    config = settings.model
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=config.test_size, random_state=settings.seed
    )

    model = build_model(config, settings.seed)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    test_predictions = model.predict(x_test)
    test_metrics = regression_metrics(y_test, test_predictions)
    train_metrics = regression_metrics(y_train, model.predict(x_train))
    metrics = {
        **{f"test_{name}": value for name, value in test_metrics.items()},
        **{f"train_{name}": value for name, value in train_metrics.items()},
        "baseline_test_mae": baseline_mae(y_train, y_test),
        "fit_seconds": fit_seconds,
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
    }

    commit, dirty = code_version()
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"train-{versions.label}") as run:
        mlflow.log_params(
            {
                "seed": settings.seed,
                **{f"model.{name}": value for name, value in config.model_dump().items()},
                "n_features": len(FEATURES),
            }
        )
        mlflow.set_tags(
            {
                **{f"data.{name}": str(value) for name, value in asdict(versions).items()},
                "code.commit": commit,
                "code.dirty": str(dirty),
                "sklearn.version": sklearn.__version__,
                "data.validation": "passed" if reports and all(r.success for r in reports) else "",
            }
        )
        mlflow.log_metrics(metrics)
        _log_artifacts(model, features, target, x_test, y_test, test_predictions, settings, reports)

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            # Le pipeline contient un transformateur du projet : sérialisation cloudpickle.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=settings.mlflow.registered_model_name,
            input_example=x_test.head(3),
        )

    model_version = str(model_info.registered_model_version)
    _tag_model_version(settings, model_version, versions, metrics, commit, dirty)
    logger.info(
        "Modèle %s v%s enregistré : MAE test %.4f (référence naïve %.4f), run %s.",
        settings.mlflow.registered_model_name,
        model_version,
        metrics["test_mae"],
        metrics["baseline_test_mae"],
        run.info.run_id,
    )
    return TrainingResult(
        run.info.run_id, settings.mlflow.registered_model_name, model_version, metrics
    )


def _log_artifacts(model, features, target, x_test, y_test, predictions, settings, reports) -> None:
    """Dépose dans le run tout ce qui aide à comprendre et à rejouer l'entraînement."""
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)

        feature_importances(model).to_csv(folder / "feature_importances.csv", index=False)
        error_by_group(x_test["media_type"], y_test, predictions).to_csv(
            folder / "test_error_by_media_type.csv", index=False
        )
        pd.DataFrame(
            {"id": x_test.index, "score_reel": y_test.to_numpy(), "score_predit": predictions}
        ).to_csv(folder / "test_predictions.csv", index=False)

        profile = {
            "rows": len(features),
            "features": FEATURES,
            "target": target.describe().round(4).to_dict(),
            "missing_rate": features.isna().mean().round(4).to_dict(),
        }
        (folder / "dataset_profile.json").write_text(
            json.dumps(profile, indent=2, ensure_ascii=False), "utf-8"
        )
        # Configuration effective, sans la section base de données (identifiants).
        (folder / "config.json").write_text(
            settings.model_dump_json(indent=2, exclude={"database"}), "utf-8"
        )
        mlflow.log_artifacts(str(folder))

    for report in reports:
        if report.report_path is not None:
            mlflow.log_artifact(str(report.report_path), artifact_path="validation")


def _tag_model_version(settings, model_version, versions, metrics, commit, dirty) -> None:
    """Tags de la version dans le registre : origine des données, du code, et performance."""
    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    tags = {
        "mal_commit": versions.mal_commit,
        "anilist_extraction_date": versions.anilist_extraction_date,
        "anilist_is_fallback": str(versions.anilist_is_fallback),
        "code_commit": commit,
        "code_dirty": str(dirty),
        "test_mae": f"{metrics['test_mae']:.4f}",
        "test_rmse": f"{metrics['test_rmse']:.4f}",
        "train_rows": str(int(metrics["train_rows"])),
    }
    for key, value in tags.items():
        client.set_model_version_tag(
            settings.mlflow.registered_model_name, model_version, key, value
        )


def main() -> None:
    # MLflow affiche des émojis : sans UTF-8, la console Windows fait échouer le run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = get_settings()
    prepared = prepare_training_data(settings)
    features, target = split_features_target(prepared.training)
    result = train_and_log(features, target, prepared.versions, settings, prepared.reports)
    # Mise en service éventuelle, selon les règles de configs/promotion.yaml.
    promote(settings, load_promotion_rules(), result.model_version)


if __name__ == "__main__":
    main()
