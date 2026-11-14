import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from anime_mlops.config import Settings
from anime_mlops.data.dataset import DataVersions
from anime_mlops.features.build import FEATURES
from anime_mlops.training.evaluate import baseline_mae, error_by_group, regression_metrics
from anime_mlops.training.model import OTHER_LABEL, TopNMultiHot, build_model, feature_importances
from anime_mlops.training.train import train_and_log

ROWS = 120


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """Configuration de test : petit modèle, MLflow local dans un dossier temporaire."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("ANIME_MODEL__N_ESTIMATORS", "10")
    monkeypatch.setenv("ANIME_MODEL__N_JOBS", "1")
    monkeypatch.setenv("ANIME_MODEL__TOP_N_STUDIOS", "2")
    monkeypatch.setenv("ANIME_MODEL__TOP_N_TAGS", "2")
    monkeypatch.setenv("ANIME_MLFLOW__TRACKING_URI", f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.chdir(tmp_path)
    return Settings()


@pytest.fixture
def dataset():
    """Jeu synthétique où le score dépend du genre et du statut de suite."""
    rng = np.random.default_rng(0)
    is_action = rng.random(ROWS) < 0.5
    is_sequel = rng.random(ROWS) < 0.3
    features = pd.DataFrame(
        {
            "num_episodes": rng.integers(1, 50, ROWS).astype(float),
            "average_episode_duration": np.where(rng.random(ROWS) < 0.1, np.nan, 1440.0),
            "start_year": rng.integers(1990, 2026, ROWS).astype(float),
            "prequel_score": np.where(is_sequel, 7.5, np.nan),
            "media_type": rng.choice(["tv", "movie", "ova"], ROWS),
            "rating": rng.choice(["pg_13", "r", None], ROWS),
            "source": rng.choice(["MANGA", "ORIGINAL", None], ROWS),
            "start_season": rng.choice(["winter", "spring", "summer", "fall"], ROWS),
            "start_weekday": rng.choice(["monday", "friday", None], ROWS),
            "is_sequel": is_sequel,
            "genres": [["Action", "Drama"] if action else ["Comedy"] for action in is_action],
            "studios": [list(rng.choice(["A", "B", "C", "D"], 1)) for _ in range(ROWS)],
            "tags": [list(rng.choice(["x", "y", "z"], 2, replace=False)) for _ in range(ROWS)],
        },
        index=pd.Index(range(1, ROWS + 1), name="id"),
    )[FEATURES]
    target = pd.Series(
        6.0 + is_action * 1.5 + is_sequel * 0.5 + rng.normal(0, 0.1, ROWS),
        index=features.index,
        name="mean",
    )
    return features, target


@pytest.fixture
def versions():
    return DataVersions(
        mal_commit="a" * 40,
        mal_sha256="b" * 64,
        anilist_extraction_date="2026-11-14",
        anilist_sha256="c" * 64,
        anilist_is_fallback=False,
    )


# --- Encodage multi-libellés -----------------------------------------------------


def test_multi_hot_keeps_most_frequent_labels():
    column = pd.Series([["a", "b"], ["a"], ["a", "c"], ["b"], []])

    encoder = TopNMultiHot(top_n=2).fit(column)

    assert encoder.labels_ == ["a", "b"]
    assert encoder.get_feature_names_out().tolist() == ["a", "b", OTHER_LABEL]
    assert encoder.transform(column).tolist() == [
        [1, 1, 0],
        [1, 0, 0],
        [1, 0, 1],  # "c" est hors du top 2 : colonne « autre »
        [0, 1, 0],
        [0, 0, 0],
    ]


def test_multi_hot_handles_unseen_and_missing_values():
    encoder = TopNMultiHot().fit(pd.Series([["a"], ["b"]]))

    encoded = encoder.transform(pd.Series([["jamais-vu"], None, np.nan, np.array(["a"])]))

    assert encoded.tolist() == [[0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0]]


def test_multi_hot_column_order_is_deterministic():
    first = TopNMultiHot().fit(pd.Series([["b", "a"], ["c"]]))
    second = TopNMultiHot().fit(pd.Series([["c"], ["a", "b"]]))

    assert first.labels_ == second.labels_ == ["a", "b", "c"]


# --- Pipeline --------------------------------------------------------------------


def test_pipeline_learns_and_predicts_from_raw_fields(settings, dataset):
    features, target = dataset
    model = build_model(settings.model, settings.seed).fit(features, target)

    predictions = model.predict(features)

    assert predictions.shape == (ROWS,)
    assert regression_metrics(target, predictions)["mae"] < 0.3


def test_pipeline_accepts_unknown_categories_and_missing_values(settings, dataset):
    features, target = dataset
    model = build_model(settings.model, settings.seed).fit(features, target)
    newcomer = features.head(1).copy()
    newcomer["media_type"] = "format-inconnu"
    newcomer["rating"] = None
    newcomer["num_episodes"] = np.nan
    newcomer["genres"] = [["Genre inédit"]]
    newcomer["tags"] = [[]]

    prediction = model.predict(newcomer)

    assert 0 <= prediction[0] <= 10


def test_training_is_reproducible_with_the_same_seed(settings, dataset):
    features, target = dataset

    first = build_model(settings.model, seed=7).fit(features, target).predict(features)
    second = build_model(settings.model, seed=7).fit(features, target).predict(features)

    assert np.array_equal(first, second)


def test_unsupported_model_type_is_rejected(settings):
    config = settings.model.model_copy(update={"type": "reseau-de-neurones"})

    with pytest.raises(ValueError, match="non pris en charge"):
        build_model(config, settings.seed)


def test_feature_importances_cover_every_encoded_column(settings, dataset):
    features, target = dataset
    model = build_model(settings.model, settings.seed).fit(features, target)

    table = feature_importances(model)

    assert table["importance"].sum() == pytest.approx(1.0)
    assert table["importance"].is_monotonic_decreasing
    assert any("Action" in name for name in table["feature"].head(5))


# --- Évaluation ------------------------------------------------------------------


def test_regression_metrics():
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])

    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx((4 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(-1.0)


def test_baseline_predicts_the_training_mean():
    assert baseline_mae([4.0, 6.0], [5.0, 7.0]) == pytest.approx(1.0)


def test_error_by_group():
    table = error_by_group(pd.Series(["tv", "tv", None]), [1.0, 2.0, 3.0], [2.0, 2.0, 5.0])

    assert table.to_dict("records") == [
        {"groupe": "tv", "mae": 0.5, "effectif": 2},
        {"groupe": "inconnu", "mae": 2.0, "effectif": 1},
    ]


# --- Traçabilité MLflow ----------------------------------------------------------


def test_training_run_is_fully_traced(settings, dataset, versions):
    features, target = dataset

    result = train_and_log(features, target, versions, settings)

    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    run = client.get_run(result.run_id)

    # Hyperparamètres et graine.
    assert run.data.params["seed"] == "42"
    assert run.data.params["model.n_estimators"] == "10"
    assert run.data.params["model.test_size"] == "0.2"
    # Versions des données et du code.
    assert run.data.tags["data.mal_commit"] == "a" * 40
    assert run.data.tags["data.anilist_extraction_date"] == "2026-11-14"
    assert "code.commit" in run.data.tags
    # Métriques.
    assert run.data.metrics["test_mae"] == pytest.approx(result.metrics["test_mae"])
    assert run.data.metrics["test_mae"] < run.data.metrics["baseline_test_mae"]
    assert run.data.metrics["train_rows"] + run.data.metrics["test_rows"] == ROWS
    # Artefacts.
    artifacts = {artifact.path for artifact in client.list_artifacts(result.run_id)}
    assert {
        "feature_importances.csv",
        "test_error_by_media_type.csv",
        "test_predictions.csv",
        "dataset_profile.json",
        "config.json",
    } <= artifacts


def test_logged_config_contains_no_database_credentials(settings, dataset, versions):
    features, target = dataset
    result = train_and_log(features, target, versions, settings)

    path = mlflow.artifacts.download_artifacts(
        run_id=result.run_id,
        artifact_path="config.json",
        tracking_uri=settings.mlflow.tracking_uri,
    )
    config = json.loads(Path(path).read_text("utf-8"))

    assert "database" not in config
    assert config["seed"] == 42


def test_model_is_registered_with_tags_and_can_be_reloaded(settings, dataset, versions):
    features, target = dataset

    result = train_and_log(features, target, versions, settings)

    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    version = client.get_model_version(result.model_name, result.model_version)
    assert version.run_id == result.run_id
    assert version.tags["mal_commit"] == "a" * 40
    assert version.tags["test_mae"] == f"{result.metrics['test_mae']:.4f}"

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    reloaded = mlflow.sklearn.load_model(f"models:/{result.model_name}/{result.model_version}")
    assert reloaded.predict(features.head(5)).shape == (5,)


def test_each_training_creates_a_new_model_version(settings, dataset, versions):
    features, target = dataset

    first = train_and_log(features, target, versions, settings)
    second = train_and_log(features, target, versions, settings)

    assert int(second.model_version) == int(first.model_version) + 1
