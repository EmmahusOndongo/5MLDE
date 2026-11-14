import pytest
from pydantic import ValidationError

from anime_mlops.config import CONFIG_FILE_ENV_VAR, Settings


@pytest.fixture(autouse=True)
def _isolate_from_local_env(monkeypatch):
    """Les tests ne dépendent ni du .env local ni des variables de la machine."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.delenv("ANIME_DATABASE__PASSWORD", raising=False)


def test_default_config_loads():
    settings = Settings()

    assert settings.seed == 42
    assert settings.model.n_estimators > 0
    assert 0 < settings.quality.min_join_match_rate <= 1
    assert settings.database.name == "anime"
    assert settings.api.model_alias == "champion"


def test_env_variable_overrides_yaml(monkeypatch):
    monkeypatch.setenv("ANIME_SEED", "7")
    monkeypatch.setenv("ANIME_MODEL__N_ESTIMATORS", "10")
    monkeypatch.setenv("ANIME_DATABASE__HOST", "postgres")

    settings = Settings()

    assert settings.seed == 7
    assert settings.model.n_estimators == 10
    assert settings.database.host == "postgres"


def test_invalid_value_is_rejected(monkeypatch):
    monkeypatch.setenv("ANIME_QUALITY__MIN_JOIN_MATCH_RATE", "1.5")

    with pytest.raises(ValidationError):
        Settings()


def test_missing_config_file_fails_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv(CONFIG_FILE_ENV_VAR, str(tmp_path / "absent.yaml"))

    with pytest.raises(FileNotFoundError):
        Settings()


def test_database_password_is_not_exposed(monkeypatch):
    monkeypatch.setenv("ANIME_DATABASE__PASSWORD", "s3cret")

    settings = Settings()

    assert "s3cret" not in repr(settings)
    assert "s3cret" in settings.database.url


def test_database_credentials_are_url_encoded(monkeypatch):
    monkeypatch.setenv("ANIME_DATABASE__PASSWORD", "p@ss/word")

    url = Settings().database.url

    assert "p%40ss%2Fword" in url
    assert "p@ss/word" not in url


def test_mal_raw_url_requires_a_commit():
    mal = Settings().sources.mal

    with pytest.raises(ValueError, match="commit"):
        mal.raw_url()

    assert mal.raw_url("abc123").endswith("/abc123/data/anime-standalone.csv")
