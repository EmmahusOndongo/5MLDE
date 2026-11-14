"""Configuration typée du projet.

Ordre de priorité (du plus fort au plus faible) : arguments explicites,
variables d'environnement ``ANIME_*``, fichier ``.env``, puis ``configs/config.yaml``.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_PROMOTION_FILE = PROJECT_ROOT / "configs" / "promotion.yaml"
CONFIG_FILE_ENV_VAR = "ANIME_CONFIG_FILE"


class PathsConfig(BaseModel):
    data_dir: Path
    raw_dir: Path
    cache_dir: Path
    reports_dir: Path


class MalSourceConfig(BaseModel):
    repo: str
    file_path: str
    commit: str | None = None

    def raw_url(self, commit: str | None = None) -> str:
        """URL de téléchargement du CSV pour un commit donné (version des données)."""
        ref = commit or self.commit
        if not ref:
            raise ValueError("Aucun hash de commit MAL fourni : version des données requise.")
        return f"https://raw.githubusercontent.com/{self.repo}/{ref}/{self.file_path}"


class AnilistSourceConfig(BaseModel):
    url: str
    page_size: int = Field(gt=0, le=50)
    max_retries: int = Field(ge=0)
    retry_delay_seconds: float = Field(ge=0)
    # Pause entre deux pages, pour rester sous la limite de requêtes de l'API.
    request_interval_seconds: float = Field(ge=0)
    # Plafond de lots (None = tout le catalogue) ; utile pour une démonstration rapide.
    max_batches: int | None = Field(default=None, gt=0)


class SourcesConfig(BaseModel):
    mal: MalSourceConfig
    anilist: AnilistSourceConfig


class QualityConfig(BaseModel):
    min_rows: int = Field(gt=0)
    min_join_match_rate: float = Field(ge=0, le=1)
    max_missing_rate: float = Field(ge=0, le=1)
    min_episode_agreement_rate: float = Field(ge=0, le=1)
    target_min: float
    target_max: float
    # Part minimale du catalogue MAL qui doit avoir un score (la cible).
    min_target_rate: float = Field(ge=0, le=1)
    # Part minimale de valeurs appartenant aux catégories connues (tolère une nouveauté marginale).
    min_known_category_rate: float = Field(ge=0, le=1)


class ModelConfig(BaseModel):
    type: str
    n_estimators: int = Field(gt=0)
    max_depth: int | None = Field(default=None, gt=0)
    min_samples_leaf: int = Field(gt=0)
    n_jobs: int
    test_size: float = Field(gt=0, lt=1)
    top_n_studios: int = Field(gt=0)
    top_n_tags: int = Field(gt=0)


class MlflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


class DatabaseConfig(BaseModel):
    host: str
    port: int
    name: str
    user: str
    # Jamais dans le YAML : fourni par ANIME_DATABASE__PASSWORD (.env ou environnement).
    password: SecretStr = SecretStr("")

    @property
    def url(self) -> str:
        """URL SQLAlchemy (contient le mot de passe : ne pas journaliser)."""
        password = self.password.get_secret_value()
        return f"postgresql+psycopg://{self.user}:{password}@{self.host}:{self.port}/{self.name}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANIME_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    seed: int
    paths: PathsConfig
    sources: SourcesConfig
    quality: QualityConfig
    model: ModelConfig
    mlflow: MlflowConfig
    database: DatabaseConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = Path(os.environ.get(CONFIG_FILE_ENV_VAR, DEFAULT_CONFIG_FILE))
        if not yaml_file.is_file():
            raise FileNotFoundError(f"Fichier de configuration introuvable : {yaml_file}")
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
        return init_settings, env_settings, dotenv_settings, yaml_settings


class PromotionAliases(BaseModel):
    champion: str
    challenger: str


class PromotionRules(BaseModel):
    """Règles de mise en service, lues dans ``configs/promotion.yaml``."""

    decision_metric: str
    lower_is_better: bool
    tolerance: float = Field(ge=0)
    require_validated_data: bool
    must_beat_baseline: bool
    aliases: PromotionAliases

    @property
    def metric_key(self) -> str:
        """Nom de la métrique de décision dans le run et dans les tags de version."""
        return f"test_{self.decision_metric}"

    @property
    def baseline_key(self) -> str:
        return f"baseline_test_{self.decision_metric}"


def load_promotion_rules(path: Path = DEFAULT_PROMOTION_FILE) -> PromotionRules:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de règles de promotion introuvable : {path}")
    return PromotionRules.model_validate(yaml.safe_load(path.read_text("utf-8")))


@lru_cache
def get_settings() -> Settings:
    """Configuration chargée une seule fois par processus."""
    return Settings()
