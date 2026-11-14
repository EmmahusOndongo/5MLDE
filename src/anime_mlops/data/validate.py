"""Contrôles qualité bloquants (Great Expectations), exécutés avant tout prétraitement.

Chaque source a sa suite de règles. Une validation produit toujours un rapport JSON
archivé ; en cas de non-conformité, ``ensure_valid`` lève ``DataQualityError`` et
l'entraînement ne démarre pas.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from anime_mlops.config import QualityConfig

# Pas de télémétrie vers l'éditeur : à fixer avant l'import de la bibliothèque.
os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")

import great_expectations as gx  # noqa: E402
import great_expectations.expectations as gxe  # noqa: E402

logger = logging.getLogger(__name__)

MAL_REQUIRED_COLUMNS = [
    "id",
    "title",
    "mean",
    "num_episodes",
    "start_date",
    "media_type",
    "status",
    "rating",
    "average_episode_duration",
    "genres",
    "studios",
]
MAL_STATUSES = ["finished_airing", "currently_airing", "not_yet_aired"]
MAL_MEDIA_TYPES = [
    "tv",
    "movie",
    "ova",
    "ona",
    "special",
    "tv_special",
    "music",
    "cm",
    "pv",
    "unknown",
]
# Date complète, ou partielle (année seule, année-mois) pour les animes anciens ou annoncés.
MAL_DATE_PATTERN = r"^\d{4}(?:-\d{2}){0,2}$"

ANILIST_REQUIRED_COLUMNS = [
    "id_mal",
    "anilist_id",
    "format",
    "source",
    "episodes",
    "tags",
    "is_sequel",
    "prequel_id_mal",
]

# Colonnes du jeu d'entraînement dont le taux de valeurs manquantes est plafonné.
MERGED_MONITORED_COLUMNS = [
    "media_type",
    "rating",
    "num_episodes",
    "average_episode_duration",
    "start_date",
    "genres",
    "studios",
    "source",
    "tags",
]


@dataclass(frozen=True)
class CheckFailure:
    """Une règle non respectée, décrite de façon lisible pour l'alerte et les journaux."""

    rule: str
    column: str | None
    detail: str

    def __str__(self) -> str:
        target = f"[{self.column}] " if self.column else ""
        return f"{target}{self.rule} : {self.detail}"


@dataclass(frozen=True)
class ValidationReport:
    source: str
    data_version: str
    success: bool
    evaluated_rules: int
    row_count: int
    failures: tuple[CheckFailure, ...] = ()
    report_path: Path | None = None
    details: dict = field(default_factory=dict, repr=False, compare=False)


class DataQualityError(RuntimeError):
    """Lot de données non conforme : la chaîne doit s'arrêter."""

    def __init__(self, report: ValidationReport):
        self.report = report
        lines = "\n".join(f"  - {failure}" for failure in report.failures)
        super().__init__(
            f"Données {report.source} (version {report.data_version}) non conformes : "
            f"{len(report.failures)} règle(s) sur {report.evaluated_rules} en échec.\n{lines}\n"
            f"Rapport : {report.report_path}"
        )


def mal_suite(quality: QualityConfig) -> gx.ExpectationSuite:
    """Règles du catalogue MyAnimeList brut."""
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(column_set=MAL_REQUIRED_COLUMNS, exact_match=False),
        gxe.ExpectTableRowCountToBeBetween(min_value=quality.min_rows),
        gxe.ExpectColumnValuesToNotBeNull(column="id"),
        gxe.ExpectColumnValuesToBeUnique(column="id"),
        gxe.ExpectColumnValuesToNotBeNull(column="title"),
        # La cible : bornée, et présente sur une part suffisante du catalogue.
        gxe.ExpectColumnValuesToBeBetween(
            column="mean", min_value=quality.target_min, max_value=quality.target_max
        ),
        gxe.ExpectColumnValuesToNotBeNull(column="mean", mostly=quality.min_target_rate),
        # Le statut pilote le choix des lignes d'entraînement : aucune valeur inconnue tolérée.
        gxe.ExpectColumnValuesToBeInSet(column="status", value_set=MAL_STATUSES),
        gxe.ExpectColumnValuesToBeInSet(
            column="media_type", value_set=MAL_MEDIA_TYPES, mostly=quality.min_known_category_rate
        ),
        gxe.ExpectColumnValuesToBeBetween(column="num_episodes", min_value=0),
        gxe.ExpectColumnValuesToBeBetween(column="average_episode_duration", min_value=0),
        gxe.ExpectColumnValuesToMatchRegex(
            column="start_date", regex=MAL_DATE_PATTERN, mostly=quality.min_known_category_rate
        ),
    ]
    return gx.ExpectationSuite(name="mal", expectations=expectations)


def anilist_suite(quality: QualityConfig) -> gx.ExpectationSuite:
    """Règles de l'extraction AniList aplatie."""
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(column_set=ANILIST_REQUIRED_COLUMNS, exact_match=False),
        gxe.ExpectTableRowCountToBeBetween(min_value=quality.min_rows),
        # Clé de jointure : présente et sans doublon.
        gxe.ExpectColumnValuesToNotBeNull(column="id_mal"),
        gxe.ExpectColumnValuesToBeUnique(column="id_mal"),
        gxe.ExpectColumnValuesToNotBeNull(column="anilist_id"),
        gxe.ExpectColumnValuesToBeUnique(column="anilist_id"),
        gxe.ExpectColumnValuesToNotBeNull(column="tags"),
        gxe.ExpectColumnValuesToNotBeNull(column="is_sequel"),
        gxe.ExpectColumnValuesToBeInSet(column="is_sequel", value_set=[True, False]),
        gxe.ExpectColumnValuesToBeBetween(column="episodes", min_value=0),
    ]
    return gx.ExpectationSuite(name="anilist", expectations=expectations)


def merged_suite(quality: QualityConfig) -> gx.ExpectationSuite:
    """Règles du jeu d'entraînement fusionné (MAL enrichi par AniList)."""
    completeness = 1 - quality.max_missing_rate
    expectations = [
        gxe.ExpectTableRowCountToBeBetween(min_value=quality.min_rows),
        gxe.ExpectColumnValuesToBeUnique(column="id"),
        gxe.ExpectColumnValuesToNotBeNull(column="mean"),
        gxe.ExpectColumnValuesToBeBetween(
            column="mean", min_value=quality.target_min, max_value=quality.target_max
        ),
        # Taux de correspondance MAL ↔ AniList.
        gxe.ExpectColumnValuesToBeInSet(
            column="has_anilist", value_set=[True], mostly=quality.min_join_match_rate
        ),
        # Cohérence inter-sources (les lignes où une source ignore la valeur sont exclues).
        gxe.ExpectColumnValuesToBeInSet(
            column="episodes_agree", value_set=[True], mostly=quality.min_episode_agreement_rate
        ),
        *(
            gxe.ExpectColumnValuesToNotBeNull(column=column, mostly=completeness)
            for column in MERGED_MONITORED_COLUMNS
        ),
    ]
    return gx.ExpectationSuite(name="fusion", expectations=expectations)


def validate(
    data: pd.DataFrame,
    suite: gx.ExpectationSuite,
    data_version: str,
    reports_dir: Path,
    now: datetime | None = None,
) -> ValidationReport:
    """Exécute la suite sur le lot et archive le rapport, que le lot soit conforme ou non."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = gx.data_context.types.base.ProgressBarsConfig(globally=False)
    batch_definition = (
        context.data_sources.add_pandas(suite.name)
        .add_dataframe_asset(suite.name)
        .add_batch_definition_whole_dataframe("lot")
    )
    result = batch_definition.get_batch(batch_parameters={"dataframe": data}).validate(suite)

    failures = tuple(_describe(check) for check in result.results if not check.success)
    report = ValidationReport(
        source=suite.name,
        data_version=data_version,
        success=bool(result.success),
        evaluated_rules=len(result.results),
        row_count=len(data),
        failures=failures,
        details=result.to_json_dict(),
    )
    report = _archive(report, reports_dir, now or datetime.now(UTC))

    if report.success:
        logger.info(
            "Données %s (%s) conformes : %d règles vérifiées sur %d lignes.",
            report.source,
            data_version,
            report.evaluated_rules,
            report.row_count,
        )
    else:
        logger.error(
            "Données %s (%s) NON conformes : %s", report.source, data_version, list(failures)
        )
    return report


def ensure_valid(report: ValidationReport) -> ValidationReport:
    """Point de blocage : lève ``DataQualityError`` si le lot n'est pas conforme."""
    if not report.success:
        raise DataQualityError(report)
    return report


def validate_mal(
    data: pd.DataFrame, commit: str, quality: QualityConfig, reports_dir: Path
) -> ValidationReport:
    return ensure_valid(validate(data, mal_suite(quality), commit, reports_dir))


def validate_anilist(
    data: pd.DataFrame, extraction_date: str, quality: QualityConfig, reports_dir: Path
) -> ValidationReport:
    return ensure_valid(validate(data, anilist_suite(quality), extraction_date, reports_dir))


def validate_merged(
    training: pd.DataFrame, data_version: str, quality: QualityConfig, reports_dir: Path
) -> ValidationReport:
    return ensure_valid(validate(training, merged_suite(quality), data_version, reports_dir))


def _describe(check) -> CheckFailure:
    """Traduit un résultat Great Expectations en une ligne exploitable."""
    kwargs = dict(check.expectation_config.kwargs)
    result = check.result or {}
    exception = (check.exception_info or {}).get("exception_message")
    if not exception:
        # Selon la règle, l'erreur est rangée par métrique : on prend la première.
        nested = [
            info.get("exception_message")
            for info in (check.exception_info or {}).values()
            if isinstance(info, dict) and info.get("raised_exception")
        ]
        exception = nested[0] if nested else None

    if exception:
        detail = f"contrôle impossible ({exception.strip().splitlines()[0]})"
    elif "unexpected_count" in result:
        examples = result.get("partial_unexpected_list", [])[:5]
        detail = (
            f"{result['unexpected_count']} valeur(s) invalide(s) "
            f"({result.get('unexpected_percent', 0):.2f} %), exemples : {examples}"
        )
    elif (result.get("details") or {}).get("mismatched"):
        missing = result["details"]["mismatched"].get("missing", [])
        detail = f"colonne(s) attendue(s) absente(s) : {missing}"
    else:
        detail = f"valeur observée : {result.get('observed_value')}"

    return CheckFailure(
        rule=check.expectation_config.type, column=kwargs.get("column"), detail=detail
    )


def _archive(report: ValidationReport, reports_dir: Path, now: datetime) -> ValidationReport:
    """Écrit le rapport complet sur disque et renvoie le rapport avec son chemin."""
    status = "ok" if report.success else "echec"
    path = (
        reports_dir
        / "validation"
        / f"{now:%Y%m%dT%H%M%S}_{report.source}_{report.data_version[:12]}_{status}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": report.source,
        "data_version": report.data_version,
        "validated_at": now.isoformat(),
        "success": report.success,
        "row_count": report.row_count,
        "evaluated_rules": report.evaluated_rules,
        "failures": [str(failure) for failure in report.failures],
        "great_expectations": report.details,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")
    return ValidationReport(
        source=report.source,
        data_version=report.data_version,
        success=report.success,
        evaluated_rules=report.evaluated_rules,
        row_count=report.row_count,
        failures=report.failures,
        report_path=path,
        details=report.details,
    )
