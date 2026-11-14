"""Mise en service d'une version du modèle et retour arrière.

Usage :
    python -m anime_mlops.training.promote status
    python -m anime_mlops.training.promote promote [--version N]
    python -m anime_mlops.training.promote rollback [--version N]

L'API sert toujours la version qui porte l'alias ``champion``. Ce module décide qui
le reçoit, et laisse sur chaque version des tags expliquant la décision.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from anime_mlops.config import PromotionRules, Settings, get_settings, load_promotion_rules

logger = logging.getLogger(__name__)

# Tags posés sur les versions du registre.
TAG_STATUS = "status"
TAG_REASON = "decision_reason"
TAG_DECIDED_AT = "decided_at"
TAG_REPLACED = "replaced_version"
TAG_ROLLBACK_FROM = "rollback_from_version"


class Status(StrEnum):
    CHAMPION = "champion"  # en service
    CHALLENGER = "challenger"  # valide, mais n'a pas battu le champion
    REJECTED = "rejected"  # ne remplit pas les critères d'acceptation
    ARCHIVED = "archived"  # ancien champion, remplacé par une meilleure version
    ROLLED_BACK = "rolled_back"  # champion retiré par un retour arrière


@dataclass(frozen=True)
class Decision:
    version: str
    status: Status
    reason: str
    previous_champion: str | None = None

    @property
    def promoted(self) -> bool:
        return self.status is Status.CHAMPION


class PromotionError(RuntimeError):
    """Opération impossible sur le registre (version absente, rien à restaurer…)."""


def decide(
    candidate_metric: float,
    champion_metric: float | None,
    baseline_metric: float | None,
    data_validated: bool,
    rules: PromotionRules,
) -> tuple[Status, str]:
    """Règle de décision pure : ne dépend que des chiffres, pas de MLflow."""
    name = rules.metric_key

    if rules.require_validated_data and not data_validated:
        return Status.REJECTED, "contrôles qualité des données non passés"

    if rules.must_beat_baseline:
        if baseline_metric is None:
            return Status.REJECTED, "référence naïve absente du run : comparaison impossible"
        if not _is_better_or_equal(candidate_metric, baseline_metric, 0.0, rules, strict=True):
            return Status.REJECTED, (
                f"{name}={candidate_metric:.4f} ne bat pas la référence naïve "
                f"({baseline_metric:.4f})"
            )

    if champion_metric is None:
        return Status.CHAMPION, f"aucun champion en place ; {name}={candidate_metric:.4f}"

    if _is_better_or_equal(candidate_metric, champion_metric, rules.tolerance, rules):
        return Status.CHAMPION, (
            f"{name}={candidate_metric:.4f} au moins aussi bon que le champion "
            f"({champion_metric:.4f}, tolérance {rules.tolerance})"
        )
    return Status.CHALLENGER, (
        f"{name}={candidate_metric:.4f} moins bon que le champion ({champion_metric:.4f})"
    )


def promote(settings: Settings, rules: PromotionRules, version: str | None = None) -> Decision:
    """Évalue une version (la plus récente par défaut) et pose alias et tags en conséquence."""
    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    model_name = settings.mlflow.registered_model_name
    candidate = _get_version(client, model_name, version)
    champion = _get_by_alias(client, model_name, rules.aliases.champion)

    if champion is not None and champion.version == candidate.version:
        return Decision(candidate.version, Status.CHAMPION, "version déjà en service")

    candidate_run = client.get_run(candidate.run_id)
    champion_metric = (
        client.get_run(champion.run_id).data.metrics.get(rules.metric_key) if champion else None
    )
    status, reason = decide(
        candidate_metric=candidate_run.data.metrics[rules.metric_key],
        champion_metric=champion_metric,
        baseline_metric=candidate_run.data.metrics.get(rules.baseline_key),
        data_validated=candidate_run.data.tags.get("data.validation") == "passed",
        rules=rules,
    )

    previous = champion.version if champion and status is Status.CHAMPION else None
    if status is Status.CHAMPION:
        client.set_registered_model_alias(model_name, rules.aliases.champion, candidate.version)
        _drop_alias_if_on(client, model_name, rules.aliases.challenger, candidate.version)
        if champion is not None:
            _tag(
                client,
                model_name,
                champion.version,
                Status.ARCHIVED,
                f"remplacé par la v{candidate.version}",
            )
            client.set_model_version_tag(
                model_name, candidate.version, TAG_REPLACED, champion.version
            )
    elif status is Status.CHALLENGER:
        client.set_registered_model_alias(model_name, rules.aliases.challenger, candidate.version)

    _tag(client, model_name, candidate.version, status, reason)
    logger.info("Version %s de %s : %s (%s).", candidate.version, model_name, status.value, reason)
    return Decision(candidate.version, status, reason, previous)


def rollback(settings: Settings, rules: PromotionRules, version: str | None = None) -> Decision:
    """Remet l'alias champion sur une version antérieure.

    Sans version explicite, on revient à celle que le champion actuel avait remplacée.
    """
    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    model_name = settings.mlflow.registered_model_name
    current = _get_by_alias(client, model_name, rules.aliases.champion)
    if current is None:
        raise PromotionError("Aucun champion en place : rien à annuler.")

    target_version = version or current.tags.get(TAG_REPLACED)
    if target_version is None:
        raise PromotionError(
            f"La v{current.version} n'a remplacé aucune version : précisez --version."
        )
    target = _get_version(client, model_name, str(target_version))
    if target.version == current.version:
        raise PromotionError(f"La v{target.version} est déjà en service.")

    client.set_registered_model_alias(model_name, rules.aliases.champion, target.version)
    _tag(
        client,
        model_name,
        current.version,
        Status.ROLLED_BACK,
        f"retour arrière vers la v{target.version}",
    )
    reason = f"retour arrière depuis la v{current.version}"
    _tag(client, model_name, target.version, Status.CHAMPION, reason)
    client.set_model_version_tag(model_name, target.version, TAG_ROLLBACK_FROM, current.version)

    logger.warning(
        "Retour arrière : %s sert désormais la v%s (au lieu de la v%s).",
        model_name,
        target.version,
        current.version,
    )
    return Decision(target.version, Status.CHAMPION, reason, current.version)


def registry_status(settings: Settings, rules: PromotionRules) -> list[dict]:
    """État lisible du registre : une ligne par version, de la plus récente à la plus ancienne."""
    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    model_name = settings.mlflow.registered_model_name
    versions = client.search_model_versions(f"name = '{model_name}'")
    rows = []
    for item in sorted(versions, key=lambda v: int(v.version), reverse=True):
        # La recherche ne renvoie pas les alias : on relit la version complète.
        full = _Version.of(client.get_model_version(model_name, item.version))
        rows.append(
            {
                "version": full.version,
                "alias": ",".join(full.aliases) or "-",
                "status": full.tags.get(TAG_STATUS, "-"),
                rules.metric_key: full.tags.get(rules.metric_key, "-"),
                "mal_commit": full.tags.get("mal_commit", "-")[:8],
                "raison": full.tags.get(TAG_REASON, "-"),
            }
        )
    return rows


def _is_better_or_equal(
    candidate: float,
    reference: float,
    tolerance: float,
    rules: PromotionRules,
    strict: bool = False,
) -> bool:
    if not rules.lower_is_better:
        candidate, reference = -candidate, -reference
    return candidate < reference if strict else candidate <= reference + tolerance


@dataclass(frozen=True)
class _Version:
    """Vue minimale d'une version du registre, au numéro toujours exprimé en chaîne."""

    version: str
    run_id: str
    tags: dict[str, str]
    aliases: tuple[str, ...]

    @classmethod
    def of(cls, item: ModelVersion) -> "_Version":
        return cls(str(item.version), item.run_id, dict(item.tags), tuple(item.aliases))


def _get_version(client: MlflowClient, model_name: str, version: str | None) -> _Version:
    try:
        if version is not None:
            return _Version.of(client.get_model_version(model_name, str(version)))
        versions = client.search_model_versions(f"name = '{model_name}'")
    except MlflowException as exc:
        raise PromotionError(
            f"Version {version} de {model_name} introuvable : {exc.message}"
        ) from exc
    if not versions:
        raise PromotionError(f"Aucune version de {model_name} dans le registre.")
    latest = max(versions, key=lambda item: int(item.version))
    return _Version.of(client.get_model_version(model_name, latest.version))


def _get_by_alias(client: MlflowClient, model_name: str, alias: str) -> _Version | None:
    try:
        return _Version.of(client.get_model_version_by_alias(model_name, alias))
    except MlflowException:
        return None


def _drop_alias_if_on(client: MlflowClient, model_name: str, alias: str, version: str) -> None:
    holder = _get_by_alias(client, model_name, alias)
    if holder is not None and holder.version == version:
        client.delete_registered_model_alias(model_name, alias)


def _tag(client: MlflowClient, model_name: str, version: str, status: Status, reason: str) -> None:
    for key, value in (
        (TAG_STATUS, status.value),
        (TAG_REASON, reason),
        (TAG_DECIDED_AT, datetime.now(UTC).isoformat(timespec="seconds")),
    ):
        client.set_model_version_tag(model_name, version, key, value)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Mise en service du modèle (registre MLflow).")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="affiche les versions, leurs alias et leurs tags")
    for name, text in (
        ("promote", "applique les règles de promotion (dernière version par défaut)"),
        ("rollback", "remet le champion sur une version antérieure"),
    ):
        commands.add_parser(name, help=text).add_argument("--version", help="numéro de version")
    arguments = parser.parse_args(argv)

    settings, rules = get_settings(), load_promotion_rules()
    try:
        if arguments.command == "status":
            rows = registry_status(settings, rules)
            for row in rows:
                print("  ".join(f"{key}={value}" for key, value in row.items()))  # noqa: T201
            if not rows:
                print("Aucune version dans le registre.")  # noqa: T201
        elif arguments.command == "promote":
            decision = promote(settings, rules, arguments.version)
            print(f"v{decision.version} : {decision.status.value} ({decision.reason})")  # noqa: T201
        else:
            decision = rollback(settings, rules, arguments.version)
            print(f"champion = v{decision.version} ({decision.reason})")  # noqa: T201
    except PromotionError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
