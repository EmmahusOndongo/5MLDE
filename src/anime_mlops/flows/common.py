"""Éléments partagés par les flows : alerte en cas d'échec, compte rendu de validation."""

from prefect.artifacts import create_markdown_artifact

from anime_mlops.alerts import Alert, Level, send_alert
from anime_mlops.config import get_settings
from anime_mlops.data.validate import ValidationReport


def alert_on_failure(flow, flow_run, state) -> None:
    """Crochet Prefect : tout flow en échec ou planté émet une alerte critique."""
    send_alert(
        Alert(
            level=Level.CRITICAL,
            title=f"Flow « {flow.name} » en échec",
            message=state.message or "Échec sans message.",
            context={
                "flow": flow.name,
                "flow_run": flow_run.name,
                "flow_run_id": str(flow_run.id),
                "parameters": {key: str(value) for key, value in flow_run.parameters.items()},
            },
        ),
        get_settings().alerts,
    )


def publish_validation_reports(key: str, title: str, reports: list[ValidationReport]) -> None:
    """Rend les contrôles qualité lisibles dans l'interface Prefect (onglet Artifacts)."""
    lines = [
        f"# {title}",
        "",
        "| Contrôle | Version des données | Lignes | Règles | Verdict |",
        "|---|---|---|---|---|",
    ]
    for report in reports:
        verdict = "conforme" if report.success else f"**REJETÉ** ({len(report.failures)} règle(s))"
        lines.append(
            f"| {report.source} | {report.data_version} | {report.row_count} "
            f"| {report.evaluated_rules} | {verdict} |"
        )
    failures = [(report.source, failure) for report in reports for failure in report.failures]
    if failures:
        lines += ["", "## Règles en échec", ""]
        lines += [f"- `{source}` — {failure}" for source, failure in failures]
    create_markdown_artifact(key=key, markdown="\n".join(lines), description=title)
