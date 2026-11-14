import pytest
from mlflow import MlflowClient

from anime_mlops.config import Settings, load_promotion_rules
from anime_mlops.training import promote as p
from anime_mlops.training.promote import Status


@pytest.fixture
def rules():
    return load_promotion_rules()


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("ANIME_MLFLOW__TRACKING_URI", f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    return Settings()


@pytest.fixture
def registry(settings):
    """Fabrique de versions : un run avec ses métriques, enregistré dans le registre."""
    client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    name = settings.mlflow.registered_model_name
    client.create_registered_model(name)
    experiment = client.create_experiment("tests")

    def add_version(mae, baseline=0.70, validated=True) -> str:
        run = client.create_run(experiment)
        client.log_metric(run.info.run_id, "test_mae", mae)
        if baseline is not None:
            client.log_metric(run.info.run_id, "baseline_test_mae", baseline)
        client.set_tag(run.info.run_id, "data.validation", "passed" if validated else "")
        version = client.create_model_version(
            name, source=f"runs:/{run.info.run_id}/model", run_id=run.info.run_id
        )
        return str(version.version)

    add_version.client = client
    add_version.name = name
    return add_version


def alias_holder(registry, alias):
    try:
        return str(registry.client.get_model_version_by_alias(registry.name, alias).version)
    except Exception:  # noqa: BLE001
        return None


def tags(registry, version):
    return registry.client.get_model_version(registry.name, version).tags


# --- Règle de décision (pure) ----------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "champion", "expected"),
    [
        (0.38, None, Status.CHAMPION),  # premier modèle
        (0.35, 0.38, Status.CHAMPION),  # meilleur
        (0.38, 0.38, Status.CHAMPION),  # égal : « au moins aussi bien »
        (0.40, 0.38, Status.CHALLENGER),  # moins bon
    ],
)
def test_decision_against_the_champion(rules, candidate, champion, expected):
    status, reason = p.decide(candidate, champion, 0.70, True, rules)

    assert status is expected
    assert f"{candidate:.4f}" in reason


def test_tolerance_accepts_a_small_degradation(rules):
    tolerant = rules.model_copy(update={"tolerance": 0.05})

    assert p.decide(0.40, 0.38, 0.70, True, tolerant)[0] is Status.CHAMPION
    assert p.decide(0.44, 0.38, 0.70, True, tolerant)[0] is Status.CHALLENGER


def test_unvalidated_data_is_rejected_even_with_a_better_score(rules):
    status, reason = p.decide(0.10, 0.38, 0.70, False, rules)

    assert status is Status.REJECTED
    assert "qualité" in reason


@pytest.mark.parametrize("candidate", [0.70, 0.90])
def test_model_not_beating_the_naive_baseline_is_rejected(rules, candidate):
    status, reason = p.decide(candidate, None, 0.70, True, rules)

    assert status is Status.REJECTED
    assert "référence naïve" in reason


def test_missing_baseline_is_rejected(rules):
    assert p.decide(0.38, None, None, True, rules)[0] is Status.REJECTED


def test_acceptance_criteria_can_be_disabled(rules):
    relaxed = rules.model_copy(
        update={"require_validated_data": False, "must_beat_baseline": False}
    )

    assert p.decide(0.90, None, None, False, relaxed)[0] is Status.CHAMPION


def test_higher_is_better_metrics_are_supported(rules):
    r2_rules = rules.model_copy(update={"decision_metric": "r2", "lower_is_better": False})

    assert p.decide(0.70, 0.65, 0.0, True, r2_rules)[0] is Status.CHAMPION
    assert p.decide(0.60, 0.65, 0.0, True, r2_rules)[0] is Status.CHALLENGER


# --- Promotion dans le registre ---------------------------------------------------


def test_first_valid_version_becomes_champion(settings, rules, registry):
    version = registry(mae=0.38)

    decision = p.promote(settings, rules)

    assert decision.promoted
    assert decision.version == version
    assert decision.previous_champion is None
    assert alias_holder(registry, "champion") == version
    assert tags(registry, version)["status"] == "champion"
    assert "decided_at" in tags(registry, version)


def test_better_version_replaces_the_champion_and_archives_it(settings, rules, registry):
    first = registry(mae=0.38)
    p.promote(settings, rules)
    second = registry(mae=0.35)

    decision = p.promote(settings, rules)

    assert decision.promoted
    assert decision.previous_champion == first
    assert alias_holder(registry, "champion") == second
    assert tags(registry, first)["status"] == "archived"
    assert tags(registry, second)["replaced_version"] == first


def test_worse_version_becomes_challenger_and_champion_is_untouched(settings, rules, registry):
    first = registry(mae=0.38)
    p.promote(settings, rules)
    second = registry(mae=0.45)

    decision = p.promote(settings, rules)

    assert decision.status is Status.CHALLENGER
    assert alias_holder(registry, "champion") == first
    assert alias_holder(registry, "challenger") == second
    assert tags(registry, first)["status"] == "champion"
    assert "0.3800" in tags(registry, second)["decision_reason"]


def test_rejected_version_gets_no_alias(settings, rules, registry):
    first = registry(mae=0.38)
    p.promote(settings, rules)
    invalid = registry(mae=0.10, validated=False)

    decision = p.promote(settings, rules)

    assert decision.status is Status.REJECTED
    assert alias_holder(registry, "champion") == first
    assert alias_holder(registry, "challenger") is None
    assert tags(registry, invalid)["status"] == "rejected"


def test_challenger_alias_moves_off_a_version_that_gets_promoted(settings, rules, registry):
    registry(mae=0.38)
    p.promote(settings, rules)
    challenger = registry(mae=0.45)
    p.promote(settings, rules)
    registry.client.log_metric(  # le champion se dégrade : le challenger repasse devant
        registry.client.get_model_version(registry.name, "1").run_id, "test_mae", 0.50
    )

    decision = p.promote(settings, rules, challenger)

    assert decision.promoted
    assert alias_holder(registry, "champion") == challenger
    assert alias_holder(registry, "challenger") is None


def test_promoting_the_current_champion_changes_nothing(settings, rules, registry):
    version = registry(mae=0.38)
    p.promote(settings, rules)

    decision = p.promote(settings, rules, version)

    assert decision.reason == "version déjà en service"
    assert alias_holder(registry, "champion") == version


def test_promote_fails_explicitly_on_empty_registry_or_unknown_version(settings, rules, registry):
    with pytest.raises(p.PromotionError, match="Aucune version"):
        p.promote(settings, rules)

    registry(mae=0.38)
    with pytest.raises(p.PromotionError, match="introuvable"):
        p.promote(settings, rules, "99")


# --- Retour arrière ---------------------------------------------------------------


def test_rollback_restores_the_replaced_version(settings, rules, registry):
    first = registry(mae=0.38)
    p.promote(settings, rules)
    second = registry(mae=0.35)
    p.promote(settings, rules)

    decision = p.rollback(settings, rules)

    assert decision.version == first
    assert decision.previous_champion == second
    assert alias_holder(registry, "champion") == first
    assert tags(registry, first)["status"] == "champion"
    assert tags(registry, first)["rollback_from_version"] == second
    assert tags(registry, second)["status"] == "rolled_back"


def test_rollback_to_an_explicit_version(settings, rules, registry):
    first = registry(mae=0.38)
    p.promote(settings, rules)
    registry(mae=0.36)
    p.promote(settings, rules)
    registry(mae=0.34)
    p.promote(settings, rules)

    p.rollback(settings, rules, first)

    assert alias_holder(registry, "champion") == first


def test_rollback_without_anything_to_restore_fails_explicitly(settings, rules, registry):
    with pytest.raises(p.PromotionError, match="Aucun champion"):
        p.rollback(settings, rules)

    version = registry(mae=0.38)
    p.promote(settings, rules)
    with pytest.raises(p.PromotionError, match="précisez --version"):
        p.rollback(settings, rules)
    with pytest.raises(p.PromotionError, match="déjà en service"):
        p.rollback(settings, rules, version)


# --- État du registre et ligne de commande ---------------------------------------


def test_registry_status_lists_versions_with_alias_and_decision(settings, rules, registry):
    registry(mae=0.38)
    p.promote(settings, rules)
    registry(mae=0.45)
    p.promote(settings, rules)

    rows = p.registry_status(settings, rules)

    assert [(row["version"], row["alias"], row["status"]) for row in rows] == [
        ("2", "challenger", "challenger"),
        ("1", "champion", "champion"),
    ]


def test_command_line_returns_an_error_code_on_failure(settings, registry, monkeypatch, capsys):
    monkeypatch.setattr(p, "get_settings", lambda: settings)

    assert p.main(["rollback"]) == 1

    registry(mae=0.38)
    assert p.main(["promote"]) == 0
    assert p.main(["status"]) == 0
    assert "alias=champion" in capsys.readouterr().out
