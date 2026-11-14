from types import SimpleNamespace

import pytest

from anime_mlops.api.register_mock_model import (
    MOCK_MODEL_TAG,
    MOCK_MODEL_TAG_VALUE,
    _refuse_to_replace_real_model,
)


class AliasClient:
    def __init__(self, *, model_kind: str) -> None:
        self.model_kind = model_kind

    def get_model_version_by_alias(self, _model_name: str, _alias: str):
        return SimpleNamespace(version="5", tags={MOCK_MODEL_TAG: self.model_kind})


def test_mock_registration_refuses_to_replace_a_real_champion():
    client = AliasClient(model_kind="trained")

    with pytest.raises(RuntimeError, match="non factice"):
        _refuse_to_replace_real_model(client, "anime-score-regressor", "champion", force=False)


@pytest.mark.parametrize(
    ("model_kind", "force"),
    [(MOCK_MODEL_TAG_VALUE, False), ("trained", True)],
)
def test_mock_registration_allows_safe_or_explicit_replacement(model_kind, force):
    client = AliasClient(model_kind=model_kind)

    _refuse_to_replace_real_model(client, "anime-score-regressor", "champion", force=force)
