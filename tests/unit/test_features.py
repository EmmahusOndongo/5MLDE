import numpy as np
import pandas as pd
import pytest

from anime_mlops.features.build import (
    FEATURES,
    LEAKAGE_COLUMNS,
    TARGET,
    build_features,
    split_features_target,
)


@pytest.fixture
def merged():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "mean": [8.75, 7.2, 6.1],
            "rank": [51, 900, 4000],
            "num_list_users": [2_000_000, 50_000, 900],
            "num_scoring_users": [1_000_000, 20_000, 300],
            "num_episodes": [26, 0, 1],
            "average_episode_duration": [1440, 0, 200],
            "start_date": ["1998-04-03", "2021-07", "1999"],
            "media_type": ["tv", "ona", "music"],
            "rating": ["r", None, "g"],
            "source": ["ORIGINAL", None, np.nan],
            "genres": ["Action, Sci-Fi", "Drama", np.nan],
            "studios": ["Sunrise", np.nan, "A-1 Pictures, Trigger"],
            # Les tags reviennent en tableau NumPy après lecture du cache Parquet.
            "tags": [np.array(["Space", "Crime"]), ["Idol"], np.nan],
            "is_sequel": [False, True, False],
            "prequel_score": [np.nan, 7.9, np.nan],
        }
    )


def test_output_has_exactly_the_contract_columns(merged):
    features = build_features(merged)

    assert list(features.columns) == FEATURES
    assert features.index.tolist() == [1, 2, 3]
    assert features.index.name == "id"


def test_no_leakage_column_and_no_target(merged):
    features = build_features(merged)

    assert TARGET not in features.columns
    assert not set(LEAKAGE_COLUMNS) & set(features.columns)
    assert not set(LEAKAGE_COLUMNS) & set(FEATURES)


def test_zero_counters_mean_unknown(merged):
    features = build_features(merged)

    assert features.loc[1, "num_episodes"] == 26
    assert np.isnan(features.loc[2, "num_episodes"])
    assert np.isnan(features.loc[2, "average_episode_duration"])


def test_date_parts_follow_the_precision_of_the_date(merged):
    features = build_features(merged)

    # Date complète : année, saison et jour de diffusion.
    assert features.loc[1, ["start_year", "start_season", "start_weekday"]].tolist() == [
        1998,
        "spring",
        "friday",
    ]
    # Année et mois : pas de jour.
    assert features.loc[2, "start_season"] == "summer"
    assert features.loc[2, "start_weekday"] is None
    # Année seule.
    assert features.loc[3, "start_year"] == 1999
    assert features.loc[3, "start_season"] is None


def test_multi_label_fields_are_always_lists(merged):
    features = build_features(merged)

    assert features.loc[1, "genres"] == ["Action", "Sci-Fi"]
    assert features.loc[3, "genres"] == []
    assert features.loc[3, "studios"] == ["A-1 Pictures", "Trigger"]
    assert features.loc[2, "studios"] == []
    assert features.loc[1, "tags"] == ["Space", "Crime"]
    assert features.loc[3, "tags"] == []


def test_missing_categories_are_none(merged):
    features = build_features(merged)

    assert features.loc[2, "rating"] is None
    assert features.loc[2, "source"] is None
    assert features.loc[3, "source"] is None


def test_split_features_target(merged):
    features, target = split_features_target(merged)

    assert target.tolist() == [8.75, 7.2, 6.1]
    assert target.index.equals(features.index)
    assert target.name == TARGET


def test_split_refuses_rows_without_score(merged):
    merged.loc[0, "mean"] = np.nan

    with pytest.raises(ValueError, match="sans score"):
        split_features_target(merged)
