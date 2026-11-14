import pandas as pd
import pytest

from anime_mlops.data.merge import merge_sources, training_rows


@pytest.fixture
def mal():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Saison 1", "Saison 2", "Clip", "Annoncé"],
            "mean": [8.0, 7.5, 6.0, None],
            "num_episodes": [12, 13, 1, 0],
            "status": ["finished_airing", "finished_airing", "finished_airing", "not_yet_aired"],
        }
    )


@pytest.fixture
def anilist():
    return pd.DataFrame(
        {
            "id_mal": [1, 2, 4],
            "anilist_id": [101, 102, 104],
            "format": ["TV", "TV", "TV"],
            "source": ["MANGA", "MANGA", "ORIGINAL"],
            "episodes": [12.0, 12.0, None],
            "tags": [["Space"], ["Space", "Crime"], []],
            "is_sequel": [False, True, False],
            "prequel_id_mal": [None, 1.0, None],
        }
    )


def test_every_mal_anime_is_kept(mal, anilist):
    merged = merge_sources(mal, anilist)

    assert merged["id"].tolist() == [1, 2, 3, 4]
    assert merged["has_anilist"].tolist() == [True, True, False, True]
    assert "id_mal" not in merged.columns


def test_anime_missing_from_anilist_has_empty_enrichment(mal, anilist):
    clip = merge_sources(mal, anilist).set_index("id").loc[3]

    assert pd.isna(clip["source"])
    assert pd.isna(clip["anilist_id"])
    assert clip["is_sequel"] is False or clip["is_sequel"] == False  # noqa: E712
    assert pd.isna(clip["episodes_agree"])


def test_prequel_score_comes_from_the_previous_season(mal, anilist):
    merged = merge_sources(mal, anilist).set_index("id")

    assert merged.loc[2, "prequel_score"] == 8.0
    assert pd.isna(merged.loc[1, "prequel_score"])


def test_episode_agreement_between_sources(mal, anilist):
    merged = merge_sources(mal, anilist).set_index("id")

    assert merged.loc[1, "episodes_agree"] == True  # noqa: E712
    assert merged.loc[2, "episodes_agree"] == False  # noqa: E712  (13 chez MAL, 12 chez AniList)
    # MAL indique 0 (inconnu) et AniList ne sait pas : pas de comparaison possible.
    assert pd.isna(merged.loc[4, "episodes_agree"])


def test_duplicated_join_key_is_refused(mal, anilist):
    duplicated = pd.concat([anilist, anilist.head(1)])

    with pytest.raises(pd.errors.MergeError):
        merge_sources(mal, duplicated)


def test_training_rows_keep_finished_and_scored_anime(mal, anilist):
    training = training_rows(merge_sources(mal, anilist))

    assert training["id"].tolist() == [1, 2, 3]
    assert training["mean"].notna().all()
