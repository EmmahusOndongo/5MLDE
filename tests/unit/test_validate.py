import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from anime_mlops.config import QualityConfig
from anime_mlops.data import validate as v

NOW = datetime(2026, 11, 14, 12, 0, 0, tzinfo=UTC)
ROWS = 20


@pytest.fixture
def quality():
    return QualityConfig(
        min_rows=10,
        min_join_match_rate=0.9,
        max_missing_rate=0.3,
        target_min=0.0,
        target_max=10.0,
        min_target_rate=0.5,
        min_known_category_rate=0.99,
    )


@pytest.fixture
def mal_data():
    """Lot MAL conforme : un quart des animes sans score, comme dans le vrai catalogue."""
    return pd.DataFrame(
        {
            "id": range(1, ROWS + 1),
            "title": [f"Anime {i}" for i in range(ROWS)],
            "mean": [7.5 if i % 4 else None for i in range(ROWS)],
            "num_episodes": [12] * ROWS,
            "start_date": ["2020-04-03"] * (ROWS - 2) + ["2021-07", "1999"],
            "media_type": ["tv"] * ROWS,
            "status": ["finished_airing"] * (ROWS - 1) + ["not_yet_aired"],
            "rating": ["pg_13"] * ROWS,
            "average_episode_duration": [1440] * ROWS,
            "genres": ["Action, Drama"] * ROWS,
            "studios": ["Sunrise"] * ROWS,
        }
    )


@pytest.fixture
def anilist_data():
    return pd.DataFrame(
        {
            "id_mal": range(1, ROWS + 1),
            "anilist_id": range(101, ROWS + 101),
            "format": ["TV"] * ROWS,
            "source": ["MANGA"] * ROWS,
            "episodes": [12.0] * (ROWS - 1) + [None],
            "tags": [["Space", "Crime"]] * ROWS,
            "is_sequel": [False] * (ROWS - 1) + [True],
            "prequel_id_mal": [None] * (ROWS - 1) + [1.0],
        }
    )


def failed_rules(report):
    return {(failure.rule, failure.column) for failure in report.failures}


def run_mal(data, quality, tmp_path):
    return v.validate(data, v.mal_suite(quality), "abc123", tmp_path, NOW)


def run_anilist(data, quality, tmp_path):
    return v.validate(data, v.anilist_suite(quality), "2026-11-14", tmp_path, NOW)


# --- MAL -----------------------------------------------------------------------


def test_valid_mal_batch_passes(mal_data, quality, tmp_path):
    report = run_mal(mal_data, quality, tmp_path)

    assert report.success
    assert report.failures == ()
    assert report.row_count == ROWS
    assert report.evaluated_rules == 12
    assert v.ensure_valid(report) is report


def test_score_out_of_range_is_blocking(mal_data, quality, tmp_path):
    mal_data.loc[1, "mean"] = 15.0

    report = run_mal(mal_data, quality, tmp_path)

    assert failed_rules(report) == {("expect_column_values_to_be_between", "mean")}
    assert "15.0" in report.failures[0].detail
    with pytest.raises(v.DataQualityError, match="non conformes"):
        v.ensure_valid(report)


def test_renamed_column_is_blocking(mal_data, quality, tmp_path):
    report = run_mal(mal_data.rename(columns={"status": "etat"}), quality, tmp_path)

    assert ("expect_table_columns_to_match_set", None) in failed_rules(report)
    details = " ".join(failure.detail for failure in report.failures)
    assert "status" in details


def test_duplicated_id_is_blocking(mal_data, quality, tmp_path):
    mal_data.loc[1, "id"] = mal_data.loc[0, "id"]

    report = run_mal(mal_data, quality, tmp_path)

    assert failed_rules(report) == {("expect_column_values_to_be_unique", "id")}


def test_unknown_status_is_blocking(mal_data, quality, tmp_path):
    mal_data.loc[0, "status"] = "cancelled"

    report = run_mal(mal_data, quality, tmp_path)

    assert failed_rules(report) == {("expect_column_values_to_be_in_set", "status")}


def test_too_few_rows_is_blocking(mal_data, quality, tmp_path):
    report = run_mal(mal_data.head(5), quality, tmp_path)

    assert ("expect_table_row_count_to_be_between", None) in failed_rules(report)


def test_too_many_missing_scores_is_blocking(mal_data, quality, tmp_path):
    mal_data.loc[: ROWS * 3 // 4, "mean"] = None

    report = run_mal(mal_data, quality, tmp_path)

    assert failed_rules(report) == {("expect_column_values_to_not_be_null", "mean")}


def test_negative_episode_count_is_blocking(mal_data, quality, tmp_path):
    mal_data.loc[0, "num_episodes"] = -1

    report = run_mal(mal_data, quality, tmp_path)

    assert failed_rules(report) == {("expect_column_values_to_be_between", "num_episodes")}


def test_validate_mal_raises_on_invalid_batch(mal_data, quality, tmp_path):
    mal_data.loc[1, "mean"] = 15.0

    with pytest.raises(v.DataQualityError) as error:
        v.validate_mal(mal_data, "abc123", quality, tmp_path)

    assert error.value.report.report_path.is_file()


# --- AniList -------------------------------------------------------------------


def test_valid_anilist_batch_passes(anilist_data, quality, tmp_path):
    report = run_anilist(anilist_data, quality, tmp_path)

    assert report.success
    assert report.evaluated_rules == 10


def test_duplicated_join_key_is_blocking(anilist_data, quality, tmp_path):
    anilist_data.loc[1, "id_mal"] = anilist_data.loc[0, "id_mal"]

    report = run_anilist(anilist_data, quality, tmp_path)

    assert failed_rules(report) == {("expect_column_values_to_be_unique", "id_mal")}


def test_empty_anilist_response_is_blocking(anilist_data, quality, tmp_path):
    report = run_anilist(anilist_data.head(0), quality, tmp_path)

    assert not report.success
    assert ("expect_table_row_count_to_be_between", None) in failed_rules(report)


def test_missing_tags_column_is_blocking(anilist_data, quality, tmp_path):
    report = run_anilist(anilist_data.drop(columns=["tags"]), quality, tmp_path)

    assert ("expect_table_columns_to_match_set", None) in failed_rules(report)
    with pytest.raises(v.DataQualityError):
        v.validate_anilist(anilist_data.drop(columns=["tags"]), "2026-11-14", quality, tmp_path)


# --- Rapport archivé -----------------------------------------------------------


def test_report_is_archived_on_success(mal_data, quality, tmp_path):
    report = run_mal(mal_data, quality, tmp_path)

    assert report.report_path == tmp_path / "validation" / "20261114T120000_mal_abc123_ok.json"
    payload = json.loads(report.report_path.read_text("utf-8"))
    assert payload["success"] is True
    assert payload["data_version"] == "abc123"
    assert payload["row_count"] == ROWS
    assert payload["failures"] == []
    assert payload["great_expectations"]["statistics"]["evaluated_expectations"] == 12


def test_report_is_archived_on_failure(mal_data, quality, tmp_path):
    mal_data.loc[1, "mean"] = 15.0

    report = run_mal(mal_data, quality, tmp_path)

    assert report.report_path.name == "20261114T120000_mal_abc123_echec.json"
    payload = json.loads(report.report_path.read_text("utf-8"))
    assert payload["success"] is False
    assert len(payload["failures"]) == 1
