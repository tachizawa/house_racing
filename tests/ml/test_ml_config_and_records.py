"""設定と血統レコードの分解・保存。"""

from __future__ import annotations

import json

import pytest

from keiba_ml import config as mlc
from keiba_ml.ingest import pedigree_records as pr
from keiba_ml.ingest import pedigree_store as ps
from tests.fixtures import ml_records as mr

ANCESTORS = [f"{1000000000 + i}" for i in range(1, 15)]


def test_all_paths_three_generations_match_jv_data_order():
    assert mlc.all_paths(3) == mlc.KETTO3_PATHS
    assert len(mlc.all_paths(5)) == 62
    assert mlc.all_paths(4)[14:16] == ("SSSS", "SSSD")


def test_repository_config_file_loads():
    # 値は手元のデータ期間に合わせて書き換える前提なので固定しない。読めて検査を通ればよい。
    config = mlc.load_config(mlc.DEFAULT_CONFIG_PATH)
    assert config.split.train_years
    assert config.pedigree.stat_paths


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "x.toml"
    path.write_text("[model]\nlos_function = 'Logloss'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="los_function"):
        mlc.load_config(path)


def test_stat_path_deeper_than_generations_is_rejected(tmp_path):
    path = tmp_path / "x.toml"
    path.write_text("[pedigree]\ngenerations = 2\ncategorical_generations = 2\n"
                    "inbreeding_generations = 2\nstat_paths = ['SSS']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stat_paths"):
        mlc.load_config(path)


def test_config_round_trips_through_saved_json():
    from dataclasses import asdict
    original = mlc.load_config(mlc.DEFAULT_CONFIG_PATH)
    raw = json.loads(json.dumps(asdict(original), default=str))
    assert mlc.config_from_dict(raw) == original


def test_unknown_pedigree_source_is_rejected(tmp_path):
    path = tmp_path / "x.toml"
    path.write_text("[data]\npedigree_source = 'BLOD'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pedigree_source"):
        mlc.load_config(path)


def test_blood_version_columns_map_to_the_same_ancestors_as_this_layer():
    """既存の血統の版（keiba.ingest.records.parse_sk）と、この層の分解が同じ枠に同じ番号を置く。"""
    from keiba.ingest import records as keiba_records
    from keiba_ml.dataset.pedigree import BLOOD_HN_RENAME, BLOOD_SK_RENAME
    raw = mr.make_sk(ketto_num="2015100001", ancestors=ANCESTORS)
    _, ours = pr.parse(raw)
    theirs = {BLOOD_SK_RENAME.get(k, k): v for k, v in keiba_records.parse_sk(raw).items()}
    assert [theirs[f"anc_{p}"] for p in mlc.KETTO3_PATHS] == \
        [ours[f"anc_{p}"] for p in mlc.KETTO3_PATHS] == ANCESTORS
    raw = mr.make_hn(hansyoku_num="1000000099", father=ANCESTORS[0], mother=ANCESTORS[1])
    _, ours = pr.parse(raw)
    theirs = {BLOOD_HN_RENAME.get(k, k): v for k, v in keiba_records.parse_hn(raw).items()}
    assert (theirs["father_num"], theirs["mother_num"]) == \
        (ours["father_num"], ours["mother_num"]) == (ANCESTORS[0], ANCESTORS[1])


def test_parse_sk_reads_fourteen_ancestors_in_order():
    spec, row = pr.parse(mr.make_sk(ketto_num="2015100001", ancestors=ANCESTORS))
    assert spec == "SK"
    assert row["ketto_num"] == "2015100001"
    assert row["birth_date"] == "2015-04-01"
    assert [row[f"anc_{p}"] for p in mlc.KETTO3_PATHS] == ANCESTORS
    assert row["record_length"] == pr.SK_LENGTH


def test_zero_filled_ancestor_is_blank_not_a_shared_ancestor():
    ancestors = ANCESTORS[:]
    ancestors[1] = "0000000000"
    _, row = pr.parse(mr.make_sk(ancestors=ancestors))
    assert row["anc_D"] == ""


def test_parse_hn_reads_parent_links():
    _, row = pr.parse(mr.make_hn(hansyoku_num="1000000099", father="1000000001",
                                 mother="1000000002", bamei="テストボバ", birth_year="2001"))
    assert row["hansyoku_num"] == "1000000099"
    assert row["father_num"] == "1000000001"
    assert row["mother_num"] == "1000000002"
    assert row["bamei"] == "テストボバ"
    assert row["birth_year"] == "2001"
    assert row["ketto_num"] == ""


def test_parse_um_reads_numbers_and_names():
    pairs = [(a, f"ボバ{i}") for i, a in enumerate(ANCESTORS)]
    _, row = pr.parse(mr.make_um(ancestors=pairs))
    assert row["anc_DS"] == ANCESTORS[4]
    assert row["name_DS"] == "ボバ4"
    assert row["name_DDD"] == "ボバ13"


def test_unknown_record_kind_is_ignored():
    assert pr.parse(b"BT" + b" " * 100) is None


def test_ymd_with_unknown_day_keeps_year_and_month():
    assert pr.ymd("19850300") == "1985-03-01"
    assert pr.ymd("00000000") == ""


def test_writer_commits_and_lists_versions(tmp_path):
    writer = ps.PedigreeVersionWriter(tmp_path, "20260101T000000Z", dataspecs=["BLOD"],
                                      fromdate="19860101000000", option=4)
    writer.add(*pr.parse(mr.make_sk(ancestors=ANCESTORS)))
    writer.add(*pr.parse(mr.make_hn()))
    path = writer.commit()
    manifest = ps.read_manifest(path)
    assert manifest["record_counts"] == {"SK": 1, "HN": 1, "UM": 0}
    assert manifest["length_mismatch"] == {"SK": 0, "HN": 0, "UM": 0}
    assert ps.complete_versions(tmp_path) == ["20260101T000000Z"]


def test_writer_counts_records_of_unexpected_length(tmp_path):
    writer = ps.PedigreeVersionWriter(tmp_path, "v1", dataspecs=["BLOD"],
                                      fromdate="x", option=1)
    writer.add(*pr.parse(mr.make_sk()[:-2]))
    assert writer.length_mismatch == {"SK": 1}
    writer.abort()
    assert ps.complete_versions(tmp_path) == []


def test_writer_rejects_string_dataspecs(tmp_path):
    with pytest.raises(TypeError):
        ps.PedigreeVersionWriter(tmp_path, "v1", dataspecs="BLOD", fromdate="x", option=1)


# --- PR #1 のレビュー指摘 ----------------------------------------------------

def test_default_config_is_valid_without_a_pedigree_section():
    mlc.validate(mlc.MLConfig())
    assert mlc.config_from_dict({}) == mlc.MLConfig()


def test_explicit_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="設定ファイルがありません"):
        mlc.load_config(tmp_path / "typo.toml")
    assert mlc.load_config(None) == mlc.load_config(mlc.DEFAULT_CONFIG_PATH)


@pytest.mark.parametrize(("section", "body", "match"), [
    ("pedigree", "stat_paths = []", "stat_paths"),
    ("pedigree", "inbreeding_generations = 0", "inbreeding_generations"),
    ("evaluate", "top_n = 0", "top_n"),
    ("evaluate", "calibration_bins = 0", "calibration_bins"),
    ("evaluate", "bootstrap = -1", "bootstrap"),
    ("data", "jra_only = 'false'", "jra_only"),
    ("model", "iterations = true", "iterations"),
    ("split", "train_years = ['10']", "train_years"),
    ("data", "race_versions = [20261010]", "race_versions"),
    ("data", "pedigree_versions = ['v1', 2]", "pedigree_versions"),
    ("model", "extra = { loss_function = 'YetiRank' }", "loss_function"),
    ("model", "extra = { has_time = false }", "has_time"),
])
def test_invalid_values_are_rejected_when_loading(tmp_path, section, body, match):
    path = tmp_path / "x.toml"
    path.write_text(f"[{section}]\n{body}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        mlc.load_config(path)


def test_model_extra_still_accepts_other_parameters(tmp_path):
    path = tmp_path / "x.toml"
    path.write_text("[model]\nextra = { eval_metric = 'AUC', border_count = 64 }\n",
                    encoding="utf-8")
    assert mlc.load_config(path).model.extra == {"eval_metric": "AUC", "border_count": 64}


def test_fixture_rejects_wrong_number_of_ancestors():
    with pytest.raises(ValueError):
        mr.make_sk(ancestors=ANCESTORS + ["1000000099"])
    with pytest.raises(ValueError):
        mr.make_um(ancestors=[("1000000001", "x")] * 15)


def test_writer_rejects_empty_and_duplicate_dataspecs(tmp_path):
    with pytest.raises(ValueError, match="空"):
        ps.PedigreeVersionWriter(tmp_path, "v1", dataspecs=[], fromdate="x", option=1)
    with pytest.raises(ValueError, match="重複"):
        ps.PedigreeVersionWriter(tmp_path, "v1", dataspecs=["BLOD", "BLOD"],
                                 fromdate="x", option=1)
    assert not ps.version_dir(tmp_path, "v1.partial").exists()


def test_writer_refuses_an_existing_version_before_fetching(tmp_path):
    first = ps.PedigreeVersionWriter(tmp_path, "v1", dataspecs=["BLOD"], fromdate="x", option=1)
    first.commit()
    with pytest.raises(ps.PedigreeVersionError, match="既にあります"):
        ps.PedigreeVersionWriter(tmp_path, "v1", dataspecs=["BLOD"], fromdate="x", option=1)
    assert not ps.version_dir(tmp_path, "v1.partial").exists()


def test_commit_checks_the_final_version_before_writing_the_manifest(tmp_path):
    writer = ps.PedigreeVersionWriter(tmp_path, "v1", dataspecs=["BLOD"], fromdate="x", option=1)
    ps.version_dir(tmp_path, "v1").mkdir(parents=True)    # 取得中に別のプロセスが作った
    with pytest.raises(ps.PedigreeVersionError):
        writer.commit()
    assert not (ps.version_dir(tmp_path, "v1.partial") / "manifest.json").exists()
    writer.abort()


def test_non_object_manifest_is_a_version_error(tmp_path):
    path = ps.version_dir(tmp_path, "v1")
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ps.PedigreeVersionError, match="形式"):
        ps.read_manifest(path)
    assert ps.complete_versions(tmp_path) == []
