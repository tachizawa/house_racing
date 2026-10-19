"""Phase 1: コード変換・出走表・血統の階層化・整合性検査・分割。"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from keiba_ml.config import CodeConfig, DataConfig, MLConfig, PedigreeConfig, SplitConfig
from keiba_ml.dataset import build, codes, integrity
from keiba_ml.dataset.pedigree import build_pedigree, inbreeding_table, load_sources
from keiba_ml.dataset.races import build_runners, load_raw
from keiba_ml.dataset.split import history_frame, make_window, split_frame
from keiba_ml.ingest import pedigree_store as ps
from tests.fixtures import csv_data


# --- codes -------------------------------------------------------------------

@pytest.mark.parametrize(("barei", "year", "expected"), [
    ("04", 2000, 3),   # 数え年の時代は 1 引く
    ("04", 2001, 4),
    ("00", 2010, None),
    ("", 2010, None),
    ("01", 1999, None),  # 数え 1 歳 = 満 0 歳は出走しない
])
def test_age_from_barei(barei, year, expected):
    assert codes.age_from_barei(barei, year, kazoe_until_year=2000) == expected


@pytest.mark.parametrize(("code", "tier"), [
    ("701", 0), ("703", 1), ("004", 2), ("005", 2), ("009", 3), ("010", 3),
    ("015", 4), ("016", 4), ("999", 5), ("000", None), ("", None),
])
def test_jyoken_tier(code, tier):
    assert codes.jyoken_tier(code) == tier


def test_grade_wins_over_jyoken():
    assert codes.grade_tier("A", "999") == 8
    assert codes.grade_tier("", "005") == 2


@pytest.mark.parametrize(("kyori", "band"), [
    (1000, "S"), (1399, "S"), (1400, "M"), (1800, "I"), (2200, "L"), (3200, "E"), (None, None)])
def test_distance_band(kyori, band):
    assert codes.distance_band(kyori) == band


# --- races -------------------------------------------------------------------

def _races(*rows):
    return pd.DataFrame([{k: str(v) for k, v in csv_data.make_race(**r).items()} for r in rows])


def _runners(*rows):
    return pd.DataFrame([{k: str(v) for k, v in csv_data.make_race_horse(**r).items()}
                         for r in rows])


RID = "2026032106011101"


def test_build_runners_prefers_result_rows_and_excludes_scratched():
    races = _races({"race_id": RID, "data_kubun": "2"}, {"race_id": RID, "data_kubun": "7"})
    runners = _runners(
        {"race_id": RID, "umaban": "01", "ketto_num": "2020100001", "data_kubun": "2",
         "result_kakutei_jyuni": ""},
        {"race_id": RID, "umaban": "01", "ketto_num": "2020100001", "data_kubun": "7",
         "result_kakutei_jyuni": "2"},
        {"race_id": RID, "umaban": "02", "ketto_num": "2020100002", "result_kakutei_jyuni": "1"},
        {"race_id": RID, "umaban": "03", "ketto_num": "2020100003", "result_kakutei_jyuni": "",
         "result_ijyo_cd": "1"},     # 出走取消 → 抜く
        {"race_id": RID, "umaban": "04", "ketto_num": "2020100004", "result_kakutei_jyuni": "",
         "result_ijyo_cd": "4"},     # 競走中止 → 残し、3着以内ではない
    )
    df = build_runners(races, runners, data=DataConfig(), codes_config=CodeConfig())
    assert list(df["umaban"]) == ["01", "02", "04"]
    assert list(df["target_top3"]) == [1.0, 1.0, 0.0]
    assert list(df["target_win"]) == [0.0, 1.0, 0.0]
    assert df["field_size"].unique().tolist() == [3]
    assert df["race_date"].iloc[0] == pd.Timestamp("2026-03-21")


def _version_with_fromdate(data_dir, version_id, race_ids, fromdate):
    races = [csv_data.make_race(race_id=r) for r in race_ids]
    # 同じレースの再配信（行が2つ）でもレース数は1つと数える。
    races += [csv_data.make_race(race_id=race_ids[-1], make_date="20260102")]
    csv_data.write_dataset(data_dir, races,
                           [csv_data.make_race_horse(race_id=r) for r in race_ids], [],
                           version_id=version_id)
    path = data_dir / "versions" / version_id / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**manifest, "fromdate": fromdate}), encoding="utf-8")


def test_load_raw_drops_races_no_version_covers(tmp_path):
    """セットアップ取得に混じる古い訂正再配信は、どの版の fromdate よりも前なので捨てる。"""
    old = "1977061505030201"
    _version_with_fromdate(tmp_path, "v1", [RID, old], "19860101000000")
    races, runners = load_raw(tmp_path, ["v1"])
    assert set(races["race_id"]) == {RID}
    assert runners["race_id"].tolist() == [RID]
    assert races.attrs["dropped_before_fromdate"] == {"cutoff": "19860101", "races": 1}


def test_load_raw_keeps_corrections_older_than_a_later_versions_fromdate(tmp_path):
    """fromdate は提供日。差分の版に入った古い日付のレースの訂正は、古い版が覆う範囲なら残す。"""
    corrected = "2020061505030201"
    _version_with_fromdate(tmp_path, "v1", [corrected], "19860101000000")
    _version_with_fromdate(tmp_path, "v2", [RID, corrected], "20260301000000")
    races, _ = load_raw(tmp_path, ["v1", "v2"])
    assert set(races["race_id"]) == {RID, corrected}
    assert races.attrs["dropped_before_fromdate"] == {"cutoff": "19860101", "races": 0}


def test_build_runners_entry_only_rows_have_no_target():
    races = _races({"race_id": RID, "data_kubun": "2"})
    runners = _runners({"race_id": RID, "data_kubun": "2", "result_kakutei_jyuni": "",
                        "ketto_num": "2020100001"})
    df = build_runners(races, runners, data=DataConfig(), codes_config=CodeConfig())
    assert not df["has_result"].iloc[0]
    assert pd.isna(df["target_top3"].iloc[0])


def test_build_runners_drops_jump_and_non_jra():
    races = _races({"race_id": RID, "track_cd": "51"},
                   {"race_id": "2026032130011101", "jyo_cd": "30"})
    runners = _runners({"race_id": RID, "ketto_num": "2020100001"},
                       {"race_id": "2026032130011101", "ketto_num": "2020100002"})
    df = build_runners(races, runners, data=DataConfig(), codes_config=CodeConfig())
    assert df.empty


def test_build_runners_converts_kazoe_age_before_2001():
    rid = "2000052106011101"
    races = _races({"race_id": rid})
    runners = _runners({"race_id": rid, "barei": "04", "ketto_num": "1996100001"})
    df = build_runners(races, runners, data=DataConfig(), codes_config=CodeConfig())
    assert df["age_from_barei"].iloc[0] == 3


# --- pedigree tree -----------------------------------------------------------

def _sk(ketto, ancestors, birth="2018-04-01"):
    row = {"ketto_num": ketto, "birth_date": birth}
    paths = ("S", "D", "SS", "SD", "DS", "DD", "SSS", "SSD", "SDS", "SDD", "DSS", "DSD",
             "DDS", "DDD")
    row.update({f"anc_{p}": a for p, a in zip(paths, ancestors)})
    return row


def _hn(num, father="", mother=""):
    return {"hansyoku_num": num, "father_num": father, "mother_num": mother, "bamei": f"N{num}"}


def test_sk_preferred_um_fills_and_hn_extends_to_fourth_generation():
    sk = pd.DataFrame([_sk("H1", ["S1", "D1", "SS1", "SD1", "DS1", "DD1"] + [""] * 8)])
    um = pd.DataFrame([_sk("H1", ["X", "X", "X", "X", "X", "X", "SSS1"] + [""] * 7),
                       _sk("H2", ["S2", "D2"] + [""] * 12)])
    hn = pd.DataFrame([_hn("SSS1", father="SSSS1"), _hn("S1", "SS1", "SD1")])
    ped, stats = build_pedigree(sk, hn, um, generations=4, inbreeding_generations=4)
    h1 = ped.loc["H1"]
    assert h1["ped_S"] == "S1"            # SK が優先
    assert h1["ped_SSS"] == "SSS1"        # SK の空き枠を UM で補う
    assert h1["ped_SSSS"] == "SSSS1"      # 4世代目は HN の父母リンク
    assert h1["pedigree_source"] == "SK+UM"
    assert ped.loc["H2", "pedigree_source"] == "UM"
    assert stats["parent_link"] == {"checked": 2, "agreed": 2, "rate": 1.0}


def test_parent_link_detects_disagreement():
    sk = pd.DataFrame([_sk("H1", ["S1", "D1", "WRONG", "SD1"] + [""] * 10)])
    hn = pd.DataFrame([_hn("S1", "SS1", "SD1")])
    _, stats = build_pedigree(sk, hn, pd.DataFrame(), generations=3, inbreeding_generations=3)
    assert stats["parent_link"]["checked"] == 2
    assert stats["parent_link"]["agreed"] == 1


def test_inbreeding_two_by_two():
    # 父父 と 母父 が同じ馬 A → 2×2 のクロス。F ≈ (1/2)^(2+2-1) = 1/8
    ped = pd.DataFrame({"ped_S": ["s"], "ped_D": ["d"], "ped_SS": ["A"], "ped_SD": ["x"],
                        "ped_DS": ["A"], "ped_DD": ["y"]}, index=pd.Index(["H"], name="ketto_num"))
    table = inbreeding_table(ped, 2)
    assert table.loc["H", "inbreeding_f"] == pytest.approx(1 / 8)
    assert table.loc["H", "n_cross"] == 1
    assert table.loc["H", "closest_cross"] == 4


def test_no_cross_means_zero():
    ped = pd.DataFrame({"ped_S": ["s"], "ped_D": ["d"]}, index=pd.Index(["H"], name="ketto_num"))
    table = inbreeding_table(ped, 1)
    assert table.loc["H", "inbreeding_f"] == 0.0
    assert table.loc["H", "n_cross"] == 0


# --- integrity / split -------------------------------------------------------

def test_feature_column_check_rejects_results_and_odds():
    assert integrity.check_feature_columns(["ped_S", "age"])["status"] == "ok"
    bad = integrity.check_feature_columns(["ped_S", "result_odds", "target_top3", "odds"])
    assert bad["status"] == "fail"
    assert set(bad["detail"]["forbidden"]) == {"result_odds", "target_top3", "odds"}


def test_age_convention_check_finds_wrong_boundary():
    df = pd.DataFrame({"year": [1999, 1999, 2005], "birth_year": [1995, 1995, 2001],
                       "age_from_barei": [5, 5, 4]})      # 1999 年は数え年のまま（変換漏れ）
    result = integrity.check_age_convention(df, kazoe_until_year=1990)
    assert result["status"] == "fail"
    assert result["detail"]["years_below_95pct"] == [1999]


def test_window_and_split_are_disjoint():
    split = SplitConfig(test_start=date(2025, 1, 1), test_end=date(2025, 12, 31),
                        valid_years=1, train_years=(10,))
    window = make_window(split, 10, date(2010, 1, 5))
    assert window.valid_start == date(2024, 1, 1)
    assert window.train_start == date(2014, 1, 1)
    assert not window.truncated
    assert make_window(split, 20, date(2010, 1, 5)).truncated

    days = pd.to_datetime(["2013-12-31", "2014-01-01", "2023-12-31", "2024-01-01",
                           "2025-01-01", "2025-12-31", "2026-01-01"])
    df = pd.DataFrame({"race_date": days, "race_id": [f"r{i}" for i in range(7)],
                       "has_result": True})
    train, valid, test = split_frame(df, window)
    assert list(train["race_id"]) == ["r1", "r2"]
    assert list(valid["race_id"]) == ["r3"]
    assert list(test["race_id"]) == ["r4", "r5"]
    assert integrity.check_split(train, valid, test)["status"] == "ok"


# --- PR #1 のレビュー指摘 ----------------------------------------------------

@pytest.mark.parametrize("code", ["0", "00", "0000000000"])
def test_zero_filled_jyoken_is_missing(code):
    assert codes.jyoken_tier(code) is None


def test_stopped_or_disqualified_horse_is_not_a_positive_label():
    races = _races({"race_id": RID, "data_kubun": "7"})
    runners = _runners(
        {"race_id": RID, "umaban": "01", "ketto_num": "2020100001", "result_kakutei_jyuni": "1",
         "result_ijyo_cd": "5"},     # 失格（着順の欄に値が残っていても勝ちにしない）
        {"race_id": RID, "umaban": "02", "ketto_num": "2020100002", "result_kakutei_jyuni": "2",
         "result_ijyo_cd": "4"},     # 競走中止
        {"race_id": RID, "umaban": "03", "ketto_num": "2020100003", "result_kakutei_jyuni": "3",
         "result_ijyo_cd": "7"},     # 降着は着順どおり
    )
    df = build_runners(races, runners, data=DataConfig(), codes_config=CodeConfig())
    assert list(df["target_top3"]) == [0.0, 0.0, 1.0]
    assert list(df["target_win"]) == [0.0, 0.0, 0.0]


def test_history_frame_rejects_unknown_stats_history():
    split = SplitConfig(test_start=date(2025, 1, 1), test_end=date(2025, 12, 31),
                        valid_years=1, train_years=(10,))
    window = make_window(split, 10, date(2010, 1, 5))
    df = pd.DataFrame({"race_date": pd.to_datetime(["2020-01-01"])})
    assert len(history_frame(df, window, "all")) == 1
    with pytest.raises(ValueError, match="stats_history"):
        history_frame(df, window, "windows")


def test_missing_january_counts_as_truncated():
    split = SplitConfig(test_start=date(2025, 1, 1), test_end=date(2025, 12, 31),
                        valid_years=1, train_years=(10,))
    assert not make_window(split, 10, date(2014, 1, 5)).truncated
    assert make_window(split, 10, date(2014, 2, 1)).truncated


def _committed_pedigree_version(ml_data_dir):
    writer = ps.PedigreeVersionWriter(ml_data_dir, "v1", dataspecs=["BLOD"], fromdate="x", option=1)
    return writer.commit()


def test_missing_pedigree_csv_stops_loading(tmp_path):
    path = _committed_pedigree_version(tmp_path)
    assert set(load_sources(tmp_path)) == {"SK", "HN", "UM", "versions"}
    (path / "hn.csv").unlink()
    with pytest.raises(ps.PedigreeVersionError, match="hn.csv"):
        load_sources(tmp_path)


def test_blood_versions_are_stacked_so_a_delta_current_keeps_setup_rows(tmp_path):
    """セットアップの版の後に差分だけの版を取ると CURRENT は差分を指す。全版を重ねて読む。"""
    from keiba.ingest import records as keiba_records
    from keiba.ingest import store as keiba_store
    from keiba_ml.dataset.pedigree import load_blood_sources
    from tests.fixtures import ml_records as mr

    def commit(version_id, sk_rows):
        writer = keiba_store.VersionWriter(tmp_path, version_id, dataspecs=["BLDN"],
                                           fromdate="19860101000000", option=4,
                                           files=keiba_store.BLOOD_FILES)
        for ketto, sire, make_date in sk_rows:
            writer.add("SK", keiba_records.parse_sk(mr.make_sk(
                ketto_num=ketto, ancestors=[sire] + [""] * 13, make_date=make_date)))
        writer.commit()

    commit("20260101T000000Z", [("2015100001", "1000000001", "20251201"),
                                ("2015100002", "1000000002", "20251201")])
    commit("20260201T000000Z", [("2015100002", "1000000009", "20260131")])   # 訂正だけ
    assert (tmp_path / "CURRENT").read_text(encoding="utf-8").strip() == "20260201T000000Z"
    sources = load_blood_sources(tmp_path)
    assert sources["versions"] == ["20260101T000000Z", "20260201T000000Z"]
    sires = dict(zip(sources["SK"]["ketto_num"], sources["SK"]["anc_S"]))
    assert sires == {"2015100001": "1000000001", "2015100002": "1000000009"}


def _saved_dataset(ml_data_dir, config):
    df = pd.DataFrame({"race_id": ["r1"], "race_date": pd.to_datetime(["2020-01-01"]),
                       "has_result": [True]})
    report = {"race_versions": ["a"], "pedigree_versions": ["b"], "checks": []}
    return build.save(df, pd.DataFrame({"x": [1]}), {}, report, config,
                      ml_data_dir=ml_data_dir, dataset_id="d1")


def test_load_rejects_another_schema_version(tmp_path):
    pytest.importorskip("pyarrow")
    out = _saved_dataset(tmp_path, MLConfig())
    build.load(tmp_path, "d1")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "0"
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="schema_version"):
        build.load(tmp_path, "d1")


def test_config_mismatches_ignore_versions_but_catch_structure(tmp_path):
    pytest.importorskip("pyarrow")
    _saved_dataset(tmp_path, MLConfig())
    _, _, manifest = build.load(tmp_path, "d1")
    other_versions = MLConfig(data=DataConfig(race_versions=("x", "y")))
    assert build.config_mismatches(manifest, other_versions) == []
    deeper = MLConfig(pedigree=PedigreeConfig(generations=6))
    assert build.config_mismatches(manifest, deeper) == ["pedigree_generations"]
    with_jump = MLConfig(data=DataConfig(include_jump=True))
    assert build.config_mismatches(manifest, with_jump) == ["data.include_jump"]
    other_source = MLConfig(data=DataConfig(pedigree_source="ml"))
    assert build.config_mismatches(manifest, other_source) == ["data.pedigree_source"]
