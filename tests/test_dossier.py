from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from keiba import config
from keiba.data.repository import Repository
from keiba.data.store import DataStore
from keiba.features import dossier, history
from keiba.models import HorseEntry, RaceInput
from tests.fixtures import csv_data as fx

JST = timezone(timedelta(hours=9))
TARGET = "2026032106011101"


def _dataset(tmp_path):
    races = [
        fx.make_race(race_id="2026030106010101", hasso_at="2026-03-01T15:00:00+09:00",
                     track_cd="18", kyori_m="2000"),
        # 対象レース。結果の値に目印（time 999.9 / margin 9.87）を入れ、漏れを検出する。
        fx.make_race(race_id=TARGET, hasso_at="2026-03-21T15:25:00+09:00",
                     track_cd="18", kyori_m="2000", class_level="3"),
    ]
    # 馬名照合（find_horses_by_name）は make_date が締切以前の行しか見ない。
    # past_runs は発走時刻だけで絞る。A は過去走の行（make_date 3/2）で、
    # B は馬マスタ（UM, make_date 1/1）で照合させる——B は対象レースにしか
    # 出走記録が無いので、UM が無いと「照合できない（unmatched）」になる。
    race_horses = [
        fx.make_race_horse(race_id="2026030106010101", umaban="03", ketto_num="A",
                           bamei="カコアリ", result_kakutei_jyuni="2", result_margin_sec="0.2",
                           make_date="20260302"),
        fx.make_race_horse(race_id=TARGET, umaban="01", ketto_num="A", bamei="カコアリ",
                           result_time_sec="999.9", result_margin_sec="9.87"),
        fx.make_race_horse(race_id=TARGET, umaban="02", ketto_num="B", bamei="カコナシ",
                           result_kakutei_jyuni="2", result_time_sec="999.9"),
    ]
    horses = [fx.make_horse(ketto_num="B", bamei="カコナシ", bamei_kana="ｶｺﾅｼ",
                            make_date="20260101")]
    data_dir = fx.write_dataset(tmp_path, races, race_horses, horses)
    return Repository(DataStore.open_current(data_dir))


def _race_input(**kw):
    base = dict(
        date=date(2026, 3, 21), jyo="中山", race_num=11, track_kind="turf", kyori_m=2000,
        entry_count=2, prediction_cutoff=datetime(2026, 3, 20, 21, 25, tzinfo=JST),
        horses=(HorseEntry(umaban="01", bamei="カコアリ"),
                HorseEntry(umaban="02", bamei="カコナシ")),
        jyo_cd="06", race_id=TARGET)
    base.update(kw)
    return RaceInput(**base)


def test_top_level_keys(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert tuple(d) == ("dossier_version", "prediction_cutoff", "course", "race",
                        "horses", "condition_definitions", "history_status_notes",
                        "field_summary", "data_quality")
    assert d["dossier_version"] == config.DOSSIER_VERSION


def test_condition_definitions_are_hoisted_once_and_absent_per_horse(tmp_path):
    """定義文は state の重複の主因の1つ（brief: 各馬の conditions 5区分に同じ
    definition 文が繰り返される）。top-level に1回だけ持つ。"""
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert tuple(d["condition_definitions"]) == history.COMBINED_CONDITION_KEYS
    for key in history.COMBINED_CONDITION_KEYS:
        assert isinstance(d["condition_definitions"][key], str)
        assert d["condition_definitions"][key]
    for h in d["horses"].values():
        for cond in h["conditions"].values():
            assert "definition" not in cond


def test_history_status_notes_present_only_for_occurring_statuses(tmp_path):
    """注記文も馬ごとに繰り返さない。出現した status ぶんだけ、ソート済みで持つ
    （brief: history.note が全頭に固定文言を繰り返す）。"""
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert d["field_summary"]["history_status_counts"] == {"found": 1, "not_in_window": 1}
    assert d["history_status_notes"] == {
        "found": history.STATUS_NOTES["found"],
        "not_in_window": history.STATUS_NOTES["not_in_window"],
    }
    assert tuple(d["history_status_notes"]) == tuple(sorted(d["history_status_notes"]))
    for h in d["horses"].values():
        assert "note" not in h["history"]


def test_every_horse_has_same_keys(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    key_sets = {tuple(h) for h in d["horses"].values()}
    assert key_sets == {("bamei", "sex_age", "futan_kg", "waku", "kisyu", "history",
                         "recent_runs", "vs_last_run", "conditions", "ability",
                         "running_style")}


def test_course_is_identified_from_race_conditions(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert d["course"]["identified"] is True
    assert d["course"]["track_label"] == "芝・右外"
    assert "rail_basis" in d["course"]["facts_source"]


def test_course_not_identified_without_race_id(tmp_path):
    d = dossier.build_dossier(_race_input(race_id=None), _dataset(tmp_path))
    assert d["course"]["identified"] is False
    assert d["course"]["facts"] is None
    assert d["horses"]["01"]["conditions"]["course_exact_distance"]["starts"] is None


def test_history_distinguishes_horses(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert d["horses"]["01"]["history"]["runs_found"] == 1
    assert d["horses"]["01"]["recent_runs"][0]["kakutei_jyuni"] == 2
    assert d["horses"]["02"]["history"]["status"] == "not_in_window"  # class_level 3
    assert d["horses"]["02"]["recent_runs"] == []


def test_target_race_result_never_leaks(tmp_path):
    text = json.dumps(dossier.build_dossier(_race_input(), _dataset(tmp_path)),
                      ensure_ascii=False)
    assert "999.9" not in text
    assert "9.87" not in text


def test_dossier_is_json_safe(tmp_path):
    json.dumps(dossier.build_dossier(_race_input(), _dataset(tmp_path)), ensure_ascii=False)


def test_field_summary_counts(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    fs = d["field_summary"]
    assert fs["n_horses"] == 2
    assert fs["n_with_history"] == 1
    assert fs["history_status_counts"] == {"found": 1, "not_in_window": 1}


def test_to_state_adds_pace_forecast_only_when_given(tmp_path):
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert "pace_forecast" not in dossier.to_state(d)
    state = dossier.to_state(d, pace_forecast={"probabilities": {"fast": 0.2}})
    assert state["pace_forecast"] == {"probabilities": {"fast": 0.2}}
    assert "pace_forecast" not in d  # 元の資料を書き換えない


def test_check_state_size_raises_over_limit():
    with pytest.raises(dossier.StateTooLargeError) as info:
        dossier.check_state_size({"x": "あ" * 100}, limit=50)
    assert info.value.chars > 50


def test_check_state_size_returns_chars_within_limit():
    assert dossier.check_state_size({"x": "a"}, limit=1000) == dossier.state_chars({"x": "a"})


def test_scratch_only_history_is_not_found(tmp_path):
    """過去の出走が取消（result_ijyo_cd "1"）だけの馬は、走っていないので
    found にしない（final review I-2）。recent_runs と同じ出走集合で判定する。"""
    races = [
        fx.make_race(race_id="2026030106010101", hasso_at="2026-03-01T15:00:00+09:00",
                     track_cd="18", kyori_m="2000"),
        fx.make_race(race_id=TARGET, hasso_at="2026-03-21T15:25:00+09:00",
                     track_cd="18", kyori_m="2000", class_level="3"),
    ]
    race_horses = [
        fx.make_race_horse(race_id="2026030106010101", umaban="03", ketto_num="A",
                           bamei="カコアリ", result_ijyo_cd="1", result_kakutei_jyuni="0",
                           make_date="20260302"),
        fx.make_race_horse(race_id=TARGET, umaban="01", ketto_num="A", bamei="カコアリ"),
        fx.make_race_horse(race_id=TARGET, umaban="02", ketto_num="B", bamei="カコナシ",
                           result_kakutei_jyuni="2"),
    ]
    horses = [fx.make_horse(ketto_num="B", bamei="カコナシ", bamei_kana="ｶｺﾅｼ",
                            make_date="20260101")]
    repo = Repository(DataStore.open_current(
        fx.write_dataset(tmp_path, races, race_horses, horses)))
    d = dossier.build_dossier(_race_input(), repo)
    history = d["horses"]["01"]["history"]
    assert history["status"] == "not_in_window"  # class_level 3
    assert history["runs_found"] == 0
    assert d["horses"]["01"]["recent_runs"] == []
    assert d["field_summary"]["n_with_history"] == 0


def test_front_runner_definition_states_the_threshold_exactly(tmp_path, monkeypatch):
    """文言の閾値と判定に使う値が一致する（final review I-3c）。0.33 と書いて
    1/3 で判定すると、0.333 の馬が文言では外なのに数えられる。"""
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path))
    assert config.FRONT_RUNNER_MAX_CORNER_POS == 1 / 3
    assert "1/3 以下" in d["field_summary"]["front_runner_definition"]
    monkeypatch.setattr(config, "FRONT_RUNNER_MAX_CORNER_POS", 0.35)
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path / "b"))
    assert "0.35 以下" in d["field_summary"]["front_runner_definition"]
    monkeypatch.setattr(config, "FRONT_RUNNER_MAX_CORNER_POS", 0.3412345)
    d = dossier.build_dossier(_race_input(), _dataset(tmp_path / "c"))
    assert "0.3412345 以下" in d["field_summary"]["front_runner_definition"]
