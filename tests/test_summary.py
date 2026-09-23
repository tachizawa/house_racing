"""keiba.features.summary のテスト。

このプロジェクトでもっとも重要なテストが置かれる場所。型（EntryRow /
ResultRow の分離）と Repository の as_of ゲートが未来情報混入の一段目の
防御で、ここに置く文字列検索テストが最後の防御線になる。漏れても例外は
出ず、評価の数字が静かに良くなるだけなので、ここで捕まえるしかない。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from keiba.data.repository import Repository
from keiba.data.store import DataStore
from keiba.features.context import build_context
from keiba.features.summary import (
    PAST_TRANSPORT_NOTE,
    TRANSPORT_ESTIMATE_NOTE,
    build_horse_state,
    build_race_state,
    data_quality,
    horse_criteria,
)
from keiba.models import HorseEntry, RaceInput
from tests.fixtures import csv_data as fx

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


def open_repository(tmp_path, races=None, race_horses=None, horses=None, **kw) -> Repository:
    data_dir = fx.write_dataset(tmp_path, races or [], race_horses or [], horses or [], **kw)
    return Repository(DataStore.open_current(data_dir))


def make_entry(*, umaban="01", bamei="テストウマ", **kw) -> HorseEntry:
    return HorseEntry(umaban=umaban, bamei=bamei, **kw)


def make_race_input(horses, **kw) -> RaceInput:
    defaults = dict(
        date=date(2026, 3, 21),
        jyo="中山",
        race_num=11,
        track_kind="turf",
        kyori_m=2000,
        entry_count=len(horses),
        prediction_cutoff=jst(2026, 3, 20, 20, 0, 0),
        horses=tuple(horses),
        jyo_cd="06",
        baba_cd="1",
    )
    defaults.update(kw)
    return RaceInput(**defaults)


def _mixed_context(tmp_path):
    """1頭は履歴あり、1頭は未登録という、キー集合の一致を確かめるための最小構成。"""
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(
        race_id="R1", umaban="01", ketto_num="0000000001", bamei="トウロクウマ",
        result_margin_sec="0.3",
    )]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entries = [
        make_entry(umaban="01", bamei="トウロクウマ", sex_age="牡4", futan_kg=57.0),
        make_entry(umaban="02", bamei="ミトウロクウマ"),
    ]
    race_input = make_race_input(entries)
    return build_context(race_input, repo)


# --- 未来情報の混入防止（最優先） ----------------------------------------------

@pytest.fixture
def repo_with_finished_race(tmp_path):
    """対象レースに、偶然の一致がありえない特徴的な結果値を持たせる。

    わざと現実にありえない値を使う。着順281・人気283・入線頭数285 のような値は、
    state に現れる正当な特徴量とは絶対に一致しないので、部分一致で誤検出も
    見逃しも起きない。「もっともらしい値」（12 や 1 など）に直すとテストが
    偶然通るようになり、通ったことが信用できなくなる——2桁の値は本物の
    特徴量（斤量・馬番・クラス順序値など）とたまたま部分一致しうるため。
    互いの部分文字列にもならないよう、時計・オッズの小数と桁を分けてある。

    対象レースの発走は prediction_cutoff より後（=まだ結果を知り得ない時点で
    予想する）。だが保存されている CSV にはすでに確定結果が入っている——
    バックテストと同じ状況を作り、それでも state に漏れないことを確認する。
    """
    sentinel = {
        "kakutei_jyuni": "281",
        "time_sec": "999.9",
        "margin_sec": "888.8",
        "odds": "987.6",
        "ninki": "283",
        "nyusen_tosu": "285",
    }
    race_id = "2026092106011101"
    races = [fx.make_race(
        race_id=race_id, hasso_at="2026-09-21T15:25:00+09:00",
        nyusen_tosu=sentinel["nyusen_tosu"], data_kubun="7",
    )]
    race_horses = [fx.make_race_horse(
        race_id=race_id, umaban="01", wakuban="1", ketto_num="0000000001",
        bamei="タイショウウマ",
        result_kakutei_jyuni=sentinel["kakutei_jyuni"],
        result_time_sec=sentinel["time_sec"],
        result_margin_sec=sentinel["margin_sec"],
        result_odds=sentinel["odds"],
        result_ninki=sentinel["ninki"],
        data_kubun="7",
    )]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="タイショウウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)

    entry = make_entry(umaban="01", bamei="タイショウウマ")
    race_input = make_race_input([entry], race_id=race_id)
    return repo, race_input, sentinel


def test_target_race_results_never_appear_in_state(repo_with_finished_race):
    """対象レースの結果が state に混ざらないことを、文字列検索で確かめる。

    型（EntryRow / ResultRow）と Repository の as_of ゲートで防いではいるが、
    dict を組み立てる過程で取り違える余地は残る。JSON にしてから結果の実値を
    検索し、1件も出ないことを確認する。ここが最後の防御線で、漏れても
    例外は出ず、評価の数字が静かに良くなるだけ。
    """
    repo, race_input, sentinel = repo_with_finished_race
    ctx = build_context(race_input, repo)

    race_blob = json.dumps(build_race_state(ctx), ensure_ascii=False)
    horse_blob = json.dumps(build_horse_state(ctx, "01"), ensure_ascii=False)
    # horse_criteria も Choice の criteria としてモデルに渡る。state と同じく
    # 対象レースの結果が混ざってはいけない。型では守っていない経路なので、
    # ここも同じ文字列検索の対象にする。
    criteria_blob = json.dumps(horse_criteria(ctx), ensure_ascii=False)

    for value in sentinel.values():
        assert value not in race_blob, f"{value} が第1層の state に混入している"
        assert value not in horse_blob, f"{value} が第2層の state に混入している"
        assert value not in criteria_blob, f"{value} が horse_criteria に混入している"


def test_future_races_never_appear_in_state(tmp_path):
    """as_of より後のレースの名前・日付・数値が state に混ざらない。"""
    ketto = "0000000001"
    races = [fx.make_race(
        race_id="R_FUTURE", hasso_at="2099-09-09T15:25:00+09:00",
        race_name="ミライトクベツダイキネン",
    )]
    race_horses = [fx.make_race_horse(
        race_id="R_FUTURE", umaban="01", ketto_num=ketto, bamei="タイショウウマ",
        result_margin_sec="55.5",
    )]
    horses = [fx.make_horse(ketto_num=ketto, bamei="タイショウウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)

    entry = make_entry(umaban="01", bamei="タイショウウマ")
    race_input = make_race_input([entry])  # prediction_cutoff は 2026-03-20
    ctx = build_context(race_input, repo)

    blob = json.dumps(build_race_state(ctx), ensure_ascii=False)
    assert "ミライトクベツダイキネン" not in blob
    assert "2099-09-09" not in blob
    assert "55.5" not in blob


def test_state_has_no_cumulative_um_aggregates(tmp_path):
    """UM の累計成績（Ruikei* / ChakuKaisu* / Kyakusitu）に対応するキーは
    どこにも現れない（HorseMaster がそもそも持たない値。models.py 参照）。"""
    ctx = _mixed_context(tmp_path)
    blob = json.dumps(build_race_state(ctx), ensure_ascii=False)
    for forbidden in ("ruikei", "chaku_kaisu", "kyakusitu"):
        assert forbidden not in blob.lower()


def test_state_is_json_serialisable(tmp_path):
    """json.dumps(state, ensure_ascii=False) が default= なしで通る。
    date/datetime はすべて事前に ISO 文字列へ変換してある。"""
    ctx = _mixed_context(tmp_path)
    race_state = build_race_state(ctx)
    horse_state = build_horse_state(ctx, "02")  # 未登録馬（値が全部 None）でも通る

    json.dumps(race_state, ensure_ascii=False)
    json.dumps(horse_state, ensure_ascii=False)


# --- キー名・キー集合 -----------------------------------------------------------

def test_race_state_uses_spec_key_names(tmp_path):
    """トップレベルのキーが prediction_cutoff / race / horses / data_quality
    （spec §5.1）。"""
    ctx = _mixed_context(tmp_path)
    state = build_race_state(ctx)
    assert set(state.keys()) == {"prediction_cutoff", "race", "horses", "data_quality"}


def test_pace_definition_omitted_when_absent(tmp_path):
    """pace を評価しないなら、キーごと出さない。None を置かない（race_context
    と同じ理由）。"""
    ctx = _mixed_context(tmp_path)
    assert ctx.pace_definition is None
    state = build_race_state(ctx)
    assert "pace_definition" not in state


def test_pace_definition_present_when_supplied(tmp_path):
    """pace を評価するなら、閾値の定義を state に載せる。

    第1層の pace 質問は「区分の意味は pace_definition に従ってください」と
    指示する。定義が state に無いと、モデルは自分で閾値を作ってしまう。
    """
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(race_id="R1", umaban="01", ketto_num="0000000001",
                                       bamei="トウロクウマ")]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(umaban="01", bamei="トウロクウマ")
    race_input = make_race_input([entry])
    pace_definition = {
        "stratum": "track_kind", "race_count": 480,
        "thresholds": {"fast": 34.5, "slow": 36.0},
    }

    ctx = build_context(race_input, repo, pace_definition=pace_definition)
    state = build_race_state(ctx)

    assert state["pace_definition"] == pace_definition
    assert set(state.keys()) == {
        "prediction_cutoff", "race", "horses", "data_quality", "pace_definition",
    }


def test_horse_state_uses_spec_key_names(tmp_path):
    """prediction_cutoff / race / target_horse / opponents_summary /
    race_context / data_quality（spec §6.1）。race_context を含む全キーが
    揃うのは pace を渡したときだけなので、ここでは渡して確認する。"""
    ctx = _mixed_context(tmp_path)
    race_context = {"pace_probabilities": {"slow": 0.2, "standard": 0.5, "fast": 0.3},
                     "note": "第1層で得た予測分布。確定した事実ではない"}
    state = build_horse_state(ctx, "01", race_context=race_context)
    assert set(state.keys()) == {
        "prediction_cutoff", "race", "target_horse", "opponents_summary",
        "race_context", "data_quality",
    }


def test_all_horses_share_the_same_key_set(tmp_path):
    """履歴のある馬と未登録の馬が混在していても、horses の各値のキー集合は
    全馬で同一（spec §5.1「同じ書式で入れる」）。"""
    ctx = _mixed_context(tmp_path)
    state = build_race_state(ctx)
    key_sets = [set(horse_dict.keys()) for horse_dict in state["horses"].values()]
    assert len(state["horses"]) == 2
    assert key_sets[0] == key_sets[1]


def test_missing_values_stay_null_not_zero(tmp_path):
    """未登録馬の ability.recent_margin_sec_wavg は None（0 ではない）。"""
    ctx = _mixed_context(tmp_path)
    state = build_race_state(ctx)
    assert state["horses"]["02"]["ability"]["recent_margin_sec_wavg"] is None
    assert state["horses"]["02"]["ability"]["runs_used"] == 0


def test_aggregates_always_carry_starts(tmp_path):
    """by_condition の各区分に starts が必ずある（starts=0 の区分を含む）。"""
    ctx = _mixed_context(tmp_path)
    state = build_race_state(ctx)
    for horse_dict in state["horses"].values():
        for record in horse_dict["by_condition"].values():
            assert "starts" in record
            assert isinstance(record["starts"], int)


# --- same_baba の inapplicable_reason（fix2） ------------------------------------

def _context_with_unknown_baba(tmp_path):
    """今回の馬場状態が未入力（baba_cd=None）な PredictionContext。"""
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(
        race_id="R1", umaban="01", ketto_num="0000000001", bamei="トウロクウマ",
        result_margin_sec="0.3",
    )]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(umaban="01", bamei="トウロクウマ", sex_age="牡4", futan_kg=57.0)
    race_input = make_race_input([entry], baba_cd=None)
    return build_context(race_input, repo)


def test_state_has_all_six_keys_for_an_inapplicable_record(tmp_path):
    """馬場状態が未入力のとき、same_baba は6キーすべてを持ち、件数3つは
    None、inapplicable_reason は文字列（キーを落とさない。決定4・決定5）。"""
    ctx = _context_with_unknown_baba(tmp_path)
    state = build_race_state(ctx)
    record = state["horses"]["01"]["by_condition"]["same_baba"]
    assert set(record.keys()) == {
        "starts", "wins", "top3", "margin_sec_mean", "period", "inapplicable_reason",
    }
    assert record["starts"] is None
    assert record["wins"] is None
    assert record["top3"] is None
    assert isinstance(record["inapplicable_reason"], str)


def test_state_has_all_six_keys_for_a_normal_record(tmp_path):
    """馬場状態が入力されている通常のケースでは、6キーすべてが存在し
    inapplicable_reason は None。"""
    ctx = _mixed_context(tmp_path)
    state = build_race_state(ctx)
    record = state["horses"]["01"]["by_condition"]["same_baba"]
    assert set(record.keys()) == {
        "starts", "wins", "top3", "margin_sec_mean", "period", "inapplicable_reason",
    }
    assert record["inapplicable_reason"] is None


def test_all_five_condition_keys_are_present_when_going_is_unknown(tmp_path):
    """same_baba が定義できなくても by_condition のキーは5つ揃ったまま
    （決定4: プロンプトが名指しするキーを落とさない）。"""
    ctx = _context_with_unknown_baba(tmp_path)
    state = build_race_state(ctx)
    by_condition = state["horses"]["01"]["by_condition"]
    assert set(by_condition.keys()) == {
        "exact_distance", "same_distance", "same_venue", "same_track_kind", "same_baba",
    }


# --- 輸送の注記 -----------------------------------------------------------------

def test_transport_note_marks_estimate(tmp_path):
    """estimated_note が「概数であって実際の経路・輸送日を示すものではない」
    という趣旨を持つ（spec §4.2）。"""
    ctx = _mixed_context(tmp_path)
    state = build_race_state(ctx)
    note = state["horses"]["01"]["transport"]["estimated_note"]
    assert note == TRANSPORT_ESTIMATE_NOTE
    assert "概数" in note
    assert "実際の" in note and "輸送日" in note

    past_note = state["horses"]["01"]["transport"]["past_long_transport"]["note"]
    assert past_note == PAST_TRANSPORT_NOTE
    assert "推定" in past_note
    assert "輸送であることを示すものではない" in past_note


def test_transport_records_base_source(tmp_path):
    """transport.base_source が state に出る。_mixed_context は対象レースの
    entries() を持たない（race_id 未指定）ので、履歴のある馬は直近走からの
    推定（"last_run"）になる。base を確認済みの値として読ませないため
    （spec §4.2）。"""
    ctx = _mixed_context(tmp_path)
    state = build_race_state(ctx)
    assert state["horses"]["01"]["transport"]["base_source"] == "last_run"


def test_data_quality_missing_includes_transport_when_unresolvable(tmp_path):
    """所属コードを解決する手がかりが無い馬（未登録・entries()も過去走も無い）
    は、transport も missing に含まれる（決定3）。"""
    ctx = _mixed_context(tmp_path)
    dq = data_quality(ctx)
    assert "transport" in dq["missing"]["02"]
    assert "transport" not in dq["missing"].get("01", [])


def test_transport_stay_status_unknown_by_default(tmp_path):
    """滞在有無を入力する手段が無いので、遠隔開催でも常に unknown。"""
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00", jyo_cd="01")]
    race_horses = [fx.make_race_horse(race_id="R1", umaban="01", ketto_num="0000000001",
                                       bamei="トウロクウマ")]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(umaban="01", bamei="トウロクウマ")
    # 今回のレース自体は遠隔開催（札幌）とする。
    race_input = make_race_input([entry], jyo="札幌", jyo_cd="01")

    ctx = build_context(race_input, repo)
    state = build_race_state(ctx)
    assert state["horses"]["01"]["transport"]["stay_status"] == "unknown"


# --- opponents_summary -----------------------------------------------------------

def test_opponents_summary_includes_target_horse(tmp_path):
    """相手要約に対象馬自身も含まれる（相手の中での位置が分かるように）。"""
    ctx = _mixed_context(tmp_path)
    state = build_horse_state(ctx, "01")
    assert "01" in state["opponents_summary"]
    assert "02" in state["opponents_summary"]


def test_opponents_summary_is_short(tmp_path):
    """相手要約の各馬のキー数が target_horse より少ない（ability と
    running_style だけの簡略版）。"""
    ctx = _mixed_context(tmp_path)
    state = build_horse_state(ctx, "01")
    opponent_keys = set(state["opponents_summary"]["02"].keys())
    target_keys = set(state["target_horse"].keys())
    assert opponent_keys == {"bamei", "ability", "running_style"}
    assert len(opponent_keys) < len(target_keys)


# --- race_context -----------------------------------------------------------------

def test_race_context_omitted_when_no_pace(tmp_path):
    """race_context=None のとき state にキーが無い（キー自体を省く。None を
    入れない）。"""
    ctx = _mixed_context(tmp_path)
    state = build_horse_state(ctx, "01", race_context=None)
    assert "race_context" not in state


def test_race_context_keeps_all_three_pace_probabilities(tmp_path):
    """pace を渡すと3区分すべての確率が入り、最大のものに置き換わっていない
    （spec §5.3）。"""
    ctx = _mixed_context(tmp_path)
    race_context = {
        "pace_probabilities": {"slow": 0.2, "standard": 0.5, "fast": 0.3},
        "note": "第1層で得た予測分布。確定した事実ではない",
    }
    state = build_horse_state(ctx, "01", race_context=race_context)
    probs = state["race_context"]["pace_probabilities"]
    assert set(probs.keys()) == {"slow", "standard", "fast"}
    assert probs == {"slow": 0.2, "standard": 0.5, "fast": 0.3}


# --- horse_criteria -----------------------------------------------------------------

def test_horse_criteria_keys_are_umaban(tmp_path):
    ctx = _mixed_context(tmp_path)
    criteria = horse_criteria(ctx)
    assert set(criteria.keys()) == {"01", "02"}


def test_horse_criteria_values_identify_the_horse(tmp_path):
    """値に馬名・性齢・斤量が入る。"""
    ctx = _mixed_context(tmp_path)
    criteria = horse_criteria(ctx)
    value = criteria["01"]
    assert "トウロクウマ" in value
    assert "牡4" in value
    assert "57" in value


# --- data_quality -----------------------------------------------------------------

def test_data_quality_lists_missing_per_horse(tmp_path):
    ctx = _mixed_context(tmp_path)
    dq = data_quality(ctx)
    assert "ability" in dq["missing"]["02"]
    assert "01" not in dq["missing"]


def test_data_quality_carries_data_version_and_fetched_at(tmp_path):
    ctx = _mixed_context(tmp_path)
    dq = data_quality(ctx)
    assert dq["data_version"] == ctx.data_version
    assert dq["data_fetched_at"] == ctx.data_fetched_at.isoformat()


def test_race_level_note_still_reaches_data_quality(tmp_path):
    """ctx.notes に入れたレース単位の注記が data_quality()["notes"] に現れる。

    fix4 で ``notes`` はレース単位専用になり、馬単位は ``horse_notes`` に
    分離された（決定1）。この分離後も、レース単位の注記が state / 保存記録 /
    利用者向けビューに届く唯一の経路（``list(ctx.notes)``）は生きていることを
    確かめる（fix3 で追加した回帰テストを、新しい形に合わせて引き継ぐ）。
    """
    ctx = _context_with_unknown_baba(tmp_path)
    assert len(ctx.notes) == 1  # このケースでは馬場状態の注記が1件だけ立つ

    dq = data_quality(ctx)

    assert ctx.notes[0] in dq["notes"]


def test_notes_holds_only_race_level_entries(tmp_path):
    """data_quality()["notes"] には ctx.notes（レース単位）の文だけが入り、
    馬単位の注記（horse.notes）は混ざらない（fix4 決定1）。

    ``_context_with_unknown_baba`` は馬場状態未入力によるレース単位の注記と、
    対象レースの出馬表がまだ無いことによる馬単位の注記（輸送起点の推定）を
    同時に持つ——両方が同時に存在する状態で、notes に馬単位のほうが混ざって
    いないことを確かめられる。
    """
    ctx = _context_with_unknown_baba(tmp_path)
    assert ctx.notes != ()
    assert ctx.horses[0].notes != ()  # 比較対象として、馬単位の注記も実在する

    dq = data_quality(ctx)

    assert dq["notes"] == list(ctx.notes)
    for note in ctx.horses[0].notes:
        assert note not in dq["notes"]


def test_horse_notes_is_keyed_by_umaban(tmp_path):
    """注記を持つ馬の馬番が horse_notes のキーになり、その馬の注記が値になる。"""
    ctx = _context_with_unknown_baba(tmp_path)
    dq = data_quality(ctx)
    assert dq["horse_notes"]["01"] == list(ctx.horses[0].notes)


def _context_with_a_noteless_horse(tmp_path):
    """01番は notes を持たない（登録済み・今回の出馬表があるので輸送起点の
    推定に頼らない・斤量や騎手の変更もない）。02番は未登録で notes を持つ。
    horse_notes が「無い馬は省く」ことを確かめるための対比を作る
    （test_context.py の test_marks_affiliation_from_todays_entry_when_available
    と同じ構成）。
    """
    races = [fx.make_race(race_id="R_TARGET", hasso_at="2026-03-25T15:25:00+09:00",
                           data_kubun="2")]
    race_horses = [fx.make_race_horse(
        race_id="R_TARGET", umaban="01", ketto_num="0000000001", bamei="トウロクウマ",
        tozai_cd="2", data_kubun="2",
        result_ijyo_cd="", result_kakutei_jyuni="", result_time_sec="",
        result_margin_sec="", result_odds="", result_ninki="",
    )]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entries = [
        make_entry(umaban="01", bamei="トウロクウマ"),
        make_entry(umaban="02", bamei="ミトウロクウマ"),
    ]
    race_input = make_race_input(entries, race_id="R_TARGET")
    return build_context(race_input, repo)


def test_horse_notes_omits_horses_without_notes(tmp_path):
    """注記の無い馬は horse_notes にキーとして現れない
    （missing と同じ「言うことが無い」を不在で表す約束）。"""
    ctx = _context_with_a_noteless_horse(tmp_path)
    by_umaban = {h.umaban: h for h in ctx.horses}
    assert by_umaban["01"].notes == ()
    assert by_umaban["02"].notes != ()

    dq = data_quality(ctx)

    assert "01" not in dq["horse_notes"]
    assert "02" in dq["horse_notes"]
