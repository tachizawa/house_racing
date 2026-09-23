"""keiba.input.match のテスト。

最優先は「絞れない同名馬を勝手に1頭選ばないこと」。誤った馬の履歴で
予想すると、結果はもっともらしい見た目のまま静かに壊れる（仕様 14.3 節）。
これを担保する test_does_not_auto_pick_among_same_name_horses と、
それと対をなす（未登録はエラーではない）test_unregistered_is_not_an_error を
最初に書く。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest import mock

import pytest

from keiba.data.repository import Repository
from keiba.data.store import DataStore
from keiba.input import match as match_module
from keiba.input.match import (
    AmbiguousMatchError,
    MatchResult,
    match_horses,
    normalize_name,
)
from keiba.models import HorseEntry, HorseMaster, RaceInput
from tests.fixtures import csv_data as fx

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


def open_repository(tmp_path, races=None, race_horses=None, horses=None, **kw) -> Repository:
    data_dir = fx.write_dataset(tmp_path, races or [], race_horses or [], horses or [], **kw)
    return Repository(DataStore.open_current(data_dir))


def make_entry(*, umaban="01", bamei="テストウマ", sex_age=None, **kw) -> HorseEntry:
    return HorseEntry(umaban=umaban, bamei=bamei, sex_age=sex_age, **kw)


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
    )
    defaults.update(kw)
    return RaceInput(**defaults)


AS_OF = jst(2026, 3, 21, 10, 0, 0)


# --- normalize_name -----------------------------------------------------

def test_normalize_collapses_width_and_symbols():
    """全角カタカナ・半角カタカナ・空白入り・中黒入りがすべて同じ値になる。"""
    base = normalize_name("テストウマ")
    assert normalize_name("ﾃｽﾄｳﾏ") == base
    assert normalize_name("テスト ウマ") == base
    assert normalize_name("テスト・ウマ") == base


def test_normalize_keeps_katakana_as_katakana():
    """ひらがなに寄せない。カタカナとひらがなは別の馬として扱う。"""
    assert normalize_name("テスト") != normalize_name("てすと")


# --- match_horses: 核となる規則 ------------------------------------------

def test_does_not_auto_pick_among_same_name_horses(tmp_path):
    """同名馬が2頭いて絞れないとき、勝手に1頭を選ばない。

    別の馬の履歴で予想すると、もっともらしい予想が別の馬の実績の上に
    建つ。下流の何も異常を示さないので、ここで止めるしかない。
    """
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="1"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(umaban="01", bamei="ドウメイウマ")
    race_input = make_race_input([entry])

    with pytest.raises(AmbiguousMatchError) as excinfo:
        match_horses(race_input, repo, as_of=AS_OF)

    error = excinfo.value
    assert len(error.results) == 1
    result = error.results[0]
    assert result.status == "ambiguous"
    assert result.ketto_num is None  # 選ばない
    assert {c.ketto_num for c in result.candidates} == {"0000000001", "0000000002"}


def test_match_does_not_auto_pick_first_candidate(tmp_path):
    """候補の並び順を変えても、先頭を勝手に採ったりしない（例外が出続けること）。"""
    horses = [
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="ドウメイウマ")
    race_input = make_race_input([entry])

    with pytest.raises(AmbiguousMatchError):
        match_horses(race_input, repo, as_of=AS_OF)


def test_unregistered_is_not_an_error(tmp_path):
    """未登録は曖昧とは違う。過去実績なしとして続行できる。"""
    repo = open_repository(tmp_path, horses=[])
    entry = make_entry(umaban="01", bamei="ミトウロクウマ")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert len(results) == 1
    assert results[0].status == "unregistered"
    assert results[0].ketto_num is None
    assert results[0].candidates == ()


def test_unregistered_is_not_ambiguous(tmp_path):
    """未登録の馬と一意に決まる馬が混在していても、未登録側は例外を出さない。"""
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, horses=horses)
    entries = [
        make_entry(umaban="01", bamei="トウロクウマ"),
        make_entry(umaban="02", bamei="ミトウロクウマ"),
    ]
    race_input = make_race_input(entries)

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert len(results) == 2
    by_umaban = {r.umaban: r for r in results}
    assert by_umaban["01"].status == "matched"
    assert by_umaban["01"].ketto_num == "0000000001"
    assert by_umaban["02"].status == "unregistered"
    assert by_umaban["02"].ketto_num is None


def test_match_raises_on_ambiguous_even_with_other_matched_horses(tmp_path):
    """曖昧な馬が1頭でもいれば、レース全体を例外にする。

    results にはレース全頭分（曖昧でない馬も含む）が入っている必要がある。
    ユーザーが一度で全部の問題を把握できるようにするため。
    """
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ"),
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000003", bamei="ドウメイウマ", sex_cd="1"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entries = [
        make_entry(umaban="01", bamei="トウロクウマ"),
        make_entry(umaban="02", bamei="ドウメイウマ"),
    ]
    race_input = make_race_input(entries)

    with pytest.raises(AmbiguousMatchError) as excinfo:
        match_horses(race_input, repo, as_of=AS_OF)

    results = excinfo.value.results
    assert len(results) == 2
    by_umaban = {r.umaban: r for r in results}
    assert by_umaban["01"].status == "matched"
    assert by_umaban["01"].ketto_num == "0000000001"
    assert by_umaban["02"].status == "ambiguous"
    assert len(by_umaban["02"].candidates) == 2


def test_match_resolves_unique_name(tmp_path):
    horses = [fx.make_horse(ketto_num="0000000001", bamei="ユイイツウマ")]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(umaban="03", bamei="ユイイツウマ")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, MatchResult)
    assert result.umaban == "03"
    assert result.status == "matched"
    assert result.ketto_num == "0000000001"
    assert result.candidates == ()
    assert result.entry is entry


def test_match_marks_unregistered_when_no_candidate(tmp_path):
    repo = open_repository(tmp_path, horses=[])
    entry = make_entry(bamei="ソンザイシナイウマ")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "unregistered"
    assert results[0].ketto_num is None


# --- sex_age による絞り込み -------------------------------------------------

def test_match_narrows_by_sex_age(tmp_path):
    """同名2頭を sex_age="牡4" で1頭に絞れる。"""
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1"),  # 牡
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="2"),  # 牝
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="ドウメイウマ", sex_age="牡4")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert len(results) == 1
    assert results[0].status == "matched"
    assert results[0].ketto_num == "0000000001"


def test_match_narrows_by_sex_age_sen(tmp_path):
    """セン（去勢）は2文字なので、牡/牝との前方一致誤判定がないことを確認する。

    生年は申告の馬齢（5歳 = 2026 - 5 = 2021年生）に合わせてある。合っていないと
    馬齢の段階で「この馬ではない」＝未登録になり、性別の判定に届かない。
    """
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1",
                    birth_date="2021-03-15"),  # 牡
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="3",
                    birth_date="2021-04-02"),  # セン
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="ドウメイウマ", sex_age="セン5")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "0000000002"


def test_match_does_not_narrow_on_unparseable_sex_age(tmp_path):
    """sex_age が解釈できない書式なら、絞り込みに使わずそのまま曖昧扱いにする。

    「絞ったつもりで正しい馬を落とす」事故を避けるため、解釈できない
    手がかりは無視する（フィルタしない）。フィルタしない結果、2頭とも残り
    曖昧のままなのが正しい挙動。
    """
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="2"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="ドウメイウマ", sex_age="4歳牡")  # 語順が逆で解釈できない
    race_input = make_race_input([entry])

    with pytest.raises(AmbiguousMatchError) as excinfo:
        match_horses(race_input, repo, as_of=AS_OF)

    assert len(excinfo.value.results[0].candidates) == 2


def test_match_does_not_narrow_when_sex_age_not_supplied(tmp_path):
    """sex_age を入力していないなら絞り込みは行わない（曖昧のまま）。"""
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="2"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="ドウメイウマ", sex_age=None)
    race_input = make_race_input([entry])

    with pytest.raises(AmbiguousMatchError):
        match_horses(race_input, repo, as_of=AS_OF)


def test_several_candidates_with_a_sex_disagreement_is_still_conflict(tmp_path):
    """sex_age は解釈できるが、どの候補の sex_cd とも一致せず、候補も複数の場合。

    名前は一致しているので unregistered ではない（unregistered にすると
    「過去実績なし」という嘘の data_quality を持ったまま予想が進んでしまう）。
    候補が複数残っている以上、名前でも1頭に特定できていない。入力とデータの
    どちらが誤っているかコードには判断できないので、同名馬を勝手に選ばないのと
    同じ理由で conflict として止める。

    候補が1頭だけの場合は別扱い（下の
    test_single_candidate_with_a_sex_disagreement_is_matched_with_a_note）。
    """
    horses = [
        # 同名2頭。どちらも牡で、申告の馬齢（3歳 = 2023年生）に合う。
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1",
                    birth_date="2023-04-10"),
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="1",
                    birth_date="2023-05-20"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="ドウメイウマ", sex_age="牝3")  # 実際とは違う性別を指定
    race_input = make_race_input([entry])

    with pytest.raises(AmbiguousMatchError) as excinfo:
        match_horses(race_input, repo, as_of=AS_OF)

    result = excinfo.value.results[0]
    assert result.status == "conflict"
    assert result.ketto_num is None
    # データが実際に持っている記録をユーザーが見られるよう、性別で絞る前の
    # 候補がそのまま入っている。
    assert {c.ketto_num for c in result.candidates} == {"0000000001", "0000000002"}


def test_single_candidate_with_a_sex_disagreement_is_matched_with_a_note(tmp_path):
    """名前で1頭に特定できているなら、性別が食い違ってもレースを止めない。

    性別は識別の裏付けであって識別子ではない。しかもデータ側の性別は古びうる
    （去勢された馬の UM 行が更新されないまま残る。実データで出走馬の 1.6%）。
    黙って続行しないために注記を残す。
    """
    horses = [fx.make_horse(ketto_num="9900000001", bamei="セイベツウマ", sex_cd="1",
                             birth_date="2023-04-10", make_date="20260320")]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(umaban="05", bamei="セイベツウマ", sex_age="牝3")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    result = results[0]
    assert result.status == "matched"
    assert result.ketto_num == "9900000001"
    assert len(result.notes) == 1
    note = result.notes[0]
    assert "05番" in note          # どの馬か
    assert "牝" in note            # 申告
    assert "牡" in note            # データ
    assert "master" in note        # 出どころ


def test_conflict_and_ambiguous_together_raise_once_with_both_visible(tmp_path):
    """曖昧な馬と矛盾した馬が同じレースにいても、例外は1回で両方見える。

    ユーザーが一度で全部の問題を把握できることを確認する。ラウンドトリップの
    たびに次の問題が発覚するようでは、同じ理由の警告を何度も出すことになる。
    """
    horses = [
        # 同名2頭、どちらも牡。sex_age 無しでは絞れない -> ambiguous。
        fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000002", bamei="ドウメイウマ", sex_cd="1"),
        # 同名2頭、どちらも牡（馬齢は申告どおり4歳）だが、牝で申告 -> conflict。
        # 1頭に特定できていれば matched + 注記になるので、ここは2頭にする。
        fx.make_horse(ketto_num="0000000003", bamei="ショウトツウマ", sex_cd="1"),
        fx.make_horse(ketto_num="0000000004", bamei="ショウトツウマ", sex_cd="1"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entries = [
        make_entry(umaban="01", bamei="ドウメイウマ"),
        make_entry(umaban="02", bamei="ショウトツウマ", sex_age="牝4"),
    ]
    race_input = make_race_input(entries)

    with pytest.raises(AmbiguousMatchError) as excinfo:
        match_horses(race_input, repo, as_of=AS_OF)

    results = excinfo.value.results
    assert len(results) == 2
    by_umaban = {r.umaban: r for r in results}
    assert by_umaban["01"].status == "ambiguous"
    assert by_umaban["02"].status == "conflict"


# --- sex_age の馬齢による絞り込み（性別より先） -----------------------------
#
# JRA は抹消後に馬名を再使用する。馬マスタは 1936 年まで遡るので、同名の馬は
# 普通に複数いる（実データで正規化名 190,148 名のうち 7,371 件、3.9%）。
# 馬齢で解ける: 同名グループ 2,000 件を調べて生年月日が重なるものは 0 件だった。
# 馬齢の規則も実データで検証済み（950行中950行）: 馬齢 = レースの年 - 生年。

def _two_old_namesakes() -> list[dict]:
    """同名2頭。どちらも申告の馬齢（2歳 = 2024年生）には合わない生年。

    実データで中止していたレースの形（架空の2歳牝馬 に対して候補が
    1989年生と1996年生だけ）をそのまま写したもの。
    """
    return [
        fx.make_horse(ketto_num="9900001989", bamei="サイリヨウウマ", sex_cd="2",
                    birth_date="1989-04-20"),
        fx.make_horse(ketto_num="9900001996", bamei="サイリヨウウマ", sex_cd="2",
                    birth_date="1996-05-02"),
    ]


def test_no_candidate_of_the_declared_age_is_unregistered(tmp_path):
    """申告の馬齢に合う候補が1頭も居ないなら、その馬はデータの中に居ない。

    馬齢 = レースの年 - 生年 は実データで2度検証されている（950/950 の直接照合と
    204/204 の突き合わせ）。生年が合わない候補はこの馬ではないので、
    「決めきれない（ambiguous）」ではなく「未登録（過去実績なし）」が正しい。
    候補を残して ambiguous にしていた頃は、これだけでレース全体が中止していた
    （実測で中止31件の原因）。
    """
    repo = open_repository(tmp_path, horses=_two_old_namesakes())
    entry = make_entry(umaban="07", bamei="サイリヨウウマ", sex_age="牝2")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert len(results) == 1
    result = results[0]
    assert result.status == "unregistered"
    assert result.ketto_num is None
    assert result.candidates == ()             # 選ばないので入れない


def test_that_note_names_the_same_name_candidates(tmp_path):
    """注記に、打ち間違いを見つけられる情報（申告馬齢と同名馬の生年）を入れる。"""
    repo = open_repository(tmp_path, horses=_two_old_namesakes())
    entry = make_entry(umaban="07", bamei="サイリヨウウマ", sex_age="牝2")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert len(results[0].notes) == 1
    note = results[0].notes[0]
    assert "07番" in note
    assert "2歳" in note                       # 申告された馬齢
    assert "1989" in note
    assert "1996" in note


def test_the_note_does_not_name_other_horses_ketto_num(tmp_path):
    """注記に他馬の血統登録番号を出さない（生年だけにする。fix5 決定10）。

    この注記は to_user_view 経由で利用者に届く。入力した馬とは別の馬の
    識別子まで見せる必要はなく、打ち間違いの検出という目的には生年で足りる。
    """
    repo = open_repository(tmp_path, horses=_two_old_namesakes())
    entry = make_entry(umaban="07", bamei="サイリヨウウマ", sex_age="牝2")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    note = results[0].notes[0]
    assert "9900001989" not in note
    assert "9900001996" not in note


def test_the_note_truncates_a_long_candidate_list(tmp_path):
    """同名候補が多いときは3頭までにする。

    18頭立てで全馬に長い注記が付くと読めなくなる。残りは頭数だけ示す。
    """
    horses = [
        fx.make_horse(ketto_num=f"990000200{i}", bamei="オオゼイウマ", sex_cd="2",
                    birth_date=f"199{i}-04-20")
        for i in range(5)
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(umaban="07", bamei="オオゼイウマ", sex_age="牝2")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    note = results[0].notes[0]
    # 生年だけを並べる（血統登録番号は出さない）。5頭のうち3頭まで。
    assert sum(note.count(f"199{i}年生") for i in range(5)) == 3
    assert "他 2 頭" in note


def test_the_race_is_not_aborted(tmp_path):
    """該当する馬が居ても、レース全体を中止しない。

    これが実データで 466レース中30件（6.4%）を捨てていた経路。
    """
    horses = _two_old_namesakes() + [
        fx.make_horse(ketto_num="9900002100", bamei="フツウノウマ", sex_cd="1",
                    birth_date="2022-03-15"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entries = [
        make_entry(umaban="01", bamei="サイリヨウウマ", sex_age="牝2"),
        make_entry(umaban="02", bamei="フツウノウマ", sex_age="牡4"),
    ]
    race_input = make_race_input(entries)

    results = match_horses(race_input, repo, as_of=AS_OF)   # 例外を投げない

    by_umaban = {r.umaban: r for r in results}
    assert by_umaban["01"].status == "unregistered"
    assert by_umaban["02"].status == "matched"
    assert by_umaban["02"].ketto_num == "9900002100"


def test_an_unreadable_age_still_narrows_by_sex_only(tmp_path):
    """馬齢が読めないときは絞らないだけ。unregistered にはしない。

    「馬齢が分からない」は「馬齢が合わない」とは違う。sex_age から数字が
    読めなければ馬齢は使わず、性別だけで絞る従来の流れに戻る
    （sex_age が None のときに絞り込みを一切行わないことは
    test_match_does_not_narrow_when_sex_age_not_supplied が見ている）。
    """
    horses = [
        fx.make_horse(ketto_num="9900002201", bamei="バレイヨメナイウマ", sex_cd="1",
                    birth_date="2008-03-30"),
        fx.make_horse(ketto_num="9900002202", bamei="バレイヨメナイウマ", sex_cd="2",
                    birth_date="1990-03-10"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="バレイヨメナイウマ", sex_age="牡")   # 数字が無い
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"          # unregistered ではない
    assert results[0].ketto_num == "9900002201"
    assert results[0].notes == ()


def test_a_candidate_without_a_birth_year_keeps_the_horse_resolvable(tmp_path):
    """生年不明の候補が残るなら「馬齢に合う候補0頭」ではない。

    birth_year が None の候補は年齢で落ちないので、_narrow_by_age の結果は
    空にならない。判定は「結果が空かどうか」だけで行うので、この場合は
    未登録扱いが発動しない。
    """
    races = [fx.make_race(race_id="2026031506010101",
                           hasso_at="2026-03-15T15:25:00+09:00")]
    race_horses = [
        # SE にしか居ない馬。barei が空なので生年を計算できない。
        fx.make_race_horse(race_id="2026031506010101", ketto_num="9900002301",
                            bamei="セイネンマチマチウマ", sex_cd="1", barei="",
                            make_date="20260316"),
    ]
    horses = [
        fx.make_horse(ketto_num="9900002302", bamei="セイネンマチマチウマ", sex_cd="1",
                    birth_date="1989-04-20"),
    ]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(bamei="セイネンマチマチウマ", sex_age="牡3")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "9900002301"
    assert results[0].notes == ()              # 未登録扱いの注記は付かない


def test_a_matching_age_still_wins(tmp_path):
    """1頭が申告の馬齢に合うなら、その馬に決まる（従来どおり）。"""
    horses = [
        fx.make_horse(ketto_num="9900002401", bamei="ガイトウウマ", sex_cd="1",
                    birth_date="2023-04-10"),
        fx.make_horse(ketto_num="9900002402", bamei="ガイトウウマ", sex_cd="1",
                    birth_date="1989-04-20"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="ガイトウウマ", sex_age="牡3")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "9900002401"
    assert results[0].notes == ()


def test_age_picks_the_right_horse_among_reused_names(tmp_path):
    """同名の再使用は馬齢で解ける。生年が申告の馬齢に合う1頭だけが残る。"""
    horses = [
        fx.make_horse(ketto_num="9900000013", bamei="サイリヨウウマ", sex_cd="1",
                    birth_date="2023-04-10"),
        fx.make_horse(ketto_num="9900000014", bamei="サイリヨウウマ", sex_cd="1",
                    birth_date="1990-03-10"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="サイリヨウウマ", sex_age="牡3")   # 2026 - 3 = 2023年生
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "9900000013"
    assert results[0].notes == ()


def test_candidates_without_a_birth_year_survive_the_age_filter(tmp_path):
    """birth_year が None の候補は馬齢で落とさない。

    生年が分からないだけで、その馬でないとは言えない。None を「年齢が
    合わない」と同じに扱うと、生年の手がかりが無い馬が静かに消える。
    """
    races = [fx.make_race(race_id="2026031506010101",
                           hasso_at="2026-03-15T15:25:00+09:00")]
    race_horses = [
        # SE にしか居ない馬。barei が空なので生年を計算できない（birth_year=None）。
        fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000015",
                            bamei="セイネンフメイウマ", sex_cd="1", barei="",
                            make_date="20260316"),
    ]
    horses = [
        # 同名だが生年が申告の馬齢に合わない馬。こちらは馬齢で落ちる。
        fx.make_horse(ketto_num="9900000016", bamei="セイネンフメイウマ", sex_cd="1",
                    birth_date="1990-03-10"),
    ]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(bamei="セイネンフメイウマ", sex_age="牡3")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "9900000015"
    assert results[0].notes == ()             # 空にならなかったので注記は無い


def test_age_narrows_using_a_race_derived_birth_year(tmp_path):
    """マスタに無い馬でも、出走記録から計算した生年で同名を絞れる。

    セットアップ取得で得られるマスタは生年 1936〜2021 年で 96% が抹消馬、
    出走馬の 86.8%（5,177 頭中 4,494 頭）が入っていない。年齢の絞り込みが
    マスタの birth_date を必要とする限り、現役馬（出走馬の大半）では年齢が
    使えず、同名が出たらレースごと止まる。
    """
    races = [fx.make_race(race_id="2026031506010101",
                           hasso_at="2026-03-15T15:25:00+09:00")]
    race_horses = [
        # SE にしか居ない現役馬。2026年のレースで馬齢3 -> 2023年生まれ。
        fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000021",
                            bamei="ゲンエキウマ", sex_cd="1", barei="03",
                            make_date="20260316"),
    ]
    horses = [
        # 同名の抹消馬。マスタにはこちらしか居ない。
        fx.make_horse(ketto_num="9900000022", bamei="ゲンエキウマ", sex_cd="1",
                       birth_date="1990-03-10", data_kubun="9"),
    ]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(bamei="ゲンエキウマ", sex_age="牡3")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "9900000021"
    assert results[0].notes == ()
    # 絞り込みに使われた生年が、マスタではなく出走記録から来ていること。
    candidates = repo.find_horses_by_name(normalize_name("ゲンエキウマ"), as_of=AS_OF)
    by_ketto = {c.ketto_num: c for c in candidates}
    assert by_ketto["9900000021"].birth_year == 2023
    assert by_ketto["9900000021"].birth_year_source == "race"


def test_age_drops_a_race_derived_birth_year_that_disagrees(tmp_path):
    """出走記録由来の生年も、合わなければ落とす（マスタ由来と同じ扱い）。

    両方の候補が SE にしか居ない場合。birth_date を見ていた頃は、どちらも
    birth_date=None なので絞れず ambiguous でレースごと止まっていた。
    """
    races = [
        fx.make_race(race_id="2026031506010101",
                      hasso_at="2026-03-15T15:25:00+09:00"),
        fx.make_race(race_id="2026031506010102",
                      hasso_at="2026-03-15T16:00:00+09:00"),
    ]
    race_horses = [
        fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000023",
                            bamei="サイリヨウゲンエキウマ", sex_cd="1", barei="03",
                            make_date="20260316"),        # 2023年生まれ
        fx.make_race_horse(race_id="2026031506010102", ketto_num="9900000024",
                            bamei="サイリヨウゲンエキウマ", sex_cd="1", barei="08",
                            make_date="20260316"),        # 2018年生まれ
    ]
    repo = open_repository(tmp_path, races, race_horses, horses=[])
    entry = make_entry(bamei="サイリヨウゲンエキウマ", sex_age="牡3")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "9900000023"
    assert results[0].notes == ()


def test_age_is_applied_before_sex(tmp_path):
    """絞り込みは 名前 -> 馬齢 -> 性別 の順。

    性別を先に当てると、馬齢の合わない同名馬（性別だけが申告と一致）を
    選んでしまう。馬齢のほうが識別子として強い（同名かつ同じ生年月日の馬は
    実データで 0 件）ので、先に馬齢で絞る。
    """
    horses = [
        # 申告の馬齢に合う馬。性別は申告と食い違う。
        fx.make_horse(ketto_num="9900000017", bamei="ジュンバンウマ", sex_cd="1",
                    birth_date="2023-04-10"),
        # 性別だけが申告と一致する、馬齢の合わない同名馬。
        fx.make_horse(ketto_num="9900000018", bamei="ジュンバンウマ", sex_cd="2",
                    birth_date="1990-03-10"),
    ]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(umaban="09", bamei="ジュンバンウマ", sex_age="牝3")
    race_input = make_race_input([entry])

    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert results[0].ketto_num == "9900000017"   # 馬齢で選ばれた馬
    assert len(results[0].notes) == 1             # 性別の食い違いは注記になる


def test_sex_age_is_parsed_once(tmp_path):
    """性別と馬齢は1箇所で解析し、同じ結果を両方の絞り込みに渡す。

    同じ文字列を2箇所で解析すると、片方だけが直される日が来る。
    """
    horses = [fx.make_horse(ketto_num="9900000019", bamei="カイセキウマ", sex_cd="1",
                          birth_date="2023-04-10")]
    repo = open_repository(tmp_path, horses=horses)
    entry = make_entry(bamei="カイセキウマ", sex_age="牡3")
    race_input = make_race_input([entry])

    with mock.patch.object(match_module, "_parse_sex_age",
                            wraps=match_module._parse_sex_age) as parse_spy, \
         mock.patch.object(match_module, "_narrow_by_age",
                            wraps=match_module._narrow_by_age) as age_spy, \
         mock.patch.object(match_module, "_narrow_by_sex",
                            wraps=match_module._narrow_by_sex) as sex_spy:
        results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "matched"
    assert parse_spy.call_count == 1                      # 1頭につき1回だけ解析する
    parsed = age_spy.call_args.args[1]
    assert sex_spy.call_args.args[1] is parsed            # 同じ解析結果が両方に渡る
    assert (parsed.sex_name, parsed.barei) == ("牡", 3)   # 性別と馬齢の両方が入る


# --- 未登録は他馬の履歴を借りない -------------------------------------------

def test_match_does_not_borrow_another_horses_history(tmp_path):
    """unregistered の ketto_num が None なので past_runs を引けない。

    誤って None を渡しても、別の馬（実在する ketto_num）の過去走を
    返してしまわないことを確認する。
    """
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(race_id="R1", ketto_num="0000000099")]
    horses = [fx.make_horse(ketto_num="0000000099", bamei="ジツザイウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)

    entry = make_entry(bamei="ミトウロクウマ")
    race_input = make_race_input([entry])
    results = match_horses(race_input, repo, as_of=AS_OF)

    assert results[0].status == "unregistered"
    assert results[0].ketto_num is None
    # None を渡しても、実在する別の馬（0000000099）の過去走を返さない。
    assert repo.past_runs(results[0].ketto_num, as_of=AS_OF) == []


# --- as_of の受け渡し --------------------------------------------------------

def test_match_passes_as_of_through(tmp_path):
    """repo.find_horses_by_name が as_of 付きで呼ばれる（モックで確認）。"""
    mock_repo = mock.create_autospec(Repository, instance=True)
    mock_repo.find_horses_by_name.return_value = [
        HorseMaster(ketto_num="0000000001", bamei="ユイイツウマ",
                    bamei_kana="ﾕｲｲﾂｳﾏ", birth_date=None, sex_cd="1",
                    sex_cd_source="master", birth_year=2022,
                    birth_year_source="race"),
    ]
    entry = make_entry(bamei="ユイイツウマ")
    race_input = make_race_input([entry])
    as_of = jst(2026, 3, 21, 12, 34, 56)

    match_horses(race_input, mock_repo, as_of=as_of)

    mock_repo.find_horses_by_name.assert_called_once_with(
        normalize_name("ユイイツウマ"), as_of=as_of)
