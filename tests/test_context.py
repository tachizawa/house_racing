"""keiba.features.context のテスト。

PredictionContext は Repository の呼び出しと ability.py / derive.py の
計算結果を束ねるだけの層。ここでは「束ね方」（as_of の受け渡し・未登録馬の
扱い・field_rank の横断計算・missing/notes への記録）だけを確かめる。
値そのものの計算（着差の加重平均など）は test_ability.py / test_derive.py が
すでに検証済みなので、ここでは再検証しない。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest import mock

import pytest

from keiba.data.repository import Repository
from keiba.data.store import DataStore
from keiba.features.ability import field_ranks
from keiba.features.context import build_context
from keiba.features.derive import BABA_UNKNOWN_REASON
from keiba.models import HorseEntry, HorseMaster, RaceInput
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


# --- as_of の受け渡し --------------------------------------------------------

def test_uses_prediction_cutoff_as_as_of():
    """repo.past_runs が as_of=race_input.prediction_cutoff で呼ばれる。

    build_context には as_of を渡すパラメータが無い。race_input.prediction_cutoff
    以外の時刻を使う経路が無いことを、モックで直接確かめる。
    """
    mock_repo = mock.create_autospec(Repository, instance=True)
    mock_repo.find_horses_by_name.return_value = [
        HorseMaster(ketto_num="0000000001", bamei="タイショウウマ",
                    bamei_kana="ﾀｲｼｮｳｳﾏ", birth_date=None, sex_cd="1",
                    sex_cd_source="master", birth_year=2022,
                    birth_year_source="race"),
    ]
    mock_repo.past_runs.return_value = []
    entry = make_entry(bamei="タイショウウマ")
    race_input = make_race_input([entry])

    build_context(race_input, mock_repo)

    mock_repo.past_runs.assert_called_once_with(
        "0000000001", as_of=race_input.prediction_cutoff)


# --- 未登録馬 -----------------------------------------------------------------

def test_unregistered_horse_has_empty_features(tmp_path):
    """ketto_num が無い馬は、過去走を引けないので能力指標が空になる。"""
    repo = open_repository(tmp_path, horses=[])
    entry = make_entry(bamei="ミトウロクウマ")
    race_input = make_race_input([entry])

    ctx = build_context(race_input, repo)

    assert len(ctx.horses) == 1
    features = ctx.horses[0]
    assert features.ketto_num is None
    assert features.ability.runs_used == 0
    assert features.ability.recent_margin_sec_wavg is None
    assert features.missing != ()


def test_unregistered_horse_notes_reason(tmp_path):
    """未登録の理由が notes に残る。data_quality がデータについて虚偽を語らない
    ため（仕様14.3.1節）。"""
    repo = open_repository(tmp_path, horses=[])
    entry = make_entry(bamei="ミトウロクウマ")
    race_input = make_race_input([entry])

    ctx = build_context(race_input, repo)

    joined = "".join(ctx.horses[0].notes)
    # 索引は馬マスタ（UM）と出走記録（SE）の両方から作るので、「マスタ未登録」
    # では実態を表さない。どちらにも無かったことが理由。
    assert "馬マスタにも取得済みの出走記録にも" in joined
    assert "過去実績なし" in joined


def test_match_notes_reach_the_horse(tmp_path):
    """MatchResult.notes が馬単位の notes に取り込まれる。

    照合側は「申告の馬齢に合う馬が居なかった」「性別が食い違った」といった
    具体的な理由を notes に入れる。取り込まないと、照合が静かに unregistered を
    返す経路で理由が利用者にも state にも届かない（打ち間違いが黙って
    「過去実績なし」として予想に乗る）。
    """
    mock_repo = mock.create_autospec(Repository, instance=True)
    mock_repo.find_horses_by_name.return_value = []
    mock_repo.past_runs.return_value = []
    mock_repo.entries.return_value = []
    mock_repo.race.return_value = None
    mock_repo.data_version = "vtest"
    mock_repo.data_fetched_at = jst(2026, 3, 20)

    entry = make_entry(bamei="ミトウロクウマ")
    race_input = make_race_input([entry])

    with mock.patch("keiba.features.context.match_horses") as patched:
        patched.return_value = [
            mock.Mock(umaban="01", entry=entry, ketto_num=None,
                      status="unregistered", candidates=(),
                      notes=("01番は申告の馬齢（牡3）に合う馬が見つからなかった。",)),
        ]
        ctx = build_context(race_input, mock_repo)

    joined = "".join(ctx.horses[0].notes)
    assert "申告の馬齢（牡3）に合う馬が見つからなかった" in joined
    # 具体的な理由があるときは一般的な文を重ねない。
    assert "馬マスタにも取得済みの出走記録にも" not in joined


def test_missing_lists_paths_not_values(tmp_path):
    """missing は値ではなく「どの特徴量群が計算できなかったか」のパス名。

    未登録馬は過去走に依存する4群に加え、所属コードを解決する手がかりが
    何も無い（entries() の行も過去走も無い）ので transport も足される
    （ability.transport_base_source が None になるケース。決定3）。
    """
    repo = open_repository(tmp_path, horses=[])
    entry = make_entry(bamei="ミトウロクウマ")
    race_input = make_race_input([entry])

    ctx = build_context(race_input, repo)

    assert set(ctx.horses[0].missing) == {
        "ability", "by_condition", "recent", "running_style", "transport",
    }
    for item in ctx.horses[0].missing:
        assert isinstance(item, str)


def test_missing_includes_history_groups_when_no_past_runs(tmp_path):
    """照合できても、**引けた過去走が0本なら**過去走依存の4群は計算不能。

    照合できたかどうか（ketto_num の有無）で立てると、「馬マスタには居るが
    取得済みのデータ版に先行出走が無い馬」が欠損を申告しないまま
    ``ability`` は空・``by_condition`` は 0戦 として state に載る。
    Jev から見ると「計算できなかった」ではなく「0戦0勝の馬」に読める。
    取得窓を遡ったあとも欠損の申告が正しく保たれるように、**実際に引けた
    走の数**で立てる（report2 §6 の #6）。
    """
    horses = [fx.make_horse(ketto_num="0000000001", bamei="ハツソウウマ")]
    repo = open_repository(tmp_path, [], [], horses)
    entry = make_entry(umaban="01", bamei="ハツソウウマ")
    race_input = make_race_input([entry])

    ctx = build_context(race_input, repo)

    assert ctx.horses[0].ketto_num == "0000000001"
    assert set(ctx.horses[0].missing) >= {
        "ability", "by_condition", "recent", "running_style"}


def test_missing_omits_history_groups_when_a_past_run_exists(tmp_path):
    """過去走が1本でも引ければ4群は計算できているので missing に入らない。"""
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(race_id="R1", umaban="01", ketto_num="0000000001",
                                       bamei="ジッセキウマ", tozai_cd="1")]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="ジッセキウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(umaban="01", bamei="ジッセキウマ")
    race_input = make_race_input([entry])

    ctx = build_context(race_input, repo)

    assert ctx.horses[0].ability.runs_used > 0
    assert ctx.horses[0].missing == ()


def test_missing_omits_transport_when_base_known(tmp_path):
    """所属が解決できれば transport は missing に入らない（stay_status が
    既定の "unknown" であること自体は理由にしない——決定3）。"""
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(race_id="R1", umaban="01", ketto_num="0000000001",
                                       bamei="トウロクウマ", tozai_cd="1")]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(umaban="01", bamei="トウロクウマ")
    race_input = make_race_input([entry])

    ctx = build_context(race_input, repo)

    assert "transport" not in ctx.horses[0].missing
    assert ctx.horses[0].transport.stay_status == "unknown"


def test_missing_includes_transport_when_base_unresolvable(tmp_path):
    """履歴のある馬でも、所属コードが 美浦/栗東 のどちらでもない（地方・外国
    招待など）なら base が None になり、transport が missing に入る。"""
    races = [fx.make_race(race_id="R_TARGET", hasso_at="2026-03-25T15:25:00+09:00",
                           data_kubun="2")]
    race_horses = [fx.make_race_horse(
        race_id="R_TARGET", umaban="01", ketto_num="0000000001", bamei="チホウウマ",
        tozai_cd="3", data_kubun="2",
        result_ijyo_cd="", result_kakutei_jyuni="", result_time_sec="",
        result_margin_sec="", result_odds="", result_ninki="",
    )]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="チホウウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(umaban="01", bamei="チホウウマ")
    race_input = make_race_input([entry], race_id="R_TARGET")

    ctx = build_context(race_input, repo)

    assert ctx.horses[0].transport.base is None
    assert "transport" in ctx.horses[0].missing
    # この馬の行は対象レース（締切より後）の1件だけなので、引けた過去走は0本。
    # transport は**それとは独立に** base が解決できなかったから載る、という
    # 2つの経路の独立性がここの眼目。以前この行は「ketto_num が分かっている
    # ので4群は計算できている」として missing == {"transport"} を主張して
    # いたが、それは照合の成否と材料の有無を取り違えていた（report2 §6 の #6）。
    assert set(ctx.horses[0].missing) == {
        "ability", "by_condition", "recent", "running_style", "transport"}


# --- field_rank ---------------------------------------------------------------

def test_field_rank_computed_across_all_horses(tmp_path):
    """全馬の recent_margin_sec_wavg から順位が付く（ability.field_ranks と同じ
    値になることを直接照合する）。"""
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [
        fx.make_race_horse(race_id="R1", umaban="01", ketto_num="0000000001",
                            bamei="イチバンウマ", result_margin_sec="0.0"),
        fx.make_race_horse(race_id="R1", umaban="02", ketto_num="0000000002",
                            bamei="ニバンウマ", result_margin_sec="1.5"),
    ]
    horses = [
        fx.make_horse(ketto_num="0000000001", bamei="イチバンウマ"),
        fx.make_horse(ketto_num="0000000002", bamei="ニバンウマ"),
    ]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entries = [make_entry(umaban="01", bamei="イチバンウマ"),
               make_entry(umaban="02", bamei="ニバンウマ")]
    race_input = make_race_input(entries)

    ctx = build_context(race_input, repo)

    by_umaban = {h.umaban: h for h in ctx.horses}
    expected = field_ranks({
        "01": by_umaban["01"].ability.recent_margin_sec_wavg,
        "02": by_umaban["02"].ability.recent_margin_sec_wavg,
    })
    assert by_umaban["01"].field_rank == expected["01"]
    assert by_umaban["02"].field_rank == expected["02"]
    # 着差0.0（1着相当）のほうが1.5より良い順位（1.0）のはず。
    assert by_umaban["01"].field_rank == pytest.approx(1.0)
    assert by_umaban["02"].field_rank == pytest.approx(2.0)


# --- 登録後の変更 ---------------------------------------------------------------

def test_changed_entry_is_noted(tmp_path):
    """EntryRow.changed_after_entry が真の馬は notes に記録される。

    futan_before_kg が設定されているのは「前日時点から斤量が変わった」印
    （models.py の EntryRow.changed_after_entry）。対象レースの EntryRow は
    Repository.entries() からしか読めない（利用者入力の HorseEntry には
    この情報が無い）。
    """
    races = [fx.make_race(race_id="R_TARGET", hasso_at="2026-03-25T15:25:00+09:00",
                           data_kubun="2")]
    race_horses = [fx.make_race_horse(
        race_id="R_TARGET", umaban="01", ketto_num="0000000001", bamei="ヘンコウウマ",
        futan_before_kg="55.0", futan_kg="57.0", data_kubun="2",
        result_ijyo_cd="", result_kakutei_jyuni="", result_time_sec="",
        result_margin_sec="", result_odds="", result_ninki="",
    )]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="ヘンコウウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)

    entry = make_entry(umaban="01", bamei="ヘンコウウマ")
    race_input = make_race_input([entry], race_id="R_TARGET")

    ctx = build_context(race_input, repo)

    joined = "".join(ctx.horses[0].notes)
    assert "変更" in joined


def test_marks_affiliation_taken_from_a_past_run(tmp_path):
    """今日の出馬表が内部データに無いとき、所属は直近走から借りたと明示する。

    黙って借りると、state は「栗東所属」と断定したことになる。推測と
    確認済みの情報を区別するのが spec §4.2 の要点（決定2への修正: base_source）。
    """
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
    race_horses = [fx.make_race_horse(race_id="R1", umaban="01", ketto_num="0000000001",
                                       bamei="トウロクウマ", tozai_cd="2")]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="トウロクウマ")]
    repo = open_repository(tmp_path, races, race_horses, horses)
    entry = make_entry(umaban="01", bamei="トウロクウマ")
    # race_id を指定しない = 対象レースの公式な出馬表（entries()）が無い状態。
    race_input = make_race_input([entry])

    ctx = build_context(race_input, repo)
    features = ctx.horses[0]

    assert features.transport_base_source == "last_run"
    assert features.transport.base == "栗東"
    joined = "".join(features.notes)
    assert "推定" in joined
    assert "01" in joined


def test_marks_affiliation_from_todays_entry_when_available(tmp_path):
    """対象レースの出馬表（entries()）が既にあるなら、そこから所属を取り、
    「推定」ではなく確認済み（"entry"）として扱う。"""
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
    entry = make_entry(umaban="01", bamei="トウロクウマ")
    race_input = make_race_input([entry], race_id="R_TARGET")

    ctx = build_context(race_input, repo)
    features = ctx.horses[0]

    assert features.transport_base_source == "entry"
    assert features.transport.base == "栗東"
    assert "推定" not in "".join(features.notes)


# --- レース単位の注記（fix2: 馬場状態未入力） -------------------------------------

def test_race_note_when_going_is_unknown(tmp_path):
    """baba を入力しなかったとき、ctx.notes にちょうど1件、
    ``derive.BABA_UNKNOWN_REASON`` を部分文字列として含む注記が入る
    （notes は現在このケース以外では常に空タプル）。

    文言を丸写しでハードコードしない——理由の文字列は derive.py の
    BABA_UNKNOWN_REASON 1箇所だけから組む、という決定（fix3 決定2）が
    守られているかをここで確かめる。文言そのものをハードコードすると、
    context.py 側が独自の文字列を書いていても気づけない。
    """
    repo = open_repository(tmp_path, horses=[])
    entry = make_entry(bamei="ミトウロクウマ")
    race_input = make_race_input([entry], baba_cd=None)

    ctx = build_context(race_input, repo)

    assert len(ctx.notes) == 1
    assert BABA_UNKNOWN_REASON in ctx.notes[0]


def test_no_race_note_when_going_is_given(tmp_path):
    """baba を与えたら ctx.notes は空タプルのまま。"""
    repo = open_repository(tmp_path, horses=[])
    entry = make_entry(bamei="ミトウロクウマ")
    race_input = make_race_input([entry], baba_cd="1")

    ctx = build_context(race_input, repo)

    assert ctx.notes == ()
