"""Repository / ResultRepository のテスト。

最優先は「未来情報が予想側に混ざらないこと」。as_of の必須化・比較の向き
（同時刻は含めない）・DataKubun の優先順位選択を最初にテストする。
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from keiba.data.repository import Repository, ResultRepository
from keiba.data.store import DataStore
from keiba.ingest.store import DataVersionError, version_dir
from keiba.models import EntryRow, HorseMaster
from keiba.names import normalize_name
from tests.fixtures import csv_data as fx

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


def open_repository(tmp_path, races=None, race_horses=None, horses=None, **kw) -> Repository:
    data_dir = fx.write_dataset(tmp_path, races or [], race_horses or [], horses or [], **kw)
    return Repository(DataStore.open_current(data_dir))


def open_result_repository(tmp_path, races=None, race_horses=None, horses=None,
                            **kw) -> ResultRepository:
    data_dir = fx.write_dataset(tmp_path, races or [], race_horses or [], horses or [], **kw)
    return ResultRepository(DataStore.open_current(data_dir))


# --- 署名の検査。ここが崩れると as_of なしで呼べる経路ができてしまう ---------

def test_every_public_method_requires_as_of():
    """Repository の公開メソッドはすべて as_of を必須で取る。

    既定値を与えないのは、うっかり as_of なしで呼べる経路を作らないため。
    予想に未来の情報が混ざっても例外は出ず、評価の数字が静かに良くなるだけなので、
    署名で塞ぐしかない。
    """
    for name, member in inspect.getmembers(Repository, inspect.isfunction):
        if name.startswith("_"):
            continue
        parameter = inspect.signature(member).parameters.get("as_of")
        assert parameter is not None, f"{name} に as_of がない"
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, f"{name} の as_of がキーワード専用でない"
        assert parameter.default is inspect.Parameter.empty, f"{name} の as_of に既定値がある"


def test_result_repository_has_no_as_of():
    """ResultRepository は評価専用。as_of を取れてしまうと Repository との
    使い分け（結果を見てよい場面かどうか）が型で区別できなくなる。"""
    for name, member in inspect.getmembers(ResultRepository, inspect.isfunction):
        if name.startswith("_"):
            continue
        parameter = inspect.signature(member).parameters.get("as_of")
        assert parameter is None, f"ResultRepository.{name} に as_of がある"


# --- past_runs ---------------------------------------------------------------

class TestPastRuns:
    KETTO = "0000000001"

    def _three_runs_dataset(self, tmp_path):
        races = [
            fx.make_race(race_id="2026030106010101", hasso_at="2026-03-01T15:25:00+09:00"),
            fx.make_race(race_id="2026030806010101", hasso_at="2026-03-08T15:25:00+09:00"),
            fx.make_race(race_id="2026031506010101", hasso_at="2026-03-15T15:25:00+09:00"),
        ]
        race_horses = [
            fx.make_race_horse(race_id="2026030106010101", ketto_num=self.KETTO),
            fx.make_race_horse(race_id="2026030806010101", ketto_num=self.KETTO),
            fx.make_race_horse(race_id="2026031506010101", ketto_num=self.KETTO),
        ]
        return open_repository(tmp_path, races, race_horses)

    def test_excludes_races_after_as_of(self, tmp_path):
        repo = self._three_runs_dataset(tmp_path)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 10))
        dates = {run.race.hasso_at.date().isoformat() for run in runs}
        assert dates == {"2026-03-01", "2026-03-08"}

    def test_excludes_race_starting_exactly_at_as_of(self, tmp_path):
        repo = self._three_runs_dataset(tmp_path)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 8, 15, 25, 0))
        dates = {run.race.hasso_at.date().isoformat() for run in runs}
        assert "2026-03-08" not in dates
        assert dates == {"2026-03-01"}

    def test_orders_newest_first(self, tmp_path):
        repo = self._three_runs_dataset(tmp_path)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20))
        hasso_ats = [run.race.hasso_at for run in runs]
        assert hasso_ats == sorted(hasso_ats, reverse=True)
        assert len(runs) == 3

    def test_respects_limit(self, tmp_path):
        repo = self._three_runs_dataset(tmp_path)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20), limit=2)
        assert len(runs) == 2
        assert runs[0].race.hasso_at > runs[1].race.hasso_at

    def test_limit_zero_returns_no_runs(self, tmp_path):
        """limit=0（「0走使う」）は空リスト。limit=None（「全部」）と別物。

        `if limit` で分岐していると 0 が None と同じ扱いになり、
        「0走使う」と言ったのに全走返ってくる。
        """
        repo = self._three_runs_dataset(tmp_path)
        assert repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20), limit=0) == []

    def test_excludes_jump_races(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00", track_kind="turf"),
            fx.make_race(race_id="R2", hasso_at="2026-03-08T15:25:00+09:00", track_kind="jump"),
        ]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO),
            fx.make_race_horse(race_id="R2", ketto_num=self.KETTO),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20))
        assert len(runs) == 1
        assert runs[0].race.race_id == "R1"

    def test_prefers_data_kubun_7(self, tmp_path):
        races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00",
                               data_kubun="7", make_date="20260301")]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="3",
                                make_date="20260301", result_time_sec="999.9"),
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="7",
                                make_date="20260301", result_time_sec="123.4"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20))
        assert len(runs) == 1
        assert runs[0].result.time_sec == 123.4

    def test_falls_back_through_priority(self, tmp_path):
        races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="3",
                                make_date="20260301", result_time_sec="999.9"),
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="5",
                                make_date="20260301", result_time_sec="111.1"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20))
        assert len(runs) == 1
        assert runs[0].result.time_sec == 111.1

    def test_drops_deleted_and_cancelled(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00"),
            fx.make_race(race_id="R2", hasso_at="2026-03-02T15:25:00+09:00", data_kubun="9"),
        ]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="7"),
            fx.make_race_horse(race_id="R2", ketto_num=self.KETTO, data_kubun="9"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20))
        assert len(runs) == 1
        assert runs[0].race.race_id == "R1"

    def test_uses_newest_make_date_on_duplicate(self, tmp_path):
        races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00")]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="7",
                                make_date="20260301", result_time_sec="111.1"),
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="7",
                                make_date="20260302", result_time_sec="222.2"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20))
        assert len(runs) == 1
        assert runs[0].result.time_sec == 222.2

    def test_uses_the_latest_hasso_at_across_revisions(self, tmp_path):
        """発走が予定より早まった版があっても、最大値で判定する。

        race() と同じ規則を通ること。片方だけが単一版を信じていると、
        同じデータで答えが食い違う。
        """
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:10:00+09:00",
                          data_kubun="2", make_date="20260315", nyusen_tosu=""),
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:00:00+09:00",
                          data_kubun="7", make_date="20260320", nyusen_tosu="12"),
        ]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num=self.KETTO, data_kubun="7",
                                make_date="20260320"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        runs = repo.past_runs(self.KETTO, as_of=jst(2026, 3, 20, 15, 5, 0))
        assert runs == []


# --- entries -------------------------------------------------------------

class TestEntries:
    def test_prefers_data_kubun_2_then_1(self, tmp_path):
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", umaban="01", data_kubun="1", futan_kg="55.0"),
            fx.make_race_horse(race_id="R1", umaban="01", data_kubun="2", futan_kg="56.0"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        entries = repo.entries("R1", as_of=jst(2026, 3, 20))
        assert len(entries) == 1
        assert entries[0].futan_kg == 56.0

    def test_uses_data_kubun_1_when_2_absent(self, tmp_path):
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", umaban="01", data_kubun="1", futan_kg="55.0"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        entries = repo.entries("R1", as_of=jst(2026, 3, 20))
        assert len(entries) == 1
        assert entries[0].futan_kg == 55.0

    def test_falls_back_to_result_rows_for_entry_fields_only(self, tmp_path):
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", umaban="01", data_kubun="7",
                                futan_kg="55.0", result_kakutei_jyuni="1"),
        ]
        repo = open_repository(tmp_path, races, race_horses)
        entries = repo.entries("R1", as_of=jst(2026, 3, 20))
        assert len(entries) == 1
        assert isinstance(entries[0], EntryRow)
        assert entries[0].futan_kg == 55.0
        assert not hasattr(entries[0], "kakutei_jyuni")

    def test_returns_entry_row_type(self, tmp_path):
        races = [fx.make_race(race_id="R1")]
        race_horses = [fx.make_race_horse(race_id="R1", umaban="01")]
        repo = open_repository(tmp_path, races, race_horses)
        entries = repo.entries("R1", as_of=jst(2026, 3, 20))
        assert len(entries) == 1
        entry = entries[0]
        assert isinstance(entry, EntryRow)
        for forbidden in ("kakutei_jyuni", "time_sec", "odds", "ninki", "ijyo_cd"):
            assert not hasattr(entry, forbidden)


# --- race()。将来レース／対象レースへ結果側の値を漏らさないことが核心 -------

class TestRaceConditions:
    """レース前に確定する列だけを返す経路（fix5 決定3）。

    実データの races.csv は data_kubun が {"2": 84, "7": 480} で、結果行を
    持つ480レースのうち出馬表行を併せ持つものは0件だった。race() は
    「発走が as_of 以降なら結果行を一切使わない」規律のため、バックテストの
    全レースで None を返し、class_level が取れなかった（27列のうち
    class_level_change と class_level_change_missing が常に無効）。
    class_level は jyoken_cd 由来で発走前に確定しているので、結果行から
    読んでも未来情報にならない。
    """

    def test_race_conditions_reads_class_level_from_a_result_row(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-04-01T15:25:00+09:00",
                          class_level="5", data_kubun="7", make_date="20260401"),
        ]
        repo = open_repository(tmp_path, races, [])

        conditions = repo.race_conditions("R1", as_of=jst(2026, 3, 20))

        assert conditions is not None
        assert conditions.class_level == 5

    def test_race_conditions_returns_none_for_unknown_race_id(self, tmp_path):
        repo = open_repository(tmp_path, [], [])
        assert repo.race_conditions("NOPE", as_of=jst(2026, 3, 20)) is None

    def test_race_conditions_prefers_the_entry_row(self, tmp_path):
        """出馬表行があればそちらを使う（_pick_by_data_kubun の優先順位）。"""
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-04-01T15:25:00+09:00",
                          class_level="2", data_kubun="2", make_date="20260326"),
            fx.make_race(race_id="R1", hasso_at="2026-04-01T15:25:00+09:00",
                          class_level="5", data_kubun="7", make_date="20260401"),
        ]
        repo = open_repository(tmp_path, races, [])

        conditions = repo.race_conditions("R1", as_of=jst(2026, 3, 20))

        assert conditions is not None
        assert conditions.class_level == 2

    def test_race_conditions_returns_exactly_one_column(self):
        """返り値の型のフィールドは、発走前に確定する列だけ（allow-list）。

        禁止名の一覧（deny-list）では足りない——レビュアーが実測で
        ``lap_times_sec``（走ってからしか決まらない）と ``race_name`` を足したら、
        1069件のテストが全部通った。リストに無い結果由来の列は素通りする。
        allow-list なら、どんな列を足してもここで必ず落ちる。

        フィールドを足すときは、その列が発走前に確定することを確かめ、この
        等式も更新すること（更新するときが「本当に発走前に確定するのか」を
        考える場所になる）。

        ``track_cd``（使用するコース）と ``course_kubun_cd``（仮柵の位置
        A/B/C…）は番組と開催前の発表で決まり、発走前に確定する（report3
        §4.2 の『使用コース』）。馬場状態（``baba_cd``）は当日確定なので
        入れない。
        """
        from keiba.models import RaceConditions

        assert {f.name for f in dataclasses.fields(RaceConditions)} == {
            "class_level", "track_cd", "course_kubun_cd"}

    def test_race_conditions_returns_track_and_course_kubun(self, tmp_path):
        races = [dict(fx.make_race(race_id="R1", track_cd="18", data_kubun="7"),
                      course_kubun_cd="B")]
        repo = open_repository(tmp_path, races, [])
        conditions = repo.race_conditions("R1", as_of=jst(2026, 3, 20))
        assert conditions.track_cd == "18"
        assert conditions.course_kubun_cd == "B"


class TestRace:
    def test_returns_none_for_unknown_race_id(self, tmp_path):
        repo = open_repository(tmp_path, [], [])
        assert repo.race("NOPE", as_of=jst(2026, 3, 20)) is None

    def test_uses_best_available_when_race_is_before_as_of(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00",
                          data_kubun="2", make_date="20260226", nyusen_tosu=""),
            fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00",
                          data_kubun="7", make_date="20260301", nyusen_tosu="12"),
        ]
        repo = open_repository(tmp_path, races, [])
        race = repo.race("R1", as_of=jst(2026, 3, 20))
        assert race is not None
        assert race.nyusen_tosu == 12

    def test_hides_result_fields_when_race_is_at_or_after_as_of(self, tmp_path):
        """データ版自体は as_of と無関係に「今取得できる全件」を含む。

        対象レースの発走が as_of 以降なら、結果側 (data_kubun 3-7) の行に
        最終着順などが入っていても、それを使ってはいけない。
        """
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-04-01T15:25:00+09:00",
                          data_kubun="2", make_date="20260326", nyusen_tosu=""),
            fx.make_race(race_id="R1", hasso_at="2026-04-01T15:25:00+09:00",
                          data_kubun="7", make_date="20260401", nyusen_tosu="12"),
        ]
        repo = open_repository(tmp_path, races, [])
        race = repo.race("R1", as_of=jst(2026, 3, 20))
        assert race is not None
        assert race.nyusen_tosu is None

    def test_hides_result_fields_when_hasso_at_equals_as_of(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:25:00+09:00",
                          data_kubun="2", make_date="20260315", nyusen_tosu=""),
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:25:00+09:00",
                          data_kubun="7", make_date="20260320", nyusen_tosu="12"),
        ]
        repo = open_repository(tmp_path, races, [])
        race = repo.race("R1", as_of=jst(2026, 3, 20, 15, 25, 0))
        assert race is not None
        assert race.nyusen_tosu is None

    def test_returns_none_when_only_result_data_exists_for_a_future_race(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-04-01T15:25:00+09:00",
                          data_kubun="7", make_date="20260401", nyusen_tosu="12"),
        ]
        repo = open_repository(tmp_path, races, [])
        assert repo.race("R1", as_of=jst(2026, 3, 20)) is None

    def test_gate_uses_the_latest_known_hasso_at(self, tmp_path):
        """発走時刻は版ごとに書き換わる。古いほう（出馬表の予定時刻）で
        判定すると、実際にはまだ発走していないレースの結果が漏れる。

        出馬表(2)は予定 15:00、成績(7)は実際の発走が遅れて 15:10。
        as_of=15:05 の時点では（実際の発走時刻を基準にすれば）まだ発走して
        いないので、結果側の項目を返してはいけない。
        """
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:00:00+09:00",
                          data_kubun="2", make_date="20260315",
                          nyusen_tosu="", syusso_tosu="", first_3f_sec=""),
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:10:00+09:00",
                          data_kubun="7", make_date="20260320",
                          nyusen_tosu="12", syusso_tosu="12", first_3f_sec="35.2"),
        ]
        repo = open_repository(tmp_path, races, [])
        race = repo.race("R1", as_of=jst(2026, 3, 20, 15, 5, 0))
        assert race is not None
        assert race.nyusen_tosu is None
        assert race.syusso_tosu is None
        assert race.first_3f_sec is None

    def test_gate_opens_once_the_latest_hasso_at_has_passed(self, tmp_path):
        """実際の発走時刻（版のうち最大の hasso_at）を過ぎていれば、
        通常どおり結果側（手に入る中で最も情報の多い行）を返す。"""
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:00:00+09:00",
                          data_kubun="2", make_date="20260315",
                          nyusen_tosu="", syusso_tosu="", first_3f_sec=""),
            fx.make_race(race_id="R1", hasso_at="2026-03-20T15:10:00+09:00",
                          data_kubun="7", make_date="20260320",
                          nyusen_tosu="12", syusso_tosu="12", first_3f_sec="35.2"),
        ]
        repo = open_repository(tmp_path, races, [])
        race = repo.race("R1", as_of=jst(2026, 3, 20, 15, 30, 0))
        assert race is not None
        assert race.nyusen_tosu == 12


# --- find_horses_by_name ------------------------------------------------

class TestFindHorsesByName:
    def test_returns_matching_horses(self, tmp_path):
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ")]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name("テストウマ", as_of=jst(2026, 3, 22))
        assert len(found) == 1
        assert isinstance(found[0], HorseMaster)
        assert found[0].ketto_num == "0000000001"

    def test_no_match_returns_empty(self, tmp_path):
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ")]
        repo = open_repository(tmp_path, horses=horses)
        assert repo.find_horses_by_name("いないウマ", as_of=jst(2026, 3, 22)) == []

    def test_excludes_deleted(self, tmp_path):
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ", data_kubun="0")]
        repo = open_repository(tmp_path, horses=horses)
        assert repo.find_horses_by_name("テストウマ", as_of=jst(2026, 3, 22)) == []


class TestFindHorsesByNameNormalisation:
    """索引側（DataStore.name_index）の正規化が効いていることの確認。

    生の bamei と正規化済みの入力を直接比較していた旧実装では、長音符
    「ー」を含む馬名（JRA の馬名では珍しくない）がすべて一致しなくなる
    不具合があった。合成フィクスチャが長音符を含まない名前だとテストが
    通ってしまうため、ここで実際に長音符を含む名前を使う。
    """

    def test_finds_a_horse_whose_name_contains_a_long_vowel_mark(self, tmp_path):
        """長音符を含む馬名が見つかること。

        正規化した名前を素の bamei と比較していると、JRA の馬名に頻出する
        長音符つきの馬がすべて「未登録」になる。合成フィクスチャが長音符を
        含まない名前だと、テストは通るのに実データで壊れる。
        """
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストーヴル")]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("テストーヴル"),
                                          as_of=jst(2026, 3, 22))
        assert [h.bamei for h in found] == ["テストーヴル"]

    def test_finds_a_horse_by_its_kana_name(self, tmp_path):
        """利用者が半角カナで入力しても拾えること。

        bamei と bamei_kana をわざと別の名前にする。同じ値にすると、
        bamei 側の一致だけで通ってしまい、bamei_kana を実際に索引に使って
        いるかを確認したことにならない（旧実装が生の bamei としか比較して
        いなかったときの見落としがこれだった）。
        """
        horses = [fx.make_horse(ketto_num="0000000001", bamei="ドウメイウマ",
                                 bamei_kana="ﾃｽﾄｳﾏ")]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("ﾃｽﾄｳﾏ"), as_of=jst(2026, 3, 22))
        assert len(found) == 1
        assert found[0].ketto_num == "0000000001"

    def test_returns_every_horse_sharing_a_normalised_name(self, tmp_path):
        """同名馬は全頭返す。1頭に絞るのは呼び出し側の仕事ではなく、絞れないなら止める。"""
        horses = [
            fx.make_horse(ketto_num="0000000001", bamei="テストウマ"),
            fx.make_horse(ketto_num="0000000002", bamei="テストウマ"),
        ]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert len({h.ketto_num for h in found}) == 2

    def test_a_horse_indexed_under_both_columns_appears_once(self, tmp_path):
        """bamei と bamei_kana が同じキーに正規化される馬を二重に返さない。

        既定値の bamei="テストウマ" / bamei_kana="ﾃｽﾄｳﾏ" は NFKC で同じ
        キーに正規化されるため、索引にはこの1頭の ketto_num が bamei 経由
        ・bamei_kana 経由の2回分入る。find_horses_by_name はそれを
        ketto_num 単位でまとめるので、返るのは1件のはず。
        """
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ",
                                 bamei_kana="ﾃｽﾄｳﾏ")]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert len(found) == 1


class TestHorseMasterDataKubun:
    """UM（競走馬マスタ）のデータ区分は RA/SE と別の表。

    RA/SE では 9 は「中止」だが UM では「抹消」＝引退した実在の馬を指し、
    戦績は完全に残っている。RA/SE 用の DATA_KUBUN_DROP をそのまま使うと
    引退馬が全部消え、バックテストで予想する過去レースの出走馬（現在の
    master ではほぼ全頭が抹消済み）の馬名照合がほぼ全滅する。

    **この fixture に race_horses を足さないこと。** UM 単独であることは
    意図した条件で、UM 側の除外集合の取り違えを検出できる唯一の場所だから。
    SE 経路があると、抹消(9) の UM 行と SE 行の両方を持つ馬は取り違えても
    依然として見つかり（失われるのは誰も読んでいない bamei_kana / birth_date
    だけ）、本番では症状が出ない。
    """

    def test_finds_a_deregistered_horse(self, tmp_path):
        """抹消(9)の馬も見つかること。

        UM のデータ区分は RA/SE と別の表で、9 は「抹消」= 引退した実在の馬。
        RA/SE 用の除外集合をそのまま使うと引退馬が全部消え、バックテストで
        予想する過去レースの出走馬はほぼ全頭が該当する。
        """
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ", data_kubun="9")]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert [h.ketto_num for h in found] == ["0000000001"]

    @pytest.mark.parametrize("data_kubun", ["1", "2", "3", "4", "9"])
    def test_keeps_every_meaningful_um_kubun(self, tmp_path, data_kubun):
        """0 以外の UM データ区分はすべて実在の馬を表す。"""
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ",
                                 data_kubun=data_kubun)]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert len(found) == 1

    def test_drops_only_the_deleted_record_kubun(self, tmp_path):
        """0（提供ミスによる削除）だけは落とす。"""
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ", data_kubun="0")]
        repo = open_repository(tmp_path, horses=horses)
        assert repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22)) == []

    def test_deregistration_row_wins_over_an_older_registration_row(self, tmp_path):
        """同じ馬に 1(新規) と 9(抹消) の行があれば、make_date の新しい方を採る。

        抹消が最新の状態なので、そちらの birth_date を返す。9 が
        DATA_KUBUN_DROP_HORSE で落とされていた旧実装では、候補は 1 の行
        しか残らず、常に古い方の値が返っていた。
        """
        horses = [
            fx.make_horse(ketto_num="0000000001", bamei="テストウマ",
                           birth_date="2018-04-01", data_kubun="1", make_date="20220101"),
            fx.make_horse(ketto_num="0000000001", bamei="テストウマ",
                           birth_date="2019-05-02", data_kubun="9", make_date="20260301"),
        ]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert len(found) == 1
        assert found[0].birth_date == date(2019, 5, 2)

    def test_tie_break_keeps_the_first_csv_row_when_make_date_ties(self, tmp_path):
        """同一 make_date の UM 行が複数あるとき、CSV の先頭行が残る。

        同値どうしのどちらの値が正しいかという優先順位を主張するテストでは
        なく、_newest_by_ketto_num の安定ソート（kind="mergesort"）＋
        drop_duplicates(keep="first") という決定性の仕組みを固定するテスト
        （Task 6 の残り1）。make_date が同値の2行を、見分けのつく sex_cd だけ
        変えて与える。
        """
        horses = [
            fx.make_horse(ketto_num="0000000001", bamei="テストウマ",
                           sex_cd="1", make_date="20260301"),
            fx.make_horse(ketto_num="0000000001", bamei="テストウマ",
                           sex_cd="2", make_date="20260301"),
        ]
        repo = open_repository(tmp_path, horses=horses)
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert len(found) == 1
        assert found[0].sex_cd == "1"           # CSV の先頭行 (sex_cd="1") が残る


class TestNameIndexCoversRaceHorses:
    """JVOpen("RACE") は UM を返さないことがあり、UM を運ぶ差分配信（DIFF）も
    取得期間中に更新の無かった馬を含まないため、馬マスタ（horses.csv）は
    空・不完全でありうる。出走記録（race_horses.csv）には、このデータ版の
    取得期間に走った全頭の bamei・ketto_num が入っているので、名前の照合は
    そこからも成立しなければならない。マスタの完全性という運用者の操作次第
    の前提に、予想が成立するかどうかが依存してはいけない。
    """

    def test_finds_a_horse_that_is_only_in_race_horses(self, tmp_path):
        """馬マスタに無くても、出走記録から名前で引けること。

        JVOpen("RACE") は UM を返さないので horses.csv は空になりうる。
        しかも UM を運ぶ DIFF は差分なので、取得期間中に更新が無かった馬は
        含まれない。照合がマスタの完全性に依存すると、運用者がどう更新を
        走らせたかで予想が成立するかが変わってしまう。
        """
        races = [fx.make_race(race_id="R1")]
        race_horses = [fx.make_race_horse(race_id="R1", ketto_num="0000000001",
                                           bamei="テストウマ", sex_cd="1")]
        repo = open_repository(tmp_path, races, race_horses, horses=[])
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert [h.ketto_num for h in found] == ["0000000001"]
        assert found[0].bamei_kana is None      # SE 由来なのでカナは持たない
        assert found[0].birth_date is None

    def test_master_row_wins_when_the_horse_is_in_both(self, tmp_path):
        """両方にある馬は1件だけ返し、カナと生年月日を持つマスタ側を採る。"""
        races = [fx.make_race(race_id="R1")]
        race_horses = [fx.make_race_horse(race_id="R1", ketto_num="0000000001",
                                           bamei="テストウマ")]
        horses = [fx.make_horse(ketto_num="0000000001", bamei="テストウマ",
                                 bamei_kana="ﾃｽﾄｳﾏ", birth_date="2020-01-01")]
        repo = open_repository(tmp_path, races, race_horses, horses)
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert len(found) == 1
        assert found[0].bamei_kana is not None
        assert found[0].birth_date is not None

    def test_deduplicates_a_horse_that_ran_many_times(self, tmp_path):
        """同じ馬の出走行が複数あっても1件に絞り、採るのは make_date が最新の行。

        件数（1件）だけを見ても重複排除はテストできない。find_horses_by_name は
        最後に ketto_num をキーにした dict を作るので、どの行が採られても1件に
        畳まれる。そこで3行に別々の make_date と別々の sex_cd を持たせ、
        **どの行が採られたか**を sex_cd で見る。CSV 順は古い順に並べてあるので、
        _newest_by_ketto_num の sort_values("make_date") が無いと先頭（最古）の
        行が採られ、このアサートが落ちる。

        ここが壊れると症状は「1件多い/少ない」ではなく誤った属性値になる。
        去勢された馬（古い行が sex_cd="1"（牡）、最新が "3"（セン））で古い行を
        採ると、入力「セン5」に対して性別で絞り込めず status が conflict になり、
        AmbiguousMatchError でレース全体の予想が中止になる。

        sex_cd の3値は「どの行が採られたか」を見分けるための印として置いてある
        （実際に起こりうる変化は牡→セン）。
        """
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00"),
            fx.make_race(race_id="R2", hasso_at="2026-03-08T15:25:00+09:00"),
            fx.make_race(race_id="R3", hasso_at="2026-03-15T15:25:00+09:00"),
        ]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num="0000000001", bamei="テストウマ",
                                sex_cd="1", make_date="20260302"),
            fx.make_race_horse(race_id="R2", ketto_num="0000000001", bamei="テストウマ",
                                sex_cd="2", make_date="20260309"),
            fx.make_race_horse(race_id="R3", ketto_num="0000000001", bamei="テストウマ",
                                sex_cd="3", make_date="20260316"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses=[])
        found = repo.find_horses_by_name(normalize_name("テストウマ"), as_of=jst(2026, 3, 22))
        assert len(found) == 1
        assert found[0].sex_cd == "3"           # make_date が最新（20260316）の行

    def test_applies_the_right_drop_set_to_each_file(self, tmp_path):
        """UM の除外集合を SE 行に適用しない（その逆も）。

        UM の 9 は抹消（実在の馬）、SE の 9 はレース中止。同じ数字で意味が違う。

        チュウシウマ の `== []` に意味を持たせるため、SE にしか居て区分が
        正常（7）な馬（セイジョウウマ）を positive control として同じ fixture に
        置く。これが引けることで、`== []` が「除外集合が効いた」ことを示すと
        言える（SE 経路が丸ごと無いだけでも `== []` は成立してしまう）。
        """
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            # SE 側の 9（中止）。この馬は horses.csv に行が無いので、
            # SE 側で落ちれば結果は空になるはず。
            fx.make_race_horse(race_id="R1", umaban="01", ketto_num="0000000002",
                                bamei="チュウシウマ", data_kubun="9"),
            # positive control: SE にしか居ない、区分が正常（7）な馬。
            fx.make_race_horse(race_id="R1", umaban="02", ketto_num="0000000003",
                                bamei="セイジョウウマ", data_kubun="7"),
        ]
        horses = [
            # UM 側の 9（抹消）。除外されてはいけない。
            fx.make_horse(ketto_num="0000000001", bamei="マッショウウマ", data_kubun="9"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses)

        assert repo.find_horses_by_name(normalize_name("チュウシウマ"),
                                         as_of=jst(2026, 3, 22)) == []
        control = repo.find_horses_by_name(normalize_name("セイジョウウマ"),
                                            as_of=jst(2026, 3, 22))
        assert [h.ketto_num for h in control] == ["0000000003"]
        found = repo.find_horses_by_name(normalize_name("マッショウウマ"),
                                          as_of=jst(2026, 3, 22))
        assert [h.ketto_num for h in found] == ["0000000001"]


class TestHorseMasterSexSource:
    """sex_cd は時点依存の値なので、SE 行があればそちらから採る。

    去勢（牡 -> セン）は起こるが、UM 行は最後に更新された時点の値しか持たない。
    実データでは出走馬 5,177 頭のうち 82 頭（1.6%）が UM と SE で食い違い、
    向きはすべて SE=3（セン）/ UM=1（牡）だった（UM 行が古い）。マスタ側を
    採ると出馬表どおり「セン」と入力した利用者と食い違い、性別で絞った結果が
    0 件になってレース全体が中止になる。
    """

    def test_sex_comes_from_the_race_row_when_one_exists(self, tmp_path):
        """UM が "1"（牡）、SE が "3"（セン）なら "3" を採り、出どころは "race"。"""
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num="9900000001",
                                bamei="キョセイウマ", sex_cd="3", make_date="20260316"),
        ]
        horses = [
            fx.make_horse(ketto_num="9900000001", bamei="キョセイウマ",
                           sex_cd="1", make_date="20230626"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses)

        found = repo.find_horses_by_name(normalize_name("キョセイウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].sex_cd == "3"
        assert found[0].sex_cd_source == "race"

    def test_sex_comes_from_the_master_when_no_race_row_exists(self, tmp_path):
        """SE 行が無い馬（この版の取得期間に走っていない馬）はマスタから採る。"""
        horses = [
            fx.make_horse(ketto_num="9900000002", bamei="キュウヨウウマ",
                           sex_cd="2", make_date="20260301"),
        ]
        repo = open_repository(tmp_path, horses=horses)

        found = repo.find_horses_by_name(normalize_name("キュウヨウウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].sex_cd == "2"
        assert found[0].sex_cd_source == "master"

    def test_master_still_supplies_kana_and_birth_date(self, tmp_path):
        """SE から採るのは sex_cd だけ。SE に無い欄は引き続きマスタから採る。"""
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num="9900000003",
                                bamei="リョウホウウマ", sex_cd="3", make_date="20260316"),
        ]
        horses = [
            fx.make_horse(ketto_num="9900000003", bamei="リョウホウウマ",
                           bamei_kana="ﾘﾖｳﾎｳｳﾏ", birth_date="2021-04-05",
                           sex_cd="1", make_date="20260301"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses)

        found = repo.find_horses_by_name(normalize_name("リョウホウウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].bamei_kana == "ﾘﾖｳﾎｳｳﾏ"
        assert found[0].birth_date == date(2021, 4, 5)
        assert found[0].sex_cd == "3"           # sex_cd だけが SE 由来


class TestFindHorsesByNameAsOf:
    """行の選択そのものが as_of を守る。

    索引（DataStore.name_index）は cached_property なので as_of を持たない。
    索引に名前があっても、as_of 以前の行が1つも無ければ候補にならない。
    これが無いと、対象レース自身の結果行（make_date はレース後）が候補に
    選ばれ、そのレース後の性別が「レース時点の性別」として使われてしまう。
    """

    def test_rows_after_as_of_are_not_considered(self, tmp_path):
        """make_date が as_of より後の行しか持たない馬は候補に出ない。"""
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num="9900000004",
                                bamei="ミライウマ", make_date="20260323"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses=[])

        assert repo.find_horses_by_name(normalize_name("ミライウマ"),
                                         as_of=jst(2026, 3, 22)) == []

    def test_a_row_made_on_the_as_of_date_is_included(self, tmp_path):
        """make_date は日付だけで時刻が無いので、as_of と同日の行は含める。

        前日夜の締切に対して、同日に作られる SE 行は出馬表の行である。
        出馬表を取りこぼすより、1日ぶん寛容な側に倒すほうが安全。
        """
        horses = [
            fx.make_horse(ketto_num="9900000005", bamei="ドウジツウマ",
                           make_date="20260322"),
        ]
        repo = open_repository(tmp_path, horses=horses)

        found = repo.find_horses_by_name(normalize_name("ドウジツウマ"),
                                          as_of=jst(2026, 3, 22, 18, 0, 0))

        assert [h.ketto_num for h in found] == ["9900000005"]

    def test_sex_reflects_the_newest_row_at_or_before_as_of(self, tmp_path):
        """as_of より後の行は、たとえ最新でも sex_cd の出どころにしない。

        対象レース自身の結果行（レース後に作られる）を採ってしまうと、
        「そのレース時点の性別」ではなくレース後の性別で照合することになる。
        """
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-15T15:25:00+09:00"),
            fx.make_race(race_id="R2", hasso_at="2026-03-22T15:25:00+09:00"),
        ]
        race_horses = [
            fx.make_race_horse(race_id="R1", ketto_num="9900000006", bamei="ジテンウマ",
                                sex_cd="1", make_date="20260316"),
            fx.make_race_horse(race_id="R2", ketto_num="9900000006", bamei="ジテンウマ",
                                sex_cd="3", make_date="20260323"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses=[])

        found = repo.find_horses_by_name(normalize_name("ジテンウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].sex_cd == "1"


class TestBirthYear:
    """生年は、マスタが無くても出走記録（SE）から計算できる。

    セットアップ取得で得られるマスタ（198,212件）は生年が 1936〜2021 年で、
    del_kubun は抹消 189,526 / 現役 8,686（96% が抹消馬）。実測では出走馬
    5,177 頭のうち 4,494 頭（86.8%）がマスタに無い。同名を年齢で絞る仕組みが
    マスタの birth_date を必要とすると、現役馬（出走馬の大半）では年齢が
    使えない。SE 行は barei を持ち race_id の先頭4桁がレースの年なので、
    馬齢 = レースの年 - 生年（実データ 950/950 で検証済み）から生年を戻せる。
    """

    def test_birth_year_comes_from_the_master_when_available(self, tmp_path):
        """マスタの birth_date があればその年を使う（日まで分かるのはマスタだけ）。

        SE 行の barei からは別の年（2023）が出るようにしてある。どちらを
        使ったかが見分けられるようにするための印で、実データでは一致する。
        """
        races = [fx.make_race(race_id="2026031506010101",
                               hasso_at="2026-03-15T15:25:00+09:00")]
        race_horses = [
            fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000031",
                                bamei="マスタアリウマ", barei="03", make_date="20260316"),
        ]
        horses = [
            fx.make_horse(ketto_num="9900000031", bamei="マスタアリウマ",
                           birth_date="2021-04-05", make_date="20260301"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses)

        found = repo.find_horses_by_name(normalize_name("マスタアリウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].birth_year == 2021
        assert found[0].birth_year == found[0].birth_date.year
        assert found[0].birth_year_source == "master"

    def test_birth_year_is_computed_from_the_race_row_without_a_master(self, tmp_path):
        """マスタに無い馬は SE から計算する: race_id の年 - barei。"""
        races = [fx.make_race(race_id="2026031506010101",
                               hasso_at="2026-03-15T15:25:00+09:00")]
        race_horses = [
            fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000032",
                                bamei="マスタナシウマ", barei="03", make_date="20260316"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses=[])

        found = repo.find_horses_by_name(normalize_name("マスタナシウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].birth_year == 2023          # 2026 - 3
        assert found[0].birth_year_source == "race"

    @pytest.mark.parametrize("barei", ["", "00"])
    def test_birth_year_is_none_when_barei_is_unreadable(self, tmp_path, barei):
        """barei が正の整数として読めなければ計算しない。

        特に barei="00" を黙って「今年生まれ」にしないこと。0 は年齢としては
        ありえない値で、「不明」を意味する。0 を実在の馬齢として扱うと、
        その馬だけ生年が当年になり、年齢の絞り込みで静かに落ちる。
        """
        races = [fx.make_race(race_id="2026031506010101",
                               hasso_at="2026-03-15T15:25:00+09:00")]
        race_horses = [
            fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000033",
                                bamei="バレイフメイウマ", barei=barei,
                                make_date="20260316"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses=[])

        found = repo.find_horses_by_name(normalize_name("バレイフメイウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].birth_year is None
        assert found[0].birth_year_source is None

    def test_birth_year_uses_the_same_row_as_sex(self, tmp_path):
        """生年は sex_cd を採ったのと同じ SE 行から計算する。

        行を選ぶ経路が2つあると、片方だけが as_of や data_kubun の規律から
        外れる。2行に別々の sex_cd と、別々の生年になる barei を持たせて、
        どちらの行から採ったかを両方の欄で見る（実データでは2行の生年は
        一致する。ここで食い違わせているのは行を見分けるための印）。
        """
        races = [
            fx.make_race(race_id="2025091506010101",
                          hasso_at="2025-09-15T15:25:00+09:00"),
            fx.make_race(race_id="2026031506010101",
                          hasso_at="2026-03-15T15:25:00+09:00"),
        ]
        race_horses = [
            fx.make_race_horse(race_id="2025091506010101", ketto_num="9900000034",
                                bamei="ドウイツギョウウマ", sex_cd="1", barei="02",
                                make_date="20250916"),          # 2025 - 2 = 2023
            fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000034",
                                bamei="ドウイツギョウウマ", sex_cd="3", barei="04",
                                make_date="20260316"),          # 2026 - 4 = 2022
        ]
        repo = open_repository(tmp_path, races, race_horses, horses=[])

        found = repo.find_horses_by_name(normalize_name("ドウイツギョウウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].sex_cd == "3"               # make_date が最新の行
        assert found[0].birth_year == 2022          # 同じ行の barei から

    def test_birth_date_is_still_master_only(self, tmp_path):
        """birth_date はマスタのみ。SE から日付をこしらえない。"""
        races = [fx.make_race(race_id="2026031506010101",
                               hasso_at="2026-03-15T15:25:00+09:00")]
        race_horses = [
            fx.make_race_horse(race_id="2026031506010101", ketto_num="9900000035",
                                bamei="ヒヅケナシウマ", barei="03", make_date="20260316"),
        ]
        repo = open_repository(tmp_path, races, race_horses, horses=[])

        found = repo.find_horses_by_name(normalize_name("ヒヅケナシウマ"),
                                          as_of=jst(2026, 3, 22))

        assert len(found) == 1
        assert found[0].birth_date is None          # 日まではマスタしか持たない
        assert found[0].birth_year == 2023          # 年だけは SE から分かる


# --- ResultRepository ------------------------------------------------------

class TestResultRepository:
    def test_race_prefers_data_kubun_priority(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", data_kubun="3", make_date="20260301",
                          nyusen_tosu="10"),
            fx.make_race(race_id="R1", data_kubun="7", make_date="20260301",
                          nyusen_tosu="12"),
        ]
        repo = open_result_repository(tmp_path, races, [])
        race = repo.race("R1")
        assert race is not None
        assert race.nyusen_tosu == 12

    def test_race_returns_none_for_unknown_race_id(self, tmp_path):
        repo = open_result_repository(tmp_path, [], [])
        assert repo.race("NOPE") is None

    def test_results_include_scratched_horses(self, tmp_path):
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", umaban="01", result_ijyo_cd="0"),
            fx.make_race_horse(race_id="R1", umaban="02", result_ijyo_cd="1",
                                result_kakutei_jyuni=""),
        ]
        repo = open_result_repository(tmp_path, races, race_horses)
        results = repo.results("R1")
        assert len(results) == 2
        assert {r.entry.umaban for r in results} == {"01", "02"}

    def test_winners_returns_all_dead_heat_horses(self, tmp_path):
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", umaban="01", result_kakutei_jyuni="1"),
            fx.make_race_horse(race_id="R1", umaban="02", result_kakutei_jyuni="1"),
        ]
        repo = open_result_repository(tmp_path, races, race_horses)
        assert set(repo.winners("R1")) == {"01", "02"}

    def test_winners_excludes_disqualified(self, tmp_path):
        races = [fx.make_race(race_id="R1")]
        race_horses = [
            fx.make_race_horse(race_id="R1", umaban="01", result_ijyo_cd="5",
                                result_kakutei_jyuni="1"),
            fx.make_race_horse(race_id="R1", umaban="02", result_ijyo_cd="0",
                                result_kakutei_jyuni="2"),
        ]
        repo = open_result_repository(tmp_path, races, race_horses)
        assert repo.winners("R1") == []

    def test_races_between_is_inclusive_and_sorted(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00"),
            fx.make_race(race_id="R2", hasso_at="2026-03-08T15:25:00+09:00"),
            fx.make_race(race_id="R3", hasso_at="2026-03-15T15:25:00+09:00"),
        ]
        repo = open_result_repository(tmp_path, races, [])
        found = repo.races_between(jst(2026, 3, 1, 15, 25, 0), jst(2026, 3, 15, 15, 25, 0))
        assert [r.race_id for r in found] == ["R1", "R2", "R3"]

    def test_races_between_excludes_jump(self, tmp_path):
        races = [
            fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00", track_kind="turf"),
            fx.make_race(race_id="R2", hasso_at="2026-03-08T15:25:00+09:00", track_kind="jump"),
        ]
        repo = open_result_repository(tmp_path, races, [])
        found = repo.races_between(jst(2026, 1, 1), jst(2026, 12, 31))
        assert [r.race_id for r in found] == ["R1"]


# --- 型・ID -----------------------------------------------------------------

def test_ids_are_strings_with_leading_zeros(tmp_path):
    races = [fx.make_race(race_id="R1", jyo_cd="06")]
    race_horses = [fx.make_race_horse(race_id="R1", ketto_num="0000000001", umaban="01")]
    repo = open_repository(tmp_path, races, race_horses)

    entries = repo.entries("R1", as_of=jst(2026, 3, 20))
    assert entries[0].ketto_num == "0000000001"

    # 既定の hasso_at (2026-03-21) より後にすることで、race() が結果側の行
    # （既定の data_kubun="7"）を使える状態にする。
    race = repo.race("R1", as_of=jst(2026, 3, 22))
    assert race.jyo_cd == "06"


def test_empty_cells_become_none_not_zero_or_empty_string(tmp_path):
    """CSV の空セルは None にする。0 にも "" にもしない。

    0 は着順・増減など多くの数値項目で「有効な値」として実際にありうるため、
    空文字列を 0 に丸めると「0という結果」と「値が無い」を区別できなくなる。
    """
    races = [fx.make_race(race_id="R1", hasso_at="2026-03-01T15:25:00+09:00",
                           class_level="")]
    race_horses = [
        fx.make_race_horse(race_id="R1", ketto_num="0000000001", umaban="01",
                            futan_before_kg="", result_dochaku_tosu="",
                            result_kakutei_jyuni=""),
    ]
    horses = [fx.make_horse(ketto_num="0000000001", bamei="空欄ウマ", birth_date="")]
    data_dir = fx.write_dataset(tmp_path, races, race_horses, horses)
    store = DataStore.open_current(data_dir)
    repo = Repository(store)

    entries = repo.entries("R1", as_of=jst(2026, 3, 20))
    assert entries[0].futan_before_kg is None

    race = repo.race("R1", as_of=jst(2026, 3, 20))
    assert race.class_level is None

    # as_of は fixture の make_date（20260322）以降にする。find_horses_by_name は
    # make_date が as_of 以前の行しか見ない。
    found = repo.find_horses_by_name("空欄ウマ", as_of=jst(2026, 3, 22))
    assert found[0].birth_date is None

    result_repo = ResultRepository(store)
    results = result_repo.results("R1")
    assert results[0].dochaku_tosu is None
    assert results[0].kakutei_jyuni is None


# --- DataStore / バージョン管理 ----------------------------------------------

def test_open_current_rejects_incomplete_version(tmp_path):
    version_id = "20260322T030000Z"
    path = version_dir(tmp_path, version_id)
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"version_id": version_id, "status": "incomplete"}), encoding="utf-8")
    (tmp_path / "CURRENT").write_text(version_id, encoding="utf-8")

    with pytest.raises(DataVersionError):
        DataStore.open_current(tmp_path)


def test_open_current_raises_when_no_current(tmp_path):
    with pytest.raises(DataVersionError, match="update_data"):
        DataStore.open_current(tmp_path)


def test_data_version_and_fetched_at_exposed(tmp_path):
    data_dir = fx.write_dataset(tmp_path, [], [], [], version_id="20260322T010000Z")
    repo = Repository(DataStore.open_current(data_dir))
    assert repo.data_version == "20260322T010000Z"
    assert repo.data_fetched_at == datetime.fromisoformat(fx.DEFAULT_FETCHED_AT)
