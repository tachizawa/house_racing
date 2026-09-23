"""EntryRow と ResultRow の分離が未来情報の混入を防ぐことを確認する。"""

from datetime import date, datetime

import pytest

from keiba.models import (
    EntryRow,
    HorseEntry,
    PastRun,
    RaceInput,
    RaceRow,
    ResultRow,
)


def make_entry(umaban: str = "1", **kw) -> EntryRow:
    base = dict(
        umaban=umaban,
        wakuban="1",
        ketto_num="0000000001",
        bamei="テスト馬",
        sex_cd="1",
        barei="4",
        futan_kg=57.0,
        kisyu_code="00001",
        tozai_cd="1",
        chokyosi_code="00001",
    )
    base.update(kw)
    return EntryRow(**base)


def make_race(**kw) -> RaceRow:
    base = dict(
        race_id="2026032106050311",
        hasso_at=datetime(2026, 3, 21, 15, 25),
        jyo_cd="06",
        kyori_m=2000,
        track_cd="17",
        track_kind="turf",
        baba_cd="1",
        class_level=4,
        grade_cd=" ",
        jyoken_cd_youngest="016",
        toroku_tosu=12,
        syusso_tosu=12,
        nyusen_tosu=12,
        first_3f_sec=35.2,
        race_name="テスト賞",
    )
    base.update(kw)
    return RaceRow(**base)


def make_result(**kw) -> ResultRow:
    base = dict(
        entry=make_entry(),
        ijyo_cd="0",
        kakutei_jyuni=1,
        dochaku_kubun="0",
        dochaku_tosu=None,
        time_sec=120.5,
        margin_sec=0.0,
        jyuni_1c=3,
        jyuni_2c=3,
        jyuni_3c=2,
        jyuni_4c=2,
        last_3f_sec=34.1,
        ba_taijyu_kg=486,
        zogen_sa_kg=-2,
        odds=3.4,
        ninki=1,
    )
    base.update(kw)
    return ResultRow(**base)


class TestEntryRow:
    def test_has_no_result_fields(self):
        """出馬表相当の型に結果の項目が生えていないこと。"""
        entry = make_entry()
        for forbidden in (
            "kakutei_jyuni", "time_sec", "margin_sec", "odds", "ninki",
            "jyuni_1c", "jyuni_2c", "jyuni_3c", "jyuni_4c", "last_3f_sec",
            "ba_taijyu_kg", "zogen_sa_kg", "ijyo_cd", "dochaku_kubun", "dochaku_tosu",
        ):
            assert not hasattr(entry, forbidden), forbidden

    def test_is_frozen(self):
        entry = make_entry()
        with pytest.raises(Exception):
            entry.umaban = "2"


class TestChangedAfterEntry:
    """``kisyu_code_before`` の「変更なし」表現を「変更あり」と読まないこと。

    JV-Data は変更なしを空白ではなくゼロ埋め（``"00000"``）で入れてくる。
    ingest 側でも正規化するが、**すでに保存済みのデータ版には生の
    ``"00000"`` が入っている**ので、読む側のここでも「変更なし」と読む。
    """

    def test_blank_is_not_a_change(self):
        assert make_entry().changed_after_entry is False

    def test_zero_filled_kisyu_code_is_not_a_change(self):
        assert make_entry(kisyu_code_before="00000").changed_after_entry is False

    def test_real_kisyu_code_is_a_change(self):
        assert make_entry(kisyu_code_before="01184").changed_after_entry is True

    def test_futan_before_is_a_change(self):
        assert make_entry(futan_before_kg=56.0).changed_after_entry is True


class TestResultRow:
    @pytest.mark.parametrize(
        ("ijyo_cd", "starter", "can_win"),
        [
            ("0", True, True),    # 正常
            ("6", True, True),    # 落馬再騎乗
            ("7", True, True),    # 降着。確定着順がある
            ("1", False, False),  # 出走取消
            ("2", False, False),  # 発走除外
            ("3", False, False),  # 競走除外
            ("4", True, False),   # 競走中止。集合に残すが勝ち馬になり得ない
            ("5", True, False),   # 失格
        ],
    )
    def test_starter_and_winner_eligibility(self, ijyo_cd, starter, can_win):
        result = make_result(ijyo_cd=ijyo_cd)
        assert result.counts_as_starter is starter
        assert result.can_win is can_win


class TestRaceRow:
    def test_is_flat_excludes_jump(self):
        assert make_race(track_kind="turf").is_flat is True
        assert make_race(track_kind="dirt").is_flat is True
        assert make_race(track_kind="jump").is_flat is False
        assert make_race(track_kind=None).is_flat is False


class TestPastRun:
    def test_margin_sec_is_none_when_no_time(self):
        run = PastRun(race=make_race(), result=make_result(ijyo_cd="4", margin_sec=None))
        assert run.margin_sec is None

    def test_margin_sec_passes_through_for_normal_run(self):
        run = PastRun(race=make_race(), result=make_result(margin_sec=0.3))
        assert run.margin_sec == 0.3


class TestRaceInput:
    def test_defaults(self):
        race_input = RaceInput(
            date=date(2026, 3, 21),
            jyo="中山",
            race_num=11,
            track_kind="turf",
            kyori_m=2000,
            entry_count=2,
            prediction_cutoff=datetime(2026, 3, 20, 18, 0),
            horses=(HorseEntry(umaban="1", bamei="テスト馬"),
                    HorseEntry(umaban="2", bamei="サンプル馬")),
        )
        assert race_input.top_n == 3
        assert race_input.baba is None
        assert race_input.class_name is None


class TestSexAge:
    """fix5 決定8: barei が読めないとき「0歳」を捏造しない。"""

    def test_reads_a_zero_padded_barei(self):
        assert make_entry(sex_cd="1", barei="04").sex_age == "牡4"

    def test_keeps_two_digit_ages(self):
        assert make_entry(sex_cd="2", barei="10").sex_age == "牝10"

    @pytest.mark.parametrize("barei", ["00", "0", "", "  ", "??"])
    def test_returns_only_the_sex_when_barei_is_unreadable(self, barei):
        """barei が正の整数として読めなければ性別だけ（「0歳」を作らない）。

        0 は年齢としてありえない値で「不明」を意味する
        （repository._birth_year_from_race_row も years <= 0 を意図的に
        不明にしている）。「牡0」を作ると、その値が申告として照合に渡り、
        生年 = 当年 - 0 で絞って0件 → その馬の過去走がすべて失われ、しかも
        注記が「申告の馬齢（0歳）」と入力について虚偽を述べる。
        """
        assert make_entry(sex_cd="1", barei=barei).sex_age == "牡"

    def test_unknown_sex_code_still_has_no_fabricated_age(self):
        assert make_entry(sex_cd="9", barei="00").sex_age == "?"
