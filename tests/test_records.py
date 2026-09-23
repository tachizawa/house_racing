"""固定長 cp932 レコードの分解。

仕様書のフォーマットシート: TimeDiff は「符号(+または-)+99秒9 符号は1着:-、
2着以下:+」。1着馬の欄には「2着馬とのタイム差」が入るので margin は 0.0。
"""

import pytest

from keiba.ingest import records
from tests.fixtures import records as fx


def test_record_spec_reads_first_two_bytes():
    assert records.record_spec(fx.make_ra()) == "RA"
    assert records.record_spec(fx.make_se()) == "SE"
    assert records.record_spec(fx.make_um()) == "UM"


@pytest.mark.parametrize(
    ("time_diff", "expected"),
    [("-003", 0.0), ("-012", 0.0),        # 自身が1着。値は2着馬との差なので捨てる
     ("+003", 0.3), ("+000", 0.0), ("+125", 12.5),
     ("9999", None), ("    ", None), ("", None), ("+abc", None), ("?003", None)],
)
def test_margin_sec(time_diff, expected):
    assert records.margin_sec(time_diff) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1234", 83.4), ("0598", 59.8), ("2015", 121.5),
     ("0000", None), ("    ", None), ("9999", None)],
)
def test_race_time_sec(value, expected):
    assert records.race_time_sec(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("341", 34.1), ("352", 35.2), ("999", None), ("000", None), ("   ", None)],
)
def test_tenth_sec(value, expected):
    assert records.tenth_sec(value) == expected


class TestParseRa:
    def test_identity_and_conditions(self):
        row = records.parse_ra(fx.make_ra())
        assert row["race_id"] == "2026032106010511"
        assert row["jyo_cd"] == "06"
        assert row["race_num"] == "11"
        assert row["data_kubun"] == "7"
        assert row["race_name"] == "テスト特別"
        assert row["kyori_m"] == 2000
        assert row["track_kind"] == "turf"
        assert row["jyoken_cd_youngest"] == "016"
        assert row["class_level"] == 4
        assert row["toroku_tosu"] == 12
        assert row["hasso_at"] == "2026-03-21T15:25:00+09:00"

    def test_going_follows_track_kind(self):
        turf = records.parse_ra(fx.make_ra(track_cd="17", siba_baba="3", dirt_baba="0"))
        dirt = records.parse_ra(fx.make_ra(track_cd="23", siba_baba="0", dirt_baba="4"))
        assert turf["baba_cd"] == "3"
        assert dirt["track_kind"] == "dirt" and dirt["baba_cd"] == "4"

    def test_first_3f_comes_from_haron_s3_not_lap_sum(self):
        # ラップは 12.3 × 10 = 36.9 になるが、前3ハロン欄の 35.2 を使うこと
        assert records.parse_ra(fx.make_ra(first_3f="352"))["first_3f_sec"] == 35.2

    def test_grade_overrides_condition_code(self):
        row = records.parse_ra(fx.make_ra(grade_cd="A", jyoken_youngest="999"))
        assert row["class_level"] == 8


class TestParseSe:
    def test_entry_fields(self):
        row = records.parse_se(fx.make_se())
        assert row["race_id"] == "2026032106010511"
        assert row["umaban"] == "01"
        assert row["ketto_num"] == "0000000001"
        assert row["bamei"] == "テストウマ"
        assert row["tozai_cd"] == "1"
        assert row["futan_kg"] == 57.0
        assert row["kisyu_code"] == "00001"

    def test_change_fields_none_when_unset(self):
        row = records.parse_se(fx.make_se())
        assert row["futan_before_kg"] is None
        assert row["kisyu_code_before"] == ""

    def test_change_fields_kept_when_set(self):
        row = records.parse_se(fx.make_se(futan_before="565", kisyu_code_before="00002"))
        assert row["futan_before_kg"] == 56.5
        assert row["kisyu_code_before"] == "00002"

    def test_zero_filled_kisyu_code_before_is_blank(self):
        """騎手変更なしのゼロ埋め "00000" は空文字列にそろえる。

        JV-Data は「変更なし」を空白ではなくゼロ埋めで入れてくる（実データの
        race_horses.csv では 7,455 行中 7,389 行が "00000"）。生の "00000" を
        そのまま通すと EntryRow.changed_after_entry が bool("00000") = True を
        読んで、ほぼ全頭に「登録後に変更あり」の偽の注記が付く。
        """
        row = records.parse_se(fx.make_se(kisyu_code_before="00000"))
        assert row["kisyu_code_before"] == ""

    def test_result_fields_are_prefixed(self):
        row = records.parse_se(fx.make_se())
        assert row["result_kakutei_jyuni"] == 1
        assert row["result_time_sec"] == 83.4
        assert row["result_margin_sec"] == 0.0        # 1着
        assert row["result_jyuni_4c"] == 2
        assert row["result_last_3f_sec"] == 34.1
        assert row["result_odds"] == 3.4
        assert row["result_ijyo_cd"] == "0"

    def test_weight_change_uses_sign_byte(self):
        minus = records.parse_se(fx.make_se(zogen_fugo="-", zogen_sa="014"))
        plus = records.parse_se(fx.make_se(zogen_fugo="+", zogen_sa="006"))
        assert minus["result_zogen_sa_kg"] == -14
        assert plus["result_zogen_sa_kg"] == 6

    @pytest.mark.parametrize("zogen_sa", ["999", "   "])
    def test_weight_change_none_when_unmeasurable(self, zogen_sa):
        assert records.parse_se(fx.make_se(zogen_sa=zogen_sa))["result_zogen_sa_kg"] is None

    def test_weight_change_zero_kept(self):
        row = records.parse_se(fx.make_se(zogen_fugo=" ", zogen_sa="000"))
        assert row["result_zogen_sa_kg"] == 0

    def test_scratched_horse_has_no_result_values(self):
        """出走取消（ijyo_cd=1）の馬は time と time_diff が無効値に設定される。

        time_sec、margin_sec、kakutei_jyuni はすべて None になる（初期値）。
        ResultRow.counts_as_starter / can_win は ijyo_cd で判定する。
        """
        row = records.parse_se(
            fx.make_se(ijyo_cd="1", kakutei_jyuni="00", time="0000", time_diff="9999"))
        assert row["result_ijyo_cd"] == "1"
        assert row["result_kakutei_jyuni"] is None  # 初期値 "00" → None
        assert row["result_time_sec"] is None       # 初期値 "0000" → None
        assert row["result_margin_sec"] is None     # 初期値 "9999" → None

    @pytest.mark.parametrize("ijyo_cd", ["6", "7"])
    def test_keeps_result_values_for_demoted_and_remounted(self, ijyo_cd):
        """6 落馬再騎乗 と 7 降着 は完走していて確定着順がある。捨てない。"""
        row = records.parse_se(
            fx.make_se(ijyo_cd=ijyo_cd, kakutei_jyuni="03", time="1234",
                       time_diff="+003", odds="0125", ninki="04"))
        assert row["result_ijyo_cd"] == ijyo_cd
        assert row["result_kakutei_jyuni"] == 3
        assert row["result_time_sec"] == 83.4
        assert row["result_margin_sec"] == 0.3
        assert row["result_odds"] == 12.5

    @pytest.mark.parametrize(
        "key",
        ["result_kakutei_jyuni", "result_ninki",
         "result_jyuni_1c", "result_jyuni_2c", "result_jyuni_3c", "result_jyuni_4c"],
    )
    def test_unset_zero_becomes_none(self, key):
        """0 が値として成立しない項目は、初期値 "00" を 0 にせず None にする。"""
        row = records.parse_se(
            fx.make_se(kakutei_jyuni="00", ninki="00",
                       jyuni_1c="00", jyuni_2c="00", jyuni_3c="00", jyuni_4c="00"))
        assert row[key] is None

    def test_dead_heat_count_zero_is_kept(self):
        """同着頭数の 0 は「同着なし」という実際の値なので残す。"""
        row = records.parse_se(fx.make_se(dochaku_tosu="0"))
        assert row["result_dochaku_tosu"] == 0


class TestParseUm:
    def test_time_independent_fields(self):
        row = records.parse_um(fx.make_um())
        assert row["ketto_num"] == "0000000001"
        assert row["bamei"] == "テストウマ"
        assert row["birth_date"] == "2022-03-15"
        assert row["sex_cd"] == "1"

    def test_no_cumulative_aggregates(self):
        """UM の累計成績は過去時点を復元できないので dict に載せない。"""
        row = records.parse_um(fx.make_um())
        for forbidden in ("ruikei", "chaku_kaisu", "kyakusitu", "race_count",
                          "chaku_sogo", "chaku_chuo"):
            assert not any(forbidden in key for key in row), forbidden


def test_parse_dispatches_and_ignores_other_specs():
    spec, row = records.parse(fx.make_se())
    assert spec == "SE" and row["umaban"] == "01"
    raw = bytearray(fx.make_se())
    raw[0:2] = b"O1"
    assert records.parse(bytes(raw)) is None
