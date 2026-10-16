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


class TestParseSk:
    def test_identity_fields(self):
        row = records.parse_sk(fx.make_sk())
        assert row["ketto_num"] == "0000000001"
        assert row["birth_date"] == "2022-03-15"
        assert row["sex_cd"] == "1"
        assert row["sanchi_name"] == "日高町"
        assert row["breeder_code"] == "00001234"
        assert row["data_kubun"] == "1"

    def test_ancestors_follow_the_three_generation_order(self):
        """HansyokuNum[0..13] は 父・母・父父・父母・母父・母母・父父父…の順。"""
        row = records.parse_sk(fx.make_sk())
        assert [row[f"{name}_num"] for name in records.SK_ANCESTOR_NAMES] == fx.SK_ANCESTORS
        assert row["sire_num"] == "0000000001"
        assert row["dam_num"] == "0000000002"
        assert row["dam_sire_num"] == "0000000005"     # 母父
        assert row["dam_dam_dam_num"] == "0000000014"

    def test_zero_filled_ancestor_is_blank(self):
        """不明な祖先はゼロ埋めで来る。値があるように見せない。"""
        ancestors = ["0000000000"] + fx.SK_ANCESTORS[1:]
        row = records.parse_sk(fx.make_sk(ancestors=ancestors))
        assert row["sire_num"] == ""


class TestParseHn:
    def test_fields(self):
        row = records.parse_hn(fx.make_hn())
        assert row["hansyoku_num"] == "0000000001"
        assert row["bamei"] == "テストシュボバ"
        assert row["bamei_eng"] == "Test Sire"
        assert row["birth_year"] == "2005"
        assert row["sex_cd"] == "1"
        assert row["sire_num"] == "0000000003"
        assert row["dam_num"] == "0000000004"
        assert row["sanchi_name"] == "日高町"

    def test_zero_filled_codes_are_blank(self):
        row = records.parse_hn(fx.make_hn(ketto_num="0000000000", sire_num="0000000000",
                                          import_year="0000"))
        assert row["ketto_num"] == ""
        assert row["sire_num"] == ""
        assert row["import_year"] == ""


class TestParseBt:
    def test_fields_without_the_long_description(self):
        row = records.parse_bt(fx.make_bt())
        assert row["hansyoku_num"] == "0000000001"
        assert row["keito_id"] == "0101"
        assert row["keito_name"] == "テスト系"
        assert not any("ex" in key or "setsumei" in key for key in row)


def test_parse_dispatches_blood_records():
    assert records.parse(fx.make_sk())[0] == "SK"
    assert records.parse(fx.make_hn())[0] == "HN"
    assert records.parse(fx.make_bt())[0] == "BT"


@pytest.mark.parametrize(
    ("parser", "make", "old_length"),
    [(records.parse_sk, fx.make_sk, 178),      # 祖先14頭 × 8桁、生産者コード6桁
     (records.parse_hn, fx.make_hn, 245),      # 繁殖登録番号3か所 × 8桁
     (records.parse_bt, fx.make_bt, 6887)],    # 繁殖登録番号 8桁
)
def test_blood_parsers_reject_the_pre_2023_format(parser, make, old_length):
    """JV-Data 4.9.0（2023-08）より前の形式は番号が8桁で、位置が全部ずれる。

    dataspec BLOD はこの旧形式のまま 2023-07 で止まっている（2026-09-25 実測）。
    黙って読むと別の馬の番号になるので、長さで弾く。新形式は BLDN が運ぶ。
    """
    with pytest.raises(ValueError, match="BLDN"):
        parser(make()[:old_length - 2] + b"\r\n")


# --- 払戻（HR）・オッズ（O1〜O6） -----------------------------------------------


def _rows(pairs, table):
    return [row for key, row in pairs if key == table]


class TestParseHr:
    def test_race_level_row(self):
        (race,) = _rows(records.parse_hr(fx.make_hr()), "payout_races")
        assert race["race_id"] == "2026032106010511"
        assert race["toroku_tosu"] == 12 and race["syusso_tosu"] == 12
        assert race["data_kubun"] == "2" and race["make_date"] == "20260323"
        for column in ("fuseiritu", "tokubarai", "henkan", "henkan_umaban",
                       "henkan_wakuban", "henkan_dowaku"):
            assert race[column] == "", column

    def test_one_row_per_winning_combination(self):
        payouts = _rows(records.parse_hr(fx.make_hr()), "payouts")
        got = {(r["bet_type"], r["combination"], r["payout_yen"], r["popularity"])
               for r in payouts}
        assert got == {
            ("tansyo", "05", 350, 2),
            ("fukusyo", "05", 140, 2), ("fukusyo", "03", 210, 4),
            ("fukusyo", "07", 480, 8),
            ("wakuren", "3-5", 890, 5),
            ("umaren", "03-05", 1230, 6),
            ("wide", "03-05", 450, 5), ("wide", "05-07", 1020, 12),
            ("wide", "03-07", 1580, 19),
            ("umatan", "05-03", 2010, 9),
            ("sanrenpuku", "03-05-07", 4560, 15),
            ("sanrentan", "05-03-07", 21340, 71),
        }
        assert len(payouts) == 12            # 空白の繰返しは行にしない
        for row in payouts:
            assert row["race_id"] == "2026032106010511"
            assert row["data_kubun"] == "2" and row["make_date"] == "20260323"

    def test_unordered_bet_types_are_sorted_and_ordered_ones_are_not(self):
        """馬連・ワイド・枠連・3連複は昇順にそろえる。馬単・3連単は着順のまま。"""
        pays = {"umaren": [("0503", "000001230", "006")],
                "sanrenpuku": [("070503", "000004560", "015")],
                "wakuren": [("53", "000000890", "05")],
                "umatan": [("0705", "000002010", "009")],
                "sanrentan": [("070305", "000021340", "0071")]}
        payouts = _rows(records.parse_hr(fx.make_hr(pays=pays)), "payouts")
        combos = {r["bet_type"]: r["combination"] for r in payouts}
        assert combos == {"umaren": "03-05", "sanrenpuku": "03-05-07",
                          "wakuren": "3-5", "umatan": "07-05",
                          "sanrentan": "07-03-05"}

    def test_dead_heat_gives_two_rows(self):
        """同着は繰返しの2件目以降に入る。1着2頭なら単勝の行が2つ。"""
        pays = {"tansyo": [("05", "000000240", "01"), ("08", "000000410", "03")]}
        payouts = _rows(records.parse_hr(fx.make_hr(pays=pays)), "payouts")
        assert [(r["combination"], r["payout_yen"]) for r in payouts] == [
            ("05", 240), ("08", 410)]

    def test_place_paid_to_second_only_skips_the_zero_slot(self):
        """複勝2着払いでは3件目の馬番が "00"（仕様書 特記事項 43a）。"""
        pays = {"fukusyo": [("05", "000000140", "02"), ("03", "000000210", "04"),
                            ("00", "000000000", "  ")]}
        payouts = _rows(records.parse_hr(fx.make_hr(pays=pays)), "payouts")
        assert [r["combination"] for r in payouts] == ["05", "03"]

    def test_special_payout_has_blank_combination(self):
        """特払・不成立は馬番 "00" に金額だけが入る。全券に払う金額として空の組番で残す。"""
        pays = {"tansyo": [("00", "000000070", "  ")]}
        pairs = list(records.parse_hr(fx.make_hr(pays=pays, tokubarai="100000000")))
        (race,) = _rows(pairs, "payout_races")
        (payout,) = _rows(pairs, "payouts")
        assert race["tokubarai"] == "tansyo"
        assert payout["combination"] == "" and payout["payout_yen"] == 70
        assert payout["popularity"] is None

    def test_flags_skip_the_reserved_slot(self):
        """各フラグの6番目は予備。7〜9番目が馬単・3連複・3連単。"""
        (race,) = _rows(records.parse_hr(fx.make_hr(
            fuseiritu="000001100", tokubarai="000000011", henkan="110000001")),
            "payout_races")
        assert race["fuseiritu"] == "umatan"
        assert race["tokubarai"] == "sanrenpuku;sanrentan"
        assert race["henkan"] == "tansyo;fukusyo;sanrentan"

    def test_refunded_horse_and_frame_numbers(self):
        (race,) = _rows(records.parse_hr(fx.make_hr(
            henkan_uma="0000100000010" + "0" * 15, henkan_waku="00001000",
            henkan_dowaku="00000010")), "payout_races")
        assert race["henkan_umaban"] == "05;12"
        assert race["henkan_wakuban"] == "5"
        assert race["henkan_dowaku"] == "7"

    def test_rejects_an_unexpected_length(self):
        with pytest.raises(ValueError, match="HR"):
            list(records.parse_hr(fx.make_hr()[:-3] + b"\r\n"))


class TestParseO1:
    @pytest.fixture
    def pairs(self):
        return list(records.parse_o1(fx.make_o1(
            tansyo=[(0, "01003401"), (1, "02999903"), (2, "03------"),
                    (3, "04****01"), (4, "05000012")],
            fukusyo=[(0, "010012001803"), (1, "02--------  ")],
            wakuren=[(0, "110012312"), (5, "16*******")],
        )))

    def test_win_odds(self, pairs):
        rows = _rows(pairs, "odds_tansyo")
        # 取消・無投票・空白は行にしない。"9999" は 999.9倍以上（下限として 999.9）
        assert [(r["combination"], r["odds"], r["popularity"]) for r in rows] == [
            ("01", 3.4, 1), ("02", 999.9, 3)]
        assert rows[0]["race_id"] == "2026032106010511"
        assert rows[0]["data_kubun"] == "4" and rows[0]["make_date"] == "20260323"
        assert rows[0]["happyo_time"] == ""

    def test_place_odds_have_low_and_high(self, pairs):
        (row,) = _rows(pairs, "odds_fukusyo")
        assert (row["combination"], row["odds"], row["odds_high"], row["popularity"]) == (
            "01", 1.2, 1.8, 3)

    def test_bracket_quinella_odds(self, pairs):
        (row,) = _rows(pairs, "odds_wakuren")
        assert (row["combination"], row["odds"], row["popularity"]) == ("1-1", 12.3, 12)

    def test_pool_rows(self, pairs):
        pools = {r["bet_type"]: r for r in _rows(pairs, "odds_pools")}
        assert set(pools) == {"tansyo", "fukusyo", "wakuren"}
        assert pools["tansyo"]["total_votes_100yen"] == 123456
        assert pools["fukusyo"]["total_votes_100yen"] == 234567
        assert pools["wakuren"]["total_votes_100yen"] == 34567
        assert all(p["sale_flag"] == "7" for p in pools.values())
        assert pools["fukusyo"]["fukusyo_chakubarai_key"] == "3"
        assert pools["tansyo"]["fukusyo_chakubarai_key"] == ""
        assert pools["tansyo"]["toroku_tosu"] == 12 and pools["tansyo"]["syusso_tosu"] == 12
        assert pools["tansyo"]["race_id"] == "2026032106010511"
        assert pools["tansyo"]["data_kubun"] == "4"

    def test_intermediate_odds_keep_the_announcement_time(self):
        pairs = list(records.parse_o1(fx.make_o1(data_kubun="1", happyo="03211455",
                                                 tansyo=[(0, "01003401")])))
        (row,) = _rows(pairs, "odds_tansyo")
        assert row["happyo_time"] == "03211455"
        assert _rows(pairs, "odds_pools")[0]["happyo_time"] == "03211455"

    def test_rejects_an_unexpected_length(self):
        with pytest.raises(ValueError, match="O1"):
            list(records.parse_o1(fx.make_o1()[:-3] + b"\r\n"))


@pytest.mark.parametrize(
    ("spec", "table", "slot", "expected"),
    [("O2", "odds_umaren", (0, "0102001234001"),
      {"combination": "01-02", "odds": 123.4, "popularity": 1}),
     ("O3", "odds_wide", (0, "01020001200018001"),
      {"combination": "01-02", "odds": 1.2, "odds_high": 1.8, "popularity": 1}),
     ("O4", "odds_umatan", (17, "0201000567002"),
      {"combination": "02-01", "odds": 56.7, "popularity": 2}),
     ("O5", "odds_sanrenpuku", (0, "010203001234001"),
      {"combination": "01-02-03", "odds": 123.4, "popularity": 1}),
     ("O6", "odds_sanrentan", (272, "0301020123456" + "4896"),
      {"combination": "03-01-02", "odds": 12345.6, "popularity": 4896})],
)
def test_combination_odds(spec, table, slot, expected):
    """組番の並びは JV のまま（馬単・3連単は着順、他は昇順で届く）。"""
    parser = getattr(records, f"parse_{spec.lower()}")
    width = fx.ODDS_LAYOUT[spec][1]
    kumi_len = 6 if spec in ("O5", "O6") else 4
    cancelled = (slot[0] + 1, slot[1][:kumi_len] + "*" * (width - kumi_len))
    pairs = list(parser(fx.make_odds(spec, slots=[slot, cancelled])))
    (row,) = _rows(pairs, table)                   # 取消の組番は行にしない
    for key, value in expected.items():
        assert row[key] == value, key
    assert row["race_id"] == "2026032106010511"
    assert row["data_kubun"] == "4"
    (pool,) = _rows(pairs, "odds_pools")
    assert pool["bet_type"] == table.removeprefix("odds_")
    assert pool["total_votes_100yen"] == 1234567
    assert pool["sale_flag"] == "7"


@pytest.mark.parametrize("spec", ["O2", "O3", "O4", "O5", "O6"])
def test_odds_parsers_reject_an_unexpected_length(spec):
    parser = getattr(records, f"parse_{spec.lower()}")
    with pytest.raises(ValueError, match=spec):
        list(parser(fx.make_odds(spec)[:-3] + b"\r\n"))


def test_no_votes_is_not_stored():
    """オッズ "000000" は無投票。オッズが無いので行にしない。"""
    pairs = list(records.parse_o2(fx.make_odds("O2", slots=[(0, "0102000000153")])))
    assert _rows(pairs, "odds_umaren") == []


def test_trifecta_rows_are_produced_lazily():
    """3連単は1レコード最大4,896行。リストに溜めずに1行ずつ渡す。

    Iterator かどうかでは足りない（作り終えたリストを iter() で包んでも通る）。
    ジェネレータであること＝行を作るのが取り出すときであることを確かめる。
    """
    import inspect
    raw = fx.make_odds("O6", slots=[(i, f"01{i % 17 + 2:02d}{i % 16 + 3:02d}00123450001")
                                    for i in range(0, 4896, 7)])
    rows = records.parse_rows(raw)
    assert inspect.isgenerator(rows)
    assert sum(1 for key, _ in rows if key == "odds_sanrentan") == len(range(0, 4896, 7))


class TestParseRows:
    def test_single_row_specs_are_keyed_by_the_record_spec(self):
        assert [key for key, _ in records.parse_rows(fx.make_ra())] == ["RA"]
        assert [key for key, _ in records.parse_rows(fx.make_sk())] == ["SK"]

    def test_payout_and_odds_records_split_into_tables(self):
        keys = {key for key, _ in records.parse_rows(fx.make_hr())}
        assert keys == {"payout_races", "payouts"}
        keys = {key for key, _ in records.parse_rows(
            fx.make_o1(tansyo=[(0, "01003401")]))}
        assert keys == {"odds_pools", "odds_tansyo"}

    def test_unknown_specs_give_nothing(self):
        raw = bytearray(fx.make_se())
        raw[0:2] = b"H1"
        assert list(records.parse_rows(bytes(raw))) == []

    def test_table_keys(self):
        assert records.table_keys("RA") == ("RA",)
        assert records.table_keys("HR") == ("payout_races", "payouts")
        assert records.table_keys("O1") == (
            "odds_pools", "odds_tansyo", "odds_fukusyo", "odds_wakuren")
        assert records.table_keys("O6") == ("odds_pools", "odds_sanrentan")
        assert records.table_keys("H1") == ()
