"""合成レコード。バイト列の構造とオフセットだけを再現する。位置は 1 始まり。"""

from __future__ import annotations


def _blank(length: int) -> bytearray:
    return bytearray(b" " * length)


def _put(buf: bytearray, pos: int, value: str, length: int) -> None:
    encoded = value.encode("cp932")
    if len(encoded) > length:
        raise ValueError(f"{value!r} は {length} バイトに収まらない")
    buf[pos - 1 : pos - 1 + length] = encoded + b" " * (length - len(encoded))


def make_ra(*, data_kubun="7", year="2026", monthday="0321", jyo_cd="06",
            kaiji="01", nichiji="05", race_num="11", hondai="テスト特別",
            grade_cd=" ", jyoken_youngest="016", kyori="2000", track_cd="17",
            hasso_time="1525", toroku_tosu="12", syusso_tosu="12",
            nyusen_tosu="12", siba_baba="1", dirt_baba="0",
            first_3f="352", last_3f="341") -> bytes:
    buf = _blank(1272)
    _put(buf, 1, "RA", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, year, 4); _put(buf, 16, monthday, 4); _put(buf, 20, jyo_cd, 2)
    _put(buf, 22, kaiji, 2); _put(buf, 24, nichiji, 2); _put(buf, 26, race_num, 2)
    _put(buf, 33, hondai, 60)
    _put(buf, 615, grade_cd, 1)
    _put(buf, 617, "13", 2); _put(buf, 622, "4", 1)
    for i in range(5):                       # 5要素すべて同じ値を入れておく
        _put(buf, 623 + 3 * i, jyoken_youngest, 3)
    _put(buf, 698, kyori, 4); _put(buf, 706, track_cd, 2); _put(buf, 710, "A ", 2)
    _put(buf, 874, hasso_time, 4)
    _put(buf, 882, toroku_tosu, 2); _put(buf, 884, syusso_tosu, 2)
    _put(buf, 886, nyusen_tosu, 2)
    _put(buf, 888, "1", 1); _put(buf, 889, siba_baba, 1); _put(buf, 890, dirt_baba, 1)
    for i in range(10):                      # ラップ 12.3秒 × 10。合計は 36.9
        _put(buf, 891 + 3 * i, "123", 3)
    _put(buf, 970, first_3f, 3); _put(buf, 976, last_3f, 3)
    buf[1270] = ord("\r"); buf[1271] = ord("\n")
    return bytes(buf)


def make_se(*, data_kubun="7", year="2026", monthday="0321", jyo_cd="06",
            kaiji="01", nichiji="05", race_num="11", wakuban="1", umaban="01",
            ketto_num="0000000001", bamei="テストウマ", sex_cd="1", barei="04",
            tozai_cd="1", chokyosi_code="00001", futan="570", futan_before="   ",
            kisyu_code="00001", kisyu_code_before="     ", ba_taijyu="486",
            zogen_fugo="-", zogen_sa="002", ijyo_cd="0", kakutei_jyuni="01",
            dochaku_kubun="0", dochaku_tosu="0", time="1234", jyuni_1c="03",
            jyuni_2c="03", jyuni_3c="02", jyuni_4c="02", odds="0034", ninki="01",
            haron_l3="341", time_diff="-003") -> bytes:
    buf = _blank(555)
    _put(buf, 1, "SE", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, year, 4); _put(buf, 16, monthday, 4); _put(buf, 20, jyo_cd, 2)
    _put(buf, 22, kaiji, 2); _put(buf, 24, nichiji, 2); _put(buf, 26, race_num, 2)
    _put(buf, 28, wakuban, 1); _put(buf, 29, umaban, 2); _put(buf, 31, ketto_num, 10)
    _put(buf, 41, bamei, 36); _put(buf, 79, sex_cd, 1); _put(buf, 83, barei, 2)
    _put(buf, 85, tozai_cd, 1); _put(buf, 86, chokyosi_code, 5)
    _put(buf, 289, futan, 3); _put(buf, 292, futan_before, 3)
    _put(buf, 297, kisyu_code, 5); _put(buf, 302, kisyu_code_before, 5)
    _put(buf, 325, ba_taijyu, 3); _put(buf, 328, zogen_fugo, 1)
    _put(buf, 329, zogen_sa, 3); _put(buf, 332, ijyo_cd, 1)
    _put(buf, 335, kakutei_jyuni, 2); _put(buf, 337, dochaku_kubun, 1)
    _put(buf, 338, dochaku_tosu, 1); _put(buf, 339, time, 4)
    _put(buf, 352, jyuni_1c, 2); _put(buf, 354, jyuni_2c, 2)
    _put(buf, 356, jyuni_3c, 2); _put(buf, 358, jyuni_4c, 2)
    _put(buf, 360, odds, 4); _put(buf, 364, ninki, 2)
    _put(buf, 391, haron_l3, 3); _put(buf, 532, time_diff, 4)
    buf[553] = ord("\r"); buf[554] = ord("\n")
    return bytes(buf)


def make_um(*, data_kubun="0", ketto_num="0000000001", del_kubun="0",
            birth_date="20220315", bamei="テストウマ", bamei_kana="ﾃｽﾄｳﾏ",
            sex_cd="1") -> bytes:
    buf = _blank(1000)
    _put(buf, 1, "UM", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, ketto_num, 10); _put(buf, 22, del_kubun, 1)
    _put(buf, 23, "20240401", 8); _put(buf, 39, birth_date, 8)
    _put(buf, 47, bamei, 36); _put(buf, 83, bamei_kana, 36); _put(buf, 201, sex_cd, 1)
    return bytes(buf)


# 3代血統の並び（SK の HansyokuNum[0..13]）。父・母・父父・父母・母父・母母…
SK_ANCESTORS = [f"{i + 1:010d}" for i in range(14)]


def make_sk(*, data_kubun="1", ketto_num="0000000001", birth_date="20220315",
            sex_cd="1", ancestors=None, sanchi="日高町") -> bytes:
    buf = _blank(208)
    _put(buf, 1, "SK", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, ketto_num, 10); _put(buf, 22, birth_date, 8)
    _put(buf, 30, sex_cd, 1); _put(buf, 31, "1", 1); _put(buf, 32, "03", 2)
    _put(buf, 34, "0", 1); _put(buf, 35, "0000", 4); _put(buf, 39, "00001234", 8)
    _put(buf, 47, sanchi, 20)
    for i, num in enumerate(ancestors if ancestors is not None else SK_ANCESTORS):
        _put(buf, 67 + 10 * i, num, 10)
    buf[206] = ord("\r"); buf[207] = ord("\n")
    return bytes(buf)


def make_hn(*, data_kubun="1", hansyoku_num="0000000001", ketto_num="0000000000",
            bamei="テストシュボバ", bamei_eng="Test Sire", birth_year="2005",
            sex_cd="1", sire_num="0000000003", dam_num="0000000004",
            import_year="0000") -> bytes:
    buf = _blank(251)
    _put(buf, 1, "HN", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, hansyoku_num, 10); _put(buf, 30, ketto_num, 10); _put(buf, 40, "0", 1)
    _put(buf, 41, bamei, 36); _put(buf, 77, "ﾃｽﾄｼｭﾎﾞﾊﾞ", 40); _put(buf, 117, bamei_eng, 80)
    _put(buf, 197, birth_year, 4); _put(buf, 201, sex_cd, 1); _put(buf, 202, "1", 1)
    _put(buf, 203, "01", 2); _put(buf, 205, "0", 1); _put(buf, 206, import_year, 4)
    _put(buf, 210, "日高町", 20); _put(buf, 230, sire_num, 10); _put(buf, 240, dam_num, 10)
    buf[249] = ord("\r"); buf[250] = ord("\n")
    return bytes(buf)


def make_bt(*, data_kubun="1", hansyoku_num="0000000001",
            keito_id="0101", keito_name="テスト系") -> bytes:
    buf = _blank(6889)
    _put(buf, 1, "BT", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, hansyoku_num, 10); _put(buf, 22, keito_id, 30); _put(buf, 52, keito_name, 36)
    _put(buf, 88, "説明文" * 10, 6800)
    buf[6887] = ord("\r"); buf[6888] = ord("\n")
    return bytes(buf)


def _race_header(buf: bytearray, spec: str, data_kubun: str, *, year="2026",
                 monthday="0321", jyo_cd="06", kaiji="01", nichiji="05",
                 race_num="11") -> None:
    _put(buf, 1, spec, 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, "20260323", 8)
    _put(buf, 12, year, 4); _put(buf, 16, monthday, 4); _put(buf, 20, jyo_cd, 2)
    _put(buf, 22, kaiji, 2); _put(buf, 24, nichiji, 2); _put(buf, 26, race_num, 2)


def _crlf(buf: bytearray) -> bytes:
    buf[-2] = ord("\r"); buf[-1] = ord("\n")
    return bytes(buf)


# HR の払戻ブロック: 賭式 -> (開始位置, 繰返し, 1件のバイト長, 組番長, 人気順長)
HR_PAY_LAYOUT = {
    "tansyo": (103, 3, 13, 2, 2), "fukusyo": (142, 5, 13, 2, 2),
    "wakuren": (207, 3, 13, 2, 2), "umaren": (246, 3, 16, 4, 3),
    "wide": (294, 7, 16, 4, 3), "umatan": (454, 6, 16, 4, 3),
    "sanrenpuku": (550, 3, 18, 6, 3), "sanrentan": (604, 6, 19, 6, 4),
}

# 12頭立てで1着5番（5枠）・2着3番（3枠）・3着7番（6枠）
HR_DEFAULT_PAYS = {
    "tansyo": [("05", "000000350", "02")],
    "fukusyo": [("05", "000000140", "02"), ("03", "000000210", "04"),
                ("07", "000000480", "08")],
    "wakuren": [("35", "000000890", "05")],
    "umaren": [("0305", "000001230", "006")],
    "wide": [("0305", "000000450", "005"), ("0507", "000001020", "012"),
             ("0307", "000001580", "019")],
    "umatan": [("0503", "000002010", "009")],
    "sanrenpuku": [("030507", "000004560", "015")],
    "sanrentan": [("050307", "000021340", "0071")],
}


def make_hr(*, data_kubun="2", pays=None, fuseiritu="000000000",
            tokubarai="000000000", henkan="000000000",
            henkan_uma="0" * 28, henkan_waku="0" * 8, henkan_dowaku="0" * 8,
            **race) -> bytes:
    """払戻。pays は賭式 -> [(組番, 払戻金9桁, 人気順)]。省略した賭式は空白のまま。"""
    buf = _blank(719)
    _race_header(buf, "HR", data_kubun, **race)
    _put(buf, 28, "12", 2); _put(buf, 30, "12", 2)
    _put(buf, 32, fuseiritu, 9); _put(buf, 41, tokubarai, 9); _put(buf, 50, henkan, 9)
    _put(buf, 59, henkan_uma, 28); _put(buf, 87, henkan_waku, 8)
    _put(buf, 95, henkan_dowaku, 8)
    for bet_type, entries in (HR_DEFAULT_PAYS if pays is None else pays).items():
        start, repeat, width, kumi_len, _ = HR_PAY_LAYOUT[bet_type]
        assert len(entries) <= repeat
        for i, (kumi, pay, ninki) in enumerate(entries):
            pos = start + width * i
            _put(buf, pos, kumi, kumi_len); _put(buf, pos + kumi_len, pay, 9)
            _put(buf, pos + kumi_len + 9, ninki, width - kumi_len - 9)
    return _crlf(buf)


def _odds_header(buf: bytearray, spec: str, data_kubun: str, flags: str,
                 happyo: str, race: dict) -> None:
    _race_header(buf, spec, data_kubun, **race)
    _put(buf, 28, happyo, 8); _put(buf, 36, "12", 2); _put(buf, 38, "12", 2)
    _put(buf, 40, flags, len(flags))


def _odds_slots(buf: bytearray, start: int, width: int, slots) -> None:
    """slots は (繰返しの添字, 1件分の文字列) の並び。1件分は位置どおりに連結したもの。"""
    for index, text in slots:
        assert len(text.encode("cp932")) == width, text
        _put(buf, start + width * index, text, width)


def make_o1(*, data_kubun="4", flags="7773", happyo="00000000",
            tansyo=(), fukusyo=(), wakuren=(),
            totals=("00000123456", "00000234567", "00000034567"), **race) -> bytes:
    """単複枠。flags は 単勝・複勝・枠連の発売フラグ + 複勝着払キー。

    tansyo / fukusyo / wakuren は (添字, 1件分の文字列)。1件分の長さは 8 / 12 / 9。
    """
    buf = _blank(962)
    _odds_header(buf, "O1", data_kubun, flags, happyo, race)
    _odds_slots(buf, 44, 8, tansyo)
    _odds_slots(buf, 268, 12, fukusyo)
    _odds_slots(buf, 604, 9, wakuren)
    _put(buf, 928, totals[0], 11); _put(buf, 939, totals[1], 11)
    _put(buf, 950, totals[2], 11)
    return _crlf(buf)


# O2〜O6: 種別 -> (レコード長, 1件のバイト長, 票数合計の位置)
ODDS_LAYOUT = {
    "O2": (2042, 13, 2030), "O3": (2654, 17, 2642), "O4": (4031, 13, 4019),
    "O5": (12293, 15, 12281), "O6": (83285, 17, 83273),
}


def make_odds(spec, *, data_kubun="4", flag="7", happyo="00000000", slots=(),
              total="00001234567", **race) -> bytes:
    """O2〜O6。slots は (添字, 1件分の文字列)。組番昇順の添字は呼び出し側が決める。"""
    length, width, total_pos = ODDS_LAYOUT[spec]
    buf = _blank(length)
    _odds_header(buf, spec, data_kubun, flag, happyo, race)
    _odds_slots(buf, 41, width, slots)
    _put(buf, total_pos, total, 11)
    return _crlf(buf)
