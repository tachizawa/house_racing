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
