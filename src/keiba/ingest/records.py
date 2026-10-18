"""JV-Data の固定長 cp932 レコードを dict にする。

位置と長さは JVData_Struct.py の SetDataB と JV-Data仕様書のフォーマットシート
から写した。vendor/ は再配布不可なので import しない。位置は 1 始まり。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from keiba import config

RA_LENGTH = 1272
SE_LENGTH = 555
# 血統（JV-Data 4.9.0 以降の形式。繁殖登録番号が10桁）。
SK_LENGTH = 208
HN_LENGTH = 251
BT_LENGTH = 6889
_JST = "+09:00"          # JRA-VAN の日時はすべて日本時間


def _text(raw: bytes, pos: int, length: int) -> str:
    return raw[pos - 1 : pos - 1 + length].decode("cp932", errors="replace").strip()


def record_spec(raw: bytes) -> str:
    return _text(raw, 1, 2)


def to_int(value: str) -> int | None:
    """数字だけなら int。空白や非数字は None（0 にしない）。"""
    value = value.strip()
    return int(value) if value.isdigit() else None


def positive_int(value: str) -> int | None:
    """0 が値として成立しない項目のための変換。

    確定着順・人気・コーナー通過順位は、仕様書の初期値が "00" で、
    0 という順位は存在しない。to_int はそのまま 0 を返してしまうので、
    走破タイム("0000")やハロンタイム("000")と同じく None に寄せる。
    0 を残すと「1着より前」に並んでしまい、ソート時に不正な結果を生む。
    """
    parsed = to_int(value)
    return parsed if parsed else None


def tenth_sec(value: str) -> float | None:
    """99.9秒形式（3桁、0.1秒単位）。000 と 999 は無効値。"""
    value = value.strip()
    if not value.isdigit() or value in ("000", "999"):
        return None
    return int(value) / 10.0


def race_time_sec(value: str) -> float | None:
    """走破タイム。9分99秒9 形式の4桁。'1234' → 1分23.4秒 = 83.4秒。"""
    value = value.strip()
    if len(value) != 4 or not value.isdigit() or value in ("0000", "9999"):
        return None
    return int(value[0]) * 60 + int(value[1:3]) + int(value[3]) / 10.0


def margin_sec(time_diff: str) -> float | None:
    """SE.TimeDiff から「1着馬とのタイム差（秒）」を復元する。

    符号は1着が '-'、2着以下が '+'。1着馬の欄には2着馬とのタイム差が入るが、
    1着馬の margin は定義上 0.0 なので符号が '-' なら値を捨てて 0.0 を返す。
    取消・除外・中止は '9999'。
    """
    value = (time_diff or "").strip()
    if not value or value == "9999" or len(value) != 4:
        return None
    sign, digits = value[0], value[1:]
    if not digits.isdigit():
        return None
    if sign == "-":
        return 0.0
    if sign == "+":
        return int(digits) / 10.0
    return None


def _race_id(raw: bytes) -> str:
    """RACE_ID（絶対位置 12 から 16 バイト）を連結した16桁のキー。"""
    return (_text(raw, 12, 4) + _text(raw, 16, 4) + _text(raw, 20, 2)
            + _text(raw, 22, 2) + _text(raw, 24, 2) + _text(raw, 26, 2))


def _hasso_at(year: str, monthday: str, hhmm: str) -> str | None:
    if not (len(year) == 4 and len(monthday) == 4 and len(hhmm) == 4):
        return None
    if not (year + monthday + hhmm).isdigit():
        return None
    return f"{year}-{monthday[:2]}-{monthday[2:]}T{hhmm[:2]}:{hhmm[2:]}:00{_JST}"


def parse_ra(raw: bytes) -> dict:
    year, monthday = _text(raw, 12, 4), _text(raw, 16, 4)
    track_cd = _text(raw, 706, 2)
    kind = config.track_kind(track_cd)
    grade_cd = raw[614:615].decode("cp932", errors="replace")   # 空白を潰さない
    jyoken_youngest = _text(raw, 635, 3)                        # 最若年条件
    siba_baba, dirt_baba = _text(raw, 889, 1), _text(raw, 890, 1)
    baba_cd = siba_baba if kind == "turf" else dirt_baba if kind == "dirt" else ""
    laps = [tenth_sec(_text(raw, 891 + 3 * i, 3)) for i in range(25)]
    return {
        "race_id": _race_id(raw), "year": year, "monthday": monthday,
        "jyo_cd": _text(raw, 20, 2), "kaiji": _text(raw, 22, 2),
        "nichiji": _text(raw, 24, 2), "race_num": _text(raw, 26, 2),
        "hasso_at": _hasso_at(year, monthday, _text(raw, 874, 4)),
        "kyori_m": to_int(_text(raw, 698, 4)),
        "track_cd": track_cd, "track_kind": kind,
        "jyoken_cd_youngest": jyoken_youngest,
        "jyoken_cd_all": ";".join(_text(raw, 623 + 3 * i, 3) for i in range(5)),
        "grade_cd": grade_cd.strip(),
        "class_level": config.class_level(grade_cd, jyoken_youngest),
        "syubetu_cd": _text(raw, 617, 2), "kigo_cd": _text(raw, 619, 3),
        "jyuryo_cd": _text(raw, 622, 1), "course_kubun_cd": _text(raw, 710, 2),
        "tenko_cd": _text(raw, 888, 1),
        "siba_baba_cd": siba_baba, "dirt_baba_cd": dirt_baba, "baba_cd": baba_cd,
        "toroku_tosu": to_int(_text(raw, 882, 2)),
        "syusso_tosu": to_int(_text(raw, 884, 2)),
        "nyusen_tosu": to_int(_text(raw, 886, 2)),
        "lap_times_sec": ";".join("" if v is None else f"{v:.1f}" for v in laps),
        # 前3ハロンは元データに合計値が入っている。ラップを自分で足さない。
        "first_3f_sec": tenth_sec(_text(raw, 970, 3)),
        "last_3f_sec": tenth_sec(_text(raw, 976, 3)),
        "race_name": _text(raw, 33, 60),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


def _zogen(raw: bytes) -> int | None:
    """馬体重の増減。符号1バイト(328) + 3桁(329)。999 計量不能 / 空白 初出走。"""
    digits = _text(raw, 329, 3)
    if not digits.isdigit() or digits == "999":
        return None
    sign = raw[327:328].decode("cp932", errors="replace")
    return -int(digits) if sign == "-" else int(digits)


def _futan(value: str) -> float | None:
    """負担重量。3桁、0.1kg 単位。'570' → 57.0。"""
    value = value.strip()
    return int(value) / 10.0 if value.isdigit() and value != "000" else None


def _code_or_blank(value: str) -> str:
    """コードの「未設定」を空文字列にそろえる。

    JV-Data は未設定を空白ではなくゼロ埋めで入れてくる（騎手変更なしの
    ``kisyu_code_before`` は ``"00000"``）。生のまま通すと読む側が
    ``bool("00000")`` を「値がある」と読む——``EntryRow.changed_after_entry``
    が実際にそうなっていて、ほぼ全頭に偽の「登録後に変更あり」が付いていた。
    ここで正規化できるのは**これから作る版だけ**なので、読む側
    （``models.py``）にも同じ判定を置いてある。
    """
    value = value.strip()
    return value if value.strip("0") else ""


def parse_se(raw: bytes) -> dict:
    """結果側の列には result_ 接頭辞を付ける。誤用しにくくするため。"""
    odds = to_int(_text(raw, 360, 4))
    return {
        "race_id": _race_id(raw),
        "umaban": _text(raw, 29, 2), "wakuban": _text(raw, 28, 1),
        "ketto_num": _text(raw, 31, 10), "bamei": _text(raw, 41, 36),
        "sex_cd": _text(raw, 79, 1), "barei": _text(raw, 83, 2),
        "futan_kg": _futan(_text(raw, 289, 3)),
        "futan_before_kg": _futan(_text(raw, 292, 3)),
        "kisyu_code": _text(raw, 297, 5),
        "kisyu_code_before": _code_or_blank(_text(raw, 302, 5)),
        "tozai_cd": _text(raw, 85, 1), "chokyosi_code": _text(raw, 86, 5),
        "uma_kigo_cd": _text(raw, 77, 2), "blinker": _text(raw, 295, 1),
        "result_ijyo_cd": _text(raw, 332, 1),
        "result_kakutei_jyuni": positive_int(_text(raw, 335, 2)),
        "result_dochaku_kubun": _text(raw, 337, 1),
        "result_dochaku_tosu": to_int(_text(raw, 338, 1)),
        "result_time_sec": race_time_sec(_text(raw, 339, 4)),
        "result_margin_sec": margin_sec(_text(raw, 532, 4)),
        "result_jyuni_1c": positive_int(_text(raw, 352, 2)),
        "result_jyuni_2c": positive_int(_text(raw, 354, 2)),
        "result_jyuni_3c": positive_int(_text(raw, 356, 2)),
        "result_jyuni_4c": positive_int(_text(raw, 358, 2)),
        "result_last_3f_sec": tenth_sec(_text(raw, 391, 3)),
        "result_haron_l4_sec": tenth_sec(_text(raw, 388, 3)),
        "result_ba_taijyu_kg": to_int(_text(raw, 325, 3)),
        "result_zogen_sa_kg": _zogen(raw),
        "result_odds": odds / 10.0 if odds else None,
        "result_ninki": positive_int(_text(raw, 364, 2)),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


def _ymd(value: str) -> str | None:
    value = value.strip()
    if len(value) != 8 or not value.isdigit() or value == "00000000":
        return None
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def parse_um(raw: bytes) -> dict:
    """時点に依存しない項目だけを読む。

    Ruikei* / ChakuKaisu* / Kyakusitu / RaceCount は「現在までの累計」で
    過去時点の値を復元できないため、ここでは一切読まない。
    """
    return {
        "ketto_num": _text(raw, 12, 10), "del_kubun": _text(raw, 22, 1),
        "reg_date": _ymd(_text(raw, 23, 8)), "del_date": _ymd(_text(raw, 31, 8)),
        "birth_date": _ymd(_text(raw, 39, 8)),
        "bamei": _text(raw, 47, 36), "bamei_kana": _text(raw, 83, 36),
        "sex_cd": _text(raw, 201, 1), "hinsyu_cd": _text(raw, 202, 1),
        "keiro_cd": _text(raw, 203, 2),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


# SK の HansyokuNum[0..13]（3代血統の繁殖登録番号）の並び。仕様書どおり
# 父・母・父父・父母・母父・母母・父父父・父父母・父母父・父母母・母父父・母父母・母母父・母母母。
SK_ANCESTOR_NAMES: tuple[str, ...] = (
    "sire", "dam",
    "sire_sire", "sire_dam", "dam_sire", "dam_dam",
    "sire_sire_sire", "sire_sire_dam", "sire_dam_sire", "sire_dam_dam",
    "dam_sire_sire", "dam_sire_dam", "dam_dam_sire", "dam_dam_dam",
)


def _require_blood_length(raw: bytes, expected: int) -> None:
    """JV-Data 4.9.0（2023-08）より前の血統レコードを弾く。

    旧形式は繁殖登録番号が8桁（生産者コードも6桁）で、それより後ろの位置が
    全部ずれる。dataspec BLOD はこの旧形式のまま 2023-07 作成分で止まっており
    （2026-09-25 実測）、黙って読むと祖先の番号が隣の馬の番号と混ざる。
    新形式を運ぶのは BLDN。
    """
    if len(raw) != expected:
        raise ValueError(
            f"{record_spec(raw)} レコードの長さが {len(raw)} バイトです（新形式は {expected}）。"
            " 繁殖登録番号が8桁の旧形式（dataspec BLOD）とみられます。BLDN で取得してください。")


def parse_sk(raw: bytes) -> dict:
    """産駒マスタ。祖先は繁殖登録番号で持ち、名前は HN から引く。"""
    _require_blood_length(raw, SK_LENGTH)
    row = {
        "ketto_num": _text(raw, 12, 10), "birth_date": _ymd(_text(raw, 22, 8)),
        "sex_cd": _text(raw, 30, 1), "hinsyu_cd": _text(raw, 31, 1),
        "keiro_cd": _text(raw, 32, 2), "sanku_mochi_kubun": _text(raw, 34, 1),
        "import_year": _code_or_blank(_text(raw, 35, 4)),
        "breeder_code": _text(raw, 39, 8), "sanchi_name": _text(raw, 47, 20),
    }
    for i, name in enumerate(SK_ANCESTOR_NAMES):
        row[f"{name}_num"] = _code_or_blank(_text(raw, 67 + 10 * i, 10))
    row["data_kubun"] = _text(raw, 3, 1)
    row["make_date"] = _text(raw, 4, 8)
    return row


def parse_hn(raw: bytes) -> dict:
    """繁殖馬マスタ。種牡馬・繁殖牝馬の両方が入る。"""
    _require_blood_length(raw, HN_LENGTH)
    return {
        "hansyoku_num": _text(raw, 12, 10),
        "ketto_num": _code_or_blank(_text(raw, 30, 10)),
        "bamei": _text(raw, 41, 36), "bamei_kana": _text(raw, 77, 40),
        "bamei_eng": _text(raw, 117, 80),
        "birth_year": _code_or_blank(_text(raw, 197, 4)),
        "sex_cd": _text(raw, 201, 1), "hinsyu_cd": _text(raw, 202, 1),
        "keiro_cd": _text(raw, 203, 2), "mochi_kubun": _text(raw, 205, 1),
        "import_year": _code_or_blank(_text(raw, 206, 4)),
        "sanchi_name": _text(raw, 210, 20),
        "sire_num": _code_or_blank(_text(raw, 230, 10)),
        "dam_num": _code_or_blank(_text(raw, 240, 10)),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


def parse_bt(raw: bytes) -> dict:
    """系統情報。6800バイトの系統説明（自由文）は読まない。"""
    _require_blood_length(raw, BT_LENGTH)
    return {
        "hansyoku_num": _text(raw, 12, 10), "keito_id": _text(raw, 22, 30),
        "keito_name": _text(raw, 52, 36),
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


# --- 払戻（HR）・オッズ（O1〜O6） -------------------------------------------------
#
# 1レコードが複数の表の行になる（HR は払戻のレース単位の行と組番ごとの行、
# O1 は単勝・複勝・枠連の3つの賭式）。そのため parse() ではなく parse_rows() が
# (表のキー, 行) を1行ずつ返す。3連単は1レコード最大 4,896 行あるので、
# リストに溜めずにジェネレータで渡す（5年ぶんのセットアップで数千万行になる）。
#
# データ区分（仕様書 フォーマット 4・7〜12 の項番2）:
#   HR   : 1 速報成績（払戻金確定） 2 成績（月曜） 9 レース中止 0 該当レコード削除。
#          払戻金が確定しているのは 1 と 2。同じレースに両方あれば 2 を採る。
#   O1〜O6: 1 中間 2 前日売最終 3 最終 4 確定 5 確定（月曜） 9 レース中止 0 削除。
#          確定オッズは 4 と 5。同じレースに両方あれば 5 を採る。1〜3 は確定前の
#          値で、払戻の計算に使ってはいけない。
#   どちらも 0 と 9 は捨てる。優先順位は config.DATA_KUBUN_PAYOUT_PRIORITY /
#   config.DATA_KUBUN_ODDS_PRIORITY に置いた（選ぶのは読む側）。
#
# 組番は馬番2桁（枠連は枠番1桁）を "-" でつなぐ。馬単・3連単は着順どおり、
# それ以外（枠連・馬連・ワイド・3連複）は昇順にそろえる。

HR_LENGTH = 719
O1_LENGTH = 962
O2_LENGTH = 2042
O3_LENGTH = 2654
O4_LENGTH = 4031
O5_LENGTH = 12293
O6_LENGTH = 83285

#: HR の不成立・特払・返還フラグ（各9バイト）の並び。6番目は予備なので None。
_HR_FLAG_BET_TYPES: tuple[str | None, ...] = (
    "tansyo", "fukusyo", "wakuren", "umaren", "wide", None,
    "umatan", "sanrenpuku", "sanrentan",
)

#: HR の払戻ブロック。(賭式, 開始位置, 繰返し, 1件の長さ, 組番の長さ, 1頭ぶんの桁数,
#: 人気順の長さ, 着順どおりか)。払戻金はどれも9桁（円。100円あたり）。
#: 繰返しは同着を見込んだ数で、使わない繰返しは空白（または "00"）。
_HR_PAY_BLOCKS: tuple[tuple[str, int, int, int, int, int, int, bool], ...] = (
    ("tansyo", 103, 3, 13, 2, 2, 2, True),
    ("fukusyo", 142, 5, 13, 2, 2, 2, True),
    ("wakuren", 207, 3, 13, 2, 1, 2, False),
    ("umaren", 246, 3, 16, 4, 2, 3, False),
    ("wide", 294, 7, 16, 4, 2, 3, False),
    # 406〜453 は予備（3件 × 16バイト）
    ("umatan", 454, 6, 16, 4, 2, 3, True),
    ("sanrenpuku", 550, 3, 18, 6, 2, 3, False),
    ("sanrentan", 604, 6, 19, 6, 2, 4, True),
)


def _require_length(raw: bytes, expected: int) -> None:
    """レコード長が仕様書と違えば止める。

    払戻・オッズは繰返し項目の塊で、位置が1バイトずれると別の組番のオッズを
    読む。形式が変わったとき（BLOD→BLDN のように）黙って読み進めないよう、
    長さで確かめる。
    """
    if len(raw) != expected:
        raise ValueError(
            f"{record_spec(raw)} レコードの長さが {len(raw)} バイトです（仕様書では {expected}）。"
            " JV-Data の形式が変わった可能性があります。位置を確かめるまで取り込めません。")


def _kumi(value: str, part: int, ordered: bool) -> str:
    """"0503" → "05-03"（着順どおり）または "03-05"（昇順）。"""
    parts = [value[i:i + part] for i in range(0, len(value), part)]
    if not ordered:
        parts.sort()
    return "-".join(parts)


def _flag_bet_types(raw: bytes, pos: int) -> str:
    return ";".join(bet_type for i, bet_type in enumerate(_HR_FLAG_BET_TYPES)
                    if bet_type and _text(raw, pos + i, 1) == "1")


def _flag_numbers(raw: bytes, pos: int, count: int, width: int) -> str:
    """"1" の立っている位置の番号（1始まり、width 桁のゼロ埋め）を ";" でつなぐ。"""
    return ";".join(f"{i + 1:0{width}d}" for i in range(count)
                    if _text(raw, pos + i, 1) == "1")


def parse_hr(raw: bytes) -> Iterator[tuple[str, dict]]:
    """払戻。("payout_races", 1行) と ("payouts", 的中組番ごとの行) を返す。

    payout_races（レース単位）:
      fuseiritu / tokubarai / henkan … 不成立・特払・返還が「あり」の賭式を ";" で
        つないだもの（なければ空）。特払・返還フラグは 2002-06-15 以降のみ設定。
      henkan_umaban / henkan_wakuban / henkan_dowaku … 発売後取消で返還対象に
        なった馬番・枠番・同枠（例: 10頭立てで8番取消なら 7-7 だけが返還）。
        返還馬番を含む券は的中の有無にかかわらず全額返還になる。
    payouts（組番ごと）:
      payout_yen は 100円あたりの払戻金。同着なら同じ賭式に複数行。
      特払・不成立は馬番が "00" で金額だけが入る。これは「その賭式の全券に払う
      金額」なので combination を空文字列にして残す。
    """
    _require_length(raw, HR_LENGTH)
    race_id = _race_id(raw)
    data_kubun, make_date = _text(raw, 3, 1), _text(raw, 4, 8)
    yield "payout_races", {
        "race_id": race_id,
        "toroku_tosu": to_int(_text(raw, 28, 2)),
        "syusso_tosu": to_int(_text(raw, 30, 2)),
        "fuseiritu": _flag_bet_types(raw, 32),
        "tokubarai": _flag_bet_types(raw, 41),
        "henkan": _flag_bet_types(raw, 50),
        "henkan_umaban": _flag_numbers(raw, 59, 28, 2),
        "henkan_wakuban": _flag_numbers(raw, 87, 8, 1),
        "henkan_dowaku": _flag_numbers(raw, 95, 8, 1),
        "data_kubun": data_kubun, "make_date": make_date,
    }
    for bet_type, start, repeat, width, kumi_len, part, ninki_len, ordered in _HR_PAY_BLOCKS:
        special_seen = False
        for i in range(repeat):
            pos = start + width * i
            kumi = _text(raw, pos, kumi_len)
            payout = to_int(_text(raw, pos + kumi_len, 9))
            if kumi.isdigit() and kumi.strip("0"):
                combination = _kumi(kumi, part, ordered)
            elif payout and not special_seen:
                combination, special_seen = "", True       # 特払・不成立
            else:
                continue                                    # 空き・2着払いの "00"
            yield "payouts", {
                "race_id": race_id, "bet_type": bet_type, "combination": combination,
                "payout_yen": payout,
                "popularity": positive_int(_text(raw, pos + kumi_len + 9, ninki_len)),
                "data_kubun": data_kubun, "make_date": make_date,
            }


def odds_value(value: str) -> float | None:
    """オッズ（0.1倍単位の整数）。数値でないもの・0 は None。

    特殊値（仕様書 フォーマット 7〜12）:
      "0000…" 無投票 / "----…" 発売前取消 / "****…" 発売後取消 / 空白 登録なし → None
      上限の値（単勝 "9999" = 999.9倍以上、2004-08-13 以前の複勝 "0999" = 99.9倍以上
      など）はその値のまま返す。実際のオッズはそれ以上なので**下限**として読むこと。
    """
    value = value.strip()
    if not value.isdigit():
        return None
    number = int(value)
    return number / 10.0 if number else None


def _odds_head(raw: bytes) -> dict:
    return {"race_id": _race_id(raw), "happyo_time": _code_or_blank(_text(raw, 28, 8))}


def _odds_pool(head: dict, raw: bytes, bet_type: str, flag_pos: int, total_pos: int,
               chakubarai_key: str = "") -> tuple[str, dict]:
    """賭式ごとの発売状況と票数合計（単位百円、返還分を含む）。"""
    return "odds_pools", {
        **head, "bet_type": bet_type,
        "toroku_tosu": to_int(_text(raw, 36, 2)),
        "syusso_tosu": to_int(_text(raw, 38, 2)),
        # 0 発売なし / 1 発売前取消 / 3 発売後取消 / 7 発売あり
        "sale_flag": _text(raw, flag_pos, 1),
        "total_votes_100yen": to_int(_text(raw, total_pos, 11)),
        # 0 複勝発売なし / 2 2着まで払い / 3 3着まで払い（2002-06-15 以降）
        "fukusyo_chakubarai_key": chakubarai_key,
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
    }


def _odds_slots(raw: bytes, head: dict, table: str, start: int, repeat: int,
                kumi_len: int, part: int, odds_len: int, ninki_len: int,
                ordered: bool, has_high: bool) -> Iterator[tuple[str, dict]]:
    """繰返し項目を1組番1行にする。オッズが無い組番（取消・無投票・空き）は出さない。

    3連単では1レコードに 4,896 回まわるので、先にバイト列のまま空き・取消を
    弾いてから文字列にする。
    """
    width = kumi_len + odds_len * (2 if has_high else 1) + ninki_len
    data_kubun, make_date = _text(raw, 3, 1), _text(raw, 4, 8)
    for i in range(repeat):
        offset = start - 1 + width * i
        field = raw[offset + kumi_len:offset + kumi_len + odds_len]
        odds = odds_value(field.decode("ascii", "replace"))
        if odds is None:
            continue
        slot = raw[offset:offset + width].decode("ascii", "replace")
        row = {**head, "combination": _kumi(slot[:kumi_len], part, ordered), "odds": odds}
        tail = kumi_len + odds_len
        if has_high:
            row["odds_high"] = odds_value(slot[tail:tail + odds_len])
            tail += odds_len
        row["popularity"] = positive_int(slot[tail:tail + ninki_len])
        row["data_kubun"] = data_kubun
        row["make_date"] = make_date
        yield table, row


def parse_o1(raw: bytes) -> Iterator[tuple[str, dict]]:
    """オッズ1（単勝・複勝・枠連）。複勝は最低・最高オッズを odds / odds_high に。"""
    _require_length(raw, O1_LENGTH)
    head = _odds_head(raw)
    yield _odds_pool(head, raw, "tansyo", 40, 928)
    yield _odds_pool(head, raw, "fukusyo", 41, 939, _text(raw, 43, 1))
    yield _odds_pool(head, raw, "wakuren", 42, 950)
    yield from _odds_slots(raw, head, "odds_tansyo", 44, 28, 2, 2, 4, 2, True, False)
    yield from _odds_slots(raw, head, "odds_fukusyo", 268, 28, 2, 2, 4, 2, True, True)
    yield from _odds_slots(raw, head, "odds_wakuren", 604, 36, 2, 1, 5, 2, False, False)


def _combination_odds(raw: bytes, length: int, bet_type: str, repeat: int,
                      kumi_len: int, odds_len: int, ninki_len: int, ordered: bool,
                      has_high: bool, total_pos: int) -> Iterator[tuple[str, dict]]:
    """O2〜O6 の共通形。発売フラグは 40、繰返しは 41 から、票数合計はその後ろ。"""
    _require_length(raw, length)
    head = _odds_head(raw)
    yield _odds_pool(head, raw, bet_type, 40, total_pos)
    yield from _odds_slots(raw, head, f"odds_{bet_type}", 41, repeat, kumi_len, 2,
                           odds_len, ninki_len, ordered, has_high)


def parse_o2(raw: bytes) -> Iterator[tuple[str, dict]]:
    """オッズ2（馬連）。153組、組番昇順 01-02〜17-18。"""
    return _combination_odds(raw, O2_LENGTH, "umaren", 153, 4, 6, 3, False, False, 2030)


def parse_o3(raw: bytes) -> Iterator[tuple[str, dict]]:
    """オッズ3（ワイド）。153組。最低・最高オッズを odds / odds_high に。"""
    return _combination_odds(raw, O3_LENGTH, "wide", 153, 4, 5, 3, False, True, 2642)


def parse_o4(raw: bytes) -> Iterator[tuple[str, dict]]:
    """オッズ4（馬単）。306組、01-02〜18-17（着順どおり）。"""
    return _combination_odds(raw, O4_LENGTH, "umatan", 306, 4, 6, 3, True, False, 4019)


def parse_o5(raw: bytes) -> Iterator[tuple[str, dict]]:
    """オッズ5（3連複）。816組、01-02-03〜16-17-18。"""
    return _combination_odds(raw, O5_LENGTH, "sanrenpuku", 816, 6, 6, 3, False, False, 12281)


def parse_o6(raw: bytes) -> Iterator[tuple[str, dict]]:
    """オッズ6（3連単）。4,896組、01-02-03〜18-17-16（着順どおり）。"""
    return _combination_odds(raw, O6_LENGTH, "sanrentan", 4896, 6, 7, 4, True, False, 83273)


_PARSERS = {"RA": parse_ra, "SE": parse_se, "UM": parse_um,
            "SK": parse_sk, "HN": parse_hn, "BT": parse_bt}

#: 1レコードが複数の表の行になる種別。値は (パーサ, 書き込む表のキー)。
_ROW_PARSERS: dict[str, tuple[Callable[[bytes], Iterator[tuple[str, dict]]],
                              tuple[str, ...]]] = {
    "HR": (parse_hr, ("payout_races", "payouts")),
    "O1": (parse_o1, ("odds_pools", "odds_tansyo", "odds_fukusyo", "odds_wakuren")),
    "O2": (parse_o2, ("odds_pools", "odds_umaren")),
    "O3": (parse_o3, ("odds_pools", "odds_wide")),
    "O4": (parse_o4, ("odds_pools", "odds_umatan")),
    "O5": (parse_o5, ("odds_pools", "odds_sanrenpuku")),
    "O6": (parse_o6, ("odds_pools", "odds_sanrentan")),
}


def parse(raw: bytes) -> tuple[str, dict] | None:
    """1レコード1行の種別を振り分ける。それ以外（HR・O1 など）は None。

    HR・O1〜O6 は1レコードが複数の表の行になるので parse_rows() を使う。
    """
    spec = record_spec(raw)
    parser = _PARSERS.get(spec)
    return (spec, parser(raw)) if parser else None


def table_keys(spec: str) -> tuple[str, ...]:
    """レコード種別が行を書く表のキー。扱わない種別は空。"""
    if spec in _PARSERS:
        return (spec,)
    if spec in _ROW_PARSERS:
        return _ROW_PARSERS[spec][1]
    return ()


def parse_rows(raw: bytes) -> Iterator[tuple[str, dict]]:
    """(表のキー, 行) を1行ずつ返す。1行の種別は表のキー = レコード種別。

    扱わない種別（H1 など）は何も返さない。
    """
    spec = record_spec(raw)
    if spec in _PARSERS:
        return iter(((spec, _PARSERS[spec](raw)),))
    if spec in _ROW_PARSERS:
        return iter(_ROW_PARSERS[spec][0](raw))
    return iter(())
