"""JV-Data の固定長 cp932 レコードを dict にする。

位置と長さは JVData_Struct.py の SetDataB と JV-Data仕様書のフォーマットシート
から写した。vendor/ は再配布不可なので import しない。位置は 1 始まり。
"""

from __future__ import annotations

from keiba import config

RA_LENGTH = 1272
SE_LENGTH = 555
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


_PARSERS = {"RA": parse_ra, "SE": parse_se, "UM": parse_um}


def parse(raw: bytes) -> tuple[str, dict] | None:
    """レコード種別で振り分ける。扱わない種別（O1 など）は None。"""
    spec = record_spec(raw)
    parser = _PARSERS.get(spec)
    return (spec, parser(raw)) if parser else None
