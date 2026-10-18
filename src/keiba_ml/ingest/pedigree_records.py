"""血統系の JV-Data レコード（SK / HN / UM の3代血統）を dict にする。

位置は 1 始まり。JV-Data仕様書 4.9 のフォーマットから写し、2026-10-19 に
``scripts/ml/check_offsets.py`` で SDK 5.0.0 の ``JVData_Struct.py`` と照合した（全項目一致）。
SK/HN の位置は既存の ``keiba.ingest.records.parse_sk`` / ``parse_hn`` とも同じで、
既存の血統の版（BLDN）から作った祖先表では ``pedigree_parent_link`` が 99.85%。

| 種別 | 内容 | dataspec | 長さ |
|---|---|---|---|
| SK | 産駒マスタ。血統登録番号 → 3代血統の繁殖登録番号14個 | BLDN | 208 |
| HN | 繁殖馬マスタ。繁殖登録番号 → 馬名・生年・父母の繁殖登録番号 | BLDN | 251 |
| UM | 競走馬マスタ。3代血統の繁殖登録番号と馬名14組（SK の補完） | DIFF/DIFN | 1609 |

**BLOD は使わない。** BLOD は JV-Data 4.9.0（2023-08）より前の形式（繁殖登録番号8桁）の
まま 2023-07 作成分で止まっており、この長さ・位置で読むと全部ずれる（CLAUDE.md）。
SK/HN だけなら、既存の ``update_data.py --kind blood`` が取った版をそのまま読める
（``DataConfig.pedigree_source = "blood"``、既定）。この層で取るのは UM で補いたいときだけ。

既存の ``keiba.ingest.records.parse_um`` は3代血統を読まない（過去成績を持たない列
だけを読む方針）。既存の版の形式を変えないよう、血統はこちらで別に読む。
"""

from __future__ import annotations

from keiba_ml.config import KETTO3_PATHS

SK_LENGTH = 208
HN_LENGTH = 251
UM_LENGTH = 1609

# --- SK: 産駒マスタ ----------------------------------------------------------
SK_KETTO_NUM = (12, 10)
SK_BIRTH_DATE = (22, 8)
SK_SEX_CD = (30, 1)
SK_HINSYU_CD = (31, 1)
SK_KEIRO_CD = (32, 2)
SK_MOCHI_KUBUN = (34, 1)          # 産駒持込区分
SK_IMPORT_YEAR = (35, 4)
SK_BREEDER_CODE = (39, 8)
SK_SANCHI_NAME = (47, 20)
SK_HANSYOKU_START = 67            # 3代血統 繁殖登録番号 10バイト × 14
SK_HANSYOKU_LENGTH = 10

# --- HN: 繁殖馬マスタ --------------------------------------------------------
HN_HANSYOKU_NUM = (12, 10)
HN_KETTO_NUM = (30, 10)
HN_DEL_KUBUN = (40, 1)
HN_BAMEI = (41, 36)
HN_BAMEI_KANA = (77, 40)
HN_BAMEI_ENG = (117, 80)
HN_BIRTH_YEAR = (197, 4)
HN_SEX_CD = (201, 1)
HN_HINSYU_CD = (202, 1)
HN_KEIRO_CD = (203, 2)
HN_MOCHI_KUBUN = (205, 1)         # 繁殖馬持込区分
HN_IMPORT_YEAR = (206, 4)
HN_SANCHI_NAME = (210, 20)
HN_FATHER_NUM = (230, 10)
HN_MOTHER_NUM = (240, 10)

# --- UM: 競走馬マスタの3代血統 -----------------------------------------------
UM_KETTO_NUM = (12, 10)
UM_BIRTH_DATE = (39, 8)
UM_BAMEI = (47, 36)
UM_SEX_CD = (201, 1)
UM_KETTO3_START = 205             # (繁殖登録番号 10 + 馬名 36) × 14
UM_KETTO3_NUM_LENGTH = 10
UM_KETTO3_NAME_LENGTH = 36

#: 書き出す列。anc_<位置> が繁殖登録番号、name_<位置> が馬名。
SK_COLUMNS: tuple[str, ...] = (
    "ketto_num", "birth_date", "sex_cd", "hinsyu_cd", "keiro_cd",
    "mochi_kubun", "import_year", "breeder_code", "sanchi_name",
    *(f"anc_{p}" for p in KETTO3_PATHS),
    "data_kubun", "make_date", "record_length",
)
HN_COLUMNS: tuple[str, ...] = (
    "hansyoku_num", "ketto_num", "del_kubun", "bamei", "bamei_kana", "bamei_eng",
    "birth_year", "sex_cd", "hinsyu_cd", "keiro_cd", "mochi_kubun", "import_year",
    "sanchi_name", "father_num", "mother_num",
    "data_kubun", "make_date", "record_length",
)
UM_KETTO_COLUMNS: tuple[str, ...] = (
    "ketto_num", "bamei", "birth_date", "sex_cd",
    *(f"anc_{p}" for p in KETTO3_PATHS),
    *(f"name_{p}" for p in KETTO3_PATHS),
    "data_kubun", "make_date", "record_length",
)

EXPECTED_LENGTHS = {"SK": SK_LENGTH, "HN": HN_LENGTH, "UM": UM_LENGTH}


def _text(raw: bytes, pos: int, length: int) -> str:
    return raw[pos - 1 : pos - 1 + length].decode("cp932", errors="replace").strip()


def _field(raw: bytes, spec: tuple[int, int]) -> str:
    return _text(raw, *spec)


def record_spec(raw: bytes) -> str:
    return _text(raw, 1, 2)


def id_or_blank(value: str) -> str:
    """登録番号の「未設定」（空白・ゼロ埋め）を空文字列にそろえる。

    JV-Data は未設定をゼロ埋めで入れてくる。"0000000000" を祖先 ID として残すと
    「祖先不明の馬」全員が同じ1頭の祖先を共有していることになり、CatBoost が
    それを一頭の名種牡馬のように学習してしまう。
    """
    value = value.strip()
    return value if value.strip("0") else ""


def ymd(value: str) -> str:
    value = value.strip()
    if len(value) != 8 or not value.isdigit() or value == "00000000":
        return ""
    month, day = value[4:6], value[6:]
    if month == "00":
        return ""
    if day == "00":            # 古い馬は生日不明で 00 のことがある。年月だけ残す
        day = "01"
    return f"{value[:4]}-{month}-{day}"


def _year(value: str) -> str:
    value = value.strip()
    return value if len(value) == 4 and value.isdigit() and value != "0000" else ""


def _header(raw: bytes) -> dict:
    return {
        "data_kubun": _text(raw, 3, 1), "make_date": _text(raw, 4, 8),
        "record_length": len(raw),
    }


def parse_sk(raw: bytes) -> dict:
    row = {
        "ketto_num": id_or_blank(_field(raw, SK_KETTO_NUM)),
        "birth_date": ymd(_field(raw, SK_BIRTH_DATE)),
        "sex_cd": _field(raw, SK_SEX_CD), "hinsyu_cd": _field(raw, SK_HINSYU_CD),
        "keiro_cd": _field(raw, SK_KEIRO_CD), "mochi_kubun": _field(raw, SK_MOCHI_KUBUN),
        "import_year": _year(_field(raw, SK_IMPORT_YEAR)),
        "breeder_code": _field(raw, SK_BREEDER_CODE),
        "sanchi_name": _field(raw, SK_SANCHI_NAME),
    }
    for i, path in enumerate(KETTO3_PATHS):
        pos = SK_HANSYOKU_START + SK_HANSYOKU_LENGTH * i
        row[f"anc_{path}"] = id_or_blank(_text(raw, pos, SK_HANSYOKU_LENGTH))
    row.update(_header(raw))
    return row


def parse_hn(raw: bytes) -> dict:
    row = {
        "hansyoku_num": id_or_blank(_field(raw, HN_HANSYOKU_NUM)),
        "ketto_num": id_or_blank(_field(raw, HN_KETTO_NUM)),
        "del_kubun": _field(raw, HN_DEL_KUBUN),
        "bamei": _field(raw, HN_BAMEI), "bamei_kana": _field(raw, HN_BAMEI_KANA),
        "bamei_eng": _field(raw, HN_BAMEI_ENG),
        "birth_year": _year(_field(raw, HN_BIRTH_YEAR)),
        "sex_cd": _field(raw, HN_SEX_CD), "hinsyu_cd": _field(raw, HN_HINSYU_CD),
        "keiro_cd": _field(raw, HN_KEIRO_CD), "mochi_kubun": _field(raw, HN_MOCHI_KUBUN),
        "import_year": _year(_field(raw, HN_IMPORT_YEAR)),
        "sanchi_name": _field(raw, HN_SANCHI_NAME),
        "father_num": id_or_blank(_field(raw, HN_FATHER_NUM)),
        "mother_num": id_or_blank(_field(raw, HN_MOTHER_NUM)),
    }
    row.update(_header(raw))
    return row


def parse_um_ketto(raw: bytes) -> dict:
    row = {
        "ketto_num": id_or_blank(_field(raw, UM_KETTO_NUM)),
        "bamei": _field(raw, UM_BAMEI),
        "birth_date": ymd(_field(raw, UM_BIRTH_DATE)),
        "sex_cd": _field(raw, UM_SEX_CD),
    }
    step = UM_KETTO3_NUM_LENGTH + UM_KETTO3_NAME_LENGTH
    for i, path in enumerate(KETTO3_PATHS):
        pos = UM_KETTO3_START + step * i
        row[f"anc_{path}"] = id_or_blank(_text(raw, pos, UM_KETTO3_NUM_LENGTH))
        row[f"name_{path}"] = _text(raw, pos + UM_KETTO3_NUM_LENGTH, UM_KETTO3_NAME_LENGTH)
    row.update(_header(raw))
    return row


_PARSERS = {"SK": parse_sk, "HN": parse_hn, "UM": parse_um_ketto}


def parse(raw: bytes) -> tuple[str, dict] | None:
    """扱う種別だけを dict にする。それ以外（BT など）は None。"""
    spec = record_spec(raw)
    parser = _PARSERS.get(spec)
    return (spec, parser(raw)) if parser else None
