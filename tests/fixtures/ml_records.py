"""血統系レコード（SK / HN / UM）の合成バイト列。位置は 1 始まり。

keiba_ml/ingest/pedigree_records.py の定数を使わず、仕様書の位置を数字で直に書く。
同じ定数を使うと、定数が間違っていてもテストが通ってしまうため。
"""

from __future__ import annotations

from tests.fixtures.records import _blank, _put


def make_sk(*, ketto_num="2015100001", birth_date="20150401", sex_cd="1",
            ancestors: list[str] | None = None, data_kubun="1",
            make_date="20240101") -> bytes:
    ancestors = ancestors or [""] * 14
    buf = _blank(208)
    _put(buf, 1, "SK", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, make_date, 8)
    _put(buf, 12, ketto_num, 10); _put(buf, 22, birth_date, 8); _put(buf, 30, sex_cd, 1)
    _put(buf, 31, "1", 1); _put(buf, 32, "03", 2); _put(buf, 34, "0", 1)
    _put(buf, 35, "0000", 4); _put(buf, 39, "00000001", 8); _put(buf, 47, "日高", 20)
    for i, value in enumerate(ancestors):
        _put(buf, 67 + 10 * i, value or "0000000000", 10)
    buf[206] = ord("\r"); buf[207] = ord("\n")
    return bytes(buf)


def make_hn(*, hansyoku_num="1000000001", ketto_num="", bamei="テストシュボバ",
            birth_year="2005", sex_cd="1", father="", mother="", data_kubun="1",
            make_date="20240101") -> bytes:
    buf = _blank(251)
    _put(buf, 1, "HN", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, make_date, 8)
    _put(buf, 12, hansyoku_num, 10); _put(buf, 22, "00000000", 8)
    _put(buf, 30, ketto_num or "0000000000", 10); _put(buf, 40, "0", 1)
    _put(buf, 41, bamei, 36); _put(buf, 197, birth_year, 4); _put(buf, 201, sex_cd, 1)
    _put(buf, 202, "1", 1); _put(buf, 203, "01", 2); _put(buf, 205, "0", 1)
    _put(buf, 206, "0000", 4); _put(buf, 210, "米", 20)
    _put(buf, 230, father or "0000000000", 10); _put(buf, 240, mother or "0000000000", 10)
    buf[249] = ord("\r"); buf[250] = ord("\n")
    return bytes(buf)


def make_um(*, ketto_num="2015100001", bamei="テストウマ", birth_date="20150401",
            sex_cd="1", ancestors: list[tuple[str, str]] | None = None,
            data_kubun="1", make_date="20240101") -> bytes:
    ancestors = ancestors or [("", "")] * 14
    buf = _blank(1609)
    _put(buf, 1, "UM", 2); _put(buf, 3, data_kubun, 1); _put(buf, 4, make_date, 8)
    _put(buf, 12, ketto_num, 10); _put(buf, 39, birth_date, 8); _put(buf, 47, bamei, 36)
    _put(buf, 201, sex_cd, 1)
    for i, (num, name) in enumerate(ancestors):
        _put(buf, 205 + 46 * i, num or "0000000000", 10)
        _put(buf, 215 + 46 * i, name, 36)
    buf[1607] = ord("\r"); buf[1608] = ord("\n")
    return bytes(buf)
