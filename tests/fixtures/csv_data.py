"""Repository のテスト用に、CSV とマニフェストを直接書く。

``VersionWriter``（Task 4）を経由しないのは、Repository のテストを ingest 側の
挙動から切り離すため。列名は ``keiba.ingest.store`` の定数をそのまま使い、
本物の CSV と形を揃える。値を指定しなかった列はすべて空文字列になる
（``csv.DictWriter`` の ``restval``）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from keiba import config
from keiba.ingest.store import HORSE_COLUMNS, RACE_COLUMNS, RACE_HORSE_COLUMNS, version_dir

# manifest の fetched_at。テストが Repository.data_fetched_at と突き合わせられる
# よう固定値にしてある。
DEFAULT_FETCHED_AT = "2026-03-22T01:00:00+09:00"


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            # None を渡した列も空文字列にする（本物の CSV の欠損表現と揃える）。
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def write_dataset(tmp_path: Path, races: list[dict], race_horses: list[dict],
                   horses: list[dict], *, version_id: str = "20260322T010000Z") -> Path:
    """races / race_horses / horses から version_id の CSV 一式を直接書く。

    manifest は status="complete" で書き、CURRENT も version_id を指すように
    する。戻り値は data_dir（tmp_path そのもの）。DataStore.open_current /
    open_version にそのまま渡せる。
    """
    data_dir = tmp_path
    path = version_dir(data_dir, version_id)
    path.mkdir(parents=True)

    _write_csv(path / "races.csv", RACE_COLUMNS, races)
    _write_csv(path / "race_horses.csv", RACE_HORSE_COLUMNS, race_horses)
    _write_csv(path / "horses.csv", HORSE_COLUMNS, horses)

    manifest = {
        "version_id": version_id,
        "schema_version": config.CSV_SCHEMA_VERSION,
        "fetched_at": DEFAULT_FETCHED_AT,
        "record_counts": {"RA": len(races), "SE": len(race_horses), "UM": len(horses)},
        "status": "complete",
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_dir / "CURRENT").write_text(version_id, encoding="utf-8")
    return data_dir


def make_race(*, race_id="2026032106011101", hasso_at="2026-03-21T15:25:00+09:00",
              jyo_cd="06", kyori_m="2000", track_cd="17", track_kind="turf",
              baba_cd="1", class_level="4", grade_cd="", jyoken_cd_youngest="016",
              toroku_tosu="12", syusso_tosu="12", nyusen_tosu="12",
              first_3f_sec="35.2", last_3f_sec="34.1", race_name="テスト特別",
              data_kubun="7", make_date="20260322") -> dict:
    """races.csv の1行。既定値は平地・中山・2000m の架空レース。"""
    return {
        "race_id": race_id, "hasso_at": hasso_at, "jyo_cd": jyo_cd,
        "kyori_m": kyori_m, "track_cd": track_cd, "track_kind": track_kind,
        "baba_cd": baba_cd, "class_level": class_level, "grade_cd": grade_cd,
        "jyoken_cd_youngest": jyoken_cd_youngest,
        "toroku_tosu": toroku_tosu, "syusso_tosu": syusso_tosu, "nyusen_tosu": nyusen_tosu,
        "first_3f_sec": first_3f_sec, "last_3f_sec": last_3f_sec, "race_name": race_name,
        "data_kubun": data_kubun, "make_date": make_date,
    }


def make_race_horse(*, race_id="2026032106011101", umaban="01", wakuban="1",
                     ketto_num="0000000001", bamei="テストウマ", sex_cd="1",
                     barei="04", futan_kg="57.0", futan_before_kg="",
                     kisyu_code="00001", kisyu_code_before="", tozai_cd="1",
                     chokyosi_code="00001",
                     result_ijyo_cd="0", result_kakutei_jyuni="1",
                     result_dochaku_kubun="0", result_dochaku_tosu="",
                     result_time_sec="123.4", result_margin_sec="0.0",
                     result_jyuni_1c="3", result_jyuni_2c="3",
                     result_jyuni_3c="2", result_jyuni_4c="2",
                     result_last_3f_sec="34.1", result_ba_taijyu_kg="486",
                     result_zogen_sa_kg="-2", result_odds="3.4", result_ninki="1",
                     data_kubun="7", make_date="20260322") -> dict:
    """race_horses.csv の1行。既定値は正常決着・1着の架空の馬。"""
    return {
        "race_id": race_id, "umaban": umaban, "wakuban": wakuban,
        "ketto_num": ketto_num, "bamei": bamei, "sex_cd": sex_cd, "barei": barei,
        "futan_kg": futan_kg, "futan_before_kg": futan_before_kg,
        "kisyu_code": kisyu_code, "kisyu_code_before": kisyu_code_before,
        "tozai_cd": tozai_cd, "chokyosi_code": chokyosi_code,
        "result_ijyo_cd": result_ijyo_cd, "result_kakutei_jyuni": result_kakutei_jyuni,
        "result_dochaku_kubun": result_dochaku_kubun, "result_dochaku_tosu": result_dochaku_tosu,
        "result_time_sec": result_time_sec, "result_margin_sec": result_margin_sec,
        "result_jyuni_1c": result_jyuni_1c, "result_jyuni_2c": result_jyuni_2c,
        "result_jyuni_3c": result_jyuni_3c, "result_jyuni_4c": result_jyuni_4c,
        "result_last_3f_sec": result_last_3f_sec, "result_ba_taijyu_kg": result_ba_taijyu_kg,
        "result_zogen_sa_kg": result_zogen_sa_kg, "result_odds": result_odds,
        "result_ninki": result_ninki,
        "data_kubun": data_kubun, "make_date": make_date,
    }


def make_horse(*, ketto_num="0000000001", bamei="テストウマ", bamei_kana="ﾃｽﾄｳﾏ",
               birth_date="2022-03-15", sex_cd="1",
               data_kubun="1", make_date="20260320") -> dict:
    """horses.csv の1行。

    make_date の既定は、テストが使う予想締切（prediction_cutoff の既定は
    2026-03-20 20:00）以前にしてある。find_horses_by_name は make_date が
    as_of 以前の行しか見ないので、締切より後に作られた行しか持たない馬は
    候補にならない（実際のマスタ行も、そのレースに出走する馬なら締切より
    前に作られている）。行の選択そのものを試すテストは make_date を明示する。
    """
    return {
        "ketto_num": ketto_num, "bamei": bamei, "bamei_kana": bamei_kana,
        "birth_date": birth_date, "sex_cd": sex_cd,
        "data_kubun": data_kubun, "make_date": make_date,
    }
