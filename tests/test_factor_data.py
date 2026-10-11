"""走の表（keiba.factors.data）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.data import RUN_COLUMNS, build_runs


def frames():
    races = pd.DataFrame([
        dict(race_id="2025010505010101", jyo_cd="05", track_cd="24", track_kind="turf", kyori_m="1600",
             class_level="2", grade_cd="", syubetu_cd="13", data_kubun="7", make_date="20250106"),
        dict(race_id="2025010505010102", jyo_cd="05", track_cd="51", track_kind="jump", kyori_m="3000",
             class_level="2", grade_cd="", syubetu_cd="18", data_kubun="7", make_date="20250106"),
        dict(race_id="2025010530010101", jyo_cd="30", track_cd="24", track_kind="dirt", kyori_m="1400",
             class_level="2", grade_cd="", syubetu_cd="13", data_kubun="7", make_date="20250106"),
    ], dtype=str)
    base = dict(race_id="2025010505010101", wakuban="1", barei="04", futan_kg="57.0", kisyu_code="J1",
                chokyosi_code="T1", tozai_cd="1", result_odds="3.5", result_ninki="1", result_time_sec="96.0",
                result_zogen_sa_kg="-2", result_ijyo_cd="0", data_kubun="7", make_date="20250106")
    horses = pd.DataFrame([
        dict(base, umaban="01", ketto_num="A", result_kakutei_jyuni="1"),
        dict(base, umaban="02", ketto_num="B", result_kakutei_jyuni="2", result_zogen_sa_kg=""),
        dict(base, umaban="03", ketto_num="C", result_kakutei_jyuni="0", result_ijyo_cd="4", result_time_sec="0"),
        dict(base, umaban="04", ketto_num="D", result_kakutei_jyuni="0", result_ijyo_cd="1"),
        dict(base, race_id="2025010505010102", umaban="01", ketto_num="E", result_kakutei_jyuni="1"),
        dict(base, race_id="2025010530010101", umaban="01", ketto_num="F", result_kakutei_jyuni="1"),
    ], dtype=str)
    ped = pd.DataFrame([dict(ketto_num="A", sire_num="S1"), dict(ketto_num="B", sire_num="S2")], dtype=str)
    return races, horses, ped


def test_build_runs_rows_and_columns():
    R = build_runs(*frames())
    assert list(R.columns) == RUN_COLUMNS
    assert R.ketto_num.tolist() == ["A", "B", "C"]          # 取消(1)・障害・地方は除く。中止(4)は出走として残す
    assert R.field.tolist() == [3, 3, 3]
    a = R.set_index("ketto_num").loc["A"]
    assert a.top3 == 1.0 and a.fin_pct == 0.0 and a.cohort == 2021 and a.sire_num == "S1"
    c = R.set_index("ketto_num").loc["C"]
    assert c.top3 == 0.0 and np.isnan(c.fin_pct) and np.isnan(c.time_sec)
    assert np.isnan(R.set_index("ketto_num").loc["B"].zogen)
    assert R.set_index("ketto_num").loc["C"].sire_num == ""
