"""タイム指数（仕様 §3）のテスト。

最優先は2つ。as_of より後のレースが混ざらないこと（Task 1）と、2歳戦の遅さが馬場差に
吸収されないこと（Task 2、§3.4）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from keiba.features.speed_figure import RUN_COLUMNS, prepare_runs

JST = timezone(timedelta(hours=9))


def jst(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=JST)


def race_row(race_id, *, hasso_at="2025-01-05T15:00:00+09:00", jyo_cd="05", track_cd="24",
             track_kind="turf", kyori_m="1600", syubetu_cd="13", class_level="2",
             data_kubun="7", make_date="20250106") -> dict:
    return dict(race_id=race_id, hasso_at=hasso_at, jyo_cd=jyo_cd, track_cd=track_cd,
                track_kind=track_kind, kyori_m=kyori_m, syubetu_cd=syubetu_cd,
                class_level=class_level, data_kubun=data_kubun, make_date=make_date)


def horse_row(race_id, umaban="01", *, ketto_num="H1", barei="04", time="96.0", jyuni="1",
              ijyo="0", data_kubun="7", make_date="20250106") -> dict:
    return dict(race_id=race_id, umaban=umaban, ketto_num=ketto_num, barei=barei,
                result_time_sec=time, result_kakutei_jyuni=jyuni, result_ijyo_cd=ijyo,
                data_kubun=data_kubun, make_date=make_date)


def frames(races, horses):
    return pd.DataFrame(races, dtype=str), pd.DataFrame(horses, dtype=str)


def test_prepare_runs_columns_and_cohort():
    races, horses = frames([race_row("2025010505010101")], [horse_row("2025010505010101")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert list(runs.columns) == RUN_COLUMNS
    row = runs.iloc[0]
    assert row.day == "20250105" and row.year == 2025 and row.barei == 4
    assert row.cohort == 2021                     # 世代 = 年 − 馬齢
    assert row.kyori_m == 1600 and row.time_sec == 96.0 and row["class_level"] == 2.0


def test_prepare_runs_excludes_races_at_or_after_as_of():
    races, horses = frames(
        [race_row("2025010505010101", hasso_at="2025-01-05T15:00:00+09:00"),
         race_row("2025010505010102", hasso_at="2025-01-05T15:30:00+09:00")],
        [horse_row("2025010505010101"), horse_row("2025010505010102")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 1, 5, 15, 30))
    assert runs.race_id.tolist() == ["2025010505010101"]   # 同時刻は含めない


# race_id の先頭8桁は開催日として読むので、作り物でも本物と同じ形にする
OK, JUMP, NAR, ENTRY = "2025010505010101", "2025010505010102", "2025010530010103", "2025010505010104"


def test_prepare_runs_exclusions():
    races, horses = frames(
        [race_row(OK), race_row(JUMP, track_kind="jump"), race_row(NAR, jyo_cd="30"),
         race_row(ENTRY, data_kubun="2")],
        [horse_row(OK, "01"), horse_row(OK, "02", ijyo="1"), horse_row(OK, "03", ijyo="4"),
         horse_row(OK, "04", time=""), horse_row(OK, "05", time="0"),
         horse_row(JUMP), horse_row(NAR), horse_row(ENTRY, data_kubun="2")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert runs.race_id.tolist() == [OK] and runs.umaban.tolist() == ["01"]


def test_prepare_runs_prefers_result_over_entry_rows():
    races, horses = frames(
        [race_row(OK, data_kubun="2", make_date="20250103"), race_row(OK)],
        [horse_row(OK, time="", data_kubun="2", make_date="20250103"), horse_row(OK)])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert len(runs) == 1 and runs.iloc[0].time_sec == 96.0


def test_prepare_runs_requires_timezone():
    races, horses = frames([race_row(OK)], [horse_row(OK)])
    with pytest.raises(ValueError):
        prepare_runs(races, horses, as_of=datetime(2025, 2, 1))
