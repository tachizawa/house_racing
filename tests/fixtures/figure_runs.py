"""タイム指数のテスト用の作り物の「走」。prepare_runs の出力と同じ列を持つ。

古馬戦は 1000m あたりの秒が s（距離ごと）+ c（クラス）+ v（日ごと）ちょうどになるよう作る。
雑音を入れないので、推定はほぼ厳密に真値を取り戻す。
"""

from __future__ import annotations

import pandas as pd

S_TRUE = {1600: 60.0, 2000: 61.0}       # 基準（1000m あたり秒）
C_TRUE = {2: 0.0, 3: -0.5}              # クラス差。2（1勝）が基準
V_TRUE = [-0.3, -0.1, 0.0, 0.1, 0.25, 0.05]   # 2025-01-01〜06 の馬場差


def run_row(**kw) -> dict:
    row = dict(race_id="R", umaban="01", ketto_num="H0", day="20250101",
               date=pd.Timestamp("2025-01-01"), year=2025, jyo_cd="05", track_cd="24",
               track_kind="turf", kyori_m=1600, syubetu_cd="13", class_level=2.0,
               time_sec=96.0, rank=1.0, barei=4, cohort=2021)
    row.update(kw)
    return row


def _day(i: int) -> str:
    return f"202501{i + 1:02d}"


def older_runs() -> pd.DataFrame:
    """6日 × 2距離 × 2クラスの古馬戦の1着。"""
    rows = []
    for i, v in enumerate(V_TRUE):
        day = _day(i)
        for kyori, s in S_TRUE.items():
            for cls, c in C_TRUE.items():
                rows.append(run_row(race_id=f"{day}-{kyori}-{cls}", day=day, date=pd.Timestamp(day),
                                    kyori_m=kyori, class_level=float(cls),
                                    time_sec=(s + c + v) * kyori / 1000))
    return pd.DataFrame(rows)


def young_runs(slow: float) -> pd.DataFrame:
    """各日に1つずつの2歳戦（1600m）。1000m あたり slow 秒だけ遅い。"""
    rows = []
    for i, v in enumerate(V_TRUE):
        day = _day(i)
        rows.append(run_row(race_id=f"{day}-Y", day=day, date=pd.Timestamp(day), syubetu_cd="11",
                            class_level=1.0, time_sec=(S_TRUE[1600] + v + slow) * 1.6,
                            barei=2, cohort=2023, ketto_num=f"Y{i}"))
    return pd.DataFrame(rows)
