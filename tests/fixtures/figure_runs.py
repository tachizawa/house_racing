"""タイム指数のテスト用の作り物の「走」。prepare_runs の出力と同じ列を持つ。

古馬戦は 1000m あたりの秒が s（距離ごと）+ c（クラス）+ v（日ごと）ちょうどになるよう作る。
雑音を入れないので、推定はほぼ厳密に真値を取り戻す。
"""

from __future__ import annotations

import numpy as np
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


def season_runs(*, years=(2024, 2025), days_per_year=4, field=12, seed=0) -> pd.DataFrame:
    """複数年の5月の開催日に、古馬戦（3〜5歳）と2歳戦（翌年分の世代）を並べる。

    年齢ごとに馬を field 頭ずつ作り、毎日すべての馬が走る。タイムは基準 + 馬場差 + 馬の地力 + 雑音。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year in years:
        ability = {age: rng.normal(0, 0.3, field) for age in (2, 3, 4, 5)}
        for d in range(days_per_year):
            day = f"{year}05{d + 1:02d}"
            v = float(rng.normal(0, 0.2))
            for kyori, s in S_TRUE.items():
                race_id = f"{day}-{kyori}-O"
                runners = [(age, i) for age in (3, 4, 5) for i in range(field)]
                times = [(s - ability[age][i] + v + rng.normal(0, 0.05)) * kyori / 1000
                         for age, i in runners]
                order = np.argsort(times)
                for rank, k in enumerate(order, start=1):
                    age, i = runners[k]
                    rows.append(run_row(race_id=race_id, umaban=f"{k + 1:02d}", ketto_num=f"C{year - age}-{i}",
                                        day=day, date=pd.Timestamp(day), year=year, kyori_m=kyori,
                                        time_sec=times[k], rank=float(rank), barei=age, cohort=year - age))
            young_id = f"{day}-1600-Y"
            for i in range(field):
                rows.append(run_row(race_id=young_id, umaban=f"{i + 1:02d}", ketto_num=f"C{year - 2}-{i}",
                                    day=day, date=pd.Timestamp(day), year=year, syubetu_cd="11",
                                    class_level=1.0,
                                    time_sec=(S_TRUE[1600] + 1.0 - ability[2][i] + v) * 1.6,
                                    rank=float(i + 1), barei=2, cohort=year - 2))
    return pd.DataFrame(rows)


def sparse_season_runs(*, years=(2024, 2025), n_days=16, horses_per_age=80, runs_per_horse=4,
                       noise=0.4, seed=0) -> pd.DataFrame:
    """区間の偏りを確かめるための作り物。馬は開催日の一部（runs_per_horse 日）にしか走らず、1走ごとの雑音が大きい。

    実データと同じく「開催日を抜くと馬の走が減り、最高値が下がる」形にしてある。毎年、年齢ごとに
    horses_per_age 頭。1〜8月の開催日に、古馬戦（3〜5歳・1勝クラス）を距離ごとに1つと2歳戦を1つ置く。
    """
    rng = np.random.default_rng(seed)
    rows = []
    days = [f"{m:02d}{d:02d}" for m in range(1, 9) for d in (5, 20)][:n_days]
    for year in years:
        horses = []
        for age in (2, 3, 4, 5):
            for i in range(horses_per_age):
                horses.append(dict(age=age, i=i, ability=float(rng.normal(0, 0.3)),
                                   kyori=1600 if age == 2 else (1600, 2000)[i % 2],
                                   days=set(rng.choice(len(days), runs_per_horse, replace=False).tolist())))
        for d, md in enumerate(days):
            day = f"{year}{md}"
            v = float(rng.normal(0, 0.2))
            groups = {("O", 1600): [], ("O", 2000): [], ("Y", 1600): []}
            for h in horses:
                if d in h["days"]:
                    groups[("Y" if h["age"] == 2 else "O", h["kyori"])].append(h)
            for (kind, kyori), members in groups.items():
                if not members:
                    continue
                times = [(S_TRUE[kyori] + v - h["ability"] + (1.0 if kind == "Y" else 0.0)
                          + float(rng.normal(0, noise))) * kyori / 1000 for h in members]
                for rank, k in enumerate(np.argsort(times), start=1):
                    h = members[k]
                    rows.append(run_row(race_id=f"{day}-{kyori}-{kind}", umaban=f"{k + 1:02d}",
                                        ketto_num=f"C{year - h['age']}-{h['i']}", day=day,
                                        date=pd.Timestamp(day), year=year, kyori_m=kyori,
                                        syubetu_cd="11" if kind == "Y" else "13",
                                        class_level=1.0 if kind == "Y" else 2.0,
                                        time_sec=times[k], rank=float(rank), barei=h["age"],
                                        cohort=year - h["age"]))
    return pd.DataFrame(rows)
