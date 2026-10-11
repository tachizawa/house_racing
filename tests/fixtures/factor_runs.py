"""要素別の予想のテスト用の作り物の「走」の表（keiba.factors.data.RUN_COLUMNS と同じ列）。

馬ごとの地力 + 騎手の上乗せ（J0 だけ +1.0）+ 雑音で着順を決める。オッズは騎手の上乗せを知らない
（地力だけから作る）ので、J0 は「人気から見込むより多く3着内に来る」騎手になる。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.data import RUN_COLUMNS

STRONG_JOCKEY = "J0"


def make_factor_runs(*, years=(2015, 2016, 2017), days_per_year=8, races_per_day=6, field=10,
                     seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_horses = 160
    ability = rng.normal(0, 1, n_horses)
    birth = rng.integers(min(years) - 6, min(years) - 1, n_horses)
    sire = np.array([f"S{i % 6}" for i in range(n_horses)])
    rows = []
    for year in years:
        for d in range(days_per_year):
            date = pd.Timestamp(year, 1 + d * 12 // days_per_year, 5 + d % 3)
            for r in range(races_per_day):
                jyo = "05" if r % 2 == 0 else "08"
                kind = "turf" if r % 3 else "dirt"
                kyori = 1600 if r % 2 == 0 else 2000
                track_cd = "24" if kind == "turf" else "23"
                race_id = f"{date:%Y%m%d}{jyo}0101{r + 1:02d}"
                horses = rng.choice(n_horses, field, replace=False)
                jockeys = rng.choice([f"J{j}" for j in range(12)], field, replace=False)
                bonus = np.where(jockeys == STRONG_JOCKEY, 1.0, 0.0)
                perf = ability[horses] + bonus + rng.normal(0, 1, field)
                order = np.argsort(-perf)
                jyuni = np.empty(field, int)
                jyuni[order] = np.arange(1, field + 1)
                p = np.exp(ability[horses]) / np.exp(ability[horses]).sum()
                odds = np.round(0.8 / p, 1)
                ninki = (-p).argsort().argsort() + 1
                for i, h in enumerate(horses):
                    rows.append(dict(
                        race_id=race_id, date=date, umaban=f"{i + 1:02d}", waku=i // 2 + 1, ketto_num=f"H{h:03d}",
                        barei=int(year - birth[h]), futan=55.0 if i % 4 else 57.0, kisyu=jockeys[i],
                        chokyosi=f"T{h % 5}", tozai="1" if h % 2 else "2", sire_num=sire[h], jyuni=float(jyuni[i]),
                        odds=float(odds[i]), ninki=float(ninki[i]),
                        time_sec=float((60.5 - 0.3 * perf[i]) * kyori / 1000), zogen=float(rng.integers(-8, 9)),
                        jyo=jyo, track_cd=track_cd, track_kind=kind, kyori=kyori, class_level=2.0, grade="",
                        syubetu="13"))
    R = pd.DataFrame(rows)
    R["year"] = R["date"].dt.year
    R["cohort"] = R["year"] - R["barei"]
    R["field"] = R.groupby("race_id")["race_id"].transform("size")
    R["top3"] = R["jyuni"].between(1, 3).astype(float)
    R["fin_pct"] = (R["jyuni"] - 1) / (R["field"] - 1)
    return R.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)[RUN_COLUMNS]


def to_runs_all(R: pd.DataFrame) -> pd.DataFrame:
    """speed_figure.RUN_COLUMNS の形（指数の物差しの推定に使う）に直す。"""
    return pd.DataFrame({
        "race_id": R["race_id"], "umaban": R["umaban"], "ketto_num": R["ketto_num"],
        "day": R["date"].dt.strftime("%Y%m%d"), "date": R["date"], "year": R["year"], "jyo_cd": R["jyo"],
        "track_cd": R["track_cd"], "track_kind": R["track_kind"], "kyori_m": R["kyori"].astype(int),
        "syubetu_cd": R["syubetu"], "class_level": R["class_level"], "time_sec": R["time_sec"],
        "rank": R["jyuni"], "barei": R["barei"], "cohort": R["cohort"],
    }).reset_index(drop=True)


def perturb_results_from(R: pd.DataFrame, day) -> pd.DataFrame:
    """day 以降の行の結果（着順・3着内・着順の割合・タイム）を入れ替えた写し。レース前の情報はそのまま。"""
    P = R.copy()
    m = P["date"] >= pd.Timestamp(day)
    P.loc[m, "jyuni"] = P.loc[m, "field"] - P.loc[m, "jyuni"] + 1
    P.loc[m, "top3"] = P.loc[m, "jyuni"].between(1, 3).astype(float)
    P.loc[m, "fin_pct"] = (P.loc[m, "jyuni"] - 1) / (P.loc[m, "field"] - 1)
    P.loc[m, "time_sec"] = P.loc[m, "time_sec"] + 3.0
    return P


def assert_no_lookahead(compute, R: pd.DataFrame, day) -> None:
    """day の行の点数が、day 以降の結果を変えても変わらないことを確かめる（仕様 §9「時点の規則」）。"""
    before = np.asarray(compute(R), float)
    after = np.asarray(compute(perturb_results_from(R, day)), float)
    rows = (R["date"] == pd.Timestamp(day)).to_numpy()
    assert rows.any(), "その日の行が無い"
    np.testing.assert_allclose(before[rows], after[rows], equal_nan=True)
