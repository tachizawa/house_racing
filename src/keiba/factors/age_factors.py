"""年齢・晩成度×年齢・世代レベル（仕様 §3.2 #6〜8）。

どれも年ごとの物差し（figure_factors.yearly_figures）の指数を使い、評価する年の1/1（世代レベルは月の1日）
より前のデータだけで作る。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from keiba.evaluate.age_curve import age_curve, curve_levels, period_values
from keiba.evaluate.generation import cohort_table, yearly_comparison
from keiba.factors.cumulative import eb_lambda

FIRST_DATA_YEAR = 2015
COHORT_AGES = list(range(2, 10))
LATE_A_MIN_N, LATE_B_MIN_N = 10, 5
LATE_B_FIRST_COHORT = 2013   # 2015年の2歳戦から見える最初の世代


def _rows(R: pd.DataFrame, year: int) -> np.ndarray:
    return (R["year"] == year).to_numpy()


def age_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """前年末までのデータで作った年齢曲線の、その馬の年齢・半期の水準。"""
    out = np.full(len(R), np.nan)
    curve = age_curve(fig, years=list(range(FIRST_DATA_YEAR, year)), before=date(year, 1, 1), n_boot=1, seed=0)
    lv = curve_levels(curve)
    rows = _rows(R, year)
    half = (R["date"].dt.month > 6).astype(int).to_numpy()
    ages = R["barei"].to_numpy()
    out[rows] = [lv.get((int(a), int(h)), np.nan) for a, h in zip(ages[rows], half[rows])]
    return out


def _shrunk_means(values: pd.Series, keys: pd.Series) -> tuple[pd.Series, pd.Series]:
    d = pd.DataFrame({"v": values.to_numpy(float), "k": keys.to_numpy()})
    d = d[(d["k"].astype(str) != "") & d["v"].notna()]
    g = d.groupby("k")["v"]
    n, m = g.size(), g.mean()
    lam = eb_lambda(d["v"].to_numpy(), d["k"].to_numpy())
    eff = (m - d["v"].mean()) * (n / (n + lam) if lam is not None else 0.0)
    return eff, n


def sire_lateness(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> pd.Series:
    """父の晩成度 L: 長い期間の伸び（A）と出世の時期（B）の縮小効果をそれぞれ標準化して平均したもの。

    材料は前年末までに5歳前半を終えた世代（cohort ≤ year − 6）だけ（仕様 §3.2 #6）。
    """
    max_c = year - 6
    cut = pd.Timestamp(year, 1, 1)
    sire = R[["ketto_num", "sire_num"]].drop_duplicates("ketto_num")
    parts = []
    pv = period_values(fig, years=list(range(FIRST_DATA_YEAR, year)), before=date(year, 1, 1))
    early = pv[(pv.age == 3) & (pv.half == 0)][["ketto_num", "year", "value"]].rename(
        columns={"value": "v_early", "year": "y3"})
    late = pv[((pv.age == 4) & (pv.half == 1)) | ((pv.age == 5) & (pv.half == 0))]
    late = late.groupby("ketto_num")["value"].mean().rename("v_late").reset_index()
    ga = early.merge(late, on="ketto_num")
    ga["cohort"] = ga["y3"] - 3
    ga = ga[ga["cohort"] <= max_c].copy()
    if len(ga) > 2:
        ga["delta"] = ga["v_late"] - ga["v_early"]

        def resid(d: pd.DataFrame) -> pd.Series:
            x, y = d["v_early"] - d["v_early"].mean(), d["delta"]
            den = float((x ** 2).sum())
            beta = float((x * (y - y.mean())).sum()) / den if den > 0 else 0.0
            return y - y.mean() - beta * x

        ga["resid"] = pd.concat([resid(d) for _, d in ga.groupby("cohort")[["v_early", "delta"]]]).reindex(ga.index)
        ga = ga.merge(sire, on="ketto_num")
        eA, nA = _shrunk_means(ga["resid"], ga["sire_num"])
        eA = eA[nA >= LATE_A_MIN_N]
        if len(eA) > 1 and eA.std() > 0:
            parts.append(((eA - eA.mean()) / eA.std()).rename("A"))
    op = R[(R["date"] < cut) & (R["class_level"] >= 5) & (R["top3"] == 1) & R["barei"].between(2, 5)]
    first = op.sort_values("date").drop_duplicates("ketto_num")
    first = first[(first["year"] - first["barei"]).between(LATE_B_FIRST_COHORT, max_c)]
    if len(first) > 2:
        eB, nB = _shrunk_means((first["barei"] >= 4).astype(float), first["sire_num"])
        eB = eB[nB >= LATE_B_MIN_N]
        if len(eB) > 1 and eB.std() > 0:
            parts.append(((eB - eB.mean()) / eB.std()).rename("B"))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def late_age_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """L × clip(年齢 − 3.5, −1.5, +1.5)。L の無い父は NaN。"""
    out = np.full(len(R), np.nan)
    L = sire_lateness(R, fig, year)
    rows = _rows(R, year)
    lv = R["sire_num"].map(L).to_numpy(float)
    coef = np.clip(R["barei"].to_numpy(float) - 3.5, -1.5, 1.5)
    out[rows] = (lv * coef)[rows]
    return out


def cohort_level_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """その馬の世代の例年比（中央値）。月の1日の 0:00 を時点として月ごとに計算する。"""
    out = np.full(len(R), np.nan)
    months = R["date"].dt.month.to_numpy()
    yr = _rows(R, year)
    for month in range(1, 13):
        rows = yr & (months == month)
        if not rows.any():
            continue
        table = cohort_table(fig, as_of=date(year, month, 1), ages=COHORT_AGES,
                             years=list(range(FIRST_DATA_YEAR, year + 1)))
        yc = yearly_comparison(table, year=year)
        med = yc[yc["metric"] == "median"].set_index("age")["diff"]
        out[rows] = R.loc[rows, "barei"].map(med).to_numpy(float)
    return out
