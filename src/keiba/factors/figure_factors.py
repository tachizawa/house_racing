"""タイム指数と相手関係（仕様 §3.2 #1, 2）。

物差しは評価する年ごとに作り直す。基準タイムとクラス差は「その年の1/1より前」で推定し、その年の各日の馬場差は
その日の古馬戦の1着だけから（基準・クラス差を固定して）推定する。こうすると、ある日のレースの予想に使う過去走の
指数には、その日以降の結果が一切入らない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.features.speed_figure import FACE, KEY, OLDER_SYUBETU, FigureModel, apply_figures, fit_model


def day_variants(runs: pd.DataFrame, model: FigureModel) -> pd.Series:
    """runs の各開催面の馬場差を、その面の古馬戦の1着だけから推定する（基準・クラス差は model のまま）。"""
    w = runs[runs["syubetu_cd"].isin(OLDER_SYUBETU) & (runs["rank"] == 1) & runs["class_level"].notna()]
    w = w.drop_duplicates("race_id")
    if w.empty:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], [], []], names=FACE))
    s = model.standards.reindex(pd.MultiIndex.from_frame(w[KEY])).to_numpy()
    c = model.class_effects.reindex(pd.MultiIndex.from_frame(w[["class_level", "track_kind"]])).to_numpy()
    r = w["time_sec"].to_numpy() / (w["kyori_m"].to_numpy() / 1000.0) - s - c
    ok = ~np.isnan(r)
    w = w[ok].assign(_r=r[ok])
    g = w.groupby(FACE)["_r"]
    lam = model.sigma2 / model.tau2 if model.tau2 > 0 else 0.0
    return g.sum() / (g.size() + lam)


def yearly_model(runs_all: pd.DataFrame, year: int) -> FigureModel:
    cut = pd.Timestamp(year, 1, 1)
    model = fit_model(runs_all[runs_all["date"] < cut])
    cur = runs_all[(runs_all["date"] >= cut) & (runs_all["date"] < pd.Timestamp(year + 1, 1, 1))]
    new = day_variants(cur, model)
    variants = pd.concat([model.variants, new[~new.index.isin(model.variants.index)]])
    return FigureModel(model.standards, model.class_effects, variants, model.sigma2, model.tau2, model.n_races)


def yearly_figures(runs_all: pd.DataFrame, year: int) -> pd.DataFrame:
    """year の物差しで、year+1 年1/1 より前の全走に指数を付ける。"""
    model = yearly_model(runs_all, year)
    return apply_figures(runs_all[runs_all["date"] < pd.Timestamp(year + 1, 1, 1)], model)


def level_before(fig: pd.DataFrame, targets: pd.DataFrame, *, window_days: int = 365) -> np.ndarray:
    """targets の各行（ketto_num, date）について、同じ馬の [date − window, date) の指数の上位2走の平均。"""
    out = np.full(len(targets), np.nan)
    f = fig[fig["figure"].notna()].sort_values("date")
    groups = {k: (g["date"].to_numpy(), g["figure"].to_numpy()) for k, g in f.groupby("ketto_num", sort=False)}
    tk, td = targets["ketto_num"].to_numpy(), targets["date"].to_numpy()
    win = np.timedelta64(window_days, "D")
    for i in range(len(targets)):
        g = groups.get(tk[i])
        if g is None:
            continue
        d, v = g
        lo, hi = np.searchsorted(d, td[i] - win, "left"), np.searchsorted(d, td[i], "left")
        if hi > lo:
            seg = v[lo:hi]
            out[i] = float(seg.mean() if len(seg) == 1 else np.sort(seg)[-2:].mean())
    return out


def time_figure_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    out = np.full(len(R), np.nan)
    rows = (R["year"] == year).to_numpy()
    out[rows] = level_before(fig, R[rows])
    return out


def opponents_scores(R: pd.DataFrame, fig: pd.DataFrame, year: int) -> np.ndarray:
    """過去5走（1年以内）の「他の出走馬の当時の指数の平均 − 自分の着差（1600m 換算の秒）」の上位3走の平均。"""
    out = np.full(len(R), np.nan)
    win = R[(R["date"] >= pd.Timestamp(year - 1, 1, 1)) & (R["date"] < pd.Timestamp(year + 1, 1, 1))]
    lv = level_before(fig, win)
    d = win[["race_id", "ketto_num", "date", "time_sec", "kyori"]].assign(lv=lv)
    s = d.groupby("race_id")["lv"].transform("sum")
    c = d.groupby("race_id")["lv"].transform("count")
    own = d["lv"].notna().astype(float)
    others_n = c - own
    others = (s - d["lv"].fillna(0.0)) / others_n.where(others_n > 0)
    winner = d.groupby("race_id")["time_sec"].transform("min")
    margin = (d["time_sec"] - winner) * 1600.0 / d["kyori"]
    d = d.assign(perf=(others - margin).to_numpy())
    by = {k: (g["date"].to_numpy(), g["perf"].to_numpy()) for k, g in d.sort_values("date").groupby("ketto_num", sort=False)}
    year_rows = np.flatnonzero((R["year"] == year).to_numpy())
    keys, dates = R["ketto_num"].to_numpy(), R["date"].to_numpy()
    one_year = np.timedelta64(365, "D")
    for i in year_rows:
        g = by.get(keys[i])
        if g is None:
            continue
        dd, pp = g
        hi = np.searchsorted(dd, dates[i], "left")
        lo = max(np.searchsorted(dd, dates[i] - one_year, "left"), hi - 5)
        seg = pp[lo:hi]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = float(np.sort(seg)[-3:].mean())
    return out
