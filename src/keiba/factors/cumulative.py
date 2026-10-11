"""「その行の日より前」だけを使う累積と、経験ベイズの縮小（仕様 §2「縮小」・§3.2）。

累積は日付単位で取り、同じ日の行は含めない（その日のほかのレースの結果は、予想の時点ではまだ分からない）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prior_day_sums(df: pd.DataFrame, keys: list[str], value: str) -> tuple[np.ndarray, np.ndarray]:
    """各行について、同じ keys の行のうち日付がその行より前のものの value の合計と件数。NaN は数えない。"""
    v = df[value].to_numpy(float)
    d = df[keys].copy()
    d["date"] = df["date"].to_numpy()
    d["_s"] = np.nan_to_num(v)
    d["_n"] = (~np.isnan(v)).astype(float)
    daily = d.groupby(keys + ["date"], sort=True)[["_s", "_n"]].sum()
    before = daily.groupby(level=list(range(len(keys))), sort=False).cumsum() - daily
    got = before.reindex(pd.MultiIndex.from_frame(d[keys + ["date"]]))
    return got["_s"].to_numpy(), got["_n"].to_numpy()


def eb_lambda(values: np.ndarray, keys: np.ndarray) -> float | None:
    """組ごとの平均を縮める強さ λ = σ²/τ²（モーメント法）。組どうしの差が偶然の範囲（τ² ≤ 0）なら None。"""
    d = pd.DataFrame({"v": np.asarray(values, float), "k": np.asarray(keys)}).dropna(subset=["v"])
    if d.empty:
        return None
    g = d.groupby("k")["v"]
    n, m = g.size(), g.mean()
    if len(n) < 2:
        return None
    within = float(((d["v"] - d["k"].map(m)) ** 2).sum() / max(len(d) - len(n), 1))
    tau2 = float(np.average((m - d["v"].mean()) ** 2, weights=n)) - within * len(n) / float(n.sum())
    if tau2 <= 0 or within <= 0:
        return None
    return within / tau2


def shrunk_prior_scores(df: pd.DataFrame, keys: list[str], value: str, *, first_year: int) -> np.ndarray:
    """各行の点数 = (S − n·μ) / (n + λ_Y)。

    S・n はその行の日より前の同じ組の value の合計と件数、μ はその行の日より前の全体の平均。
    λ_Y は行の年 Y の前年末までの行で推定する。λ が無い年（組の差が偶然の範囲）と n = 0 の行は 0（平均並み）。
    first_year より前の年の行は NaN（助走）。
    """
    S, n = prior_day_sums(df, keys, value)
    allS, alln = prior_day_sums(df.assign(_all=0), ["_all"], value)
    mu = np.divide(allS, alln, out=np.zeros_like(allS), where=alln > 0)
    keyarr = df[keys].astype(str).agg("|".join, axis=1).to_numpy()
    vals = df[value].to_numpy(float)
    dates = df["date"].to_numpy()
    years = df["year"].to_numpy()
    out = np.full(len(df), np.nan)
    for y in np.unique(years):
        if y < first_year:
            continue
        rows = years == y
        prior = dates < np.datetime64(f"{int(y)}-01-01")
        lam = eb_lambda(vals[prior], keyarr[prior])
        if lam is None:
            out[rows] = 0.0
            continue
        out[rows] = (S[rows] - n[rows] * mu[rows]) / (n[rows] + lam)
    return out


def add_expected_top3(df: pd.DataFrame) -> pd.DataFrame:
    """見込み3着内率 e3（人気順位ごとの3着内率、その日より前）と A−E = 3着内 − e3 を足す。

    人気が無い行・その人気の前例が無い行は NaN。人気は 18 で頭打ち。
    """
    d = df.copy()
    d["_ninki"] = d["ninki"].clip(upper=18).fillna(-1).astype(int)
    S, n = prior_day_sums(d, ["_ninki"], "top3")
    e3 = np.divide(S, n, out=np.full(len(d), np.nan), where=n > 0)
    e3[d["_ninki"].to_numpy() < 1] = np.nan
    d["e3"] = e3
    d["ae"] = d["top3"] - d["e3"]
    return d.drop(columns="_ninki")
