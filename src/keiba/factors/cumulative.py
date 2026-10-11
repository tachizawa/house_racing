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
    """組ごとの平均を縮める強さ λ = σ²/τ²（モーメント法、v1）。組どうしの差が偶然の範囲（τ² ≤ 0）なら None。

    σ² は組の中の散らばりを全組まとめた1つの値として扱う。A−E のように組ごとに散らばりが大きく違うと
    （人気馬に多く乗る大きな組は ≈0.25、穴馬ばかりの小さな組はずっと小さい）、まとめた σ² を組の数だけ
    引きすぎて τ² が小さく出る（λ が大きすぎる）。表の要素の v1 の記録を再現するために残している。
    v2 は eb_lambda_hetero を使う。
    """
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


def eb_variance_components(values: np.ndarray, keys: np.ndarray) -> tuple[float, float] | None:
    """組ごとに散らばりの違う場合のモーメント法（v2）。(σ̄², τ²) を返す。推定できなければ None。

    組 k の件数 n_k、平均 m_k、組の中の不偏分散 s²_k、N = Σn_k、m̄ = Σn_k m_k / N として、
    m_k = μ + u_k + e_k（Var u_k = τ²、Var e_k = σ²_k / n_k）のもとで

        E[Σ n_k (m_k − m̄)² / N] = τ² (1 − Σ n_k² / N²) + Σ σ²_k (1 − n_k / N) / N

    が成り立つので、τ² = [Σ n_k (m_k − m̄)² / N − Σ s²_k (1 − n_k/N) / N] / (1 − Σ n_k² / N²)。
    引く量を組ごとの s²_k で作るので、散らばりの小さい組が多くても引きすぎない。n_k < 2 の組は s²_k が
    取れないので、組の中の散らばりを全組まとめた値（v1 の within）で代える。
    σ̄² は s²_k を n_k で重み付けした平均（縮小の強さを1つの λ にまとめるため。散らばりの小さい小さな組には
    やや強めに縮める側＝保守的にずれる）。
    """
    d = pd.DataFrame({"v": np.asarray(values, float), "k": np.asarray(keys)}).dropna(subset=["v"])
    if d.empty:
        return None
    g = d.groupby("k")["v"]
    n, m = g.size().astype(float), g.mean()
    if len(n) < 2:
        return None
    N = float(n.sum())
    resid2 = (d["v"] - d["k"].map(m)) ** 2
    pooled = float(resid2.sum() / max(len(d) - len(n), 1))
    s2 = (resid2.groupby(d["k"]).sum() / (n - 1).where(n >= 2)).fillna(pooled)
    mbar = float((n * m).sum() / N)
    between = float((n * (m - mbar) ** 2).sum() / N)
    noise = float((s2 * (1 - n / N)).sum() / N)
    denom = 1.0 - float((n ** 2).sum()) / N ** 2
    if denom <= 0:
        return None
    tau2 = (between - noise) / denom
    sigma2 = float((n * s2).sum() / N)
    if not (tau2 > 0 and sigma2 > 0):
        return None
    return sigma2, tau2


def eb_lambda_hetero(values: np.ndarray, keys: np.ndarray) -> float | None:
    """λ = σ̄² / τ²（eb_variance_components の値。v2）。組どうしの差が偶然の範囲なら None。"""
    vc = eb_variance_components(values, keys)
    return None if vc is None else vc[0] / vc[1]


def shrunk_prior_scores(df: pd.DataFrame, keys: list[str], value: str, *, first_year: int,
                        lam_fn=eb_lambda) -> np.ndarray:
    """各行の点数 = (S − n·μ) / (n + λ_Y)。

    S・n はその行の日より前の同じ組の value の合計と件数、μ はその行の日より前の全体の平均。
    λ_Y は行の年 Y の前年末までの行で lam_fn により推定する（v1 は eb_lambda、v2 は eb_lambda_hetero。
    既定を旧のままにしているのは、v1 の記録を同じ呼び方で再現できるようにするため）。
    λ が無い年（組の差が偶然の範囲）と n = 0 の行は 0（平均並み）。first_year より前の年の行は NaN（助走）。
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
        lam = lam_fn(vals[prior], keyarr[prior])
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
