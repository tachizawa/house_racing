"""段ごとの条件付きロジット（仕様 §4）。段 s では、まだ着順の決まっていない馬の中から s 着の馬が選ばれる確率を
softmax(w_s · x) とする。スプリンターズS の combine.py と同じ形。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

STAGES = (1, 2, 3)


def exclusion_reasons(R: pd.DataFrame) -> pd.DataFrame:
    """race_id → 除く理由（重なりあり）。レポートで除いた件数を理由ごとに示すため。

    - dead_heat: 1・2・3着のどれかが1頭でない（同着、または該当の着順の馬がいない）
    - bad_odds: 単勝オッズが無い・0以下の馬がいる
    - small_field: 4頭未満
    """
    g = R.groupby("race_id", sort=False)
    counts = pd.concat({k: R["jyuni"].eq(k).groupby(R["race_id"], sort=False).sum() for k in STAGES}, axis=1)
    return pd.DataFrame({
        "dead_heat": (counts != 1).any(axis=1),
        "bad_odds": (~(R["odds"] > 0)).groupby(R["race_id"], sort=False).any(),
        "small_field": g["field"].first() < 4,
    })


def valid_race_mask(R: pd.DataFrame) -> np.ndarray:
    """評価に使える行: 単勝オッズが全頭正、1・2・3着が各1頭、4頭以上のレース（exclusion_reasons のどれにも当たらない）。"""
    bad = exclusion_reasons(R).any(axis=1)
    return (~R["race_id"].map(bad).astype(bool)).to_numpy()


def add_market_features(R: pd.DataFrame) -> pd.DataFrame:
    """lq = log(レース内で正規化した 1/単勝オッズ)、lq2 = lq の2乗 を足す。"""
    inv = 1.0 / R["odds"]
    q = inv / inv.groupby(R["race_id"]).transform("sum")
    lq = np.log(q)
    return R.assign(lq=lq, lq2=lq ** 2)


def _stage_rows(d: pd.DataFrame, s: int) -> pd.DataFrame:
    f = d[~d["jyuni"].between(1, s - 1)] if s > 1 else d
    return f.assign(_y=(f["jyuni"] == s).astype(float))


def _codes(race_id: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    codes = pd.factorize(race_id)[0]
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    return codes, starts, int(codes.max()) + 1


def _nll_grad(w, X, y, codes, starts, n_r):
    z = X @ w
    zmax = np.maximum.reduceat(z, starts)[codes]
    e = np.exp(z - zmax)
    den = np.bincount(codes, e, n_r)
    p = e / den[codes]
    ll = -(y * (z - zmax - np.log(den[codes]))).sum()
    grad = -(X * (y - p)[:, None]).sum(0)
    return ll, grad


def fit_stages(d: pd.DataFrame, feats: list[str]) -> list[np.ndarray]:
    """段1〜3の重みを最尤推定する。d は valid な行だけで、race_id ごとにまとまって並ぶこと。"""
    weights = []
    for s in STAGES:
        f = _stage_rows(d, s)
        codes, starts, n_r = _codes(f["race_id"])
        X = f[feats].to_numpy(float)
        r = minimize(_nll_grad, np.zeros(len(feats)), args=(X, f["_y"].to_numpy(), codes, starts, n_r),
                     jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-8})
        weights.append(r.x)
    return weights


def race_nll(d: pd.DataFrame, feats: list[str], weights: list[np.ndarray]) -> pd.Series:
    """race_id → 段1〜3 の −対数尤度の合計。"""
    total = None
    for s, w in zip(STAGES, weights):
        f = _stage_rows(d, s)
        codes, starts, n_r = _codes(f["race_id"])
        z = f[feats].to_numpy(float) @ w
        zmax = np.maximum.reduceat(z, starts)
        lse = np.log(np.bincount(codes, np.exp(z - zmax[codes]), n_r)) + zmax
        zt = np.bincount(codes, z * f["_y"].to_numpy(), n_r)
        part = pd.Series(lse - zt, index=pd.unique(f["race_id"]))
        total = part if total is None else total.add(part, fill_value=0.0)
    return total


def _order_dist(etas: list[np.ndarray]) -> np.ndarray:
    e = [np.exp(x - x.max()) for x in etas]
    p1 = e[0] / e[0].sum()
    P2 = e[1][None, :] / (e[1].sum() - e[1])[:, None]
    np.fill_diagonal(P2, 0)
    d3 = e[2].sum() - e[2][:, None] - e[2][None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        P3 = e[2][None, None, :] / d3[:, :, None]
    idx = np.arange(len(p1))
    P3[idx, idx, :] = 0
    P3[:, idx, idx] = 0
    P3[idx, :, idx] = 0
    return p1[:, None, None] * P2[:, :, None] * np.nan_to_num(P3)


def place_probabilities(d: pd.DataFrame, feats: list[str], weights: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """行ごとの1着確率と3着内確率（d の行順に揃えて返す）。"""
    win = np.zeros(len(d))
    top3 = np.zeros(len(d))
    X = d[feats].to_numpy(float)
    for idx in d.groupby("race_id", sort=False).indices.values():
        t = _order_dist([X[idx] @ w for w in weights])
        win[idx] = t.sum((1, 2))
        top3[idx] = t.sum((1, 2)) + t.sum((0, 2)) + t.sum((0, 1))
    return win, top3
