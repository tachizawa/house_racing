"""履歴の要素（仕様 §3.2 #9, 10, 12）。その馬のその日より前の走だけを使う。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _prior_mean(R: pd.DataFrame, same) -> np.ndarray:
    """各行について、同じ馬のその日より前の走のうち same(prev, i) を満たすものの「1 − 着順の割合」の平均。"""
    out = np.full(len(R), np.nan)
    val = 1.0 - R["fin_pct"].to_numpy(float)
    dates = R["date"].to_numpy()
    for idx in R.groupby("ketto_num", sort=False).indices.values():
        idx = idx[np.argsort(dates[idx], kind="stable")]
        for j in range(1, len(idx)):
            i = idx[j]
            prev = idx[:j]
            prev = prev[dates[prev] < dates[i]]
            if len(prev) == 0:
                continue
            v = val[prev][same(prev, i)]
            v = v[~np.isnan(v)]
            if len(v):
                out[i] = float(v.mean())
    return out


def distance_fit(R: pd.DataFrame) -> np.ndarray:
    """同じ芝ダで今回の距離 ±200m の過去走の「1 − 着順の割合」の平均。"""
    kind, kyori = R["track_kind"].to_numpy(), R["kyori"].to_numpy()
    return _prior_mean(R, lambda prev, i: (kind[prev] == kind[i]) & (np.abs(kyori[prev] - kyori[i]) <= 200))


def course_fit(R: pd.DataFrame) -> np.ndarray:
    """同じ競馬場・同じ芝ダの過去走の同じ値。"""
    kind, jyo = R["track_kind"].to_numpy(), R["jyo"].to_numpy()
    return _prior_mean(R, lambda prev, i: (kind[prev] == kind[i]) & (jyo[prev] == jyo[i]))


def weight_carried(R: pd.DataFrame) -> np.ndarray:
    """−（斤量 − そのレースの平均斤量）。軽いほど高い。"""
    return -(R["futan"] - R.groupby("race_id")["futan"].transform("mean")).to_numpy(float)
