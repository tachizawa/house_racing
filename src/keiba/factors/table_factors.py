"""表の要素（仕様 §3.2 #3〜5, 11, 13〜15）。

どれも「その日より前」の A−E（実際の3着内 − 人気から見込む3着内率）を組ごとに縮小平均したもの。
人気（市場）が既に知っている分を差し引いた「上乗せ」なので、市場への上乗せの評価と向きがそろう。
枠順だけは、同じコースの3着内率からのずれを使う（人気は枠をあまり織り込まないため、仕様どおり）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.cumulative import prior_day_sums, shrunk_prior_scores
from keiba.factors.data import REGION

FIRST_YEAR = 2016   # 2015年は助走（仕様 §5.1）
LAYOFF_EDGES = [-1, 13, 27, 55, 118, 364, 10 ** 6]
LAYOFF_LABELS = ["〜13", "14〜27", "28〜55", "56〜118", "119〜364", "365〜"]
WEIGHT_EDGES = [-1000, -10, -4, 3, 9, 1000]
WEIGHT_LABELS = ["〜−10", "−9〜−4", "−3〜+3", "+4〜+9", "+10〜"]


def _by(R: pd.DataFrame, key: np.ndarray, value: str = "ae") -> np.ndarray:
    return shrunk_prior_scores(R.assign(_k=key), ["_k"], value, first_year=FIRST_YEAR)


def jockey(R: pd.DataFrame) -> np.ndarray:
    return _by(R, R["kisyu"].astype(str).to_numpy())


def trainer(R: pd.DataFrame) -> np.ndarray:
    return _by(R, R["chokyosi"].astype(str).to_numpy())


def sire_surface(R: pd.DataFrame) -> np.ndarray:
    return _by(R, (R["sire_num"].astype(str) + ":" + R["track_kind"]).to_numpy())


def travel(R: pd.DataFrame) -> np.ndarray:
    base = R["tozai"].map({"1": "美浦", "2": "栗東"}).fillna("その他")
    return _by(R, (base + "→" + R["jyo"].map(REGION).fillna("その他")).to_numpy())


def layoff_bin(R: pd.DataFrame) -> np.ndarray:
    """前走（同じ馬の前の行）からの日数の区分。

    前の行が無い行は、表の最初の日から365日より後なら「初出走」。365日以内なら、表の始まりより前に
    走っていたかもしれず初出走と区別できないので「不明」（初出走の区分を汚さないため別にする）。
    """
    order = np.lexsort((R["date"].to_numpy(), R["ketto_num"].to_numpy()))
    d = R.iloc[order]
    prev = d.groupby("ketto_num", sort=False)["date"].shift(1)
    gap = (d["date"] - prev).dt.days
    first_label = np.where((d["date"] - R["date"].min()).dt.days > 365, "初出走", "不明")
    lab = pd.cut(gap, bins=LAYOFF_EDGES, labels=LAYOFF_LABELS).astype(object)
    lab = lab.where(gap.notna(), pd.Series(first_label, index=d.index))
    out = np.empty(len(R), dtype=object)
    out[order] = lab.to_numpy()
    return out


def layoff(R: pd.DataFrame) -> np.ndarray:
    return _by(R, layoff_bin(R))


def weight_bin(R: pd.DataFrame) -> np.ndarray:
    z = R["zogen"]
    lab = pd.cut(z, bins=WEIGHT_EDGES, labels=WEIGHT_LABELS).astype(object)
    return lab.where(z.notna(), "不明").to_numpy()


def body_weight(R: pd.DataFrame) -> np.ndarray:
    return _by(R, weight_bin(R))


def draw(R: pd.DataFrame) -> np.ndarray:
    """（競馬場, track_cd, 距離, 枠番）の3着内率の、同じコース全体からのずれの縮小値。"""
    course = (R["jyo"].astype(str) + "|" + R["track_cd"].astype(str) + "|" + R["kyori"].astype(str)).to_numpy()
    d = R.assign(_c=course, _w=R["waku"].fillna(-1).astype(int).astype(str))
    S, n = prior_day_sums(d, ["_c"], "top3")
    course_mean = np.divide(S, n, out=np.full(len(d), np.nan), where=n > 0)
    d["_dev"] = d["top3"].to_numpy(float) - course_mean
    return shrunk_prior_scores(d, ["_c", "_w"], "_dev", first_year=FIRST_YEAR)
