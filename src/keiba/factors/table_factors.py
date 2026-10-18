"""表の要素（仕様 §3.2 #3〜5, 11, 13〜15）。

どれも「その日より前」の A−E（実際の3着内 − 人気から見込む3着内率）を組ごとに縮小平均したもの。
人気（市場）が既に知っている分を差し引いた「上乗せ」なので、市場への上乗せの評価と向きがそろう。
枠順だけは、同じコースの3着内率からのずれを使う（人気は枠をあまり織り込まないため、仕様どおり）。

版（仕様 §2「縮小」）: v1 は縮小の強さ λ を組の中の散らばりを1つにまとめたモーメント法（eb_lambda）で、
v2 は組ごとの散らばりを引くモーメント法（eb_lambda_hetero）で推定する。違いは λ だけで、組の作り方と点数の
式は同じ。v1 は組ごとの散らばりの違い（A−E は人気馬の多い組ほど大きい）を無視して τ² を小さく出し、λ が
大きすぎて年ごとに物差しがずれていた（2016年は τ² ≤ 0 で全行 0）。v1 の記録を再現できるよう両方残す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.cumulative import eb_lambda, eb_lambda_hetero, prior_day_sums, shrunk_prior_scores
from keiba.factors.data import REGION

FIRST_YEAR = 2016   # 2015年は助走（仕様 §5.1）
LAYOFF_EDGES = [-1, 13, 27, 55, 118, 364, 10 ** 6]
LAYOFF_LABELS = ["〜13", "14〜27", "28〜55", "56〜118", "119〜364", "365〜"]
WEIGHT_EDGES = [-1000, -10, -4, 3, 9, 1000]
WEIGHT_LABELS = ["〜−10", "−9〜−4", "−3〜+3", "+4〜+9", "+10〜"]
ESTIMATORS = {"v1": eb_lambda, "v2": eb_lambda_hetero}
VERSION = "v2"   # 今の版。registry.FACTORS の表の要素の版と同じ


def _by(R: pd.DataFrame, key: np.ndarray, value: str = "ae", *, version: str) -> np.ndarray:
    return shrunk_prior_scores(R.assign(_k=key), ["_k"], value, first_year=FIRST_YEAR, lam_fn=ESTIMATORS[version])


def jockey_key(R: pd.DataFrame) -> np.ndarray:
    return R["kisyu"].astype(str).to_numpy()


def trainer_key(R: pd.DataFrame) -> np.ndarray:
    return R["chokyosi"].astype(str).to_numpy()


def sire_surface_key(R: pd.DataFrame) -> np.ndarray:
    return (R["sire_num"].astype(str) + ":" + R["track_kind"]).to_numpy()


def travel_key(R: pd.DataFrame) -> np.ndarray:
    base = R["tozai"].map({"1": "美浦", "2": "栗東"}).fillna("その他")
    return (base + "→" + R["jyo"].map(REGION).fillna("その他")).to_numpy()


def jockey(R: pd.DataFrame, *, version: str = VERSION) -> np.ndarray:
    return _by(R, jockey_key(R), version=version)


def trainer(R: pd.DataFrame, *, version: str = VERSION) -> np.ndarray:
    return _by(R, trainer_key(R), version=version)


def sire_surface(R: pd.DataFrame, *, version: str = VERSION) -> np.ndarray:
    return _by(R, sire_surface_key(R), version=version)


def travel(R: pd.DataFrame, *, version: str = VERSION) -> np.ndarray:
    return _by(R, travel_key(R), version=version)


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


def layoff(R: pd.DataFrame, *, version: str = VERSION) -> np.ndarray:
    return _by(R, layoff_bin(R), version=version)


def weight_bin(R: pd.DataFrame) -> np.ndarray:
    z = R["zogen"]
    lab = pd.cut(z, bins=WEIGHT_EDGES, labels=WEIGHT_LABELS).astype(object)
    return lab.where(z.notna(), "不明").to_numpy()


def body_weight(R: pd.DataFrame, *, version: str = VERSION) -> np.ndarray:
    return _by(R, weight_bin(R), version=version)


def _draw_frame(R: pd.DataFrame) -> pd.DataFrame:
    """枠順の材料: _c = コース、_w = 枠番、_dev = 3着内 − そのコースのその日より前の3着内率。"""
    course = (R["jyo"].astype(str) + "|" + R["track_cd"].astype(str) + "|" + R["kyori"].astype(str)).to_numpy()
    d = R.assign(_c=course, _w=R["waku"].fillna(-1).astype(int).astype(str))
    S, n = prior_day_sums(d, ["_c"], "top3")
    course_mean = np.divide(S, n, out=np.full(len(d), np.nan), where=n > 0)
    d["_dev"] = d["top3"].to_numpy(float) - course_mean
    return d


def draw(R: pd.DataFrame, *, version: str = VERSION) -> np.ndarray:
    """（競馬場, track_cd, 距離, 枠番）の3着内率の、同じコース全体からのずれの縮小値。"""
    return shrunk_prior_scores(_draw_frame(R), ["_c", "_w"], "_dev", first_year=FIRST_YEAR,
                               lam_fn=ESTIMATORS[version])


TABLE_FACTORS = {"jockey": jockey, "trainer": trainer, "sire_surface": sire_surface, "draw": draw,
                 "layoff": layoff, "body_weight": body_weight, "travel": travel}


def _lambda_inputs(name: str, R: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """λ の推定に渡す (値, 組) の行ごとの配列。点数の計算（shrunk_prior_scores）に渡すものと同じ。"""
    if name == "draw":
        d = _draw_frame(R)
        return d["_dev"].to_numpy(float), (d["_c"].astype(str) + "|" + d["_w"].astype(str)).to_numpy()
    keyfn = {"jockey": jockey_key, "trainer": trainer_key, "sire_surface": sire_surface_key, "travel": travel_key,
             "layoff": layoff_bin, "body_weight": weight_bin}[name]
    return R["ae"].to_numpy(float), np.asarray(keyfn(R)).astype(str)


def lambda_table(R: pd.DataFrame, years, names=tuple(TABLE_FACTORS)) -> pd.DataFrame:
    """表の要素ごと・年ごとの λ（v1 と v2）。年 Y の λ は Y の前年末までの行で推定する（点数の計算と同じ）。

    v2 を作った理由（v1 の λ が大きすぎた）をレポートで数字で示すために使う。None（推定できない）は NaN。
    """
    rows = []
    dates = R["date"].to_numpy()
    for name in names:
        vals, keys = _lambda_inputs(name, R)
        for y in years:
            prior = dates < np.datetime64(f"{int(y)}-01-01")
            row = {"factor": name, "year": int(y)}
            for ver, fn in ESTIMATORS.items():
                lam = fn(vals[prior], keys[prior])
                row[f"lambda_{ver}"] = np.nan if lam is None else lam
            rows.append(row)
    return pd.DataFrame(rows)
