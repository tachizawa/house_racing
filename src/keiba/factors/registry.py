"""15要素の一覧（仕様 §3.2）と一括計算。

表の要素（騎手・調教師・父の芝ダ・枠順・間隔・馬体重・遠征）は v2（縮小の強さ λ の推定を組ごとの散らばりを
引く形に直した版。table_factors の docstring）。ほかの要素は縮小を使わないか（履歴の要素）、別の物差しで
作るので v1 のまま。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from keiba.factors import history_factors as hf
from keiba.factors import table_factors as tf
from keiba.factors.age_factors import age_scores, cohort_level_scores, late_age_scores
from keiba.factors.cumulative import add_expected_top3
from keiba.factors.figure_factors import opponents_scores, time_figure_scores, yearly_figures


@dataclass(frozen=True)
class FactorSpec:
    name: str
    version: str
    ja: str
    forward: bool   # 当日の朝に出せるか（馬体重は発走前の発表までわからない）


FACTORS = [
    FactorSpec("time_figure", "v1", "タイム指数", True),
    FactorSpec("opponents", "v1", "相手関係", True),
    FactorSpec("jockey", "v2", "騎手", True),
    FactorSpec("trainer", "v2", "調教師", True),
    FactorSpec("sire_surface", "v2", "父の芝ダ適性", True),
    FactorSpec("late_age", "v1", "晩成度×年齢", True),
    FactorSpec("age", "v1", "年齢", True),
    FactorSpec("cohort_level", "v1", "世代レベル", True),
    FactorSpec("distance_fit", "v1", "距離適性", True),
    FactorSpec("course_fit", "v1", "コース適性", True),
    FactorSpec("draw", "v2", "枠順", True),
    FactorSpec("weight_carried", "v1", "斤量", True),
    FactorSpec("layoff", "v2", "間隔", True),
    FactorSpec("body_weight", "v2", "馬体重", False),
    FactorSpec("travel", "v2", "遠征", True),
]
FACTOR_NAMES = [f.name for f in FACTORS]
VERSIONS = {f.name: f.version for f in FACTORS}
_TABLE = tf.TABLE_FACTORS
_HISTORY = {"distance_fit": hf.distance_fit, "course_fit": hf.course_fit, "weight_carried": hf.weight_carried}
_YEARLY = {"time_figure": time_figure_scores, "opponents": opponents_scores, "age": age_scores,
           "late_age": late_age_scores, "cohort_level": cohort_level_scores}


def compute_all(R: pd.DataFrame, runs_all: pd.DataFrame, *, years, log=print) -> pd.DataFrame:
    """R の写しに e3・ae と15要素の列を足す。年ごとの要素は years の年だけ計算し、ほかの年は NaN。"""
    out = add_expected_top3(R)
    out = recompute_table_factors(out, log=log)
    for name, fn in _HISTORY.items():
        log(f"{name} …")
        out[name] = fn(out)
    for name in _YEARLY:
        out[name] = np.nan
    for y in years:
        log(f"{y}年の物差しと年ごとの要素 …")
        fig = yearly_figures(runs_all, y)
        rows = (out["year"] == y).to_numpy()
        for name, fn in _YEARLY.items():
            out.loc[rows, name] = fn(out, fig, y)[rows]
    return out


def recompute_table_factors(R: pd.DataFrame, *, log=print) -> pd.DataFrame:
    """表の要素の列を今の版（VERSIONS）で作り直した写し。R は e3・ae を持つこと。

    表の要素は R の列と e3・ae だけから決まるので、古い版の点数の表（キャッシュ）に対してこれだけ呼べば、
    物差しの推定に30分以上かかる年ごとの要素を作り直さずに版を上げられる。
    """
    out = R.copy()
    for name, fn in _TABLE.items():
        log(f"{name}（{VERSIONS[name]}）…")
        out[name] = fn(out, version=VERSIONS[name])
    return out


def versions_tag(versions: dict[str, str] | None = None) -> str:
    """要素の名前と版の組から作る短い印（キャッシュのファイル名に入れ、版の違う点数の表を取り違えないため）。"""
    v = VERSIONS if versions is None else versions
    text = ",".join(f"{k}:{v[k]}" for k in sorted(v))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
