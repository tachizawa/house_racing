"""15要素の一覧（仕様 §3.2）と一括計算。"""

from __future__ import annotations

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
    FactorSpec("jockey", "v1", "騎手", True),
    FactorSpec("trainer", "v1", "調教師", True),
    FactorSpec("sire_surface", "v1", "父の芝ダ適性", True),
    FactorSpec("late_age", "v1", "晩成度×年齢", True),
    FactorSpec("age", "v1", "年齢", True),
    FactorSpec("cohort_level", "v1", "世代レベル", True),
    FactorSpec("distance_fit", "v1", "距離適性", True),
    FactorSpec("course_fit", "v1", "コース適性", True),
    FactorSpec("draw", "v1", "枠順", True),
    FactorSpec("weight_carried", "v1", "斤量", True),
    FactorSpec("layoff", "v1", "間隔", True),
    FactorSpec("body_weight", "v1", "馬体重", False),
    FactorSpec("travel", "v1", "遠征", True),
]
FACTOR_NAMES = [f.name for f in FACTORS]
_WHOLE = {"jockey": tf.jockey, "trainer": tf.trainer, "sire_surface": tf.sire_surface, "draw": tf.draw,
          "layoff": tf.layoff, "body_weight": tf.body_weight, "travel": tf.travel,
          "distance_fit": hf.distance_fit, "course_fit": hf.course_fit, "weight_carried": hf.weight_carried}
_YEARLY = {"time_figure": time_figure_scores, "opponents": opponents_scores, "age": age_scores,
           "late_age": late_age_scores, "cohort_level": cohort_level_scores}


def compute_all(R: pd.DataFrame, runs_all: pd.DataFrame, *, years, log=print) -> pd.DataFrame:
    """R の写しに e3・ae と15要素の列を足す。年ごとの要素は years の年だけ計算し、ほかの年は NaN。"""
    out = add_expected_top3(R)
    for name, fn in _WHOLE.items():
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
