"""総合モデル（combined）の特徴量: 血統 + 距離適性 + 本人の過去走・タイム指数・人・直前の条件。

予想屋がジャンルごとに見るもの（血統・距離・近走・人・条件）を1つの表に並べる。それぞれの作り方は
features/pedigree.py・features/distance.py・features/form.py で、ここでは束ねるだけ。
**オッズ・人気は入れない。** すべて予想するレースより前の情報（馬体重と増減は発走の約1時間前に
発表される値）。
"""

from __future__ import annotations

import pandas as pd

from keiba_ml.config import MLConfig
from keiba_ml.features import distance as distance_features
from keiba_ml.features import form
from keiba_ml.features import pedigree as pedigree_features
from keiba_ml.features.base import FeatureSet, context_columns

#: 距離適性の特徴量のうち、総合モデルに入れる列の接頭辞（条件の列は血統側と重なるので除く）。
DISTANCE_PREFIXES: tuple[str, ...] = ("own_", "dist_", "ped_")


def build_features(df: pd.DataFrame, config: MLConfig) -> FeatureSet:
    c = config.combined
    if c.use_pedigree:
        base = pedigree_features.build_features(df, pedigree=config.pedigree,
                                                features=config.features)
        frame, cats, nums = base.frame, list(base.cat_features), list(base.num_features)
    else:
        prepared = pedigree_features.prepare(df)
        frame, cats, nums = context_columns(prepared, config.features.context)
    prepared = pedigree_features.prepare(df)
    speed = form.speed_figures(df, standard_years=c.speed_standard_years)
    parts = [
        form.own_history_features(df),
        speed,
        form.people_features(prepared, shrinkage=c.people_shrinkage),
        form.condition_features(df),
    ]
    if c.opponent:
        parts.append(form.opponent_strength(df, speed["s_fig_best"]))
    if c.recent_days:
        parts.append(form.recent_people_features(prepared, days=c.recent_days,
                                                 shrinkage=c.recent_shrinkage))
    if c.use_distance:
        dist = distance_features.build_features(df, config.distance)
        cols = [n for n in dist.num_features if n.startswith(DISTANCE_PREFIXES)]
        parts.append(dist.frame[cols].add_prefix("d_").astype("float32"))
    for part in parts:
        frame = frame.join(part)
        nums.extend(part.columns)
    relative = form.relative_features(frame, df["race_id"])
    frame = frame.join(relative)
    nums.extend(relative.columns)
    return FeatureSet(frame=frame, cat_features=tuple(cats), num_features=tuple(nums))
