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
from keiba_ml.features import form, form_extra
from keiba_ml.features import pedigree as pedigree_features
from keiba_ml.features.base import FeatureSet, context_columns

#: 距離適性の特徴量のうち、総合モデルに入れる列の接頭辞（条件の列は血統側と重なるので除く）。
DISTANCE_PREFIXES: tuple[str, ...] = ("own_", "dist_", "ped_")
#: 馬体重から作る列（発走の約1時間前に発表。前日の出馬表には無い）。CombinedConfig.use_weight。
WEIGHT_COLUMNS: tuple[str, ...] = ("c_weight", "c_zogen", "x_weight_vs_mean")


def build_features(df: pd.DataFrame, config: MLConfig) -> FeatureSet:
    c = config.combined
    # prepare() は補助列を足した表のコピー。血統側と人の特徴量で共有し、1回だけ作る
    prepared = pedigree_features.prepare(df)
    if c.use_pedigree:
        base = pedigree_features.build_features(df, pedigree=config.pedigree,
                                                features=config.features, prepared=prepared)
        frame, cats, nums = base.frame, list(base.cat_features), list(base.num_features)
    else:
        frame, cats, nums = context_columns(prepared, config.features.context)
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
    extra = []
    if c.extra_form:
        extra.append(form_extra.extra_form_features(
            df, frame[list(form_extra.BASE_COLUMNS)], standard_years=c.speed_standard_years))
    if c.race_level:
        extra.append(form_extra.race_level_features(df, standard_years=c.speed_standard_years))
    for part in extra:
        frame = frame.join(part)
        nums.extend(part.columns)
    if not c.use_weight:
        drop = [col for col in WEIGHT_COLUMNS if col in frame.columns]
        frame = frame.drop(columns=drop)
        nums = [n for n in nums if n not in drop]
    return FeatureSet(frame=frame, cat_features=tuple(cats), num_features=tuple(nums))
