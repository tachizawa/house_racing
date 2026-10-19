"""どの個別モデルの特徴量にも共通する型と、条件（レースの条件・馬の性齢）の列。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

MISSING = "NA"

#: 条件の列のうちカテゴリとして渡すもの。それ以外は数値。
CATEGORICAL_CONTEXT = frozenset({"track_kind", "distance_band", "baba_cd", "jyo_cd", "sex_cd"})


@dataclass(frozen=True)
class FeatureSet:
    """1行 = 1頭 × 1レースの特徴量表。結果の列は1つも含まない。"""

    frame: pd.DataFrame
    cat_features: tuple[str, ...]
    num_features: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return [*self.cat_features, *self.num_features]


def context_columns(df: pd.DataFrame, names: Sequence[str]
                    ) -> tuple[pd.DataFrame, list[str], list[str]]:
    """条件の列を (表, カテゴリの列名, 数値の列名) で返す。"""
    frame = pd.DataFrame(index=df.index)
    cats: list[str] = []
    nums: list[str] = []
    for name in names:
        if name in CATEGORICAL_CONTEXT:
            frame[name] = df[name].fillna(MISSING).replace("", MISSING).astype(str)
            cats.append(name)
        else:
            frame[name] = pd.to_numeric(df[name], errors="coerce").astype(float)
            nums.append(name)
    return frame, cats, nums
