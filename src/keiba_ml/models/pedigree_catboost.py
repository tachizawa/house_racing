"""血統モデル（CatBoost）。中身は models/catboost_model.py の共通の包み。

以前のコードとテストが ``PedigreeCatBoost`` と ``MODEL_NAME`` をここから import するので残す。
"""

from __future__ import annotations

from keiba_ml.config import ModelConfig
from keiba_ml.models.catboost_model import (  # noqa: F401  （以前の import 先を保つ）
    RANKING_LOSSES, CatBoostModel, _target, is_ranking,
)

MODEL_NAME = "pedigree_catboost"


class PedigreeCatBoost(CatBoostModel):
    def __init__(self, config: ModelConfig, *, model_version: str,
                 model_name: str = MODEL_NAME) -> None:
        super().__init__(config, model_name=model_name, model_version=model_version)
