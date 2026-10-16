"""血統モデル（CatBoost）。

- 目的関数は設定（ModelConfig.loss_function）で切り替える。未決定事項なので、
  Logloss（3着以内の2値・確率が出る）と順位学習（YetiRank など。group_id=race_id）を
  同じ入口で比較できるようにしてある。
- 検証期間で early stopping する。テスト期間は学習にも早期終了の判定にも使わない。
- catboost は ``requirements-ml.txt`` の依存。既存モデルのテストを catboost の有無に
  左右させないよう、このモジュールの中でだけ import する。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_ml.config import ModelConfig
from keiba_ml.features.pedigree import FeatureSet
from keiba_ml.models.base import make_predictions

MODEL_NAME = "pedigree_catboost"

#: レース単位でまとめて学習する（group_id が要る）目的関数。
RANKING_LOSSES = frozenset({
    "YetiRank", "YetiRankPairwise", "PairLogit", "PairLogitPairwise",
    "QueryRMSE", "QuerySoftMax", "QueryCrossEntropy", "StochasticRank",
})


def _catboost():
    try:
        import catboost
    except ImportError as error:
        raise ImportError(
            "catboost がインストールされていません。"
            " .venv/Scripts/python.exe -m pip install -r requirements-ml.txt"
            " を実行してください。") from error
    return catboost


def is_ranking(loss_function: str) -> bool:
    return loss_function.split(":")[0] in RANKING_LOSSES


def _target(rows: pd.DataFrame, loss_function: str) -> np.ndarray:
    """2値なら 3着以内。順位学習なら関連度（1着 3 / 2着 2 / 3着 1 / それ以外 0）。

    QueryCrossEntropy は 0〜1 の目的変数を取るので 2値のまま渡す。
    """
    name = loss_function.split(":")[0]
    if not is_ranking(name) or name == "QueryCrossEntropy":
        return rows["target_top3"].to_numpy(dtype=float)
    rank = rows["result_rank"].astype("Float64").fillna(99).to_numpy(dtype=float)
    return np.where(rank <= 3, 4.0 - rank, 0.0)


class PedigreeCatBoost:
    def __init__(self, config: ModelConfig, *, model_version: str) -> None:
        self.config = config
        self.model_version = model_version
        self.model = None
        self.feature_columns: list[str] = []
        self.cat_features: list[str] = []

    def _pool(self, features: FeatureSet, rows: pd.DataFrame, *, with_label: bool):
        catboost = _catboost()
        order = rows.sort_values(["race_id", "umaban"], kind="mergesort").index
        X = features.frame.loc[order, self.feature_columns]
        kwargs = {"cat_features": self.cat_features}
        if with_label:
            kwargs["label"] = _target(rows.loc[order], self.config.loss_function)
        if is_ranking(self.config.loss_function):
            kwargs["group_id"] = rows.loc[order, "race_id"].to_numpy()
        return catboost.Pool(X, **kwargs), order

    def fit(self, features: FeatureSet, train: pd.DataFrame, valid: pd.DataFrame,
            *, log_period: int = 200) -> "PedigreeCatBoost":
        catboost = _catboost()
        self.feature_columns = features.columns
        self.cat_features = list(features.cat_features)
        train_pool, _ = self._pool(features, train, with_label=True)
        valid_pool, _ = self._pool(features, valid, with_label=True)
        params = {
            "loss_function": self.config.loss_function,
            "iterations": self.config.iterations,
            "learning_rate": self.config.learning_rate,
            "depth": self.config.depth,
            "l2_leaf_reg": self.config.l2_leaf_reg,
            "random_seed": self.config.random_seed,
            "thread_count": self.config.thread_count,
            "od_type": "Iter",
            "od_wait": self.config.early_stopping_rounds,
            "use_best_model": True,
            # 学習データは race_id（先頭8桁が日付）順に並べてある。has_time=True で
            # CatBoost の ordered target statistics が「並び順 = 時間」を使い、
            # 祖先カテゴリの数値化に後のレースの結果を使わなくなる。
            "has_time": True,
            "allow_writing_files": False,
            **self.config.extra,
        }
        if "eval_metric" not in params:
            params["eval_metric"] = ("NDCG:top=3" if is_ranking(self.config.loss_function)
                                     else "Logloss")
        model_cls = catboost.CatBoostRanker if is_ranking(self.config.loss_function) \
            else catboost.CatBoostClassifier
        self.model = model_cls(**params)
        self.model.fit(train_pool, eval_set=valid_pool, verbose=log_period)
        return self

    def predict(self, features: FeatureSet, rows: pd.DataFrame, *, fold: str) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit() か load() の後に呼んでください。")
        pool, order = self._pool(features, rows, with_label=False)
        ordered = rows.loc[order]
        if is_ranking(self.config.loss_function):
            score = self.model.predict(pool)
            prob = None
        else:
            prob = self.model.predict_proba(pool)[:, 1]
            score = prob
        return make_predictions(ordered, score, model_name=MODEL_NAME,
                                model_version=self.model_version, fold=fold,
                                prob_top3=prob)

    def feature_importance(self) -> pd.DataFrame:
        # 既定の種類は順位学習だと学習データを要求する（LossFunctionChange）。
        # どの目的関数でもデータ無しで出せる PredictionValuesChange にそろえる。
        values = self.model.get_feature_importance(type="PredictionValuesChange")
        return (pd.DataFrame({"feature": self.feature_columns, "importance": values})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    @property
    def best_iteration(self) -> int | None:
        return None if self.model is None else self.model.get_best_iteration()

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(directory / "model.cbm"))
        meta = {
            "model_name": MODEL_NAME, "model_version": self.model_version,
            "feature_columns": self.feature_columns, "cat_features": self.cat_features,
            "model_config": asdict(self.config), "best_iteration": self.best_iteration,
        }
        (directory / "model_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "PedigreeCatBoost":
        catboost = _catboost()
        directory = Path(directory)
        meta = json.loads((directory / "model_meta.json").read_text(encoding="utf-8"))
        config = ModelConfig(**meta["model_config"])
        obj = cls(config, model_version=meta["model_version"])
        model_cls = catboost.CatBoostRanker if is_ranking(config.loss_function) \
            else catboost.CatBoostClassifier
        obj.model = model_cls()
        obj.model.load_model(str(directory / "model.cbm"))
        obj.feature_columns = meta["feature_columns"]
        obj.cat_features = meta["cat_features"]
        return obj
