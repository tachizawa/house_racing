"""個別モデルの登録表。

学習・予想・評価の手順（experiment.py と scripts/ml/train.py ほか）はどのモデルでも同じで、
モデルごとに違うものだけをここに ``ModelSpec`` として集める:

| 項目 | 中身 |
|---|---|
| name | 設定ファイル（configs/ml/<name>.toml）と出力（data/ml/runs/<name>/）の名前 |
| model_name | 予測の表（models/base.py の共通形式）に入るモデル名 |
| build_features | データセットの行 → 特徴量表（as-of。結果の列を含まない） |
| stats_history | 履歴を学習期間に限るか（"window"）全期間か（"all"） |
| baselines | 学習しない比較対象（学習したモデルがこれを超えないなら学習の意味がない） |
| asof_checks | 未来情報が入っていないことの検算（学習・評価の前に必ず通す） |
| required_columns | データセットに要る列（古いデータセットで学習しようとしたら止める） |
| describe | 予想の表に並べる根拠の列 |

**新しいモデルを足す手順**（docs/ml/README.md §9 にも同じ内容）:

1. ``features/<name>.py`` に ``build_features`` を書く（as-of・結果の列を返さない）
2. ``config.py`` に ``<Name>Config`` の節を足し、``MLConfig`` と ``_SECTIONS``、``MODEL_KINDS`` に登録する
3. ``configs/ml/<name>.toml`` を作り、``[experiment] model = "<name>"`` を書く
4. ここに ``ModelSpec`` を1つ足す（基準・検算・表示も）
5. ``tests/ml/`` に合成データでの検算・端から端までのテストを足す
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from keiba_ml.config import MODEL_KINDS, MLConfig
from keiba_ml.dataset import integrity
from keiba_ml.features import distance as distance_features
from keiba_ml.features import pedigree as pedigree_features
from keiba_ml.features.base import FeatureSet
from keiba_ml.models.base import make_predictions


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_name: str
    title: str
    build_features: Callable[[pd.DataFrame, MLConfig], FeatureSet]
    stats_history: Callable[[MLConfig], str]
    baselines: Callable[[FeatureSet, pd.DataFrame, str], dict[str, pd.DataFrame]]
    asof_checks: Callable[[pd.DataFrame, FeatureSet, MLConfig, pd.Index | None], list[dict]]
    required_columns: tuple[str, ...]
    describe: Callable[[pd.DataFrame, FeatureSet, dict[str, str]], pd.DataFrame]

    def missing_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.required_columns if c not in df.columns]


def _heuristics(features: FeatureSet, rows: pd.DataFrame, fold: str,
                columns: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    return {name: make_predictions(rows, score.to_numpy(), model_name=name,
                                   model_version="heuristic", fold=fold)
            for name, score in columns.items()}


# --- 血統 -------------------------------------------------------------------

def _pedigree_features(df: pd.DataFrame, config: MLConfig) -> FeatureSet:
    return pedigree_features.build_features(df, pedigree=config.pedigree,
                                            features=config.features)


def _pedigree_baselines(features: FeatureSet, rows: pd.DataFrame,
                        fold: str) -> dict[str, pd.DataFrame]:
    """sire_ae: 父の産駒の as-of 3着以内 A/E だけ。sire_cond_ae: 父 × 芝ダ × 距離区分 の A/E だけ。"""
    frame = features.frame.loc[rows.index]
    columns = {name: frame[col] for name, col in
               (("sire_ae", "st_S_ae_top3"), ("sire_cond_ae", "st_S_ae_top3_surface_dband"))
               if col in frame}
    return _heuristics(features, rows, fold, columns)


def _pedigree_checks(history: pd.DataFrame, features: FeatureSet, config: MLConfig,
                     only: pd.Index | None) -> list[dict]:
    p = config.pedigree
    return [integrity.check_asof_samples(history, features.frame, paths=p.stat_paths,
                                         shrinkage=p.shrinkage, exclude_self=p.exclude_self,
                                         sample=100, only=only)]


def _pedigree_describe(rows: pd.DataFrame, features: FeatureSet,
                       names: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"父": rows["ped_S"].map(names).fillna(""),
                         "母父": rows["ped_DS"].map(names).fillna("")}, index=rows.index)


# --- 距離適性 ---------------------------------------------------------------

def _distance_features(df: pd.DataFrame, config: MLConfig) -> FeatureSet:
    return distance_features.build_features(df, config.distance)


def _distance_baselines(features: FeatureSet, rows: pd.DataFrame,
                        fold: str) -> dict[str, pd.DataFrame]:
    """学習しない距離の目安。

    - dist_rel_band: 本人の「芝ダ × 距離帯の走り − 同じ馬場の普段の走り」だけで並べる
    - dist_gap: 今回の距離が平均出走距離に近いほど上（陣営が選んできた距離 = 合う距離とみなす）。
      過去走の無い馬はそのレースの中央値で埋める
    """
    frame = features.frame.loc[rows.index]
    rel = frame["own_rel_pf_surface_dband"].astype(float).fillna(0.0)
    gap = -frame["dist_minus_mean"].astype(float).abs()
    gap = gap.fillna(gap.groupby(rows["race_id"]).transform("median")).fillna(0.0)
    return _heuristics(features, rows, fold, {"dist_rel_band": rel, "dist_gap": gap})


def _distance_checks(history: pd.DataFrame, features: FeatureSet, config: MLConfig,
                     only: pd.Index | None) -> list[dict]:
    return [integrity.check_distance_asof_sample(history, features.frame,
                                                 config=config.distance, sample=100,
                                                 only=only)]


def _distance_describe(rows: pd.DataFrame, features: FeatureSet,
                       names: dict[str, str]) -> pd.DataFrame:
    frame = features.frame.loc[rows.index]
    km = pd.to_numeric(rows["kyori_m"], errors="coerce").astype(float)

    def back(col: str) -> pd.Series:
        """「今回の距離 − X」から X（m）に戻す。"""
        return (km - frame[col].astype(float)).round(0).astype("Int64")

    return pd.DataFrame({
        "前走距離": back("dist_minus_last"),
        "平均距離": back("dist_minus_mean"),
        "好走距離": back("dist_minus_top3_mean"),
        "同距離帯の出走": np.expm1(frame["own_log_runs_surface_dband"].astype(float))
        .round(0).astype("Int64"),
        "父": rows["ped_S"].map(names).fillna("") if "ped_S" in rows else "",
    }, index=rows.index)


MODELS: dict[str, ModelSpec] = {
    "pedigree": ModelSpec(
        name="pedigree", model_name="pedigree_catboost", title="血統",
        build_features=_pedigree_features,
        stats_history=lambda config: config.pedigree.stats_history,
        baselines=_pedigree_baselines, asof_checks=_pedigree_checks,
        required_columns=(), describe=_pedigree_describe),
    "distance": ModelSpec(
        name="distance", model_name="distance_catboost", title="距離適性",
        build_features=_distance_features,
        stats_history=lambda config: config.distance.stats_history,
        baselines=_distance_baselines, asof_checks=_distance_checks,
        required_columns=("result_margin_sec", "result_weight_kg"),
        describe=_distance_describe),
}

assert tuple(MODELS) == MODEL_KINDS, "config.MODEL_KINDS と registry.MODELS を揃えてください"


def get(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"登録されていないモデルです: {name!r}（{list(MODELS)}）")
    return MODELS[name]


def for_config(config: MLConfig) -> ModelSpec:
    return get(config.experiment.model)
