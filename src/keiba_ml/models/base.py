"""個別モデルの共通の予測出力形式（Phase 4「共通形式に統一」の先取り）。

どの独立モデル（血統・距離・コース …）も、予測を次の列を持つ表で返す。
統合モデル（スタッキング・重み付きアンサンブル・Jev など、未決定）は
この表だけを受け取れば組み合わせられる。

| 列 | 意味 |
|---|---|
| race_id, umaban, ketto_num | 行の識別 |
| model_name, model_version | どのモデルの・どの学習の予測か |
| score | 大きいほど上位。確率とは限らない |
| prob_top3 | 3着以内の確率（Logloss で学習したときだけ。順位学習では NaN） |
| pred_rank | レース内の予測順位（1 始まり。score の降順、同点は馬番順） |
| fold | "test" / "valid" / "oof:<年>" など。統合モデルの学習には oof だけを使う |
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREDICTION_COLUMNS: tuple[str, ...] = (
    "race_id", "umaban", "ketto_num", "model_name", "model_version",
    "score", "prob_top3", "pred_rank", "fold",
)


def rank_within_race(frame: pd.DataFrame, score_col: str = "score") -> pd.Series:
    """レース内の順位。score 降順、同点は馬番の若い順（決定的にするため）。"""
    ordered = frame.sort_values(["race_id", score_col, "umaban"],
                                ascending=[True, False, True], kind="mergesort")
    ranks = ordered.groupby("race_id").cumcount() + 1
    return ranks.reindex(frame.index)


def make_predictions(rows: pd.DataFrame, score: np.ndarray, *, model_name: str,
                     model_version: str, fold: str,
                     prob_top3: np.ndarray | None = None) -> pd.DataFrame:
    out = rows[["race_id", "umaban", "ketto_num"]].copy()
    out["model_name"] = model_name
    out["model_version"] = model_version
    out["score"] = np.asarray(score, dtype=float)
    out["prob_top3"] = np.nan if prob_top3 is None else np.asarray(prob_top3, dtype=float)
    out["pred_rank"] = rank_within_race(out)
    out["fold"] = fold
    return out[list(PREDICTION_COLUMNS)]
