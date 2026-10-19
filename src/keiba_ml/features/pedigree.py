"""血統モデルの特徴量。

入力はデータセット（dataset/build.py の runners）。出力は「1行 = 1頭 × 1レース」の
特徴量表で、結果の列は1つも含まない（integrity.check_feature_columns で検査する）。

特徴量は3種類:

1. **祖先のカテゴリ**（``ped_<位置>``）… 3世代14枠の繁殖登録番号そのもの。
   CatBoost の ordered target statistics が高カーディナリティのカテゴリを
   未来情報なしで数値化する（学習データ内で「その行より前の行」だけを使う）。
2. **祖先の産駒成績（as-of）**（``st_<位置>_...``）… その祖先の産駒が
   **予想対象のレース日より前に**挙げた3着以内数を、期待値（出走頭数から決まる
   偶然の3着以内率）で割った A/E。距離区分・馬場・年齢などの条件別にも出す。
   同じ日のレースは含めない（前日予想の前提）。既定で本人の過去走も除く
   （PedigreeConfig.exclude_self）。
3. **条件**（レースの条件と馬の性齢）と **近交**（近交係数・クロス数）。

早熟・晩成（開発方針の「特に調査する内容」）は 2. の年齢別 A/E
（``st_<位置>_ae_age2`` / ``_age3`` / ``_age4p``）と、その比
``st_<位置>_maturity``（= log(4歳以上の A/E / 2歳の A/E)。正なら晩成寄り）で表す。
晩成・早熟の**定義**は未決定（開発方針 §9）なので、ここでは連続量のまま渡し、
しきい値で区切らない。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from keiba_ml.config import FeatureConfig, PedigreeConfig, all_paths
from keiba_ml.features.base import MISSING, FeatureSet, context_columns

__all__ = ["MISSING", "FeatureSet", "build_features", "expected_top3", "prepare"]

#: 年齢の区分（成長曲線用）。
AGE_BUCKETS: tuple[tuple[str, int, int], ...] = (("2", 2, 2), ("3", 3, 3), ("4p", 4, 99))

#: 条件別の A/E を出すときの条件列。キーは特徴量名の接尾辞。
CONDITIONS: dict[str, tuple[str, ...]] = {
    "surface": ("track_kind",),
    "dband": ("distance_band",),
    "surface_dband": ("track_kind", "distance_band"),
    "going": ("going_class",),
    "age": ("age_bucket",),
    "jyo": ("jyo_cd",),
}

_SUMS = ("n", "k3", "e3", "kw", "ew")


def expected_top3(field_size: pd.Series | np.ndarray) -> np.ndarray:
    """偶然で3着以内に入る確率。出走 n 頭なら min(3, n) / n。"""
    n = np.asarray(field_size, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, np.minimum(3.0, n) / n, np.nan)


def age_bucket(age: pd.Series) -> pd.Series:
    out = pd.Series(MISSING, index=age.index, dtype=object)
    for name, low, high in AGE_BUCKETS:
        out[(age >= low) & (age <= high)] = name
    return out


def going_class(baba: pd.Series) -> pd.Series:
    """良（1）とそれ以外（稍重・重・不良）の2区分。4区分では標本が割れすぎる。"""
    return baba.map({"1": "good", "2": "off", "3": "off", "4": "off"}).fillna(MISSING)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量計算に要る補助列を足す（結果列は足さない）。"""
    df = df.copy()
    df["age_bucket"] = age_bucket(df["age"])
    df["going_class"] = going_class(df["baba_cd"])
    df["_e3"] = expected_top3(df["field_size"])
    df["_ew"] = 1.0 / df["field_size"].astype(float)
    return df


def _asof_sums(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """keys ごとに、各行のレース日**より前**の (出走数, 3着内数, 期待3着内数, 勝数, 期待勝数)。

    1. 結果のある行を (keys, 日付) で集計し、keys 内で日付順に累積する
       （= その日までの合計。その日を含む）。
    2. 各行に、keys が同じで日付が**厳密に前**の最新の累積を merge_asof で付ける
       （allow_exact_matches=False）。同じ日のレースの結果は入らない。

    keys のいずれかが欠損（MISSING / 空）の行は 0 を返す。
    """
    keys = list(keys)
    valid = np.ones(len(df), dtype=bool)
    for key in keys:
        valid &= (df[key] != "") & (df[key] != MISSING) & df[key].notna()
    result = pd.DataFrame(0.0, index=df.index, columns=list(_SUMS))
    hist = df[valid & df["has_result"].to_numpy()]
    if hist.empty:
        return result
    agg = (hist.assign(_one=1.0)
           .groupby(keys + ["race_date"], sort=False)
           .agg(n=("_one", "sum"), k3=("target_top3", "sum"), e3=("_e3", "sum"),
                kw=("target_win", "sum"), ew=("_ew", "sum"))
           .reset_index()
           .sort_values(keys + ["race_date"], kind="mergesort"))
    agg[list(_SUMS)] = agg.groupby(keys, sort=False)[list(_SUMS)].cumsum()
    agg = agg.sort_values("race_date", kind="mergesort")

    left = df.loc[valid, keys + ["race_date"]].copy()
    # pandas のバージョンで groupby 後のキーの dtype（object / string）が変わる。
    # merge_asof は by の dtype の一致を要求するので両側をそろえる。
    for key in keys:
        left[key] = left[key].astype(str)
        agg[key] = agg[key].astype(str)
    left["_row"] = left.index
    left = left.sort_values("race_date", kind="mergesort")
    merged = pd.merge_asof(left, agg, on="race_date", by=keys,
                           allow_exact_matches=False, direction="backward")
    merged = merged.set_index("_row")
    result.loc[merged.index, list(_SUMS)] = merged[list(_SUMS)].fillna(0.0).to_numpy()
    return result


def asof_sums(df: pd.DataFrame, keys: Sequence[str], *, exclude_self: bool) -> pd.DataFrame:
    total = _asof_sums(df, keys)
    if exclude_self:
        total = total - _asof_sums(df, [*keys, "ketto_num"])
    return total


def _ae(actual: pd.Series, expected: pd.Series, shrinkage: float) -> pd.Series:
    """縮小推定した A/E。出走0なら 1.0（偶然と同じ）。"""
    return (actual + shrinkage) / (expected + shrinkage)


def _bucket_sums(df: pd.DataFrame, path_col: str, bucket: str, *,
                 exclude_self: bool) -> pd.DataFrame:
    """その祖先の産駒の「年齢区分 = bucket」の成績（行自身の年齢に関係なく）。"""
    masked = df.copy()
    # bucket 以外の行は履歴に入れない。参照する側（全行）は bucket を問わない。
    masked["_has"] = masked["has_result"] & (masked["age_bucket"] == bucket)
    masked["has_result"] = masked["_has"]
    return asof_sums(masked, [path_col], exclude_self=exclude_self)


def stat_features(df: pd.DataFrame, pedigree: PedigreeConfig) -> pd.DataFrame:
    """産駒成績の特徴量。列は作った時点で float32 にする。

    40年ぶん（約180万行 × 100列）を float64 で溜めてから変換すると、変換前の表と
    変換後の表が同時に載って数GBになる。計算は float64 で行い、列に入れるときに
    丸めるので、最後にまとめて変換するのと値は同じ。
    """
    out = pd.DataFrame(index=df.index)
    m = pedigree.shrinkage

    def put(name: str, values: pd.Series) -> None:
        out[name] = values.astype(np.float32)

    for path in pedigree.stat_paths:
        col = f"ped_{path}"
        base = asof_sums(df, [col], exclude_self=pedigree.exclude_self)
        put(f"st_{path}_log_n", np.log1p(base["n"]))
        put(f"st_{path}_ae_top3", _ae(base["k3"], base["e3"], m))
        put(f"st_{path}_ae_win", _ae(base["kw"], base["ew"], m))
        for name, cond in CONDITIONS.items():
            sums = asof_sums(df, [col, *cond], exclude_self=pedigree.exclude_self)
            put(f"st_{path}_ae_top3_{name}", _ae(sums["k3"], sums["e3"], m))
            put(f"st_{path}_log_n_{name}", np.log1p(sums["n"]))
        by_age = {}
        for bucket, _, _ in AGE_BUCKETS:
            sums = _bucket_sums(df, col, bucket, exclude_self=pedigree.exclude_self)
            by_age[bucket] = _ae(sums["k3"], sums["e3"], m)
            put(f"st_{path}_ae_age{bucket}", by_age[bucket])
        put(f"st_{path}_maturity", np.log(by_age["4p"] / by_age["2"]))
    return out


def build_features(df: pd.DataFrame, *, pedigree: PedigreeConfig,
                   features: FeatureConfig) -> FeatureSet:
    """特徴量表を作る。df は prepare() 前のデータセットでよい。"""
    prepared = prepare(df)
    generations = pedigree.categorical_generations
    cat_paths = [p for p in (all_paths(generations) if generations else ())
                 if p not in pedigree.categorical_exclude]
    frame = pd.DataFrame(index=prepared.index)
    cats: list[str] = []
    nums: list[str] = []
    for path in cat_paths:
        name = f"ped_{path}"
        frame[name] = prepared[name].replace("", MISSING).fillna(MISSING).astype(str)
        cats.append(name)
    context, context_cats, context_nums = context_columns(prepared, features.context)
    frame = frame.join(context)
    cats.extend(context_cats)
    nums.extend(context_nums)
    for name in ("inbreeding_f", "n_cross", "closest_cross"):
        frame[name] = pd.to_numeric(prepared[name], errors="coerce").astype(float)
        nums.append(name)
    # 40年ぶん（約180万行 × 100列）を float64 で持つと 1.4GB を超える。float32 で半分にする
    # （stat_features が列ごとに float32 で作る）。
    stats = stat_features(prepared, pedigree)
    frame = frame.join(stats)
    nums.extend(stats.columns)
    return FeatureSet(frame=frame, cat_features=tuple(cats), num_features=tuple(nums))
