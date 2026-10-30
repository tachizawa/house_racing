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
from keiba_ml.features.asof import asof
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
    """特徴量計算に要る補助列を足す（結果列は足さない）。

    ``birth_doy`` は生まれた日の年内の通し日（1月1日 = 1）。同じ世代の中での早生まれ・
    遅生まれで、2〜3歳の成長の差に効く。条件の列（``features.context``）で使える。
    """
    df = df.copy()
    df["age_bucket"] = age_bucket(df["age"])
    df["going_class"] = going_class(df["baba_cd"])
    df["_e3"] = expected_top3(df["field_size"])
    df["_ew"] = 1.0 / df["field_size"].astype(float)
    if "birth_date" in df:
        birth = pd.to_datetime(df["birth_date"], errors="coerce")
        df["birth_doy"] = birth.dt.dayofyear.astype(float)
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


#: 拡張の条件別 A/E（PedigreeConfig.extended_stats）。位置 → {接尾辞: 条件列}。
#: 条件列は extended_stat_features が足す結合キー。
EXTENDED_CONDITIONS: dict[str, dict[str, str]] = {
    "S": {"sdg": "_surf_dband_going", "sj": "_surf_jyo", "sa": "_surf_age",
          "st": "_surf_tier", "ss": "_surf_sex"},
    "DS": {"sdg": "_surf_dband_going", "sj": "_surf_jyo", "sa": "_surf_age",
           "st": "_surf_tier", "ss": "_surf_sex"},
    "D": {"st": "_surf_tier", "sa": "_surf_age"},
    "SS": {"st": "_surf_tier", "sa": "_surf_age"},
}
#: 全体と芝ダ別の A/E だけを出す追加の位置（父母・母母・母父父・母母父）。
EXTENDED_PATHS: tuple[str, ...] = ("SD", "DD", "DSS", "DDS")
#: 産駒の走りの質と格を出す位置（父・母 = 兄弟・母父・母母 = 一族）。
QUALITY_PATHS: tuple[str, ...] = ("S", "D", "DS", "DD")
#: 「格の高いレース」とみなす grade_tier の下限（4 = 3勝クラス相当以上）。
HIGH_TIER = 4

#: レース内で相対化する列（PedigreeConfig.race_relative）。無い列は飛ばす。
RELATIVE_SOURCES: tuple[str, ...] = (
    "st_S_ae_top3", "st_S_ae_top3_surface_dband", "st_S_ae_win", "st_D_ae_top3",
    "st_DS_ae_top3", "st_DS_ae_top3_surface_dband", "st_S_ae_top3_age", "st_D_ae_win",
    "stx_S_pf", "stx_D_pf", "stx_DS_pf", "stx_S_pf_surface", "stx_D_pf_surface",
    "stx_S_tier_top3", "stx_D_tier_top3", "stx_S_ae_sdg", "stx_DS_ae_sa", "stx_S_ae_sa",
    "stx_DD_ae_top3", "birth_doy", "stx_S_ae_st", "stx_D_tier_mean",
    # parent_racing / breeder_stats の列（有効なときだけある）
    "par_D_pf", "par_D_ae", "par_D_tier3", "par_S_pf", "par_DS_pf",
    "brd_ae_all", "brd_pf", "brd_ae_age",
)


def beat_fraction(df: pd.DataFrame) -> pd.Series:
    """1走の走りの質 = (頭数 − 着順) / (頭数 − 1)。1着 1.0、最下位 0.0。中止・失格・着順なしは欠損。"""
    rank = pd.to_numeric(df["result_rank"], errors="coerce").astype(float)
    n = df["field_size"].astype(float)
    ok = df["result_ijyo_cd"].astype(str).isin(["0", ""]) & rank.notna() & (rank > 0)
    return ((n - rank) / (n - 1).clip(lower=1)).where(ok)


def asof_mean(df: pd.DataFrame, keys: Sequence[str], column: str, include: pd.Series, *,
              prior_n: float, exclude_self: bool) -> pd.Series:
    """keys ごとの column の as-of 平均（レース日より前・exclude_self なら本人を除く）。

    全体の平均へ prior_n 件ぶん縮める。全体の平均も as-of（その日より前の include の行
    すべて）で、履歴が1行も無い日は 0。履歴の無い祖先は全体の平均になる。
    """
    d = df.assign(_one=1.0, _all="all")
    d[column] = d[column].where(include)
    d["_one"] = d["_one"].where(include & d[column].notna(), 0.0)
    columns = {"s": (column, "sum"), "c": ("_one", "sum")}
    total = asof(d, keys, columns, include=include)
    if exclude_self:
        total = total - asof(d, [*keys, "ketto_num"], columns, include=include)
    overall = asof(d, ["_all"], columns, include=include)
    prior_mean = (overall["s"] / overall["c"].where(overall["c"] > 0)).fillna(0.0)
    return (total["s"] + prior_n * prior_mean) / (total["c"] + prior_n)


def extended_stat_features(df: pd.DataFrame, pedigree: PedigreeConfig) -> pd.DataFrame:
    """拡張の産駒成績（``stx_``）。df は prepare() 後の表。すべて as-of・本人を除く。

    1. 条件を組み合わせた A/E（EXTENDED_CONDITIONS）: 芝ダ×距離区分×馬場、芝ダ×競馬場、
       芝ダ×年齢区分、芝ダ×格、芝ダ×性
    2. 追加の位置の A/E（EXTENDED_PATHS）
    3. 産駒の走りの質（beat fraction の平均）と格（出走した格の平均・格の高いレースへの出走の割合・
       3着以内に入ったときの格の平均）（QUALITY_PATHS）。母の産駒 = 兄弟の到達度
    """
    d = df.copy()
    include = pd.Series(d["has_result"].to_numpy(dtype=bool), index=d.index)
    d["_pf"] = beat_fraction(d)
    d["_tier"] = pd.to_numeric(d["grade_tier"], errors="coerce").astype(float)
    d["_tier_top3"] = d["_tier"].where(d["target_top3"] == 1)
    d["_hi"] = (d["_tier"] >= HIGH_TIER).astype(float)
    surface = d["track_kind"].astype(str)
    d["_surf_dband_going"] = (surface + "_" + d["distance_band"].astype(str)
                              + "_" + d["going_class"].astype(str))
    d["_surf_jyo"] = surface + "_" + d["jyo_cd"].astype(str)
    d["_surf_age"] = surface + "_" + d["age_bucket"].astype(str)
    d["_surf_tier"] = surface + "_" + d["grade_tier"].astype(str)
    d["_surf_sex"] = surface + "_" + d["sex_cd"].astype(str)
    out = pd.DataFrame(index=d.index)
    m = pedigree.shrinkage
    exclude = pedigree.exclude_self

    def put(name: str, values: pd.Series) -> None:
        out[name] = values.astype(np.float32)

    for path, conds in EXTENDED_CONDITIONS.items():
        for name, cond in conds.items():
            s = asof_sums(d, [f"ped_{path}", cond], exclude_self=exclude)
            put(f"stx_{path}_ae_{name}", _ae(s["k3"], s["e3"], m))
    for path in EXTENDED_PATHS:
        col = f"ped_{path}"
        s = asof_sums(d, [col], exclude_self=exclude)
        put(f"stx_{path}_ae_top3", _ae(s["k3"], s["e3"], m))
        put(f"stx_{path}_log_n", np.log1p(s["n"]))
        s = asof_sums(d, [col, "track_kind"], exclude_self=exclude)
        put(f"stx_{path}_ae_surface", _ae(s["k3"], s["e3"], m))
    has_pf = include & d["_pf"].notna()
    for path in QUALITY_PATHS:
        col = f"ped_{path}"
        put(f"stx_{path}_pf", asof_mean(d, [col], "_pf", has_pf, prior_n=m,
                                        exclude_self=exclude))
        put(f"stx_{path}_pf_surface", asof_mean(d, [col, "track_kind"], "_pf", has_pf,
                                                prior_n=m, exclude_self=exclude))
        put(f"stx_{path}_tier_mean", asof_mean(d, [col], "_tier", include, prior_n=m,
                                               exclude_self=exclude))
        put(f"stx_{path}_hi_share", asof_mean(d, [col], "_hi", include, prior_n=m,
                                              exclude_self=exclude))
        put(f"stx_{path}_tier_top3", asof_mean(d, [col], "_tier_top3",
                                               include & d["_tier_top3"].notna(), prior_n=3.0,
                                               exclude_self=exclude))
    return out


def race_relative_features(frame: pd.DataFrame, race_id: pd.Series) -> pd.DataFrame:
    """RELATIVE_SOURCES の列を、同じレースの出走馬の中での差（− レース平均）と順位（百分位）にする。

    血統の値そのものではなく「このレースの相手と比べてどうか」を渡す。どの列も as-of なので、
    同じレースの他の馬の値を使っても未来の結果は入らない。
    """
    cols = [c for c in RELATIVE_SOURCES if c in frame.columns]
    out = pd.DataFrame(index=frame.index)
    if not cols:
        return out
    grouped = frame[cols].groupby(race_id.to_numpy())
    mean = grouped.transform("mean")
    pct = grouped.rank(pct=True)
    for c in cols:
        out[f"rel_{c}_diff"] = (frame[c] - mean[c]).astype(np.float32)
        out[f"rel_{c}_pct"] = pct[c].astype(np.float32)
    return out


#: parent_racing_features が使う、データセットの parent_* 列（dataset.build.attach_parent_racing）。
PARENT_PATHS: tuple[str, ...] = ("S", "D", "DS")


def parent_racing_features(df: pd.DataFrame) -> pd.DataFrame:
    """父・母・母父**自身**の中央での成績（``par_``）。その馬が生まれる前のレースだけ。

    集計はデータセットを作るときに済ませてある（dataset.build.attach_parent_racing）。ここでは
    A/E（3頭ぶん縮める）・走りの質・到達した格にする。中央で走っていない祖先は欠損。
    """
    missing = [f"parent_{p}_n" for p in PARENT_PATHS if f"parent_{p}_n" not in df.columns]
    if missing:
        raise ValueError(f"データセットに {missing} がありません。parent_racing を使うには"
                         " scripts/ml/build_dataset.py でデータセットを作り直してください（schema 3）。")
    out = pd.DataFrame(index=df.index)
    for path in PARENT_PATHS:
        def col(item: str) -> pd.Series:
            return pd.to_numeric(df[f"parent_{path}_{item}"], errors="coerce").astype(float)

        n = col("n").fillna(0.0)
        raced = n > 0
        out[f"par_{path}_log_n"] = np.log1p(n).astype(np.float32)
        out[f"par_{path}_ae"] = ((col("k3").fillna(0.0) + 3.0) / (col("e3").fillna(0.0) + 3.0)) \
            .where(raced).astype(np.float32)
        out[f"par_{path}_pf"] = col("pf").astype(np.float32)
        out[f"par_{path}_tier3"] = col("tier3").fillna(-1.0).where(raced).astype(np.float32)
        out[f"par_{path}_tiermax"] = col("tiermax").astype(np.float32)
        out[f"par_{path}_raced"] = (col("linked").fillna(0.0) + raced).astype(np.float32)
    return out


def breeder_features(df: pd.DataFrame, pedigree: PedigreeConfig) -> pd.DataFrame:
    """生産者の産駒成績（``brd_``）。df は prepare() 後の表。as-of・本人を除く。

    生産者は血統ではなく「生まれ」の情報（同じ牧場の生産馬がどれだけ走っているか）。
    本人の過去成績・騎手・オッズのどれでもないので血統モデルに入れられるが、境界の判断なので
    PedigreeConfig.breeder_stats で外せるようにしてある。
    """
    if "breeder_code" not in df.columns:
        raise ValueError("データセットに breeder_code がありません。breeder_stats を使うには"
                         " scripts/ml/build_dataset.py でデータセットを作り直してください（schema 3）。")
    d = df.assign(_breeder=df["breeder_code"].fillna("").astype(str))
    m = pedigree.shrinkage
    exclude = pedigree.exclude_self
    out = pd.DataFrame(index=d.index)
    for name, cond in (("all", []), ("surface", ["track_kind"]), ("age", ["age_bucket"])):
        s = asof_sums(d, ["_breeder", *cond], exclude_self=exclude)
        out[f"brd_ae_{name}"] = _ae(s["k3"], s["e3"], m).astype(np.float32)
        if name == "all":
            out["brd_log_n"] = np.log1p(s["n"]).astype(np.float32)
    include = pd.Series(d["has_result"].to_numpy(dtype=bool), index=d.index)
    d["_pf"] = beat_fraction(d)
    out["brd_pf"] = asof_mean(d, ["_breeder"], "_pf", include & d["_pf"].notna(), prior_n=m,
                              exclude_self=exclude).astype(np.float32)
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
    if pedigree.extended_stats:
        extended = extended_stat_features(prepared, pedigree)
        frame = frame.join(extended)
        nums.extend(extended.columns)
    if pedigree.parent_racing:
        parents = parent_racing_features(prepared)
        frame = frame.join(parents)
        nums.extend(parents.columns)
    if pedigree.breeder_stats:
        breeders = breeder_features(prepared, pedigree)
        frame = frame.join(breeders)
        nums.extend(breeders.columns)
    if pedigree.race_relative:
        relative = race_relative_features(frame, prepared["race_id"])
        frame = frame.join(relative)
        nums.extend(relative.columns)
    return FeatureSet(frame=frame, cat_features=tuple(cats), num_features=tuple(nums))
