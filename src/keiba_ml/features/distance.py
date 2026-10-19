"""距離適性モデルの特徴量。

見たいのは「今回の距離がその馬に合うか」で、「その馬がどれだけ強いか」ではない。
強さ（全体の着順の水準・3着以内の A/E）は別の独立モデル（過去成績）の担当なので、
ここでは**その馬の普段の走りに対する相対的な量**だけを作る。強さの水準が入らないので、
統合するときに過去成績モデルと情報が二重に数えられにくい。

**守ること: 強さだけが違う2頭（どの走りも同じだけ良い／悪い）は、本人の特徴量が完全に
一致する**（tests/ml/test_ml_distance.py が固定している）。次のものは強さの代わりになるので
入れない（2026-10-20、実データの重要度で見つけて外した。docs/ml/README.md §10）:

- 3着以内に入った距離の平均 … 3着以内の経験が無い馬で欠損になり、欠損かどうかが強さを表す
- 走りの質で重みを付けた平均距離 … 走りの水準で重みの付き方が変わる
- 総出走数・前走の馬体重 … 距離と関係なく好走のしやすさと相関する
- 祖先の産駒の出走数 … 種牡馬の人気 = 血統の質（血統モデルの担当）

1走ごとの「走りの質」は2つで測る（どちらも大きいほど良い）:

- ``pf``（beat fraction）… 負かした馬の割合。(頭数 − 着順) / (頭数 − 1)。1着 1.0、最下位 0.0
- ``pm`` … 1着馬とのタイム差を距離で割って符号を反転したもの（秒/km。0 が最良）。
  大敗の1走が平均を支配しないよう DistanceConfig.margin_cap_sec で切る

競走中止・失格の走りは走りの質に入れない（距離の経験としては数える）。

特徴量（すべて as-of。予想対象のレース日より**前**の走りだけを使う。同じ日は含めない）:

1. **本人の距離帯ごとの走り**（``own_rel_*``）… その距離帯での走りの平均を、本人の全体の
   平均に向けて縮めたもの − 全体の平均。芝ダ × 距離帯は同じ馬場の平均と比べる（芝ダの
   向き不向きはコース適性の担当なので差し引く）
2. **本人の距離と走りの傾き**（``own_slope_*`` / ``own_fit_*``）… 走りの質を距離（km）に
   回帰した傾きと、それを今回の距離に当てはめた値（傾き × (今回の距離 − 平均出走距離)）。
   正なら「今回の距離はこの馬の普段より走れる側」
3. **距離の経験**（``dist_minus_*`` / ``own_log_runs_*``）… 今回の距離 − 平均出走距離・
   前走距離・最長・最短（m）と、今回の距離帯での出走数。延長・短縮もここで表す
4. **血統から見た距離の傾向**（``ped_<位置>_*``）… 父・母父の産駒の、距離帯ごとの走り
   （産駒全体の平均と比べた相対値）、距離と走りの傾き、産駒の平均距離・3着以内に入った
   距離の平均（好走した産駒がまだいなければ平均距離で埋める）。本人の過去走は除く
   （DistanceConfig.exclude_self）。初出走の馬は 1〜3 が欠損になり、ここだけが手がかりになる
5. **条件**（DistanceConfig.context）

距離は km で計算し、距離との差は m で出す。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba_ml.config import DistanceConfig
from keiba_ml.features.asof import asof
from keiba_ml.features.base import FeatureSet, context_columns

#: 走りの質を集める列（prepare() が作る）。合計で累積する。
_PERF_SUMS = {
    "n_p": "_perf", "s_pf": "_pf", "s_pm": "_pm", "s_d": "_pd", "s_dd": "_pdd",
    "s_pfd": "_pfd", "s_pmd": "_pmd",
}
#: 距離の経験を集める列。
_RUN_SUMS = {"n_r": "_run", "s_dr": "_d"}


def prepare(df: pd.DataFrame, config: DistanceConfig) -> pd.DataFrame:
    """1走ごとの補助列を足す（結果の行だけに値が入る。結果の無い行は 0 / NaN）。"""
    df = df.copy()
    has = df["has_result"].to_numpy(dtype=bool)
    km = pd.to_numeric(df["kyori_m"], errors="coerce").astype(float) / 1000.0
    rank = pd.to_numeric(df["result_rank"], errors="coerce").astype(float)
    n = pd.to_numeric(df["field_size"], errors="coerce").astype(float)
    margin = pd.to_numeric(df["result_margin_sec"], errors="coerce").astype(float)
    perf = has & margin.notna().to_numpy() & rank.notna().to_numpy() & (n >= 2).to_numpy() \
        & km.notna().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = ((n - rank) / (n - 1)).clip(0.0, 1.0)
        pm = -margin.clip(0.0, config.margin_cap_sec) / km
    run = has & km.notna().to_numpy()
    top3 = run & (df["target_top3"].to_numpy(dtype=float) == 1)
    df["_perf"] = perf.astype(float)
    df["_pf"] = np.where(perf, pf, 0.0)
    df["_pm"] = np.where(perf, pm, 0.0)
    df["_pd"] = np.where(perf, km, 0.0)
    df["_pdd"] = np.where(perf, km * km, 0.0)
    df["_pfd"] = np.where(perf, pf * km, 0.0)
    df["_pmd"] = np.where(perf, pm * km, 0.0)
    df["_run"] = run.astype(float)
    df["_d"] = np.where(run, km, 0.0)
    df["_t3"] = top3.astype(float)
    df["_dt3"] = np.where(top3, km, 0.0)
    df["_km"] = np.where(run, km, np.nan)
    df["_target_km"] = km
    return df


def _sums(prepared: pd.DataFrame, keys: list[str], names: dict[str, str], *,
          exclude_self: bool = False) -> pd.DataFrame:
    columns = {name: (col, "sum") for name, col in names.items()}
    include = prepared["has_result"].to_numpy(dtype=bool)
    total = asof(prepared, keys, columns, include=include)
    if exclude_self:
        total = total - asof(prepared, [*keys, "ketto_num"], columns, include=include)
    return total


def _mean(total: pd.Series, count: pd.Series) -> pd.Series:
    return (total / count).where(count > 0)


def _shrunk_relative(part_sum: pd.Series, part_n: pd.Series, base_sum: pd.Series,
                     base_n: pd.Series, shrinkage: float) -> pd.Series:
    """部分（距離帯）の平均を基準（全体）の平均に向けて縮めた値 − 基準の平均。

    部分の走りが0走なら 0（基準と同じとみなす）。基準が0走なら NaN（比べられない）。
    """
    base = _mean(base_sum, base_n)
    return (part_sum + shrinkage * base) / (part_n + shrinkage) - base


def _slope(s: pd.DataFrame, value: str, cross: str, ridge: float) -> pd.Series:
    """走りの質を距離（km）に回帰した傾き。分母に ridge を足して縮める。0走なら NaN。"""
    n = s["n_p"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = s[cross] - s[value] * s["s_d"] / n
        var = s["s_dd"] - s["s_d"] ** 2 / n
        # 距離が1種類だけなら var は 0（浮動小数の誤差で僅かに負になりうる）。
        slope = cov / (var.clip(lower=0.0) + ridge)
    return slope.where((n > 0) & np.isfinite(slope))


def own_features(prepared: pd.DataFrame, config: DistanceConfig) -> pd.DataFrame:
    """本人の過去走から作る特徴量（上の docstring の 1〜3）。"""
    m = config.own_shrinkage
    target_km = prepared["_target_km"]
    out = pd.DataFrame(index=prepared.index)

    whole = _sums(prepared, ["ketto_num"], {**_PERF_SUMS, **_RUN_SUMS})
    extremes = asof(prepared, ["ketto_num"],
                    {"last_km": ("_km", "last"), "max_km": ("_km", "max"),
                     "min_km": ("_km", "min")},
                    include=prepared["has_result"].to_numpy(dtype=bool))
    perf_names = {"n_p": "_perf", "s_pf": "_pf", "s_pm": "_pm", "n_r": "_run"}
    band = _sums(prepared, ["ketto_num", "distance_band"], perf_names)
    surface = _sums(prepared, ["ketto_num", "track_kind"], perf_names)
    surface_band = _sums(prepared, ["ketto_num", "track_kind", "distance_band"], perf_names)

    out["own_log_runs_dband"] = np.log1p(band["n_r"])
    out["own_log_runs_surface_dband"] = np.log1p(surface_band["n_r"])
    for q in ("pf", "pm"):
        out[f"own_rel_{q}_dband"] = _shrunk_relative(
            band[f"s_{q}"], band["n_p"], whole[f"s_{q}"], whole["n_p"], m)
        out[f"own_rel_{q}_surface_dband"] = _shrunk_relative(
            surface_band[f"s_{q}"], surface_band["n_p"], surface[f"s_{q}"], surface["n_p"], m)
        slope = _slope(whole, f"s_{q}", f"s_{q}d", config.slope_ridge)
        out[f"own_slope_{q}"] = slope
        out[f"own_fit_{q}"] = slope * (target_km - _mean(whole["s_d"], whole["n_p"]))
    to_m = 1000.0
    out["dist_minus_mean"] = (target_km - _mean(whole["s_dr"], whole["n_r"])) * to_m
    out["dist_minus_last"] = (target_km - extremes["last_km"]) * to_m
    out["dist_minus_max"] = (target_km - extremes["max_km"]) * to_m
    out["dist_minus_min"] = (target_km - extremes["min_km"]) * to_m
    return out


def pedigree_features(prepared: pd.DataFrame, config: DistanceConfig) -> pd.DataFrame:
    """父・母父などの産駒の距離の傾向（上の docstring の 4）。"""
    m = config.pedigree_shrinkage
    target_km = prepared["_target_km"]
    out = pd.DataFrame(index=prepared.index)
    perf_names = {"n_p": "_perf", "s_pf": "_pf"}
    for path in config.pedigree_paths:
        col = f"ped_{path}"
        whole = _sums(prepared, [col], {**_PERF_SUMS, "n_t3": "_t3", "s_dt3": "_dt3"},
                      exclude_self=config.exclude_self)
        surface = _sums(prepared, [col, "track_kind"], perf_names,
                        exclude_self=config.exclude_self)
        surface_band = _sums(prepared, [col, "track_kind", "distance_band"], perf_names,
                             exclude_self=config.exclude_self)
        prefix = f"ped_{path}"
        mean_km = _mean(whole["s_d"], whole["n_p"])
        out[f"{prefix}_rel_pf_surface_dband"] = _shrunk_relative(
            surface_band["s_pf"], surface_band["n_p"], surface["s_pf"], surface["n_p"], m)
        slope = _slope(whole, "s_pf", "s_pfd", config.slope_ridge)
        out[f"{prefix}_slope_pf"] = slope
        out[f"{prefix}_fit_pf"] = slope * (target_km - mean_km)
        out[f"{prefix}_dist_minus_mean"] = (target_km - mean_km) * 1000.0
        # 3着以内の産駒がまだいない祖先は、産駒の平均距離で埋める。欠損のままだと
        # 「好走した産駒がいるか」= 祖先の質を表してしまう（血統モデルの担当）。
        top3_km = _mean(whole["s_dt3"], whole["n_t3"]).fillna(mean_km)
        out[f"{prefix}_dist_minus_top3_mean"] = (target_km - top3_km) * 1000.0
    return out


def build_features(df: pd.DataFrame, config: DistanceConfig) -> FeatureSet:
    """特徴量表を作る。df はデータセット（runners）の行でよい。"""
    prepared = prepare(df, config)
    frame, cats, nums = context_columns(prepared, config.context)
    # 40年ぶん（約180万行）でも軽く持てるよう float32 にする（血統モデルと同じ）。
    stats = pd.concat([own_features(prepared, config), pedigree_features(prepared, config)],
                      axis=1).astype(np.float32)
    frame = frame.join(stats)
    nums.extend(stats.columns)
    return FeatureSet(frame=frame, cat_features=tuple(cats), num_features=tuple(nums))
