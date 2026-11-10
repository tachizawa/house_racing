"""総合モデルの追加特徴量（2026-11-08）: 近走の深掘り（``x_``）とレースレベル（``z_``）。

すべて**予想するレースより前**の情報だけで作る（馬ごとに日付順に並べて1つずらす。レース内の
比較はその日の as-of の値どうし）。学習・評価の前に ``integrity.check_truncation_sample`` で、
ある日の結果とそれより後の行を消した履歴で作り直しても同じ値になることを確かめる。

近走の深掘り（``extra_form_features``。``CombinedConfig.extra_form``）:

- タイム指数の指数加重平均・5走平均・ばらつき・上向き/下向き（前走 − 2〜4走前の平均）
- 同じ芝ダ × 距離帯・同じ競馬場 × 芝ダでのタイム指数（前回・最高・3走平均）
- 前走・3走平均の着順、5走の3着内率、5走の最小のタイム差、直近365日の出走数
- 90日以上あけた後の走りの質・今回が休み明けか
- 乗り替わりの質（今回の騎手の A/E − 前走の騎手の A/E）、斤量の変化、馬体重 − 過去の平均
- 展開: レース内の先行馬（4角の位置の3走平均 < 0.3）の数と割合、位置取りのレース内順位
- 3番手・1番手との差（上位3頭に入るかの境目）と、主な値のレース内の差・百分位

レースレベル（``race_level_features``。``CombinedConfig.race_level``）:

- 1〜3走前のレースの出走馬が、その後の**次走**でどれだけ走ったか（走りの質・タイム指数・3着内率）。
  次走の日付が予想するレースの日より前のものだけを数える（その日の結果は入らない。本人の次走は
  今回のレースなので1走前のレベルには入らない）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba_ml.features.form import MARGIN_CAP_SEC, _num, race_figures
from keiba_ml.features.pedigree import beat_fraction

#: extra_form_features が使う、総合モデルの既存の列（features.form が作る）。
BASE_COLUMNS: tuple[str, ...] = (
    "h_jockey_ae", "f_c4_mean3", "f_days_since", "s_fig_best5", "s_fig_mean3",
    "s_fig_last1", "f_pf_ewm", "o_beat_mean3", "f_ae_own",
)
#: 3番手・1番手との差を出す列。
GAP_SOURCES: tuple[str, ...] = (
    "s_fig_best5", "s_fig_mean3", "s_fig_last1", "f_pf_ewm", "o_beat_mean3", "h_jockey_ae",
    "f_ae_own",
)
#: 先行馬とみなす4角の位置（3走平均、頭数で割った値）。
FRONT_C4 = 0.3
#: 休み明けとみなす間隔（日）。
FIRSTUP_DAYS = 90


def _sort(d: pd.DataFrame) -> pd.DataFrame:
    if d.duplicated(["ketto_num", "race_date"]).any():
        raise ValueError("同じ馬が同じ日に2回出走している行があります。過去走の特徴量は"
                         "「その日より前」を前提にしているので作れません。")
    return d.sort_values(["ketto_num", "race_date", "race_id"], kind="mergesort")


def _roll(series: pd.Series, by, window: int, how: str, index: pd.Index) -> pd.Series:
    r = getattr(series.groupby(by, sort=False).rolling(window, min_periods=1), how)()
    return r.reset_index(level=list(range(r.index.nlevels - 1)), drop=True).reindex(index)


def _race_gap(values: pd.Series, race_id: pd.Series) -> tuple[pd.Series, pd.Series]:
    """レース内の3番目に大きい値との差と、最大値との差（3頭未満なら3番手は欠損）。"""
    grouped = values.groupby(race_id)
    rank = grouped.rank(ascending=False, method="first")
    third = values.where(rank == 3).groupby(race_id).transform("max")
    return values - third, values - grouped.transform("max")


def extra_form_features(df: pd.DataFrame, base: pd.DataFrame, *,
                        standard_years: int) -> pd.DataFrame:
    """近走の深掘り（x_）。base は df と同じ index で BASE_COLUMNS を持つ表。"""
    missing = [c for c in BASE_COLUMNS if c not in base.columns]
    if missing:
        raise ValueError(f"extra_form_features に要る列がありません: {missing}")
    has = df["has_result"].astype(bool)
    d = df[["race_id", "race_date", "ketto_num", "track_kind", "distance_band",
            "jyo_cd"]].copy()
    d["_fig"] = race_figures(df, standard_years=standard_years)
    d["_pf"] = beat_fraction(df)
    rank = pd.to_numeric(df["result_rank"], errors="coerce").where(has)
    d["_rank"] = rank.where(rank > 0)
    d["_top3"] = df["target_top3"].where(has)
    d["_margin"] = df["result_margin_sec"].clip(upper=MARGIN_CAP_SEC).where(has)
    weight = df["result_weight_kg"].astype(float)
    d["_wt"] = weight.where(weight > 300)
    d["_futan"] = _num(df["futan_kg"]) / 10.0
    for c in ("h_jockey_ae", "f_days_since"):
        d[c] = base[c].astype(float)
    d = _sort(d)
    key = d["ketto_num"]
    g = d.groupby(key, sort=False)
    idx = d.index
    out = pd.DataFrame(index=idx)

    fig1 = g["_fig"].shift(1)
    out["x_fig_ewm"] = fig1.groupby(key, sort=False).ewm(alpha=0.35, ignore_na=True).mean() \
        .reset_index(level=0, drop=True).reindex(idx)
    out["x_fig_mean5"] = _roll(fig1, key, 5, "mean", idx)
    out["x_fig_std5"] = _roll(fig1, key, 5, "std", idx)
    out["x_fig_trend"] = fig1 - _roll(g["_fig"].shift(2), key, 3, "mean", idx)
    for name, cols in (("band", ["ketto_num", "track_kind", "distance_band"]),
                       ("venue", ["ketto_num", "jyo_cd", "track_kind"])):
        by = [d[c] for c in cols]
        s = d.groupby(by, sort=False)["_fig"].shift(1)
        out[f"x_fig_{name}_last"] = s.groupby(by, sort=False).ffill()
        out[f"x_fig_{name}_best"] = s.groupby(by, sort=False).cummax() \
            .groupby(by, sort=False).ffill()
        out[f"x_fig_{name}_mean3"] = _roll(s, by, 3, "mean", idx)
    r1 = g["_rank"].shift(1)
    out["x_rank_last1"] = r1
    out["x_rank_mean3"] = _roll(r1, key, 3, "mean", idx)
    out["x_top3_rate5"] = _roll(g["_top3"].shift(1), key, 5, "mean", idx)
    out["x_margin_best5"] = _roll(g["_margin"].shift(1), key, 5, "min", idx)
    # 直近365日の出走数（その日より前。馬ごとに日付で二分探索）。結果の無い行（出馬表だけの行）は
    # 走っていないので数えない（結果のある行の累積数の差で数える）
    runs = np.zeros(len(d))
    kv = key.to_numpy()
    days = d["race_date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    ran = has.reindex(idx).to_numpy(dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(kv[1:] != kv[:-1]) + 1, len(d)]
    for s0, s1 in zip(starts[:-1], starts[1:]):
        seg = days[s0:s1]
        before = np.r_[0, np.cumsum(ran[s0:s1])]   # before[i] = i 番目より前の結果のある行の数
        lo = np.searchsorted(seg, seg - 365, side="left")
        runs[s0:s1] = before[:-1] - before[lo]
    out["x_runs_365"] = runs
    firstup = d["f_days_since"] >= FIRSTUP_DAYS
    pf_fu = d["_pf"].where(firstup).groupby(key, sort=False).shift(1)
    num = pf_fu.fillna(0.0).groupby(key, sort=False).cumsum()
    den = pf_fu.notna().astype(float).groupby(key, sort=False).cumsum()
    out["x_pf_firstup"] = num / den.where(den > 0)
    out["x_is_firstup"] = firstup.astype(float).where(d["f_days_since"].notna())
    out["x_jockey_upgrade"] = d["h_jockey_ae"] - g["h_jockey_ae"].shift(1)
    out["x_futan_change"] = d["_futan"] - g["_futan"].shift(1)
    wmean = g["_wt"].shift(1).groupby(key, sort=False).expanding().mean() \
        .reset_index(level=0, drop=True).reindex(idx)
    out["x_weight_vs_mean"] = d["_wt"] - wmean
    out = out.reindex(df.index)

    rid = df["race_id"]
    c4 = base["f_c4_mean3"].astype(float)
    out["x_pace_front_n"] = (c4 < FRONT_C4).groupby(rid).transform("sum")
    out["x_pace_mean_c4"] = c4.groupby(rid).transform("mean")
    out["x_c4_rank"] = c4.groupby(rid).rank(pct=True)
    out["x_front_share"] = out["x_pace_front_n"] / df["field_size"]
    for c in GAP_SOURCES:
        gap3, gapmax = _race_gap(base[c].astype(float), rid)
        out[f"x_gap3_{c}"] = gap3
        out[f"x_gapmax_{c}"] = gapmax
    for c in ("x_fig_ewm", "x_fig_band_best", "x_fig_venue_best", "x_top3_rate5"):
        grouped = out[c].groupby(rid)
        out[f"x_rel_{c}_pct"] = grouped.rank(pct=True)
        out[f"x_rel_{c}_diff"] = out[c] - grouped.transform("mean")
    return out.astype(np.float32)


def race_level_features(df: pd.DataFrame, *, standard_years: int) -> pd.DataFrame:
    """レースレベル（z_）: 1〜3走前のレースの出走馬の、その後の次走の成績（その日より前の分だけ）。"""
    has = df["has_result"].astype(bool)
    d = df[["race_id", "race_date", "ketto_num"]].copy()
    d["_fig"] = race_figures(df, standard_years=standard_years)
    d["_pf"] = beat_fraction(df).where(has)
    d["_top3"] = df["target_top3"].where(has)
    d = _sort(d)
    g = d.groupby("ketto_num", sort=False)
    nxt = pd.DataFrame({"race_id": d["race_id"], "nd": g["race_date"].shift(-1),
                        "npf": g["_pf"].shift(-1), "nfig": g["_fig"].shift(-1),
                        "ntop3": g["_top3"].shift(-1)})
    nxt = nxt[nxt["nd"].notna()].sort_values(["race_id", "nd"], kind="mergesort")
    race = nxt["race_id"]
    values = ("npf", "nfig", "ntop3")
    for c in values:
        nxt[f"{c}_n"] = nxt[c].notna().astype(float).groupby(race, sort=False).cumsum()
        nxt[f"{c}_s"] = nxt[c].fillna(0.0).groupby(race, sort=False).cumsum()
    # 同じレースの次走が同じ日に複数あれば、その日までの累積（最後の行）を使う
    agg = nxt.groupby(["race_id", "nd"], sort=False).last().reset_index() \
        .sort_values("nd", kind="mergesort")
    cols = ["race_id", "nd"] + [f"{c}_{x}" for c in values for x in ("s", "n")]
    out = pd.DataFrame(index=d.index)
    for lag in (1, 2, 3):
        q = pd.DataFrame({"race_id": g["race_id"].shift(lag), "nd": d["race_date"],
                          "_row": d.index})
        q = q[q["race_id"].notna()].sort_values("nd", kind="mergesort")
        # allow_exact_matches=False: 次走の日付が予想する日より**前**のものだけ
        m = pd.merge_asof(q, agg[cols], on="nd", by="race_id",
                          allow_exact_matches=False).set_index("_row")
        for c in values:
            out[f"z_lvl{lag}_{c}"] = (m[f"{c}_s"] / m[f"{c}_n"].where(m[f"{c}_n"] > 0)) \
                .reindex(d.index)
        out[f"z_lvl{lag}_n"] = m["npf_n"].reindex(d.index)
    out["z_lvl_mean_nfig"] = out[["z_lvl1_nfig", "z_lvl2_nfig", "z_lvl3_nfig"]].mean(axis=1)
    out["z_lvl_mean_npf"] = out[["z_lvl1_npf", "z_lvl2_npf", "z_lvl3_npf"]].mean(axis=1)
    out = out.reindex(df.index)
    rid = df["race_id"]
    for c in ("z_lvl1_nfig", "z_lvl_mean_nfig", "z_lvl1_ntop3"):
        grouped_r = out[c].groupby(rid)
        out[f"{c}_rdiff"] = out[c] - grouped_r.transform("mean")
        out[f"{c}_rpct"] = grouped_r.rank(pct=True)
    return out.astype(np.float32)
