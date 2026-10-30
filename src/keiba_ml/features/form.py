"""総合モデル（combined）の、血統以外の特徴量: 本人の過去走・タイム指数・人・直前の条件。

すべて**予想するレースより前**の情報だけで作る。

- 本人の過去走（``f_``）: 馬ごとに日付順に並べて1つずらす（同じ馬は同じ日に2回走らない）。
  走りの質（beat fraction）・1着とのタイム差・上がり3F のレース内の順位・4角の位置・間隔・格・
  本人の3着以内の A/E（3走ぶん縮める）・前走との距離の差・芝ダ / 騎手の替わり
- タイム指数（``s_``）: 走破タイムを (競馬場, 芝ダ, 距離) の基準タイムと、その日の馬場差で補正した
  秒/km（大きいほど速い）。基準タイムは**履歴の最初の speed_standard_years 年ぶん**の1着だけから作る
  （履歴は学習開始日の stats_history 年前から始まるので、学習・評価の行にとって過去の情報）。
  馬場差は同じ日・同じ競馬場・同じ芝ダの1着から作る（その日の結果なので、過去走にだけ使う）
- 人（``h_``）: 騎手・調教師・騎手×芝ダ・調教師×芝ダ・騎手×調教師の3着以内の A/E（as-of。
  その日より前。ガンマ・ポアソンの事前分布で people_shrinkage 頭ぶん縮める）
- 直前の条件（``c_``）: 斤量（とレース内の差）・枠・馬番・馬体重・増減・ブリンカー・所属。
  馬体重と増減は発走の約1時間前に発表される値
- レース内の相対化（``rel_``）: 主な値の、同じレースの中での差と百分位
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba_ml.features.pedigree import asof_sums, beat_fraction, expected_top3

#: 1着とのタイム差の上限（秒）。大敗の1走が平均を支配しないよう切る。
MARGIN_CAP_SEC = 3.0
#: 本人の A/E を偶然の側へ縮める強さ（期待3着以内数で何頭ぶん）。
OWN_SHRINKAGE = 3.0
#: 人の A/E を出す組（接尾辞 → 列）。
PEOPLE_KEYS: dict[str, tuple[str, ...]] = {
    "jockey": ("kisyu_code",), "jockey_surface": ("kisyu_code", "track_kind"),
    "trainer": ("chokyosi_code",), "trainer_surface": ("chokyosi_code", "track_kind"),
    "jockey_trainer": ("kisyu_code", "chokyosi_code"),
}
#: レース内で相対化する列。
RELATIVE_SOURCES: tuple[str, ...] = (
    "f_pf_ewm", "f_pf_mean5", "f_ae_own", "f_margin_mean3", "f_l3f_mean3", "h_jockey_ae",
    "h_trainer_ae", "f_tier3_max", "c_futan", "f_pf_same_surface",
    "s_fig_last1", "s_fig_mean3", "s_fig_best5", "s_fig_best", "s_fig_surface_last",
    "o_str_mean3", "o_beat_mean3", "o_beat_last1", "h_jockey_ae_recent", "h_trainer_ae_recent",
)
#: データセットに要る列（schema 4）。
REQUIRED_COLUMNS: tuple[str, ...] = (
    "wakuban", "futan_kg", "kisyu_code", "chokyosi_code", "tozai_cd", "blinker",
    "weight_change_kg", "result_time_sec", "result_jyuni_4c", "result_last_3f_sec",
)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce").astype(float)


def own_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """本人の過去走（f_）。df はデータセットの行（index は保つ）。

    「馬ごとに日付順に並べて1つずらす」ので、同じ馬が同じ日に2回出走していると同じ日の結果が
    入ってしまう。中央の規則では起きない（40年ぶん 180万行で 0件、2026-10-30 実測）が、
    前提が崩れたら黙って漏らさず止める。
    """
    if df.duplicated(["ketto_num", "race_date"]).any():
        raise ValueError("同じ馬が同じ日に2回出走している行があります。過去走の特徴量は"
                         "「その日より前」を前提にしているので作れません。")
    d = df.copy()
    has = d["has_result"].astype(bool)
    d["_pf"] = beat_fraction(d)
    d["_e3"] = expected_top3(d["field_size"])
    d["_top3"] = d["target_top3"].where(has)
    d["_win"] = d["target_win"].where(has)
    d["_margin"] = d["result_margin_sec"].clip(upper=MARGIN_CAP_SEC).where(has)
    l3f = _num(d["result_last_3f_sec"]).where(lambda s: s > 20)
    d["_l3f_pct"] = l3f.groupby(d["race_id"]).rank(pct=True)          # 小さいほど速い
    c4 = _num(d["result_jyuni_4c"]).where(lambda s: s > 0)
    d["_c4"] = c4 / d["field_size"]
    d["_tier"] = pd.to_numeric(d["grade_tier"], errors="coerce").astype(float)
    d["_tier3"] = d["_tier"].where(d["_top3"] == 1)
    d["_km"] = pd.to_numeric(d["kyori_m"], errors="coerce").astype(float)
    d["_e3h"] = d["_e3"].where(has, 0.0)

    d = d.sort_values(["ketto_num", "race_date", "race_id"], kind="mergesort")
    key = d["ketto_num"]
    g = d.groupby(key, sort=False)

    def prev(col: str, lag: int = 1) -> pd.Series:
        return g[col].shift(lag)

    def roll(series: pd.Series, window: int, how: str = "mean") -> pd.Series:
        r = getattr(series.groupby(key, sort=False).rolling(window, min_periods=1), how)()
        return r.reset_index(level=0, drop=True).reindex(d.index)

    def cum_before(col: str) -> pd.Series:
        v = d[col].fillna(0.0)
        return v.groupby(key, sort=False).cumsum() - v

    out = pd.DataFrame(index=d.index)
    out["f_log_runs"] = np.log1p(g.cumcount())
    for lag in (1, 2, 3):
        out[f"f_pf_last{lag}"] = prev("_pf", lag)
    pf = prev("_pf")
    out["f_pf_mean5"] = roll(pf, 5)
    ewm = pf.groupby(key, sort=False).ewm(alpha=0.35, ignore_na=True).mean()
    out["f_pf_ewm"] = ewm.reset_index(level=0, drop=True).reindex(d.index)
    margin = prev("_margin")
    out["f_margin_last1"] = margin
    out["f_margin_mean3"] = roll(margin, 3)
    out["f_l3f_last1"] = prev("_l3f_pct")
    out["f_l3f_mean3"] = roll(prev("_l3f_pct"), 3)
    out["f_c4_mean3"] = roll(prev("_c4"), 3)
    out["f_days_since"] = (d["race_date"] - g["race_date"].shift(1)).dt.days
    out["f_tier_last1"] = prev("_tier")
    out["f_tier_diff"] = d["_tier"] - prev("_tier")
    best_tier = prev("_tier3").groupby(key, sort=False).cummax()
    out["f_tier3_max"] = best_tier.groupby(key, sort=False).ffill()
    out["f_ae_own"] = (cum_before("_top3") + OWN_SHRINKAGE) / (cum_before("_e3h") + OWN_SHRINKAGE)
    out["f_wins"] = cum_before("_win")
    out["f_dist_diff_last"] = d["_km"] - prev("_km")
    last_surface = prev("track_kind")
    out["f_surface_change"] = (d["track_kind"] != last_surface).astype(float) \
        .where(last_surface.notna())
    last_jockey = prev("kisyu_code")
    out["f_jockey_change"] = (d["kisyu_code"] != last_jockey).astype(float) \
        .where(last_jockey.notna())
    by_surface = [d["ketto_num"], d["track_kind"]]
    pf_s = d.groupby(by_surface, sort=False)["_pf"].shift(1)
    num = pf_s.fillna(0.0).groupby(by_surface, sort=False).cumsum()
    den = pf_s.notna().astype(float).groupby(by_surface, sort=False).cumsum()
    out["f_pf_same_surface"] = num / den.where(den > 0)
    return out.reindex(df.index).astype(np.float32)


def speed_figures(df: pd.DataFrame, *, standard_years: int) -> pd.DataFrame:
    """タイム指数（s_）。基準タイムは df の最初の standard_years 年ぶんの1着から作る。"""
    d = df[["race_id", "race_date", "jyo_cd", "track_kind", "kyori_m", "ketto_num",
            "has_result", "result_rank"]].copy()
    d["_t"] = _num(df["result_time_sec"]).where(lambda s: s > 30)
    d["_km"] = pd.to_numeric(d["kyori_m"], errors="coerce").astype(float) / 1000.0
    win = d[d["has_result"].astype(bool)
            & (pd.to_numeric(d["result_rank"], errors="coerce") == 1) & d["_t"].notna()]
    course = ["jyo_cd", "track_kind", "kyori_m"]
    until = d["race_date"].min() + pd.DateOffset(years=standard_years)
    std = win[win["race_date"] < until].groupby(course)["_t"].median().rename("_std")
    d = d.join(std, on=course)
    d["_dev"] = (d["_t"] - d["_std"]) / d["_km"]
    variant = d.loc[win.index].groupby(["race_date", "jyo_cd", "track_kind"])["_dev"] \
        .mean().rename("_var")
    d = d.join(variant, on=["race_date", "jyo_cd", "track_kind"])
    d["_fig"] = -(d["_dev"] - d["_var"]).where(d["has_result"].astype(bool)).clip(-5, 5)
    d = d.sort_values(["ketto_num", "race_date", "race_id"], kind="mergesort")
    key = d["ketto_num"]
    g = d.groupby(key, sort=False)["_fig"]
    prev = g.shift(1)
    out = pd.DataFrame(index=d.index)
    out["s_fig_last1"] = prev
    out["s_fig_last2"] = g.shift(2)
    mean3 = prev.groupby(key, sort=False).rolling(3, min_periods=1).mean()
    out["s_fig_mean3"] = mean3.reset_index(level=0, drop=True).reindex(d.index)
    best5 = prev.groupby(key, sort=False).rolling(5, min_periods=1).max()
    out["s_fig_best5"] = best5.reset_index(level=0, drop=True).reindex(d.index)
    best = prev.groupby(key, sort=False).cummax()
    out["s_fig_best"] = best.groupby(key, sort=False).ffill()
    by_surface = [d["ketto_num"], d["track_kind"]]
    out["s_fig_surface_last"] = d.groupby(by_surface, sort=False)["_fig"].shift(1) \
        .groupby(by_surface, sort=False).ffill()
    return out.reindex(df.index).astype(np.float32)


def opponent_strength(df: pd.DataFrame, fig_best: pd.Series) -> pd.DataFrame:
    """相手関係（o_）: 過去走で走ったレースのメンバーの強さ。

    レースの強さ = その日の出走馬の as-of のタイム指数の最高値（s_fig_best。そのレースより前の
    走りだけで決まる）の平均と、上位3頭の平均。走りの質を掛けた値（強い相手にどれだけ先着したか）も
    出す。本人には馬ごとに1つずらして渡す（同じ日の結果は入らない）。
    """
    d = df[["race_id", "race_date", "ketto_num"]].copy()
    d["_fb"] = fig_best.astype(float)
    d["_pf"] = beat_fraction(df)
    d["_str"] = d.groupby("race_id")["_fb"].transform("mean")
    rank = d.groupby("race_id")["_fb"].rank(ascending=False, method="first")
    d["_top"] = d["_fb"].where(rank <= 3).groupby(d["race_id"]).transform("mean")
    d["_beat"] = d["_str"] * d["_pf"]
    d = d.sort_values(["ketto_num", "race_date", "race_id"], kind="mergesort")
    key = d["ketto_num"]
    out = pd.DataFrame(index=d.index)
    for col, name in (("_str", "o_str"), ("_top", "o_top3"), ("_beat", "o_beat")):
        prev = d.groupby(key, sort=False)[col].shift(1)
        out[f"{name}_last1"] = prev
        mean3 = prev.groupby(key, sort=False).rolling(3, min_periods=1).mean()
        out[f"{name}_mean3"] = mean3.reset_index(level=0, drop=True).reindex(d.index)
    return out.reindex(df.index).astype(np.float32)


def _sums_before(d: pd.DataFrame, key: str, *, days: int) -> pd.DataFrame:
    """各行の (race_date − days) より前の、key ごとの累積 (k3, e3)。"""
    hist = d[d["has_result"].astype(bool) & (d[key] != "")]
    agg = (hist.groupby([key, "race_date"]).agg(k3=("target_top3", "sum"), e3=("_e3", "sum"))
           .reset_index().sort_values([key, "race_date"], kind="mergesort"))
    agg[["k3", "e3"]] = agg.groupby(key)[["k3", "e3"]].cumsum()
    agg = agg.sort_values("race_date", kind="mergesort")
    left = d[[key, "race_date"]].copy()
    left["_row"] = left.index
    left["race_date"] = left["race_date"] - pd.Timedelta(days=days)
    left = left.sort_values("race_date", kind="mergesort")
    agg[key] = agg[key].astype(str)
    left[key] = left[key].astype(str)
    merged = pd.merge_asof(left, agg, on="race_date", by=key, allow_exact_matches=False)
    return merged.set_index("_row")[["k3", "e3"]].reindex(d.index).fillna(0.0)


def recent_people_features(prepared: pd.DataFrame, *, days: int,
                           shrinkage: float) -> pd.DataFrame:
    """騎手・調教師の直近 days 日の3着以内 A/E（その日より前）。今の累積 − days 日前の累積。"""
    d = prepared.copy()
    for col in ("kisyu_code", "chokyosi_code"):
        d[col] = d[col].fillna("").astype(str).str.strip()
    out = pd.DataFrame(index=d.index)
    for name, key in (("jockey", "kisyu_code"), ("trainer", "chokyosi_code")):
        now = asof_sums(d, [key], exclude_self=False)
        old = _sums_before(d, key, days=days)
        k3, e3 = now["k3"] - old["k3"], now["e3"] - old["e3"]
        out[f"h_{name}_ae_recent"] = (k3 + shrinkage) / (e3 + shrinkage)
    return out.astype(np.float32)


def people_features(prepared: pd.DataFrame, *, shrinkage: float) -> pd.DataFrame:
    """騎手・調教師の as-of の3着以内 A/E（h_）。prepared は features.pedigree.prepare() 後の表。"""
    d = prepared.copy()
    for col in ("kisyu_code", "chokyosi_code"):
        d[col] = d[col].fillna("").astype(str).str.strip()
    out = pd.DataFrame(index=d.index)
    for name, keys in PEOPLE_KEYS.items():
        s = asof_sums(d, list(keys), exclude_self=False)
        out[f"h_{name}_ae"] = (s["k3"] + shrinkage) / (s["e3"] + shrinkage)
        if name in ("jockey", "trainer"):
            out[f"h_{name}_log_n"] = np.log1p(s["n"])
    return out.astype(np.float32)


def condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """直前に分かる条件（c_）。"""
    out = pd.DataFrame(index=df.index)
    futan = _num(df["futan_kg"]) / 10.0
    out["c_futan"] = futan
    out["c_futan_rel"] = futan - futan.groupby(df["race_id"]).transform("mean")
    out["c_waku"] = _num(df["wakuban"])
    out["c_umaban"] = _num(df["umaban"])
    out["c_weight"] = df["result_weight_kg"].astype(float)
    out["c_zogen"] = _num(df["weight_change_kg"]).where(lambda s: s.abs() < 100)
    out["c_blinker"] = (df["blinker"].astype(str).str.strip() == "1").astype(float)
    out["c_tozai"] = _num(df["tozai_cd"])
    return out.astype(np.float32)


def relative_features(frame: pd.DataFrame, race_id: pd.Series) -> pd.DataFrame:
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
