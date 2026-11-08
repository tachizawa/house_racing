"""コース適性モデルの特徴量。

見たいのは「今回のコース（競馬場・芝ダ・回り・内外回り・直線の長さ・坂・馬場状態）がその馬に
合うか」で、「その馬がどれだけ強いか」ではない（距離適性モデル features/distance.py と同じ方針）。
強さは別の独立モデルの担当なので、本人の特徴量は**その馬の普段の走りに対する相対的な量**と
経験の数だけにする。

**守ること: 強さだけが違う2頭（どの走りも同じだけ良い／悪い）は、本人の特徴量（own_）が
完全に一致する**（tests/ml/test_ml_course.py が固定している）。総出走数・全体の着順の水準・
3着以内の A/E は入れない。

走りの質は距離適性モデルと同じ2つ（features.distance.prepare）:
``pf`` 負かした馬の割合、``pm`` 1着馬とのタイム差 / 距離（秒/km、符号反転、上限あり）。

コースの分け方（prepare_course が列を作る。キーが欠損の行は集計しない）:

| 名前 | キー | 比べる相手 |
|---|---|---|
| surface | 芝ダ | 本人の全体 |
| venue | 競馬場 × 芝ダ | 本人の同じ芝ダ |
| course | 競馬場 × トラックコード（芝ダ・回り・内外回り） | 本人の同じ芝ダ |
| course_dist | 競馬場 × トラックコード × 距離 | 本人の同じ芝ダ × 距離帯 |
| dir | 回り（右・左・直線） × 芝ダ | 本人の同じ芝ダ |
| straight | 最後の直線の長さの区分 × 芝ダ | 本人の同じ芝ダ |
| slope | ゴール前の坂の有無 × 芝ダ | 本人の同じ芝ダ |
| going | 馬場状態（良・稍重・重以上） × 芝ダ | 本人の同じ芝ダ |
| venue_dband（extra_groups） | 競馬場 × 芝ダ × 距離帯 | 本人の同じ芝ダ × 距離帯 |
| trip（extra_groups） | 所属から見た開催地（地元・遠征・北海道） × 芝ダ | 本人の同じ芝ダ |

特徴量（すべて as-of。予想対象のレース日より**前**の走りだけ。同じ日は含めない）:

1. ``own_rel_{pf,pm}_<分け方>`` … 今回と同じ区分での走りの平均を、比べる相手の平均に向けて
   縮めたもの − 比べる相手の平均
2. ``own_log_runs_<分け方>`` … 今回と同じ区分での出走数（log1p）。surface は数えない
   （全体の出走数 = 経験の量は強さと相関する）
3. ``ped_<位置>_rel_pf_<分け方>`` … 父・母父の産駒（本人を除く）の同じ区分での走りの相対値
4. ``bias_draw_ae`` … このコース・距離での、今回の馬番の位置（内・中・外の3区分）の
   as-of 3着以内 A/E（1 に向けて縮める）。馬の強さとは関係ないコースの性質
5. ``bias_front_ae`` / ``own_style`` / ``style_fit``（CourseConfig.style が True のときだけ）… この
   コース・距離帯で4角を前の方で回った馬の 3着以内 A/E と、本人の普段の4角の位置（0 = 先頭、
   1 = 最後方）、その2つの積。**4角の位置そのものは強さと相関する**（強い馬ほど前に付ける。
   2013〜2022年の5走以上の馬で、4角の位置の平均と負かした割合の平均の相関 −0.47）ので、
   本人の位置は ``pos + style_strength_slope × (pf − 0.5)`` に補正してから平均する。
   0.58 は同じ期間の「馬ごとの平均の位置 ~ 馬ごとの平均の pf」の傾き（3走以上の馬で −0.58）で、
   補正後の馬ごとの平均と強さの相関は −0.03〜+0.09（1/3/5走以上）
5b. ``bias_kick_rel`` / ``own_kick`` / ``kick_fit``（CourseConfig.kick が True のときだけ）… 末脚（上がり3F の
   レース内の順位。0 = 最速、1 = 最遅）。このコース・距離帯で上がりが速い側 1/3 に入った馬の 3着以内 A/E を、
   同じ芝ダ × 距離帯の全コースの A/E で割ったもの（このコースで末脚がどれだけ利くか）と、本人の普段の
   末脚、その2つの積。末脚の順位は強さと強く相関する（2013〜2022年の3走以上の馬で −0.79）ので、本人の
   末脚は ``kick + kick_strength_slope × (pf − 0.5)``（傾き 0.755。補正後の相関は 0.00）に補正する
5c. ``meet_*``（CourseConfig.meeting_days > 0 のときだけ）… 開催中のトラックバイアス。直近 meeting_days 日
   （予想する日の前日まで）の同じ競馬場 × 芝ダで、今回の馬番の位置（内・中・外）の 3着以内 A/E
   （``meet_draw_ae``）と、4角で前の方にいた馬の A/E をその競馬場 × 芝ダの全期間の A/E で割ったもの
   （``meet_front_rel``。style のとき本人の位置取りとの積 ``meet_style_fit`` も）
6. 条件（CourseConfig.context）と ``umaban_frac``（馬番の位置 0〜1）
7. ``rr_<列>``（CourseConfig.race_relative が True のときだけ）… RELATIVE_SOURCES の列の、同じレースの
   出走馬の平均との差（欠損の馬は平均に入れず、差も欠損）。レースの中で誰が一番合うかを直接渡す
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba_ml.config import CourseConfig, DistanceConfig
from keiba_ml.features.asof import asof
from keiba_ml.features.base import MISSING, FeatureSet, context_columns
from keiba_ml.features.distance import _shrunk_relative, _sums, prepare as prepare_runs

#: 最後の直線の長さ（m、Aコース）。JRA のコース紹介の値（src/keiba/features/course_facts.json と
#: 同じ出どころ。あちらで欠けている中山外回り・中京・京都外回り芝なども埋めた）。キーは
#: (競馬場, 芝ダ, 内外回り)。内外回りの区別が無い競馬場は "std"。
STRAIGHT_M: dict[tuple[str, str, str], float] = {
    ("01", "turf", "std"): 266.1, ("01", "dirt", "std"): 264.3,
    ("02", "turf", "std"): 262.1, ("02", "dirt", "std"): 260.3,
    ("03", "turf", "std"): 292.0, ("03", "dirt", "std"): 295.7,
    ("04", "turf", "in"): 358.7, ("04", "turf", "out"): 658.7, ("04", "dirt", "std"): 353.9,
    ("05", "turf", "std"): 525.9, ("05", "dirt", "std"): 501.6,
    ("06", "turf", "in"): 310.0, ("06", "turf", "out"): 310.0, ("06", "dirt", "std"): 308.0,
    ("07", "turf", "std"): 412.5, ("07", "dirt", "std"): 410.7,
    ("08", "turf", "in"): 328.4, ("08", "turf", "out"): 403.7, ("08", "dirt", "std"): 329.1,
    ("09", "turf", "in"): 356.5, ("09", "turf", "out"): 473.6, ("09", "dirt", "std"): 352.7,
    ("10", "turf", "std"): 293.0, ("10", "dirt", "std"): 291.3,
}
#: 直線の区分の境界（m）。〜320 短い / 〜430 中 / それ以上 長い。直線コース（新潟1000m）は別区分。
STRAIGHT_BANDS: tuple[tuple[float, str], ...] = ((0.0, "short"), (320.0, "mid"), (430.0, "long"))
#: ゴール前に急な上り坂がある競馬場（中山・阪神・中京・東京）。
SLOPE_JYO: frozenset[str] = frozenset({"06", "09", "07", "05"})
#: 競馬場の地域。北海道（札幌・函館）は東西どちらからも遠征（滞在が多い）なので別。
#: 東 = 福島・新潟・東京・中山、西 = 中京・京都・阪神・小倉。
JYO_REGION: dict[str, str] = {"01": "hokkaido", "02": "hokkaido", "03": "east", "04": "east",
                              "05": "east", "06": "east", "07": "west", "08": "west",
                              "09": "west", "10": "west"}
#: 所属（東西所属コード）→ 地域。1 美浦 = 東、2 栗東 = 西。それ以外（地方・外国）は欠損。
TOZAI_REGION: dict[str, str] = {"1": "east", "2": "west"}


def trip_class(jyo: str, tozai) -> str:
    """所属から見た開催地: home（同じ地域）/ away（反対の地域）/ hokkaido。分からなければ ""。"""
    region = JYO_REGION.get(str(jyo), "")
    home = TOZAI_REGION.get("" if tozai is None else str(tozai).strip(), "")
    if not region:
        return ""
    if region == "hokkaido":
        return "hokkaido"
    if not home:
        return ""
    return "home" if region == home else "away"


def track_shape(track_cd: str) -> tuple[str, str]:
    """トラックコード（2009）→ (回り L/R/S, 内外回り in/out/std)。分からなければ ("", "")。"""
    if not str(track_cd).strip().isdigit():
        return "", ""
    v = int(track_cd)
    if v in (10, 29):
        return "S", "std"
    left = {11: "std", 12: "out", 13: "in", 14: "out", 15: "in", 16: "out",
            23: "std", 25: "in", 27: "std"}
    right = {17: "std", 18: "out", 19: "in", 20: "out", 21: "in", 22: "out",
             24: "std", 26: "out", 28: "std"}
    if v in left:
        return "L", left[v]
    if v in right:
        return "R", right[v]
    return "", ""


def _layout_for_straight(jyo: str, kind: str, layout: str) -> str:
    """内外回りの区別が直線の表に無い競馬場は std として引く（内2周などの "in" も同じ）。"""
    if (jyo, kind, layout) in STRAIGHT_M:
        return layout
    if layout == "std" and (jyo, kind, "in") in STRAIGHT_M:
        return "in"     # 回りの区別の無いコード（17 など）は内回りとみなす（京都・阪神の芝の旧コード）
    return "std"


def straight_band(jyo: str, kind: str, layout: str, direction: str) -> str:
    if direction == "S":
        return "straight"
    meters = STRAIGHT_M.get((jyo, kind, _layout_for_straight(jyo, kind, layout)))
    if meters is None:
        return ""
    band = ""
    for lower, name in STRAIGHT_BANDS:
        if meters >= lower:
            band = name
    return band


def going_class(baba_cd) -> str:
    value = "" if baba_cd is None or (isinstance(baba_cd, float) and np.isnan(baba_cd)) \
        else str(baba_cd).strip()
    return {"1": "firm", "2": "soft", "3": "heavy", "4": "heavy"}.get(value, "")


def _key(*parts: pd.Series) -> pd.Series:
    """文字列のキーをつなぐ。どれか1つでも欠損なら MISSING（as-of の集計から外れる）。"""
    out = parts[0].astype(str)
    bad = parts[0].isna() | parts[0].astype(str).isin(["", MISSING, "nan", "None", "<NA>"])
    for part in parts[1:]:
        out = out + "|" + part.astype(str)
        bad |= part.isna() | part.astype(str).isin(["", MISSING, "nan", "None", "<NA>"])
    return out.where(~bad, MISSING)


def prepare_course(df: pd.DataFrame, config: CourseConfig) -> pd.DataFrame:
    """距離適性の 1走ごとの補助列（prepare）に、コースの区分の列を足す。"""
    dist = DistanceConfig(margin_cap_sec=config.margin_cap_sec)
    df = prepare_runs(df, dist)
    jyo = df["jyo_cd"].astype(object).fillna("").astype(str)
    kind = df["track_kind"].astype(object).fillna("").astype(str)
    tcd = df["track_cd"].astype(object).fillna("").astype(str).str.strip()
    shapes = [track_shape(t) for t in tcd]
    direction = pd.Series([s[0] for s in shapes], index=df.index)
    layout = pd.Series([s[1] for s in shapes], index=df.index)
    straight = pd.Series([straight_band(j, k, la, d) for j, k, la, d in
                          zip(jyo, kind, layout, direction)], index=df.index)
    slope = jyo.isin(SLOPE_JYO).map({True: "slope", False: "flat"})
    going = df["baba_cd"].map(going_class) if "baba_cd" in df else pd.Series("", index=df.index)
    kyori = pd.to_numeric(df["kyori_m"], errors="coerce")
    kyori_s = kyori.astype("Int64").astype(str).where(kyori.notna() & (kyori > 0), "")
    dband = df["distance_band"].astype(object).fillna("").astype(str)

    df["k_surface"] = _key(kind)
    df["k_surface_dband"] = _key(kind, dband)
    df["k_venue"] = _key(jyo, kind)
    df["k_course"] = _key(jyo, tcd)
    df["k_course_dist"] = _key(jyo, tcd, kyori_s)
    df["k_dir"] = _key(direction, kind)
    df["k_straight"] = _key(straight, kind)
    df["k_slope"] = _key(slope, kind)
    df["k_going"] = _key(going, kind)
    df["k_all"] = "all"
    df["k_course_dband"] = _key(jyo, tcd, dband)
    df["k_venue_dband"] = _key(jyo, kind, dband)
    tozai = df["tozai_cd"] if "tozai_cd" in df else pd.Series("", index=df.index)
    trip = pd.Series([trip_class(j, z) for j, z in zip(jyo, tozai)], index=df.index)
    df["k_trip"] = _key(trip, kind)

    # 馬番の位置（0 = 最内、1 = 最外）と内・中・外の3区分。
    umaban = pd.to_numeric(df["umaban"], errors="coerce").astype(float)
    n = pd.to_numeric(df["field_size"], errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = ((umaban - 1) / (n - 1)).clip(0.0, 1.0)
    df["umaban_frac"] = frac.where(n >= 2)
    # 馬番の位置の区分（既定は内・中・外の3つ。CourseConfig.draw_zones）
    k = config.draw_zones
    edges = [-0.01, *[i / k for i in range(1, k)], 1.01]
    zone = pd.cut(df["umaban_frac"], edges, labels=[f"z{i}" for i in range(k)])
    df["k_venue_zone"] = _key(jyo, kind, zone.astype(object).fillna(""))
    df["k_draw"] = _key(df["k_course_dist"].where(df["k_course_dist"] != MISSING, ""),
                        zone.astype(object).fillna(""))
    going_key = going.astype(object).fillna("")
    df["k_draw_going"] = _key(df["k_draw"].where(df["k_draw"] != MISSING, ""), going_key)
    df["k_course_dband_going"] = _key(
        df["k_course_dband"].where(df["k_course_dband"] != MISSING, ""), going_key)
    # 3着以内の期待値（偶然）。頭数 n のレースで 3 / n。
    has = df["has_result"].to_numpy(dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        expect = np.minimum(3.0 / n, 1.0)
    df["_e3"] = np.where(has & np.isfinite(expect), expect, 0.0)
    df["_k3"] = np.where(has, (df["target_top3"].to_numpy(dtype=float) == 1).astype(float), 0.0)
    # 4角の位置（0 = 先頭、1 = 最後方）。値が無い走りは数えない。
    if "result_jyuni_4c" in df:
        c4 = pd.to_numeric(df["result_jyuni_4c"].replace("", np.nan), errors="coerce").astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = ((c4 - 1) / (n - 1)).clip(0.0, 1.0)
        ok = has & pos.notna().to_numpy() & (c4 > 0).to_numpy() & (n >= 2).to_numpy()
    else:
        pos = pd.Series(np.nan, index=df.index)
        ok = np.zeros(len(df), dtype=bool)
    # 本人の位置取りは強さで説明できる分を除く（上の docstring の 5）。pf の無い走りは数えない。
    pf = df["_pf"].to_numpy(dtype=float)
    ok_own = ok & (df["_perf"].to_numpy(dtype=float) > 0)
    df["_c4n"] = ok_own.astype(float)
    df["_c4"] = np.where(ok_own, pos + config.style_strength_slope * (pf - 0.5), 0.0)
    front = ok & (pos <= config.front_threshold).to_numpy()
    df["_fr_n"] = front.astype(float)
    df["_fr_e3"] = np.where(front, df["_e3"], 0.0)
    df["_fr_k3"] = np.where(front, df["_k3"], 0.0)
    # 末脚（上がり3F のレース内の順位）。結果のある走りだけ。
    if "result_last_3f_sec" in df:
        f3 = pd.to_numeric(df["result_last_3f_sec"].replace("", np.nan), errors="coerce").astype(float)
        f3 = f3.where(has & (f3 > 0).to_numpy())
        rank = f3.groupby(df["race_id"]).rank(method="average")
        count = f3.groupby(df["race_id"]).transform("count")
        with np.errstate(divide="ignore", invalid="ignore"):
            kick = ((rank - 1) / (count - 1)).clip(0.0, 1.0).where(count >= 2)
    else:
        kick = pd.Series(np.nan, index=df.index)
    ok_kick = kick.notna().to_numpy() & (df["_perf"].to_numpy(dtype=float) > 0)
    df["_kn"] = ok_kick.astype(float)
    df["_kick"] = np.where(ok_kick, kick + config.kick_strength_slope * (pf - 0.5), 0.0)
    fast = kick.notna().to_numpy() & (kick <= 1 / 3).to_numpy()
    df["_fk_e3"] = np.where(fast, df["_e3"], 0.0)
    df["_fk_k3"] = np.where(fast, df["_k3"], 0.0)
    return df


#: (分け方, 比べる相手のキー)。
OWN_GROUPS: tuple[tuple[str, str], ...] = (
    ("surface", "k_all"),
    ("venue", "k_surface"),
    ("course", "k_surface"),
    ("course_dist", "k_surface_dband"),
    ("dir", "k_surface"),
    ("straight", "k_surface"),
    ("slope", "k_surface"),
    ("going", "k_surface"),
)
#: 設定（CourseConfig.pedigree_extra_groups）で産駒の傾向に足せる分け方。
PEDIGREE_EXTRA_GROUPS: dict[str, str] = {"course": "k_surface", "venue_dband": "k_surface_dband",
                                         "course_dist": "k_surface_dband"}
#: 設定（CourseConfig.extra_groups）で足せる分け方。
EXTRA_GROUPS: dict[str, str] = {"venue_dband": "k_surface_dband", "trip": "k_surface"}
#: 経験の数を出す分け方（surface は全体の出走数に近いので出さない）。
COUNT_GROUPS: tuple[str, ...] = ("venue", "course", "course_dist", "dir", "going")
#: 産駒の傾向を出す分け方。
PEDIGREE_GROUPS: tuple[tuple[str, str], ...] = (
    ("surface", "k_all"),
    ("venue", "k_surface"),
    ("dir", "k_surface"),
    ("straight", "k_surface"),
    ("slope", "k_surface"),
    ("going", "k_surface"),
)


def _without_going(groups, config: CourseConfig):
    """use_going が False なら馬場状態の分け方を除く（前日には馬場状態が分からない）。"""
    return [(g, b) for g, b in groups if config.use_going or g != "going"]


def _decayed(prepared: pd.DataFrame, days: float) -> tuple[pd.DataFrame, pd.Series]:
    """走りの列に exp((その走りの日 − 基準日) / days) を掛けた表と、各行の exp((レース日 − 基準日) / days)。

    as-of の合計をその行の値で割ると Σ exp(−経過日数 / days) × 値 になる（古い走りほど軽い）。
    """
    day = (prepared["race_date"] - prepared["race_date"].min()).dt.days.astype(float)
    weight = np.exp(day / days)
    out = prepared.copy()
    for col in ("_perf", "_pf", "_pm"):
        out[col] = prepared[col] * weight
    return out, pd.Series(weight, index=prepared.index)


def own_features(prepared: pd.DataFrame, config: CourseConfig) -> pd.DataFrame:
    m = config.own_shrinkage
    names = {"n_p": "_perf", "s_pf": "_pf", "s_pm": "_pm", "n_r": "_run"}
    out = pd.DataFrame(index=prepared.index)
    cache: dict[str, pd.DataFrame] = {}
    if config.decay_days > 0:
        # 相対的な走り（n_p・s_pf・s_pm）だけを減衰させる。出走数（n_r）はそのまま数える。
        weighted, now = _decayed(prepared, config.decay_days)
    else:
        weighted, now = prepared, None

    def sums(key: str) -> pd.DataFrame:
        if key not in cache:
            total = _sums(weighted, ["ketto_num", key], names)
            if now is not None:
                for col in ("n_p", "s_pf", "s_pm"):
                    total[col] = total[col] / now
            cache[key] = total
        return cache[key]

    groups = [*_without_going(OWN_GROUPS, config), *((g, EXTRA_GROUPS[g]) for g in config.extra_groups)]
    for group, base_key in groups:
        part, base = sums(f"k_{group}"), sums(base_key)
        for q in config.qualities:
            out[f"own_rel_{q}_{group}"] = _shrunk_relative(
                part[f"s_{q}"], part["n_p"], base[f"s_{q}"], base["n_p"], m)
    counts = [g for g in COUNT_GROUPS if config.use_going or g != "going"]
    for group in (*counts, *config.extra_groups):
        out[f"own_log_runs_{group}"] = np.log1p(sums(f"k_{group}")["n_r"])
    return out


def pedigree_features(prepared: pd.DataFrame, config: CourseConfig) -> pd.DataFrame:
    m = config.pedigree_shrinkage
    names = {"n_p": "_perf", "s_pf": "_pf"}
    out = pd.DataFrame(index=prepared.index)
    for path in config.pedigree_paths:
        col = f"ped_{path}"
        cache: dict[str, pd.DataFrame] = {}

        def sums(key: str) -> pd.DataFrame:
            if key not in cache:
                cache[key] = _sums(prepared, [col, key], names, exclude_self=config.exclude_self)
            return cache[key]

        groups = [*_without_going(PEDIGREE_GROUPS, config),
                  *((g, PEDIGREE_EXTRA_GROUPS[g]) for g in config.pedigree_extra_groups)]
        for group, base_key in groups:
            part, base = sums(f"k_{group}"), sums(base_key)
            out[f"ped_{path}_rel_pf_{group}"] = _shrunk_relative(
                part["s_pf"], part["n_p"], base["s_pf"], base["n_p"], m)
    return out


def _ae(k3: pd.Series, e3: pd.Series, shrinkage: float) -> pd.Series:
    """3着以内の A/E を 1 に向けて縮めたもの（期待値で shrinkage 頭ぶんの事前）。"""
    return (k3 + shrinkage) / (e3 + shrinkage)


def bias_features(prepared: pd.DataFrame, config: CourseConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=prepared.index)
    include = prepared["has_result"].to_numpy(dtype=bool)
    if config.draw_bias:
        draw = asof(prepared, ["k_draw"], {"k3": ("_k3", "sum"), "e3": ("_e3", "sum")},
                    include=include)
        out["bias_draw_ae"] = _ae(draw["k3"], draw["e3"], config.bias_shrinkage)
        if config.bias_by_going:
            draw = asof(prepared, ["k_draw_going"], {"k3": ("_k3", "sum"), "e3": ("_e3", "sum")},
                        include=include)
            out["bias_draw_going_ae"] = _ae(draw["k3"], draw["e3"], config.bias_shrinkage)
    if config.style:
        front = asof(prepared, ["k_course_dband"],
                     {"k3": ("_fr_k3", "sum"), "e3": ("_fr_e3", "sum")}, include=include)
        out["bias_front_ae"] = _ae(front["k3"], front["e3"], config.bias_shrinkage)
        if config.bias_by_going:
            front = asof(prepared, ["k_course_dband_going"],
                         {"k3": ("_fr_k3", "sum"), "e3": ("_fr_e3", "sum")}, include=include)
            out["bias_front_going_ae"] = _ae(front["k3"], front["e3"], config.bias_shrinkage)
        if config.decay_style and config.decay_days > 0:
            # 位置取りの平均も古い走りほど軽くする（比なので基準日の重みは打ち消し合う）
            day = (prepared["race_date"] - prepared["race_date"].min()).dt.days.astype(float)
            w = np.exp(day / config.decay_days)
            src = prepared.assign(_c4n=prepared["_c4n"] * w, _c4=prepared["_c4"] * w)
        else:
            src = prepared
        own = asof(src, ["ketto_num"], {"n": ("_c4n", "sum"), "s": ("_c4", "sum")},
                   include=include)
        out["own_style"] = (own["s"] / own["n"]).where(own["n"] > 0)
        out["style_fit"] = (out["bias_front_ae"] - 1.0) * (0.5 - out["own_style"])
        if config.front_relative:
            # 前に付けた馬は強い馬が多く A/E はどのコースでも 1 を超えるので、同じ芝ダ × 距離帯の
            # 全コースの A/E で割って「このコースで前がどれだけ残るか」にする
            every = asof(prepared, ["k_surface_dband"],
                         {"k3": ("_fr_k3", "sum"), "e3": ("_fr_e3", "sum")}, include=include)
            out["bias_front_rel"] = out["bias_front_ae"] / _ae(every["k3"], every["e3"],
                                                                config.bias_shrinkage)
            out["style_fit_rel"] = (out["bias_front_rel"] - 1.0) * (0.5 - out["own_style"])
    if config.kick:
        cols = {"k3": ("_fk_k3", "sum"), "e3": ("_fk_e3", "sum")}
        here = asof(prepared, ["k_course_dband"], cols, include=include)
        every = asof(prepared, ["k_surface_dband"], cols, include=include)
        s = config.bias_shrinkage
        out["bias_kick_rel"] = _ae(here["k3"], here["e3"], s) / _ae(every["k3"], every["e3"], s)
        own = asof(prepared, ["ketto_num"], {"n": ("_kn", "sum"), "s": ("_kick", "sum")},
                   include=include)
        out["own_kick"] = (own["s"] / own["n"]).where(own["n"] > 0)
        out["kick_fit"] = (out["bias_kick_rel"] - 1.0) * (0.5 - out["own_kick"])
    return out


#: レースの中での差にする列（あるものだけ）。
RELATIVE_SOURCES: tuple[str, ...] = (
    "own_rel_pf_surface", "own_rel_pf_venue", "own_rel_pf_course", "own_rel_pf_course_dist",
    "own_rel_pf_going", "own_rel_pf_straight", "own_rel_pm_course_dist",
    "ped_S_rel_pf_surface", "ped_S_rel_pf_venue", "bias_draw_ae", "style_fit",
    "meet_draw_ae", "meet_style_fit",
)


def race_relative(stats: pd.DataFrame, race_id: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=stats.index)
    for col in RELATIVE_SOURCES:
        if col in stats:
            value = stats[col].astype(float)
            out[f"rr_{col}"] = value - value.groupby(race_id).transform("mean")
    return out


def window_sums(prepared: pd.DataFrame, keys: list[str], columns: dict, days: int) -> pd.DataFrame:
    """as-of の合計を直近 days 日（予想する日の days 日前 〜 前日）に限ったもの。

    全期間の as-of（前日まで）から、日付を days 日ずらした行で取った as-of（days 日前の前日まで）を引く。
    """
    include = prepared["has_result"].to_numpy(dtype=bool)
    full = asof(prepared, keys, columns, include=include)
    shifted = prepared.copy()
    shifted["race_date"] = shifted["race_date"] - pd.Timedelta(days=days - 1)
    both = pd.concat([prepared, shifted], ignore_index=True)
    mask = np.concatenate([include, np.zeros(len(prepared), dtype=bool)])
    lag = asof(both, keys, columns, include=mask).iloc[len(prepared):]
    lag.index = prepared.index
    return full - lag


def meeting_features(prepared: pd.DataFrame, config: CourseConfig,
                     own_style: pd.Series | None) -> pd.DataFrame:
    out = pd.DataFrame(index=prepared.index)
    days, s = config.meeting_days, config.meeting_shrinkage
    cols = {"k3": ("_k3", "sum"), "e3": ("_e3", "sum")}
    draw = window_sums(prepared, ["k_venue_zone"], cols, days)
    out["meet_draw_ae"] = _ae(draw["k3"], draw["e3"], s)
    front_cols = {"k3": ("_fr_k3", "sum"), "e3": ("_fr_e3", "sum")}
    recent = window_sums(prepared, ["k_venue"], front_cols, days)
    every = asof(prepared, ["k_venue"], front_cols,
                 include=prepared["has_result"].to_numpy(dtype=bool))
    out["meet_front_rel"] = _ae(recent["k3"], recent["e3"], s) / _ae(every["k3"], every["e3"], s)
    if own_style is not None:
        out["meet_style_fit"] = (out["meet_front_rel"] - 1.0) * (0.5 - own_style)
    return out


def build_features(df: pd.DataFrame, config: CourseConfig) -> FeatureSet:
    """特徴量表を作る。df はデータセット（runners）の行でよい（track_cd が要る。schema 5）。"""
    prepared = prepare_course(df, config)
    frame, cats, nums = context_columns(prepared, config.context)
    if config.draw_bias:
        frame["umaban_frac"] = prepared["umaban_frac"].astype(float)
        nums.append("umaban_frac")
    stats = pd.concat([own_features(prepared, config), pedigree_features(prepared, config),
                       bias_features(prepared, config)], axis=1)
    if config.meeting_days > 0:
        style = stats["own_style"] if "own_style" in stats else None
        stats = pd.concat([stats, meeting_features(prepared, config, style)], axis=1)
    if config.race_relative:
        stats = pd.concat([stats, race_relative(stats, prepared["race_id"])], axis=1)
    stats = stats.astype(np.float32)
    frame = frame.join(stats)
    nums.extend(stats.columns)
    return FeatureSet(frame=frame, cat_features=tuple(cats), num_features=tuple(nums))
