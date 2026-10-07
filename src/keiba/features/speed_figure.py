"""タイム指数。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md §3。

1走ごとに「基準タイム + その日の馬場差 − 走破タイム」を 1600m 相当の秒にする（正ほど速い）。
基準タイム・クラス差・馬場差は**古馬戦（3歳以上・4歳以上）の1着だけ**から推定する。
2歳戦・3歳戦を混ぜると、世代の強弱が「その日の馬場が速かった／遅かった」として馬場差に
吸収され、世代差が消える（§3.4）。

DataKubun の選び方は repository.py の `_pick_by_data_kubun` をそのまま使う。選び方を
2か所に持つと、片方だけ直したときに「確定前の値を確定値として使う」事故が起きるため。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from keiba import config
from keiba.data.repository import _pick_by_data_kubun

JRA_JYO = frozenset(f"{i:02d}" for i in range(1, 11))
FLAT = ("turf", "dirt")
OLDER_SYUBETU = frozenset({"13", "14"})   # 3歳以上・4歳以上
KEY = ["jyo_cd", "track_cd", "kyori_m"]    # 基準タイムの単位（track_cd は芝ダと内外回りを含む）
FACE = ["day", "jyo_cd", "track_kind"]     # 馬場差の単位（開催面）
RUN_COLUMNS = ["race_id", "umaban", "ketto_num", "day", "date", "year", "jyo_cd", "track_cd",
               "track_kind", "kyori_m", "syubetu_cd", "class_level", "time_sec", "rank",
               "barei", "cohort"]
# 取消・除外（走っていない）と中止・失格（走破タイムが使えない）は指数の対象にしない
_EXCLUDED_IJYO = config.IJYO_EXCLUDED | config.IJYO_NO_TIME
_RACE_FIELDS = ["race_id", "jyo_cd", "track_cd", "track_kind", "kyori_m", "syubetu_cd", "class_level"]

REFERENCE_CLASS = 2            # 1勝クラス。クラス差の基準（c = 0）
PER_1000M_TO_FIGURE = 1.6      # 1000m あたり秒 → 1600m 相当の秒


@dataclass(frozen=True)
class FigureModel:
    """推定した物差し。値はすべて 1000m あたりの秒。"""

    standards: pd.Series       # KEY → 基準 s
    class_effects: pd.Series   # (class_level, track_kind) → クラス差 c。基準クラスは 0
    variants: pd.Series        # FACE → 馬場差 v（経験ベイズで縮小済み）
    sigma2: float              # 開催面の中の残差の分散
    tau2: float                # 開催面どうしの馬場差の分散
    n_races: int               # 推定に使った古馬戦の数


def prepare_runs(races: pd.DataFrame, race_horses: pd.DataFrame, *, as_of: datetime) -> pd.DataFrame:
    """指数の対象になる走を1行ずつ返す（§3.1）。発走が as_of より前の確定成績だけ。

    同時刻に発走したレースは含めない（Repository と同じく「その時点でまだ結果を知り得ない」）。
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of にはタイムゾーンが必要です（日本時間の 0:00 など）")
    r = _pick_by_data_kubun(races, config.DATA_KUBUN_RESULT_PRIORITY)
    r = r[r["jyo_cd"].isin(JRA_JYO) & r["track_kind"].isin(FLAT)]
    hasso = pd.to_datetime(r["hasso_at"], errors="coerce", utc=True)
    r = r[hasso < pd.Timestamp(as_of).tz_convert("UTC")]
    h = race_horses[race_horses["race_id"].isin(r["race_id"])]
    h = _pick_by_data_kubun(h, config.DATA_KUBUN_RESULT_PRIORITY)
    h = h[~h["result_ijyo_cd"].isin(_EXCLUDED_IJYO)]
    m = h.merge(r[_RACE_FIELDS], on="race_id", how="inner")
    m = m.assign(time_sec=pd.to_numeric(m["result_time_sec"], errors="coerce"),
                 kyori=pd.to_numeric(m["kyori_m"], errors="coerce"),
                 age=pd.to_numeric(m["barei"], errors="coerce"))
    m = m[(m["time_sec"] > 0) & m["kyori"].notna() & m["age"].notna()]   # NaN の比較は False なので落ちる
    day = m["race_id"].str[:8]
    year = day.str[:4].astype(int)
    out = pd.DataFrame({
        "race_id": m["race_id"], "umaban": m["umaban"], "ketto_num": m["ketto_num"],
        "day": day, "date": pd.to_datetime(day, format="%Y%m%d"), "year": year,
        "jyo_cd": m["jyo_cd"], "track_cd": m["track_cd"], "track_kind": m["track_kind"],
        "kyori_m": m["kyori"].astype(int), "syubetu_cd": m["syubetu_cd"],
        "class_level": pd.to_numeric(m["class_level"], errors="coerce"),
        "time_sec": m["time_sec"].astype(float),
        "rank": pd.to_numeric(m["result_kakutei_jyuni"], errors="coerce"),
        "barei": m["age"].astype(int), "cohort": year - m["age"].astype(int),
    })
    return out[RUN_COLUMNS].reset_index(drop=True)


def _older_winners(runs: pd.DataFrame) -> pd.DataFrame:
    """推定に使う行: 古馬戦の1着で、クラスが決まるもの。1着同着は1行にする（同じタイム）。"""
    w = runs[runs["syubetu_cd"].isin(OLDER_SYUBETU) & (runs["rank"] == 1) & runs["class_level"].notna()]
    w = w.drop_duplicates("race_id")
    return w.assign(y=w["time_sec"] / (w["kyori_m"] / 1000.0))


def _shrunk_face_effects(r: np.ndarray, fc: np.ndarray, nf: int) -> tuple[np.ndarray, float, float]:
    """開催面ごとの残差の平均を経験ベイズで 0 に縮める（§3.2）。

    σ²（面の中の分散）と τ²（面どうしの分散）はモーメント法。レースが少ない面ほど強く縮む。
    """
    n = np.bincount(fc, minlength=nf).astype(float)
    sums = np.bincount(fc, weights=r, minlength=nf)
    means = sums / n
    dof = len(r) - nf
    sigma2 = float(((r - means[fc]) ** 2).sum() / dof) if dof > 0 else 0.0
    tau2 = max(float(means.var()) - sigma2 * float((1.0 / n).mean()), 1e-12)
    return sums / (n + sigma2 / tau2), sigma2, tau2


def fit_model(runs: pd.DataFrame, *, max_iter: int = 500, tol: float = 1e-10) -> FigureModel:
    """古馬戦の1着だけから s・c・v を交互に推定する（§3.2）。s と c は縮めない。"""
    w = _older_winners(runs)
    if w.empty:
        raise ValueError("古馬戦の1着が1件もありません。基準タイムを推定できません")
    gk = w.groupby(KEY, sort=True)
    gc = w.groupby(["class_level", "track_kind"], sort=True)
    gf = w.groupby(FACE, sort=True)
    kc, cc, fc = gk.ngroup().to_numpy(), gc.ngroup().to_numpy(), gf.ngroup().to_numpy()
    key_index, class_index, face_index = gk.size().index, gc.size().index, gf.size().index
    nk, nc, nf = len(key_index), len(class_index), len(face_index)
    y = w["y"].to_numpy()
    n_k = np.bincount(kc, minlength=nk).astype(float)
    n_c = np.bincount(cc, minlength=nc).astype(float)
    key_kind = gk["track_kind"].first().to_numpy()
    class_kind = class_index.get_level_values("track_kind").to_numpy()
    class_lv = class_index.get_level_values("class_level").to_numpy()
    s = np.bincount(kc, weights=y, minlength=nk) / n_k
    c, v = np.zeros(nc), np.zeros(nf)
    sigma2 = tau2 = 0.0
    fitted = s[kc]
    for _ in range(max_iter):
        s = np.bincount(kc, weights=y - c[cc] - v[fc], minlength=nk) / n_k
        c = np.bincount(cc, weights=y - s[kc] - v[fc], minlength=nc) / n_c
        # 芝ダごとに基準クラス（1勝）の c を 0 にし、その分を s に移す（当てはまりは変わらない）
        for kind in np.unique(class_kind):
            ref_mask = (class_kind == kind) & (class_lv == REFERENCE_CLASS)
            ref = float(c[ref_mask][0]) if ref_mask.any() else 0.0
            c[class_kind == kind] -= ref
            s[key_kind == kind] += ref
        v, sigma2, tau2 = _shrunk_face_effects(y - s[kc] - c[cc], fc, nf)
        new = s[kc] + c[cc] + v[fc]
        done = float(np.max(np.abs(new - fitted))) < tol
        fitted = new
        if done:
            break
    return FigureModel(standards=pd.Series(s, index=key_index),
                       class_effects=pd.Series(c, index=class_index),
                       variants=pd.Series(v, index=face_index),
                       sigma2=sigma2, tau2=tau2, n_races=len(w))


def apply_figures(runs: pd.DataFrame, model: FigureModel) -> pd.DataFrame:
    """全走（2歳戦・3歳戦を含む）に指数を付ける（§3.3）。クラス差は足さない。

    基準が無いキーの走は指数を欠損にする。古馬戦の無い開催面は馬場差 0（縮小の事前の値）。
    どちらも has_standard / has_variant で数えられるようにする。
    """
    out = runs.copy()
    s = model.standards.reindex(pd.MultiIndex.from_frame(out[KEY])).to_numpy()
    v = model.variants.reindex(pd.MultiIndex.from_frame(out[FACE])).to_numpy()
    out["has_standard"] = ~np.isnan(s)
    out["has_variant"] = ~np.isnan(v)
    per_1000m = out["time_sec"].to_numpy() / (out["kyori_m"].to_numpy() / 1000.0)
    out["figure"] = (s + np.nan_to_num(v, nan=0.0) - per_1000m) * PER_1000M_TO_FIGURE
    return out


def compute_figures(races: pd.DataFrame, race_horses: pd.DataFrame, *,
                    as_of: datetime) -> tuple[pd.DataFrame, FigureModel]:
    """prepare_runs → fit_model → apply_figures をまとめて呼ぶ。"""
    runs = prepare_runs(races, race_horses, as_of=as_of)
    model = fit_model(runs)
    return apply_figures(runs, model), model


def repeatability(fig: pd.DataFrame, *, max_gap_days: int = 120) -> tuple[float, int]:
    """同じ馬の続けて2走（間隔 max_gap_days 日以内）の指数の相関係数と組の数（§3.6）。"""
    d = fig[fig["figure"].notna()].sort_values(["ketto_num", "date", "race_id"])
    horse = d["ketto_num"].to_numpy()
    gap = np.diff(d["date"].to_numpy()).astype("timedelta64[D]").astype(int)
    ok = (horse[1:] == horse[:-1]) & (gap > 0) & (gap <= max_gap_days)
    f = d["figure"].to_numpy()
    a, b = f[:-1][ok], f[1:][ok]
    if len(a) < 3:
        return float("nan"), int(len(a))
    return float(np.corrcoef(a, b)[0, 1]), int(len(a))
