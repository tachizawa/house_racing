"""年齢曲線（仕様 §6）。同じ馬の半年ごとの変化を積み上げ、3歳後半を 0 にする。

年齢ごとの全体の平均を並べると、弱い馬ほど早く引退するので高齢ほど強く見える（生き残りの偏り）。
同じ馬の隣り合う期間どうしの差だけを使うことで、それを避ける。ただし大きく衰えた馬はそのまま
引退して差が残らないので、衰えは小さめに出る（§6.3）。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

ANCHOR = (3, 1)        # 3歳後半 = 0
MIN_RUNS = 2           # 期間の値を作るのに要る走の数
TOP_RUNS = 2           # 期間の値 = 上位2走の平均（最高値1つだと走った回数が多い期間ほど高く出る）
MIN_PEAK_N = 30        # 全盛期・1年後の落ち幅に使う期間の最低頭数（これ未満はレポートで「頭数少」）
CURVE_COLUMNS = ["age", "half", "label", "delta", "n", "delta_lo", "delta_hi",
                 "level", "level_lo", "level_hi"]


def period_label(age: int, half: int) -> str:
    return f"{age}歳{'前半' if half == 0 else '後半'}"


def half_of(d: date) -> int:
    """1〜6月は 0（前半）、7〜12月は 1（後半）。"""
    return 0 if d.month <= 6 else 1


def _ordinal(age: int, half: int) -> int:
    return 2 * age + half


def period_values(fig: pd.DataFrame, *, years: list[int], before: date,
                  surface: str | None = None) -> pd.DataFrame:
    """馬 × 期間の値（上位2走の平均）。終わっている期間だけ、2走以上ある期間だけ。"""
    d = fig[fig["figure"].notna() & fig["year"].isin(years)]
    if surface is not None:
        d = d[d["track_kind"] == surface]
    d = d.assign(half=(d["date"].dt.month > 6).astype(int))
    end = pd.to_datetime(pd.DataFrame({"year": d["year"] + d["half"],
                                       "month": np.where(d["half"] == 0, 7, 1), "day": 1}))
    d = d[end <= pd.Timestamp(before)]
    keys = ["ketto_num", "year", "half", "barei"]
    d = d.sort_values("figure", ascending=False)
    d = d.assign(_k=d.groupby(keys).cumcount())
    count = d.groupby(keys).size()
    top = d[d["_k"] < TOP_RUNS].groupby(keys)["figure"].mean()
    top = top.loc[count[count >= MIN_RUNS].index]
    out = top.rename("value").reset_index().rename(columns={"barei": "age"})
    return out[["ketto_num", "year", "half", "age", "value"]].sort_values(
        ["ketto_num", "year", "half"]).reset_index(drop=True)


def period_changes(pv: pd.DataFrame) -> pd.DataFrame:
    """同じ馬の隣り合う期間（前半→後半、後半→翌年前半）の値の差。"""
    p = pv.assign(t=pv["year"] * 2 + pv["half"]).sort_values(["ketto_num", "t"])
    nxt = p.groupby("ketto_num")[["t", "age", "half", "value"]].shift(-1)
    ok = (nxt["t"] == p["t"] + 1).to_numpy()
    return pd.DataFrame({
        "ketto_num": p["ketto_num"].to_numpy()[ok],
        "to_age": nxt["age"].to_numpy()[ok].astype(int),
        "to_half": nxt["half"].to_numpy()[ok].astype(int),
        "delta": (nxt["value"] - p["value"]).to_numpy()[ok],
    })


def _levels(to_ord: list[int], deltas: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """期間 lo..hi（序数）の水準。ANCHOR を 0 にして隣の期間の変化を足し引きする。"""
    d = dict(zip(to_ord, deltas.tolist()))
    a = _ordinal(*ANCHOR)
    lv: dict[int, float] = {}
    if lo <= a <= hi:
        lv[a] = 0.0
        for o in range(a + 1, hi + 1):
            lv[o] = lv[o - 1] + d.get(o, np.nan)
        for o in range(a - 1, lo - 1, -1):
            lv[o] = lv[o + 1] - d.get(o + 1, np.nan)
    return np.array([lv.get(o, np.nan) for o in range(lo, hi + 1)])


def _interval(a: np.ndarray) -> tuple[float, float]:
    """95% 区間。全て欠測なら (nan, nan)。"""
    if np.all(np.isnan(a)):
        return np.nan, np.nan
    return float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))


def age_curve(fig: pd.DataFrame, *, years: list[int], before: date, surface: str | None = None,
              n_boot: int = 200, seed: int = 20261004) -> pd.DataFrame:
    """年齢曲線と区間（馬単位のブートストラップ）。変化が1つも無ければ空の表。"""
    ch = period_changes(period_values(fig, years=years, before=before, surface=surface))
    if ch.empty:
        return pd.DataFrame(columns=CURVE_COLUMNS)
    ords = (ch["to_age"] * 2 + ch["to_half"]).to_numpy()
    to_ord = sorted(set(ords.tolist()))
    tc = np.searchsorted(np.array(to_ord), ords)
    hc, horses = pd.factorize(ch["ketto_num"])
    delta = ch["delta"].to_numpy()
    T, H = len(to_ord), len(horses)

    def means(weights: np.ndarray) -> np.ndarray:
        num = np.bincount(tc, weights=delta * weights, minlength=T)
        den = np.bincount(tc, weights=weights, minlength=T)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    lo, hi = min(to_ord) - 1, max(to_ord)
    point = means(np.ones(len(ch)))
    rng = np.random.default_rng(seed)
    boot_d, boot_l = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, H, H), minlength=H)[hc].astype(float)
        m = means(w)
        boot_d.append(m)
        boot_l.append(_levels(to_ord, m, lo, hi))
    boot_d, boot_l = np.array(boot_d), np.array(boot_l)
    level = _levels(to_ord, point, lo, hi)
    n_by = np.bincount(tc, minlength=T)
    rows = []
    for i, o in enumerate(range(lo, hi + 1)):
        age, half = o // 2, o % 2
        j = to_ord.index(o) if o in to_ord else None
        d_lo, d_hi = _interval(boot_d[:, j]) if j is not None else (np.nan, np.nan)
        l_lo, l_hi = _interval(boot_l[:, i])
        rows.append(dict(age=age, half=half, label=period_label(age, half),
                         delta=float(point[j]) if j is not None else np.nan,
                         n=int(n_by[j]) if j is not None else 0,
                         delta_lo=d_lo, delta_hi=d_hi, level=float(level[i]),
                         level_lo=l_lo, level_hi=l_hi))
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def curve_levels(curve: pd.DataFrame) -> dict[tuple[int, int], float]:
    return {(int(r.age), int(r.half)): float(r.level) for r in curve.itertuples()}


def peak_summary(curve: pd.DataFrame, *, min_n: int = MIN_PEAK_N) -> dict:
    """全盛期と、その1年後（2期間後）までの落ち幅（正 = 落ちた）。

    - 候補は頭数 min_n 以上の期間だけ。高齢の数頭の期間は水準の区間が広く、偶然の高さを拾うため
    - 実データの曲線は 3歳後半〜5歳前半でほぼ平らで、最大の期間を1つ挙げると精度を装うことになる。
      水準の95%区間が最大の期間の区間と重なる期間（頭数 min_n 以上）を「横ばい」(plateau) として返す
    - 1年後の落ち幅は、その期間の頭数も min_n 以上のときだけ出す
    """
    empty = {"peak": None, "peak_level": float("nan"), "drop_1y": float("nan"), "plateau": []}
    c = curve.dropna(subset=["level"])
    c = c[c["n"].astype(int) >= min_n]
    if c.empty:
        return empty
    top = c.loc[c["level"].astype(float).idxmax()]
    overlap = (c["level_lo"] <= top.level_hi) & (c["level_hi"] >= top.level_lo)
    overlap |= c.index == top.name                     # 区間が欠損でも最大の期間そのものは入れる
    plateau = c[overlap].assign(_o=lambda d: d["age"] * 2 + d["half"]).sort_values("_o")["label"].tolist()
    later = c[(c["age"] * 2 + c["half"]) == _ordinal(int(top.age), int(top.half)) + 2]
    drop = float(top.level - later["level"].iloc[0]) if len(later) else float("nan")
    return {"peak": top.label, "peak_level": float(top.level), "drop_1y": drop, "plateau": plateau}
