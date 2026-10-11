"""要素別の予想の過去での評価（仕様 §5）。

評価年 Y ごとに、市場だけの着順モデルと「市場＋1要素」の着順モデルを 2016〜Y−1年で推定し、Y年のレースで
−対数尤度の差 Δ（正 = 要素で良くなった）を出す。判定は9年まとめた Δ の区間と、年ごとの安定性で行う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.order_model import add_market_features, fit_stages, place_probabilities, race_nll, valid_race_mask

EVAL_YEARS = list(range(2017, 2026))
TRAIN_FIRST = 2016
N_BOOT = 1000
SEED = 20261011
MIN_YEARS_IMPROVED = 7
MKT = ["lq", "lq2"]
ZERO_TOL = 1e-9   # これより小さい中心化した値は丸めの残りとみなして 0 にする（zscore）
Z_CLIP = 4.0      # z の上限・下限（zscore）


def topk_weights(values: np.ndarray, race_ids: np.ndarray, k: int = 3) -> np.ndarray:
    """各レースで値の大きい順に k 頭の重み。NaN は最後。境の同点は残りの枠を等分した分数（馬番で切らない）。"""
    out = np.zeros(len(values))
    vals = np.where(np.isnan(values), -np.inf, values)
    for idx in pd.Series(np.arange(len(values))).groupby(np.asarray(race_ids), sort=False).indices.values():
        v = vals[idx]
        if len(v) <= k:
            out[idx] = 1.0
            continue
        b = np.sort(v)[::-1][k - 1]
        above, tied = v > b, v == b
        out[idx[above]] = 1.0
        out[idx[tied]] = (k - above.sum()) / tied.sum()
    return out


def zscore(d: pd.DataFrame, col: str, train: np.ndarray) -> np.ndarray:
    """レース内で中心化し、学習行の中心化した値の標準偏差で割る。NaN（点数なし）はレース平均（0）にする。

    仕様 §4.2 に2点を足している（2026-10-11 の実データでの破綻への対処）:

    - 中心化した値の絶対値が ZERO_TOL 未満なら 0 とみなす。レース内で全頭同じ値（年齢の 2016年など）でも
      平均を引くと 1e−17 程度の丸めの残りが出て、学習の年がそれだけだと標準偏差が 1e−17 になり、評価の年の
      本物の差を割って z が 1e17 になった（年齢の Δ ×1000 が −8e16）。0 とみなせば標準偏差 0 → z は全部 0
      （市場だけと同じ）になる。
    - z を ±Z_CLIP で切る。表の要素は縮小の強さ λ を年ごとに推定し直すので、材料の少ない初めの年は散らばりが
      小さく、後の年で一気に広がる（父の芝ダ: 2018年の中心化した値の標準偏差 0.0011、2019年 0.0059）。学習の年の
      標準偏差で割ると評価の年の z が ±38 になり、学習の範囲外で推定した重みを当てはめて Δ が壊れた。
      学習の行も同じく切るので、学習と評価で z の範囲がそろう（学習の行で ±4 を超えるのはごく一部）。
    """
    v = d[col].to_numpy(float)
    mean = pd.Series(v).groupby(d["race_id"].to_numpy()).transform("mean").to_numpy()
    c = v - mean
    c[np.abs(c) < ZERO_TOL] = 0.0          # NaN は比較が False なので NaN のまま残る
    sd = np.nanstd(c[train]) if np.isfinite(c[train]).any() else 0.0
    z = c / sd if sd > 0 else np.zeros_like(c)
    return np.clip(np.nan_to_num(z, nan=0.0), -Z_CLIP, Z_CLIP)


def verdict(lo: float, years_improved: int) -> str:
    return "上乗せあり" if (lo > 0 and years_improved >= MIN_YEARS_IMPROVED) else "上乗せは確認できない"


def evaluate(R: pd.DataFrame, factors: list[str], *, years=EVAL_YEARS, log=print) -> dict:
    keep = valid_race_mask(R)
    in_years = R["year"].isin(years).to_numpy()
    n_races = R.loc[in_years, "race_id"].nunique()
    n_valid = R.loc[in_years & keep, "race_id"].nunique()
    d = add_market_features(R[keep & (R["year"] >= TRAIN_FIRST).to_numpy()].copy()).reset_index(drop=True)
    d = d.sort_values(["race_id", "umaban"]).reset_index(drop=True)
    parts = []
    for y in years:
        log(f"評価 {y}年 …")
        train = (d["year"] >= TRAIN_FIRST) & (d["year"] < y)
        test = (d["year"] == y).to_numpy()
        if not test.any() or not train.any():
            continue
        tr, te = d[train.to_numpy()], d[test].copy()
        w_m = fit_stages(tr, MKT)
        nll_m = race_nll(te, MKT, w_m)
        _, p_m = place_probabilities(te, MKT, w_m)
        rid = te["race_id"].to_numpy()
        top3 = te["top3"].to_numpy(float)
        e3 = te["e3"].to_numpy(float) if "e3" in te else np.full(len(te), np.nan)
        w_odds = topk_weights(-te["odds"].to_numpy(float), rid)
        w_mm = topk_weights(p_m, rid)
        # 列を1本ずつ DataFrame に足すと断片化して遅くなる（PerformanceWarning）ので、辞書に集めて最後に1回で作る
        cols = {"race_id": rid, "mkt_odds_hits": w_odds * top3, "mkt_model_hits": w_mm * top3}
        deltas = {}
        for f in factors:
            z_all = zscore(d, f, train.to_numpy())
            if not np.any(z_all[train.to_numpy()]) or not np.any(z_all[test]):
                # 点数が全部欠けている・レース内で全頭同じ → 足しても市場だけと同じ（数値誤差で判定を揺らさない）
                nll_f, p_f = nll_m, p_m
            else:
                te["_z"] = z_all[test]
                tr_z = tr.assign(_z=z_all[train.to_numpy()])
                w_f = fit_stages(tr_z, MKT + ["_z"])
                nll_f = race_nll(te, MKT + ["_z"], w_f)
                _, p_f = place_probabilities(te, MKT + ["_z"], w_f)
            deltas[f] = nll_m - nll_f
            w_solo = topk_weights(te[f].to_numpy(float), rid)
            w_comb = topk_weights(p_f, rid)
            fo = w_solo * (1 - w_odds)
            mo = w_odds * (1 - w_solo)
            cols[f"solo:{f}"] = w_solo * top3
            cols[f"comb:{f}"] = w_comb * top3
            cols[f"fo_n:{f}"], cols[f"fo_hit:{f}"], cols[f"fo_e3:{f}"] = fo, fo * top3, fo * np.nan_to_num(e3)
            cols[f"mo_n:{f}"], cols[f"mo_hit:{f}"], cols[f"mo_e3:{f}"] = mo, mo * top3, mo * np.nan_to_num(e3)
        agg = pd.DataFrame(cols).groupby("race_id", sort=False).sum()
        agg = pd.concat([agg, pd.DataFrame({f"delta:{f}": deltas[f].reindex(agg.index).to_numpy() for f in factors},
                                           index=agg.index)], axis=1)
        meta = te.groupby("race_id", sort=False).agg(date=("date", "first"), class_level=("class_level", "first"),
                                                       syubetu=("syubetu", "first"))
        agg = agg.join(meta)
        agg["year"] = y
        agg["op"] = agg["class_level"] >= 5
        agg["mixed"] = agg["syubetu"] == "13"
        parts.append(agg.reset_index())
    per_race = pd.concat(parts, ignore_index=True)
    return {"per_race": per_race, "excluded": {"races": int(n_races), "valid": int(n_valid),
                                               "excluded": int(n_races - n_valid)}}


def _cluster_ci(per_race: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    codes, uniq = pd.factorize(per_race["date"])
    sums = np.stack([np.bincount(codes, per_race[c].to_numpy(float), len(uniq)) for c in cols], axis=1)
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        means[b] = (w @ sums) / (w @ counts)
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def summarize(per_race: pd.DataFrame, factors: list[str], *, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    cols = [f"delta:{f}" for f in factors]
    lo, hi = _cluster_ci(per_race, cols, n_boot, seed)
    yearly = per_race.groupby("year")[cols].mean()
    rows = []
    for i, f in enumerate(factors):
        improved = int((yearly[f"delta:{f}"] > 0).sum())
        fo_n, mo_n = per_race[f"fo_n:{f}"].sum(), per_race[f"mo_n:{f}"].sum()
        rows.append(dict(
            factor=f, delta=per_race[f"delta:{f}"].mean(), lo=lo[i], hi=hi[i], years_improved=improved,
            verdict=verdict(lo[i], improved), solo=per_race[f"solo:{f}"].mean(),
            mkt_odds=per_race["mkt_odds_hits"].mean(), comb=per_race[f"comb:{f}"].mean(),
            mkt_model=per_race["mkt_model_hits"].mean(), fo_n=fo_n,
            fo_rate=per_race[f"fo_hit:{f}"].sum() / fo_n if fo_n else np.nan,
            fo_e3=per_race[f"fo_e3:{f}"].sum() / fo_n if fo_n else np.nan, mo_n=mo_n,
            mo_rate=per_race[f"mo_hit:{f}"].sum() / mo_n if mo_n else np.nan,
            mo_e3=per_race[f"mo_e3:{f}"].sum() / mo_n if mo_n else np.nan))
    return pd.DataFrame(rows).set_index("factor")


def by_year(per_race: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """年 × 要素の表。Δ・的中に加え、食い違いの件数・的中率・平均の期待3着内率も年ごとに出す（仕様 §5.3）。"""
    rows = []
    for y, g in per_race.groupby("year"):
        for f in factors:
            row = dict(year=y, factor=f, delta=g[f"delta:{f}"].mean(), solo=g[f"solo:{f}"].mean(),
                       mkt_odds=g["mkt_odds_hits"].mean(), comb=g[f"comb:{f}"].mean(),
                       mkt_model=g["mkt_model_hits"].mean())
            for tag in ("fo", "mo"):
                n = g[f"{tag}_n:{f}"].sum()
                row[f"{tag}_n"] = n
                row[f"{tag}_rate"] = g[f"{tag}_hit:{f}"].sum() / n if n else np.nan
                row[f"{tag}_e3"] = g[f"{tag}_e3:{f}"].sum() / n if n else np.nan
            rows.append(row)
    return pd.DataFrame(rows).set_index(["year", "factor"])
