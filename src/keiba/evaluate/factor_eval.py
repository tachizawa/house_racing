"""要素別の予想の過去での評価（仕様 §5）。

評価年 Y ごとに、市場だけの着順モデルと「市場＋1要素」の着順モデルを 2016〜Y−1年で推定し、Y年のレースで
−対数尤度の差 Δ（正 = 要素で良くなった）を出す。判定は9年まとめた Δ の区間と、年ごとの安定性で行う。

「測れなかった年」: 学習の年の z が全部 0（点数が欠けている・レース内で全頭同じ）か、評価の年の z が全部 0 なら、
要素を足しても市場だけと同じなので Δ = 0 ちょうどになる。これは「改善しなかった」ではなく「その年は測れなかった」
なので、年 × 要素ごとに measured で区別し、レポートでは Δ を「—」、改善した年を「k/m（測れた年 m）」と出す。
判定のルール（9年中7年以上で改善）は変えない（仕様 §5.2）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from keiba.factors.order_model import (add_market_features, exclusion_reasons, fit_stages, place_probabilities,
                                      race_nll, valid_race_mask)

EVAL_YEARS = list(range(2017, 2026))
TRAIN_FIRST = 2016
N_BOOT = 1000
SEED = 20261011
MIN_YEARS_IMPROVED = 7
ALPHA = 0.05
N_COMPARE = 15    # Bonferroni の参考の区間で割る比較の数（15要素）。判定には使わない
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
    reasons = exclusion_reasons(R[in_years])
    by_reason = {c: int(reasons[c].sum()) for c in reasons.columns}
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
        has_e3 = np.isfinite(e3)     # 見込み（e3）の平均は e3 のある馬だけで取る（欠けを 0 として薄めない）
        w_odds = topk_weights(-te["odds"].to_numpy(float), rid)
        w_mm = topk_weights(p_m, rid)
        # 列を1本ずつ DataFrame に足すと断片化して遅くなる（PerformanceWarning）ので、辞書に集めて最後に1回で作る
        cols = {"race_id": rid, "mkt_odds_hits": w_odds * top3, "mkt_model_hits": w_mm * top3,
                "n_runners": np.ones(len(te))}
        deltas, measured = {}, {}
        for f in factors:
            z_all = zscore(d, f, train.to_numpy())
            measured[f] = bool(np.any(z_all[train.to_numpy()]) and np.any(z_all[test]))
            cols[f"clip_n:{f}"] = (np.abs(z_all[test]) >= Z_CLIP).astype(float)
            if not measured[f]:
                # 点数が全部欠けている・レース内で全頭同じ → 足しても市場だけと同じ（数値誤差で判定を揺らさない）。
                # Δ = 0 になるが「測れなかった年」として measured で区別する
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
            for tag, wt in (("fo", fo), ("mo", mo)):
                cols[f"{tag}_n:{f}"], cols[f"{tag}_hit:{f}"] = wt, wt * top3
                cols[f"{tag}_e3:{f}"] = np.where(has_e3, wt * np.nan_to_num(e3), 0.0)
                cols[f"{tag}_e3n:{f}"] = np.where(has_e3, wt, 0.0)
        agg = pd.DataFrame(cols).groupby("race_id", sort=False).sum()
        extra = {f"delta:{f}": deltas[f].reindex(agg.index).to_numpy() for f in factors}
        extra.update({f"measured:{f}": np.full(len(agg), measured[f]) for f in factors})
        extra["nll_mkt"] = nll_m.reindex(agg.index).to_numpy()
        agg = pd.concat([agg, pd.DataFrame(extra, index=agg.index)], axis=1)
        meta = te.groupby("race_id", sort=False).agg(date=("date", "first"), class_level=("class_level", "first"),
                                                       syubetu=("syubetu", "first"))
        agg = agg.join(meta)
        agg["year"] = y
        agg["op"] = agg["class_level"] >= 5
        agg["mixed"] = agg["syubetu"] == "13"
        parts.append(agg.reset_index())
    per_race = pd.concat(parts, ignore_index=True)
    return {"per_race": per_race, "excluded": {"races": int(n_races), "valid": int(n_valid),
                                               "excluded": int(n_races - n_valid), "by_reason": by_reason}}


def _cluster_boot(per_race: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> np.ndarray:
    """開催日単位のクラスタ・ブートストラップの平均（n_boot × 列）。"""
    codes, uniq = pd.factorize(per_race["date"])
    sums = np.stack([np.bincount(codes, per_race[c].to_numpy(float), len(uniq)) for c in cols], axis=1)
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        means[b] = (w @ sums) / (w @ counts)
    return means


def _e3_mean(per_race: pd.DataFrame, tag: str, f: str) -> float:
    """食い違った馬（tag = fo / mo）の見込み3着内率の平均。e3 の無い馬は分母にも入れない。"""
    n = per_race[f"{tag}_e3n:{f}"].sum()
    return per_race[f"{tag}_e3:{f}"].sum() / n if n else np.nan


def summarize(per_race: pd.DataFrame, factors: list[str], *, n_boot: int = N_BOOT, seed: int = SEED,
              n_compare: int = N_COMPARE) -> pd.DataFrame:
    """要素ごとのまとめ。判定は 95% 区間の下限（lo）と改善した年の数で決める（仕様 §5.2。変えない）。

    - lo_bonf / hi_bonf: n_compare 比較の Bonferroni（15要素なら 1 − 0.05/15 ≈ 99.67%）の区間。15要素を同時に
      調べていることの参考に並べるだけで、判定には使わない。95% の区間と同じブートストラップの標本から取るので、
      種をそろえれば 95% の区間は v1 の実行と同じ値になる。1,000 回では端の分位点（約 0.17%）は下から2番目程度の
      標本で決まるので粗い
    - years_measured: 測れた年の数（モジュールの docstring）。years_improved は測れた年のうち Δ > 0 の年の数
    - base_nll: 市場だけのモデルの1レースあたり −対数尤度の平均。delta_pct = Δ / base_nll（大きさの目安）
    """
    cols = [f"delta:{f}" for f in factors]
    means = _cluster_boot(per_race, cols, n_boot, seed)
    lo, hi = np.quantile(means, ALPHA / 2, axis=0), np.quantile(means, 1 - ALPHA / 2, axis=0)
    a_b = ALPHA / n_compare
    lo_b, hi_b = np.quantile(means, a_b / 2, axis=0), np.quantile(means, 1 - a_b / 2, axis=0)
    yearly = per_race.groupby("year")[cols].mean()
    meas = per_race.groupby("year")[[f"measured:{f}" for f in factors]].all()
    base = float(per_race["nll_mkt"].mean())
    rows = []
    for i, f in enumerate(factors):
        m = meas[f"measured:{f}"]
        improved = int(((yearly[f"delta:{f}"] > 0) & m).sum())
        fo_n, mo_n = per_race[f"fo_n:{f}"].sum(), per_race[f"mo_n:{f}"].sum()
        delta = per_race[f"delta:{f}"].mean()
        rows.append(dict(
            factor=f, delta=delta, lo=lo[i], hi=hi[i], years_improved=improved, years_measured=int(m.sum()),
            years_total=int(len(m)), verdict=verdict(lo[i], improved), lo_bonf=lo_b[i], hi_bonf=hi_b[i],
            base_nll=base, delta_pct=delta / base if base else np.nan, solo=per_race[f"solo:{f}"].mean(),
            mkt_odds=per_race["mkt_odds_hits"].mean(), comb=per_race[f"comb:{f}"].mean(),
            mkt_model=per_race["mkt_model_hits"].mean(), fo_n=fo_n,
            fo_rate=per_race[f"fo_hit:{f}"].sum() / fo_n if fo_n else np.nan,
            fo_e3=_e3_mean(per_race, "fo", f), mo_n=mo_n,
            mo_rate=per_race[f"mo_hit:{f}"].sum() / mo_n if mo_n else np.nan,
            mo_e3=_e3_mean(per_race, "mo", f)))
    return pd.DataFrame(rows).set_index("factor")


def by_year(per_race: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """年 × 要素の表。Δ・的中に加え、食い違いの件数・的中率・平均の期待3着内率も年ごとに出す（仕様 §5.3）。

    measured はその年を測れたか（モジュールの docstring）。clipped_share は評価の行のうち z が ±Z_CLIP で切られた
    行の割合（仕様 §4.2 の追記。高い年の Δ は要素の力より物差しのずれを映しうる）。base_nll はその年の市場だけの
    モデルの1レースあたり −対数尤度の平均。
    """
    rows = []
    for y, g in per_race.groupby("year"):
        runners = g["n_runners"].sum()
        for f in factors:
            row = dict(year=y, factor=f, delta=g[f"delta:{f}"].mean(), measured=bool(g[f"measured:{f}"].all()),
                       clipped_share=g[f"clip_n:{f}"].sum() / runners if runners else np.nan,
                       base_nll=g["nll_mkt"].mean(), solo=g[f"solo:{f}"].mean(),
                       mkt_odds=g["mkt_odds_hits"].mean(), comb=g[f"comb:{f}"].mean(),
                       mkt_model=g["mkt_model_hits"].mean())
            for tag in ("fo", "mo"):
                n = g[f"{tag}_n:{f}"].sum()
                row[f"{tag}_n"] = n
                row[f"{tag}_rate"] = g[f"{tag}_hit:{f}"].sum() / n if n else np.nan
                row[f"{tag}_e3"] = _e3_mean(g, tag, f)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["year", "factor"])
