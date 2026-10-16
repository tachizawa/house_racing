"""上位 n 頭の指標と確率の較正（開発方針 §5「評価指標・初期段階」）。

1. 予測上位3頭のうち実際に3着以内に入った頭数（hits、0〜3）
2. 予測上位3頭がすべて3着以内に入った割合（all3 = 三連複ボックス3頭の的中に相当）
3. 順位予測性能（NDCG@3、1着馬を上位3頭に含む率、1位予測の勝率）
4. 確率を出すモデルなら較正（brier・log loss・ECE・ビン別の表）

どの指標にも**偶然の期待値**を並べる。出走 n 頭・実際の3着以内 |A| 頭のレースで
無作為に3頭選ぶと、hits の期待値は 3|A|/n、all3 の確率は C(|A|,3)/C(n,3)。
頭数の違うレースを混ぜて「何%当たった」だけを見ると、少頭数のレースが多い期間ほど
良く見えてしまうので、必ず偶然との差で読むこと。
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


def _dcg(relevances: np.ndarray) -> float:
    return float(np.sum((2.0 ** relevances - 1.0) / np.log2(np.arange(2, len(relevances) + 2))))


def per_race(predictions: pd.DataFrame, truth: pd.DataFrame, *, top_n: int = 3
             ) -> pd.DataFrame:
    """レースごとの指標。predictions は models.base の形式、truth はデータセットの行。"""
    cols = ["race_id", "umaban", "result_rank", "target_top3"]
    merged = predictions[["race_id", "umaban", "score", "pred_rank"]].merge(
        truth[cols], on=["race_id", "umaban"], how="inner", validate="one_to_one")
    rows = []
    for race_id, race in merged.groupby("race_id", sort=True):
        n = len(race)
        k = min(top_n, n)
        actual = set(race.loc[race["target_top3"] == 1, "umaban"])
        picked = race.nsmallest(k, "pred_rank")
        hits = int(picked["umaban"].isin(actual).sum())
        rank = race["result_rank"].astype("Float64").fillna(99).to_numpy(dtype=float)
        rel_by_umaban = dict(zip(race["umaban"], np.where(rank <= 3, 4.0 - rank, 0.0)))
        rel_pred = np.array([rel_by_umaban[u] for u in picked["umaban"]])
        rel_ideal = np.sort(np.array(list(rel_by_umaban.values())))[::-1][:k]
        idcg = _dcg(rel_ideal)
        winners = set(race.loc[race["result_rank"] == 1, "umaban"])
        top1 = picked.iloc[0]["umaban"] if len(picked) else None
        a = len(actual)
        rows.append({
            "race_id": race_id, "n": n, "actual_top3": a, "hits": hits,
            "all_hit": float(hits == k and k > 0),
            "winner_in_pred": float(bool(winners & set(picked["umaban"]))),
            "top1_win": float(top1 in winners),
            "ndcg": _dcg(rel_pred) / idcg if idcg > 0 else np.nan,
            "random_hits": k * a / n if n else np.nan,
            "random_all_hit": comb(a, k) / comb(n, k) if n >= k and k > 0 else np.nan,
            "random_winner_in_pred": k / n if n else np.nan,
            "random_top1_win": len(winners) / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0 or n_boot <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def summarize(races: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """平均と、レース単位のブートストラップによる 95% 区間。偶然との差も出す。"""
    if races.empty:
        return {"races": 0}
    out: dict = {"races": int(len(races)), "mean_field_size": float(races["n"].mean())}
    for name in ("hits", "all_hit", "winner_in_pred", "top1_win", "ndcg"):
        values = races[name].to_numpy(dtype=float)
        low, high = _bootstrap_ci(values, n_boot=n_boot, seed=seed)
        entry = {"mean": float(np.nanmean(values)), "ci95": [low, high]}
        random_col = f"random_{name}"
        if random_col in races:
            chance = races[random_col].to_numpy(dtype=float)
            diff = values - chance
            d_low, d_high = _bootstrap_ci(diff, n_boot=n_boot, seed=seed)
            entry.update({"random": float(np.nanmean(chance)),
                          "lift_vs_random": float(np.nanmean(diff)),
                          "lift_ci95": [d_low, d_high]})
        out[name] = entry
    out["hits_distribution"] = {str(k): int(v) for k, v in
                                races["hits"].value_counts().sort_index().items()}
    return out


def calibration(prob: np.ndarray, y: np.ndarray, *, bins: int = 10) -> dict:
    """2値（3着以内）の確率の較正。ビンは予測確率の等幅。"""
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(prob) & ~np.isnan(y)
    prob, y = prob[mask], y[mask]
    if len(prob) == 0:
        return {}
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        sel = which == b
        if not sel.any():
            continue
        mean_p, mean_y = float(prob[sel].mean()), float(y[sel].mean())
        ece += sel.mean() * abs(mean_p - mean_y)
        table.append({"bin": [float(edges[b]), float(edges[b + 1])], "rows": int(sel.sum()),
                      "mean_pred": mean_p, "observed": mean_y})
    base = float(y.mean())
    return {
        "rows": int(len(prob)),
        "brier": float(np.mean((prob - y) ** 2)),
        "brier_base_rate": float(np.mean((base - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))),
        "ece": float(ece),
        "mean_pred": float(prob.mean()), "observed_rate": base,
        "table": table,
    }


def evaluate(predictions: pd.DataFrame, truth: pd.DataFrame, *, top_n: int = 3,
             bins: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    races = per_race(predictions, truth, top_n=top_n)
    result = {"ranking": summarize(races, n_boot=n_boot, seed=seed)}
    if predictions["prob_top3"].notna().any():
        merged = predictions.merge(truth[["race_id", "umaban", "target_top3"]],
                                   on=["race_id", "umaban"], how="inner")
        result["calibration"] = calibration(merged["prob_top3"].to_numpy(),
                                            merged["target_top3"].to_numpy(), bins=bins)
    return result
