"""世代（生まれ年）の数値と比較。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md §4・§5。

世代の数値は「その年の 1/1 から時点の前日までの、馬ごとの指数の最高値」を高い順に並べて作る。
過去の年とは同じ月日の時点で比べる（年の途中の成長を揃えるため）。
2歳は「今いる世代の並び」に入れない。2歳と3歳以上は実力差が大きすぎ、並べても比較にならない（所有者の判断）。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from keiba.evaluate.age_curve import half_of
from keiba.features.speed_figure import apply_figures, fit_model

METRICS = ("top10", "top50", "median")
TOP_K = {"top10": 10, "top50": 50}
HYPOTHESIS_PAIRS = ((4, 5), (5, 3), (4, 3))   # 所有者の見立て 4歳 > 5歳 > 3歳。左が強いと見立てた側
MIN_RANKING_AGE = 3
TABLE_COLUMNS = ["year", "age", "cohort", "n", *METRICS]


def same_monthday(as_of: date, year: int) -> date:
    """過去の年の同じ月日。2/29 は常に 2/28 にそろえる。"""
    if (as_of.month, as_of.day) == (2, 29):
        return date(year, 2, 28)
    return date(year, as_of.month, as_of.day)


def comparable_years(coverage_start: date, last_year: int) -> list[int]:
    """比べてよい年（データが 1/1 から揃う年）。coverage_start は版の取得開始日。"""
    first = coverage_start.year if (coverage_start.month, coverage_start.day) == (1, 1) else coverage_start.year + 1
    return list(range(first, last_year + 1))


def cohort_values(fig: pd.DataFrame, *, year: int, cutoff: date, cohort: int) -> dict:
    """year 年 1/1 から cutoff の前日までの、世代 cohort の数値（§4）。K 頭未満の topK は欠損。"""
    d = fig[(fig["year"] == year) & (fig["cohort"] == cohort)
            & (fig["date"] < pd.Timestamp(cutoff)) & fig["figure"].notna()]
    best = np.sort(d.groupby("ketto_num")["figure"].max().to_numpy())[::-1]
    n = len(best)
    out = {"n": n, "median": float(np.median(best)) if n else float("nan")}
    for name, k in TOP_K.items():
        out[name] = float(best[:k].mean()) if n >= k else float("nan")
    return out


def cohort_table(fig: pd.DataFrame, *, as_of: date, ages: list[int], years: list[int]) -> pd.DataFrame:
    """年 × 年齢 の世代の数値。各年の時点は as_of と同じ月日。"""
    fig = fig[fig["year"].isin(years)]
    rows = []
    for y in years:
        cutoff = same_monthday(as_of, y)
        for a in ages:
            rows.append({"year": y, "age": a, "cohort": y - a,
                         **cohort_values(fig, year=y, cutoff=cutoff, cohort=y - a)})
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def yearly_comparison(table: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """仮説①: 今年の世代 − 過去の同じ年齢の世代の平均。順位は今年を含めた中での順位（1 が最強）。"""
    rows = []
    for age, g in table.groupby("age"):
        cur = g[g["year"] == year]
        if cur.empty:
            continue
        past = g[g["year"] != year]
        for m in METRICS:
            value = float(cur[m].iloc[0])
            vals = g[m].dropna()
            past_mean = float(past[m].mean())
            rows.append({"age": int(age), "cohort": int(cur["cohort"].iloc[0]), "metric": m,
                         "current": value, "past_mean": past_mean, "diff": value - past_mean,
                         "rank": int((vals > value).sum() + 1) if not np.isnan(value) else np.nan,
                         "n_years": int(vals.size)})
    return pd.DataFrame(rows, columns=["age", "cohort", "metric", "current", "past_mean", "diff",
                                       "rank", "n_years"])


def trajectory(table: pd.DataFrame, *, year: int) -> pd.DataFrame:
    """仮説②: 今年3歳以上の各世代の、各年齢での「その年齢の他の世代の平均との差」。"""
    rows = []
    now = table[(table["year"] == year) & (table["age"] >= MIN_RANKING_AGE)]
    for cohort in sorted(now["cohort"].unique(), reverse=True):
        for r in table[table["cohort"] == cohort].sort_values("age").itertuples():
            others = table[(table["age"] == r.age) & (table["year"] != r.year)]
            for m in METRICS:
                value = float(getattr(r, m))
                rows.append({"cohort": int(cohort), "age_now": int(year - cohort), "age": int(r.age),
                             "year": int(r.year), "metric": m, "value": value,
                             "deviation": value - float(others[m].mean())})
    return pd.DataFrame(rows, columns=["cohort", "age_now", "age", "year", "metric", "value", "deviation"])


def ranking_table(table: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """年 × 年齢（3歳以上）の数値。今年の並びが例年と違うかを見る（§5.3）。"""
    t = table[table["age"] >= MIN_RANKING_AGE]
    return t.pivot(index="year", columns="age", values=metric)


def pair_differences(table: pd.DataFrame, *, year: int, pairs, curve_levels: dict | None,
                     half: int) -> pd.DataFrame:
    """見立ての照合（§5.3）。そのままの差と、年齢曲線で年齢分を差し引いた差。"""
    for a, b in pairs:
        if min(a, b) < MIN_RANKING_AGE:
            raise ValueError("2歳は今いる世代の並びに入れません（仕様 §5.3）")
    now = table[table["year"] == year].set_index("age")
    rows = []
    for a, b in pairs:
        if curve_levels is None:
            gap = float("nan")
        else:
            gap = curve_levels.get((a, half), float("nan")) - curve_levels.get((b, half), float("nan"))
        for m in METRICS:
            diff = float(now.at[a, m]) - float(now.at[b, m])
            rows.append({"pair": f"{a}-{b}", "stronger": a, "weaker": b, "metric": m,
                         "diff": diff, "age_gap": gap, "adjusted": diff - gap})
    return pd.DataFrame(rows, columns=["pair", "stronger", "weaker", "metric", "diff", "age_gap", "adjusted"])


def verdict(lo: float, hi: float) -> str:
    """区間から照合の判定（§5.3）。差は「見立てで強い側 − 弱い側」なので正が見立ての向き。"""
    if np.isnan(lo) or np.isnan(hi):
        return "判定できない"
    if lo > 0:
        return "見立てと合う"
    if hi < 0:
        return "見立てと逆"
    return "差があるとは言えない"


def replicate_stats(fig: pd.DataFrame, *, as_of: date, years: list[int], ages: list[int],
                    curve_levels: dict | None) -> dict[str, float]:
    """1回の推定から出る統計を、ブートストラップで集めやすい「キー → 値」の形にする。"""
    table = cohort_table(fig, as_of=as_of, ages=ages, years=years)
    out: dict[str, float] = {}
    for r in table.itertuples():
        for m in METRICS:
            out[f"value|{r.year}|{r.age}|{m}"] = float(getattr(r, m))
    for r in yearly_comparison(table, year=as_of.year).itertuples():
        out[f"yearly|{r.age}|{r.metric}"] = float(r.diff)
    pairs = [p for p in HYPOTHESIS_PAIRS if set(p) <= set(ages)]
    if pairs:
        pdf = pair_differences(table, year=as_of.year, pairs=pairs, curve_levels=curve_levels,
                               half=half_of(as_of))
        for r in pdf.itertuples():
            out[f"pair_raw|{r.pair}|{r.metric}"] = float(r.diff)
            out[f"pair_adj|{r.pair}|{r.metric}"] = float(r.adjusted)
    for r in trajectory(table, year=as_of.year).itertuples():
        out[f"traj|{r.cohort}|{r.age}|{r.metric}"] = float(r.deviation)
    return out


def resample_days(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """開催日を復元抽出する（§5.5）。同じ日が2回選ばれたら別の日として扱う（day と race_id に #番号）。"""
    groups = runs.groupby("day", sort=True).indices
    days = np.array(sorted(groups))
    pick = rng.choice(days, size=len(days), replace=True)
    idx = np.concatenate([groups[d] for d in pick])
    copy_no = np.repeat(np.arange(len(pick)), [len(groups[d]) for d in pick])
    out = runs.iloc[idx].reset_index(drop=True)
    tag = pd.Series(copy_no).astype(str).radd("#")
    out["day"] = out["day"].astype(str) + tag
    out["race_id"] = out["race_id"].astype(str) + tag
    return out


def bootstrap_intervals(runs: pd.DataFrame, *, as_of: date, years: list[int], ages: list[int],
                        curve_levels: dict | None, n_boot: int, seed: int) -> pd.DataFrame:
    """開催日単位のクラスタ・ブートストラップで 95% 区間（§5.5）。物差しの推定からやり直す。

    年齢分を差し引いた差では curve_levels を固定する（曲線の誤差は区間に入らない）。
    """
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        rs = resample_days(runs, rng)
        fig = apply_figures(rs, fit_model(rs))
        reps.append(replicate_stats(fig, as_of=as_of, years=years, ages=ages, curve_levels=curve_levels))
    df = pd.DataFrame(reps)
    return pd.DataFrame({"lo": df.quantile(0.025), "hi": df.quantile(0.975)})
