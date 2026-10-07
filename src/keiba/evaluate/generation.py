"""世代（生まれ年）の数値と比較。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md §4・§5。

世代の数値は「その年の 1/1 から時点の前日までの、馬ごとの指数の最高値」を高い順に並べて作る。
過去の年とは同じ月日の時点で比べる（年の途中の成長を揃えるため）。
2歳は「今いる世代の並び」に入れない。2歳と3歳以上は実力差が大きすぎ、並べても比較にならない（所有者の判断）。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd

from keiba.evaluate.age_curve import half_of
from keiba.features.speed_figure import FigureModel, apply_figures, fit_model

METRICS = ("top10", "top50", "median")
TOP_K = {"top10": 10, "top50": 50}
HYPOTHESIS_PAIRS = ((4, 5), (5, 3), (4, 3))   # 所有者の見立て 4歳 > 5歳 > 3歳。左が強いと見立てた側
MIN_RANKING_AGE = 3
TABLE_COLUMNS = ["year", "age", "cohort", "n", *METRICS]
PAIR_COLUMNS = ["pair", "stronger", "weaker", "metric", "diff", "age_gap", "adjusted", "past_diff", "did"]


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
    """見立ての照合（§5.3）。組ごとに4通りの差を出す。判定に使うのは did だけ。

    - diff: 今年のそのままの差
    - adjusted: 年齢曲線で年齢分を差し引いた差。同じ馬の加齢しか除かない。4・5歳は弱い馬が引退した後の
      集団で、3歳は未勝利馬まで含む集団なので、生き残り・顔ぶれの差と3歳の斤量の減量、年の途中の時期の
      ずれ（3歳の最高値は3歳前半に出やすい）が残る。毎年ほぼ同じ形で残るので参考にとどめる
    - past_diff: 過去の比べてよい年の同じ月日での、同じ組の差の平均（例年の差）
    - did: 今年の差 − 例年の差。年ごとに変わらない偏り（上の残りもの）を丸ごと除くので、
      「今年の世代の並びが例年と違うか」に直接答える。yearly[a] − yearly[b] と同じ値
    """
    for a, b in pairs:
        if min(a, b) < MIN_RANKING_AGE:
            raise ValueError("2歳は今いる世代の並びに入れません（仕様 §5.3）")
    now = table[table["year"] == year].set_index("age")
    past_mean = table[table["year"] != year].groupby("age")[list(METRICS)].mean()
    rows = []
    for a, b in pairs:
        if curve_levels is None:
            gap = float("nan")
        else:
            gap = curve_levels.get((a, half), float("nan")) - curve_levels.get((b, half), float("nan"))
        for m in METRICS:
            diff = float(now.at[a, m]) - float(now.at[b, m])
            pa = float(past_mean.at[a, m]) if a in past_mean.index else float("nan")
            pb = float(past_mean.at[b, m]) if b in past_mean.index else float("nan")
            # (今年 a − 過去 a) − (今年 b − 過去 b)。過去の平均は年齢ごとに欠損を除いて取る（yearly と同じ）
            did = (float(now.at[a, m]) - pa) - (float(now.at[b, m]) - pb)
            rows.append({"pair": f"{a}-{b}", "stronger": a, "weaker": b, "metric": m,
                         "diff": diff, "age_gap": gap, "adjusted": diff - gap,
                         "past_diff": pa - pb, "did": did})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


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
            out[f"pair_past|{r.pair}|{r.metric}"] = float(r.past_diff)
            out[f"did|{r.pair}|{r.metric}"] = float(r.did)
    for r in trajectory(table, year=as_of.year).itertuples():
        out[f"traj|{r.cohort}|{r.age}|{r.metric}"] = float(r.deviation)
    return out


def resample_days(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """開催日を復元抽出する（§5.5 の手順 1。物差しの推定し直しにだけ使う）。同じ日が2回選ばれたら別の日として扱う（day と race_id に #番号）。"""
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


def replicate_model(point: FigureModel, rep: FigureModel) -> FigureModel:
    """複製で推定し直した物差しを、元の走に当てられる形にする（§5.5 の手順 2）。

    複製の開催面は日付に「#番号」が付いているので外す（同じ日が2回選ばれた写しは同じ値になるので平均する）。
    複製に選ばれなかった日の開催面と、複製に古馬戦が無かったキーは点推定の値を使う。
    そうしないと、選ばれなかった日の走が指数を失ったり馬場差 0 で計算されたりして、
    最高値・上位K頭の平均が下に偏る（復元抽出では約 37% の日が選ばれない）。
    """
    v = rep.variants
    faces = v.index.to_frame(index=False)
    faces["day"] = faces["day"].astype(str).str.split("#").str[0]
    plain = pd.Series(v.to_numpy(), index=pd.MultiIndex.from_frame(faces))
    plain = plain.groupby(level=list(range(plain.index.nlevels))).mean()
    return replace(rep, standards=rep.standards.combine_first(point.standards),
                   variants=plain.combine_first(point.variants))


def resample_horses(fig: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """世代ごとに馬を復元抽出する（§5.5 の手順 3）。馬の全走（年をまたいでも）をまとめて動かす。

    同じ馬が2回選ばれたら別の馬として数える。ketto_num は抽出の通し番号（整数）に置き換えるので、
    元の血統登録番号は残らない（世代の数値は馬の区別にしか使わない）。
    """
    unit_keys = ["cohort", "ketto_num"]
    f = fig.sort_values(unit_keys, kind="stable").reset_index(drop=True)
    g = f.groupby(unit_keys, sort=False)
    unit = g.ngroup().to_numpy()
    counts = np.bincount(unit)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    unit_cohort = f["cohort"].to_numpy()[starts]
    picks = []
    for c in np.unique(unit_cohort):
        members = np.flatnonzero(unit_cohort == c)
        picks.append(rng.choice(members, size=len(members), replace=True))
    pick = np.concatenate(picks)
    lens = counts[pick]
    offset = np.repeat(np.cumsum(lens) - lens, lens)
    idx = np.repeat(starts[pick], lens) + (np.arange(lens.sum()) - offset)
    out = f.iloc[idx].reset_index(drop=True)
    out["ketto_num"] = np.repeat(np.arange(len(pick)), lens)
    return out


def bootstrap_replicates(runs: pd.DataFrame, *, as_of: date, years: list[int], ages: list[int],
                         curve_levels: dict | None, n_boot: int, seed: int,
                         model: FigureModel | None = None) -> pd.DataFrame:
    """ブートストラップの複製ごとの統計（行 = 複製、列 = 統計のキー）。手順は §5.5。

    1. 開催日を復元抽出して物差し（基準・クラス差・馬場差）を推定し直す（物差しの誤差）
    2. その物差しを**元の走すべて**に当てる。抽出した走に当てると、選ばれなかった日の走が消えて
       馬ごとの最高値が下がり、上位K頭の平均や中央値が下に偏る（点推定が区間の外に出ていた）
    3. 世代ごとに馬を復元抽出する（どの馬がその世代にいたかの誤差）

    model は点推定の物差し。省略すると runs から推定する。年齢分の差では curve_levels を固定する。
    """
    model = fit_model(runs) if model is None else model
    runs_in_years = runs[runs["year"].isin(years)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        rep = replicate_model(model, fit_model(resample_days(runs, rng)))
        fig = resample_horses(apply_figures(runs_in_years, rep), rng)
        reps.append(replicate_stats(fig, as_of=as_of, years=years, ages=ages, curve_levels=curve_levels))
    return pd.DataFrame(reps)


def bootstrap_intervals(runs: pd.DataFrame, *, as_of: date, years: list[int], ages: list[int],
                        curve_levels: dict | None, n_boot: int, seed: int,
                        model: FigureModel | None = None) -> pd.DataFrame:
    """95% 区間（2.5% と 97.5% の分位点）。複製の作り方は bootstrap_replicates（§5.5）。

    年齢分を差し引いた差では curve_levels を固定する（曲線の誤差は区間に入らない）。
    """
    df = bootstrap_replicates(runs, as_of=as_of, years=years, ages=ages, curve_levels=curve_levels,
                              n_boot=n_boot, seed=seed, model=model)
    return pd.DataFrame({"lo": df.quantile(0.025), "hi": df.quantile(0.975)})
