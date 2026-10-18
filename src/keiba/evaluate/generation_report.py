"""世代レベルの report.md を組み立てる（仕様 §8）。計算はしない。並べて言葉を付けるだけ。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from keiba.evaluate.age_curve import MIN_PEAK_N, peak_summary
from keiba.evaluate.generation import (HYPOTHESIS_PAIRS, METRICS, MIN_RANKING_AGE, ranking_table,
                                       trajectory_correlations, verdict, yearly_comparison)
from keiba.features.speed_figure import FigureModel

METRIC_JA = {"top10": "上位10頭", "top50": "上位50頭", "median": "中央値"}


@dataclass(frozen=True)
class ExampleHorse:
    umaban: str
    bamei: str
    age: int
    cohort: int


@dataclass(frozen=True)
class ReportInputs:
    as_of: date
    version_id: str
    years: list[int]
    ages: list[int]
    model: FigureModel
    fig: pd.DataFrame
    table: pd.DataFrame
    point: dict[str, float]
    ci: pd.DataFrame                    # index = 統計のキー、列 lo / hi
    curves: dict[str, pd.DataFrame]     # "全体" / "芝" / "ダート"
    repeat: tuple[float, int]
    n_boot: int
    seed: int
    examples: dict[str, list[ExampleHorse]] = field(default_factory=dict)
    example_names: dict[str, str] = field(default_factory=dict)   # race_id → レース名（見出し用）


def _num(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"


def _plain(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def _ci(inp: ReportInputs, key: str) -> tuple[float, float]:
    if key not in inp.ci.index:
        return float("nan"), float("nan")
    return float(inp.ci.at[key, "lo"]), float(inp.ci.at[key, "hi"])


def _with_ci(inp: ReportInputs, key: str) -> str:
    lo, hi = _ci(inp, key)
    return f"{_num(inp.point.get(key, float('nan')))} [{_num(lo)}, {_num(hi)}]"


def _cohort_name(cohort: int, age: int, year_now: int, year: int) -> str:
    if year == year_now:
        return f"今年の{age}歳（{cohort}年生まれ）"
    age_now = year_now - cohort
    return f"{cohort}年生まれ（今の{age_now}歳）の{age}歳時"


def _section_figures(inp: ReportInputs) -> list[str]:
    fig, m = inp.fig, inp.model
    n = len(fig)
    no_std = int((~fig["has_standard"]).sum())
    no_var = int((fig["has_standard"] & ~fig["has_variant"]).sum())
    r, pairs = inp.repeat
    return [
        "## タイム指数", "",
        "- 指数 =（基準タイム + その日の馬場差 − 走破タイム）を 1600m 相当の秒にしたもの。正ほど速い",
        f"- 基準タイム・クラス差・馬場差は古馬戦の1着だけから推定（{m.n_races:,} レース）。"
        f"馬場差の縮小: σ² = {m.sigma2:.4f}、τ² = {m.tau2:.4f}",
        f"- 対象の走: {n:,} 件。基準タイムが無い走: {no_std} 件（{100 * no_std / max(n, 1):.1f}%、指数は欠損）。"
        f"馬場差が無い走（古馬戦の無い開催面、馬場差 0 で計算）: {no_var} 件",
        f"- 再現性（同じ馬の続けて2走、120 日以内）: 相関 {_plain(r)}（{pairs:,} 組）",
        f"- 比べた年: {', '.join(map(str, inp.years))}。区間は 95% のブートストラップ（{inp.n_boot} 回、種 {inp.seed}）。"
        "複製ごとに、開催日を復元抽出して物差しを推定し直し、それを元の走すべてに当て、世代ごとに馬を復元抽出する",
        "",
    ]


def _section_yearly(inp: ReportInputs) -> list[str]:
    out = ["## 仮説① 同じ年齢での世代比較（例年比）", "",
           f"各年の {inp.as_of.month}/{inp.as_of.day} 時点、その年の 1/1 からの馬ごとの最高値で比べる。", ""]
    year_now = inp.as_of.year
    yc = yearly_comparison(inp.table, year=year_now).set_index(["age", "metric"])
    for age in inp.ages:
        t = inp.table[inp.table["age"] == age].sort_values("year")
        out += [f"### {age}歳", "", "| 世代 | 年 | 上位10頭 | 上位50頭 | 中央値 | 頭数 |", "|---|---|---|---|---|---|"]
        for r in t.itertuples():
            out.append(f"| {_cohort_name(int(r.cohort), age, year_now, int(r.year))} | {r.year} | "
                       f"{_plain(r.top10)} | {_plain(r.top50)} | {_plain(r.median)} | {r.n} |")
        rows = {m: yc.loc[(age, m)] for m in METRICS if (age, m) in yc.index}
        past = "、".join(f"{METRIC_JA[m]} {_plain(float(r['past_mean']))}" for m, r in rows.items())
        diffs = "、".join(f"{METRIC_JA[m]} {_with_ci(inp, f'yearly|{age}|{m}')}" for m in METRICS)
        ranks = "、".join(f"{METRIC_JA[m]} {_rank(r)}" for m, r in rows.items())
        out += ["", f"過去の平均: {past}", "", f"今年 − 過去の平均: {diffs}", "",
                f"過去の世代の中での順位（今年を含む。1 が最強）: {ranks}", ""]
    return out


def _rank(r) -> str:
    if np.isnan(r["rank"]):
        return "—"
    return f"{int(r['rank'])}位（{int(r['n_years'])}世代中）"


def _section_trajectory(inp: ReportInputs) -> list[str]:
    year_now = inp.as_of.year
    ages = [a for a in inp.ages if a >= MIN_RANKING_AGE]
    max_age = max(inp.ages)
    out = ["## 仮説② 世代の歩み", "",
           "各世代が各年齢だったときの「その年齢の他の世代の平均との差」（上位10頭）。空欄は比べてよい年でない。", "",
           "| 世代 | " + " | ".join(f"{a}歳時" for a in range(2, max_age + 1)) + " |",
           "|---|" + "---|" * (max_age - 1)]
    for age_now in sorted(ages):
        cohort = year_now - age_now
        cells = []
        for a in range(2, max_age + 1):
            key = f"traj|{cohort}|{a}|top10"
            cells.append(_with_ci(inp, key) if key in inp.point else "")
        out.append(f"| {cohort}年生まれ（今の{age_now}歳） | " + " | ".join(cells) + " |")
    tc = trajectory_correlations(inp.table, metric="top10")
    out += ["", "隣り合う年齢の例年比（上位10頭）の、世代をまたいだ相関"
                "（比べてよい年にある全世代。3世代未満は出さない）:", "",
            "| 年齢の組 | 世代数 | 相関 |", "|---|---|---|"]
    for r in tc.itertuples():
        out.append(f"| {r.age_from}歳時 → {r.age_to}歳時 | {r.n_cohorts} | {_num(float(r.corr))} |")
    out += ["", "世代の数が少ないので、相関は記述のためだけの数字で、世代の地の差がある証拠にはならない。", ""]
    return out


def _section_ranking(inp: ReportInputs) -> list[str]:
    out = ["## 見立て（4歳 > 5歳 > 3歳）の照合", "",
           "差は「見立てで強い側 − 弱い側」。正なら見立ての向き。**判定は「例年の差と比べた差」**"
           "（今年の差 − 過去の同じ月日の同じ組の差の平均）だけで出す。", "",
           "| 組 | 指標 | 例年の差と比べた差 [95%] | 判定 | 今年の差（参考）[95%] | 例年の差（過去の平均） "
           "| 年齢分を差し引いた差（参考）[95%] |",
           "|---|---|---|---|---|---|---|"]
    for a, b in HYPOTHESIS_PAIRS:
        if not {a, b} <= set(inp.ages):
            continue
        for m in METRICS:
            did, raw = f"did|{a}-{b}|{m}", f"pair_raw|{a}-{b}|{m}"
            past, adj = f"pair_past|{a}-{b}|{m}", f"pair_adj|{a}-{b}|{m}"
            out.append(f"| {a}歳 − {b}歳 | {METRIC_JA[m]} | {_with_ci(inp, did)} | {verdict(*_ci(inp, did))} | "
                       f"{_with_ci(inp, raw)} | {_num(inp.point.get(past, float('nan')))} | {_with_ci(inp, adj)} |")
    out += ["", "参考の2列には判定を付けない。年齢分の差し引きは同じ馬の加齢しか除かず、4・5歳は弱い馬が引退した後の"
                "集団で3歳は未勝利馬まで含む集団であること（生き残り・顔ぶれの差）と、3歳の斤量の減量の分が残る。"
                "例年の差と比べた差は、年ごとに変わらないこれらの偏りを丸ごと除く。", ""]
    for m in METRICS:
        rt = ranking_table(inp.table, metric=m)
        out += [f"例年の同じ月日の並び（{METRIC_JA[m]}）:", "",
                "| 年 | " + " | ".join(f"{a}歳" for a in rt.columns) + " |", "|---|" + "---|" * len(rt.columns)]
        for year, row in rt.iterrows():
            out.append(f"| {year} | " + " | ".join(_plain(v) for v in row.to_numpy()) + " |")
        out.append("")
    return out


def _curve_table(curve: pd.DataFrame) -> list[str]:
    if curve.empty:
        return ["データ不足", ""]
    out = ["| 期間 | 前の期間からの変化 [95%] | 頭数 | 水準（3歳後半 = 0）[95%] |", "|---|---|---|---|"]
    for r in curve.itertuples():
        n = f"{r.n}（頭数少）" if r.n < MIN_PEAK_N else f"{r.n}"
        out.append(f"| {r.label} | {_num(r.delta)} [{_num(r.delta_lo)}, {_num(r.delta_hi)}] | {n} | "
                   f"{_num(r.level)} [{_num(r.level_lo)}, {_num(r.level_hi)}] |")
    s = peak_summary(curve)
    out += ["", f"全盛期（頭数 {MIN_PEAK_N} 以上の期間だけ）: {_peak_text(s)}。"
                f"そこから1年後の落ち幅: {_plain(s['drop_1y'])} 秒", ""]
    return out


def _peak_text(s: dict) -> str:
    """横ばいの期間が2つ以上なら「A〜Bでほぼ横ばい、最大は X」。区間が重なる期間を1つの全盛期と言い切らない。"""
    if s["peak"] is None:
        return "—"
    plateau = s["plateau"]
    if len(plateau) <= 1:
        return f"{s['peak']}（水準 {_num(s['peak_level'])}）"
    # 続いていれば「A〜Bでほぼ横ばい」。間に区間の重ならない期間を挟むなら横ばいとは言えないので並べるだけ
    span = f"{plateau[0]}〜{plateau[-1]}でほぼ横ばい" if _contiguous(plateau) else f"{'・'.join(plateau)}が同程度"
    return f"{span}（水準の95%区間が最大の期間と重なる）、最大は {s['peak']} {_num(s['peak_level'])}"


def _contiguous(labels: list[str]) -> bool:
    order = [int(x.split("歳")[0]) * 2 + (0 if x.endswith("前半") else 1) for x in labels]
    return order == list(range(order[0], order[0] + len(order)))


def _section_curve(inp: ReportInputs) -> list[str]:
    out = ["## 仮説③ 年齢曲線", "",
           "同じ馬の半年ごとの変化（期間の値 = 上位2走の平均、2走以上の期間だけ）を積み上げる。", ""]
    for name in ("全体", "芝", "ダート"):
        out += [f"### {name}", ""] + _curve_table(inp.curves.get(name, pd.DataFrame()))
    out += ["大きく衰えた馬はそのまま引退して変化が残らないので、衰えは小さめに出る。", ""]
    return out


def _section_examples(inp: ReportInputs) -> list[str]:
    out = []
    max_age = max(inp.ages)
    for race_id, horses in inp.examples.items():
        name = inp.example_names.get(race_id, "")
        out += [f"## 例: {race_id}" + (f"（{name}）" if name else ""), "",
                "結果を見てから作った指標なので、**精度の証拠としては扱わない**。各馬の世代の例年比（上位10頭）。", "",
                "| 馬番 | 馬名 | 年齢 | 世代 | その年齢の例年比（同じ年齢の過去平均との差）[95%] |",
                "|---|---|---|---|---|"]
        for h in horses:
            out.append(f"| {h.umaban} | {h.bamei} | {h.age} | {h.cohort}年生まれ | "
                       f"{_with_ci(inp, f'yearly|{h.age}|top10')} |")
        out += ["", f"{max_age}歳より上の馬は比べていない（—）。比べる年齢は --ages の最大で決まる。", ""]
    return out


def render(inp: ReportInputs) -> str:
    lines = [f"# 世代レベル（{inp.as_of.isoformat()} 時点）", "",
             f"版 `{inp.version_id}`。仕様 docs/superpowers/specs/2026-10-04-generation-level-design.md。"
             "Jev は使っていない。", ""]
    lines += _section_figures(inp) + _section_yearly(inp) + _section_trajectory(inp)
    lines += _section_ranking(inp) + _section_curve(inp) + _section_examples(inp)
    lines += ["## 既知の偏り", "",
              "- スローペースのレースは時計が遅く出る（最高値・上位2走で和らげるが消えない）",
              "- 斤量は補正しない。3歳と古馬の並びでは3歳が減量の分だけ有利に出る",
              "- 比べられる過去の世代は少ない（例年比の「過去の平均」は粗い）",
              "- 古馬戦にも6月以降は3歳が混ざるので、3歳世代の強弱はわずかに馬場差へ漏れる",
              "- 年齢曲線は衰えを小さめに出す", ""]
    return "\n".join(lines)
