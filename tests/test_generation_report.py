"""report.md の組み立て（仕様 §8）。欠損・データ不足でも落ちずに「—」「データ不足」を出す。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from keiba.evaluate.age_curve import CURVE_COLUMNS, age_curve
from keiba.evaluate.generation import cohort_table, replicate_stats
from keiba.evaluate.generation_report import ExampleHorse, ReportInputs, render
from keiba.features.speed_figure import apply_figures, fit_model
from tests.fixtures.figure_runs import season_runs

AS_OF = date(2025, 10, 4)
YEARS = [2024, 2025]
AGES = [2, 3, 4, 5]


def _inputs(**over) -> ReportInputs:
    runs = season_runs()
    model = fit_model(runs)
    fig = apply_figures(runs, model)
    table = cohort_table(fig, as_of=AS_OF, ages=AGES, years=YEARS)
    point = replicate_stats(fig, as_of=AS_OF, years=YEARS, ages=AGES, curve_levels=None)
    ci = pd.DataFrame({"lo": {k: v - 0.1 for k, v in point.items()},
                       "hi": {k: v + 0.1 for k, v in point.items()}})
    curves = {"全体": age_curve(fig, years=YEARS, before=AS_OF, n_boot=5, seed=1),
              "芝": pd.DataFrame(columns=CURVE_COLUMNS), "ダート": pd.DataFrame(columns=CURVE_COLUMNS)}
    base = dict(as_of=AS_OF, version_id="TEST", years=YEARS, ages=AGES, model=model, fig=fig,
                table=table, point=point, ci=ci, curves=curves, repeat=(0.5, 100), n_boot=5, seed=1,
                examples={"2025100405040211": [ExampleHorse("01", "テストウマ", 3, 2022)]})
    base.update(over)
    return ReportInputs(**base)


def test_report_has_all_sections():
    text = render(_inputs())
    for heading in ("# 世代レベル", "## タイム指数", "## 仮説① 同じ年齢での世代比較",
                    "## 仮説② 世代の歩み", "## 見立て（4歳 > 5歳 > 3歳）の照合",
                    "## 仮説③ 年齢曲線", "## 例: 2025100405040211", "## 既知の偏り"):
        assert heading in text
    assert "精度の証拠としては扱わない" in text


def test_report_shows_missing_standard_count():
    inputs = _inputs()
    fig = inputs.fig.copy()
    fig.loc[fig.index[:3], "has_standard"] = False
    text = render(_inputs(fig=fig))
    assert "基準タイムが無い走: 3 件" in text


def test_report_renders_missing_values_as_dash():
    inputs = _inputs()
    point = {k: float("nan") for k in inputs.point}
    text = render(_inputs(point=point))
    assert "—" in text and "判定できない" in text


def test_report_handles_empty_curve():
    text = render(_inputs())
    assert "### 芝\n\nデータ不足" in text


def test_ranking_section_has_no_two_year_olds():
    text = render(_inputs())
    section = text.split("## 見立て（4歳 > 5歳 > 3歳）の照合")[1].split("## 仮説③")[0]
    assert "2歳" not in section


def _section(text: str, start: str, end: str) -> str:
    return text.split(start)[1].split(end)[0]


def test_ranking_verdict_comes_from_difference_against_usual_gap():
    """§5.3: 判定は「例年の差と比べた差」だけで出す。そのまま・年齢分の差は参考で判定を付けない。"""
    inputs = _inputs()
    ci = inputs.ci.copy()
    ci[["lo", "hi"]] = (-1.0, 1.0)                     # 他の行はすべて「差があるとは言えない」
    ci.loc["did|4-3|top10"] = (0.2, 0.4)
    ci.loc["pair_raw|4-3|top10"] = (-0.4, -0.2)
    ci.loc["pair_adj|4-3|top10"] = (-0.4, -0.2)
    section = _section(render(_inputs(ci=ci)), "## 見立て（4歳 > 5歳 > 3歳）の照合", "## 仮説③")
    row = next(line for line in section.splitlines() if line.startswith("| 4歳 − 3歳 | 上位10頭 |"))
    assert "見立てと合う" in row and section.count("見立てと合う") == 1 and "見立てと逆" not in section
    assert "例年の差と比べた差" in section and "同じ馬の加齢" in section and "斤量" in section
    for m in ("上位10頭", "上位50頭", "中央値"):
        assert f"例年の同じ月日の並び（{m}）" in section


def test_yearly_section_shows_past_mean_and_rank():
    section = _section(render(_inputs()), "## 仮説① 同じ年齢での世代比較", "## 仮説②")
    assert "過去の平均:" in section and "過去の世代の中での順位" in section and "位（2世代中）" in section


def test_trajectory_section_shows_correlations_descriptively():
    section = _section(render(_inputs()), "## 仮説② 世代の歩み", "## 見立て")
    assert "| 2歳時 → 3歳時 |" in section and "世代数" in section and "証拠" in section


def test_curve_marks_small_periods_and_plateau():
    curve = pd.DataFrame([
        dict(age=a, half=h, label=f"{a}歳{'前半' if h == 0 else '後半'}", delta=0.0, n=n, delta_lo=0.0,
             delta_hi=0.0, level=lv, level_lo=lo, level_hi=hi)
        for a, h, lv, lo, hi, n in [(3, 1, 0.0, 0.0, 0.0, 600), (4, 0, 0.09, 0.06, 0.11, 400),
                                    (4, 1, 0.03, 0.00, 0.07, 350), (5, 0, 0.05, 0.01, 0.09, 290),
                                    (11, 1, 0.5, 0.1, 0.9, 2)]], columns=CURVE_COLUMNS)
    text = render(_inputs(curves={"全体": curve, "芝": curve.iloc[:0], "ダート": curve.iloc[:0]}))
    assert "| 11歳後半 |" in text and "2（頭数少）" in text
    assert "4歳前半〜5歳前半でほぼ横ばい" in text and "最大は 4歳前半 +0.09" in text


def test_example_heading_names_race_and_explains_older_horses():
    text = render(_inputs(example_names={"2025100405040211": "毎日王冠"}))
    assert "## 例: 2025100405040211（毎日王冠）" in text
    assert "その年齢の例年比（同じ年齢の過去平均との差）" in text
    assert "5歳より上" in text
