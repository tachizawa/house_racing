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
