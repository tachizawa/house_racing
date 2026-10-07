"""世代の数値と比較（仕様 §4・§5）のテスト。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from keiba.evaluate.generation import (HYPOTHESIS_PAIRS, cohort_table, cohort_values,
                                       comparable_years, pair_differences, ranking_table,
                                       same_monthday, trajectory, verdict, yearly_comparison)
from keiba.evaluate.generation import bootstrap_intervals, replicate_stats, resample_days
from keiba.features.speed_figure import apply_figures, fit_model
from tests.fixtures.figure_runs import season_runs


def fig_rows(year, cohort, figures, *, month=5, prefix="H"):
    return [dict(ketto_num=f"{prefix}{cohort}-{i}", year=year, cohort=cohort,
                 date=pd.Timestamp(year, month, 1), figure=f) for i, f in enumerate(figures)]


def test_same_monthday_and_leap_day():
    assert same_monthday(date(2026, 10, 4), 2024) == date(2024, 10, 4)
    assert same_monthday(date(2024, 2, 29), 2023) == date(2023, 2, 28)
    assert same_monthday(date(2024, 2, 29), 2020) == date(2020, 2, 28)


def test_comparable_years_start_after_partial_year():
    assert comparable_years(date(2021, 9, 1), 2026) == [2022, 2023, 2024, 2025, 2026]
    assert comparable_years(date(2022, 1, 1), 2024) == [2022, 2023, 2024]


def test_cohort_values_best_per_horse_and_top_k():
    rows = fig_rows(2025, 2022, range(60))
    rows += [dict(ketto_num="H2022-0", year=2025, cohort=2022, date=pd.Timestamp(2025, 6, 1), figure=100.0)]
    v = cohort_values(pd.DataFrame(rows), year=2025, cutoff=date(2025, 10, 4), cohort=2022)
    best = sorted([100.0] + list(range(1, 60)), reverse=True)    # H2022-0 の最高値は 100
    assert v["n"] == 60
    assert v["top10"] == pytest.approx(np.mean(best[:10]))
    assert v["top50"] == pytest.approx(np.mean(best[:50]))
    assert v["median"] == pytest.approx(np.median(best))


def test_cohort_values_respects_cutoff_and_missing_top_k():
    rows = fig_rows(2025, 2022, [1.0, 2.0], month=5) + fig_rows(2025, 2022, [9.0], month=11, prefix="L")
    v = cohort_values(pd.DataFrame(rows), year=2025, cutoff=date(2025, 10, 4), cohort=2022)
    assert v["n"] == 2 and np.isnan(v["top10"]) and v["median"] == pytest.approx(1.5)


def test_cohort_values_empty_cohort():
    v = cohort_values(pd.DataFrame(fig_rows(2025, 2022, [1.0])), year=2025, cutoff=date(2025, 10, 4), cohort=2023)
    assert v["n"] == 0 and np.isnan(v["median"])


def _table():
    rows = []
    for year, shift in ((2023, 0.0), (2024, 1.0), (2025, -1.0)):
        for age, base in ((2, -3.0), (3, 0.0), (4, 1.0), (5, 0.5)):
            rows += fig_rows(year, year - age, np.full(60, base + shift + (age == 3) * (year == 2025) * -0.5))
    fig = pd.DataFrame(rows)
    return cohort_table(fig, as_of=date(2025, 10, 4), ages=[2, 3, 4, 5], years=[2023, 2024, 2025])


def test_cohort_table_shape():
    t = _table()
    assert list(t.columns) == ["year", "age", "cohort", "n", "top10", "top50", "median"]
    assert len(t) == 12
    assert t.set_index(["year", "age"]).at[(2024, 3), "cohort"] == 2021


def test_yearly_comparison_against_past_mean():
    yc = yearly_comparison(_table(), year=2025).set_index(["age", "metric"])
    row = yc.loc[(3, "top10")]
    assert row.current == pytest.approx(-1.5)          # 0 − 1（年の分）− 0.5
    assert row.past_mean == pytest.approx(0.5)          # 2023 の 0 と 2024 の 1
    assert row["diff"] == pytest.approx(-2.0)
    assert row["rank"] == 3 and row.n_years == 3


def test_trajectory_lines_up_each_cohort_by_age():
    tr = trajectory(_table(), year=2025)
    assert set(tr.cohort) == {2022, 2021, 2020}          # 2025年に3歳以上の世代だけ
    cell = tr[(tr.cohort == 2021) & (tr.age == 3) & (tr.metric == "top10")].iloc[0]
    assert cell.year == 2024
    assert cell.deviation == pytest.approx(1.0 - np.mean([0.0, -1.5]))   # 2024年の3歳 − 他の年の3歳


def test_ranking_table_excludes_two_year_olds():
    rt = ranking_table(_table(), metric="top10")
    assert 2 not in rt.columns and list(rt.columns) == [3, 4, 5]


def test_pair_differences_rejects_two_year_olds():
    with pytest.raises(ValueError):
        pair_differences(_table(), year=2025, pairs=[(3, 2)], curve_levels=None, half=1)


def test_pair_differences_age_adjustment():
    levels = {(3, 1): 0.0, (4, 1): 1.0, (5, 1): 0.5}
    pdf = pair_differences(_table(), year=2025, pairs=HYPOTHESIS_PAIRS, curve_levels=levels,
                           half=1).set_index(["pair", "metric"])
    row = pdf.loc[("4-3", "top10")]
    assert row["diff"] == pytest.approx(0.0 - (-1.5))
    assert row.age_gap == pytest.approx(1.0)
    assert row.adjusted == pytest.approx(1.5 - 1.0)
    assert row.past_diff == pytest.approx(1.0)          # 2023: 1 − 0、2024: 2 − 1
    assert row.did == pytest.approx(1.5 - 1.0)           # 今年の 3歳だけ 0.5 弱い分
    yc = yearly_comparison(_table(), year=2025).set_index(["age", "metric"])["diff"]
    assert row.did == pytest.approx(yc[(4, "top10")] - yc[(3, "top10")])
    assert np.isnan(pair_differences(_table(), year=2025, pairs=[(4, 5)], curve_levels=None,
                                     half=1).iloc[0].adjusted)


def test_verdict():
    assert verdict(0.1, 0.5) == "見立てと合う"
    assert verdict(-0.5, -0.1) == "見立てと逆"
    assert verdict(-0.1, 0.2) == "差があるとは言えない"
    assert verdict(float("nan"), 0.2) == "判定できない"


def _season_fig():
    runs = season_runs()
    return runs, apply_figures(runs, fit_model(runs))


def test_replicate_stats_keys():
    _, fig = _season_fig()
    stats = replicate_stats(fig, as_of=date(2025, 10, 4), years=[2024, 2025], ages=[2, 3, 4, 5],
                            curve_levels={(3, 1): 0.0, (4, 1): 0.2, (5, 1): 0.1})
    for key in ("value|2025|3|median", "yearly|2|median", "pair_raw|4-5|median",
                "pair_adj|4-3|median", "pair_past|4-3|median", "did|4-3|median", "traj|2021|3|median"):
        assert key in stats
    assert not any(k.startswith("pair_") and ("-2|" in k or "|2-" in k) for k in stats)


def test_resample_days_keeps_each_day_whole():
    runs, _ = _season_fig()
    out = resample_days(runs, np.random.default_rng(0))
    assert out["day"].str.split("#").str[0].isin(runs["day"]).all()
    sizes = runs.groupby("day").size()
    for label, g in out.groupby("day"):
        assert len(g) == sizes[label.split("#")[0]]       # 選ばれた日は丸ごと入る
    assert out["day"].nunique() == runs["day"].nunique()   # 復元抽出の回数 = 日数（同じ日の写しは別名）


def test_bootstrap_is_reproducible_and_brackets_point():
    runs, fig = _season_fig()
    kw = dict(as_of=date(2025, 10, 4), years=[2024, 2025], ages=[2, 3, 4, 5], curve_levels=None)
    a = bootstrap_intervals(runs, n_boot=15, seed=11, **kw)
    b = bootstrap_intervals(runs, n_boot=15, seed=11, **kw)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["lo", "hi"]
    point = replicate_stats(fig, **kw)["value|2025|4|median"]
    lo, hi = a.loc["value|2025|4|median"]
    assert lo - 0.5 <= point <= hi + 0.5


# --- 見立ての照合を端から端まで（仕様 §9「年齢分を差し引いた差」「例年の差と比べた差」） ---

# 真の年齢曲線（3歳後半 = 0）。test_age_curve と同じ形
TRUE_CURVE = {(2, 1): -2.0, (3, 0): -1.0, (3, 1): 0.0, (4, 0): 0.3, (4, 1): 0.4, (5, 0): 0.3, (5, 1): 0.1}
E2E_YEARS = list(range(2018, 2026))
E2E_AS_OF = date(2025, 10, 4)


def stable_population_fig(*, n: int = 200) -> pd.DataFrame:
    """毎年まったく同じ構造の世代を並べた作り物（世代の強弱は無い）。

    馬の指数 = 地力 + 年齢の水準。弱い馬ほど早く引退する（下位 50% は 3歳後半で、下位 70% は 4歳後半で）。
    年齢曲線は同じ馬の変化から正しく取れるが、4・5歳は強い馬だけが残った集団なので、
    3歳と並べると「年齢分を差し引いても」古馬が強く見える。毎年同じなので、例年の差と比べれば 0 になるはず。
    """
    abilities = np.linspace(-2.0, 2.0, n)
    rows = []
    for cohort in range(min(E2E_YEARS) - 5, max(E2E_YEARS) - 1):
        for i, ab in enumerate(abilities):
            for (age, half), lvl in TRUE_CURVE.items():
                if (age, half) >= (4, 0) and i < 0.5 * n:
                    break
                if (age, half) >= (5, 0) and i < 0.7 * n:
                    break
                year = cohort + age
                if year not in E2E_YEARS:
                    continue
                for month in ((3, 5) if half == 0 else (8, 9)):
                    rows.append(dict(ketto_num=f"H{cohort}-{i}", year=year, cohort=cohort,
                                     date=pd.Timestamp(year, month, 1), barei=age, track_kind="turf",
                                     figure=float(ab + lvl)))
    return pd.DataFrame(rows)


def test_pair_difference_against_usual_gap_removes_survivorship():
    """年齢分を差し引いても生き残りの偏りは残る。例年の差と比べた差（did）なら 0 になり、判定も差なしになる。"""
    from keiba.evaluate.age_curve import age_curve, curve_levels, half_of

    fig = stable_population_fig()
    curve = age_curve(fig, years=E2E_YEARS, before=E2E_AS_OF, n_boot=5, seed=1)
    levels = curve_levels(curve)
    for period, lvl in TRUE_CURVE.items():
        if period != (2, 1):
            assert levels[period] == pytest.approx(lvl, abs=1e-9)     # 曲線そのものは正しく取れている
    table = cohort_table(fig, as_of=E2E_AS_OF, ages=[2, 3, 4, 5], years=E2E_YEARS)
    pdf = pair_differences(table, year=2025, pairs=HYPOTHESIS_PAIRS, curve_levels=levels,
                           half=half_of(E2E_AS_OF)).set_index(["pair", "metric"])
    assert abs(pdf.loc[("4-3", "median"), "adjusted"]) > 0.5           # 年齢分を引いても 0 にならない
    assert abs(pdf.loc[("5-3", "median"), "adjusted"]) > 0.5
    for pair in ("4-5", "5-3", "4-3"):
        for m in ("top10", "top50", "median"):
            assert pdf.loc[(pair, m), "did"] == pytest.approx(0.0, abs=1e-9)
    stats = replicate_stats(fig, as_of=E2E_AS_OF, years=E2E_YEARS, ages=[2, 3, 4, 5], curve_levels=levels)
    assert stats["did|4-3|median"] == pytest.approx(0.0, abs=1e-9)
    assert verdict(-0.1 + stats["did|4-3|median"], 0.1 + stats["did|4-3|median"]) == "差があるとは言えない"
