"""世代の数値と比較（仕様 §4・§5）のテスト。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from keiba.evaluate.generation import (HYPOTHESIS_PAIRS, cohort_table, cohort_values,
                                       comparable_years, pair_differences, ranking_table,
                                       same_monthday, trajectory, verdict, yearly_comparison)


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
    assert np.isnan(pair_differences(_table(), year=2025, pairs=[(4, 5)], curve_levels=None,
                                     half=1).iloc[0].adjusted)


def test_verdict():
    assert verdict(0.1, 0.5) == "見立てと合う"
    assert verdict(-0.5, -0.1) == "見立てと逆"
    assert verdict(-0.1, 0.2) == "差があるとは言えない"
    assert verdict(float("nan"), 0.2) == "判定できない"
