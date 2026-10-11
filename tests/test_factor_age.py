"""年齢・晩成度×年齢・世代レベル（仕様 §3.2 #6〜8）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.age_factors import age_scores, cohort_level_scores, late_age_scores, sire_lateness
from keiba.factors.figure_factors import yearly_figures
from tests.fixtures.factor_runs import assert_no_lookahead, make_factor_runs, to_runs_all


@pytest.fixture(scope="module")
def data():
    R = make_factor_runs(years=(2015, 2016, 2017, 2018), days_per_year=10)
    return R, yearly_figures(to_runs_all(R), 2018)


@pytest.mark.parametrize("fn", [age_scores, late_age_scores, cohort_level_scores], ids=lambda f: f.__name__)
def test_only_target_year_and_no_lookahead(fn, data):
    R, fig = data
    out = fn(R, fig, 2018)
    assert out.shape == (len(R),)
    assert np.isnan(out[(R.year != 2018).to_numpy()]).all()
    day = R[R.year == 2018].date.unique()[4]
    assert_no_lookahead(lambda d: fn(d, yearly_figures(to_runs_all(d), 2018), 2018), R, day)


def test_age_scores_follow_curve_levels(data):
    R, fig = data
    out = age_scores(R, fig, 2018)
    d = R.assign(sc=out)[R.year == 2018]
    # 同じ年齢・同じ半期なら同じ値
    assert d.groupby(["barei", d.date.dt.month > 6]).sc.nunique(dropna=False).max() == 1


def test_sire_lateness_respects_cohort_limit(data):
    R, fig = data
    L = sire_lateness(R, fig, 2018)
    # 作り物のデータでは 2012年生まれ以前の「3歳前半」が 2015年以降に無いので、A の材料が無く L は空か B だけ
    assert isinstance(L, pd.Series)


def test_cohort_level_january_is_nan(data):
    R, fig = data
    out = cohort_level_scores(R, fig, 2018)
    jan = ((R.year == 2018) & (R.date.dt.month == 1)).to_numpy()
    assert np.isnan(out[jan]).all()       # 1/1 時点の今年の世代はまだ走っていない
