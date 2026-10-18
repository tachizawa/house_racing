"""タイム指数・相手関係（仕様 §3.2 #1, 2）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.figure_factors import (level_before, opponents_scores, time_figure_scores, yearly_figures,
                                          yearly_model)
from tests.fixtures.factor_runs import assert_no_lookahead, make_factor_runs, to_runs_all


def test_yearly_model_uses_only_prior_years_for_standards():
    R = make_factor_runs()
    m1 = yearly_model(to_runs_all(R), 2017)
    R2 = R.copy()
    R2.loc[R2.year == 2017, "time_sec"] += 5.0
    m2 = yearly_model(to_runs_all(R2), 2017)
    pd.testing.assert_series_equal(m1.standards, m2.standards)
    # 2017年の日の馬場差は、その日のレースが遅くなった分だけ動く
    d17 = [k for k in m1.variants.index if k[0].startswith("2017")]
    assert (m2.variants[d17] > m1.variants[d17]).all()


def test_level_before_window_and_top2():
    fig = pd.DataFrame({"ketto_num": ["A"] * 4, "figure": [1.0, 3.0, 2.0, 9.0],
                        "date": pd.to_datetime(["2019-01-01", "2020-01-01", "2020-06-01", "2020-07-01"])})
    tg = pd.DataFrame({"ketto_num": ["A", "A", "B"], "date": pd.to_datetime(["2020-07-01", "2019-01-01", "2020-07-01"])})
    out = level_before(fig, tg)
    assert out[0] == pytest.approx(2.5)        # 2019-07-02 以降・当日を除く → 3.0 と 2.0
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_time_figure_no_lookahead():
    R = make_factor_runs()
    day = R[R.year == 2017].date.unique()[3]

    def compute(d):
        return time_figure_scores(d, yearly_figures(to_runs_all(d), 2017), 2017)

    assert_no_lookahead(compute, R, day)


def test_opponents_no_lookahead():
    R = make_factor_runs()
    day = R[R.year == 2017].date.unique()[3]

    def compute(d):
        return opponents_scores(d, yearly_figures(to_runs_all(d), 2017), 2017)

    assert_no_lookahead(compute, R, day)


def test_scores_only_for_target_year():
    R = make_factor_runs()
    fig = yearly_figures(to_runs_all(R), 2017)
    tf_ = time_figure_scores(R, fig, 2017)
    op = opponents_scores(R, fig, 2017)
    assert np.isnan(tf_[(R.year != 2017).to_numpy()]).all()
    assert np.isnan(op[(R.year != 2017).to_numpy()]).all()
    assert np.isfinite(tf_[(R.year == 2017).to_numpy()]).mean() > 0.5


def test_time_figure_tracks_ability():
    R = make_factor_runs()
    fig = yearly_figures(to_runs_all(R), 2017)
    sc = time_figure_scores(R, fig, 2017)
    d = R.assign(sc=sc)[R.year == 2017].dropna(subset=["sc"])
    # 指数が高い馬ほど着順の割合が小さい（よい）
    assert np.corrcoef(d.sc, d.fin_pct)[0, 1] < -0.2
