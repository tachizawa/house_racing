"""日より前だけの累積と縮小（keiba.factors.cumulative）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.cumulative import add_expected_top3, eb_lambda, prior_day_sums, shrunk_prior_scores
from tests.fixtures.factor_runs import STRONG_JOCKEY, assert_no_lookahead, make_factor_runs


def test_prior_day_sums_excludes_same_day_and_future():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"]),
                       "k": ["a", "a", "a", "b"], "v": [1.0, 2.0, np.nan, 5.0]})
    s, n = prior_day_sums(df, ["k"], "v")
    assert s.tolist() == [0.0, 0.0, 3.0, 0.0]
    assert n.tolist() == [0.0, 0.0, 2.0, 0.0]


def test_eb_lambda_none_when_groups_do_not_differ():
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 4000)
    k = np.repeat(np.arange(40), 100)
    assert eb_lambda(v, k) is None or eb_lambda(v, k) > 50   # 差が無いので縮小は非常に強い（または無し）
    v2 = v + np.repeat(rng.normal(0, 1, 40), 100)
    lam = eb_lambda(v2, k)
    assert lam is not None and 0.5 < lam < 2.0               # σ² = 1, τ² ≈ 1


def test_shrunk_scores_recover_strong_jockey():
    R = add_expected_top3(make_factor_runs())
    sc = shrunk_prior_scores(R, ["kisyu"], "ae", first_year=2016)
    late = (R.year == 2017).to_numpy()
    j0 = sc[late & (R.kisyu == STRONG_JOCKEY).to_numpy()].mean()
    others = sc[late & (R.kisyu != STRONG_JOCKEY).to_numpy()].mean()
    assert j0 > 0.05 and j0 > others + 0.05
    assert np.isnan(sc[(R.year == 2015).to_numpy()]).all()   # 助走の年


def test_add_expected_top3_uses_prior_days_only():
    R = make_factor_runs()
    day = R.date.sort_values().unique()[10]
    assert_no_lookahead(lambda d: add_expected_top3(d)["e3"].to_numpy(), R, day)


def test_shrunk_scores_no_lookahead():
    R = make_factor_runs()
    day = R[R.year == 2016].date.unique()[3]
    assert_no_lookahead(lambda d: shrunk_prior_scores(add_expected_top3(d), ["kisyu"], "ae", first_year=2016), R, day)
