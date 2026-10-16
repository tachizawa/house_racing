"""日より前だけの累積と縮小（keiba.factors.cumulative）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.cumulative import (add_expected_top3, eb_lambda, eb_lambda_hetero, eb_variance_components,
                                     prior_day_sums, shrunk_prior_scores)
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


def _unequal_variance_groups(seed: int = 3, tau: float = 0.02):
    """実データの A−E の形を真似た作り物: 人気馬に多く乗る少数の大きな組は散らばりが大きく（σ² = 0.25）、
    多数の小さな組は散らばりが小さい（σ² = 0.01）。組の真の差 u_k は ±tau を交互に置き、τ² = tau² ちょうどにする。"""
    rng = np.random.default_rng(seed)
    sizes = [2000] * 10 + [20] * 600
    sds = [0.5] * 10 + [0.1] * 600
    u = np.where(np.arange(len(sizes)) % 2 == 0, tau, -tau)
    v = np.concatenate([u[k] + rng.normal(0, sd, n) for k, (n, sd) in enumerate(zip(sizes, sds))])
    keys = np.repeat(np.arange(len(sizes)), sizes)
    return v, keys, tau ** 2


def test_hetero_estimator_recovers_tau2_where_pooled_is_badly_biased():
    """組ごとに散らばりが違うと、散らばりを1つにまとめる旧の推定（eb_lambda）は τ² を大きく引きすぎる。
    散らばりを組ごとに引く新の推定（eb_lambda_hetero）は真の τ² の近くに来る（仕様 §2「縮小」の v2）。"""
    v, keys, tau2 = _unequal_variance_groups()
    sigma2, t2 = eb_variance_components(v, keys)
    assert t2 == pytest.approx(tau2, rel=0.3)
    sigma2_true = (2000 * 10 * 0.25 + 20 * 600 * 0.01) / (2000 * 10 + 20 * 600)   # n_k で重み付けした σ²_k = 0.16
    assert sigma2 == pytest.approx(sigma2_true, rel=0.05)
    lam_true = sigma2_true / tau2
    lam_new = eb_lambda_hetero(v, keys)
    assert lam_new == pytest.approx(lam_true, rel=0.3)
    lam_old = eb_lambda(v, keys)
    assert lam_old is None or lam_old > 3 * lam_true          # 旧は「差なし」か、縮めすぎ


def test_hetero_estimator_matches_pooled_when_variances_are_equal():
    rng = np.random.default_rng(1)
    k = np.repeat(np.arange(40), 100)
    v = rng.normal(0, 1, 4000) + np.repeat(rng.normal(0, 1, 40), 100)
    old, new = eb_lambda(v, k), eb_lambda_hetero(v, k)
    assert old is not None and new is not None and new == pytest.approx(old, rel=0.1)
    flat = eb_lambda_hetero(rng.normal(0, 1, 4000), k)
    assert flat is None or flat > 50                         # 組の差が無い → 縮小は非常に強い（または無し）
    assert eb_lambda_hetero(np.ones(3), np.array(["a", "a", "a"])) is None   # 組が1つ


def test_shrunk_scores_take_the_estimator():
    """shrunk_prior_scores は λ の推定を選べる（v1 = 旧、v2 = 新）。既定は旧のまま（v1 を再現できるように）。"""
    R = add_expected_top3(make_factor_runs())
    a = shrunk_prior_scores(R, ["kisyu"], "ae", first_year=2016)
    b = shrunk_prior_scores(R, ["kisyu"], "ae", first_year=2016, lam_fn=eb_lambda)
    c = shrunk_prior_scores(R, ["kisyu"], "ae", first_year=2016, lam_fn=eb_lambda_hetero)
    np.testing.assert_array_equal(a, b)
    rows = (R.year >= 2016).to_numpy()
    assert np.isfinite(c[rows]).all()
    assert not np.allclose(b[rows], c[rows])                 # 推定を替えると点数が変わる（引数が効いている）
    # 1行を式 (S − n·μ) / (n + λ) で確かめる（λ は v2 の推定に前年末までの行を渡したもの）
    i = int(np.flatnonzero((R.year == 2017).to_numpy() & (R.kisyu == STRONG_JOCKEY).to_numpy())[-1])
    S, n = prior_day_sums(R, ["kisyu"], "ae")
    allS, alln = prior_day_sums(R.assign(_all=0), ["_all"], "ae")
    prior = (R.date < pd.Timestamp("2017-01-01")).to_numpy()
    lam = eb_lambda_hetero(R["ae"].to_numpy(float)[prior], R["kisyu"].astype(str).to_numpy()[prior])
    assert lam is not None
    assert c[i] == pytest.approx((S[i] - n[i] * allS[i] / alln[i]) / (n[i] + lam))


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
