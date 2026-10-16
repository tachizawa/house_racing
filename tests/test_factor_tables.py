"""表の要素・履歴の要素（仕様 §3.2 #3〜5, 9〜15）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors import history_factors as hf
from keiba.factors import table_factors as tf
from keiba.factors.cumulative import add_expected_top3
from tests.fixtures.factor_runs import STRONG_JOCKEY, assert_no_lookahead, make_factor_runs

TABLE = [tf.jockey, tf.trainer, tf.sire_surface, tf.travel, tf.layoff, tf.body_weight, tf.draw]
HISTORY = [hf.distance_fit, hf.course_fit, hf.weight_carried]


@pytest.fixture(scope="module")
def R():
    return add_expected_top3(make_factor_runs())


@pytest.mark.parametrize("fn", TABLE + HISTORY, ids=lambda f: f.__name__)
def test_no_lookahead(fn):
    base = make_factor_runs()
    day = base[base.year == 2016].date.unique()[4]
    assert_no_lookahead(lambda d: fn(add_expected_top3(d)), base, day)


@pytest.mark.parametrize("fn", TABLE + HISTORY, ids=lambda f: f.__name__)
def test_length_and_direction(fn, R):
    out = fn(R)
    assert out.shape == (len(R),)


def test_jockey_finds_strong_jockey(R):
    sc = tf.jockey(R)
    late = (R.year == 2017).to_numpy()
    assert sc[late & (R.kisyu == STRONG_JOCKEY).to_numpy()].mean() > sc[late & (R.kisyu != STRONG_JOCKEY).to_numpy()].mean()


def test_layoff_bins():
    R = pd.DataFrame({"ketto_num": ["A", "A", "A", "B"],
                      "date": pd.to_datetime(["2020-01-01", "2020-01-10", "2021-03-01", "2020-01-01"])})
    assert tf.layoff_bin(R).tolist() == ["不明", "〜13", "365〜", "不明"]


def test_layoff_bin_unknown_near_data_start():
    R = pd.DataFrame({"ketto_num": ["X", "A", "B"],
                      "date": pd.to_datetime(["2015-01-04", "2015-03-01", "2016-06-01"])})
    assert tf.layoff_bin(R).tolist() == ["不明", "不明", "初出走"]


def test_weight_bins():
    R = pd.DataFrame({"zogen": [-12.0, -4.0, 0.0, 4.0, 10.0, np.nan]})
    assert tf.weight_bin(R).tolist() == ["〜−10", "−9〜−4", "−3〜+3", "+4〜+9", "+10〜", "不明"]


def test_distance_fit_uses_only_prior_similar_runs():
    R = pd.DataFrame({
        "ketto_num": ["A"] * 4, "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        "track_kind": ["turf", "turf", "dirt", "turf"], "kyori": [1600, 2400, 1600, 1800], "jyo": ["05"] * 4,
        "fin_pct": [0.0, 1.0, 0.0, 0.5]})
    out = hf.distance_fit(R)
    assert np.isnan(out[0])
    assert out[3] == pytest.approx(1.0)        # 1800±200 の芝: 1走目（1−0）だけ。2400m とダートは入らない


def test_weight_carried_sign():
    R = pd.DataFrame({"race_id": ["x", "x", "x"], "futan": [54.0, 56.0, 58.0]})
    assert hf.weight_carried(R).tolist() == [2.0, 0.0, -2.0]


def test_draw_is_zero_when_no_draw_effect(R):
    sc = tf.draw(R)
    rows = (R.year >= 2016).to_numpy()
    assert np.allclose(sc[rows], 0.0, atol=0.05)   # 作り物は枠に効果を入れていない


@pytest.mark.parametrize("fn", TABLE, ids=lambda f: f.__name__)
def test_table_factor_versions_differ_only_in_lambda(fn, R):
    """v1・v2 は λ の推定だけが違う。どちらも有限で、助走の年（2015）は NaN。"""
    a, b = fn(R, version="v1"), fn(R, version="v2")
    rows = (R.year >= 2016).to_numpy()
    assert np.isfinite(a[rows]).all() and np.isfinite(b[rows]).all()
    assert np.isnan(b[~rows]).all()
    assert np.array_equal(fn(R), b, equal_nan=True)    # 既定は今の版（v2）
    # 点数が違うのは λ が違う年だけ（λ が同じ年は同じ点数、違う年は違う点数）
    lam = tf.lambda_table(R, [2016, 2017], names=(fn.__name__,)).set_index("year")
    for y in (2016, 2017):
        same_lam = (np.isnan(lam.at[y, "lambda_v1"]) and np.isnan(lam.at[y, "lambda_v2"])) \
            or lam.at[y, "lambda_v1"] == lam.at[y, "lambda_v2"]
        yr = (R.year == y).to_numpy()
        assert np.array_equal(a[yr], b[yr]) == same_lam, y


def test_table_factor_versions_differ_somewhere(R):
    """v1 と v2 で点数が変わる要素が作り物でも少なくとも1つある（version の引数が推定を選んでいる）。"""
    assert any(not np.array_equal(fn(R, version="v1"), fn(R, version="v2"), equal_nan=True) for fn in TABLE)


def test_lambda_table_has_both_estimators(R):
    t = tf.lambda_table(R, [2016, 2017], names=("jockey", "draw"))
    assert set(t.columns) == {"factor", "year", "lambda_v1", "lambda_v2"}
    assert len(t) == 4
    j = t[(t.factor == "jockey") & (t.year == 2017)].iloc[0]
    assert j.lambda_v2 > 0
