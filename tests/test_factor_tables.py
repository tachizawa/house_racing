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
    assert tf.layoff_bin(R).tolist() == ["初出走", "〜13", "365〜", "初出走"]


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
