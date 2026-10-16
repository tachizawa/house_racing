"""要素の一覧と一括計算のテスト。"""

from __future__ import annotations

import numpy as np

from keiba.factors import table_factors as tf
from keiba.factors.cumulative import add_expected_top3
from keiba.factors.registry import (FACTOR_NAMES, FACTORS, VERSIONS, compute_all, recompute_table_factors,
                                    versions_tag)
from tests.fixtures.factor_runs import make_factor_runs, to_runs_all

EXPECTED = ["time_figure", "opponents", "jockey", "trainer", "sire_surface", "late_age", "age", "cohort_level",
            "distance_fit", "course_fit", "draw", "weight_carried", "layoff", "body_weight", "travel"]


def test_factor_list_matches_spec():
    assert FACTOR_NAMES == EXPECTED
    table = {"jockey", "trainer", "sire_surface", "draw", "layoff", "body_weight", "travel"}
    # 表の要素は縮小の推定を直した v2、ほかは v1（final-findings #1）
    assert {f.name: f.version for f in FACTORS} == {n: ("v2" if n in table else "v1") for n in EXPECTED}
    assert [f.name for f in FACTORS if not f.forward] == ["body_weight"]


def test_compute_all_adds_every_factor():
    R = make_factor_runs(years=(2015, 2016, 2017))
    out = compute_all(R, to_runs_all(R), years=[2016, 2017], log=lambda *_: None)
    for name in EXPECTED + ["e3", "ae"]:
        assert name in out.columns
    assert len(out) == len(R)
    assert np.isnan(out.loc[out.year == 2015, "jockey"]).all()
    assert out.loc[out.year == 2017, "time_figure"].notna().mean() > 0.5


def test_recompute_table_factors_uses_current_versions():
    R = add_expected_top3(make_factor_runs(years=(2015, 2016, 2017)))
    stale = R.assign(**{name: np.nan for name in tf.TABLE_FACTORS})
    out = recompute_table_factors(stale, log=lambda *_: None)
    expected = {"jockey": tf.jockey, "trainer": tf.trainer, "sire_surface": tf.sire_surface, "draw": tf.draw,
                "layoff": tf.layoff, "body_weight": tf.body_weight, "travel": tf.travel}
    assert set(tf.TABLE_FACTORS) == set(expected)
    for name, fn in expected.items():                   # 表の要素7つすべてを、その名前の関数の v2 と比べる
        np.testing.assert_array_equal(out[name].to_numpy(), fn(R, version="v2"), err_msg=name)  # NaN 同士は等しい扱い
        assert VERSIONS[name] == "v2"
    assert "jockey" not in R.columns                   # 元の表は変えない


def test_versions_tag_changes_with_versions():
    a = versions_tag({"jockey": "v1", "age": "v1"})
    assert a == versions_tag({"age": "v1", "jockey": "v1"})        # 並び順に依らない
    assert a != versions_tag({"jockey": "v2", "age": "v1"})
    assert len(versions_tag()) == 8
