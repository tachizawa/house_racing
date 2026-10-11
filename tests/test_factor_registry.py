"""要素の一覧と一括計算のテスト。"""

from __future__ import annotations

import numpy as np

from keiba.factors.registry import FACTOR_NAMES, FACTORS, compute_all
from tests.fixtures.factor_runs import make_factor_runs, to_runs_all

EXPECTED = ["time_figure", "opponents", "jockey", "trainer", "sire_surface", "late_age", "age", "cohort_level",
            "distance_fit", "course_fit", "draw", "weight_carried", "layoff", "body_weight", "travel"]


def test_factor_list_matches_spec():
    assert FACTOR_NAMES == EXPECTED
    assert all(f.version == "v1" for f in FACTORS)
    assert [f.name for f in FACTORS if not f.forward] == ["body_weight"]


def test_compute_all_adds_every_factor():
    R = make_factor_runs(years=(2015, 2016, 2017))
    out = compute_all(R, to_runs_all(R), years=[2016, 2017], log=lambda *_: None)
    for name in EXPECTED + ["e3", "ae"]:
        assert name in out.columns
    assert len(out) == len(R)
    assert np.isnan(out.loc[out.year == 2015, "jockey"]).all()
    assert out.loc[out.year == 2017, "time_figure"].notna().mean() > 0.5
