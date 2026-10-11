"""要素別の評価（仕様 §5.2〜5.4）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.evaluate.factor_eval import by_year, evaluate, summarize, topk_weights, verdict, zscore


def synth_eval(n_per_year=150, field=10, seed=0):
    """市場は地力だけ、要素 good は市場が知らない上乗せ、要素 noise は無関係。"""
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2016, 2026):
        for r in range(n_per_year):
            date = pd.Timestamp(year, 1 + r % 12, 1 + r % 27)
            ab = rng.normal(0, 1, field)
            extra = rng.normal(0, 1, field)
            g = rng.gumbel(size=field)
            perf = ab + 0.8 * extra + g
            order = np.argsort(-perf)
            jyuni = np.empty(field)
            jyuni[order] = np.arange(1, field + 1)
            p = np.exp(ab) / np.exp(ab).sum()
            for i in range(field):
                rows.append(dict(race_id=f"{year}{r:04d}", date=date, year=year, umaban=f"{i + 1:02d}",
                                 jyuni=jyuni[i], top3=float(jyuni[i] <= 3), field=field, odds=0.8 / p[i],
                                 e3=0.3, class_level=5.0 if r % 5 == 0 else 2.0, syubetu="13" if r % 2 else "12",
                                 good=extra[i], noise=rng.normal(), allnan=np.nan, const=1.0))
    return pd.DataFrame(rows)


def test_topk_weights_split_ties():
    v = np.array([3.0, 2.0, 2.0, 2.0, 1.0, np.nan])
    w = topk_weights(v, np.array(["a"] * 6))
    assert w.tolist() == pytest.approx([1.0, 2 / 3, 2 / 3, 2 / 3, 0.0, 0.0])


def test_zscore_missing_is_race_mean():
    d = pd.DataFrame({"race_id": ["a", "a", "a", "b", "b"], "f": [1.0, 3.0, np.nan, 5.0, 5.0]})
    z = zscore(d, "f", np.ones(5, bool))
    assert z[2] == 0.0 and z[3] == 0.0 and z[4] == 0.0
    assert z[0] < 0 < z[1]


def test_good_factor_lifts_and_noise_does_not():
    R = synth_eval()
    res = evaluate(R, ["good", "noise"], log=lambda *_: None)
    s = summarize(res["per_race"], ["good", "noise"], n_boot=200)
    assert s.loc["good", "verdict"] == "上乗せあり"
    assert s.loc["good", "lo"] > 0
    assert s.loc["noise", "verdict"] == "上乗せは確認できない"


def test_all_nan_factor_gives_no_lift():
    R = synth_eval(n_per_year=60)
    res = evaluate(R, ["allnan", "const"], log=lambda *_: None)
    s = summarize(res["per_race"], ["allnan", "const"], n_boot=100)
    for f in ("allnan", "const"):
        assert abs(s.loc[f, "delta"]) < 1e-6
        assert s.loc[f, "verdict"] == "上乗せは確認できない"


def test_by_year_and_verdict_rule():
    R = synth_eval(n_per_year=60)
    res = evaluate(R, ["good"], log=lambda *_: None)
    t = by_year(res["per_race"], ["good"])
    assert set(t.index.get_level_values("year")) == set(range(2017, 2026))
    assert verdict(0.01, 7) == "上乗せあり"
    assert verdict(0.01, 6) == "上乗せは確認できない"
    assert verdict(-0.01, 9) == "上乗せは確認できない"


def test_expanding_window_uses_only_prior_years():
    R = synth_eval(n_per_year=60)
    R2 = R.copy()
    late = R2.year == 2025
    R2.loc[late, "jyuni"] = R2.loc[late, "field"] - R2.loc[late, "jyuni"] + 1
    R2.loc[late, "top3"] = (R2.loc[late, "jyuni"] <= 3).astype(float)
    a = evaluate(R, ["good"], log=lambda *_: None)["per_race"]
    b = evaluate(R2, ["good"], log=lambda *_: None)["per_race"]
    ea, eb = a[a.year < 2025].set_index("race_id"), b[b.year < 2025].set_index("race_id")
    np.testing.assert_allclose(ea["delta:good"], eb.loc[ea.index, "delta:good"])


def test_disagreement_counts_hand_checked():
    R = synth_eval(n_per_year=40)
    res = evaluate(R, ["good"], log=lambda *_: None)
    pr = res["per_race"]
    # 要素の上位3頭にいて市場の上位3頭にいない「重み」の合計は 0〜3 の範囲
    assert pr["fo_n:good"].between(0, 3).all() and pr["mo_n:good"].between(0, 3).all()
    # 両者の重みの合計は同じ（同じ3枠を入れ替えるだけ）
    assert np.allclose(pr["fo_n:good"], pr["mo_n:good"])
