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


def test_by_year_has_disagreement_columns_matching_pooled():
    R = synth_eval(n_per_year=60)
    res = evaluate(R, ["good"], years=[2017], log=lambda *_: None)
    t = by_year(res["per_race"], ["good"])
    s = summarize(res["per_race"], ["good"], n_boot=50)
    for c in ("fo_n", "fo_rate", "fo_e3", "mo_n", "mo_rate", "mo_e3"):
        assert c in t.columns
        assert t.loc[(2017, "good"), c] == pytest.approx(s.loc["good", c])


def test_by_year_disagreement_nan_when_empty():
    pr = pd.DataFrame({"year": [2017], "delta:f": [0.0], "solo:f": [0.0], "comb:f": [0.0], "mkt_odds_hits": [0.0],
                       "mkt_model_hits": [0.0], "fo_n:f": [0.0], "fo_hit:f": [0.0], "fo_e3:f": [0.0],
                       "mo_n:f": [0.0], "mo_hit:f": [0.0], "mo_e3:f": [0.0]})
    t = by_year(pr, ["f"])
    assert t.loc[(2017, "f"), "fo_n"] == 0 and np.isnan(t.loc[(2017, "f"), "fo_rate"])


def test_expanding_window_uses_only_prior_years(monkeypatch):
    from keiba.evaluate import factor_eval as fe

    R = synth_eval(n_per_year=60)
    seen = []
    orig = fe.fit_stages

    def spy(d, feats):
        seen.append((int(d["year"].min()), int(d["year"].max())))
        return orig(d, feats)

    monkeypatch.setattr(fe, "fit_stages", spy)
    evaluate(R, ["good"], log=lambda *_: None)
    # 評価年ごとに（市場だけ・市場＋要素）の2回推定する。学習は 2016〜Y−1年だけ
    assert len(seen) == 2 * len(fe.EVAL_YEARS)
    for y, (lo, hi) in zip(fe.EVAL_YEARS, seen[0::2]):
        assert lo == 2016 and hi == y - 1
    for y, (lo, hi) in zip(fe.EVAL_YEARS, seen[1::2]):
        assert lo == 2016 and hi == y - 1


def test_future_results_do_not_change_past_deltas():
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
    """1レース6頭を手で計算する。オッズ上位3頭 = 1〜3番。要素の値は 3番手が3頭同点（境界の同点）で NaN も含む。"""
    train = synth_eval(n_per_year=30)
    train = train[train.year == 2016]
    n = 6
    hand = pd.DataFrame(dict(
        race_id="H0001", date=pd.Timestamp(2017, 5, 1), year=2017, umaban=[f"{i + 1:02d}" for i in range(n)],
        jyuni=[3.0, 1.0, 6.0, 2.0, 4.0, 5.0], top3=[1.0, 1.0, 0.0, 1.0, 0.0, 0.0], field=n,
        odds=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], e3=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        class_level=2.0, syubetu="13", good=[0.0, 9.0, 7.0, 7.0, 7.0, np.nan]))
    R = pd.concat([train, hand], ignore_index=True)
    pr = evaluate(R, ["good"], years=[2017], log=lambda *_: None)["per_race"]
    assert len(pr) == 1
    r = pr.iloc[0]
    # 要素の重み = [0, 1, 2/3, 2/3, 2/3, 0]（9点が1枠、残り2枠を7点3頭で等分）、オッズの重み = [1, 1, 1, 0, 0, 0]
    # 要素だけが選ぶ(fo) = 4番・5番に各2/3、オッズだけが選ぶ(mo) = 1番に1、3番に1/3
    assert r["fo_n:good"] == pytest.approx(4 / 3)
    assert r["fo_hit:good"] == pytest.approx(2 / 3)           # 4番は3着内、5番は圏外
    assert r["fo_e3:good"] == pytest.approx(2 / 3 * 40 + 2 / 3 * 50)
    assert r["mo_n:good"] == pytest.approx(4 / 3)
    assert r["mo_hit:good"] == pytest.approx(1.0)             # 1番は3着内、3番は圏外
    assert r["mo_e3:good"] == pytest.approx(10 + 30 / 3)
    assert r["solo:good"] == pytest.approx(1 + 2 / 3)         # 2番(3着内) と 4番(3着内)の2/3
    assert r["mkt_odds_hits"] == pytest.approx(2.0)           # 1番・2番が3着内、3番は圏外


