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
                       "fo_e3n:f": [0.0], "mo_n:f": [0.0], "mo_hit:f": [0.0], "mo_e3:f": [0.0], "mo_e3n:f": [0.0],
                       "measured:f": [True],
                       "clip_n:f": [0.0], "n_runners": [8.0], "nll_mkt": [5.0]})
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
    assert r["fo_e3n:good"] == pytest.approx(4 / 3) and r["mo_e3n:good"] == pytest.approx(4 / 3)


def test_missing_e3_is_left_out_of_the_expected_rate():
    """e3 の無い馬は見込みの平均の分母にも入れない（0 として数えると見込みが低く出る）。"""
    train = synth_eval(n_per_year=30)
    train = train[train.year == 2016]
    n = 6
    hand = pd.DataFrame(dict(
        race_id="H0001", date=pd.Timestamp(2017, 5, 1), year=2017, umaban=[f"{i + 1:02d}" for i in range(n)],
        jyuni=[3.0, 1.0, 6.0, 2.0, 4.0, 5.0], top3=[1.0, 1.0, 0.0, 1.0, 0.0, 0.0], field=n,
        odds=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], e3=[10.0, 20.0, 30.0, np.nan, 50.0, 60.0],
        class_level=2.0, syubetu="13", good=[0.0, 9.0, 7.0, 7.0, 7.0, np.nan]))
    pr = evaluate(pd.concat([train, hand], ignore_index=True), ["good"], years=[2017], log=lambda *_: None)["per_race"]
    # 要素だけが選ぶ馬 = 4番（e3 なし）・5番に各2/3 → 見込みは 5番の 50 だけで決まる（分母は 2/3）
    assert pr.iloc[0]["fo_n:good"] == pytest.approx(4 / 3)
    assert pr.iloc[0]["fo_e3n:good"] == pytest.approx(2 / 3)
    s = summarize(pr, ["good"], n_boot=10)
    assert s.loc["good", "fo_e3"] == pytest.approx(50.0)
    assert s.loc["good", "mo_e3"] == pytest.approx((10 + 30 / 3) / (4 / 3))
    assert by_year(pr, ["good"]).loc[(2017, "good"), "fo_e3"] == pytest.approx(50.0)




def _constant_within_race(R: pd.DataFrame, seed: int = 1) -> np.ndarray:
    """レースごとに1つの値を全頭に入れる（年齢の 2016年のような「レース内で全頭同じ」要素）。

    0.1 刻みでない半端な値にして、レース内平均を引いたときに丸めの残り（1e−17 程度）が出るようにする。
    """
    rng = np.random.default_rng(seed)
    per_race = pd.Series(rng.uniform(-1.2, 0.1, R["race_id"].nunique()) / 7.0, index=R["race_id"].unique())
    return R["race_id"].map(per_race).to_numpy(float, copy=True)


def test_zscore_rounding_residuals_count_as_zero():
    """学習の行がどのレースでも全頭同じ値なら、中心化の丸めの残りで割って巨大な z を作ってはいけない。"""
    R = synth_eval(n_per_year=40)
    R = R[R.year <= 2017].reset_index(drop=True)
    v = _constant_within_race(R)
    late = (R.year == 2017).to_numpy()
    v[late] = R.loc[late, "noise"].to_numpy()          # 評価年だけレース内で差がある
    d = R.assign(f=v)
    train = (R.year == 2016).to_numpy()
    c = v - pd.Series(v).groupby(R["race_id"]).transform("mean").to_numpy()
    assert np.any(c[train] != 0.0), "作り物に丸めの残りが出ていない（テストの前提）"
    z = zscore(d, "f", train)
    assert np.all(z[train] == 0.0)
    assert np.all(z == 0.0)                            # 学習で差が無い → 標準偏差 0 → 全部 0（市場だけと同じ）


def test_zscore_is_clipped_when_test_spread_exceeds_train():
    """学習の年より評価の年の方が散らばりが大きい要素（縮小が年ごとに弱まる表の要素）でも |z| は 4 まで。"""
    d = pd.DataFrame({"race_id": ["a"] * 4 + ["b"] * 4,
                      "f": [0.001, -0.001, 0.002, -0.002, 0.5, -0.5, 0.3, -0.3]})
    train = np.array([True] * 4 + [False] * 4)
    z = zscore(d, "f", train)
    assert np.all(np.abs(z) <= 4.0)
    assert z[4] == 4.0 and z[5] == -4.0
    assert 0 < z[0] < 4.0                              # 学習の範囲の値は切られない


def test_delta_finite_and_bounded_for_late_starting_and_scale_shift_factors(monkeypatch):
    """実データで壊れた2つの形を仕込み、Δ が有限で桁外れにならないことを確かめる。

    - late_start（年齢の形）: 2016年はレース内で全頭同じ値、2017年からレース内で差がある。2017年の学習は
      2016年だけなので、要素は何も足さない（Δ = 0 ちょうど）。直す前は丸めの残りで割って Δ が −51 だった
    - scale_shift（父の芝ダの形）: 弱く効く要素で、散らばりが 2017年まで 0.001、2018年 0.0015、2019年から 0.01。
      モデルに入る z は ±4 に収まり、2019年の Δ も 1 未満（直す前は z が ±80 前後になり −2.2）

    作り物は1年80レースと小さく、雑音の要素でも推定のぶれで Δ が ±0.05 程度動くので、境は緩くしている。
    """
    from keiba.evaluate import factor_eval as fe

    R = synth_eval(n_per_year=80)
    v = _constant_within_race(R)
    later = (R.year >= 2017).to_numpy()
    v[later] = R.loc[later, "noise"].to_numpy() * 0.1
    scale = np.where(R.year <= 2017, 0.001, np.where(R.year == 2018, 0.0015, 0.01))
    R = R.assign(late_start=v, scale_shift=(0.1 * R["good"] + R["noise"]).to_numpy() * scale)
    seen_z = []
    orig = fe.race_nll

    def spy(d, feats, weights):
        if "_z" in feats:
            seen_z.append(float(np.abs(d["_z"]).max()))
        return orig(d, feats, weights)

    monkeypatch.setattr(fe, "race_nll", spy)
    pr = evaluate(R, ["late_start", "scale_shift"], log=lambda *_: None)["per_race"]
    assert seen_z and max(seen_z) <= fe.Z_CLIP
    for f in ("late_start", "scale_shift"):
        yearly = pr.groupby("year")[f"delta:{f}"].mean()
        assert np.all(np.isfinite(yearly)), f
        assert np.all(np.abs(yearly) < 1.0), (f, yearly.to_dict())
    assert pr.loc[pr.year == 2017, "delta:late_start"].eq(0.0).all()


def _late_start(R: pd.DataFrame) -> np.ndarray:
    """2016年はレース内で全頭同じ（測れない）、2017年からレース内で差がある要素。"""
    v = _constant_within_race(R)
    later = (R.year >= 2017).to_numpy()
    v[later] = R.loc[later, "good"].to_numpy()
    return v


def test_unmeasured_years_are_marked_and_not_counted():
    """学習の z が全部 0 の年は Δ = 0 ちょうどだが「改善しなかった年」ではなく「測れなかった年」（final-findings #2）。"""
    R = synth_eval(n_per_year=60)
    R = R.assign(late_start=_late_start(R))
    pr = evaluate(R, ["late_start", "allnan", "good"], log=lambda *_: None)["per_race"]
    t = by_year(pr, ["late_start", "allnan", "good"])
    assert t.loc[(2017, "late_start"), "measured"] == False      # noqa: E712  2016年だけで学習 → 測れない
    assert t.loc[(2018, "late_start"), "measured"] == True       # noqa: E712
    assert not t.xs("allnan", level="factor")["measured"].any()
    assert t.xs("good", level="factor")["measured"].all()
    s = summarize(pr, ["late_start", "allnan", "good"], n_boot=50)
    assert s.loc["late_start", "years_measured"] == 8
    assert s.loc["allnan", "years_measured"] == 0 and s.loc["allnan", "years_improved"] == 0
    assert s.loc["good", "years_measured"] == 9
    assert s.loc["late_start", "years_improved"] <= 8


def test_clipped_share_per_year_matches_zscore():
    """年 × 要素ごとに、評価の行のうち |z| が上限で切られた割合を出す（final-findings #3）。"""
    from keiba.evaluate import factor_eval as fe

    R = synth_eval(n_per_year=60)
    R = R.assign(wide=np.where(R.year >= 2019, 50.0, 1.0) * R["noise"])   # 2019年から散らばりが50倍
    pr = evaluate(R, ["wide", "good"], log=lambda *_: None)["per_race"]
    t = by_year(pr, ["wide", "good"])
    assert "clipped_share" in t.columns
    d = fe.add_market_features(R[fe.valid_race_mask(R)].copy()).sort_values(["race_id", "umaban"]).reset_index(drop=True)
    train = ((d.year >= 2016) & (d.year < 2019)).to_numpy()
    z = fe.zscore(d, "wide", train)
    test = (d.year == 2019).to_numpy()
    assert t.loc[(2019, "wide"), "clipped_share"] == pytest.approx(np.mean(np.abs(z[test]) >= fe.Z_CLIP))
    assert t.loc[(2019, "wide"), "clipped_share"] > 0.5
    assert t.loc[(2018, "wide"), "clipped_share"] < 0.01
    assert t.xs("good", level="factor")["clipped_share"].max() < 0.01


def test_summary_has_baseline_nll_pct_and_bonferroni_interval():
    """市場だけのモデルの1レースあたり −対数尤度と、Δ のその割合、15比較の Bonferroni の区間（final-findings #5）。"""
    R = synth_eval(n_per_year=60)
    pr = evaluate(R, ["good", "noise"], log=lambda *_: None)["per_race"]
    s = summarize(pr, ["good", "noise"], n_boot=200, n_compare=15)
    assert s.loc["good", "base_nll"] == pytest.approx(pr["nll_mkt"].mean())
    assert s.loc["good", "delta_pct"] == pytest.approx(s.loc["good", "delta"] / pr["nll_mkt"].mean())
    for f in ("good", "noise"):
        assert s.loc[f, "lo_bonf"] <= s.loc[f, "lo"] <= s.loc[f, "hi"] <= s.loc[f, "hi_bonf"]
    # 判定は 95% 区間のまま（Bonferroni は参考の列）
    assert s.loc["good", "verdict"] == verdict(s.loc["good", "lo"], int(s.loc["good", "years_improved"]))


def test_excluded_counts_by_reason():
    R = synth_eval(n_per_year=30)
    first = R.loc[R.year == 2017, "race_id"].unique()[:3]
    R.loc[(R.race_id == first[0]) & (R.jyuni == 2), "jyuni"] = 1.0
    R.loc[(R.race_id == first[1]) & (R.umaban == "01"), "odds"] = np.nan
    R = R[~((R.race_id == first[2]) & (R.jyuni > 3))]
    R.loc[R.race_id == first[2], "field"] = 3
    ex = evaluate(R, ["good"], years=[2017], log=lambda *_: None)["excluded"]
    assert ex["excluded"] == 3
    assert ex["by_reason"] == {"dead_heat": 1, "bad_odds": 1, "small_field": 1}
