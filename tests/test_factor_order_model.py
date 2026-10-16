"""段ごとの条件付きロジットのテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.order_model import (add_market_features, fit_stages, place_probabilities, race_nll,
                                       valid_race_mask)


def synth(n_races=400, field=8, w_true=(1.5, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        x = rng.normal(0, 1, field)
        g = rng.gumbel(size=field)
        order = np.argsort(-(w_true[0] * x + g))
        jyuni = np.empty(field)
        jyuni[order] = np.arange(1, field + 1)
        for i in range(field):
            rows.append(dict(race_id=f"R{r:04d}", umaban=f"{i + 1:02d}", x=x[i], jyuni=jyuni[i],
                             odds=float(np.round(2 + 10 * rng.random(), 1)), field=field))
    return pd.DataFrame(rows)


def test_valid_race_mask_excludes_dead_heat_and_missing_odds():
    d = synth(n_races=3)
    d.loc[(d.race_id == "R0001") & (d.jyuni == 2), "jyuni"] = 1.0       # 1着同着
    d.loc[(d.race_id == "R0002") & (d.umaban == "01"), "odds"] = np.nan
    m = valid_race_mask(d)
    assert set(d[m].race_id) == {"R0000"}


def test_exclusion_reasons_per_race():
    """除く理由をレースごとに分けて数える（重なりあり）。valid_race_mask と同じ判定。"""
    from keiba.factors.order_model import exclusion_reasons

    d = synth(n_races=5)
    d.loc[(d.race_id == "R0001") & (d.jyuni == 3), "jyuni"] = 2.0       # 2着同着 → 3着がいない
    d.loc[(d.race_id == "R0002") & (d.umaban == "01"), "odds"] = 0.0
    d.loc[d.race_id == "R0003", "field"] = 3
    d.loc[(d.race_id == "R0004") & (d.jyuni == 2), "jyuni"] = 1.0       # 同着かつオッズ欠け
    d.loc[(d.race_id == "R0004") & (d.umaban == "02"), "odds"] = np.nan
    t = exclusion_reasons(d)
    assert t.loc["R0000"].tolist() == [False, False, False]
    assert t.loc["R0001"].tolist() == [True, False, False]
    assert t.loc["R0002"].tolist() == [False, True, False]
    assert t.loc["R0003"].tolist() == [False, False, True]
    assert t.loc["R0004"].tolist() == [True, True, False]
    assert list(t.columns) == ["dead_heat", "bad_odds", "small_field"]
    ok = set(d[valid_race_mask(d)].race_id)
    assert ok == set(t.index[~t.any(axis=1)])


def test_fit_recovers_stage1_weight():
    d = synth()
    w = fit_stages(d, ["x"])
    assert w[0][0] == pytest.approx(1.5, abs=0.25)


def test_race_nll_and_probabilities():
    d = synth(n_races=50)
    w = fit_stages(d, ["x"])
    nll = race_nll(d, ["x"], w)
    assert len(nll) == 50 and (nll > 0).all()
    win, top3 = place_probabilities(d, ["x"], w)
    s = pd.DataFrame({"r": d.race_id, "w": win, "t": top3}).groupby("r").sum()
    assert np.allclose(s.w, 1.0) and np.allclose(s.t, 3.0)


def _brute_force(x: np.ndarray, weights: list[np.ndarray]) -> dict[tuple[int, int, int], float]:
    """1レースの (1着, 2着, 3着) の確率を、段ごとの softmax を順に掛けて素直に数え上げる。"""
    eta = [x @ w for w in weights]
    n = len(x)
    out = {}
    for i in range(n):
        p1 = np.exp(eta[0][i]) / np.exp(eta[0]).sum()
        for j in range(n):
            if j == i:
                continue
            rest2 = [k for k in range(n) if k != i]
            p2 = np.exp(eta[1][j]) / np.exp(eta[1][rest2]).sum()
            for k in range(n):
                if k in (i, j):
                    continue
                rest3 = [m for m in range(n) if m not in (i, j)]
                out[(i, j, k)] = p1 * p2 * np.exp(eta[2][k]) / np.exp(eta[2][rest3]).sum()
    return out


def test_probabilities_and_nll_match_brute_force_with_stage_specific_weights():
    """段ごとに違う重みで、place_probabilities と race_nll を数え上げと突き合わせる（段2・段3の条件付き確率の確認）。"""
    rng = np.random.default_rng(5)
    n = 6
    d = pd.DataFrame({"race_id": "R0", "umaban": [f"{i + 1:02d}" for i in range(n)],
                      "x": rng.normal(0, 1, n), "u": rng.normal(0, 1, n),
                      "jyuni": [4.0, 2.0, 6.0, 1.0, 3.0, 5.0]})
    feats = ["x", "u"]
    weights = [np.array([1.2, -0.3]), np.array([0.4, 0.9]), np.array([-0.7, 0.5])]
    probs = _brute_force(d[feats].to_numpy(), weights)
    assert sum(probs.values()) == pytest.approx(1.0)
    win, top3 = place_probabilities(d, feats, weights)
    for h in range(n):
        assert win[h] == pytest.approx(sum(p for (i, _, _), p in probs.items() if i == h))
        assert top3[h] == pytest.approx(sum(p for t, p in probs.items() if h in t))
    actual = (3, 1, 4)                                    # 1着 = 4番、2着 = 2番、3着 = 5番
    assert race_nll(d, feats, weights)["R0"] == pytest.approx(-np.log(probs[actual]))


def test_fit_recovers_later_stage_weights():
    """段2・段3 も同じ地力の Plackett–Luce で決めた着順なので、真の重みを取り戻す。"""
    d = synth(n_races=800, seed=3)
    w = fit_stages(d, ["x"])
    assert w[1][0] == pytest.approx(1.5, abs=0.25) and w[2][0] == pytest.approx(1.5, abs=0.25)


def test_order_dist_stays_a_probability_when_stage3_denominator_cancels():
    """3段目の分母 S − e_i − e_j が桁落ちで 0 以下になっても、負や巨大な確率を出さない。"""
    from keiba.factors.order_model import _order_dist

    flat = np.zeros(4)
    t = _order_dist([flat, flat, np.array([0.0, -40.0, -40.0, -40.0])])   # e = [1, 4e−18, …] → 1 − 1 − 4e−18 < 0
    assert np.isfinite(t).all() and (t >= 0).all() and t.sum() <= 1 + 1e-12


def test_fit_stops_when_a_stage_does_not_converge(monkeypatch):
    from keiba.factors import order_model as om

    real = om.minimize

    def stalled(*a, **k):                                 # 反復の上限に当たった結果を装う
        r = real(*a, **k)
        r.success, r.message = False, "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        return r

    monkeypatch.setattr(om, "minimize", stalled)
    with pytest.raises(RuntimeError, match="段1"):
        fit_stages(synth(n_races=50), ["x"])


def test_market_features():
    d = pd.DataFrame({"race_id": ["a", "a"], "odds": [2.0, 4.0]})
    out = add_market_features(d)
    q = np.array([2 / 3, 1 / 3])                          # 1/オッズ = 0.5, 0.25 をレース内で合計 1 に
    assert np.allclose(out.lq, np.log(q)) and np.allclose(out.lq2, np.log(q) ** 2)
