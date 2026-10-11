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


def test_market_features():
    d = pd.DataFrame({"race_id": ["a", "a"], "odds": [2.0, 2.0]})
    out = add_market_features(d)
    assert np.allclose(out.lq, np.log(0.5)) and np.allclose(out.lq2, np.log(0.5) ** 2)
