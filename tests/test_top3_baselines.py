from __future__ import annotations

from keiba.predict import top3_baselines as tb


def d(ranks: dict):
    return {"horses": {u: {"ability": {"field_rank": r}} for u, r in ranks.items()}}


def test_ranking_by_field_rank():
    r = tb.ability_ranking(d({"01": 3.0, "02": 1.0, "03": 2.0}))
    assert r.ranking == ("02", "03", "01")
    assert r.all_tied is False
    assert r.n_unranked == 0


def test_horses_without_rank_go_last_in_umaban_order():
    r = tb.ability_ranking(d({"03": None, "01": 2.0, "02": None}))
    assert r.ranking == ("01", "02", "03")
    assert r.n_unranked == 2


def test_all_missing_is_reported_as_tied():
    r = tb.ability_ranking(d({"01": None, "02": None}))
    assert r.all_tied is True


def test_uniform_probabilities_sum_to_three():
    p = tb.uniform_probabilities(d({f"{i:02d}": None for i in range(1, 11)}))
    assert abs(sum(p.values()) - 3.0) < 1e-9


def test_uniform_probabilities_small_field():
    p = tb.uniform_probabilities(d({"01": None, "02": None}))
    assert p == {"01": 1.0, "02": 1.0}
