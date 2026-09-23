from __future__ import annotations

import math

import pytest

from keiba.evaluate import top3_metrics as tm
from keiba.models import EntryRow, ResultRow


def rr(umaban, jyuni, ijyo_cd="0"):
    entry = EntryRow(umaban=umaban, wakuban="1", ketto_num=umaban, bamei=umaban,
                     sex_cd="1", barei="04", futan_kg=57.0, kisyu_code="1",
                     tozai_cd="1", chokyosi_code="1")
    return ResultRow(entry=entry, ijyo_cd=ijyo_cd, kakutei_jyuni=jyuni, dochaku_kubun="0",
                     dochaku_tosu=None, time_sec=None, margin_sec=None, jyuni_1c=None,
                     jyuni_2c=None, jyuni_3c=None, jyuni_4c=None, last_3f_sec=None,
                     ba_taijyu_kg=None, zogen_sa_kg=None, odds=None, ninki=None)


def test_outcome_basic():
    o = tm.outcome_from_results("R", [rr("01", 1), rr("02", 2), rr("03", 3), rr("04", 4)])
    assert o.top3 == ("01", "02", "03")
    assert o.starters == ("01", "02", "03", "04")


def test_outcome_dead_heat_for_third_has_four():
    o = tm.outcome_from_results("R", [rr("01", 1), rr("02", 2), rr("03", 3), rr("04", 3), rr("05", 5)])
    assert o.top3 == ("01", "02", "03", "04")


def test_outcome_excludes_disqualified_and_scratched():
    o = tm.outcome_from_results("R", [rr("01", 1), rr("02", 2, ijyo_cd="5"), rr("03", 3),
                                      rr("04", None, ijyo_cd="1"), rr("05", 4)])
    assert o.top3 == ("01", "03")
    assert "02" in o.starters
    assert o.excluded == ("04",)


def test_outcome_none_without_placed_horses():
    assert tm.outcome_from_results("R", [rr("01", None, ijyo_cd="4"), rr("02", None, ijyo_cd="4")]) is None
    assert tm.outcome_from_results("R", []) is None


def _o(top3=("01", "02", "03"), starters=("01", "02", "03", "04", "05"), excluded=()):
    return tm.Top3Outcome("20260921R", top3, starters, excluded)


def test_score_hits_and_all_selected():
    s = tm.score_race(["01", "02", "04"], None, _o())
    assert s["hits"] == 2
    assert s["all_top3_selected"] is False
    assert s["brier"] is None


def test_score_scratched_selection_counts_as_miss():
    s = tm.score_race(["01", "02", "09"], None, _o(excluded=("09",)))
    assert s["hits"] == 2
    assert s["selected_scratched"] == 1


def test_score_top3_exceeds_n_cannot_be_all_selected():
    s = tm.score_race(["01", "02", "03"], None, _o(top3=("01", "02", "03", "04")))
    assert s["all_top3_selected"] is False
    assert s["top3_exceeds_n"] is True


def test_score_brier_over_starters_only():
    probs = {"01": 1.0, "02": 1.0, "03": 1.0, "04": 0.0, "05": 0.0, "09": 0.7}
    s = tm.score_race(["01", "02", "03"], probs, _o(excluded=("09",)))
    assert s["brier"] == 0.0
    assert s["prob_sum"] == 3.0
    assert s["prob_sum_minus_actual"] == 0.0


def test_score_requires_probability_for_every_starter():
    with pytest.raises(ValueError):
        tm.score_race(["01"], {"01": 0.5}, _o())


def test_aggregate_distribution_and_rates():
    scored = [tm.score_race(["01", "02", "03"], None, _o()),
              tm.score_race(["04", "05", "01"], None, _o())]
    agg = tm.aggregate(scored, n=3)
    assert agg["n_races"] == 2
    assert agg["mean_hits"] == 2.0
    assert agg["hit_distribution"] == {"0": 0, "1": 1, "2": 0, "3": 1}
    assert agg["all_top3_selected_rate"] == 0.5
    assert math.isnan(agg["brier"])


def test_aggregate_empty_is_nan_not_zero():
    agg = tm.aggregate([], n=3)
    assert agg["n_races"] == 0
    assert math.isnan(agg["mean_hits"])


def test_race_date_cluster():
    assert tm.race_date_cluster("2026092106040511") == "20260921"


def test_paired_bootstrap_zero_difference():
    a = {f"2026092{d}0604051{r}": 2.0 for d in range(1, 4) for r in range(1, 4)}
    result = tm.paired_bootstrap(a, dict(a), cluster_of=tm.race_date_cluster, n_resamples=200)
    assert result["mean_diff"] == 0.0
    assert result["lo"] == 0.0 and result["hi"] == 0.0
    assert result["n_clusters"] == 3


def test_paired_bootstrap_uses_only_common_races():
    a = {"2026092106040511": 3.0, "2026092206040511": 1.0}
    b = {"2026092106040511": 1.0}
    result = tm.paired_bootstrap(a, b, cluster_of=tm.race_date_cluster, n_resamples=50)
    assert result["n_races"] == 1
    assert result["mean_diff"] == 2.0
