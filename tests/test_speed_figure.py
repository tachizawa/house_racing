"""タイム指数（仕様 §3）のテスト。

最優先は2つ。as_of より後のレースが混ざらないこと（Task 1）と、2歳戦の遅さが馬場差に
吸収されないこと（Task 2、§3.4）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from keiba.features.speed_figure import (RUN_COLUMNS, apply_figures, fit_model, prepare_runs,
                                         repeatability)
from tests.fixtures.figure_runs import C_TRUE, V_TRUE, older_runs, run_row, young_runs

JST = timezone(timedelta(hours=9))


def jst(y, m, d, hh=0, mm=0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=JST)


def race_row(race_id, *, hasso_at="2025-01-05T15:00:00+09:00", jyo_cd="05", track_cd="24",
             track_kind="turf", kyori_m="1600", syubetu_cd="13", class_level="2",
             data_kubun="7", make_date="20250106") -> dict:
    return dict(race_id=race_id, hasso_at=hasso_at, jyo_cd=jyo_cd, track_cd=track_cd,
                track_kind=track_kind, kyori_m=kyori_m, syubetu_cd=syubetu_cd,
                class_level=class_level, data_kubun=data_kubun, make_date=make_date)


def horse_row(race_id, umaban="01", *, ketto_num="H1", barei="04", time="96.0", jyuni="1",
              ijyo="0", data_kubun="7", make_date="20250106") -> dict:
    return dict(race_id=race_id, umaban=umaban, ketto_num=ketto_num, barei=barei,
                result_time_sec=time, result_kakutei_jyuni=jyuni, result_ijyo_cd=ijyo,
                data_kubun=data_kubun, make_date=make_date)


def frames(races, horses):
    return pd.DataFrame(races, dtype=str), pd.DataFrame(horses, dtype=str)


def test_prepare_runs_columns_and_cohort():
    races, horses = frames([race_row("2025010505010101")], [horse_row("2025010505010101")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert list(runs.columns) == RUN_COLUMNS
    row = runs.iloc[0]
    assert row.day == "20250105" and row.year == 2025 and row.barei == 4
    assert row.cohort == 2021                     # 世代 = 年 − 馬齢
    assert row.kyori_m == 1600 and row.time_sec == 96.0 and row["class_level"] == 2.0


def test_prepare_runs_excludes_races_at_or_after_as_of():
    races, horses = frames(
        [race_row("2025010505010101", hasso_at="2025-01-05T15:00:00+09:00"),
         race_row("2025010505010102", hasso_at="2025-01-05T15:30:00+09:00")],
        [horse_row("2025010505010101"), horse_row("2025010505010102")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 1, 5, 15, 30))
    assert runs.race_id.tolist() == ["2025010505010101"]   # 同時刻は含めない


# race_id の先頭8桁は開催日として読むので、作り物でも本物と同じ形にする
OK, JUMP, NAR, ENTRY = "2025010505010101", "2025010505010102", "2025010530010103", "2025010505010104"


def test_prepare_runs_exclusions():
    races, horses = frames(
        [race_row(OK), race_row(JUMP, track_kind="jump"), race_row(NAR, jyo_cd="30"),
         race_row(ENTRY, data_kubun="2")],
        [horse_row(OK, "01"), horse_row(OK, "02", ijyo="1"), horse_row(OK, "03", ijyo="4"),
         horse_row(OK, "04", time=""), horse_row(OK, "05", time="0"),
         horse_row(JUMP), horse_row(NAR), horse_row(ENTRY, data_kubun="2")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert runs.race_id.tolist() == [OK] and runs.umaban.tolist() == ["01"]


def test_prepare_runs_prefers_result_over_entry_rows():
    races, horses = frames(
        [race_row(OK, data_kubun="2", make_date="20250103"), race_row(OK)],
        [horse_row(OK, time="", data_kubun="2", make_date="20250103"), horse_row(OK)])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert len(runs) == 1 and runs.iloc[0].time_sec == 96.0


def test_prepare_runs_uses_final_results_only():
    """確定成績（7）がまだ無く速報（3〜6）しかないレース・走は使わない（仕様 §3.1）。"""
    races, horses = frames(
        [race_row(OK), race_row(OK, data_kubun="5", make_date="20250107"), race_row(JUMP, data_kubun="5")],
        [horse_row(OK, "01"), horse_row(OK, "02", data_kubun="6"), horse_row(JUMP, data_kubun="5")])
    runs = prepare_runs(races, horses, as_of=jst(2025, 2, 1))
    assert runs.race_id.tolist() == [OK] and runs.umaban.tolist() == ["01"]


def test_prepare_runs_requires_timezone():
    races, horses = frames([race_row(OK)], [horse_row(OK)])
    with pytest.raises(ValueError):
        prepare_runs(races, horses, as_of=datetime(2025, 2, 1))


def test_fit_recovers_track_variants():
    model = fit_model(older_runs())
    est = model.variants.xs(("05", "turf"), level=("jyo_cd", "track_kind")).to_numpy()
    truth = np.array(V_TRUE)
    # 基準と馬場差は全体の定数を取り合うので、平均を引いた形で比べる
    assert np.allclose(est - est.mean(), truth - truth.mean(), atol=1e-6)
    assert model.n_races == 24


def test_fit_recovers_class_effects_and_standards():
    model = fit_model(older_runs())
    assert model.class_effects[(2.0, "turf")] == pytest.approx(0.0, abs=1e-9)
    assert model.class_effects[(3.0, "turf")] == pytest.approx(C_TRUE[3], abs=1e-6)
    s = model.standards
    assert s[("05", "24", 2000)] - s[("05", "24", 1600)] == pytest.approx(1.0, abs=1e-6)


def test_slow_two_year_olds_are_not_absorbed_into_variant():
    """§3.4 の核心。2歳戦が一律に遅くても馬場差は動かず、2歳の指数が下がる。"""
    base = fit_model(older_runs())
    runs = pd.concat([older_runs(), young_runs(slow=0.5)], ignore_index=True)
    model = fit_model(runs)
    pd.testing.assert_series_equal(model.variants, base.variants)
    fig = apply_figures(runs, model)
    young = fig[fig.syubetu_cd == "11"]
    assert np.allclose(young.figure, -0.5 * 1.6, atol=1e-6)
    older_cls3 = fig[(fig.syubetu_cd == "13") & (fig.class_level == 3.0)]
    assert np.allclose(older_cls3.figure, 0.5 * 1.6, atol=1e-6)   # 上のクラスほど速い


def test_non_winners_do_not_enter_the_fit():
    base = fit_model(older_runs())
    # 1着のいない別のレースにする（同じ race_id だと drop_duplicates で消え、着順の条件を確かめられない）
    extra = pd.DataFrame([run_row(race_id="20250101-1600-X", umaban="02", rank=2.0, time_sec=200.0)])
    model = fit_model(pd.concat([older_runs(), extra], ignore_index=True))
    pd.testing.assert_series_equal(model.variants, base.variants)
    assert model.n_races == base.n_races


def test_dead_heat_winner_counted_once():
    heat = pd.DataFrame([run_row(race_id="20250101-1600-2", umaban="09", ketto_num="H9",
                                 time_sec=older_runs().iloc[0].time_sec)])
    model = fit_model(pd.concat([older_runs(), heat], ignore_index=True))
    assert model.n_races == 24


def test_run_on_key_without_older_races_has_no_figure():
    odd = pd.DataFrame([run_row(race_id="X", kyori_m=1200, syubetu_cd="11", time_sec=72.0)])
    runs = pd.concat([older_runs(), odd], ignore_index=True)
    fig = apply_figures(runs, fit_model(runs))
    row = fig[fig.race_id == "X"].iloc[0]
    assert not row.has_standard and np.isnan(row.figure)


def test_run_on_day_without_older_races_uses_zero_variant():
    z = pd.DataFrame([run_row(race_id="Z", day="20250301", date=pd.Timestamp("2025-03-01"),
                              syubetu_cd="11", time_sec=60.0 * 1.6)])
    runs = pd.concat([older_runs(), z], ignore_index=True)
    model = fit_model(runs)
    row = apply_figures(runs, model).set_index("race_id").loc["Z"]
    assert not row.has_variant
    assert row.figure == pytest.approx((model.standards[("05", "24", 1600)] - 60.0) * 1.6)


def test_fit_without_older_races_raises():
    with pytest.raises(ValueError):
        fit_model(young_runs(slow=0.0))


def test_repeatability_pairs_consecutive_runs_within_gap():
    rows = []
    for h, base in enumerate([-1.0, 0.0, 1.0, 2.0]):
        for k, d in enumerate(["2025-01-01", "2025-02-01", "2025-12-01"]):
            rows.append(run_row(race_id=f"{h}{k}", ketto_num=f"H{h}", date=pd.Timestamp(d)))
    fig = pd.DataFrame(rows)
    fig["figure"] = [b + 0.01 * k for b in [-1.0, 0.0, 1.0, 2.0] for k in range(3)]
    r, n = repeatability(fig, max_gap_days=120)
    assert n == 4                       # 2月→12月は 120 日を超えるので数えない
    assert r == pytest.approx(1.0)


def test_fit_raises_without_reference_class_for_a_surface():
    dirt = pd.DataFrame([run_row(race_id="D", track_kind="dirt", track_cd="23", class_level=3.0)])
    with pytest.raises(ValueError, match="dirt"):
        fit_model(pd.concat([older_runs(), dirt], ignore_index=True))
