"""年齢曲線（仕様 §6）のテスト。同じ馬の変化で測れば、弱い馬の引退（生き残りの偏り）に強い。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from keiba.evaluate.age_curve import (ANCHOR, CURVE_COLUMNS, age_curve, curve_levels, half_of,
                                      peak_summary, period_label, period_values)

# 真の年齢曲線（3歳後半 = 0）
L = {(2, 1): -2.0, (3, 0): -1.0, (3, 1): 0.0, (4, 0): 0.3, (4, 1): 0.4, (5, 0): 0.3, (5, 1): 0.1}
YEARS = [2022, 2023, 2024, 2025]


def synth_fig(*, n_horses=200, retire_weak_after=None, seed=0) -> pd.DataFrame:
    """馬ごとの地力 base + 年齢の水準 L。各期間に2走（同じ値）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_horses):
        base = float(rng.normal(0, 1))
        cohort = 2020 + i % 2
        for (age, half), lvl in L.items():
            if retire_weak_after and base < 0 and 2 * age + half > 2 * retire_weak_after[0] + retire_weak_after[1]:
                break
            year = cohort + age
            if year not in YEARS:
                continue
            for month in ((3, 5) if half == 0 else (9, 11)):
                rows.append(dict(ketto_num=f"H{i}", year=year, date=pd.Timestamp(year, month, 1),
                                 barei=age, track_kind="turf", figure=base + lvl))
    return pd.DataFrame(rows)


def test_labels_and_halves():
    assert period_label(3, 1) == "3歳後半" and period_label(4, 0) == "4歳前半"
    assert half_of(date(2026, 6, 30)) == 0 and half_of(date(2026, 7, 1)) == 1
    assert ANCHOR == (3, 1)


def test_curve_recovers_true_levels():
    curve = age_curve(synth_fig(), years=YEARS, before=date(2026, 1, 1), n_boot=20, seed=1)
    assert list(curve.columns) == CURVE_COLUMNS
    got = curve_levels(curve)
    for period, lvl in L.items():
        assert got[period] == pytest.approx(lvl, abs=1e-9)


def test_paired_changes_resist_survivorship_bias():
    """弱い馬が3歳後半で引退しても、同じ馬の変化なら真の曲線に近い。全体平均は高齢を強く見せる。"""
    fig = synth_fig(retire_weak_after=(3, 1))
    got = curve_levels(age_curve(fig, years=YEARS, before=date(2026, 1, 1), n_boot=20, seed=1))
    assert got[(4, 0)] == pytest.approx(L[(4, 0)], abs=1e-9)
    naive = fig.assign(half=(fig.date.dt.month > 6).astype(int)).groupby(["barei", "half"]).figure.mean()
    naive_level = naive[(4, 0)] - naive[(3, 1)]
    assert abs(naive_level - L[(4, 0)]) > 0.1


def test_single_run_periods_are_not_used():
    fig = pd.DataFrame([
        dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 3, 1), barei=3, track_kind="turf", figure=1.0),
        dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 9, 1), barei=3, track_kind="turf", figure=2.0),
        dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 10, 1), barei=3, track_kind="turf", figure=3.0),
    ])
    pv = period_values(fig, years=YEARS, before=date(2026, 1, 1))
    assert pv[["year", "half", "value"]].values.tolist() == [[2024, 1, 2.5]]   # 前半は1走なので使わない


def test_top_two_runs_are_averaged():
    rows = [dict(ketto_num="A", year=2024, date=pd.Timestamp(2024, 8, d), barei=3, track_kind="turf", figure=f)
            for d, f in ((1, 1.0), (2, 5.0), (3, 3.0))]
    pv = period_values(pd.DataFrame(rows), years=YEARS, before=date(2026, 1, 1))
    assert pv.value.tolist() == [4.0]


def test_incomplete_periods_are_not_used():
    pv = period_values(synth_fig(), years=YEARS, before=date(2025, 10, 4))
    assert not ((pv.year == 2025) & (pv.half == 1)).any()
    assert ((pv.year == 2025) & (pv.half == 0)).any()


def test_surface_filter():
    fig = synth_fig()
    fig.loc[fig.index % 2 == 0, "track_kind"] = "dirt"    # 各期間の1走目をダートに
    assert period_values(fig, years=YEARS, before=date(2026, 1, 1), surface="turf").empty   # 芝は1走ずつ


def test_peak_and_one_year_drop():
    curve = age_curve(synth_fig(), years=YEARS, before=date(2026, 1, 1), n_boot=10, seed=1)
    summary = peak_summary(curve)
    assert summary["peak"] == "4歳後半"
    assert summary["drop_1y"] == pytest.approx(0.4 - 0.1, abs=1e-9)


def test_bootstrap_is_reproducible():
    fig = synth_fig()
    fig["figure"] += np.random.default_rng(3).normal(0, 0.3, len(fig))
    a = age_curve(fig, years=YEARS, before=date(2026, 1, 1), n_boot=30, seed=7)
    b = age_curve(fig, years=YEARS, before=date(2026, 1, 1), n_boot=30, seed=7)
    pd.testing.assert_frame_equal(a, b)
    row = a.set_index("label").loc["4歳前半"]
    assert row.level_lo <= row.level <= row.level_hi


def test_age_curve_empty_when_no_changes():
    one = synth_fig(n_horses=1).iloc[:2]       # 1期間しかない
    curve = age_curve(one, years=YEARS, before=date(2026, 1, 1), n_boot=5, seed=1)
    assert curve.empty and list(curve.columns) == CURVE_COLUMNS
    assert peak_summary(curve)["peak"] is None
