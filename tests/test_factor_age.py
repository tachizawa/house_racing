"""年齢・晩成度×年齢・世代レベル（仕様 §3.2 #6〜8）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba.factors.age_factors import age_scores, cohort_level_scores, late_age_scores, sire_lateness
from keiba.factors.figure_factors import yearly_figures
from tests.fixtures.factor_runs import assert_no_lookahead, make_factor_runs, to_runs_all


@pytest.fixture(scope="module")
def data():
    R = make_factor_runs(years=(2015, 2016, 2017, 2018), days_per_year=10)
    return R, yearly_figures(to_runs_all(R), 2018)


@pytest.mark.parametrize("fn", [age_scores, late_age_scores, cohort_level_scores], ids=lambda f: f.__name__)
# 作り物のデータでは L の材料が無く late_age_scores は全て NaN になる。L の経路そのものは
# test_sire_lateness_signs_and_cohort_limit が手組みのデータで調べる。
def test_only_target_year_and_no_lookahead(fn, data):
    R, fig = data
    out = fn(R, fig, 2018)
    assert out.shape == (len(R),)
    assert np.isnan(out[(R.year != 2018).to_numpy()]).all()
    day = R[R.year == 2018].date.unique()[4]
    assert_no_lookahead(lambda d: fn(d, yearly_figures(to_runs_all(d), 2018), 2018), R, day)


def test_age_scores_follow_curve_levels(data):
    R, fig = data
    out = age_scores(R, fig, 2018)
    d = R.assign(sc=out)[R.year == 2018]
    # 同じ年齢・同じ半期なら同じ値
    assert d.groupby(["barei", d.date.dt.month > 6]).sc.nunique(dropna=False).max() == 1


def _hand_built(with_future_cohort: bool):
    """評価年 2022（使える世代は 2016 年生まれまで）の手組みデータ。

    LATE は伸びて出世が遅い父、EARLY は伸びず出世が早い父（A・B 両方の材料あり）。
    A_ONLY は A の最小頭数を満たすが OP の上位入着が無い父、B_ONLY は B の最小頭数を満たすが
    A の材料（図の行）が無い父。FUTURE は 2017 年生まれ（2022 年には使えない）で A・B とも極端。
    """
    fig_rows, run_rows = [], []

    def add_sire(sire, cohorts, per_cohort, first_ages, delta):
        """first_ages が None なら OP の行を作らない。delta が None なら図の行を作らない。"""
        k = 0
        for c in cohorts:
            for i in range(per_cohort):
                h = f"{sire}-{c}-{i}"
                early = 50.0 + 3 * (i % 5) + (k % 3)   # 同じ世代の中で 3歳前半の水準にばらつきを付ける
                if delta is not None:
                    # 3歳前半 / 4歳後半 / 5歳前半 の各期間に2走（上位2走の平均 = f − 0.5）
                    for yr, month, age, f in [(c + 3, 3, 3, early),
                                              (c + 4, 10, 4, early + delta + (k % 2) * 0.5),
                                              (c + 5, 3, 5, early + delta + (k % 3) * 0.4)]:
                        for j, m in enumerate((month, month + 1)):
                            fig_rows.append((h, yr, pd.Timestamp(yr, m, 10), age, "turf", f - j))
                if first_ages is not None:
                    fa = first_ages[k % len(first_ages)]
                    run_rows.append((h, sire, pd.Timestamp(c + fa, 5, 10), c + fa, fa, 5, 1))
                run_rows.append((h, sire, pd.Timestamp(2022, 5, 10), 2022, 2022 - c, 1, 0))
                k += 1

    add_sire("LATE", (2015, 2016), 5, [4, 4, 4, 4, 4, 4, 4, 4, 3, 3], 10.0)
    add_sire("EARLY", (2015, 2016), 5, [3, 3, 3, 3, 3, 3, 3, 3, 3, 4], -5.0)
    add_sire("A_ONLY", (2015, 2016), 5, None, 10.0)
    add_sire("B_ONLY", (2015, 2016), 3, [4, 4, 4, 4, 4, 3], None)
    if with_future_cohort:
        add_sire("FUTURE", (2017,), 10, [2], 80.0)
    fig = pd.DataFrame(fig_rows, columns=["ketto_num", "year", "date", "barei", "track_kind", "figure"])
    R = pd.DataFrame(run_rows, columns=["ketto_num", "sire_num", "date", "year", "barei", "class_level", "top3"])
    return R, fig


def test_sire_lateness_signs_and_cohort_limit():
    R, fig = _hand_built(with_future_cohort=False)
    L = sire_lateness(R, fig, 2022)
    # A_ONLY は A だけ、B_ONLY は B だけから L が出る。片方の部分を落とすとその父が消える
    assert set(L.index) == {"LATE", "EARLY", "A_ONLY", "B_ONLY"}
    assert L["LATE"] > 0 > L["EARLY"]
    assert L["A_ONLY"] > 0      # A の標準化値は +（伸びは LATE と同じ +10）
    assert L["B_ONLY"] > 0      # B の標準化値は +（初の OP 上位入着が多く 4歳以上）
    # 値そのもの（A・B 両方に依存する。部分を落とす・限界を外すと変わる）
    assert L.to_dict() == pytest.approx({"LATE": 0.570553775, "EARLY": -1.154647612,
                                         "A_ONLY": 0.577350269, "B_ONLY": 0.590837405}, abs=1e-6)
    # 2017年生まれ（2022年には 5歳前半を終えていない）の極端な馬を足しても L は変わらない
    R2, fig2 = _hand_built(with_future_cohort=True)
    L2 = sire_lateness(R2, fig2, 2022)
    assert "FUTURE" not in L2.index
    pd.testing.assert_series_equal(L2.sort_index(), L.sort_index())
    # late_age_scores: 2022年の行（4歳以上）に L × 年齢係数が入る
    out = late_age_scores(R, fig, 2022)
    rows = (R.year == 2022).to_numpy() & (R.barei >= 4).to_numpy()
    sc = pd.Series(out[rows], index=R.sire_num[rows])
    assert sc.notna().all()
    assert (sc[sc.index == "LATE"] > 0).all() and (sc[sc.index == "EARLY"] < 0).all()
    assert np.isnan(out[(R.year != 2022).to_numpy()]).all()


def test_sire_lateness_empty_without_material(data):
    R, fig = data
    # 作り物のデータは各父の頭数が少なく OP の行も無いので、L の材料が無く空になる（late_age_scores は全て NaN）
    assert sire_lateness(R, fig, 2018).empty


def test_cohort_level_january_is_nan(data):
    R, fig = data
    out = cohort_level_scores(R, fig, 2018)
    jan = ((R.year == 2018) & (R.date.dt.month == 1)).to_numpy()
    assert np.isnan(out[jan]).all()       # 1/1 時点の今年の世代はまだ走っていない
