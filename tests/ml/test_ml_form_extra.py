"""総合モデルの追加特徴量（features/form_extra.py）が as-of であること・切り詰め検査が漏れを見つけること。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from keiba_ml import registry
from keiba_ml.config import CombinedConfig, MLConfig, PedigreeConfig, validate
from keiba_ml.dataset import integrity
from keiba_ml.features import form, form_extra
from keiba_ml.features.pedigree import prepare
from tests.ml.test_ml_combined import _frame, _pedigree_frame


def _base(df: pd.DataFrame) -> pd.DataFrame:
    speed = form.speed_figures(df, standard_years=1)
    return pd.concat([form.own_history_features(df), speed,
                      form.people_features(prepare(df), shrinkage=10.0),
                      form.opponent_strength(df, speed["s_fig_best"])],
                     axis=1)[list(form_extra.BASE_COLUMNS)]


def _extra(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([form_extra.extra_form_features(df, _base(df), standard_years=1),
                      form_extra.race_level_features(df, standard_years=1)], axis=1)


def test_changing_future_results_does_not_change_past_extra_features():
    df = _frame()
    feats = _extra(df)
    cutoff = df["race_date"].iloc[len(df) // 2]
    flipped = df.copy()
    future = flipped["race_date"] >= cutoff
    flipped.loc[future, "result_rank"] = 9 - flipped.loc[future, "result_rank"]
    flipped.loc[future, "target_top3"] = (flipped.loc[future, "result_rank"] <= 3).astype(float)
    flipped.loc[future, "target_win"] = (flipped.loc[future, "result_rank"] == 1).astype(float)
    flipped.loc[future, "result_time_sec"] = "99.9"
    flipped.loc[future, "result_margin_sec"] = 2.5
    feats2 = _extra(flipped)
    upto = df["race_date"] <= cutoff
    pd.testing.assert_frame_equal(feats[upto], feats2[upto])
    assert not feats[~upto].equals(feats2[~upto])


def test_last_rank_and_runs_in_year_come_from_earlier_races():
    df = _frame()
    feats = form_extra.extra_form_features(df, _base(df), standard_years=1)
    horse = df[df["ketto_num"] == "H03"].sort_values("race_date")
    first, second, third = horse.index[:3]
    assert np.isnan(feats.at[first, "x_rank_last1"])
    assert feats.at[second, "x_rank_last1"] == horse.at[first, "result_rank"]
    assert feats.at[first, "x_runs_365"] == 0
    assert feats.at[third, "x_runs_365"] == 2   # 1日1レース・週1回なので2走とも365日以内


def test_runs_in_year_does_not_count_rows_without_results():
    df = _frame()
    horse = df[df["ketto_num"] == "H03"].sort_values("race_date")
    first, second, third = horse.index[:3]
    df.loc[first, "has_result"] = False      # 結果の無い行（出馬表だけ）は出走に数えない
    feats = form_extra.extra_form_features(df, _base(df), standard_years=1)
    assert feats.at[second, "x_runs_365"] == 0
    assert feats.at[third, "x_runs_365"] == 1


def test_gap_to_third_is_zero_for_the_third_best_horse():
    df = _frame()
    base = _base(df)
    feats = form_extra.extra_form_features(df, base, standard_years=1)
    race = df["race_id"].iloc[-1]
    rows = df.index[df["race_id"] == race]
    values = base.loc[rows, "f_ae_own"].astype(float)
    third = values.sort_values(ascending=False).index[2]
    assert feats.at[third, "x_gap3_f_ae_own"] == pytest.approx(0.0, abs=1e-6)
    best = values.idxmax()
    assert feats.at[best, "x_gapmax_f_ae_own"] == pytest.approx(0.0, abs=1e-6)


def test_race_level_counts_only_next_runs_before_the_day():
    df = _frame()
    lvl = form_extra.race_level_features(df, standard_years=1)
    from keiba_ml.features.pedigree import beat_fraction
    pf = beat_fraction(df)
    order = df.sort_values(["ketto_num", "race_date"])
    nxt_date = order.groupby("ketto_num")["race_date"].shift(-1)
    nxt_pf = pf.loc[order.index].groupby(order["ketto_num"]).shift(-1)
    prev_race = order.groupby("ketto_num")["race_id"].shift(1)
    # 総当たり: 前走のレースの出走馬の、今日より前の次走の走りの質の平均
    for idx in order.index[300:340]:
        race = prev_race.loc[idx]
        if pd.isna(race):
            continue
        day = df.at[idx, "race_date"]
        members = order.index[order["race_id"] == race]
        # 走りの質が欠損の次走は数えない（race_level_features は欠損を除いて平均する）
        vals = [nxt_pf.loc[m] for m in members
                if pd.notna(nxt_date.loc[m]) and nxt_date.loc[m] < day and pd.notna(nxt_pf.loc[m])]
        expected = np.mean(vals) if vals else np.nan
        got = float(lvl.at[idx, "z_lvl1_npf"])
        assert (np.isnan(expected) and np.isnan(got)) or np.isclose(expected, got, rtol=1e-5)


def test_truncation_check_passes_and_detects_leaks():
    df = _frame()
    feats = _extra(df)
    ok = integrity.check_truncation_sample(df, feats, name="t", columns=list(feats.columns),
                                           builder=_extra, days=3)
    assert ok["status"] == "ok", ok
    # その日の結果を使う「漏れた」列（今回の着順）を混ぜると fail になる
    leaky = feats.assign(x_rank_last1=df["result_rank"].astype(float))
    bad = integrity.check_truncation_sample(
        df, leaky, name="t", columns=["x_rank_last1"],
        builder=lambda part: _extra(part).assign(x_rank_last1=part["result_rank"].astype(float)),
        days=3)
    assert bad["status"] == "fail"


def test_combined_with_extras_builds_and_passes_checks():
    df = _pedigree_frame()
    config = MLConfig(pedigree=PedigreeConfig(generations=3, inbreeding_generations=3),
                      combined=CombinedConfig(stats_history="lead:1", speed_standard_years=1,
                                              use_distance=False, extra_form=True,
                                              race_level=True))
    spec = registry.get("combined")
    fs = spec.build_features(df, config)
    assert {"x_fig_ewm", "x_gap3_f_pf_ewm", "z_lvl1_npf"} <= set(fs.columns)
    assert integrity.check_feature_columns(fs.columns)["status"] == "ok"
    status = {c["name"]: c["status"] for c in spec.asof_checks(df, fs, config, None)}
    assert status["form_extra_truncation"] == "ok", status
    broken = replace(fs, frame=fs.frame.assign(x_fig_ewm=fs.frame["x_fig_ewm"] + 0.5))
    bad = {c["name"]: c["status"] for c in spec.asof_checks(df, broken, config, None)}
    assert bad["form_extra_truncation"] == "fail"


def test_use_weight_false_drops_weight_columns():
    df = _pedigree_frame()
    config = MLConfig(pedigree=PedigreeConfig(generations=3, inbreeding_generations=3),
                      combined=CombinedConfig(stats_history="lead:1", speed_standard_years=1,
                                              use_distance=False, extra_form=True,
                                              use_weight=False))
    fs = registry.get("combined").build_features(df, config)
    assert not {"c_weight", "c_zogen", "x_weight_vs_mean"} & set(fs.columns)
    assert {"c_futan", "x_fig_ewm"} <= set(fs.columns)
    assert set(fs.columns) <= set(fs.frame.columns)


def test_extra_form_requires_opponent():
    with pytest.raises(ValueError, match="opponent"):
        validate(replace(MLConfig(), combined=CombinedConfig(extra_form=True, opponent=False)))
