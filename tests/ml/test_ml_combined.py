"""総合モデル（combined）の特徴量: 血統以外（近走・タイム指数・人・直前の条件）が as-of であること。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from keiba_ml import registry
from keiba_ml.config import CombinedConfig, MLConfig, PedigreeConfig, validate
from keiba_ml.dataset import integrity
from keiba_ml.dataset.split import history_start
from keiba_ml.features import form


def _frame(n_races: int = 60, field: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(n_races):
        # 1日1レース（同じ馬が同じ日に2回走らない。中央の規則と同じ）
        day = pd.Timestamp("2018-01-06") + pd.Timedelta(days=7 * race)
        horses = rng.choice(30, size=field, replace=False)
        ranks = rng.permutation(field) + 1
        for j, (h, rank) in enumerate(zip(horses, ranks)):
            rows.append({
                "race_id": f"{day:%Y%m%d}{race % 3:02d}", "race_date": day,
                "umaban": f"{j + 1:02d}", "ketto_num": f"H{h:02d}", "field_size": field,
                "jyo_cd": "05", "track_kind": ["turf", "dirt"][race % 2], "kyori_m": 1600,
                "distance_band": "M", "baba_cd": "1", "grade_tier": int(race % 4),
                "age": 3, "sex_cd": "1", "month": day.month,
                "has_result": True, "result_rank": int(rank), "result_ijyo_cd": "0",
                "target_top3": float(rank <= 3), "target_win": float(rank == 1),
                "result_margin_sec": 0.2 * (rank - 1), "result_weight_kg": 470.0 + j,
                "wakuban": str(j // 2 + 1), "futan_kg": "560", "kisyu_code": f"J{(h + race) % 5}",
                "chokyosi_code": f"T{h % 4}", "tozai_cd": "1", "blinker": "0",
                "weight_change_kg": "+2", "result_time_sec": f"{96 + 0.2 * rank:.1f}",
                "result_jyuni_4c": str(rank), "result_last_3f_sec": f"{34 + 0.1 * rank:.1f}",
            })
    return pd.DataFrame(rows)


def _all_form(df: pd.DataFrame) -> pd.DataFrame:
    from keiba_ml.features.pedigree import prepare
    speed = form.speed_figures(df, standard_years=1)
    return pd.concat([form.own_history_features(df), speed,
                      form.people_features(prepare(df), shrinkage=10.0),
                      form.condition_features(df),
                      form.opponent_strength(df, speed["s_fig_best"]),
                      form.recent_people_features(prepare(df), days=28, shrinkage=5.0)], axis=1)


def test_recent_people_counts_only_the_last_days():
    from keiba_ml.features.pedigree import prepare
    df = _frame()
    recent = form.recent_people_features(prepare(df), days=28, shrinkage=5.0)
    # 総当たり: 騎手 J の、(その日 − 28日) 以降・その日より前の結果だけ
    from keiba_ml.features.pedigree import expected_top3
    for idx in df.index[200:230]:
        row = df.loc[idx]
        hist = df[(df["kisyu_code"] == row["kisyu_code"]) & (df["race_date"] < row["race_date"])
                  & (df["race_date"] >= row["race_date"] - pd.Timedelta(days=28))]
        value = (hist["target_top3"].sum() + 5.0) / (expected_top3(hist["field_size"]).sum() + 5.0)
        assert np.isclose(recent.at[idx, "h_jockey_ae_recent"], value, rtol=1e-5)


def test_changing_future_results_does_not_change_past_form_features():
    df = _frame()
    feats = _all_form(df)
    cutoff = df["race_date"].iloc[len(df) // 2]
    flipped = df.copy()
    future = flipped["race_date"] >= cutoff
    flipped.loc[future, "result_rank"] = 9 - flipped.loc[future, "result_rank"]
    flipped.loc[future, "target_top3"] = (flipped.loc[future, "result_rank"] <= 3).astype(float)
    flipped.loc[future, "target_win"] = (flipped.loc[future, "result_rank"] == 1).astype(float)
    flipped.loc[future, "result_time_sec"] = "99.9"
    flipped.loc[future, "result_last_3f_sec"] = "39.9"
    flipped.loc[future, "result_margin_sec"] = 2.5
    feats2 = _all_form(flipped)
    upto = df["race_date"] <= cutoff
    pd.testing.assert_frame_equal(feats[upto], feats2[upto])
    assert not feats[~upto].equals(feats2[~upto])


def test_same_horse_twice_on_one_day_is_refused():
    df = _frame()
    twin = df.iloc[[0]].assign(race_id="other", umaban="01")
    with pytest.raises(ValueError, match="同じ日に2回"):
        form.own_history_features(pd.concat([df, twin], ignore_index=True))


def test_last_run_values_come_from_the_previous_race():
    df = _frame()
    feats = form.own_history_features(df)
    horse = df[df["ketto_num"] == "H03"].sort_values("race_date")
    first, second = horse.index[0], horse.index[1]
    assert np.isnan(feats.at[first, "f_pf_last1"])
    rank = horse.at[first, "result_rank"]
    assert feats.at[second, "f_pf_last1"] == np.float32((8 - rank) / 7)
    assert feats.at[second, "f_days_since"] == (horse.at[second, "race_date"]
                                                - horse.at[first, "race_date"]).days


def test_form_asof_check_agrees_and_detects_leaks():
    df = _frame()
    feats = _all_form(df)
    ok = integrity.check_form_asof_sample(df, feats, people_shrinkage=10.0, sample=40)
    assert ok["status"] == "ok", ok
    broken = feats.copy()
    broken["h_jockey_ae"] = broken["h_jockey_ae"] + 0.01
    assert integrity.check_form_asof_sample(df, broken, people_shrinkage=10.0,
                                            sample=40)["status"] == "fail"


def test_combined_features_have_no_result_columns_and_no_odds():
    df = _frame()
    for col in ("birth_date", "breeder_code"):
        df[col] = ""
    for p in ("S", "D", "SS", "SD", "DS", "DD", "SSS", "SSD", "SDS", "SDD", "DSS", "DSD", "DDS",
              "DDD"):
        df[f"ped_{p}"] = f"{p}1"
    df[["inbreeding_f", "n_cross", "closest_cross"]] = [0.0, 0, np.nan]
    config = MLConfig(pedigree=PedigreeConfig(generations=3, inbreeding_generations=3),
                      combined=CombinedConfig(stats_history="lead:1", speed_standard_years=1,
                                              use_distance=False))
    fs = registry.get("combined").build_features(df, config)
    assert integrity.check_feature_columns(fs.columns)["status"] == "ok"
    assert not [c for c in fs.columns if "odds" in c or "ninki" in c]
    assert {"f_pf_ewm", "s_fig_mean3", "h_jockey_ae", "c_futan", "rel_f_pf_ewm_diff"} <= \
        set(fs.columns)


def test_history_start_lead_years():
    from datetime import date
    assert history_start(date(2014, 1, 1), "lead:5") == date(2009, 1, 1)
    assert history_start(date(2014, 1, 1), "window") == date(2014, 1, 1)
    assert history_start(date(2014, 1, 1), "all") is None
    with pytest.raises(ValueError):
        history_start(date(2014, 1, 1), "lead:x")


def test_combined_config_validation():
    base = MLConfig()
    validate(base)
    with pytest.raises(ValueError, match="lead"):
        validate(replace(base, combined=CombinedConfig(stats_history="window")))
    with pytest.raises(ValueError, match="speed_standard_years"):
        validate(replace(base, combined=CombinedConfig(stats_history="lead:3",
                                                       speed_standard_years=5)))
