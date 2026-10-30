"""血統モデルの拡張特徴量（stx_ / rel_ / birth_doy）。as-of で未来の結果と本人の走りが入らないこと。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from keiba_ml.config import FeatureConfig, PedigreeConfig
from keiba_ml.dataset import integrity
from keiba_ml.features.pedigree import (
    QUALITY_PATHS, asof_mean, beat_fraction, build_features, prepare)

PED = PedigreeConfig(generations=3, categorical_generations=0, inbreeding_generations=3,
                     stat_paths=("S", "D"), shrinkage=10.0)
EXT = replace(PED, extended_stats=True, race_relative=True)
CTX = FeatureConfig(context=(*FeatureConfig().context, "grade_tier", "birth_doy"))


def _random_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        race = i // 8
        rank = i % 8 + 1
        records.append({
            "race_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=race // 2),
            "race_id": f"r{race}", "umaban": f"{rank:02d}",
            "ketto_num": f"H{rng.integers(0, 60)}",
            "has_result": True, "result_rank": int(rng.permutation(8)[0] + 1),
            "result_ijyo_cd": "0", "field_size": 8,
            "track_kind": ["turf", "dirt"][race % 2], "distance_band": "M", "baba_cd": "1",
            "jyo_cd": f"0{race % 3 + 1}", "age": 2 + race % 3, "sex_cd": "1",
            "kyori_m": 1600, "month": 5, "grade_tier": int(race % 6),
            "birth_date": f"201{7 + race % 3}-0{1 + i % 5}-1{i % 9}",
            "inbreeding_f": 0.0, "n_cross": 0, "closest_cross": np.nan,
            **{f"ped_{p}": f"{p}{rng.integers(0, 5)}" for p in
               ("S", "D", "SS", "SD", "DS", "DD", "SSS", "SSD", "SDS", "SDD", "DSS", "DSD",
                "DDS", "DDD")},
        })
    df = pd.DataFrame(records)
    df["target_top3"] = (df["result_rank"] <= 3).astype(float)
    df["target_win"] = (df["result_rank"] == 1).astype(float)
    return df


def test_extended_features_are_off_by_default():
    base = build_features(_random_frame(), pedigree=PED, features=FeatureConfig())
    assert not [c for c in base.columns if c.startswith(("stx_", "rel_"))]
    assert "birth_doy" not in base.columns


def test_extended_features_are_numeric_and_present():
    fs = build_features(_random_frame(), pedigree=EXT, features=CTX)
    stx = [c for c in fs.num_features if c.startswith("stx_")]
    rel = [c for c in fs.num_features if c.startswith("rel_")]
    assert stx and rel and "birth_doy" in fs.num_features and "grade_tier" in fs.num_features
    assert integrity.check_feature_columns(fs.columns)["status"] == "ok"
    assert all(fs.frame[c].dtype == np.float32 for c in stx + rel)


def test_changing_future_results_does_not_change_past_extended_features():
    df = _random_frame()
    feats = build_features(df, pedigree=EXT, features=CTX).frame
    cutoff = df["race_date"].iloc[len(df) // 2]
    flipped = df.copy()
    future = flipped["race_date"] >= cutoff
    flipped.loc[future, "result_rank"] = 9 - flipped.loc[future, "result_rank"]
    flipped.loc[future, "target_top3"] = (flipped.loc[future, "result_rank"] <= 3).astype(float)
    flipped.loc[future, "target_win"] = (flipped.loc[future, "result_rank"] == 1).astype(float)
    feats2 = build_features(flipped, pedigree=EXT, features=CTX).frame
    upto = df["race_date"] <= cutoff
    pd.testing.assert_frame_equal(feats[upto], feats2[upto])
    assert not feats[~upto].equals(feats2[~upto])


def test_own_runs_do_not_enter_quality_features():
    df = prepare(_random_frame())
    df["_pf"] = beat_fraction(df)
    include = df["has_result"] & df["_pf"].notna()
    # 母 "solo" の産駒は H_solo だけ。本人の走りを除くと兄弟の履歴は空で、全体の平均になる
    solo = df["ketto_num"] == df["ketto_num"].iloc[0]
    df.loc[solo, "ped_D"] = "solo"
    df.loc[~solo & (df["ped_D"] == "solo"), "ped_D"] = "other"
    without = asof_mean(df, ["ped_D"], "_pf", include, prior_n=10.0, exclude_self=True)
    with_self = asof_mean(df, ["ped_D"], "_pf", include, prior_n=10.0, exclude_self=False)
    overall = asof_mean(df.assign(ped_D="all"), ["ped_D"], "_pf", include, prior_n=1e-12,
                        exclude_self=False)
    rows = df.index[solo][1:]   # 2走目以降（本人の過去走がある）
    assert len(rows) > 0
    assert np.allclose(without[rows], overall[rows], atol=1e-6)
    assert not np.allclose(with_self[rows], without[rows])


def test_extended_asof_check_agrees_and_detects_leaks():
    df = _random_frame()
    feats = build_features(df, pedigree=EXT, features=CTX).frame
    ok = integrity.check_extended_asof_sample(df, feats, paths=QUALITY_PATHS, shrinkage=10.0,
                                              exclude_self=True, sample=40)
    assert ok["status"] == "ok", ok
    broken = feats.copy()
    broken["stx_D_pf"] = broken["stx_D_pf"] + 0.01
    bad = integrity.check_extended_asof_sample(df, broken, paths=QUALITY_PATHS, shrinkage=10.0,
                                               exclude_self=True, sample=40)
    assert bad["status"] == "fail"


def test_race_relative_is_within_race():
    df = _random_frame()
    fs = build_features(df, pedigree=EXT, features=CTX).frame
    diff = fs["rel_st_S_ae_top3_diff"].groupby(df["race_id"]).sum()
    assert np.allclose(diff, 0.0, atol=1e-4)
    pct = fs["rel_st_S_ae_top3_pct"]
    assert ((pct > 0) & (pct <= 1)).all()


def test_parent_racing_uses_only_races_before_the_foal_was_born():
    from keiba_ml.dataset.build import attach_parent_racing
    from keiba_ml.features.pedigree import parent_racing_features

    def run(day, race, ketto, rank, tier, dam="", birth="2010-01-01"):
        return {"race_date": pd.Timestamp(day), "race_id": race, "ketto_num": ketto,
                "has_result": True, "result_rank": rank, "result_ijyo_cd": "0",
                "field_size": 10, "grade_tier": tier, "target_top3": float(rank <= 3),
                "birth_date": birth, "ped_S": "", "ped_D": dam, "ped_DS": ""}

    df = pd.DataFrame([
        run("2015-05-01", "a", "MARE", 1, 6, birth="2012-03-01"),    # 母自身。子の生まれる前
        run("2015-06-01", "b", "MARE", 10, 2, birth="2012-03-01"),
        run("2019-06-01", "c", "MARE", 1, 8, birth="2012-03-01"),    # 子の生まれた後（入れない）
        run("2021-06-01", "d", "FOAL", 5, 1, dam="2001", birth="2019-04-01"),
        run("2021-06-01", "d", "OTHER", 6, 1, dam="9999", birth="2019-04-01"),
    ])
    hn = pd.DataFrame({"hansyoku_num": ["2001", "9999"], "ketto_num": ["MARE", ""]})
    out = attach_parent_racing(df, hn)
    foal = out[out["ketto_num"] == "FOAL"].iloc[0]
    assert foal["parent_D_n"] == 2 and foal["parent_D_k3"] == 1
    assert foal["parent_D_tier3"] == 6 and foal["parent_D_tiermax"] == 6
    assert foal["parent_D_linked"] == 1
    other = out[out["ketto_num"] == "OTHER"].iloc[0]
    assert other["parent_D_n"] == 0 and other["parent_D_linked"] == 0
    feats = parent_racing_features(out)
    assert np.isnan(feats.loc[other.name, "par_D_ae"])
    assert feats.loc[foal.name, "par_D_ae"] == np.float32((1 + 3) / (0.6 + 3))
    assert integrity.check_feature_columns(list(feats.columns))["status"] == "ok"


def test_parent_racing_requires_schema3_columns():
    import pytest

    df = _random_frame()
    with pytest.raises(ValueError, match="schema 3"):
        build_features(df, pedigree=replace(EXT, parent_racing=True), features=CTX)


def test_breeder_stats_are_asof_and_exclude_self():
    from keiba_ml.features.pedigree import breeder_features

    df = _random_frame()
    df["breeder_code"] = [f"B{i % 4}" for i in range(len(df))]
    cfg = replace(PED, breeder_stats=True)
    feats = build_features(df, pedigree=cfg, features=CTX).frame
    assert {"brd_ae_all", "brd_ae_surface", "brd_ae_age", "brd_log_n", "brd_pf"} <= set(feats)
    cutoff = df["race_date"].iloc[len(df) // 2]
    flipped = df.copy()
    future = flipped["race_date"] >= cutoff
    flipped.loc[future, "target_top3"] = 1 - flipped.loc[future, "target_top3"]
    flipped.loc[future, "result_rank"] = 9 - flipped.loc[future, "result_rank"]
    feats2 = build_features(flipped, pedigree=cfg, features=CTX).frame
    upto = df["race_date"] <= cutoff
    pd.testing.assert_frame_equal(feats[upto], feats2[upto])
    # 生産者が分からない馬は履歴に入らず、出走0の A/E（= 1.0）になる
    blank = prepare(df.assign(breeder_code=""))
    assert (breeder_features(blank, cfg)["brd_ae_all"] == 1.0).all()


def test_breeder_stats_require_breeder_code():
    import pytest

    with pytest.raises(ValueError, match="breeder_code"):
        build_features(_random_frame(), pedigree=replace(PED, breeder_stats=True), features=CTX)


def test_birth_doy_is_day_of_year():
    df = prepare(pd.DataFrame({"age": [3], "baba_cd": ["1"], "field_size": [10],
                               "birth_date": ["2021-02-01"]}))
    assert df["birth_doy"].iloc[0] == 32
