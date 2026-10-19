"""距離適性モデル: as-of 集計・特徴量・設定・登録表。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from keiba_ml import registry
from keiba_ml.config import (
    CONFIG_DIR, MODEL_KINDS, DataConfig, DistanceConfig, MLConfig, CodeConfig, config_from_dict,
    load_config,
)
from keiba_ml.dataset import integrity
from keiba_ml.dataset.races import build_runners
from keiba_ml.features import distance
from keiba_ml.features.asof import asof
from tests.fixtures import csv_data


# --- as-of 集計 ---------------------------------------------------------------

def _day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def test_asof_uses_only_strictly_earlier_days():
    df = pd.DataFrame({
        "key": ["a", "a", "b", "a", "a", ""],
        "race_date": [_day("2020-01-01"), _day("2020-01-05"), _day("2020-01-05"),
                      _day("2020-01-09"), _day("2020-01-13"), _day("2020-01-13")],
        "v": [1.0, 2.0, 10.0, np.nan, 5.0, 7.0],
        "has": [True, True, True, True, False, True],
    })
    out = asof(df, ["key"], {"s": ("v", "sum"), "mx": ("v", "max"), "mn": ("v", "min"),
                             "last": ("v", "last")}, include=df["has"])
    # 1/5 の行は同じ日の自分（2.0）を含まない
    assert out.loc[1].to_dict() == {"s": 1.0, "mx": 1.0, "mn": 1.0, "last": 1.0}
    # 他のキー（b）の値は入らない。前の履歴が無ければ合計 0・それ以外 NaN
    assert out.loc[2, "s"] == 0.0 and np.isnan(out.loc[2, "mx"])
    # 1/13 の行: 1/9 の値は NaN なので、最大・最小・直前は 1/5 までの値のまま
    assert out.loc[4].to_dict() == {"s": 3.0, "mx": 2.0, "mn": 1.0, "last": 2.0}
    # キーが欠損の行は集計しない
    assert out.loc[5, "s"] == 0.0 and np.isnan(out.loc[5, "last"])


def test_asof_rejects_unknown_aggregation():
    df = pd.DataFrame({"key": ["a"], "race_date": [_day("2020-01-01")], "v": [1.0]})
    with pytest.raises(ValueError):
        asof(df, ["key"], {"x": ("v", "median")}, include=[True])


# --- 特徴量 -------------------------------------------------------------------

def _run(ketto, day, kyori, rank, *, n=10, has=True, surface="turf", margin=None,
         sire="S1", weight=480.0, race=None):
    band = "S" if kyori < 1400 else "M" if kyori < 1800 else "I" if kyori < 2200 else "L"
    return {
        "race_id": race or f"{day.replace('-', '')}{kyori}", "race_date": _day(day),
        "ketto_num": ketto, "kyori_m": kyori, "track_kind": surface, "distance_band": band,
        "has_result": has, "result_rank": rank if has else pd.NA, "field_size": n,
        "result_margin_sec": (0.0 if rank == 1 else 0.2 * rank) if has and margin is None
        else margin,
        "target_top3": (1.0 if rank <= 3 else 0.0) if has else np.nan,
        "result_weight_kg": weight if has else np.nan,
        "ped_S": sire, "ped_DS": "DS1", "jyo_cd": "05", "age": 4, "sex_cd": "1",
        "umaban": "01",
    }


def _sprinter_history() -> pd.DataFrame:
    """1200m で好走し、2000m で大敗する馬（X）と、同じ父の別の馬（Y）。"""
    rows = [
        _run("X", "2020-01-05", 1200, 1), _run("X", "2020-02-05", 2000, 10),
        _run("X", "2020-03-05", 1200, 2), _run("X", "2020-04-05", 2000, 9),
        _run("X", "2020-05-05", 1200, 1, weight=490.0),
        _run("Y", "2020-01-05", 1200, 3, race="y1"), _run("Y", "2020-02-05", 2400, 8, race="y2"),
        # 予想対象（結果なし）: 同じ日に 1200m と 2000m を仮に並べる
        _run("X", "2020-06-05", 1200, 0, has=False, race="t1200"),
        _run("X", "2020-06-05", 2000, 0, has=False, race="t2000"),
    ]
    return pd.DataFrame(rows)


def test_sprinter_looks_better_at_short_distance():
    df = _sprinter_history()
    frame = distance.build_features(df, DistanceConfig()).frame
    short, long_ = frame.index[df["race_id"] == "t1200"][0], frame.index[df["race_id"] == "t2000"][0]
    for col in ("own_rel_pf_dband", "own_rel_pm_dband", "own_rel_pf_surface_dband",
                "own_fit_pf", "own_fit_pm"):
        assert frame.at[short, col] > 0 > frame.at[long_, col], col
    assert frame.at[short, "own_slope_pf"] < 0
    assert frame.at[short, "own_log_runs"] == pytest.approx(np.log1p(5))
    assert frame.at[short, "own_log_runs_dband"] == pytest.approx(np.log1p(3))
    assert frame.at[short, "dist_minus_last"] == 0.0
    assert frame.at[long_, "dist_minus_last"] == 800.0
    assert frame.at[long_, "dist_minus_max"] == 0.0
    assert frame.at[short, "dist_minus_min"] == 0.0
    assert frame.at[short, "dist_minus_top3_mean"] == 0.0     # 3着以内は全部 1200m
    assert frame.at[short, "dist_minus_mean"] == pytest.approx(1200 - 7600 / 5)
    assert frame.at[short, "own_last_weight"] == 490.0
    # 父の産駒（本人を除く = Y だけ）が3着以内に入ったのは 1200m
    assert frame.at[long_, "ped_S_dist_minus_top3_mean"] == 800.0
    assert frame.at[short, "ped_S_log_runs"] == pytest.approx(np.log1p(2))


def test_first_runner_has_no_own_history_but_has_pedigree():
    df = pd.DataFrame([_run("Y", "2020-01-05", 1200, 1, race="y1"),
                       _run("Z", "2020-03-05", 1200, 0, has=False, race="z")])
    frame = distance.build_features(df, DistanceConfig()).frame
    z = frame.index[df["race_id"] == "z"][0]
    assert frame.at[z, "own_log_runs"] == 0.0
    assert np.isnan(frame.at[z, "own_rel_pf_dband"])
    assert np.isnan(frame.at[z, "dist_minus_mean"])
    assert frame.at[z, "ped_S_dist_minus_top3_mean"] == 0.0


def test_stopped_runs_count_as_experience_but_not_as_performance():
    df = pd.DataFrame([
        _run("X", "2020-01-05", 1200, 1),
        _run("X", "2020-02-05", 2400, 5, margin=np.nan),     # 競走中止（着差なし）
        _run("X", "2020-03-05", 1200, 0, has=False, race="t"),
    ])
    frame = distance.build_features(df, DistanceConfig()).frame
    t = frame.index[df["race_id"] == "t"][0]
    assert frame.at[t, "own_log_runs"] == pytest.approx(np.log1p(2))
    assert frame.at[t, "dist_minus_max"] == -1200.0
    assert frame.at[t, "own_rel_pf_dband"] == pytest.approx(0.0)  # 走りの質は 1200m の1走だけ


def test_features_do_not_change_when_a_later_result_changes():
    df = _sprinter_history()
    base = distance.build_features(df, DistanceConfig()).frame
    changed = df.copy()
    last = changed.index[(changed["ketto_num"] == "X") & (changed["race_date"] == _day("2020-05-05"))]
    changed.loc[last, ["result_rank", "result_margin_sec", "target_top3"]] = [10, 3.0, 0.0]
    after = distance.build_features(changed, DistanceConfig()).frame
    earlier = df.index[df["race_date"] <= _day("2020-05-05")]
    pd.testing.assert_frame_equal(base.loc[earlier], after.loc[earlier])
    later = df.index[df["race_date"] > _day("2020-05-05")]
    assert not base.loc[later].equals(after.loc[later])


def test_feature_names_pass_the_forbidden_column_check():
    features = distance.build_features(_sprinter_history(), DistanceConfig())
    assert integrity.check_feature_columns(features.columns)["status"] == "ok"
    assert set(features.cat_features) == {"track_kind", "distance_band", "jyo_cd", "sex_cd"}


def test_distance_asof_check_detects_a_tampered_feature():
    df = _sprinter_history()
    features = distance.build_features(df, DistanceConfig())
    ok = integrity.check_distance_asof_sample(df, features.frame, config=DistanceConfig(),
                                              sample=50)
    assert ok["status"] == "ok", ok
    tampered = features.frame.copy()
    tampered["dist_minus_last"] = tampered["dist_minus_last"] + 1.0
    bad = integrity.check_distance_asof_sample(df, tampered, config=DistanceConfig(), sample=50)
    assert bad["status"] == "fail"


def test_distance_baselines_rank_within_each_race():
    df = _sprinter_history()
    features = distance.build_features(df, DistanceConfig())
    rows = df[~df["has_result"]]
    preds = registry.get("distance").baselines(features, rows, "live")
    assert set(preds) == {"dist_rel_band", "dist_gap"}
    for pred in preds.values():
        assert list(pred["pred_rank"]) == [1, 1]          # 1頭ずつのレース


# --- データセットの列 -----------------------------------------------------------

def test_dataset_keeps_margin_and_weight_only_for_finishers():
    rid = "2026032106011101"
    races = pd.DataFrame([{k: str(v) for k, v in
                           csv_data.make_race(race_id=rid, data_kubun="7").items()}])
    runners = pd.DataFrame([{k: str(v) for k, v in csv_data.make_race_horse(**r).items()} for r in (
        {"race_id": rid, "umaban": "01", "ketto_num": "2020100001", "result_kakutei_jyuni": "1",
         "result_margin_sec": "0.0", "result_ba_taijyu_kg": "480"},
        {"race_id": rid, "umaban": "02", "ketto_num": "2020100002", "result_kakutei_jyuni": "2",
         "result_margin_sec": "0.3", "result_ba_taijyu_kg": "999"},     # 計不
        {"race_id": rid, "umaban": "03", "ketto_num": "2020100003", "result_kakutei_jyuni": "9",
         "result_ijyo_cd": "4", "result_margin_sec": "1.0"},            # 競走中止
    )])
    df = build_runners(races, runners, data=DataConfig(), codes_config=CodeConfig())
    assert df["result_margin_sec"].tolist()[:2] == [0.0, 0.3]
    assert np.isnan(df["result_margin_sec"].iloc[2])
    assert df["result_weight_kg"].iloc[0] == 480.0 and np.isnan(df["result_weight_kg"].iloc[1])


# --- 設定と登録表 -------------------------------------------------------------

def test_registry_matches_model_kinds_and_each_has_a_config_file():
    assert tuple(registry.MODELS) == MODEL_KINDS
    for name in MODEL_KINDS:
        config = load_config(model=name)
        assert config.experiment.model == name
        assert registry.for_config(config).name == name
        assert (CONFIG_DIR / f"{name}.toml").exists()


def test_distance_and_pedigree_configs_can_share_one_dataset():
    from keiba_ml.dataset.build import manifest_config
    assert manifest_config(load_config(model="distance")) == \
        manifest_config(load_config(model="pedigree"))


def test_load_config_rejects_a_file_for_another_model():
    with pytest.raises(ValueError, match="pedigree"):
        load_config(CONFIG_DIR / "pedigree.toml", model="distance")


def test_old_run_config_without_experiment_reads_as_pedigree():
    assert config_from_dict({"model": {"iterations": 10}}).experiment.model == "pedigree"


@pytest.mark.parametrize("raw", [
    {"experiment": {"model": "speed"}},
    {"distance": {"stats_history": "forever"}},
    {"distance": {"own_shrinkage": 0.0}},
    {"distance": {"slope_ridge": -1.0}},
    {"distance": {"pedigree_paths": ["SSSSSS"]}},
])
def test_invalid_distance_settings_are_rejected(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_missing_columns_are_reported_for_old_datasets():
    spec = registry.get("distance")
    assert spec.missing_columns(pd.DataFrame({"race_id": []})) == [
        "result_margin_sec", "result_weight_kg"]
    assert registry.get("pedigree").missing_columns(pd.DataFrame()) == []


def test_unknown_model_name():
    with pytest.raises(KeyError):
        registry.get("speed")
    assert MLConfig().experiment.model == "pedigree"
    assert Path(CONFIG_DIR).name == "ml"
