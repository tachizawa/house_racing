"""特徴量（as-of の未来情報なし）と評価指標。"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from keiba_ml.config import FeatureConfig, PedigreeConfig
from keiba_ml.dataset import integrity
from keiba_ml.evaluate import metrics
from keiba_ml.features.pedigree import asof_sums, build_features, expected_top3, prepare
from keiba_ml.models.base import make_predictions

PED = PedigreeConfig(generations=3, categorical_generations=3, inbreeding_generations=3,
                     stat_paths=("S", "D"), shrinkage=10.0)


def _frame(rows):
    """rows: (date, race_id, ketto, sire, dam, top3, field_size[, has_result])"""
    records = []
    for i, r in enumerate(rows):
        day, race, ketto, sire, dam, top3, n = r[:7]
        has = r[7] if len(r) > 7 else True
        records.append({
            "race_date": pd.Timestamp(day), "race_id": race, "umaban": f"{i:02d}",
            "ketto_num": ketto, "ped_S": sire, "ped_D": dam,
            "has_result": has, "target_top3": float(top3) if has else np.nan,
            "target_win": 0.0 if has else np.nan, "field_size": n,
            "track_kind": "turf", "distance_band": "M", "baba_cd": "1", "jyo_cd": "05",
            "age": 3, "sex_cd": "1", "kyori_m": 1600, "month": 5,
            "inbreeding_f": 0.0, "n_cross": 0, "closest_cross": np.nan,
            **{f"ped_{p}": "" for p in ("SS", "SD", "DS", "DD", "SSS", "SSD", "SDS",
                                       "SDD", "DSS", "DSD", "DDS", "DDD")},
        })
    return pd.DataFrame(records)


def test_expected_top3():
    assert list(expected_top3(pd.Series([3, 6, 10]))) == [1.0, 0.5, 0.3]


def test_asof_uses_only_strictly_earlier_days_and_excludes_self():
    df = prepare(_frame([
        ("2020-01-01", "r1", "A", "S1", "d1", 1, 10),
        ("2020-01-01", "r2", "B", "S1", "d2", 1, 10),   # 同じ日。互いに見えない
        ("2020-02-01", "r3", "A", "S1", "d1", 0, 10),   # A 自身の過去走は除く → B の1走だけ
        ("2020-03-01", "r4", "C", "S1", "d3", 0, 10),
    ]))
    sums = asof_sums(df, ["ped_S"], exclude_self=True)
    assert list(sums["n"]) == [0, 0, 1, 3]
    assert list(sums["k3"]) == [0, 0, 1, 2]
    with_self = asof_sums(df, ["ped_S"], exclude_self=False)
    assert list(with_self["n"]) == [0, 0, 2, 3]


def test_rows_without_result_get_history_but_add_none():
    df = prepare(_frame([
        ("2020-01-01", "r1", "A", "S1", "d1", 1, 10),
        ("2020-02-01", "r2", "B", "S1", "d2", 0, 10, False),   # 出馬表のみ
        ("2020-03-01", "r3", "C", "S1", "d3", 0, 10),
    ]))
    sums = asof_sums(df, ["ped_S"], exclude_self=True)
    assert list(sums["n"]) == [0, 1, 1]


def test_changing_future_results_does_not_change_past_features():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i // 5)
        rows.append((day, f"r{i // 5}", f"H{i % 40}", f"S{i % 4}", f"D{i % 13}",
                     int(rng.random() < 0.3), 10))
    df = _frame(rows)
    feats = build_features(df, pedigree=PED, features=FeatureConfig()).frame
    cutoff = pd.Timestamp("2020-02-01")
    flipped = df.copy()
    future = flipped["race_date"] >= cutoff
    flipped.loc[future, "target_top3"] = 1 - flipped.loc[future, "target_top3"]
    flipped.loc[future, "target_win"] = 1 - flipped.loc[future, "target_win"]   # st_*_ae_win も見る
    feats2 = build_features(flipped, pedigree=PED, features=FeatureConfig()).frame
    past = df["race_date"] < cutoff
    # cutoff 当日の行も、cutoff 当日の結果を見ないので変わらない
    same_day_or_before = df["race_date"] <= cutoff
    pd.testing.assert_frame_equal(feats[same_day_or_before], feats2[same_day_or_before])
    assert not feats[~past].equals(feats2[~past])


def test_asof_sample_check_agrees_with_features():
    rng = np.random.default_rng(1)
    rows = [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i // 4), f"r{i // 4}",
             f"H{i % 30}", f"S{i % 3}", f"D{i % 9}", int(rng.random() < 0.3), 8 + i % 5)
            for i in range(200)]
    df = _frame(rows)
    feats = build_features(df, pedigree=PED, features=FeatureConfig()).frame
    check = integrity.check_asof_sample(df, feats, path="S", shrinkage=10.0,
                                        exclude_self=True, sample=50)
    assert check["status"] == "ok", check
    broken = feats.copy()
    broken["st_S_ae_top3"] = broken["st_S_ae_top3"] + 0.01
    assert integrity.check_asof_sample(df, broken, path="S", shrinkage=10.0,
                                       exclude_self=True, sample=50)["status"] == "fail"


def test_feature_set_has_no_result_columns():
    df = _frame([("2020-01-01", "r1", "A", "S1", "d1", 1, 10)])
    fs = build_features(df, pedigree=PED, features=FeatureConfig())
    assert integrity.check_feature_columns(fs.columns)["status"] == "ok"
    assert "ped_S" in fs.cat_features and "age" in fs.num_features
    assert fs.frame.loc[0, "ped_SS"] == "NA"


# --- metrics ---------------------------------------------------------------

def _race(scores, ranks, race_id="r1"):
    rows = pd.DataFrame({"race_id": race_id, "umaban": [f"{i + 1:02d}" for i in range(len(scores))],
                         "ketto_num": [f"k{i}" for i in range(len(scores))]})
    truth = rows.copy()
    truth["result_rank"] = pd.array(ranks, dtype="Int64")
    truth["target_top3"] = [1.0 if r is not None and r <= 3 else 0.0 for r in ranks]
    pred = make_predictions(rows, np.array(scores, dtype=float), model_name="m",
                            model_version="v", fold="test")
    return pred, truth


def test_per_race_hits_and_chance():
    pred, truth = _race([0.9, 0.8, 0.1, 0.7, 0.2, 0.3], [1, 5, 2, 3, 4, 6])
    races = metrics.per_race(pred, truth)
    row = races.iloc[0]
    assert row["hits"] == 2            # 01, 04 が当たり、02 は外れ
    assert row["all_hit"] == 0.0
    assert row["winner_in_pred"] == 1.0
    assert row["top1_win"] == 1.0
    assert row["random_hits"] == pytest.approx(3 * 3 / 6)
    assert row["random_all_hit"] == pytest.approx(1 / comb(6, 3))


def test_dead_heat_counts_four_horses_in_top3():
    pred, truth = _race([0.9, 0.8, 0.7, 0.1, 0.0], [1, 2, 3, 3, 5])
    row = metrics.per_race(pred, truth).iloc[0]
    assert row["actual_top3"] == 4
    assert row["all_hit"] == 1.0
    assert row["random_all_hit"] == pytest.approx(comb(4, 3) / comb(5, 3))


def test_perfect_prediction_has_ndcg_one():
    pred, truth = _race([0.9, 0.8, 0.7, 0.1], [1, 2, 3, 4])
    assert metrics.per_race(pred, truth).iloc[0]["ndcg"] == pytest.approx(1.0)


def test_ties_in_score_break_by_umaban():
    pred, _ = _race([0.5, 0.5, 0.5], [1, 2, 3])
    assert list(pred["pred_rank"]) == [1, 2, 3]


def test_calibration_of_perfectly_calibrated_bins():
    prob = np.array([0.2] * 10 + [0.8] * 10)
    y = np.array([1, 1] + [0] * 8 + [1] * 8 + [0, 0], dtype=float)
    result = metrics.calibration(prob, y, bins=10)
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)
    assert result["observed_rate"] == pytest.approx(0.5)


def test_summary_reports_lift_against_chance():
    pred, truth = _race([0.9, 0.8, 0.7, 0.1, 0.0, 0.0], [1, 2, 3, 4, 5, 6])
    summary = metrics.summarize(metrics.per_race(pred, truth), n_boot=10)
    assert summary["hits"]["mean"] == 3
    assert summary["hits"]["lift_vs_random"] == pytest.approx(3 - 1.5)


# --- PR #1 のレビュー指摘 ----------------------------------------------------

def _random_frame(seed=1, n=200):
    rng = np.random.default_rng(seed)
    rows = [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=i // 4), f"r{i // 4}",
             f"H{i % 30}", f"S{i % 3}", f"D{i % 9}", int(rng.random() < 0.3), 8 + i % 5)
            for i in range(n)]
    return _frame(rows)


def test_asof_check_covers_every_stat_path():
    df = _random_frame()
    feats = build_features(df, pedigree=PED, features=FeatureConfig()).frame
    ok = integrity.check_asof_samples(df, feats, paths=PED.stat_paths, shrinkage=10.0,
                                      exclude_self=True, sample=30)
    assert ok["name"] == "asof_sample" and ok["status"] == "ok", ok
    assert set(ok["detail"]["by_path"]) == {"S", "D"}
    broken = feats.copy()
    broken["st_D_ae_top3"] = broken["st_D_ae_top3"] + 0.01     # 先頭（S）以外の位置の漏れ
    bad = integrity.check_asof_samples(df, broken, paths=PED.stat_paths, shrinkage=10.0,
                                       exclude_self=True, sample=30)
    assert bad["status"] == "fail"
    assert bad["detail"]["by_path"]["S"]["status"] == "ok"


def test_stat_features_are_float32():
    fs = build_features(_random_frame(), pedigree=PED, features=FeatureConfig())
    stats = [c for c in fs.frame.columns if c.startswith("st_")]
    assert stats and all(fs.frame[c].dtype == np.float32 for c in stats)


def test_rank_within_race_tolerates_duplicate_index():
    from keiba_ml.models.base import rank_within_race
    frame = pd.DataFrame({"race_id": ["r1", "r1", "r2"], "umaban": ["01", "02", "01"],
                          "score": [0.1, 0.9, 0.5]}, index=[0, 0, 1])
    assert list(rank_within_race(frame)) == [2, 1, 1]


def test_random_winner_baseline_counts_dead_heat_winners():
    pred, truth = _race([0.9, 0.8, 0.7, 0.1, 0.0], [1, 1, 3, 4, 5])
    row = metrics.per_race(pred, truth).iloc[0]
    assert row["random_winner_in_pred"] == pytest.approx(1 - comb(3, 3) / comb(5, 3))
    pred, truth = _race([0.9, 0.8, 0.7, 0.1, 0.0], [1, 2, 3, 4, 5])
    assert metrics.per_race(pred, truth).iloc[0]["random_winner_in_pred"] == pytest.approx(3 / 5)


def test_age_curve_has_name_column_without_names():
    from keiba_ml.evaluate.growth import age_curve
    df = _random_frame()
    df["age"] = [2 + i % 4 for i in range(len(df))]
    table = age_curve(df, path="S", min_starts=1, names={})
    assert "name" in table.columns
    assert (table["name"] == "").all()


def test_stopped_horse_with_rank_is_not_a_winner_or_relevant():
    """中止・失格（target_top3 が 0）の馬は、着順の欄が 1 でも勝ち・関連度に数えない。"""
    pred, truth = _race([0.9, 0.8, 0.7, 0.1], [1, 1, 2, 3])
    truth.loc[0, "target_top3"] = 0.0          # 01 は失格
    row = metrics.per_race(pred, truth).iloc[0]
    assert row["top1_win"] == 0.0              # 1位に選んだ 01 は勝ちではない
    assert row["random_top1_win"] == pytest.approx(1 / 4)
    assert row["ndcg"] < 1.0                   # 01 を上に置いた予想は理想の並びではない


def test_ranking_target_ignores_stopped_horses():
    from keiba_ml.models.pedigree_catboost import _target
    rows = pd.DataFrame({"result_rank": pd.array([1, 1, 2, 3, None], dtype="Int64"),
                         "target_top3": [0.0, 1.0, 1.0, 1.0, 0.0]})
    assert list(_target(rows, "YetiRank")) == [0.0, 3.0, 2.0, 1.0, 0.0]
    assert list(_target(rows, "Logloss")) == [0.0, 1.0, 1.0, 1.0, 0.0]
