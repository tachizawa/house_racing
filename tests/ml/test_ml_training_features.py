"""調教の特徴量（features/training.py）: レースの日より前の調教だけを使うこと。"""

import numpy as np
import pandas as pd

from keiba_ml import registry
from keiba_ml.dataset import integrity
from keiba_ml.features import training


def _workouts(rows):
    """(馬, 日, z) の行から load_workouts と同じ列の表を作る。"""
    d = pd.DataFrame(rows, columns=["ketto_num", "day", "z"])
    d["day"] = pd.to_datetime(d["day"])
    d["kind"] = np.int8(0)
    for col in ("time_4f", "time_3f", "lap_1", "zl1", "accel", "za"):
        d[col] = np.float32(d["z"] + 50)
    d["z"] = d["z"].astype("float32")
    return d.sort_values(["ketto_num", "day"]).reset_index(drop=True)


def _runners(rows):
    return pd.DataFrame({
        "race_id": [r[0] for r in rows], "race_date": pd.to_datetime([r[1] for r in rows]),
        "ketto_num": [r[2] for r in rows], "chokyosi_code": [r[3] for r in rows],
        "has_result": [True] * len(rows), "target_top3": [0.0] * len(rows),
    })


RUNNERS = _runners([("R1", "2024-05-05", "A", "T1"), ("R1", "2024-05-05", "B", "T1"),
                    ("R1", "2024-05-05", "C", "T2"), ("R1", "2024-05-05", "D", "T2")])
# A の 04-25（10日前）は 05-01（4日前）より速い。追い切りは「7日以内で最速」なので 05-01 を選ぶ
BASE = [("A", "2024-05-01", -1.0), ("A", "2024-04-25", -1.5), ("B", "2024-05-02", -0.2),
        ("C", "2024-05-03", 0.3), ("A", "2024-03-01", -0.5)]


def test_only_workouts_before_race_day_are_used():
    a = training.training_features(RUNNERS, _workouts(BASE))
    # 当日・翌日の調教を足しても値は変わらない
    leak = [("A", "2024-05-05", -3.9), ("B", "2024-05-05", -3.8), ("A", "2024-05-06", -3.7)]
    b = training.training_features(RUNNERS, _workouts(BASE + leak))
    pd.testing.assert_frame_equal(a, b)


def test_final_workout_is_fastest_within_seven_days():
    out = training.training_features(RUNNERS, _workouts(BASE))
    assert out.loc[0, "t_oi_z"] == -1.0
    assert out.loc[0, "t_oi_days"] == 4
    assert out.loc[0, "t_n_14"] == 2
    assert out.loc[0, "t_best_prior_z"] == -0.5
    assert out.loc[0, "t_oi_vs_self"] == -0.5


def test_unknown_horse_is_missing_not_zero():
    out = training.training_features(RUNNERS, _workouts(BASE))
    assert np.isnan(out.loc[3, "t_n_14"])
    assert np.isnan(out.loc[3, "t_oi_z"])


def test_known_horse_without_recent_workout_counts_zero():
    # 調教の記録はあるが 60 日より前だけの馬は「本数 0」
    rows = BASE + [("D", "2024-01-01", 0.1)]
    out = training.training_features(RUNNERS, _workouts(rows))
    assert out.loc[3, "t_n_14"] == 0
    assert np.isnan(out.loc[3, "t_oi_z"])


def test_trainer_mean_uses_only_earlier_days():
    rows = BASE + [("E", "2024-04-30", -2.0), ("F", "2024-05-08", -2.5)]
    runners = pd.concat([RUNNERS, _runners([("R0", "2024-05-04", "E", "T1"),
                                            ("R2", "2024-05-12", "F", "T1")])],
                        ignore_index=True)
    out = training.training_features(runners, _workouts(rows))
    # 同じ日の A と B の値は混ざらない。05-04 の E だけが基準で、−1.0 − (−2.0)
    assert out.loc[0, "t_oi_vs_trainer"] == 1.0
    # 最初の日の馬（E）には基準が無い
    assert np.isnan(out.loc[4, "t_oi_vs_trainer"])


def test_day_z_needs_enough_workouts_and_lower_is_faster():
    n = training.MIN_DAY_GROUP
    slope = pd.DataFrame({
        "tresen_kubun": ["0"] * n, "chokyo_date": ["2024-05-01"] * n,
        "ketto_num": [str(i) for i in range(n)],
        "time_4f": [str(50 + i) for i in range(n)], "time_3f": [str(38 + i) for i in range(n)],
        "lap_2": ["13.0"] * n, "lap_1": [str(12 + i * 0.1) for i in range(n)],
        "data_kubun": ["1"] * n})
    wood = pd.DataFrame(columns=["tresen_kubun", "chokyo_date", "ketto_num", "course", "time_5f",
                                 "time_4f", "time_3f", "lap_2", "lap_1", "data_kubun"])
    table = training.normalize_workouts(slope, wood)
    assert len(table) == n
    fastest = table.loc[table["ketto_num"] == "0", "z"].item()
    assert fastest == table["z"].min() and fastest < 0
    assert training.normalize_workouts(slope.head(n - 1), wood).empty


def test_truncation_check_catches_same_day_workouts(monkeypatch):
    workouts = _workouts(BASE + [("A", "2024-05-05", -3.9), ("B", "2024-05-05", -3.8)])
    honest = training.training_features(RUNNERS, workouts)
    columns = ["t_oi_z", "t_n_7"]
    ok = integrity.check_truncation_sample(
        RUNNERS, honest, name="t", columns=columns,
        builder=lambda part: registry._training_rebuild(part, workouts))
    assert ok["status"] == "ok", ok

    # 当日の調教まで読む「漏れた」作り方（1日後のレースとして数える）。作り直しにも同じ関数を使うので、
    # 作り直しに渡す調教をその日より前に絞らないと、同じように漏れて一致してしまう
    real = training.training_features

    def leaky(df, w):
        return real(df.assign(race_date=df["race_date"] + pd.Timedelta(days=1)), w)

    monkeypatch.setattr(training, "training_features", leaky)
    leaked = leaky(RUNNERS, workouts)
    bad = integrity.check_truncation_sample(
        RUNNERS, leaked, name="t", columns=columns,
        builder=lambda part: registry._training_rebuild(part, workouts))
    assert bad["status"] == "fail", bad
