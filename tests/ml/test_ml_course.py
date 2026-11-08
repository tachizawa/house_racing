"""コース適性モデル: コースの区分・特徴量・検算・設定・登録表。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from keiba_ml import registry
from keiba_ml.config import CodeConfig, CourseConfig, DataConfig, config_from_dict, load_config
from keiba_ml.dataset import integrity
from keiba_ml.dataset.races import build_runners
from keiba_ml.features import course
from tests.fixtures import csv_data


def _day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def _run(ketto, day, rank, *, jyo="06", track_cd="17", kyori=2000, n=10, has=True,
         margin=None, sire="S1", baba="1", umaban="05", race=None, c4=None):
    surface = "turf" if int(track_cd) <= 22 else "dirt"
    band = "S" if kyori < 1400 else "M" if kyori < 1800 else "I" if kyori < 2200 else "L"
    return {
        "race_id": race or f"{ketto}{day}{jyo}", "race_date": _day(day),
        "ketto_num": ketto, "kyori_m": kyori, "track_cd": track_cd, "track_kind": surface,
        "distance_band": band, "jyo_cd": jyo, "baba_cd": baba,
        "has_result": has, "result_rank": rank if has else pd.NA, "field_size": n,
        "result_margin_sec": (0.0 if rank == 1 else 0.2 * rank) if has and margin is None
        else margin,
        "target_top3": (1.0 if rank <= 3 else 0.0) if has else np.nan,
        "result_jyuni_4c": "" if c4 is None else str(c4),
        "ped_S": sire, "ped_DS": "DS1", "age": 4, "sex_cd": "1", "umaban": umaban,
    }


def _nakayama_specialist() -> pd.DataFrame:
    """X は中山（右・坂・直線が短い）で好走し、東京（左・坂・直線が長い）で大敗する。"""
    rows = [
        _run("X", "2020-01-05", 1, jyo="06", track_cd="17"),
        _run("X", "2020-02-05", 9, jyo="05", track_cd="11"),
        _run("X", "2020-03-05", 2, jyo="06", track_cd="17"),
        _run("X", "2020-04-05", 10, jyo="05", track_cd="11"),
        _run("Y", "2020-01-05", 2, jyo="06", track_cd="17", race="y1"),
        _run("X", "2020-06-05", 0, jyo="06", track_cd="17", has=False, race="t06"),
        _run("X", "2020-06-05", 0, jyo="05", track_cd="11", has=False, race="t05"),
    ]
    return pd.DataFrame(rows)


def test_track_shape_and_straight_band():
    assert course.track_shape("17") == ("R", "std")
    assert course.track_shape("18") == ("R", "out")
    assert course.track_shape("11") == ("L", "std")
    assert course.track_shape("10") == ("S", "std")
    assert course.track_shape("24") == ("R", "std")
    assert course.track_shape("") == ("", "")
    assert course.straight_band("05", "turf", "std", "L") == "long"
    assert course.straight_band("06", "turf", "std", "R") == "short"
    assert course.straight_band("09", "turf", "out", "R") == "long"
    assert course.straight_band("09", "turf", "std", "R") == "mid"   # 内回りとみなす
    assert course.straight_band("04", "turf", "std", "S") == "straight"
    assert course.going_class("4") == "heavy" and course.going_class(None) == ""


def test_specialist_looks_better_at_the_course_it_suits():
    df = _nakayama_specialist()
    frame = course.build_features(df, CourseConfig()).frame
    good, bad = frame.index[df["race_id"] == "t06"][0], frame.index[df["race_id"] == "t05"][0]
    for col in ("own_rel_pf_venue", "own_rel_pm_venue", "own_rel_pf_course",
                "own_rel_pf_course_dist", "own_rel_pf_dir", "own_rel_pf_straight"):
        assert frame.at[good, col] > 0 > frame.at[bad, col], col
    # 坂は中山も東京も「あり」なので差が出ない
    assert frame.at[good, "own_rel_pf_slope"] == pytest.approx(frame.at[bad, "own_rel_pf_slope"])
    assert frame.at[good, "own_log_runs_venue"] == pytest.approx(np.log1p(2))
    assert "own_log_runs_surface" not in frame.columns


def _two_horses_differing_only_in_strength() -> pd.DataFrame:
    """A と B は、どのコースでも B が同じだけ悪い（負かした割合も、1km あたりの着差も）。"""
    rows = []
    for ketto, shift_rank, shift_pm, sire in (("A", 0, 0.0, "SA"), ("B", 3, 0.5, "SB")):
        for day, jyo, tcd, rank, pm, baba in (
                ("2020-01-05", "06", "17", 1, 0.0, "1"), ("2020-02-05", "05", "11", 3, 0.3, "3"),
                ("2020-03-05", "09", "18", 2, 0.1, "1"), ("2020-04-05", "06", "24", 2, 0.2, "2")):
            margin = (pm + shift_pm) * 2.0
            rows.append(_run(ketto, day, rank + shift_rank, jyo=jyo, track_cd=tcd, margin=margin,
                             sire=sire, baba=baba, race=f"{ketto}{day}"))
        rows.append(_run(ketto, "2020-06-05", 0, has=False, sire=sire, race=f"{ketto}t"))
    return pd.DataFrame(rows)


def test_features_do_not_carry_the_horses_strength():
    """強さだけが違う2頭は、本人の特徴量が完全に一致する（強さは別のモデルの担当）。"""
    df = _two_horses_differing_only_in_strength()
    frame = course.build_features(df, CourseConfig()).frame
    own = [c for c in frame.columns if c.startswith("own_")]
    assert len(own) >= 20
    a = frame.loc[df.index[df["race_id"] == "At"][0], own].astype(float)
    b = frame.loc[df.index[df["race_id"] == "Bt"][0], own].astype(float)
    pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-5)


def test_first_runner_has_no_own_history_but_has_pedigree():
    df = pd.DataFrame([_run("Y", "2020-01-05", 1, race="y1"),
                       _run("Z", "2020-03-05", 0, has=False, race="z")])
    frame = course.build_features(df, CourseConfig()).frame
    z = frame.index[df["race_id"] == "z"][0]
    assert frame.at[z, "own_log_runs_venue"] == 0.0
    assert np.isnan(frame.at[z, "own_rel_pf_venue"])
    assert frame.at[z, "ped_S_rel_pf_venue"] == pytest.approx(0.0)


def test_draw_bias_counts_earlier_days_only():
    rows = []
    for i, day in enumerate(("2020-01-05", "2020-01-12")):
        for u in range(1, 10):
            rows.append(_run(f"H{i}{u}", day, u, n=9, umaban=f"{u:02d}", race=f"r{i}"))
    rows.append(_run("Q", "2020-01-12", 0, n=9, umaban="01", has=False, race="r1"))
    df = pd.DataFrame(rows)
    config = CourseConfig(bias_shrinkage=1.0)
    frame = course.build_features(df, config).frame
    q = frame.index[df["ketto_num"] == "Q"][0]
    # 前の日（r0）だけ: 内（馬番1〜3）は3頭とも3着以内、期待は 3 × 3/9 = 1
    assert frame.at[q, "bias_draw_ae"] == pytest.approx((3 + 1) / (1 + 1))
    first = frame.index[df["race_id"] == "r0"][0]
    assert frame.at[first, "bias_draw_ae"] == pytest.approx(1.0)
    check = integrity.check_course_asof_sample(df, frame, config=config, sample=19)
    assert check["status"] == "ok", check


def test_style_features_are_optional():
    df = _nakayama_specialist()
    df["result_jyuni_4c"] = ["1", "8", "2", "9", "1", "", ""]
    off = course.build_features(df, CourseConfig()).frame
    assert "own_style" not in off.columns
    on = course.build_features(df, CourseConfig(style=True, style_strength_slope=0.0)).frame
    t = on.index[df["race_id"] == "t06"][0]
    assert on.at[t, "own_style"] == pytest.approx(np.mean([0, 7 / 9, 1 / 9, 8 / 9]))
    # 補正あり: X の pf は 1, 1/9, 8/9, 0（着順 1, 9, 2, 10 / 10頭）
    adj = course.build_features(df, CourseConfig(style=True, style_strength_slope=0.5)).frame
    pos, pf = np.array([0, 7 / 9, 1 / 9, 8 / 9]), np.array([1, 1 / 9, 8 / 9, 0])
    assert adj.at[t, "own_style"] == pytest.approx(np.mean(pos + 0.5 * (pf - 0.5)))


def test_features_do_not_change_when_a_later_result_changes():
    df = _nakayama_specialist()
    base = course.build_features(df, CourseConfig()).frame
    changed = df.copy()
    last = changed.index[(changed["ketto_num"] == "X") & (changed["race_date"] == _day("2020-04-05"))]
    changed.loc[last, ["result_rank", "result_margin_sec", "target_top3"]] = [1, 0.0, 1.0]
    after = course.build_features(changed, CourseConfig()).frame
    earlier = df.index[df["race_date"] <= _day("2020-04-05")]
    pd.testing.assert_frame_equal(base.loc[earlier], after.loc[earlier])
    later = df.index[df["race_date"] > _day("2020-04-05")]
    assert not base.loc[later].equals(after.loc[later])


def test_feature_names_pass_the_forbidden_column_check():
    features = course.build_features(_nakayama_specialist(), CourseConfig())
    assert integrity.check_feature_columns(features.columns)["status"] == "ok"
    assert set(features.cat_features) == {"track_kind", "distance_band", "jyo_cd", "baba_cd",
                                          "sex_cd"}


def test_course_asof_check_detects_a_tampered_feature():
    df = _two_horses_differing_only_in_strength()
    features = course.build_features(df, CourseConfig())
    ok = integrity.check_course_asof_sample(df, features.frame, config=CourseConfig(), sample=10)
    assert ok["status"] == "ok", ok
    tampered = features.frame.copy()
    tampered["own_rel_pf_venue"] = tampered["own_rel_pf_venue"] + 0.1
    bad = integrity.check_course_asof_sample(df, tampered, config=CourseConfig(), sample=10)
    assert bad["status"] == "fail"


def test_course_baselines_and_describe():
    df = _nakayama_specialist()
    features = course.build_features(df, CourseConfig())
    rows = df[~df["has_result"]]
    spec = registry.get("course")
    preds = spec.baselines(features, rows, "live")
    assert set(preds) == {"course_rel_venue", "course_draw"}
    view = spec.describe(rows, features, {"S1": "父馬"})
    assert view["父"].tolist() == ["父馬", "父馬"]
    assert view["この競馬場の出走"].tolist() == [2, 2]


def test_dataset_keeps_the_track_code():
    rid = "2026032106011101"
    races = pd.DataFrame([{k: str(v) for k, v in
                           csv_data.make_race(race_id=rid, data_kubun="7").items()}])
    runners = pd.DataFrame([{k: str(v) for k, v in csv_data.make_race_horse(**r).items()} for r in (
        {"race_id": rid, "umaban": "01", "ketto_num": "2020100001", "result_kakutei_jyuni": "1",
         "result_margin_sec": "0.0"},
    )])
    df = build_runners(races, runners, data=DataConfig(), codes_config=CodeConfig())
    assert df["track_cd"].iloc[0] == str(races["track_cd"].iloc[0]).strip()
    assert registry.get("course").missing_columns(df) == []


def test_course_config_shares_the_dataset_and_rejects_bad_values():
    from keiba_ml.dataset.build import manifest_config
    assert manifest_config(load_config(model="course")) == \
        manifest_config(load_config(model="pedigree"))
    for raw in ({"course": {"own_shrinkage": 0.0}}, {"course": {"qualities": ["speed"]}},
                {"course": {"front_threshold": 1.5}}, {"course": {"stats_history": "x"}},
                {"course": {"pedigree_paths": ["SSSSSS"]}}):
        with pytest.raises(ValueError):
            config_from_dict(raw)


def test_extra_groups_trip_and_venue_dband():
    assert course.trip_class("05", "1") == "home"
    assert course.trip_class("09", "1") == "away"
    assert course.trip_class("01", "2") == "hokkaido"
    assert course.trip_class("05", "3") == ""
    df = _nakayama_specialist()
    df["tozai_cd"] = "2"          # 栗東所属: 中山も東京も遠征
    config = CourseConfig(extra_groups=("venue_dband", "trip"))
    frame = course.build_features(df, config).frame
    t = frame.index[df["race_id"] == "t06"][0]
    assert frame.at[t, "own_rel_pf_venue_dband"] > 0
    assert frame.at[t, "own_log_runs_trip"] == pytest.approx(np.log1p(4))
    assert frame.at[t, "own_rel_pf_trip"] == pytest.approx(0.0)   # 全部遠征 = 普段どおり
    check = integrity.check_course_asof_sample(df, frame, config=config, sample=7)
    assert check["status"] == "ok", check
    with pytest.raises(ValueError):
        config_from_dict({"course": {"extra_groups": ["moon"]}})


def test_race_relative_is_the_difference_from_the_race_mean():
    df = _two_horses_differing_only_in_strength()
    df.loc[df["race_id"] == "Bt", "race_id"] = "At"          # 2頭を同じレースに
    frame = course.build_features(df, CourseConfig(race_relative=True)).frame
    rows = frame.index[df["race_id"] == "At"]
    # 2頭は本人の特徴量が一致するので、差は 0
    assert frame.loc[rows, "rr_own_rel_pf_venue"].abs().max() == pytest.approx(0.0, abs=1e-6)
    assert "rr_ped_S_rel_pf_venue" in frame.columns
    assert integrity.check_feature_columns(
        course.build_features(df, CourseConfig(race_relative=True)).columns)["status"] == "ok"


def test_going_bias_and_pedigree_extra_groups():
    rows = []
    for i, (day, baba) in enumerate((("2020-01-05", "1"), ("2020-01-12", "3"),
                                     ("2020-01-19", "3"))):
        for u in range(1, 10):
            rows.append(_run(f"H{i}{u}", day, u if baba == "1" else 10 - u, n=9,
                             umaban=f"{u:02d}", baba=baba, race=f"r{i}"))
    df = pd.DataFrame(rows)
    config = CourseConfig(bias_by_going=True, bias_shrinkage=1.0,
                          pedigree_extra_groups=("course", "course_dist"))
    frame = course.build_features(df, config).frame
    q = frame.index[(df["race_id"] == "r2") & (df["umaban"] == "09")][0]
    # 重馬場（r1）だけ: 外（7〜9番）は3頭とも3着以内。良（r0）では外は全敗
    assert frame.at[q, "bias_draw_going_ae"] == pytest.approx((3 + 1) / (1 + 1))
    assert frame.at[q, "bias_draw_ae"] == pytest.approx((3 + 1) / (2 + 1))
    assert {"ped_S_rel_pf_course", "ped_S_rel_pf_course_dist"} <= set(frame.columns)
    with pytest.raises(ValueError):
        config_from_dict({"course": {"pedigree_extra_groups": ["moon"]}})


def test_decay_weights_recent_runs_more():
    """中山で古くは大敗・最近は好走した馬は、減衰ありの方が中山の相対的な走りが高い。"""
    rows = [
        _run("X", "2018-01-05", 10, jyo="06"), _run("X", "2018-02-05", 1, jyo="05", track_cd="11"),
        _run("X", "2020-01-05", 1, jyo="06"), _run("X", "2020-02-05", 9, jyo="05", track_cd="11"),
        _run("X", "2020-03-05", 0, jyo="06", has=False, race="t"),
    ]
    df = pd.DataFrame(rows)
    plain = course.build_features(df, CourseConfig()).frame
    decay = course.build_features(df, CourseConfig(decay_days=180.0)).frame
    t = plain.index[df["race_id"] == "t"][0]
    assert decay.at[t, "own_rel_pf_venue"] > plain.at[t, "own_rel_pf_venue"]
    assert decay.at[t, "own_log_runs_venue"] == plain.at[t, "own_log_runs_venue"]
    # 強さだけが違う2頭は減衰ありでも一致する
    df2 = _two_horses_differing_only_in_strength()
    frame = course.build_features(df2, CourseConfig(decay_days=180.0)).frame
    own = [c for c in frame.columns if c.startswith("own_")]
    a = frame.loc[df2.index[df2["race_id"] == "At"][0], own].astype(float)
    b = frame.loc[df2.index[df2["race_id"] == "Bt"][0], own].astype(float)
    pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-5)
    check = integrity.check_course_asof_sample(df, decay, config=CourseConfig(decay_days=180.0),
                                               sample=5)
    assert check["status"] == "ok", check


def test_kick_features_use_the_race_rank_of_the_last_3f():
    rows = []
    for i, day in enumerate(("2020-01-05", "2020-01-12")):
        for u in range(1, 5):
            row = _run(f"H{u}", day, u, n=4, umaban=f"{u:02d}", race=f"r{i}")
            row["result_last_3f_sec"] = str(33.0 + u)       # 1着が最速
            rows.append(row)
    rows.append({**_run("H1", "2020-02-05", 0, n=4, has=False, race="t"),
                 "result_last_3f_sec": ""})
    df = pd.DataFrame(rows)
    frame = course.build_features(df, CourseConfig(kick=True, kick_strength_slope=0.0)).frame
    t = frame.index[df["race_id"] == "t"][0]
    assert frame.at[t, "own_kick"] == pytest.approx(0.0)
    # コースは1つだけなので、利き方の比は 1
    assert frame.at[t, "bias_kick_rel"] == pytest.approx(1.0)
    adj = course.build_features(df, CourseConfig(kick=True)).frame
    assert adj.at[t, "own_kick"] == pytest.approx(0.755 * 0.5)


def test_use_going_false_drops_the_going_columns():
    df = _nakayama_specialist()
    config = CourseConfig(use_going=False, context=("track_kind", "kyori_m", "jyo_cd"))
    frame = course.build_features(df, config).frame
    assert not [c for c in frame.columns if "going" in c or c == "baba_cd"]
    assert integrity.check_course_asof_sample(df, frame, config=config, sample=7)["status"] == "ok"


def test_meeting_bias_uses_only_the_recent_window_before_the_day():
    rows = []
    # 30日前: 内（1〜3番）が全部3着以内。5日前・3日前: 外（7〜9番）が全部3着以内
    for i, (day, outside) in enumerate((("2020-01-01", False), ("2020-01-26", True),
                                        ("2020-01-28", True), ("2020-01-31", True))):
        for u in range(1, 10):
            rank = (10 - u) if outside else u
            rows.append(_run(f"H{i}{u}", day, rank, n=9, umaban=f"{u:02d}", race=f"r{i}"))
    df = pd.DataFrame(rows)
    frame = course.build_features(df, CourseConfig(meeting_days=14, meeting_shrinkage=1.0)).frame
    q = frame.index[(df["race_id"] == "r3") & (df["umaban"] == "09")][0]
    # 直近14日（r1, r2）だけ: 外は 6頭とも3着以内、期待 6 × 3/9 = 2。同じ日（r3）は入らない
    assert frame.at[q, "meet_draw_ae"] == pytest.approx((6 + 1) / (2 + 1))
    inner = frame.index[(df["race_id"] == "r3") & (df["umaban"] == "01")][0]
    assert frame.at[inner, "meet_draw_ae"] == pytest.approx(1 / 3)
    first = frame.index[(df["race_id"] == "r1") & (df["umaban"] == "01")][0]
    assert frame.at[first, "meet_draw_ae"] == pytest.approx(1.0)   # 25日前の r0 は窓の外


def test_meeting_bias_passes_the_asof_check_and_detects_tampering():
    rows = []
    for i, day in enumerate(("2020-01-05", "2020-01-12", "2020-01-19", "2020-03-01")):
        for u in range(1, 8):
            rows.append(_run(f"H{u}", day, (u + i) % 7 + 1, n=7, umaban=f"{u:02d}", race=f"r{i}"))
    df = pd.DataFrame(rows)
    config = CourseConfig(meeting_days=21)
    frame = course.build_features(df, config).frame
    assert integrity.check_course_asof_sample(df, frame, config=config, sample=28)["status"] == "ok"
    tampered = frame.copy()
    tampered["meet_draw_ae"] = tampered["meet_draw_ae"] + 0.1
    assert integrity.check_course_asof_sample(df, tampered, config=config,
                                              sample=28)["status"] == "fail"


def test_race_relative_subtracts_a_non_constant_race_mean():
    df = _nakayama_specialist()
    df.loc[df["race_id"] == "y1", "race_id"] = "X2020-01-0506"     # Y を X の1走目と同じレースに
    df = pd.concat([df, pd.DataFrame([
        _run("Y", "2020-06-05", 0, jyo="06", track_cd="17", has=False, race="t06"),
    ])], ignore_index=True)
    config = CourseConfig(race_relative=True)
    frame = course.build_features(df, config).frame
    rows = frame.index[df["race_id"] == "t06"]
    assert len(rows) == 2
    for src in ("own_rel_pf_venue", "own_rel_pf_course"):
        values = frame.loc[rows, src].astype(float)
        assert values.nunique() == 2                            # 2頭の値は違う
        expected = values - values.mean()
        assert np.allclose(frame.loc[rows, f"rr_{src}"].astype(float), expected, atol=1e-6)
        assert frame.loc[rows, f"rr_{src}"].abs().min() > 1e-3   # 差は 0 でない
    assert integrity.check_course_asof_sample(df, frame, config=config, sample=8)["status"] == "ok"


def test_meeting_window_includes_its_first_day():
    rows = []
    # 前日（1日前）: 外が全部3着以内。meeting_days=1 なら前日だけが窓に入る
    for i, day in enumerate(("2020-01-04", "2020-01-05")):
        for u in range(1, 10):
            rows.append(_run(f"H{i}{u}", day, 10 - u, n=9, umaban=f"{u:02d}", race=f"r{i}"))
    df = pd.DataFrame(rows)
    config = CourseConfig(meeting_days=1, meeting_shrinkage=1.0)
    frame = course.build_features(df, config).frame
    q = frame.index[(df["race_id"] == "r1") & (df["umaban"] == "09")][0]
    assert frame.at[q, "meet_draw_ae"] == pytest.approx((3 + 1) / (1 + 1))
    assert integrity.check_course_asof_sample(df, frame, config=config, sample=18)["status"] == "ok"


def _many_runs() -> pd.DataFrame:
    rows = []
    days = ("2020-01-05", "2020-01-12", "2020-02-02", "2020-02-09", "2020-03-01")
    for i, day in enumerate(days):
        jyo, tcd = ("06", "17") if i % 2 == 0 else ("05", "11")
        for u in range(1, 8):
            rows.append(_run(f"H{u}", day, (u + 2 * i) % 7 + 1, n=7, umaban=f"{u:02d}",
                             race=f"r{i}", jyo=jyo, track_cd=tcd, baba="1" if i < 3 else "3",
                             c4=(u + i) % 7 + 1, sire=f"S{u % 3}"))
    df = pd.DataFrame(rows)
    df["result_last_3f_sec"] = [str(34.0 + (k * 7) % 5 * 0.3) for k in range(len(df))]
    df["tozai_cd"] = ["1" if k % 2 else "2" for k in range(len(df))]
    return df


def test_asof_check_covers_every_feature_column_and_detects_tampering():
    df = _many_runs()
    config = CourseConfig(extra_groups=("venue_dband", "trip"), decay_days=60.0, style=True,
                          decay_style=True, front_relative=True, kick=True, bias_by_going=True,
                          pedigree_extra_groups=("course", "venue_dband"), meeting_days=14,
                          race_relative=True)
    frame = course.build_features(df, config).frame
    check = integrity.check_course_asof_sample(df, frame, config=config, sample=len(df))
    assert check["status"] == "ok", check
    assert check["detail"]["unchecked"] == []
    for col in ("own_rel_pm_dir", "own_log_runs_trip", "ped_DS_rel_pf_venue_dband", "own_style",
                "bias_front_going_ae", "style_fit_rel", "kick_fit", "meet_front_rel",
                "rr_style_fit"):
        tampered = frame.copy()
        tampered[col] = tampered[col] + 0.1
        bad = integrity.check_course_asof_sample(df, tampered, config=config, sample=len(df))
        assert bad["status"] == "fail", col
    extra = frame.assign(mystery=0.0)
    bad = integrity.check_course_asof_sample(df, extra, config=config, sample=5)
    assert bad["status"] == "fail" and bad["detail"]["unchecked"] == ["mystery"]


def test_baseline_works_with_pm_only_qualities():
    df = _nakayama_specialist()
    features = course.build_features(df, CourseConfig(qualities=("pm",)))
    rows = df[~df["has_result"]]
    preds = registry.get("course").baselines(features, rows, "live")
    assert set(preds) == {"course_rel_venue", "course_draw"}


def test_course_config_rejects_unknown_context_and_going_contradictions():
    for raw in ({"course": {"context": ["track_kind", "kyori"]}},
                {"course": {"use_going": False}},                       # 既定の context に baba_cd
                {"course": {"use_going": False, "bias_by_going": True,
                            "context": ["track_kind", "kyori_m"]}}):
        with pytest.raises(ValueError):
            config_from_dict(raw)
    config_from_dict({"course": {"use_going": False, "context": ["track_kind", "month"]}})
