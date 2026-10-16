from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from keiba.features import course
from keiba.models import RaceConditions, RaceInput

JST = timezone(timedelta(hours=9))


def make_race_input(**kw) -> RaceInput:
    base = dict(
        date=date(2026, 8, 1), jyo="新潟", race_num=11, track_kind="turf",
        kyori_m=2000, entry_count=0,
        prediction_cutoff=datetime(2026, 7, 31, 20, 0, tzinfo=JST),
        horses=(), jyo_cd="04",
    )
    base.update(kw)
    return RaceInput(**base)


def test_decode_track_cd_right_outer_turf():
    code = course.decode_track_cd("18")
    assert code == course.TrackCode(surface="turf", direction="right", label="芝・右外")


def test_decode_track_cd_left_turf_has_no_layout_claim():
    # 11 は「芝・左」。内回りとは書かれていない（新潟内回りもこの値）ので、
    # 解読結果に「内回り」を作らない。内外の事実はコース基本情報表の仕事。
    code = course.decode_track_cd("11")
    assert code.label == "芝・左"
    assert code.direction == "left"


def test_decode_track_cd_straight():
    assert course.decode_track_cd("10").direction == "straight"
    assert course.decode_track_cd("29") == course.TrackCode("dirt", "straight", "ダート・直")


def test_decode_track_cd_unknown_or_jump_is_none():
    assert course.decode_track_cd("00") is None
    assert course.decode_track_cd("51") is None
    assert course.decode_track_cd("  ") is None


def test_identify_with_conditions():
    result = course.identify(
        make_race_input(),
        RaceConditions(class_level=5, track_cd="12", course_kubun_cd="A"))
    assert result.identified is True
    assert result.track_cd == "12"
    assert result.course_kubun_cd == "A"
    assert result.reason is None


def test_identify_without_race_id_is_not_identified():
    result = course.identify(make_race_input(), None)
    assert result.identified is False
    assert result.track is None
    assert "トラックコード" in result.reason
    # 引いていないのに「データ版に無い」とは言わない。
    assert "レースIDが無い" in result.reason
    assert "データ版に無い" not in result.reason


def test_identify_race_not_in_data_says_so():
    result = course.identify(make_race_input(race_id="2026080104020311"), None)
    assert result.identified is False
    assert "データ版に無い" in result.reason


def test_identify_rejects_surface_mismatch():
    # 入力は芝なのにトラックコードはダート。どちらかが誤っているので推測しない。
    result = course.identify(
        make_race_input(track_kind="turf"),
        RaceConditions(class_level=5, track_cd="23", course_kubun_cd=""))
    assert result.identified is False
    assert "食い違" in result.reason


def test_identify_blank_course_kubun_is_none():
    result = course.identify(
        make_race_input(track_kind="dirt"),
        RaceConditions(class_level=5, track_cd="23", course_kubun_cd="  "))
    assert result.identified is True
    assert result.course_kubun_cd is None
