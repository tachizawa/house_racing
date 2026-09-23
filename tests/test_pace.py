"""keiba.features.pace のテスト。

ペース区分の閾値は「学習期間だけから計算する」（as_of ゲート）ことと、
「層が足りなければ粗い層へ落ちる」ことが仕様の核心（spec §8.2）。
どちらも間違えると、間違いは例外ではなく静かに数字がずれる形で出るので、
境界値を具体的な数値で確かめる。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from keiba.features import pace
from keiba.models import RaceInput, RaceRow

JST = timezone(timedelta(hours=9))


def jst(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=JST)


AS_OF = jst(2026, 9, 1)
BEFORE = jst(2026, 8, 1)  # as_of より確実に前


def make_race_row(
    *,
    race_id="R",
    hasso_at=BEFORE,
    jyo_cd="06",
    kyori_m=2000,
    track_kind="turf",
    first_3f_sec=35.0,
) -> RaceRow:
    return RaceRow(
        race_id=race_id,
        hasso_at=hasso_at,
        jyo_cd=jyo_cd,
        kyori_m=kyori_m,
        track_cd="17" if track_kind == "turf" else ("24" if track_kind == "dirt" else "51"),
        track_kind=track_kind,
        baba_cd="1",
        class_level=4,
        grade_cd=" ",
        jyoken_cd_youngest="016",
        toroku_tosu=12,
        syusso_tosu=12,
        nyusen_tosu=12,
        first_3f_sec=first_3f_sec,
        race_name="テストレース",
    )


def make_target(*, track_kind="turf", kyori_m=2000, jyo_cd="06") -> RaceInput:
    return RaceInput(
        date=AS_OF.date(),
        jyo="中山",
        race_num=11,
        track_kind=track_kind,
        kyori_m=kyori_m,
        entry_count=1,
        prediction_cutoff=AS_OF,
        horses=(),
        jyo_cd=jyo_cd,
    )


def races_with_times(times, *, hasso_at=BEFORE, **kw) -> list[RaceRow]:
    return [make_race_row(race_id=f"R{i}", hasso_at=hasso_at, first_3f_sec=t, **kw)
            for i, t in enumerate(times)]


# --- distance_band ------------------------------------------------------------

def test_distance_band_groups_within_400m():
    """±200m の近傍は同じバンドに入り、400m 離れれば別バンドになる。"""
    assert pace.distance_band(1800) == pace.distance_band(2000)
    assert pace.distance_band(2000) == pace.distance_band(2100)
    assert pace.distance_band(1600) != pace.distance_band(1800)


# --- classify_pace --------------------------------------------------------------

DEFINITION = {"fast_max": 33.0, "standard_max": 36.0}


def test_smaller_first_3f_is_fast():
    assert pace.classify_pace(32.0, DEFINITION) == "fast"
    assert pace.classify_pace(37.0, DEFINITION) == "slow"


def test_boundary_goes_to_faster_band():
    """閾値ちょうどの値は速い側（<=）に寄せる。実行するたび同じ区分になるように。"""
    assert pace.classify_pace(33.0, DEFINITION) == "fast"
    assert pace.classify_pace(36.0, DEFINITION) == "standard"


def test_classify_returns_none_for_missing_value():
    assert pace.classify_pace(None, DEFINITION) is None


# --- fit_pace_definition: 三分位 -------------------------------------------------

def test_thresholds_split_at_tertiles():
    """30.0〜38.9 の90レースなら、閾値は概ね33.0と36.0になる（下位1/3がfast）。"""
    times = [30.0 + i * 0.1 for i in range(90)]
    races = races_with_times(times)

    definition = pace.fit_pace_definition(races, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert definition["fast_max"] == pytest.approx(33.0, abs=0.5)
    assert definition["standard_max"] == pytest.approx(36.0, abs=0.5)


def test_definition_records_stratum_and_count():
    times = [30.0 + i * 0.1 for i in range(30)]
    races = races_with_times(times)

    definition = pace.fit_pace_definition(races, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert definition["race_count"] == 30
    assert definition["stratum"] == {"track_kind": "turf", "distance_band": pace.distance_band(2000),
                                      "jyo_cd": "06"}
    assert "note" in definition and definition["note"]


# --- 層の段階的な粗さ ------------------------------------------------------------

def test_uses_finest_stratum_when_enough_races():
    """同じ track_kind × 距離帯 × 競馬場 が30レース以上あれば、その層を使う。"""
    fine = races_with_times([30.0 + i * 0.1 for i in range(30)], jyo_cd="06")
    # 別会場のノイズ。この層が使われるなら、閾値がこちらに引きずられないことも確認する。
    other_venue = races_with_times([50.0] * 30, jyo_cd="07")

    definition = pace.fit_pace_definition(fine + other_venue, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert definition["stratum"] == {"track_kind": "turf", "distance_band": pace.distance_band(2000),
                                      "jyo_cd": "06"}
    assert definition["race_count"] == 30
    assert definition["fast_max"] < 40.0  # 50.0 のノイズに引きずられていない


def test_falls_back_when_stratum_too_small():
    """競馬場込みの層が29レースしかなければ、競馬場を落とした層に落ちる。"""
    fine = races_with_times([30.0 + i * 0.1 for i in range(29)], jyo_cd="06")
    # 同じ track_kind × 距離帯だが別会場。これを足して初めて30レースに届く。
    other_venue = races_with_times([31.0], jyo_cd="07")

    definition = pace.fit_pace_definition(fine + other_venue, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert definition["stratum"] == {"track_kind": "turf", "distance_band": pace.distance_band(2000)}
    assert definition["race_count"] == 30


def test_falls_back_twice_when_still_too_small():
    """距離帯込みの層も足りなければ track_kind だけの層に落ちる。"""
    fine = races_with_times([30.0 + i * 0.1 for i in range(10)], jyo_cd="06", kyori_m=2000)
    # 同じ track_kind だが距離帯が違う（distance_band が変わる）。
    other_distance = races_with_times([31.0 + i * 0.1 for i in range(20)], jyo_cd="09", kyori_m=1200)

    definition = pace.fit_pace_definition(fine + other_distance, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert definition["stratum"] == {"track_kind": "turf"}
    assert definition["race_count"] == 30


def test_returns_none_when_no_stratum_has_enough():
    """どの層も30レースに届かなければ None（推測で閾値を作らない）。"""
    races = races_with_times([30.0 + i for i in range(5)])

    definition = pace.fit_pace_definition(races, target=make_target(), as_of=AS_OF)

    assert definition is None


# --- 使えるレースの絞り込み ------------------------------------------------------

def test_excludes_jump_races():
    """障害レースは is_flat が False なので除外される（first_3f_sec があっても）。"""
    turf = races_with_times([30.0 + i * 0.1 for i in range(30)], track_kind="turf")
    jump = races_with_times([20.0] * 30, track_kind="jump")  # 混ざれば閾値が大きく崩れる

    definition = pace.fit_pace_definition(turf + jump, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert definition["race_count"] == 30  # 障害の30件は数えない
    assert definition["stratum"]["track_kind"] == "turf"
    # 障害の 20.0 秒が混ざっていれば下位1/3の境界が押し下げられる。数値で確認する。
    assert definition["fast_max"] > 25.0


def test_excludes_races_without_first_3f():
    """first_3f_sec が None の行は使えるレースに数えない。"""
    with_time = races_with_times([30.0 + i * 0.1 for i in range(30)])
    without_time = races_with_times([None] * 30)

    definition = pace.fit_pace_definition(with_time + without_time, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert definition["race_count"] == 30


# --- as_of ゲート -----------------------------------------------------------------

def test_as_of_is_mandatory_and_keyword_only():
    races = races_with_times([30.0 + i * 0.1 for i in range(30)])
    with pytest.raises(TypeError):
        pace.fit_pace_definition(races, target=make_target())  # as_of を渡さない
    with pytest.raises(TypeError):
        pace.fit_pace_definition(races, make_target(), AS_OF)  # target/as_of は位置引数にできない


def test_race_at_exactly_as_of_is_excluded():
    """hasso_at が as_of ちょうどのレースは「学習期間より後」として除外する。"""
    fine = races_with_times([30.0 + i * 0.1 for i in range(29)], hasso_at=BEFORE, jyo_cd="06")
    at_cutoff = races_with_times([31.0], hasso_at=AS_OF, jyo_cd="06")  # ちょうど as_of
    # 除外されるなら29件のままで足りず、競馬場を落とした層に落ちるはず。
    other_venue = races_with_times([31.0], hasso_at=BEFORE, jyo_cd="07")

    definition = pace.fit_pace_definition(
        fine + at_cutoff + other_venue, target=make_target(), as_of=AS_OF)

    assert definition is not None
    assert "jyo_cd" not in definition["stratum"]
    assert definition["race_count"] == 30
