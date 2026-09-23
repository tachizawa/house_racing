"""keiba.features.derive のテスト。

4つの機能群（条件別成績・近況・脚質・輸送）は互いに独立しているので、
クラスを分けて1つずつ確かめる。数値は手計算で確かめられるものだけ使う。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from keiba.features.derive import (
    BABA_UNKNOWN_REASON,
    CONDITION_KEYS,
    ConditionRecord,
    RecentForm,
    RunningStyle,
    Transport,
    condition_records,
    recent_form,
    running_style,
    transport,
)
from keiba.models import HorseEntry, RaceInput
from tests.fixtures.runs import make_past_run


def make_race(**kw) -> RaceInput:
    defaults = dict(
        date=date(2026, 3, 21),
        jyo="中山",
        race_num=11,
        track_kind="turf",
        kyori_m=2000,
        entry_count=1,
        prediction_cutoff=datetime(2026, 3, 20, 20, 0, 0),
        horses=(HorseEntry(umaban="1", bamei="テストウマ"),),
        jyo_cd="06",
        baba_cd="1",
    )
    defaults.update(kw)
    return RaceInput(**defaults)


# === 条件別成績 ==============================================================


class TestConditionRecords:
    def test_exact_distance_matches_only_identical(self):
        """2000m のレースで 2000m の走だけが exact_distance に入る。"""
        race = make_race(kyori_m=2000)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=2000),
            make_past_run(hasso_at=datetime(2026, 1, 4), kyori_m=1800),
        ]
        records = condition_records(runs, race=race)
        assert records["exact_distance"].starts == 1

    @pytest.mark.parametrize("kyori_m, included", [
        (1800, True),    # diff 200（下限）。境界を含むので入る
        (2200, True),    # diff 200（上限）。境界を含むので入る
        (1799, False),   # diff 201（下限のすぐ外）。次の編集が壊しうる境界
        (2201, False),   # diff 201（上限のすぐ外）。同上
        (1600, False),   # diff 400。明らかに範囲外
        (2400, False),   # diff 400。同上
    ])
    def test_same_distance_uses_200m_tolerance(self, kyori_m, included):
        """±200m 以内（両端含む）だけが same_distance に入る。201m 差は境界の
        すぐ外なので、200m 差との区別が将来の編集で崩れていないか確かめる。"""
        race = make_race(kyori_m=2000)
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=kyori_m)]
        records = condition_records(runs, race=race)
        assert records["same_distance"].starts == (1 if included else 0)

    def test_same_venue_matches_jyo_cd(self):
        race = make_race(jyo_cd="06")
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), jyo_cd="06"),
            make_past_run(hasso_at=datetime(2026, 1, 4), jyo_cd="05"),
        ]
        records = condition_records(runs, race=race)
        assert records["same_venue"].starts == 1

    def test_same_track_kind_matches_turf_or_dirt(self):
        race = make_race(track_kind="turf")
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="turf"),
            make_past_run(hasso_at=datetime(2026, 1, 4), track_kind="dirt"),
        ]
        records = condition_records(runs, race=race)
        assert records["same_track_kind"].starts == 1

    def test_same_baba_requires_same_track_kind_too(self):
        """馬場状態コードが同じ「1（良）」でも、芝とダートでは別の基準なので
        same_baba には入らない。"""
        race = make_race(track_kind="turf", baba_cd="1")
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="turf", baba_cd="1"),
            make_past_run(hasso_at=datetime(2026, 1, 4), track_kind="dirt", baba_cd="1"),
        ]
        records = condition_records(runs, race=race)
        assert records["same_baba"].starts == 1

    def test_same_baba_is_inapplicable_when_going_is_unknown(self):
        """race.baba_cd が None（利用者が馬場状態を入力しなかった）なら、
        same_baba は「0戦」ではなく「定義できない」として返る。0戦は
        「この馬はこの馬場状態で走ったことがない」という馬についての主張に
        なってしまうが、実際には今回の馬場状態が分からないだけ（fix2 brief）。
        """
        race = make_race(track_kind="turf", baba_cd=None)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="turf", baba_cd="1"),
            make_past_run(hasso_at=datetime(2026, 1, 4), track_kind="turf", baba_cd="2"),
        ]
        records = condition_records(runs, race=race)
        assert records["same_baba"].inapplicable_reason == BABA_UNKNOWN_REASON

    def test_inapplicable_same_baba_has_no_counts(self):
        """定義できない same_baba は starts/wins/top3 がすべて None
        （0 ではない——0 は「該当する走が無かった」という別の主張のため）。"""
        race = make_race(track_kind="turf", baba_cd=None)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="turf", baba_cd="1"),
            make_past_run(hasso_at=datetime(2026, 1, 4), track_kind="turf", baba_cd="2"),
        ]
        record = condition_records(runs, race=race)["same_baba"]
        assert record.starts is None
        assert record.wins is None
        assert record.top3 is None

    def test_other_conditions_still_counted_when_going_is_unknown(self):
        """same_baba が定義できなくても、他4区分は実際の件数のまま集計される
        （None にはならない）。"""
        race = make_race(kyori_m=2000, jyo_cd="06", track_kind="turf", baba_cd=None)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=2000, jyo_cd="06",
                           track_kind="turf", baba_cd="1"),
            make_past_run(hasso_at=datetime(2026, 1, 4), kyori_m=2000, jyo_cd="06",
                           track_kind="turf", baba_cd="2"),
        ]
        records = condition_records(runs, race=race)
        for key in ("exact_distance", "same_distance", "same_venue", "same_track_kind"):
            assert records[key].starts is not None
            assert records[key].inapplicable_reason is None

    def test_same_baba_counts_when_going_is_given(self):
        """baba を与えれば同馬場状態は今まで通り件数として集計される。"""
        race = make_race(track_kind="turf", baba_cd="1")
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="turf", baba_cd="1")]
        record = condition_records(runs, race=race)["same_baba"]
        assert record.inapplicable_reason is None
        assert record.starts == 1

    def test_same_baba_requires_matching_track_kind(self):
        """馬場コードが同じ「1（良）」でも、トラック種別が違えば別基準
        （芝とダートは別基準で馬場状態コードが付く）なので same_baba には
        入らない。これは本物の0（inapplicable_reason は None のまま）。"""
        race = make_race(track_kind="turf", baba_cd="1")
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="dirt", baba_cd="1")]
        record = condition_records(runs, race=race)["same_baba"]
        assert record.starts == 0
        assert record.inapplicable_reason is None

    def test_condition_keys_are_unchanged(self):
        """CONDITION_KEYS は5要素で、順序も従来通り
        （exact_distance, same_distance, same_venue, same_track_kind, same_baba）。"""
        assert CONDITION_KEYS == (
            "exact_distance", "same_distance", "same_venue",
            "same_track_kind", "same_baba",
        )

    def test_zero_starts_reports_none_mean_not_zero(self):
        race = make_race(kyori_m=2000)
        records = condition_records([], race=race)
        record = records["exact_distance"]
        assert record.starts == 0
        assert record.wins == 0
        assert record.top3 == 0
        assert record.margin_sec_mean is None

    def test_wins_counts_only_first_place(self):
        race = make_race(kyori_m=2000)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=2000, kakutei_jyuni=1),
            make_past_run(hasso_at=datetime(2026, 1, 4), kyori_m=2000, kakutei_jyuni=2),
        ]
        records = condition_records(runs, race=race)
        assert records["exact_distance"].wins == 1

    def test_top3_counts_first_to_third(self):
        race = make_race(kyori_m=2000)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=2000, kakutei_jyuni=1),
            make_past_run(hasso_at=datetime(2026, 1, 4), kyori_m=2000, kakutei_jyuni=3),
            make_past_run(hasso_at=datetime(2026, 1, 3), kyori_m=2000, kakutei_jyuni=4),
        ]
        records = condition_records(runs, race=race)
        assert records["exact_distance"].top3 == 2

    def test_excludes_scratched_from_starts(self):
        """ijyo_cd="1"（出走取消）は走っていないので starts に数えない。"""
        race = make_race(kyori_m=2000)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=2000, ijyo_cd="1",
                           kakutei_jyuni=None),
            make_past_run(hasso_at=datetime(2026, 1, 4), kyori_m=2000, ijyo_cd="0"),
        ]
        records = condition_records(runs, race=race)
        assert records["exact_distance"].starts == 1

    def test_includes_no_time_runs_in_starts_but_not_mean(self):
        """ijyo_cd="4"（競走中止）は starts に数えるが margin_sec_mean には入れない。"""
        race = make_race(kyori_m=2000)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=2000, ijyo_cd="4",
                           kakutei_jyuni=None, margin_sec=None),
            make_past_run(hasso_at=datetime(2026, 1, 4), kyori_m=2000, ijyo_cd="0",
                           margin_sec=1.0),
        ]
        record = condition_records(runs, race=race)["exact_distance"]
        assert record.starts == 2
        assert record.margin_sec_mean == pytest.approx(1.0)

    def test_period_reflects_matched_runs(self):
        race = make_race(kyori_m=2000)
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), kyori_m=2000),
            make_past_run(hasso_at=datetime(2026, 2, 10), kyori_m=2000),
            make_past_run(hasso_at=datetime(2025, 12, 1), kyori_m=1800),  # 別区分
        ]
        record = condition_records(runs, race=race)["exact_distance"]
        assert record.period_from == date(2026, 1, 5)
        assert record.period_to == date(2026, 2, 10)

    def test_counts_are_always_present_with_rates(self):
        """どの区分でも starts が必ず入る（spec §4.3）。5区分すべてが常に返る。"""
        race = make_race()
        records = condition_records([], race=race)
        assert set(records.keys()) == set(CONDITION_KEYS)
        for key in CONDITION_KEYS:
            assert isinstance(records[key], ConditionRecord)
            assert records[key].starts == 0


# === 近況 ====================================================================


class TestRecentForm:
    def test_days_since_last_run_from_as_of(self):
        race = make_race()
        runs = [make_past_run(hasso_at=datetime(2026, 3, 1, 15, 0))]
        form = recent_form(runs, as_of=datetime(2026, 3, 21, 10, 0), race=race,
                            futan_kg=57.0, class_level=4)
        assert form.days_since_last_run == 19

    def test_changes_are_signed_differences(self):
        """距離 +200、斤量 +1.0、クラス 0 の符号付き差分。"""
        race = make_race(kyori_m=2000)
        runs = [make_past_run(hasso_at=datetime(2026, 3, 1), kyori_m=1800,
                               futan_kg=56.0, class_level=4)]
        form = recent_form(runs, as_of=datetime(2026, 3, 21), race=race,
                            futan_kg=57.0, class_level=4)
        assert form.distance_change_m == 200
        assert form.futan_change_kg == pytest.approx(1.0)
        assert form.class_level_change == 0

    def test_all_none_when_no_past_runs(self):
        race = make_race()
        form = recent_form([], as_of=datetime(2026, 3, 21), race=race,
                            futan_kg=57.0, class_level=4)
        assert form.days_since_last_run is None
        assert form.distance_change_m is None
        assert form.futan_change_kg is None
        assert form.class_level_change is None
        assert form.last_run is None

    def test_class_change_none_when_either_side_missing(self):
        race = make_race()
        runs = [make_past_run(hasso_at=datetime(2026, 3, 1), class_level=None)]
        form = recent_form(runs, as_of=datetime(2026, 3, 21), race=race,
                            futan_kg=57.0, class_level=4)
        assert form.class_level_change is None

        runs_today_missing = [make_past_run(hasso_at=datetime(2026, 3, 1), class_level=4)]
        form_today_missing = recent_form(runs_today_missing, as_of=datetime(2026, 3, 21),
                                          race=race, futan_kg=57.0, class_level=None)
        assert form_today_missing.class_level_change is None

    def test_last_run_includes_conditions_and_weight(self):
        race = make_race()
        runs = [make_past_run(
            hasso_at=datetime(2026, 3, 1), kyori_m=1800, track_kind="dirt", baba_cd="2",
            class_level=3, syusso_tosu=14, kakutei_jyuni=2, margin_sec=0.4, futan_kg=55.0,
            ba_taijyu_kg=480,
        )]
        form = recent_form(runs, as_of=datetime(2026, 3, 21), race=race,
                            futan_kg=57.0, class_level=4)
        assert form.last_run == {
            "kyori_m": 1800,
            "track_kind": "dirt",
            "baba_cd": "2",
            "class_level": 3,
            "syusso_tosu": 14,
            "kakutei_jyuni": 2,
            "margin_sec": 0.4,
            "futan_kg": 55.0,
            "ba_taijyu_kg": 480,
        }

    def test_last_run_weight_is_previous_race_value(self):
        """馬体重は前走の値。今回の値は当日発表なので last_run には出てこない。"""
        race = make_race()
        runs = [make_past_run(hasso_at=datetime(2026, 3, 1), ba_taijyu_kg=470)]
        form = recent_form(runs, as_of=datetime(2026, 3, 21), race=race,
                            futan_kg=57.0, class_level=4)
        assert form.last_run["ba_taijyu_kg"] == 470


# === 脚質 ====================================================================


class TestRunningStyle:
    def test_corner_position_is_relative(self):
        """jyuni_4c=3, syusso_tosu=12 → 3/12 = 0.25。"""
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=3, syusso_tosu=12)]
        style = running_style(runs)
        assert style.corner_pos_mean == pytest.approx(0.25)
        assert style.corner_pos_starts == 1

    def test_mean_and_stdev_over_recent_runs(self):
        """位置 [0.25, 0.5] → 平均 0.375、stdev = sqrt(((0.25-0.375)^2+(0.5-0.375)^2)/1)。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=3, syusso_tosu=12),
            make_past_run(hasso_at=datetime(2026, 1, 4), jyuni_4c=6, syusso_tosu=12),
        ]
        style = running_style(runs)
        assert style.corner_pos_mean == pytest.approx(0.375)
        assert style.corner_pos_stdev == pytest.approx(0.17677669529663687)

    def test_stdev_none_with_single_run(self):
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=3, syusso_tosu=12)]
        style = running_style(runs)
        assert style.corner_pos_stdev is None

    def test_source_is_jyuni_4c_when_available(self):
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=3, syusso_tosu=12)]
        style = running_style(runs)
        assert style.corner_source == "jyuni_4c"

    def test_falls_back_to_jyuni_3c(self):
        """4角が全走 None なら jyuni_3c を使う。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=None, jyuni_3c=4,
                           syusso_tosu=12),
            make_past_run(hasso_at=datetime(2026, 1, 4), jyuni_4c=None, jyuni_3c=6,
                           syusso_tosu=12),
        ]
        style = running_style(runs)
        assert style.corner_source == "jyuni_3c"
        assert style.corner_pos_starts == 2

    def test_falls_back_to_jyuni_2c(self):
        """4角も3角も無ければ jyuni_2c まで落とす。"""
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=None, jyuni_3c=None,
                               jyuni_2c=5, syusso_tosu=12)]
        style = running_style(runs)
        assert style.corner_source == "jyuni_2c"

    def test_all_none_when_no_corner_data(self):
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=None, jyuni_3c=None,
                               jyuni_2c=None, syusso_tosu=12)]
        style = running_style(runs)
        assert style == RunningStyle(None, None, 0, None)

    def test_does_not_mix_sources(self):
        """jyuni_4c を選んだら、4角が無い走は jyuni_3c で埋めずに集計から外す。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=3, jyuni_3c=None,
                           syusso_tosu=12),
            make_past_run(hasso_at=datetime(2026, 1, 4), jyuni_4c=None, jyuni_3c=5,
                           syusso_tosu=10),
        ]
        style = running_style(runs)
        assert style.corner_source == "jyuni_4c"
        assert style.corner_pos_starts == 1
        assert style.corner_pos_mean == pytest.approx(0.25)

    def test_skips_runs_without_field_size(self):
        """syusso_tosu が None か 0 の走は外す。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=3, syusso_tosu=None),
            make_past_run(hasso_at=datetime(2026, 1, 4), jyuni_4c=3, syusso_tosu=0),
            make_past_run(hasso_at=datetime(2026, 1, 3), jyuni_4c=3, syusso_tosu=12),
        ]
        style = running_style(runs)
        assert style.corner_pos_starts == 1

    def test_result_independent_of_input_order(self):
        """呼び出し側の並び順に依存しない。関数内で hasso_at 降順に並べ直す
        （tests/test_ability.py の test_result_independent_of_input_order と同じ理由）。

        7走ぶん（既定の recent_runs=5 より多い）用意し、日付ごとに別々の
        jyuni_4c を割り当てる。並べ替えずに先頭5件を切り出すだけだと、
        シャッフルした入力では直近5走とは違う5走が選ばれて平均が変わって
        しまう。並べ替えていれば、入力順によらず同じ直近5走・同じ平均になる。
        """
        ordered = [
            make_past_run(hasso_at=datetime(2026, 1, 7), jyuni_4c=7, syusso_tosu=10),
            make_past_run(hasso_at=datetime(2026, 1, 6), jyuni_4c=6, syusso_tosu=10),
            make_past_run(hasso_at=datetime(2026, 1, 5), jyuni_4c=5, syusso_tosu=10),
            make_past_run(hasso_at=datetime(2026, 1, 4), jyuni_4c=4, syusso_tosu=10),
            make_past_run(hasso_at=datetime(2026, 1, 3), jyuni_4c=3, syusso_tosu=10),
            make_past_run(hasso_at=datetime(2026, 1, 2), jyuni_4c=2, syusso_tosu=10),
            make_past_run(hasso_at=datetime(2026, 1, 1), jyuni_4c=1, syusso_tosu=10),
        ]
        shuffled = [ordered[1], ordered[5], ordered[0], ordered[3],
                    ordered[6], ordered[2], ordered[4]]
        result_ordered = running_style(ordered)
        result_shuffled = running_style(shuffled)
        assert result_shuffled == result_ordered
        # 直近5走（1/7〜1/3, jyuni_4c=7..3）の平均のはず。1/2, 1/1 は窓の外。
        assert result_ordered.corner_pos_mean == pytest.approx(0.5)
        assert result_ordered.corner_pos_starts == 5


# === 輸送 ====================================================================


class TestTransport:
    def test_base_from_tozai_cd(self):
        assert transport([], tozai_cd="1", jyo_cd="06").base == "美浦"
        assert transport([], tozai_cd="2", jyo_cd="06").base == "栗東"

    def test_base_none_for_non_trainee_centre(self):
        for code in ("0", "3", "4"):
            result = transport([], tozai_cd=code, jyo_cd="06")
            assert result.base is None
            assert result.estimated_transport_km is None

    def test_estimated_km_from_table(self):
        """美浦 → 中山（jyo_cd="06"）は TRANSPORT_KM の値と一致。"""
        from keiba import config
        result = transport([], tozai_cd="1", jyo_cd="06")
        assert result.estimated_transport_km == config.TRANSPORT_KM[("美浦", "06")]

    def test_stay_status_defaults_to_unknown(self):
        """遠隔開催（美浦→札幌）でも滞在有無は推測せず既定の unknown のまま。"""
        result = transport([], tozai_cd="1", jyo_cd="01")
        assert result.stay_status == "unknown"

    def test_stay_status_accepts_confirmed_value(self):
        result = transport([], tozai_cd="1", jyo_cd="01", stay_status="stay")
        assert result.stay_status == "stay"

    def test_past_long_transport_counts_only_at_or_over_threshold(self):
        """LONG_TRANSPORT_KM=300 以上の過去走だけ集計する（仕様 §6.6「以上」）。

        過去走はその走の tozai_cd/jyo_cd で距離を出す。美浦→新潟(04)は330km
        （閾値超え）、美浦→中山(06)は90km（閾値未満）。
        """
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), tozai_cd="1", jyo_cd="04"),
            make_past_run(hasso_at=datetime(2026, 1, 4), tozai_cd="1", jyo_cd="06"),
        ]
        result = transport(runs, tozai_cd="1", jyo_cd="06")
        assert result.past_long_transport_starts == 1

    def test_past_long_transport_reports_starts_even_when_zero(self):
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), tozai_cd="1", jyo_cd="06")]
        result = transport(runs, tozai_cd="1", jyo_cd="06")
        assert result.past_long_transport_starts == 0
        assert result.past_long_transport_zogen_mean is None
        assert result.past_long_transport_zogen_min is None

    def test_past_long_transport_uses_run_time_affiliation(self):
        """過去走はその走の時点の tozai_cd で距離を出す。今回の所属(呼び出し側の
        tozai_cd)では計算しない — 転厩後の馬でそれを確かめる。

        この馬は過去に栗東所属で新潟(04)へ行った（栗東→新潟=550km、閾値超え）。
        今回は美浦所属で呼び出す。今回の所属(美浦)で過去走を計算すると
        美浦→新潟=330km でも閾値超えなので区別できないため、
        栗東→中山(06)=500km（閾値超え）と美浦→中山(06)=90km（閾値未満）の
        組み合わせで、affiliation を取り違えたら starts が変わることを確かめる。
        """
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), tozai_cd="2", jyo_cd="06")]
        # 過去走時点は栗東所属 → 栗東→中山=500km（閾値超え）→ starts=1 になるはず。
        result = transport(runs, tozai_cd="1", jyo_cd="06")
        assert result.past_long_transport_starts == 1

        # 対照: 今回の所属(美浦)で誤って計算すると 美浦→中山=90km で閾値未満になり
        # starts=0 になってしまう。実装が正しければそうはならない。
        assert result.past_long_transport_starts != 0

    def test_past_long_transport_uses_the_run_s_own_venue(self):
        """過去走の開催地も、その走自身の値から取る。

        今回の開催地で過去の輸送距離を計算すると、実際には行っていない
        場所への輸送として数えてしまう。tozai_cd と jyo_cd の両方が
        「その走自身の値」でなければならない。

        過去走: 栗東 → 札幌(01)（1500km、閾値超え）。
        今回の呼び出し: 栗東 → 京都(08)（20km、閾値未満）。
        もし実装が今回の jyo_cd（京都）で過去走の距離を計算していたら
        閾値未満になり starts=0 になるはずなので、tozai_cd 側だけでなく
        jyo_cd 側の取り違えも検出できる。
        """
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), tozai_cd="2", jyo_cd="01",
                               zogen_sa_kg=-10)]
        result = transport(runs, tozai_cd="2", jyo_cd="08")
        assert result.past_long_transport_starts == 1

    def test_past_long_transport_skips_missing_weight_change(self):
        """zogen_sa_kg が None の走は starts には数えるが平均には入れない。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), tozai_cd="1", jyo_cd="04",
                           zogen_sa_kg=None),
            make_past_run(hasso_at=datetime(2026, 1, 4), tozai_cd="1", jyo_cd="04",
                           zogen_sa_kg=-6),
        ]
        result = transport(runs, tozai_cd="1", jyo_cd="06")
        assert result.past_long_transport_starts == 2
        assert result.past_long_transport_zogen_mean == pytest.approx(-6.0)
        assert result.past_long_transport_zogen_min == -6

    def test_counts_a_long_journey_to_a_jump_race(self):
        """障害戦への輸送も輸送として数える。

        測っているのは輸送の負担であって、着差の比較可能性ではない。
        能力指標（ability.py）の除外条件（is_flat）をそのまま流用すると、
        実際にあった長距離輸送の証拠を無関係な理由で捨てることになる。
        栗東→札幌(01)は TRANSPORT_KM で1500km（閾値300超）。
        """
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="jump",
                           tozai_cd="2", jyo_cd="01", zogen_sa_kg=-12),
            make_past_run(hasso_at=datetime(2026, 1, 4), track_kind="turf",
                           tozai_cd="2", jyo_cd="01", zogen_sa_kg=-8),
        ]
        result = transport(runs, tozai_cd="1", jyo_cd="06")
        assert result.past_long_transport_starts == 2
        assert result.past_long_transport_zogen_mean == pytest.approx(-10.0)
        assert result.past_long_transport_zogen_min == -12

    def test_excludes_a_scratched_run_from_long_transport(self):
        """出走取消（ijyo_cd="1"）は輸送されたかも計量もはっきりしないので
        starts に数えない（counts_as_starter だけは輸送でも除外条件として残す）。
        """
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), tozai_cd="2", jyo_cd="01",
                           ijyo_cd="1", kakutei_jyuni=None, zogen_sa_kg=None),
            make_past_run(hasso_at=datetime(2026, 1, 4), tozai_cd="2", jyo_cd="01",
                           zogen_sa_kg=-8),
        ]
        result = transport(runs, tozai_cd="1", jyo_cd="06")
        assert result.past_long_transport_starts == 1
        assert result.past_long_transport_zogen_mean == pytest.approx(-8.0)


def test_past_long_transport_includes_the_threshold_itself(monkeypatch):
    """閾値ちょうどの走を含める（仕様 §6.6 は「閾値以上」。fix5 決定10）。

    固定テーブルに 300km ちょうどの組が無いので、閾値そのものを既存の距離
    （美浦→新潟 = 330km）に合わせて境界を作る。`>` で判定していると
    330 > 330 が偽になり、境界の走が落ちる。
    """
    from keiba import config

    monkeypatch.setattr(config, "LONG_TRANSPORT_KM",
                         config.TRANSPORT_KM[("美浦", "04")])
    runs = [make_past_run(hasso_at=datetime(2026, 1, 5), tozai_cd="1", jyo_cd="04")]

    result = transport(runs, tozai_cd="1", jyo_cd="06")

    assert result.past_long_transport_starts == 1
