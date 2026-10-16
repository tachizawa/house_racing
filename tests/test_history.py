from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from keiba.features import history
from keiba.features.derive import BABA_UNKNOWN_REASON
from tests.fixtures.runs import make_past_run

JST = timezone(timedelta(hours=9))
AS_OF = datetime(2026, 9, 20, 18, 0, tzinfo=JST)
COVERAGE_FROM = datetime(2026, 8, 1, 9, 40, tzinfo=JST)


def run_at(month, day, **kw):
    return make_past_run(hasso_at=datetime(2026, month, day, 15, 0, tzinfo=JST), **kw)


def test_run_line_has_fixed_keys_and_no_future_fields():
    line = history.run_line(run_at(9, 6, kakutei_jyuni=2, margin_sec=0.1,
                                   jyuni_4c=3, track_cd="18"), as_of=AS_OF)
    assert tuple(line) == history.RUN_LINE_KEYS
    assert line["date"] == "2026-09-06"
    assert line["days_before"] == 14
    assert line["track_cd"] == "18"
    assert line["track_label"] == "芝・右外"
    assert line["kakutei_jyuni"] == 2
    assert line["corner_positions"] == [None, None, None, 3]


def test_recent_run_lines_newest_first_and_limited():
    runs = [run_at(8, 2), run_at(9, 6), run_at(8, 23)]
    lines = history.recent_run_lines(runs, as_of=AS_OF, limit=2)
    assert [line["date"] for line in lines] == ["2026-09-06", "2026-08-23"]


def test_recent_run_lines_excludes_scratched_runs():
    runs = [run_at(9, 6, ijyo_cd="1"), run_at(8, 23)]
    assert [l["date"] for l in history.recent_run_lines(runs, as_of=AS_OF, limit=5)] == ["2026-08-23"]


def test_status_found_far_from_coverage_start():
    runs = [run_at(9, 6)]
    status = history.history_status(runs, matched=True, race_class_level=2,
                                    coverage_from=datetime(2023, 8, 1, tzinfo=JST))
    assert status.status == "found"
    assert status.runs_found == 1
    assert status.period_from == date(2026, 9, 6)


def test_status_may_be_truncated_when_first_run_near_coverage_start():
    runs = [run_at(8, 10), run_at(9, 6)]
    status = history.history_status(runs, matched=True, race_class_level=2,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "found_may_be_truncated"


def test_status_debut_for_newcomer_race():
    status = history.history_status([], matched=True, race_class_level=0,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "debut"


def test_status_not_in_window_for_winners_class():
    # 1勝クラス以上（class_level >= 2）は勝利経験が必要なので過去走は実在する。
    status = history.history_status([], matched=True, race_class_level=2,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "not_in_window"


def test_status_unknown_for_maiden_race_without_runs():
    # 未勝利（1）は未出走馬も出られるので、取得期間外か未出走かを区別できない。
    status = history.history_status([], matched=True, race_class_level=1,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "unknown"


def test_status_unknown_when_class_level_is_none():
    status = history.history_status([], matched=True, race_class_level=None,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "unknown"


def test_status_unmatched_when_name_not_matched():
    status = history.history_status([], matched=False, race_class_level=0,
                                    coverage_from=COVERAGE_FROM)
    assert status.status == "unmatched"


def test_status_found_without_coverage_info_does_not_claim_truncation():
    status = history.history_status([run_at(8, 10)], matched=True, race_class_level=2,
                                    coverage_from=None)
    assert status.status == "found"


def today(**kw):
    base = dict(track_kind="turf", kyori_m=2000, jyo_cd="04", track_cd="11",
                baba_cd="1", class_level=3)
    base.update(kw)
    return history.TodayConditions(**base)


def test_combined_keys_are_fixed_and_ordered():
    records = history.combined_condition_records([], today=today())
    assert tuple(records) == history.COMBINED_CONDITION_KEYS


def test_surface_exact_distance_does_not_mix_dirt():
    runs = [run_at(9, 6, track_kind="turf", kyori_m=2000),
            run_at(8, 23, track_kind="dirt", kyori_m=2000)]
    definition, record = history.combined_condition_records(runs, today=today())["surface_exact_distance"]
    assert record.starts == 1
    assert "芝・ダート" in definition


def test_course_exact_distance_separates_inner_and_outer():
    runs = [run_at(9, 6, jyo_cd="04", track_cd="11", kyori_m=2000),
            run_at(8, 23, jyo_cd="04", track_cd="12", kyori_m=2000)]
    _, record = history.combined_condition_records(runs, today=today())["course_exact_distance"]
    assert record.starts == 1


def test_course_exact_distance_inapplicable_without_track_cd():
    _, record = history.combined_condition_records(
        [run_at(9, 6)], today=today(track_cd=None))["course_exact_distance"]
    assert record.starts is None
    assert record.inapplicable_reason is not None


def test_course_exact_distance_inapplicable_without_jyo_cd():
    # 競馬場が不明なのに 0戦 と出すと「同じコースを走ったことがない」と読める。
    _, record = history.combined_condition_records(
        [run_at(9, 6, jyo_cd="04", track_cd="11")],
        today=today(jyo_cd=None))["course_exact_distance"]
    assert record.starts is None
    assert record.inapplicable_reason == "今回の競馬場が不明のため比較できない"


def test_combined_record_counts_wins_top3_margin_and_period():
    # 同じ馬場（baba_cd "1"）の芝3走: 1着・3着（どちらも時計あり）・中止（時計なし）。
    # 別の馬場の1走は数えない。
    runs = [run_at(9, 6, kakutei_jyuni=1, margin_sec=-0.2),
            run_at(8, 23, kakutei_jyuni=3, margin_sec=0.4),
            run_at(8, 9, kakutei_jyuni=None, margin_sec=None, ijyo_cd="4"),
            run_at(8, 2, kakutei_jyuni=1, margin_sec=-0.5, baba_cd="3")]
    _, record = history.combined_condition_records(
        runs, today=today(baba_cd="1"))["surface_same_going"]
    assert record.starts == 3
    assert record.wins == 1
    assert record.top3 == 2
    assert abs(record.margin_sec_mean - 0.1) < 1e-9
    assert record.period_from == date(2026, 8, 9)
    assert record.period_to == date(2026, 9, 6)
    assert record.inapplicable_reason is None


def test_same_going_inapplicable_without_baba():
    _, record = history.combined_condition_records(
        [run_at(9, 6)], today=today(baba_cd=None))["surface_same_going"]
    assert record.starts is None
    assert record.inapplicable_reason == BABA_UNKNOWN_REASON


def test_near_class_window():
    runs = [run_at(9, 6, class_level=2), run_at(8, 30, class_level=4),
            run_at(8, 23, class_level=5)]
    _, record = history.combined_condition_records(
        runs, today=today(class_level=3))["surface_near_distance_near_class"]
    assert record.starts == 2


def test_near_class_inapplicable_without_class():
    _, record = history.combined_condition_records(
        [run_at(9, 6)], today=today(class_level=None))["surface_near_distance_near_class"]
    assert record.starts is None


def test_zero_matches_is_zero_not_none():
    _, record = history.combined_condition_records([], today=today())["surface_exact_distance"]
    assert record.starts == 0
    assert record.margin_sec_mean is None
