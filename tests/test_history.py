from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from keiba.features import history
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
