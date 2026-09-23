"""keiba.evaluate.splits のテスト。"""

from __future__ import annotations

import inspect
import json
from datetime import date, datetime, timedelta

import pytest

from keiba.evaluate import backtest as bt

from keiba.evaluate.splits import (
    SPLIT_NAMES,
    Splits,
    SplitError,
    assign,
    by_fraction,
    group_by_split,
)
from keiba.models import RaceRow


def make_race(race_id: str, hasso_at: datetime, **kw) -> RaceRow:
    base = dict(
        race_id=race_id,
        hasso_at=hasso_at,
        jyo_cd="06",
        kyori_m=2000,
        track_cd="17",
        track_kind="turf",
        baba_cd="1",
        class_level=4,
        grade_cd=" ",
        jyoken_cd_youngest="016",
        toroku_tosu=12,
        syusso_tosu=12,
        nyusen_tosu=12,
        first_3f_sec=35.2,
        race_name="テスト賞",
    )
    base.update(kw)
    return RaceRow(**base)


def _cutoff(race: RaceRow) -> datetime:
    """バックテストと同じ基準（予想締切 = 発走の18時間前）。

    定義は keiba.evaluate.backtest 側に1つだけあるので、そこから取る
    （ここで 18 を書き写すと2箇所になる）。
    """
    return bt.prediction_cutoff_for(race)


SPLITS = Splits(
    train_until=date(2026, 1, 31),
    dev_until=date(2026, 2, 20),
    calib_until=date(2026, 3, 5),
)


# --- assign ------------------------------------------------------------------

def test_assign_does_not_invite_a_hasso_at_argument():
    """引数名が ``hasso_at`` でないこと（fix6 ④）。

    呼び出し元は予想締切を渡しているのに引数名が ``hasso_at`` だったので、
    次に素直に発走日時を渡す人が fix5 §4 の欠陥（境界の翌開催日が丸ごと1区分
    ずれる）を作り直せた。挙動ではなく**再発経路**を止めるのが目的なので、
    名前そのものを固定する。
    """
    parameters = list(inspect.signature(assign).parameters)
    assert parameters[0] != "hasso_at"
    assert parameters[0] == "at"


def test_assign_puts_early_dates_in_train():
    assert assign(datetime(2026, 1, 1, 15, 0), SPLITS) == "train"


def test_assign_boundaries_are_inclusive_on_the_left():
    """hasso_at.date() == train_until は train に入る（境界を含む）。"""
    assert assign(datetime(2026, 1, 31, 23, 59), SPLITS) == "train"


def test_assign_puts_late_dates_in_test():
    assert assign(datetime(2026, 3, 6, 0, 0), SPLITS) == "test"


def test_assign_covers_all_four_splits():
    cases = {
        datetime(2026, 1, 1): "train",
        datetime(2026, 2, 1): "dev",
        datetime(2026, 2, 25): "calib",
        datetime(2026, 4, 1): "test",
    }
    for hasso_at, expected in cases.items():
        assert assign(hasso_at, SPLITS) == expected
    assert set(cases.values()) == set(SPLIT_NAMES)


# --- by_fraction ---------------------------------------------------------

def _consecutive_dates(n: int, start: date = date(2026, 1, 1)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def test_by_fraction_respects_chronological_order():
    dates = _consecutive_dates(37)
    splits = by_fraction(dates, fractions=(0.4, 0.3, 0.2, 0.1))
    assert splits.train_until <= splits.dev_until <= splits.calib_until


def test_by_fraction_uses_given_fractions():
    """100日分で (0.5,0.2,0.15,0.15) なら概ね 50/20/15/15。"""
    dates = _consecutive_dates(100)
    splits = by_fraction(dates, fractions=(0.5, 0.2, 0.15, 0.15))

    train_count = sum(1 for d in dates if d <= splits.train_until)
    dev_count = sum(1 for d in dates if splits.train_until < d <= splits.dev_until)
    calib_count = sum(1 for d in dates if splits.dev_until < d <= splits.calib_until)
    test_count = sum(1 for d in dates if d > splits.calib_until)

    assert train_count == 50
    assert dev_count == 20
    assert calib_count == 15
    assert test_count == 15


def test_by_fraction_raises_when_fractions_do_not_sum_to_one():
    with pytest.raises(SplitError):
        by_fraction(_consecutive_dates(10), fractions=(0.5, 0.2, 0.2, 0.2))


def test_by_fraction_raises_on_empty_dates():
    with pytest.raises(SplitError):
        by_fraction([])


# --- group_by_split -----------------------------------------------------

def test_group_by_split_never_splits_a_race():
    """同一 race_id は必ず1つの区分にしか現れない（spec §9.2）。

    区分の割り当ては RaceRow（レース単位）に対して1回だけ行われるので、
    その race_id の全頭が同じ答えを受け取る構造になっていることを固定する。
    """
    races = [
        make_race("R1", datetime(2026, 1, 10, 15, 0)),
        make_race("R2", datetime(2026, 2, 10, 15, 0)),
        make_race("R3", datetime(2026, 2, 25, 15, 0)),
        make_race("R4", datetime(2026, 4, 1, 15, 0)),
    ]
    # 判定に使う時刻は呼び出し側が決める（既定値は無い。fix6 ④）。ここでは
    # バックテストと同じ基準（発走の18時間前＝予想締切）を渡す。
    groups = group_by_split(races, SPLITS, at=_cutoff)

    seen: dict[str, str] = {}
    for split_name, races_in_split in groups.items():
        for race in races_in_split:
            assert race.race_id not in seen, "同じレースが2つの区分に現れた"
            seen[race.race_id] = split_name

    assert seen == {"R1": "train", "R2": "dev", "R3": "calib", "R4": "test"}


def test_group_by_split_requires_the_basis_time():
    """``at`` を省略できない（既定にできる唯一の候補が hasso_at なので）。"""
    with pytest.raises(TypeError):
        group_by_split([], SPLITS)


def test_group_by_split_returns_all_four_keys():
    groups = group_by_split([], SPLITS, at=_cutoff)
    assert set(groups.keys()) == set(SPLIT_NAMES)
    assert all(value == [] for value in groups.values())


# --- Splits.as_dict -----------------------------------------------------

def test_as_dict_is_json_serialisable():
    payload = SPLITS.as_dict()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded == {
        "train_until": "2026-01-31",
        "dev_until": "2026-02-20",
        "calib_until": "2026-03-05",
    }
