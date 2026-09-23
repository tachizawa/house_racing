"""能力指標（ability_summary / field_ranks）のテスト。

着差とクラスは別々の数値として返るべきで、両者を合成する計算はここには
一切現れない（CLAUDE.md の設計制約）。数値は手計算で確かめられるものだけ使う。
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from keiba.features.ability import ability_summary, field_ranks
from tests.fixtures.runs import make_past_run


class TestAbilitySummaryMargin:
    def test_weighted_average_uses_recency_weights(self):
        """新しい順 [0.0, 1.0, 2.0] → 重み 5,4,3 → (0*5+1*4+2*3)/(5+4+3) = 10/12。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), margin_sec=0.0),
            make_past_run(hasso_at=datetime(2026, 1, 4), margin_sec=1.0),
            make_past_run(hasso_at=datetime(2026, 1, 3), margin_sec=2.0),
        ]
        summary = ability_summary(runs)
        assert summary.recent_margin_sec_wavg == pytest.approx(10 / 12)
        assert summary.runs_used == 3

    def test_uses_only_recent_runs(self):
        """7走あっても直近5走だけを使う。古い2走は加重平均にも影響しない。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 10 - i), margin_sec=float(i))
            for i in range(7)
        ]
        summary = ability_summary(runs)
        assert summary.runs_used == 5
        expected = (0 * 5 + 1 * 4 + 2 * 3 + 3 * 2 + 4 * 1) / (5 + 4 + 3 + 2 + 1)
        assert summary.recent_margin_sec_wavg == pytest.approx(expected)

    def test_runs_used_reflects_fewer_runs(self):
        """2走しかなければ runs_used == 2。分母も使った重み(5+4)だけ。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), margin_sec=0.0),
            make_past_run(hasso_at=datetime(2026, 1, 4), margin_sec=1.0),
        ]
        summary = ability_summary(runs)
        assert summary.runs_used == 2
        assert summary.recent_margin_sec_wavg == pytest.approx((0 * 5 + 1 * 4) / (5 + 4))

    def test_returns_none_for_no_runs(self):
        summary = ability_summary([])
        assert summary.recent_margin_sec_wavg is None
        assert summary.runs_used == 0
        assert summary.recent_class_level_wavg is None
        assert summary.best_class_level is None
        assert summary.period_from is None
        assert summary.period_to is None

    def test_skips_runs_without_time(self):
        """競走中止（ijyo_cd=4）は集計から外れるが、窓の1枠は消費する。

        窓は [Jan5(margin0.0,重み5), Jan4(競走中止,重み4), Jan3(margin2.0,重み3)]。
        競走中止の走が重み4の枠を使い切るので、Jan3 は重み3のまま
        （繰り上がって重み4にはならない）ことを確認する。
        """
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), margin_sec=0.0, ijyo_cd="0"),
            make_past_run(hasso_at=datetime(2026, 1, 4), margin_sec=1.0, ijyo_cd="4"),
            make_past_run(hasso_at=datetime(2026, 1, 3), margin_sec=2.0, ijyo_cd="0"),
        ]
        summary = ability_summary(runs)
        assert summary.runs_used == 2
        assert summary.recent_margin_sec_wavg == pytest.approx((0 * 5 + 2 * 3) / (5 + 3))

    def test_counts_demoted_run_as_normal(self):
        """降着（ijyo_cd=7）は確定着順があるので通常どおり集計に入る。"""
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), margin_sec=1.5, ijyo_cd="7")]
        summary = ability_summary(runs)
        assert summary.runs_used == 1
        assert summary.recent_margin_sec_wavg == pytest.approx(1.5)

    def test_skips_jump_runs(self):
        """障害は集計対象外。窓の枠も消費しない（平地の走だけが繰り上がる）。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 5), track_kind="jump", margin_sec=0.0),
            make_past_run(hasso_at=datetime(2026, 1, 4), track_kind="turf", margin_sec=1.0),
        ]
        summary = ability_summary(runs)
        assert summary.runs_used == 1
        # 障害が枠を消費しないので、平地の1走だけが重み5で加重平均される。
        assert summary.recent_margin_sec_wavg == pytest.approx(1.0)


class TestAbilitySummaryClassLevel:
    def test_best_class_level_is_max_of_recent(self):
        """直近5走の class_level の最大値。窓外の6走目(=8)は無視される。"""
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 10), class_level=2),
            make_past_run(hasso_at=datetime(2026, 1, 9), class_level=5),
            make_past_run(hasso_at=datetime(2026, 1, 8), class_level=3),
            make_past_run(hasso_at=datetime(2026, 1, 7), class_level=1),
            make_past_run(hasso_at=datetime(2026, 1, 6), class_level=4),
            make_past_run(hasso_at=datetime(2026, 1, 5), class_level=8),
        ]
        summary = ability_summary(runs)
        assert summary.best_class_level == 5

    def test_class_level_wavg_uses_same_weights(self):
        """class_level も margin と同じ重み(5,4,3,2,1)で加重平均する。

        新しい順 [2,5,3,1,4] → (2*5+5*4+3*3+1*2+4*1)/(5+4+3+2+1) = 45/15 = 3.0。
        """
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 10), class_level=2),
            make_past_run(hasso_at=datetime(2026, 1, 9), class_level=5),
            make_past_run(hasso_at=datetime(2026, 1, 8), class_level=3),
            make_past_run(hasso_at=datetime(2026, 1, 7), class_level=1),
            make_past_run(hasso_at=datetime(2026, 1, 6), class_level=4),
        ]
        summary = ability_summary(runs)
        assert summary.recent_class_level_wavg == pytest.approx(3.0)

    def test_class_level_none_when_all_missing(self):
        """class_level が全走 None なら加重平均も最大値も None。margin は無関係に計算される。"""
        runs = [make_past_run(hasso_at=datetime(2026, 1, 5), class_level=None, margin_sec=0.5)]
        summary = ability_summary(runs)
        assert summary.recent_class_level_wavg is None
        assert summary.best_class_level is None
        assert summary.runs_used == 1


class TestAbilitySummaryPeriod:
    def test_period_covers_used_runs_only(self):
        """period は margin 集計に実際に使った走の日付範囲。

        Jan1 の走は競走中止で margin 集計から外れるので、窓には入っていても
        period には反映されない（period_from は Jan5 であって Jan1 ではない）。
        """
        runs = [
            make_past_run(hasso_at=datetime(2026, 1, 10), margin_sec=0.2, ijyo_cd="0"),
            make_past_run(hasso_at=datetime(2026, 1, 1), margin_sec=None, ijyo_cd="4"),
            make_past_run(hasso_at=datetime(2026, 1, 5), margin_sec=0.4, ijyo_cd="0"),
        ]
        summary = ability_summary(runs)
        assert summary.period_from == date(2026, 1, 5)
        assert summary.period_to == date(2026, 1, 10)


class TestAbilitySummaryOrdering:
    def test_result_independent_of_input_order(self):
        """呼び出し側の並び順に依存しない。関数内で hasso_at 降順に並べ直す。"""
        ordered = [
            make_past_run(hasso_at=datetime(2026, 1, 5), margin_sec=0.0, class_level=6),
            make_past_run(hasso_at=datetime(2026, 1, 4), margin_sec=1.0, class_level=2),
            make_past_run(hasso_at=datetime(2026, 1, 3), margin_sec=2.0, class_level=4),
        ]
        shuffled = [ordered[1], ordered[2], ordered[0]]
        assert ability_summary(shuffled) == ability_summary(ordered)


class TestFieldRanks:
    def test_field_ranks_ascending_smaller_is_better(self):
        result = field_ranks({"1": 0.1, "2": 0.5, "3": 0.3})
        assert result == {"1": 1.0, "2": 3.0, "3": 2.0}

    def test_field_ranks_averages_ties(self):
        result = field_ranks({"1": 0.3, "2": 0.3, "3": 0.5})
        assert result == {"1": 1.5, "2": 1.5, "3": 3.0}

    def test_field_ranks_keeps_none_as_none(self):
        result = field_ranks({"1": 0.2, "2": None})
        assert result["2"] is None
        assert result["1"] == 1.0

    def test_field_ranks_skips_none_in_numbering(self):
        result = field_ranks({"1": 0.1, "2": None, "3": 0.3})
        assert result == {"1": 1.0, "2": None, "3": 2.0}
