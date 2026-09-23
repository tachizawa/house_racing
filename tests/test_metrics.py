"""keiba.evaluate.metrics のテスト。

数値は手計算した小さな例と一致させる（各テストのdocstringに計算式を書く）。
"""

from __future__ import annotations

import math

import pytest

from keiba.evaluate.metrics import (
    MetricsReport,
    OutcomeError,
    RaceOutcome,
    aggregate,
    brier,
    calibration_bins,
    compare,
    expected_top_n,
    log_loss,
    log_loss_is_clipped,
    outcome_from_results,
    top1_win,
    top_n_hit,
)
from keiba.models import EntryRow, ResultRow


def make_entry(umaban: str = "1", **kw) -> EntryRow:
    base = dict(
        umaban=umaban,
        wakuban="1",
        ketto_num="0000000001",
        bamei="テスト馬",
        sex_cd="1",
        barei="4",
        futan_kg=57.0,
        kisyu_code="00001",
        tozai_cd="1",
        chokyosi_code="00001",
    )
    base.update(kw)
    return EntryRow(**base)


def make_result(umaban: str = "1", **kw) -> ResultRow:
    base = dict(
        entry=make_entry(umaban),
        ijyo_cd="0",
        kakutei_jyuni=1,
        dochaku_kubun="0",
        dochaku_tosu=None,
        time_sec=120.5,
        margin_sec=0.0,
        jyuni_1c=3,
        jyuni_2c=3,
        jyuni_3c=2,
        jyuni_4c=2,
        last_3f_sec=34.1,
        ba_taijyu_kg=486,
        zogen_sa_kg=-2,
        odds=3.4,
        ninki=1,
    )
    base.update(kw)
    return ResultRow(**base)


def make_outcome(**kw) -> RaceOutcome:
    base = dict(race_id="2026032106050311", winners=("1",), starters=("1", "2", "3"), excluded=())
    base.update(kw)
    return RaceOutcome(**base)


# --- top_n_hit / top1_win ----------------------------------------------------

def test_top_n_hit_when_winner_in_top_n():
    outcome = make_outcome(winners=("1",))
    assert top_n_hit(["3", "1", "2"], outcome, 2) is True


def test_top_n_miss_when_winner_below():
    outcome = make_outcome(winners=("1",))
    assert top_n_hit(["3", "1", "2"], outcome, 1) is False


def test_top_n_hit_with_dead_heat_any_winner():
    outcome = make_outcome(winners=("1", "5"), starters=("1", "2", "3", "4", "5"))
    assert top_n_hit(["3", "1", "2"], outcome, 2) is True


def test_top1_win_only_for_first_pick():
    outcome = make_outcome(winners=("2",))
    assert top1_win(["1", "2", "3"], outcome) is False
    assert top1_win(["2", "1", "3"], outcome) is True


# --- brier --------------------------------------------------------------------

def test_brier_hand_computed():
    """3頭 p=(0.5,0.3,0.2)、勝ち馬が1番目 → (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.38。"""
    probabilities = {"1": 0.5, "2": 0.3, "3": 0.2}
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"))
    assert brier(probabilities, outcome) == pytest.approx(0.38)


def test_brier_dead_heat_splits_outcome():
    """勝ち馬2頭なら各 y=0.5:
    (0.4-0.5)^2 + (0.4-0.5)^2 + (0.2-0)^2 = 0.01 + 0.01 + 0.04 = 0.06。
    """
    probabilities = {"1": 0.4, "2": 0.4, "3": 0.2}
    outcome = make_outcome(winners=("1", "2"), starters=("1", "2", "3"))
    assert brier(probabilities, outcome) == pytest.approx(0.06)


def test_brier_raises_when_a_starter_has_no_probability():
    """予想が starters の馬を1頭でも落としていたら、0.0 で静かに採点せず
    OutcomeError にする（fix1 修正3。見たい欠陥のため）。メッセージに
    欠けた馬番が入る。
    """
    probabilities = {"1": 0.6, "3": 0.4}  # "2" が無い
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"))

    with pytest.raises(OutcomeError) as excinfo:
        brier(probabilities, outcome)

    assert "2" in str(excinfo.value)


def test_brier_ignores_extra_probabilities_for_excluded_horses():
    """probabilities の方が starters より多いのは正常（前日の予想後に取消・
    除外になった馬）。例外にせず starters だけで計算する。
    """
    probabilities = {"1": 0.5, "2": 0.3, "3": 0.2, "9": 0.4}  # "9" は starters に無い
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"), excluded=("9",))

    assert brier(probabilities, outcome) == pytest.approx(0.38)


def test_brier_does_not_renormalise():
    """取消馬 (starters に無い "3") に確率 0.4 が乗っていて合計が 1.0 未満でも、
    再正規化した値ではなくそのままの値で採点する。

    starters=("1","2")、winners=("1",)。y=(1,0)。
    (0.3-1)^2 + (0.3-0)^2 = 0.49 + 0.09 = 0.58。
    再正規化（0.3/0.6=0.5 ずつ）した場合の (0.5-1)^2+(0.5-0)^2=0.5 とは異なる値。
    """
    probabilities = {"1": 0.3, "2": 0.3, "3": 0.4}
    outcome = make_outcome(winners=("1",), starters=("1", "2"), excluded=("3",))

    assert brier(probabilities, outcome) == pytest.approx(0.58)


# --- log_loss -------------------------------------------------------------

def test_log_loss_hand_computed():
    """勝ち馬の p=0.25 → -log(0.25) ≈ 1.3863。"""
    probabilities = {"1": 0.25, "2": 0.5, "3": 0.25}
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"))
    assert log_loss(probabilities, outcome) == pytest.approx(1.3863, abs=1e-4)


def test_log_loss_dead_heat_sums_probabilities():
    """同着2頭の p が 0.2 と 0.3 なら合計0.5 → -log(0.5)。"""
    probabilities = {"1": 0.2, "2": 0.3, "3": 0.5}
    outcome = make_outcome(winners=("1", "2"), starters=("1", "2", "3"))
    assert log_loss(probabilities, outcome) == pytest.approx(-math.log(0.5))


def test_log_loss_clips_zero_probability():
    """勝ち馬の p=0.0 でも eps で clip して有限の値になる（inf にならない）。"""
    probabilities = {"1": 0.0, "2": 1.0}
    outcome = make_outcome(winners=("1",), starters=("1", "2"))
    result = log_loss(probabilities, outcome)
    assert math.isfinite(result)
    assert result == pytest.approx(-math.log(1e-12))


def test_log_loss_is_clipped_when_winner_probability_is_zero():
    """勝ち馬の確率が 0（合計が eps 以下）のとき clip されたと判定する
    （task-19-21 fix 決定9）。
    """
    probabilities = {"1": 0.0, "2": 1.0}
    outcome = make_outcome(winners=("1",), starters=("1", "2"))

    assert log_loss_is_clipped(probabilities, outcome) is True


def test_log_loss_is_not_clipped_for_a_normal_probability():
    probabilities = {"1": 0.25, "2": 0.5, "3": 0.25}
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"))

    assert log_loss_is_clipped(probabilities, outcome) is False


def test_log_loss_raises_when_a_starter_has_no_probability():
    """brier と同じ理由・同じ判定（fix1 修正3）。実際の和は winners だけを
    見るが、非勝ち馬の欠落も starters 側の確認で検出する。
    """
    probabilities = {"1": 0.6, "3": 0.4}  # "2" が無い
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"))

    with pytest.raises(OutcomeError) as excinfo:
        log_loss(probabilities, outcome)

    assert "2" in str(excinfo.value)


# --- expected_top_n ---------------------------------------------------------

def test_expected_top_n_sums_top_probabilities():
    probabilities = {"1": 0.4, "2": 0.35, "3": 0.25}
    assert expected_top_n(probabilities, ["1", "2", "3"], 2) == pytest.approx(0.75)


def test_expected_top_n_is_unaffected_by_ties():
    """同じ確率分布なら、その行の boundary_tied（同程度の組の切り方）を
    変えても expected_top_n_mean は変わらない。expected_top_n は確率と
    ranking/n だけで決まり、tie_group を一切見ないため（将来これを崩す
    変更を止めるための固定テスト。task-19-21-supplement 決定1）。
    """
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"))
    ranking = ["1", "2", "3"]
    probabilities = {"1": 0.5, "2": 0.3, "3": 0.2}
    row_tied = {
        "skipped": False,
        "boundary_tied": True,
        "excluded_probability_mass": 0.0,
        "ranking": ranking,
        "probabilities": probabilities,
        "outcome": outcome,
    }
    row_not_tied = dict(row_tied, boundary_tied=False)

    report_tied = aggregate([row_tied], n=2)
    report_not_tied = aggregate([row_not_tied], n=2)

    assert report_tied.expected_top_n_mean == report_not_tied.expected_top_n_mean
    assert report_tied.n_boundary_tied == 1
    assert report_not_tied.n_boundary_tied == 0


# --- outcome_from_results ------------------------------------------------

def test_outcome_excludes_scratched_from_starters():
    """ijyo_cd 1/2/3（取消・発走除外・競走除外）は starters に入らず excluded に入る。"""
    results = [
        make_result("1", ijyo_cd="0", kakutei_jyuni=1),
        make_result("2", ijyo_cd="1", kakutei_jyuni=None),
        make_result("3", ijyo_cd="2", kakutei_jyuni=None),
        make_result("4", ijyo_cd="3", kakutei_jyuni=None),
        make_result("5", ijyo_cd="0", kakutei_jyuni=2),
    ]
    outcome = outcome_from_results("R1", results)
    assert outcome is not None
    assert outcome.starters == ("1", "5")
    assert outcome.excluded == ("2", "3", "4")


def test_outcome_keeps_retired_in_starters():
    """ijyo_cd 4/5（競走中止・失格）は starters に残るが winners には入らない（spec §12.2）。"""
    results = [
        make_result("1", ijyo_cd="0", kakutei_jyuni=1),
        make_result("2", ijyo_cd="4", kakutei_jyuni=None),
        make_result("3", ijyo_cd="5", kakutei_jyuni=1),
    ]
    outcome = outcome_from_results("R1", results)
    assert outcome is not None
    assert set(outcome.starters) == {"1", "2", "3"}
    assert outcome.winners == ("1",)


def test_outcome_keeps_demoted_as_possible_winner():
    """ijyo_cd="7"（降着）でも kakutei_jyuni=1 なら勝ち馬。"""
    results = [
        make_result("1", ijyo_cd="7", kakutei_jyuni=1),
        make_result("2", ijyo_cd="0", kakutei_jyuni=2),
    ]
    outcome = outcome_from_results("R1", results)
    assert outcome is not None
    assert outcome.winners == ("1",)


def test_outcome_none_when_no_winner():
    results = [
        make_result("1", ijyo_cd="4", kakutei_jyuni=None),
        make_result("2", ijyo_cd="0", kakutei_jyuni=2),
        make_result("3", ijyo_cd="0", kakutei_jyuni=3),
    ]
    assert outcome_from_results("R1", results) is None


def test_outcome_none_when_single_starter():
    results = [make_result("1", ijyo_cd="0", kakutei_jyuni=1)]
    assert outcome_from_results("R1", results) is None


def test_empty_results_returns_none():
    """results が空（そもそも結果データが無い）は「採点できない」という
    勝ち馬なし／1頭立てと同じ状況なので None（OutcomeError にはしない。fix1）。
    """
    assert outcome_from_results("R1", []) is None


def test_winner_not_among_starters_raises():
    """勝ち馬が starters に含まれていないのは矛盾したデータ（fix1）。

    ResultRow.is_winner は can_win（counts_as_starter を含む）を要求するため
    通常の ResultRow からは作れない状態なので、duck typing で最小限の
    フェイクを直接渡して防御コードそのものを検証する。
    """
    class _FakeEntry:
        def __init__(self, umaban):
            self.umaban = umaban

    class _FakeResult:
        def __init__(self, umaban, *, counts_as_starter, is_winner):
            self.entry = _FakeEntry(umaban)
            self.counts_as_starter = counts_as_starter
            self.is_winner = is_winner

    results = [
        _FakeResult("1", counts_as_starter=False, is_winner=True),
        _FakeResult("2", counts_as_starter=True, is_winner=False),
        _FakeResult("3", counts_as_starter=True, is_winner=False),
    ]

    with pytest.raises(OutcomeError) as excinfo:
        outcome_from_results("R1", results)

    assert "1" in str(excinfo.value)


def test_duplicate_umaban_raises():
    results = [
        make_result("1", ijyo_cd="0", kakutei_jyuni=1),
        make_result("1", ijyo_cd="0", kakutei_jyuni=2),
        make_result("2", ijyo_cd="0", kakutei_jyuni=3),
    ]

    with pytest.raises(OutcomeError) as excinfo:
        outcome_from_results("R1", results)

    assert "1" in str(excinfo.value)


# --- calibration_bins -------------------------------------------------------

def test_calibration_bins_group_by_predicted_probability():
    pairs = [(0.05, 0.0), (0.15, 0.0), (0.15, 1.0), (0.95, 1.0)]
    result = calibration_bins(pairs)

    assert len(result) == 3
    by_bin = {row["bin"]: row for row in result}

    assert by_bin[0]["count"] == 1
    assert by_bin[0]["predicted_mean"] == pytest.approx(0.05)
    assert by_bin[0]["actual_rate"] == pytest.approx(0.0)

    assert by_bin[1]["count"] == 2
    assert by_bin[1]["predicted_mean"] == pytest.approx(0.15)
    assert by_bin[1]["actual_rate"] == pytest.approx(0.5)

    assert by_bin[9]["count"] == 1
    assert by_bin[9]["predicted_mean"] == pytest.approx(0.95)
    assert by_bin[9]["actual_rate"] == pytest.approx(1.0)


def test_calibration_bins_skip_empty_bins():
    pairs = [(0.5, 1.0), (0.55, 0.0)]
    result = calibration_bins(pairs)
    assert len(result) == 1
    assert result[0]["bin"] == 5
    assert result[0]["count"] == 2


# --- aggregate / compare -----------------------------------------------------

def _valid_row(*, boundary_tied: bool = False, excluded_probability_mass: float = 0.0) -> dict:
    outcome = make_outcome(winners=("1",), starters=("1", "2", "3"))
    return {
        "skipped": False,
        "boundary_tied": boundary_tied,
        "excluded_probability_mass": excluded_probability_mass,
        "ranking": ["1", "2", "3"],
        "probabilities": {"1": 0.5, "2": 0.3, "3": 0.2},
        "outcome": outcome,
    }


def test_aggregate_counts_skipped_races():
    rows = [
        _valid_row(),
        {"skipped": True, "boundary_tied": False, "excluded_probability_mass": 0.0},
    ]
    report = aggregate(rows, n=2)
    assert report.n_skipped == 1
    assert report.n_races == 1


def test_aggregate_records_n_used():
    report = aggregate([_valid_row()], n=3)
    assert report.n == 3


def test_aggregate_counts_tied_boundaries():
    rows = [
        _valid_row(boundary_tied=True),
        _valid_row(boundary_tied=False),
        {"skipped": True, "boundary_tied": True, "excluded_probability_mass": 0.0},
    ]
    report = aggregate(rows, n=2)
    assert report.n_boundary_tied == 2


def test_aggregate_rejects_a_row_without_the_flag():
    row = _valid_row()
    del row["boundary_tied"]
    with pytest.raises(KeyError):
        aggregate([row], n=2)


def test_aggregate_counts_excluded_probability_mass():
    rows = [
        _valid_row(excluded_probability_mass=0.1),
        _valid_row(excluded_probability_mass=0.0),
        {"skipped": True, "boundary_tied": False, "excluded_probability_mass": 0.2},
    ]
    report = aggregate(rows, n=2)
    assert report.n_excluded_with_probability == 2


def test_aggregate_rejects_a_row_without_excluded_mass():
    row = _valid_row()
    del row["excluded_probability_mass"]
    with pytest.raises(KeyError):
        aggregate([row], n=2)


def test_aggregate_counts_winner_probability_clipped():
    """log_loss が clip されたレース数を数える（task-19-21 fix 決定9）。"""
    clipped_row = _valid_row()
    clipped_row["probabilities"] = {"1": 0.0, "2": 0.7, "3": 0.3}  # 勝ち馬 "1" の確率が 0
    normal_row = _valid_row()  # 勝ち馬 "1" の確率 0.5（clip されない）

    report = aggregate([clipped_row, normal_row], n=2)

    assert report.n_winner_probability_clipped == 1


def test_compare_reports_differences():
    baseline = MetricsReport(
        n_races=10, n_skipped=0, top_n_hit_rate=0.4, top1_win_rate=0.2,
        brier=0.3, log_loss=1.0, expected_top_n_mean=0.45, calibration=[], n=3,
        n_boundary_tied=1, n_excluded_with_probability=0,
        n_winner_probability_clipped=0,
    )
    model = MetricsReport(
        n_races=10, n_skipped=0, top_n_hit_rate=0.5, top1_win_rate=0.25,
        brier=0.25, log_loss=0.9, expected_top_n_mean=0.5, calibration=[], n=3,
        n_boundary_tied=4, n_excluded_with_probability=0,
        n_winner_probability_clipped=0,
    )
    diff = compare(baseline, model)
    assert diff["top_n_hit_rate"] == pytest.approx(0.1)
    assert diff["top1_win_rate"] == pytest.approx(0.05)
    assert diff["brier"] == pytest.approx(-0.05)
    assert diff["log_loss"] == pytest.approx(-0.1)
    assert diff["expected_top_n_mean"] == pytest.approx(0.05)


def test_compare_reports_tied_boundaries_for_both():
    baseline = MetricsReport(
        n_races=10, n_skipped=0, top_n_hit_rate=0.4, top1_win_rate=0.2,
        brier=0.3, log_loss=1.0, expected_top_n_mean=0.45, calibration=[], n=3,
        n_boundary_tied=5, n_excluded_with_probability=0,
        n_winner_probability_clipped=0,
    )
    model = MetricsReport(
        n_races=10, n_skipped=0, top_n_hit_rate=0.5, top1_win_rate=0.25,
        brier=0.25, log_loss=0.9, expected_top_n_mean=0.5, calibration=[], n=3,
        n_boundary_tied=2, n_excluded_with_probability=0,
        n_winner_probability_clipped=0,
    )
    diff = compare(baseline, model)
    assert diff["n_boundary_tied_baseline"] == 5
    assert diff["n_boundary_tied_model"] == 2


# --- fix6 ③: 行が1件も無ければ「測っていない」 ----------------------------------

def test_empty_report_does_not_claim_zero_boundary_ties():
    """行が1件も無い集計は n_boundary_tied を None にする（0 にしない）。

    baseline のみの実行では model 側の行が1件も無く、0 と書くと「境界が同程度
    だったレースは0件だった」＝測ったうえでの0件を主張してしまう。§1 の決定
    （測っていない尺度について件数を主張しない）に反する。
    """
    report = aggregate([], n=3)

    assert report.n_boundary_tied is None
    assert report.n_races == 0
