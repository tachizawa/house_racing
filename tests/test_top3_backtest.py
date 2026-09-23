from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path

import pytest

from jev_client import describe_error
from keiba.data.repository import Repository, ResultRepository
from keiba.data.store import DataStore
from keiba.evaluate import top3_backtest as tbt
from tests.fixtures import csv_data as fx

# scripts/ は keiba パッケージの外にあるため、tests/test_backtest.py と同じ方法
# (sys.path への追加) で import する（backtest_top3.py 冒頭のコメント参照）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import backtest_top3 as backtest_top3_cli  # noqa: E402

TUNING_RACE = "2026030106010101"   # 締切 2026-02-28
FINAL_RACE = "2026032106011101"    # 締切 2026-03-20


def _dataset_dir(tmp_path):
    races = [fx.make_race(race_id=TUNING_RACE, hasso_at="2026-03-01T15:00:00+09:00"),
             fx.make_race(race_id=FINAL_RACE, hasso_at="2026-03-21T15:25:00+09:00")]
    race_horses = []
    for race_id in (TUNING_RACE, FINAL_RACE):
        for i in range(1, 6):
            race_horses.append(fx.make_race_horse(
                race_id=race_id, umaban=f"{i:02d}", ketto_num=f"{race_id}{i}",
                bamei=f"ウマ{race_id[-4:]}{i}", result_kakutei_jyuni=str(i)))
    return fx.write_dataset(tmp_path, races, race_horses, [])


def _repos(tmp_path):
    store = DataStore.open_current(_dataset_dir(tmp_path))
    return Repository(store), ResultRepository(store)


class Client:
    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **kw):
        self.calls += 1
        umabans = list(state["horses"])
        nouls = {f"top3_{u}": types.SimpleNamespace(noul=0.9 - 0.1 * i)
                 for i, u in enumerate(umabans)}
        probs = {u: 1 / len(umabans) for u in umabans}
        return types.SimpleNamespace(
            nouls=nouls, choices={"winner": types.SimpleNamespace(
                choice=umabans[0], probabilities=probs, confidence=0.5)},
            model="jev-1.13.0", usage=types.SimpleNamespace(input_tokens=100))


PACE_DEF = {"fast_max": 35.0, "standard_max": 36.0, "stratum": {"track_kind": "turf"},
            "race_count": 40, "note": "n"}


class PlanBClient:
    """案B: 1回目（pace）と2回目（top3/winner）で違う応答を返す。"""

    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **kw):
        self.calls += 1
        if "pace" in questions:
            return types.SimpleNamespace(
                nouls={}, choices={"pace": types.SimpleNamespace(
                    choice="fast", probabilities={"fast": 0.5, "standard": 0.3, "slow": 0.2},
                    confidence=0.4)},
                model="jev-1.13.0", usage=types.SimpleNamespace(input_tokens=100))
        umabans = list(state["horses"])
        nouls = {f"top3_{u}": types.SimpleNamespace(noul=0.9 - 0.1 * i)
                 for i, u in enumerate(umabans)}
        probs = {u: 1 / len(umabans) for u in umabans}
        return types.SimpleNamespace(
            nouls=nouls, choices={"winner": types.SimpleNamespace(
                choice=umabans[0], probabilities=probs, confidence=0.5)},
            model="jev-1.13.0", usage=types.SimpleNamespace(input_tokens=100))


def _sdk_error():
    """課金の理由になる SDK 例外。レート制限が構成できない環境なら
    タイムアウトにフォールバックする（既存テストと同じ方針）。"""
    import typesafe_sdk as ts
    try:
        return ts.TypeSafeRateLimitError(429, {"error": "rate limited"}, {},
                                         message="rate limited", endpoint="/v1/systemone")
    except TypeError:
        return ts.TypeSafeAPITimeoutError(10.0)


class FailingClient:
    """常に SDK 例外を送出する（jev_failed が連続する状況の再現用）。"""

    def __init__(self):
        self.calls = 0

    def system_one(self, state, questions, **kw):
        self.calls += 1
        raise _sdk_error()


def _many_races(n: int, *, start_day: int = 1):
    """発走日を1日ずつずらした n レース分の races.csv / race_horses.csv の行。
    すべて同じ月内・別日にして、時系列順を保ったまま複数レースを作る。"""
    races, race_horses = [], []
    for i in range(n):
        day = start_day + i
        race_id = f"202603{day:02d}06010101"
        for h in range(1, 4):
            race_horses.append(fx.make_race_horse(
                race_id=race_id, umaban=f"{h:02d}", ketto_num=f"{race_id}{h}",
                bamei=f"ウマ{day}{h}", result_kakutei_jyuni=str(h)))
        races.append(fx.make_race(race_id=race_id, hasso_at=f"2026-03-{day:02d}T15:00:00+09:00"))
    return races, race_horses


def _run(tmp_path, *, methods=("python", "plan_a"), allow_final=False, client=None):
    repo, results = _repos(tmp_path)
    races = [results.race(TUNING_RACE), results.race(FINAL_RACE)]
    return tbt.run(races, repo, results, client or Client(), methods=methods, n=3,
                   tuning_until=date(2026, 3, 10), allow_final=allow_final,
                   max_recent_runs=5, save_records=False, data_dir=tmp_path)


def test_final_period_is_locked_by_default(tmp_path):
    client = Client()
    rows = _run(tmp_path, client=client)
    by_id = {r.race_id: r for r in rows}
    assert by_id[TUNING_RACE].period == "tuning"
    assert by_id[FINAL_RACE].skipped == "final_period_locked"
    assert client.calls == 1  # final のレースには Jev を呼んでいない


def test_allow_final_runs_final_period(tmp_path):
    rows = _run(tmp_path, allow_final=True)
    assert {r.race_id: r.skipped for r in rows}[FINAL_RACE] is None


def test_series_include_winner_same_state(tmp_path):
    row = _run(tmp_path)[0]
    assert set(row.series) == {"python", "plan_a", "winner_same_state_a"}
    assert row.series["plan_a"]["ranking"][:3] == ["01", "02", "03"]
    assert row.series["python"]["probabilities"] is not None  # 一様確率


def test_python_only_does_not_call_jev(tmp_path):
    client = Client()
    _run(tmp_path, methods=("python",), allow_final=True, client=client)
    assert client.calls == 0


def test_state_too_large_is_skipped_with_reason(tmp_path, monkeypatch):
    from keiba import config
    monkeypatch.setattr(config, "STATE_MAX_CHARS", 10)
    rows = _run(tmp_path, methods=("plan_a",))
    assert rows[0].skipped == "state_too_large"


def test_incomplete_answer_is_skipped_with_reason(tmp_path):
    class Partial(Client):
        def system_one(self, state, questions, **kw):
            response = super().system_one(state, questions, **kw)
            response.nouls.pop(next(iter(response.nouls)))
            return response
    rows = _run(tmp_path, methods=("plan_a",), client=Partial())
    assert rows[0].skipped == "incomplete_answer"
    # 応答は返ってきている＝課金は発生した。トークン数は不明のまま
    # failed_requests で「請求額が読めない課金がある」ことを可視化する
    # （fix round 1 Minor 5）。
    rep = tbt.report(rows, n=3, run_info={"note": "test"})
    assert rep["usage"]["plan_a"]["failed_requests"] == 1


def test_plan_b_skips_when_pace_definition_unavailable(tmp_path):
    repo, results = _repos(tmp_path)
    rows = tbt.run([results.race(TUNING_RACE)], repo, results, Client(),
                   methods=("plan_b",), n=3, tuning_until=date(2026, 3, 10),
                   allow_final=False, max_recent_runs=5, save_records=False,
                   data_dir=tmp_path, pace_definition_for=lambda ri: None)
    assert rows[0].skipped == "pace_unavailable"


def test_plan_b_incomplete_top3_answer_is_skipped_with_failed_request(tmp_path):
    # pace の答えは正常に返るが、top3 の答えが1頭分欠ける
    # （pace のリクエストは送信済み＝課金は発生している）。
    class PartialPlanB(PlanBClient):
        def system_one(self, state, questions, **kw):
            response = super().system_one(state, questions, **kw)
            if "pace" not in questions:
                response.nouls.pop(next(iter(response.nouls)))
            return response
    repo, results = _repos(tmp_path)
    rows = tbt.run([results.race(TUNING_RACE)], repo, results, PartialPlanB(),
                   methods=("plan_b",), n=3, tuning_until=date(2026, 3, 10),
                   allow_final=False, max_recent_runs=5, save_records=False,
                   data_dir=tmp_path, pace_definition_for=lambda ri: PACE_DEF)
    assert rows[0].skipped == "incomplete_answer"
    rep = tbt.report(rows, n=3, run_info={"note": "test"})
    assert rep["usage"]["plan_b"]["failed_requests"] == 1


def test_plan_b_records_pace_pair(tmp_path):
    repo, results = _repos(tmp_path)
    rows = tbt.run([results.race(TUNING_RACE)], repo, results, PlanBClient(),
                   methods=("plan_b",), n=3, tuning_until=date(2026, 3, 10),
                   allow_final=False, max_recent_runs=5, save_records=False,
                   data_dir=tmp_path, pace_definition_for=lambda ri: PACE_DEF)
    assert rows[0].skipped is None
    assert rows[0].pace_pair == ("fast", "standard")  # 実測 first_3f_sec=35.2 は standard


def test_report_pace_accuracy_for_plan_b(tmp_path):
    from keiba.evaluate.top3_backtest import pace_accuracy
    rows_pace = [("fast", "fast"), ("slow", "fast"), ("standard", None)]
    result = pace_accuracy(rows_pace)
    assert result == {"n_scored": 2, "n_unscorable": 1, "argmax_accuracy": 0.5}


def test_report_has_per_series_metrics_and_skips(tmp_path):
    rows = _run(tmp_path)
    rep = tbt.report(rows, n=3, run_info={"note": "test"})
    assert rep["skip_breakdown"]["final_period_locked"] == 1
    assert rep["by_period"]["tuning"]["series"]["plan_a"]["mean_hits"] == 3.0
    assert "plan_a_vs_python" in rep["by_period"]["tuning"]["comparisons"]
    assert rep["usage"]["plan_a"]["input_tokens"] == 100
    assert rep["versions"]["eval_rules_version"] == "1"


def test_report_skip_details_counts(tmp_path):
    rows = _run(tmp_path, methods=("plan_a",), client=FailingClient())
    rep = tbt.report(rows, n=3, run_info={"note": "test"})
    # final_period_locked（FINAL_RACE）は例外由来ではないので detail が無く、
    # skip_details には現れない。jev_failed（TUNING_RACE）だけが1件。
    assert sum(rep["skip_details"].values()) == 1
    key = next(iter(rep["skip_details"]))
    assert key.startswith("plan_a: ")
    assert len(key) <= len(f"plan_a: {type(_sdk_error()).__name__}: ") + 80


# ===============================================================================
# 連続失敗のサーキットブレーカー
# ===============================================================================


def _race_rows(repo, results, races):
    return [results.race(r["race_id"]) for r in races]


def test_breaker_stops_after_n_consecutive_failures(tmp_path):
    races, race_horses = _many_races(7)
    data_dir = fx.write_dataset(tmp_path, races, race_horses, [])
    store = DataStore.open_current(data_dir)
    repo, results = Repository(store), ResultRepository(store)
    race_rows = _race_rows(repo, results, races)
    client = FailingClient()

    with pytest.raises(tbt.ConsecutiveJevFailures) as excinfo:
        tbt.run(race_rows, repo, results, client, methods=("plan_a",), n=3,
               tuning_until=date(2026, 3, 31), allow_final=False, max_recent_runs=5,
               save_records=False, data_dir=tmp_path, max_consecutive_jev_failures=3)

    breaker = excinfo.value
    assert len(breaker.rows) == 3
    assert [r.skipped for r in breaker.rows] == ["jev_failed"] * 3
    assert client.calls == 3  # 4レース目以降には送っていない
    assert breaker.skip_detail == breaker.rows[-1].skip_detail


def test_breaker_resets_after_a_success(tmp_path):
    races, race_horses = _many_races(5)
    data_dir = fx.write_dataset(tmp_path, races, race_horses, [])
    store = DataStore.open_current(data_dir)
    repo, results = Repository(store), ResultRepository(store)
    race_rows = _race_rows(repo, results, races)

    class ScriptedClient:
        """1・2・4・5回目は失敗、3回目だけ成功する。

        ``Client`` を継承すると ``system_one`` 自身も ``self.calls`` を数える
        ため、委譲するとここでの加算と二重になる。数える主体をこのクラス
        だけにするため、成功時は使い捨ての ``Client()`` に委譲する。
        """

        def __init__(self):
            self.calls = 0

        def system_one(self, state, questions, **kw):
            self.calls += 1
            if self.calls in (1, 2, 4, 5):
                raise _sdk_error()
            return Client().system_one(state, questions, **kw)

    client = ScriptedClient()
    rows = tbt.run(race_rows, repo, results, client, methods=("plan_a",), n=3,
                   tuning_until=date(2026, 3, 31), allow_final=False, max_recent_runs=5,
                   save_records=False, data_dir=tmp_path, max_consecutive_jev_failures=3)

    # 3回目の成功で連続数が0に戻るので、limit=3 に達する前に5レースとも終わる。
    assert [r.skipped for r in rows] == ["jev_failed", "jev_failed", None,
                                         "jev_failed", "jev_failed"]


def test_breaker_disabled_with_none(tmp_path):
    races, race_horses = _many_races(6)
    data_dir = fx.write_dataset(tmp_path, races, race_horses, [])
    store = DataStore.open_current(data_dir)
    repo, results = Repository(store), ResultRepository(store)
    race_rows = _race_rows(repo, results, races)
    client = FailingClient()

    rows = tbt.run(race_rows, repo, results, client, methods=("plan_a",), n=3,
                   tuning_until=date(2026, 3, 31), allow_final=False, max_recent_runs=5,
                   save_records=False, data_dir=tmp_path, max_consecutive_jev_failures=None)

    assert len(rows) == 6
    assert all(r.skipped == "jev_failed" for r in rows)


def test_dry_run_reports_state_sizes_without_client(tmp_path):
    repo, results = _repos(tmp_path)
    races = [results.race(TUNING_RACE)]
    out = tbt.dry_run(races, repo, n=3, max_recent_runs=5)
    assert out["n_races"] == 1
    assert out["max_state_chars"] > 0
    assert out["over_limit"] == []


# ===============================================================================
# scripts/backtest_top3.py: --final のガード（fix round 1 Critical 1-3）
#
# すべて --methods python（Jev を呼ばない・client 不要）で main() を直接呼ぶ。
# --data-dir は tmp_path/data、--out は tmp_path/out に分けて、CLI の argparse
# を通した経路そのものを確認する（_final_guard_problem / _append_final_log を
# 単体で呼ぶのではなく main() 経由にするのは、判定と記録の呼び出し順序――
# クライアント生成の後、最終期間のレースが1件以上あるときだけ記録する――を
# 配線ごと確認するため）。
# ===============================================================================


def _final_cli_args(data_dir, out_dir, *, tuning_until="20260310", final=True,
                    reason=None):
    args = ["--data-dir", str(data_dir), "--from", "20260201", "--to", "20260401",
            "--tuning-until", tuning_until, "--methods", "python",
            "--out", str(out_dir), "--no-save-records"]
    if final:
        args.append("--final")
    if reason:
        args += ["--final-rerun-reason", reason]
    return args


def test_final_run_writes_one_log_line(tmp_path):
    data_dir = _dataset_dir(tmp_path / "data")
    args = _final_cli_args(data_dir, tmp_path / "out")

    rc = backtest_top3_cli.main(args)

    assert rc == 0
    log = data_dir / "top3" / "final_runs.jsonl"
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["tuning_until"] == "20260310"
    assert entry["n_final_races"] == 1


def test_second_final_run_without_reason_is_refused(tmp_path):
    data_dir = _dataset_dir(tmp_path / "data")
    backtest_top3_cli.main(_final_cli_args(data_dir, tmp_path / "out1"))
    log = data_dir / "top3" / "final_runs.jsonl"
    before = log.read_text(encoding="utf-8")

    rc = backtest_top3_cli.main(_final_cli_args(data_dir, tmp_path / "out2"))

    assert rc == 2
    assert log.read_text(encoding="utf-8") == before


def test_final_rerun_with_reason_appends(tmp_path):
    data_dir = _dataset_dir(tmp_path / "data")
    backtest_top3_cli.main(_final_cli_args(data_dir, tmp_path / "out1"))

    rc = backtest_top3_cli.main(
        _final_cli_args(data_dir, tmp_path / "out2", reason="再検証"))

    assert rc == 0
    log = data_dir / "top3" / "final_runs.jsonl"
    lines = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2


def test_non_final_run_past_pinned_boundary_is_refused(tmp_path):
    """--final を付けていなくても、境界（最初の --final の --tuning-until）
    より後の --tuning-until を指定する実行は拒否される（fix round 1
    Critical 2: --tuning-until をずらすだけの回避を防ぐ）。
    """
    data_dir = _dataset_dir(tmp_path / "data")
    backtest_top3_cli.main(_final_cli_args(data_dir, tmp_path / "out1"))

    rc = backtest_top3_cli.main(_final_cli_args(
        data_dir, tmp_path / "out2", tuning_until="20260315", final=False))

    assert rc == 2


def test_dry_run_final_writes_no_log(tmp_path):
    data_dir = _dataset_dir(tmp_path / "data")
    args = _final_cli_args(data_dir, tmp_path / "out") + ["--dry-run"]

    rc = backtest_top3_cli.main(args)

    assert rc == 0
    assert not (data_dir / "top3" / "final_runs.jsonl").exists()


def test_final_run_with_no_final_races_writes_no_log(tmp_path):
    """サンプルに最終期間のレースが1件も無い --final 実行は記録しない
    （fix round 1 Critical 3）。tuning_until を両レースの締切より後にすると、
    TUNING_RACE・FINAL_RACE ともに tuning 区分になる。
    """
    data_dir = _dataset_dir(tmp_path / "data")
    args = _final_cli_args(data_dir, tmp_path / "out", tuning_until="20260401")

    rc = backtest_top3_cli.main(args)

    assert rc == 0
    assert not (data_dir / "top3" / "final_runs.jsonl").exists()


# ===============================================================================
# final review の修正（I-4 / I-5 / M-6 / I-1 / save_records）
# ===============================================================================


def _run_b(tmp_path, client, *, methods=("plan_b",), pace=PACE_DEF, save_records=False):
    repo, results = _repos(tmp_path)
    return tbt.run([results.race(TUNING_RACE)], repo, results, client,
                   methods=methods, n=3, tuning_until=date(2026, 3, 10),
                   allow_final=False, max_recent_runs=5, save_records=save_records,
                   data_dir=tmp_path / "store", pace_definition_for=lambda ri: pace)


def test_pace_unavailable_skips_race_before_any_jev_call(tmp_path):
    # 案Aを先に課金してから案Bで pace が無いと分かる、を防ぐ（I-4a）。
    client = PlanBClient()
    rows = _run_b(tmp_path, client, methods=("python", "plan_a", "plan_b"), pace=None)
    assert client.calls == 0
    assert rows[0].skipped == "pace_unavailable"


def test_plan_b_partial_failure_keeps_pace_usage(tmp_path):
    class PartialPlanB(PlanBClient):
        def system_one(self, state, questions, **kw):
            response = super().system_one(state, questions, **kw)
            if "pace" not in questions:
                response.nouls.pop(next(iter(response.nouls)))
            return response
    rows = _run_b(tmp_path, PartialPlanB())
    assert rows[0].skipped == "incomplete_answer"
    usage = rows[0].usage["plan_b"]
    assert usage["input_tokens"] == 100  # pace の分
    assert usage["requests"] == 2
    assert usage["failed_requests"] == 1


def test_jev_failed_skip_detail_has_sdk_exception_class(tmp_path):
    # Task: 失敗の理由を記録する。JevRequestError は original（SDK の元例外）の
    # クラス名を使う（jev_client.JevRequestError.original 経由）。
    error = _sdk_error()
    rows = _run(tmp_path, methods=("plan_a",), client=FailingClient())
    tuning = {r.race_id: r for r in rows}[TUNING_RACE]
    assert tuning.skipped == "jev_failed"
    assert tuning.skip_detail == f"plan_a: {type(error).__name__}: {describe_error(error)}"
    assert len(tuning.skip_detail) <= 300


def test_plan_b_partial_failure_maps_jev_error(tmp_path):
    import typesafe_sdk as ts

    class FailingSecond(PlanBClient):
        def system_one(self, state, questions, **kw):
            if "pace" not in questions:
                raise ts.TypeSafeAPITimeoutError(10.0)
            return super().system_one(state, questions, **kw)
    rows = _run_b(tmp_path, FailingSecond())
    assert rows[0].skipped == "jev_failed"
    assert rows[0].usage["plan_b"]["input_tokens"] == 100
    assert rows[0].usage["plan_b"]["failed_requests"] == 1


def test_plan_b_partial_failure_before_second_send_is_not_a_failed_request(tmp_path,
                                                                            monkeypatch):
    # 2回目を送る前に止めた（StateTooLargeError）なら requests 1 / failed 0。
    from keiba.predict import top3 as top3_mod

    def fake_run_plan_b(client, dossier, *, pace_definition):
        from keiba.features.dossier import StateTooLargeError
        raise top3_mod.PlanBPartialFailure(StateTooLargeError(10, 5), input_tokens=77,
                                           requests_sent=1, seconds=0.5)
    monkeypatch.setattr(tbt.top3, "run_plan_b", fake_run_plan_b)
    rows = _run_b(tmp_path, PlanBClient())
    assert rows[0].skipped == "state_too_large"
    assert rows[0].usage["plan_b"] == {"input_tokens": 77, "requests": 1, "seconds": 0.5,
                                       "state_chars": None, "failed_requests": 0}


class PaceProbsClient(PlanBClient):
    def __init__(self, pace_probs):
        super().__init__()
        self.pace_probs = pace_probs

    def system_one(self, state, questions, **kw):
        response = super().system_one(state, questions, **kw)
        if "pace" in questions:
            response.choices["pace"].probabilities = self.pace_probs
        return response


def test_empty_pace_probabilities_are_unscorable(tmp_path):
    rows = _run_b(tmp_path, PaceProbsClient({}))
    assert rows[0].skipped is None
    assert rows[0].pace_pair == (None, "standard")
    rep = tbt.report(rows, n=3, run_info={})
    assert rep["pace_accuracy"]["n_scored"] == 0
    assert rep["pace_accuracy"]["n_unscorable"] == 1


def test_pace_argmax_ties_break_by_band_name(tmp_path):
    rows = _run_b(tmp_path, PaceProbsClient({"slow": 0.4, "fast": 0.4, "standard": 0.2}))
    assert rows[0].pace_pair[0] == "fast"


def test_pace_accuracy_counts_missing_prediction_as_unscorable():
    assert tbt.pace_accuracy([(None, "fast"), ("fast", "fast")]) == {
        "n_scored": 1, "n_unscorable": 1, "argmax_accuracy": 1.0}


class NoWinnerClient(Client):
    def system_one(self, state, questions, **kw):
        response = super().system_one(state, questions, **kw)
        response.choices = {}
        return response


def test_missing_winner_answer_is_not_scored_as_winner_series(tmp_path):
    # winner が丸ごと欠けたレースを「全頭0.0を馬番順に並べた順位」として採点しない（I-5）。
    rows = _run(tmp_path, client=NoWinnerClient())
    tuning = rows[0]
    assert tuning.skipped is None
    assert "plan_a" in tuning.series
    assert "winner_same_state_a" not in tuning.series
    rep = tbt.report(rows, n=3, run_info={})
    assert rep["by_period"]["tuning"]["n_winner_missing_races"] == {"plan_a": 1, "plan_b": 0}
    assert "winner_same_state_a" not in rep["by_period"]["tuning"]["series"]
    assert rep["by_period"]["final"]["n_winner_missing_races"] == {"plan_a": 0, "plan_b": 0}


def test_winner_present_counts_zero_missing_races(tmp_path):
    rep = tbt.report(_run(tmp_path), n=3, run_info={})
    assert rep["by_period"]["tuning"]["n_winner_missing_races"] == {"plan_a": 0, "plan_b": 0}


def test_save_records_writes_record_and_sent_prompt(tmp_path):
    repo, results = _repos(tmp_path)
    store_dir = tmp_path / "store"
    tbt.run([results.race(TUNING_RACE)], repo, results, Client(), methods=("plan_a",), n=3,
            tuning_until=date(2026, 3, 10), allow_final=False, max_recent_runs=5,
            save_records=True, data_dir=store_dir)
    records = list((store_dir / "top3" / "predictions").glob("*.jsonl"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["method"] == "plan_a"
    assert record["race_id"] == TUNING_RACE
    prompts = list((store_dir / "top3" / "prompts").glob("*/*.json"))
    assert [p.stem for p in prompts] == [record["record_id"]]
    sent = json.loads(prompts[0].read_text(encoding="utf-8"))
    assert set(sent["requests"][0]["questions"]) >= {"top3_01", "winner"}


def test_misspelled_method_is_refused_before_final_log(tmp_path, monkeypatch):
    # 方式名の検証が create_client と --final の記録より後だと、綴りを誤っただけで
    # 最終評価の境界が固定されてしまう（I-1）。
    monkeypatch.setattr(backtest_top3_cli, "create_client", lambda: Client())
    data_dir = _dataset_dir(tmp_path / "data")
    args = _final_cli_args(data_dir, tmp_path / "out")
    args[args.index("--methods") + 1] = "python,plan-x"

    rc = backtest_top3_cli.main(args)

    assert rc == 2
    assert not (data_dir / "top3" / "final_runs.jsonl").exists()


def test_final_race_count_ignores_jump_races(tmp_path):
    # 最終期間の障害レースは予想しない（jump でスキップ）ので、境界を記録するか
    # どうかの件数に入れない（M-6）。races_between は今は平地しか返さないが、
    # 件数の定義は run() のスキップ規則とここで揃えておく。
    import dataclasses
    _, results = _repos(tmp_path)
    tuning, final = results.race(TUNING_RACE), results.race(FINAL_RACE)
    jump = dataclasses.replace(final, track_kind="jump")
    until = date(2026, 3, 10)
    assert backtest_top3_cli._count_final_races([tuning, final], until) == 1
    assert backtest_top3_cli._count_final_races([tuning, jump], until) == 0


def test_cli_stops_and_writes_report_on_breaker(tmp_path, monkeypatch):
    # ConsecutiveJevFailures が飛んでも、そこまでの行から report.json を書き、
    # exit code 3 で終わる（rows.jsonl は on_row ですでに書かれている）。
    races, race_horses = _many_races(5, start_day=5)
    data_dir = fx.write_dataset(tmp_path / "data", races, race_horses, [])
    out_dir = tmp_path / "out"
    monkeypatch.setattr(backtest_top3_cli, "create_client", lambda: FailingClient())
    args = ["--data-dir", str(data_dir), "--from", "20260301", "--to", "20260315",
            "--tuning-until", "20260401", "--methods", "plan_a",
            "--out", str(out_dir), "--no-save-records",
            "--max-consecutive-jev-failures", "2", "--limit", "5"]

    rc = backtest_top3_cli.main(args)

    assert rc == 3
    report_files = list(out_dir.glob("*/report.json"))
    assert len(report_files) == 1
    report = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report["skip_breakdown"]["jev_failed"] == 2
    assert sum(report["skip_details"].values()) == 2
    rows_files = list(out_dir.glob("*/rows.jsonl"))
    assert len(rows_files) == 1
    written_rows = [json.loads(line) for line in
                    rows_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(written_rows) == 2  # 打ち切り時点までの行がすでに書かれている
