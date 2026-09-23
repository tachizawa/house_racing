from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path

import pytest

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
