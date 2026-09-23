from __future__ import annotations

import types
from datetime import date

import pytest

from keiba.data.repository import Repository, ResultRepository
from keiba.data.store import DataStore
from keiba.evaluate import top3_backtest as tbt
from tests.fixtures import csv_data as fx

TUNING_RACE = "2026030106010101"   # 締切 2026-02-28
FINAL_RACE = "2026032106011101"    # 締切 2026-03-20


def _repos(tmp_path):
    races = [fx.make_race(race_id=TUNING_RACE, hasso_at="2026-03-01T15:00:00+09:00"),
             fx.make_race(race_id=FINAL_RACE, hasso_at="2026-03-21T15:25:00+09:00")]
    race_horses = []
    for race_id in (TUNING_RACE, FINAL_RACE):
        for i in range(1, 6):
            race_horses.append(fx.make_race_horse(
                race_id=race_id, umaban=f"{i:02d}", ketto_num=f"{race_id}{i}",
                bamei=f"ウマ{race_id[-4:]}{i}", result_kakutei_jyuni=str(i)))
    store = DataStore.open_current(fx.write_dataset(tmp_path, races, race_horses, []))
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
