"""3着内予想の比較バックテスト（report3 §7 の比較手順）。

1レースの失敗では止めず、理由付きでスキップする（evaluate/backtest.py と同じ
方針）。プロンプト調整に使う期間（tuning）と、最後に評価する期間（final）を
分け、final は allow_final=True の実行でしか予想しない——結果を見ながら質問を
調整したレースは、重みを学習していなくても未使用の検証データではない
（report3 §8）。
"""

from __future__ import annotations

import statistics
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jev_client import JevRequestError
from keiba import config
from keiba.data.repository import Repository, ResultRepository
from keiba.evaluate import top3_metrics as tm
from keiba.evaluate.backtest import prediction_cutoff_for, race_input_from_history
from keiba.features import dossier as dossier_mod
from keiba.features.pace import classify_pace
from keiba.input.match import AmbiguousMatchError
from keiba.models import RaceInput, RaceRow
from keiba.predict import top3, top3_baselines
from keiba.store import top3_records

__all__ = ["METHODS", "SERIES", "Top3Row", "dry_run", "pace_accuracy", "report", "run"]

METHODS: tuple[str, ...] = ("python", "plan_a", "plan_b")
SERIES: tuple[str, ...] = ("python", "plan_a", "winner_same_state_a",
                           "plan_b", "winner_same_state_b")
Top3SkipReason = Literal["jump", "no_entries", "no_result", "match_ambiguous",
                         "state_too_large", "jev_failed", "incomplete_answer",
                         "final_period_locked", "pace_unavailable"]
_ALL_SKIPS: tuple[str, ...] = ("jump", "no_entries", "no_result", "match_ambiguous",
                               "state_too_large", "jev_failed", "incomplete_answer",
                               "final_period_locked", "pace_unavailable")


@dataclass(frozen=True)
class Top3Row:
    race_id: str
    period: str
    skipped: str | None
    outcome: tm.Top3Outcome | None
    series: dict
    usage: dict
    python_all_tied: bool | None
    #: 案Bの (予測の最尤区分, 実際の区分)。案Bを回していない行は None。
    pace_pair: tuple | None = None


def _period(race: RaceRow, tuning_until: date) -> str:
    return "tuning" if prediction_cutoff_for(race).date() <= tuning_until else "final"


def _rank(probabilities: dict[str, float]) -> list[str]:
    return [u for u, _ in sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))]


def _skip(race: RaceRow, period: str, reason: str) -> Top3Row:
    return Top3Row(race.race_id, period, reason, None, {}, {}, None)


def _record(race_input: RaceInput, dossier: dict, method: str, *, ranking, answer=None,
            probabilities=None) -> top3_records.Top3Record:
    return top3_records.Top3Record(
        record_id=uuid.uuid4().hex, race_id=race_input.race_id, method=method,
        predicted_at=datetime.now(timezone.utc),
        prediction_cutoff=race_input.prediction_cutoff,
        versions=top3_records.versions_for(
            data_version=dossier["data_quality"]["data_version"],
            model=answer.model if answer else None),
        top3_probabilities=answer.top3_probabilities if answer else probabilities,
        winner_probabilities=answer.winner_probabilities if answer else None,
        winner_missing=list(answer.winner_missing) if answer else [],
        pace_probabilities=answer.pace_probabilities if answer else None,
        ranking=list(ranking),
        usage=({"input_tokens": answer.input_tokens, "requests": answer.requests}
               if answer else {"input_tokens": 0, "requests": 0}),
        state_chars=answer.state_chars if answer else None,
        seconds=answer.seconds if answer else 0.0,
        data_quality={"history_status_counts": dossier["field_summary"]["history_status_counts"],
                      "course_identified": dossier["course"]["identified"]},
    )


def run(races: Sequence[RaceRow], repo: Repository, results: ResultRepository, client: Any, *,
        methods: Sequence[str], n: int, tuning_until: date, allow_final: bool,
        max_recent_runs: int, save_records: bool, data_dir: Path,
        pace_definition_for: Callable[[RaceInput], dict | None] | None = None,
        on_row: Callable[[Top3Row], None] | None = None) -> list[Top3Row]:
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"未知の方式: {sorted(unknown)}")
    rows: list[Top3Row] = []

    def emit(row: Top3Row) -> None:
        rows.append(row)
        if on_row is not None:
            on_row(row)

    for race in races:
        period = _period(race, tuning_until)
        if not race.is_flat:
            emit(_skip(race, period, "jump"))
            continue
        race_results = results.results(race.race_id)
        entries = [r.entry for r in race_results]
        if len(entries) < 2:
            emit(_skip(race, period, "no_entries"))
            continue
        outcome = tm.outcome_from_results(race.race_id, race_results)
        if outcome is None:
            emit(_skip(race, period, "no_result"))
            continue
        if period == "final" and not allow_final:
            emit(_skip(race, period, "final_period_locked"))
            continue

        race_input = race_input_from_history(race, entries, top_n=n)
        try:
            dossier = dossier_mod.build_dossier(race_input, repo, max_recent_runs=max_recent_runs)
        except AmbiguousMatchError:
            emit(_skip(race, period, "match_ambiguous"))
            continue

        series: dict[str, dict] = {}
        usage: dict[str, dict] = {}
        python_all_tied = None
        pace_pair = None
        skipped = None
        for method in methods:
            try:
                if method == "python":
                    base = top3_baselines.ability_ranking(dossier)
                    probs = top3_baselines.uniform_probabilities(dossier)
                    python_all_tied = base.all_tied
                    series["python"] = {"ranking": list(base.ranking), "probabilities": probs}
                    record, sent = _record(race_input, dossier, "python",
                                           ranking=base.ranking, probabilities=probs), None
                else:
                    if method == "plan_a":
                        answer = top3.run_plan_a(client, dossier)
                    else:
                        pace_definition = (pace_definition_for(race_input)
                                           if pace_definition_for else None)
                        answer = top3.run_plan_b(client, dossier,
                                                 pace_definition=pace_definition)
                        # 前半3Fは結果なので評価側（ResultRepository）からだけ読む。
                        actual_race = results.race(race.race_id)
                        actual = classify_pace(
                            actual_race.first_3f_sec if actual_race else None,
                            pace_definition)
                        predicted = max(answer.pace_probabilities,
                                        key=answer.pace_probabilities.get)
                        pace_pair = (predicted, actual)
                    suffix = method[-1]
                    ranking = _rank(answer.top3_probabilities)
                    series[method] = {"ranking": ranking,
                                      "probabilities": answer.top3_probabilities}
                    series[f"winner_same_state_{suffix}"] = {
                        "ranking": _rank(answer.winner_probabilities), "probabilities": None}
                    usage[method] = {"input_tokens": answer.input_tokens,
                                     "requests": answer.requests, "seconds": answer.seconds,
                                     "state_chars": answer.state_chars, "failed_requests": 0}
                    record, sent = _record(race_input, dossier, method, ranking=ranking,
                                           answer=answer), answer.sent
            except dossier_mod.StateTooLargeError:
                # state を送る前に弾いているので、課金は発生していない
                # （usage には何も足さない）。
                skipped = "state_too_large"
                break
            except top3.PaceUnavailableError:
                # pace_definition が作れないと分かった時点で止めており、
                # 何も送っていないので課金は発生していない（usage には足さない）。
                skipped = "pace_unavailable"
                break
            except top3.Top3IncompleteError:
                # 応答は返ってきている＝課金は発生した（トークン数は例外側が
                # 持っていないので不明のまま 0 にする）。failed_requests で
                # 「請求額が読めない課金がある」ことを report() に見せる
                # （Minor 5: 費用の可視性）。
                usage[method] = {"input_tokens": 0, "requests": 0, "seconds": 0.0,
                                 "state_chars": None, "failed_requests": 1}
                skipped = "incomplete_answer"
                break
            except JevRequestError:
                # タイムアウト・接続不能なども課金が発生した可能性があるので
                # 同様に failed_requests で見えるようにする。
                usage[method] = {"input_tokens": 0, "requests": 0, "seconds": 0.0,
                                 "state_chars": None, "failed_requests": 1}
                skipped = "jev_failed"
                break
            if save_records:
                top3_records.save(record, sent=sent, data_dir=data_dir)
        if skipped is not None:
            emit(Top3Row(race.race_id, period, skipped, None, {}, usage, None))
            continue
        emit(Top3Row(race.race_id, period, None, outcome, series, usage,
                     python_all_tied, pace_pair))
    return rows


def pace_accuracy(pairs: Sequence[tuple[str, str | None]]) -> dict:
    """(予測の最尤区分, 実際の区分) の組。実際が None（前半3Fが無い）は数えない。"""
    scored = [(p, a) for p, a in pairs if a is not None]
    return {"n_scored": len(scored), "n_unscorable": len(pairs) - len(scored),
            "argmax_accuracy": (sum(1 for p, a in scored if p == a) / len(scored)
                                if scored else float("nan"))}


def _series_scores(rows: Sequence[Top3Row], name: str, n: int) -> dict[str, dict]:
    return {row.race_id: tm.score_race(row.series[name]["ranking"][:n],
                                       row.series[name]["probabilities"], row.outcome)
            for row in rows if name in row.series}


def _period_block(rows: Sequence[Top3Row], n: int) -> dict:
    scored_rows = [r for r in rows if r.skipped is None]
    scores = {name: _series_scores(scored_rows, name, n) for name in SERIES}
    series = {name: tm.aggregate(list(s.values()), n=n) for name, s in scores.items() if s}
    comparisons = {}
    for challenger in ("plan_a", "plan_b"):
        if challenger not in series:
            continue
        suffix = challenger[-1]
        for reference in ("python", f"winner_same_state_{suffix}"):
            if reference in series:
                comparisons[f"{challenger}_vs_{reference}"] = tm.paired_bootstrap(
                    {k: v["hits"] for k, v in scores[challenger].items()},
                    {k: v["hits"] for k, v in scores[reference].items()},
                    cluster_of=tm.race_date_cluster)
    if "plan_a" in series and "plan_b" in series:
        comparisons["plan_b_vs_plan_a"] = tm.paired_bootstrap(
            {k: v["hits"] for k, v in scores["plan_b"].items()},
            {k: v["hits"] for k, v in scores["plan_a"].items()},
            cluster_of=tm.race_date_cluster)
    return {
        "n_rows": len(rows),
        "n_scored": len(scored_rows),
        "series": series,
        "comparisons": comparisons,
        "n_python_all_tied": sum(1 for r in scored_rows if r.python_all_tied),
    }


def report(rows: Sequence[Top3Row], *, n: int, run_info: dict) -> dict:
    usage: dict[str, dict] = {}
    for row in rows:
        for method, u in row.usage.items():
            total = usage.setdefault(method, {"input_tokens": 0, "requests": 0,
                                              "seconds": 0.0, "races": 0,
                                              "failed_requests": 0})
            total["input_tokens"] += u["input_tokens"]
            total["requests"] += u["requests"]
            total["seconds"] += u["seconds"]
            total["races"] += 1
            # 応答が欠けた／失敗したが課金は発生した可能性がある呼び出しの数
            # （トークン数が読めないので input_tokens には足し込めない。
            # Minor 5: このカウントを見て「usage の合計は下限」と読むこと）。
            total["failed_requests"] += u.get("failed_requests", 0)
    for total in usage.values():
        total["usd"] = total["input_tokens"] / 1_000_000 * 0.042  # 入力 $0.042 / 1M（CLAUDE.md）
    return {
        "n": n,
        "versions": top3_records.versions_for(data_version=run_info.get("data_version", ""),
                                              model=None),
        "skip_breakdown": {reason: sum(1 for r in rows if r.skipped == reason)
                           for reason in _ALL_SKIPS},
        "by_period": {period: _period_block([r for r in rows if r.period == period], n)
                      for period in ("tuning", "final")},
        "usage": usage,
        "pace_accuracy": pace_accuracy([r.pace_pair for r in rows if r.pace_pair is not None]),
        "run": run_info,
    }


def dry_run(races: Sequence[RaceRow], repo: Repository, *, n: int, max_recent_runs: int) -> dict:
    """state の大きさだけを測る。Jev を呼ばない。結果（ResultRepository）も使わない。"""
    sizes: dict[str, int] = {}
    for race in races:
        if not race.is_flat:
            continue
        entries = repo.entries(race.race_id, as_of=prediction_cutoff_for(race))
        if len(entries) < 2:
            continue
        race_input = race_input_from_history(race, entries, top_n=n)
        try:
            d = dossier_mod.build_dossier(race_input, repo, max_recent_runs=max_recent_runs)
        except AmbiguousMatchError:
            continue
        sizes[race.race_id] = dossier_mod.state_chars(dossier_mod.to_state(d))
    values = list(sizes.values())
    return {
        "n_races": len(sizes),
        "max_state_chars": max(values) if values else 0,
        "median_state_chars": statistics.median(values) if values else 0,
        "limit": config.STATE_MAX_CHARS,
        "over_limit": sorted(r for r, c in sizes.items() if c > config.STATE_MAX_CHARS),
        # 実測の比（約0.56トークン/文字）での概算。Task 14 で見直す。
        "estimated_tokens_per_race_plan_a": int(statistics.mean(values) * 0.56) if values else 0,
    }
