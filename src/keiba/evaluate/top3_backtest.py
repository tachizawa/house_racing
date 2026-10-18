"""3着内予想の比較バックテスト（report3 §7 の比較手順）。

1レースの失敗では止めず、理由付きでスキップする（evaluate/backtest.py と同じ
方針）。プロンプト調整に使う期間（tuning）と、最後に評価する期間（final）を
分け、final は allow_final=True の実行でしか予想しない——結果を見ながら質問を
調整したレースは、重みを学習していなくても未使用の検証データではない
（report3 §8）。
"""

from __future__ import annotations

import statistics
import time
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

__all__ = ["METHODS", "SERIES", "ConsecutiveJevFailures", "Top3Row", "dry_run",
           "pace_accuracy", "report", "run"]

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
    #: 予測の確率が空なら最尤区分は None（pace_accuracy は採点不能として数える）。
    pace_pair: tuple | None = None
    #: winner の答えが丸ごと欠けた方式（"plan_a" / "plan_b"）。その方式の
    #: winner_same_state_{a|b} は series に入れない（全頭 0.0 を馬番順に並べた
    #: 順位を採点しないため。final review I-5）。
    winner_missing_methods: tuple[str, ...] = ()
    #: skipped が例外由来のとき "{method}: {例外クラス名}: {メッセージ}"
    #: （300文字で切り詰め）。method は失敗した方式名、match_ambiguous のように
    #: 方式に紐づかない失敗は "dossier"。例外に由来しないスキップ（jump /
    #: no_entries / no_result / final_period_locked や、pace_definition_for が
    #: 例外なしで None を返した pace_unavailable）は None のまま
    #: （Task: 失敗の理由を記録する）。
    skip_detail: str | None = None


def _period(race: RaceRow, tuning_until: date) -> str:
    return "tuning" if prediction_cutoff_for(race).date() <= tuning_until else "final"


def _rank(probabilities: dict[str, float]) -> list[str]:
    return [u for u, _ in sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))]


def _pace_argmax(probabilities: dict[str, float]) -> str | None:
    """最尤区分。同率は区分名の順で決める（dict の順に依らない）。空なら None。"""
    ranked = _rank(probabilities)
    return ranked[0] if ranked else None


_PARTIAL_FAILURE_SKIPS: tuple[tuple[type, str], ...] = (
    (top3.Top3IncompleteError, "incomplete_answer"),
    (JevRequestError, "jev_failed"),
    (dossier_mod.StateTooLargeError, "state_too_large"),
)


def _partial_failure_skip(cause: Exception) -> str:
    for kind, reason in _PARTIAL_FAILURE_SKIPS:
        if isinstance(cause, kind):
            return reason
    raise TypeError(f"想定していない失敗: {cause!r}") from cause


def _skip(race: RaceRow, period: str, reason: str, *, detail: str | None = None) -> Top3Row:
    return Top3Row(race.race_id, period, reason, None, {}, {}, None, skip_detail=detail)


def _skip_detail(method: str, error: Exception) -> str:
    """スキップの理由になった例外を短い1行にする。

    ``JevRequestError`` はラップ用の例外なので、クラス名は ``original``
    （SDK が投げた元の例外）から取る。メッセージは ``str(error)``——
    ``JevRequestError`` は ``jev_client.describe_error()`` 済みの日本語を
    そのまま持っているので、それを使う。それ以外の例外は自分自身が
    ``original`` でありメッセージでもある。300文字を超える分は切り詰める
    （レポート・ログを肥大化させないため）。
    """
    original = error.original if isinstance(error, JevRequestError) else error
    return f"{method}: {type(original).__name__}: {error}"[:300]


def _skip_detail_prefix(detail: str) -> str:
    """report() の ``skip_details`` のキー。

    ``skip_detail`` は "{method}: {例外クラス名}: {メッセージ}" 形式なので、
    最初の2つの ": " で区切り、メッセージだけ先頭80文字に切り詰める
    （例外ごとに1文字も違わない請求 request_id などでキーが分裂しないように）。
    """
    method, _, rest = detail.partition(": ")
    cls, _, message = rest.partition(": ")
    return f"{method}: {cls}: {message[:80]}"


class ConsecutiveJevFailures(RuntimeError):
    """jev_failed が連続で規定回数に達したため ``run()`` が打ち切った。

    ``rows`` は打ち切りまでに確定した行（最後の jev_failed 行を含む）。
    ``skip_detail`` はその最後の行の理由。呼び出し側（scripts/backtest_top3.py）
    はここまでの ``rows`` から report.json を書ける。
    """

    def __init__(self, rows: list[Top3Row], skip_detail: str | None) -> None:
        super().__init__(f"jev_failed が連続したため打ち切りました: {skip_detail}")
        self.rows = rows
        self.skip_detail = skip_detail


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
        on_row: Callable[[Top3Row], None] | None = None,
        max_consecutive_jev_failures: int | None = 5) -> list[Top3Row]:
    """``max_consecutive_jev_failures``: jev_failed が連続でこの回数に達したら
    ``ConsecutiveJevFailures`` で打ち切る（rate limit・キー失効などの持続的な
    失敗に、費用の読めないリクエストを送り続けないため）。数えるのは
    jev_failed のスキップだけ——Jev 呼び出しが成功したレースで 0 に戻り、
    python のみのレース・jev_failed 以外のスキップ（state_too_large など）は
    連続数を変えない。``None`` で無効化する。
    """
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"未知の方式: {sorted(unknown)}")
    rows: list[Top3Row] = []
    consecutive_jev_failures = 0

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
        except AmbiguousMatchError as error:
            # 方式のループに入る前（どの方式にも紐づかない）失敗なので
            # method 欄は "dossier" にする。
            emit(_skip(race, period, "match_ambiguous", detail=_skip_detail("dossier", error)))
            continue

        # 案Bの pace_definition はレースごとに1回、どの方式も Jev に送る前に
        # 作る。作れないならレースごと飛ばす——案Aを課金してから案Bで止まると、
        # その代金が無駄になり、方式間で比べるレースの集合もずれる（final review I-4a）。
        pace_definition = None
        if "plan_b" in methods:
            pace_definition = pace_definition_for(race_input) if pace_definition_for else None
            if pace_definition is None:
                emit(_skip(race, period, "pace_unavailable"))
                continue

        series: dict[str, dict] = {}
        usage: dict[str, dict] = {}
        python_all_tied = None
        pace_pair = None
        winner_missing_methods: list[str] = []
        skipped = None
        skip_detail = None
        for method in methods:
            # 答えが欠けて Top3IncompleteError で来たときの所要時間（例外は時間を持たない）
            started = time.perf_counter()
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
                        answer = top3.run_plan_b(client, dossier,
                                                 pace_definition=pace_definition)
                        # 前半3Fは結果なので評価側（ResultRepository）からだけ読む。
                        actual_race = results.race(race.race_id)
                        actual = classify_pace(
                            actual_race.first_3f_sec if actual_race else None,
                            pace_definition)
                        pace_pair = (_pace_argmax(answer.pace_probabilities or {}), actual)
                    suffix = method[-1]
                    ranking = _rank(answer.top3_probabilities)
                    series[method] = {"ranking": ranking,
                                      "probabilities": answer.top3_probabilities}
                    if len(answer.winner_missing) == len(answer.winner_probabilities):
                        # winner の答えが1頭分も無い。全頭 0.0 の馬番順は予想ではない。
                        winner_missing_methods.append(method)
                    else:
                        series[f"winner_same_state_{suffix}"] = {
                            "ranking": _rank(answer.winner_probabilities),
                            "probabilities": None}
                    usage[method] = {"input_tokens": answer.input_tokens,
                                     "requests": answer.requests, "seconds": answer.seconds,
                                     "state_chars": answer.state_chars, "failed_requests": 0}
                    record, sent = _record(race_input, dossier, method, ranking=ranking,
                                           answer=answer), answer.sent
            except top3.PlanBPartialFailure as failure:
                # 案Bで pace のリクエストが成功した後の失敗。pace の代金は分かって
                # いるので落とさない。2回目を送って応答が返らなかったなら、その分は
                # 請求額の読めない課金として failed_requests に数える（final review I-4b）。
                # 2回目の答えが欠けただけ（Top3IncompleteError）なら応答の利用量は
                # failure.input_tokens に入っているので、failed_requests には数えない。
                unknown_cost = (failure.requests_sent == 2
                                and not isinstance(failure.cause, top3.Top3IncompleteError))
                usage[method] = {"input_tokens": failure.input_tokens,
                                 "requests": failure.requests_sent,
                                 "seconds": failure.seconds, "state_chars": None,
                                 "failed_requests": 1 if unknown_cost else 0}
                skipped = _partial_failure_skip(failure.cause)
                skip_detail = _skip_detail(method, failure.cause)
                break
            except dossier_mod.StateTooLargeError as error:
                # state を送る前に弾いているので、この方式の課金は発生していない
                # （この方式の usage には何も足さない。同じレースで先に回った方式の
                # 課金は usage に残る）。案Bは2回目の state も pace を聞く前に検査して
                # いる。pace を聞いた後に弾いた場合は PlanBPartialFailure で来る。
                skipped = "state_too_large"
                skip_detail = _skip_detail(method, error)
                break
            except top3.PaceUnavailableError as error:
                # 通常は方式のループの前に pace_unavailable で飛ばしている。
                # 何も送っていないので課金は発生していない（usage には足さない）。
                skipped = "pace_unavailable"
                skip_detail = _skip_detail(method, error)
                break
            except top3.Top3IncompleteError as error:
                # 応答は返ってきている＝課金は発生した。例外が応答の利用量を
                # 持っているので、それを usage に足す（足さないと使用量と費用を
                # 過少報告する）。トークン数が読めているので failed_requests には
                # 数えない。案Bの2回目の欠けは PlanBPartialFailure で来る。
                usage[method] = {"input_tokens": error.input_tokens,
                                 "requests": error.requests,
                                 "seconds": time.perf_counter() - started,
                                 "state_chars": None,
                                 "failed_requests": 0 if error.requests else 1}
                skipped = "incomplete_answer"
                skip_detail = _skip_detail(method, error)
                break
            except JevRequestError as error:
                # タイムアウト・接続不能なども課金が発生した可能性があるので
                # 同様に failed_requests で見えるようにする。
                usage[method] = {"input_tokens": 0, "requests": 0, "seconds": 0.0,
                                 "state_chars": None, "failed_requests": 1}
                skipped = "jev_failed"
                skip_detail = _skip_detail(method, error)
                break
            if save_records:
                top3_records.save(record, sent=sent, data_dir=data_dir)
        if skipped is not None:
            emit(Top3Row(race.race_id, period, skipped, None, {}, usage, None,
                         skip_detail=skip_detail))
            if skipped == "jev_failed":
                consecutive_jev_failures += 1
                if (max_consecutive_jev_failures is not None
                        and consecutive_jev_failures >= max_consecutive_jev_failures):
                    raise ConsecutiveJevFailures(rows, skip_detail)
            continue
        emit(Top3Row(race.race_id, period, None, outcome, series, usage,
                     python_all_tied, pace_pair, tuple(winner_missing_methods)))
        if any(method != "python" for method in methods):
            # Jev を呼んで成功したレース。連続失敗数を0に戻す
            # （python のみのレースは Jev を呼んでいないので変えない）。
            consecutive_jev_failures = 0
    return rows


def pace_accuracy(pairs: Sequence[tuple[str, str | None]]) -> dict:
    """(予測の最尤区分, 実際の区分) の組。実際が None（前半3Fが無い）と、
    予測が None（pace の確率が空で返った）は採点不能として数える。"""
    scored = [(p, a) for p, a in pairs if p is not None and a is not None]
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
        # winner の答えが丸ごと欠けたため winner_same_state_{a|b} を採点しなかったレース数。
        "n_winner_missing_races": {
            method: sum(1 for r in scored_rows if method in r.winner_missing_methods)
            for method in ("plan_a", "plan_b")},
    }


def _skip_detail_counts(rows: Sequence[Top3Row]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.skip_detail is None:
            continue
        key = _skip_detail_prefix(row.skip_detail)
        counts[key] = counts.get(key, 0) + 1
    return counts


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
            # 失敗して応答が読めなかったが課金は発生した可能性がある呼び出しの数
            # （トークン数が読めないので input_tokens には足し込めない。答えが
            # 欠けただけの応答は利用量が読めるので、ここではなく input_tokens に入る。
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
        # {"method: 例外クラス名: メッセージ先頭80文字": 件数}。例外に由来しない
        # スキップ（jump など）は skip_detail が無いので数えない
        # （skip_breakdown の方に既に出ている）。
        "skip_details": _skip_detail_counts(rows),
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
