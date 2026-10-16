"""3着内予想の比較バックテスト。

    # 課金なし: state の大きさを測る
    .venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260922 \
        --tuning-until 20260831 --dry-run
    # 課金あり（所有者の承認を得てから）: 調整期間を10レース
    .venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260831 \
        --tuning-until 20260831 --methods python,plan_a --limit 10

最終評価期間（--tuning-until より後）は --final を付けたときだけ予想する。
**一度 --final を実行したら最終評価の境界は固定される**——最初に記録された
--tuning-until（data/top3/final_runs.jsonl の1行目）より後の日付をどの実行
（--final の有無を問わない）に指定しても --final-rerun-reason が要り、
理由は data/top3/final_runs.jsonl に残る（report3 §8）。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest import _sample_evenly  # noqa: E402  同じ間引き規則を使う（fix3 決定1）
from jev_client import JevConfigError, create_client  # noqa: E402
from keiba import config  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore  # noqa: E402
from keiba.evaluate import top3_backtest as tbt  # noqa: E402
from keiba.evaluate.backtest import hasso_window_for_cutoff_window, prediction_cutoff_for  # noqa: E402
from keiba.features.pace import fit_pace_definition  # noqa: E402
from keiba.predict.top3_questions import TOP3_PROMPT_VERSION  # noqa: E402

JST = timezone(timedelta(hours=9))


def _day(value: str, *, end: bool = False) -> datetime:
    d = datetime.strptime(value, "%Y%m%d").replace(tzinfo=JST)
    return d.replace(hour=23, minute=59, second=59) if end else d


def _to_jsonable(value):
    if dataclasses.is_dataclass(value):
        return _to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def _final_log_path(data_dir: Path) -> Path:
    return data_dir / "top3" / "final_runs.jsonl"


def _read_final_log(data_dir: Path) -> list[dict]:
    log = _final_log_path(data_dir)
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _final_guard_problem(args, log_entries: list[dict]) -> str | None:
    """最終評価期間の再利用を防ぐ。**ここでは書き込みをしない**（純粋な判定のみ）。

    fix round 1 Critical 1: 元の実装は判定と同じ関数の中で即座にログへ1行
    書いていたため、API キー未設定（JevConfigError）や対象レースが無いなど、
    予想を1件も行わずに終わる実行でも「1回ぶんの --final 実行」を消費して
    しまっていた。判定（この関数）と記録（_append_final_log）を分け、記録は
    最終期間のレースを実際に1件予想した時点（その行を rows.jsonl に書く直前。
    実行の途中で打ち切られても記録は残る）にしか行わない。

    fix round 1 Critical 2: --tuning-until を後ろへずらすだけで検査をすり抜け
    られた（「一度 --final した期間」の定義が実行のたびの --tuning-until その
    ものだったため）。**最初に記録した --tuning-until が最終評価の境界として
    固定される**——それより後の --tuning-until を指定するその後のどの実行
    （--final の有無を問わない）も、境界を超えたレースを調整期間として扱う
    ことになり、すでに最終評価に使ったレースへもう一度（Jev 呼び出しを含めて）
    触れてしまう。--dry-run はここへ来る前に main() が return しているので、
    この関数を呼ぶ必要が無い（Jev を呼ばないので境界を侵さない）。
    """
    if not log_entries:
        return None
    pinned = log_entries[0]["tuning_until"]
    if args.final and not args.final_rerun_reason:
        return ("最終評価期間はすでに評価済みです（data/top3/final_runs.jsonl）。"
                "再評価するなら --final-rerun-reason に理由を書いてください。")
    if args.tuning_until > pinned and not args.final_rerun_reason:
        return (
            f"最終評価の境界（tuning_until={pinned}）はすでに固定されています"
            "（data/top3/final_runs.jsonl の1行目）。"
            f"指定した --tuning-until（{args.tuning_until}）はそれより後なので、"
            "境界を超えたレースはすでに最終評価に使われています。"
            "意図的に境界を動かすなら --final-rerun-reason に理由を書いてください。"
        )
    return None


def _count_final_races(races, tuning_until) -> int:
    """最終期間で予想する見込みのレース数。障害レースは run() が jump で飛ばす
    ので数えない（final review M-6）。結果に依るスキップ（no_result など）は
    結果を読まないと分からないので、ここでは数えたままにする。
    """
    return sum(1 for race in races
               if race.is_flat and prediction_cutoff_for(race).date() > tuning_until)


#: 方式を1つも走らせる前に決まるスキップ。そのレースの予想も Jev への送信もしていない。
_SKIPS_BEFORE_PREDICTING = frozenset(
    {"jump", "no_entries", "no_result", "final_period_locked", "match_ambiguous"})


def _row_used_the_race(row) -> bool:
    """その行のレースで、少なくとも1つの方式が予想した（または Jev に送った）か。

    迷う行は「使った」側に倒す（境界の記録が漏れるほうが害が大きい）。
    pace_unavailable は、方式のループの前に飛ばした行（skip_detail が無い）なら
    何も走らせていない。ループの中の PaceUnavailableError（skip_detail あり）は
    先に回った方式がすでに予想しているかもしれない。
    """
    if row.skipped is None:
        return True
    if row.skipped in _SKIPS_BEFORE_PREDICTING:
        return False
    return not (row.skipped == "pace_unavailable" and row.skip_detail is None)


def _append_final_log(args, data_dir: Path, n_final_races: int) -> None:
    """--final の実行、または固定された境界を越える --final なしの実行を記録する。

    n_final_races は境界より後で予想する見込みのレース数（--final なしなら、境界から
    --tuning-until までの調整期間として使い直すレース数）。記録する時点では予想を
    始めたばかりなので、実際に予想できた件数ではなく見込み（スキップを含む）。

    main() は、境界より後のレースを予想した最初の行（_row_used_the_race）を
    rows.jsonl に書く直前に1回だけ呼ぶ。そうしたレースが1件も無ければ境界を
    使っていないので記録しない（fix round 1 Critical 3 のテスト (f)）。
    """
    log = _final_log_path(data_dir)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "tuning_until": args.tuning_until, "to": args.date_to,
            "methods": args.methods, "top3_prompt_version": TOP3_PROMPT_VERSION,
            "dossier_version": config.DOSSIER_VERSION,
            "n_final_races": n_final_races, "final": args.final,
            "reason": args.final_rerun_reason}, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD")
    p.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD")
    p.add_argument("--tuning-until", required=True, metavar="YYYYMMDD")
    # 既定は "python" のみ（Minor 8: --methods を省略しても課金しない）。
    # Jev に課金する方式（plan_a / plan_b）は明示的に指定させる。
    p.add_argument("--methods", default="python")
    p.add_argument("--n", type=int, default=config.DEFAULT_TOP_N)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-recent-runs", type=int, default=config.DEFAULT_DOSSIER_RECENT_RUNS)
    # jev_failed が連続したら打ち切る（原因不明の持続的失敗に費用の読めない
    # リクエストを送り続けないため）。0 は無効化（tbt.run() には None を渡す）。
    p.add_argument("--max-consecutive-jev-failures", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--final-rerun-reason", default=None)
    p.add_argument("--save-records", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=Path, default=Path("data/top3/backtests"))
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = p.parse_args(argv)
    if args.max_recent_runs < 0:
        p.error("--max-recent-runs は 0 以上にしてください")

    # 重複は順序を保って1つにする。同じ方式を2回回すと Jev に2回課金するのに、
    # usage とレポートは方式ごとに1つしか持たないので費用を過少に報告する。
    methods = list(dict.fromkeys(m.strip() for m in args.methods.split(",") if m.strip()))
    # 方式名はここで確かめる。tbt.run() の中で初めて弾くと、データ版を読み
    # create_client() を済ませた後に落ちる（当時は --final の記録もその前に
    # 済んでおり、綴りを誤っただけで最終評価の境界が固定されていた。final review I-1）。
    unknown = [m for m in methods if m not in tbt.METHODS]
    if unknown or not methods:
        print(f"未知の方式です: {', '.join(unknown) or '（空）'}。"
              f"指定できるのは {', '.join(tbt.METHODS)} です（カンマ区切り）。")
        return 2
    store = DataStore.open_current(args.data_dir)
    repo, results = Repository(store), ResultRepository(store)
    start, end = hasso_window_for_cutoff_window(_day(args.date_from), _day(args.date_to, end=True))
    races = _sample_evenly(results.races_between(start, end), args.limit)
    print(f"対象レース: {len(races)}（データ版 {store.version_id}）")

    if args.dry_run:
        print(json.dumps(tbt.dry_run(races, repo, n=args.n, max_recent_runs=args.max_recent_runs),
                         ensure_ascii=False, indent=2))
        return 0

    log_entries = _read_final_log(args.data_dir)
    problem = _final_guard_problem(args, log_entries)
    if problem:
        print(problem)
        return 2

    all_races = results.races_between(_day("20000101"), end)

    def pace_definition_for(race_input):
        return fit_pace_definition(all_races, target=race_input, as_of=race_input.prediction_cutoff)

    needs_jev = any(m != "python" for m in methods)
    try:
        client = create_client() if needs_jev else None
    except JevConfigError as error:
        print(error)
        return 2

    tuning_until = _day(args.tuning_until).date()
    # ログへの記録は、境界より後のレースを実際に1件予想した時点で行う（write_row が
    # その行を書く直前）。予想の前に記録すると、境界より後のレースがすべて no_result や
    # pace_unavailable で飛ばされた実行でも境界が固定されてしまう。予想の後（tbt.run が
    # 返った後）に記録すると、途中で打ち切られた実行（連続失敗・Ctrl-C）が記録されず、
    # 最終期間を見たのに理由なしで再実行できてしまう（fix round 1 Critical 1 / 3）。
    # 残る隙間: 最初の1レースの途中（Jev に送った後、行になる前）で打ち切られると
    # 記録されない。そのレースの採点はまだ出ていない。
    log_boundary = None     # 締切がこの日より後のレースを予想したら記録する
    n_expected = 0          # 記録に残す見込みの件数（_append_final_log の n_final_races）
    if args.final:
        log_boundary = tuning_until
        n_expected = _count_final_races(races, tuning_until)
    elif log_entries and args.tuning_until > log_entries[0]["tuning_until"]:
        # --final なしで固定された境界を越える実行（ガードは理由付きで通している）も
        # 記録する。境界から --tuning-until までのレースを調整期間としてもう一度使う。
        log_boundary = _day(log_entries[0]["tuning_until"]).date()
        n_expected = (_count_final_races(races, log_boundary)
                      - _count_final_races(races, tuning_until))
    cutoff_dates = {race.race_id: prediction_cutoff_for(race).date() for race in races}
    logged = False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    def write_row(row):
        nonlocal logged
        if (not logged and log_boundary is not None and _row_used_the_race(row)
                and cutoff_dates[row.race_id] > log_boundary):
            _append_final_log(args, args.data_dir, n_expected)
            logged = True
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")

    max_consecutive_failures = (None if args.max_consecutive_jev_failures == 0
                                else args.max_consecutive_jev_failures)
    run_info = {
        "data_version": store.version_id, "from": args.date_from, "to": args.date_to,
        "tuning_until": args.tuning_until, "limit": args.limit, "methods": methods,
        "final": args.final, "max_recent_runs": args.max_recent_runs}

    def write_report(rows) -> dict:
        rep = tbt.report(rows, n=args.n, run_info=run_info)
        (out_dir / "report.json").write_text(
            json.dumps(_to_jsonable(rep), ensure_ascii=False, indent=2), encoding="utf-8")
        return rep

    try:
        rows = tbt.run(races, repo, results, client, methods=methods, n=args.n,
                       tuning_until=tuning_until, allow_final=args.final,
                       max_recent_runs=args.max_recent_runs, save_records=args.save_records,
                       data_dir=args.data_dir, pace_definition_for=pace_definition_for,
                       on_row=write_row, max_consecutive_jev_failures=max_consecutive_failures)
    except tbt.ConsecutiveJevFailures as breaker:
        # rows.jsonl には打ち切りまでの行がすでに書かれている（on_row）。
        # report.json もここまでの行（breaker.rows）から書く——一発勝負の
        # 最終評価で、原因不明の連続失敗に気づかず課金し続けないため。
        rep = write_report(breaker.rows)
        print(f"jev_failed が {args.max_consecutive_jev_failures} 回連続したため、"
              f"打ち切りました。直前の失敗: {breaker.skip_detail}")
        print(f"スキップ: {rep['skip_breakdown']}")
        print(f"skip_details: {rep['skip_details']}")
        print(f"保存: {out_dir}")
        return 3

    rep = write_report(rows)
    print(json.dumps(_to_jsonable(rep["by_period"]), ensure_ascii=False, indent=2))
    print(f"スキップ: {rep['skip_breakdown']}")
    print(f"skip_details: {rep['skip_details']}")
    print(f"使用量: {rep['usage']}")
    print(f"保存: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
