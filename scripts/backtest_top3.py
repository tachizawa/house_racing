"""3着内予想の比較バックテスト。

    # 課金なし: state の大きさを測る
    .venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260922 \
        --tuning-until 20260831 --dry-run
    # 課金あり（所有者の承認を得てから）: 調整期間を10レース
    .venv/Scripts/python.exe scripts/backtest_top3.py --from 20260801 --to 20260831 \
        --tuning-until 20260831 --methods python,plan_a --limit 10

最終評価期間（--tuning-until より後）は --final を付けたときだけ予想する。
一度 --final を実行した期間を再び評価するには --final-rerun-reason が要り、
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
from keiba.evaluate.backtest import hasso_window_for_cutoff_window  # noqa: E402
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


def _check_final_guard(args, data_dir: Path) -> str | None:
    """--final の二度目以降は理由を要求する。問題があればメッセージを返す。"""
    log = data_dir / "top3" / "final_runs.jsonl"
    if not args.final:
        return None
    if log.exists() and log.read_text(encoding="utf-8").strip() and not args.final_rerun_reason:
        return ("最終評価期間はすでに評価済みです（data/top3/final_runs.jsonl）。"
                "再評価するなら --final-rerun-reason に理由を書いてください。")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "tuning_until": args.tuning_until, "to": args.date_to,
            "methods": args.methods, "top3_prompt_version": TOP3_PROMPT_VERSION,
            "dossier_version": config.DOSSIER_VERSION,
            "reason": args.final_rerun_reason}, ensure_ascii=False) + "\n")
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD")
    p.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD")
    p.add_argument("--tuning-until", required=True, metavar="YYYYMMDD")
    p.add_argument("--methods", default="python,plan_a")
    p.add_argument("--n", type=int, default=config.DEFAULT_TOP_N)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--max-recent-runs", type=int, default=config.DEFAULT_DOSSIER_RECENT_RUNS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--final-rerun-reason", default=None)
    p.add_argument("--save-records", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=Path, default=Path("data/top3/backtests"))
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = p.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    store = DataStore.open_current(args.data_dir)
    repo, results = Repository(store), ResultRepository(store)
    start, end = hasso_window_for_cutoff_window(_day(args.date_from), _day(args.date_to, end=True))
    races = _sample_evenly(results.races_between(start, end), args.limit)
    print(f"対象レース: {len(races)}（データ版 {store.version_id}）")

    if args.dry_run:
        print(json.dumps(tbt.dry_run(races, repo, n=args.n, max_recent_runs=args.max_recent_runs),
                         ensure_ascii=False, indent=2))
        return 0

    problem = _check_final_guard(args, args.data_dir)
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"

    def write_row(row):
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")

    rows = tbt.run(races, repo, results, client, methods=methods, n=args.n,
                   tuning_until=tuning_until, allow_final=args.final,
                   max_recent_runs=args.max_recent_runs, save_records=args.save_records,
                   data_dir=args.data_dir, pace_definition_for=pace_definition_for,
                   on_row=write_row)
    rep = tbt.report(rows, n=args.n, run_info={
        "data_version": store.version_id, "from": args.date_from, "to": args.date_to,
        "tuning_until": args.tuning_until, "limit": args.limit, "methods": methods,
        "final": args.final, "max_recent_runs": args.max_recent_runs})
    (out_dir / "report.json").write_text(
        json.dumps(_to_jsonable(rep), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_to_jsonable(rep["by_period"]), ensure_ascii=False, indent=2))
    print(f"スキップ: {rep['skip_breakdown']}")
    print(f"使用量: {rep['usage']}")
    print(f"保存: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
