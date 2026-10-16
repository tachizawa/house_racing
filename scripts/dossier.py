"""1レース分の予想用資料を JSON と Markdown で書き出す。Jev は呼ばない。

    # データ版にある過去レースを、前日夜に予想した体で
    .venv/Scripts/python.exe scripts/dossier.py --race-id 2026080104020311
    # 利用者入力（scripts/predict.py と同じ形式）
    .venv/Scripts/python.exe scripts/dossier.py --input race.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore, DataVersionError  # noqa: E402
from keiba.evaluate.backtest import race_input_from_history  # noqa: E402
from keiba.features import dossier as dossier_mod  # noqa: E402
from keiba.features import dossier_markdown  # noqa: E402
from keiba.input.schema import InputError, load_race_input  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--race-id")
    source.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/dossiers"))
    parser.add_argument("--max-recent-runs", type=int, default=config.DEFAULT_DOSSIER_RECENT_RUNS)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args(argv)
    if args.max_recent_runs < 0:
        parser.error("--max-recent-runs は 0 以上にしてください")

    # scripts/predict.py と同じく、版が無い・入力が読めないときは traceback ではなく
    # 理由を1行出して 1 で終える。
    try:
        store = DataStore.open_current(args.data_dir)
    except DataVersionError as error:
        print(f"データ版を開けません: {error}")
        return 1
    repo = Repository(store)
    if args.race_id:
        results = ResultRepository(store)
        race = results.race(args.race_id)
        if race is None:
            print(f"race_id {args.race_id} はデータ版にありません。")
            return 1
        # 結果行の entry 部分だけを使う（EntryRow は結果の項目を持てない）。
        entries = [r.entry for r in results.results(args.race_id)]
        race_input = race_input_from_history(race, entries, top_n=config.DEFAULT_TOP_N)
        stem = args.race_id
    else:
        try:
            race_input = load_race_input(args.input)
        except InputError as error:
            print(f"入力を読み込めません: {error}")
            return 1
        stem = args.input.stem

    d = dossier_mod.build_dossier(race_input, repo, max_recent_runs=args.max_recent_runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{stem}.json").write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / f"{stem}.md").write_text(dossier_markdown.render(d), encoding="utf-8")
    chars = dossier_mod.state_chars(dossier_mod.to_state(d))
    print(f"書き出し: {args.out / stem}.json / .md（state {chars:,} 文字、"
          f"上限 {config.STATE_MAX_CHARS:,}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
