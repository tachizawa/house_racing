"""データ版の充足度を表示する。Jev は呼ばない。

    .venv/Scripts/python.exe scripts/data_coverage.py --from 20260901 --to 20260922

--from / --to は backtest.py・backtest_top3.py と同じく予想締切（発走の
PREDICTION_CUTOFF_HOURS 時間前）の日付。同じ範囲を渡せばバックテストと同じレースを数える。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config, console  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore, DataVersionError  # noqa: E402
from keiba.evaluate import coverage  # noqa: E402
from keiba.evaluate.backtest import hasso_window_for_cutoff_window  # noqa: E402

JST = timezone(timedelta(hours=9))


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD")
    parser.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args(argv)

    # 発走時刻ではなく予想締切で絞る（バックテストの --from / --to と同じ窓にするため）。
    start, end = hasso_window_for_cutoff_window(
        datetime.strptime(args.date_from, "%Y%m%d").replace(tzinfo=JST),
        datetime.strptime(args.date_to, "%Y%m%d").replace(hour=23, minute=59, second=59, tzinfo=JST))
    try:
        store = DataStore.open_current(args.data_dir)
    except DataVersionError as error:
        print(f"データ版を開けません: {error}")
        return 1
    repo, results = Repository(store), ResultRepository(store)
    report = coverage.measure(results.races_between(start, end), repo, results)

    print(f"データ版: {store.version_id}  収録開始: {report.coverage_from}")
    print(f"対象レース: {report.n_races}  延べ出走馬: {report.n_horses}")
    print(f"締切前の出走記録あり: {report.n_with_prior_run}"
          f"（{report.share_with_prior_run:.1%}）")
    print(f"締切前の出走記録3走以上: {report.n_with_3plus_prior_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
