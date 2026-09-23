"""データ版の充足度を表示する。Jev は呼ばない。

    .venv/Scripts/python.exe scripts/data_coverage.py --from 20260901 --to 20260922
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from keiba import config  # noqa: E402
from keiba.data.repository import Repository, ResultRepository  # noqa: E402
from keiba.data.store import DataStore  # noqa: E402
from keiba.evaluate import coverage  # noqa: E402

JST = timezone(timedelta(hours=9))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", required=True, metavar="YYYYMMDD")
    parser.add_argument("--to", dest="date_to", required=True, metavar="YYYYMMDD")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args(argv)

    start = datetime.strptime(args.date_from, "%Y%m%d").replace(tzinfo=JST)
    end = datetime.strptime(args.date_to, "%Y%m%d").replace(
        hour=23, minute=59, second=59, tzinfo=JST)
    store = DataStore.open_current(args.data_dir)
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
