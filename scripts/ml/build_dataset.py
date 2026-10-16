"""Phase 1: 既存のデータ版 × 血統の版 → 学習用データセット + 整合性レポート。

    .venv/Scripts/python.exe scripts/ml/build_dataset.py
    .venv/Scripts/python.exe scripts/ml/build_dataset.py --race-version 20261001T000000Z --race-version 20261002T000000Z

整合性検査で FAIL が出ても保存はする（原因を調べるのに表が要るため）。学習スクリプトは
FAIL のあるデータセットを既定で拒否する。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba_ml.config import DEFAULT_CONFIG_PATH, ML_DATA_DIR, RACE_DATA_DIR, load_config  # noqa: E402
from keiba_ml.dataset import build, integrity  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--race-data-dir", type=Path, default=RACE_DATA_DIR,
                        help="既存のデータ版の場所（既定 %(default)s）")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--race-version", action="append", default=None,
                        help="読むデータ版（繰り返し可）。省略時は設定ファイル、それも空なら CURRENT")
    parser.add_argument("--dataset-id", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.race_version:
        config = config.with_overrides(
            data=replace(config.data, race_versions=tuple(args.race_version)))
    df, ped, names, report = build.build(config, race_data_dir=args.race_data_dir,
                                         ml_data_dir=args.ml_data_dir)
    out = build.save(df, ped, names, report, config, ml_data_dir=args.ml_data_dir,
                     dataset_id=args.dataset_id)
    res = df[df["has_result"]]
    print(f"データセット: {out}")
    print(f"  出走 {len(df):,} 行 / レース {df['race_id'].nunique():,}"
          f"（結果あり {len(res):,} 行）")
    if len(df):
        print(f"  期間 {df['race_date'].min().date()} 〜 {df['race_date'].max().date()}")
    stats = report["pedigree_stats"]
    print(f"  血統: 対象 {stats['horses']:,} 頭 / いずれかの祖先あり {stats['with_any_pedigree']:,}"
          f" / 出どころ {stats['source_counts']}")
    print(f"  世代別の埋まり率: {stats['fill_rate_by_generation']}")
    print("\n整合性検査:")
    print(integrity.summarize(report["checks"]))
    failed = [c for c in report["checks"] if c["status"] == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
