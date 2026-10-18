"""Phase 2: 血統モデル（CatBoost）を学習し、固定のテスト期間で評価する。

    # 10年で基準モデル
    .venv/Scripts/python.exe scripts/ml/train_pedigree.py --years 10
    # 学習期間の比較（設定ファイルの split.train_years。既定 10/20/30/40）
    .venv/Scripts/python.exe scripts/ml/train_pedigree.py --compare
    # 目的関数を変えて比べる
    .venv/Scripts/python.exe scripts/ml/train_pedigree.py --years 10 --loss YetiRank

結果は data/ml/runs/<run_id>/ に出る。comparison.md が期間別の比較表。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba_ml.config import DEFAULT_CONFIG_PATH, ML_DATA_DIR, load_config  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml.experiment import run_comparison, to_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None,
                        help=f"設定ファイル（省略時 {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None, help="省略時は datasets/CURRENT")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--years", type=int, action="append",
                       help="学習期間（年）。繰り返し可。既定は設定の先頭（10）")
    group.add_argument("--compare", action="store_true",
                       help="設定の split.train_years をすべて回して比較する")
    parser.add_argument("--loss", default=None, help="CatBoost の loss_function を上書きする")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--allow-integrity-fail", action="store_true",
                        help="整合性検査の FAIL を無視して進める（調査用。結果は信用しないこと）")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as error:
        parser.error(str(error))
    except ValueError as error:
        # 値の誤りと TOML の構文誤り（tomllib.TOMLDecodeError は ValueError の派生）。
        parser.error(f"設定ファイル {args.config or DEFAULT_CONFIG_PATH} を読めません: {error}")
    model = config.model
    if args.loss:
        model = replace(model, loss_function=args.loss)
    if args.iterations is not None:
        if args.iterations < 1:
            parser.error("--iterations は 1 以上にしてください。")
        model = replace(model, iterations=args.iterations)
    config = config.with_overrides(model=model)

    df, _names, manifest = build.load(args.ml_data_dir, args.dataset_id)
    if manifest.get("checks_failed") and not args.allow_integrity_fail:
        print(f"データセット {manifest['dataset_id']} は整合性検査で FAIL があります:"
              f" {manifest['checks_failed']}")
        print("data/ml/datasets/<id>/integrity.json を確認し、直してから学習してください。")
        return 2
    mismatches = build.config_mismatches(manifest, config)
    if mismatches:
        # 世代数などが違うと、作ったときと別の設定で学習したことが config.json に残る。
        print(f"データセット {manifest['dataset_id']} は今の設定と違う設定で作られています:"
              f" {mismatches}")
        print("設定を作ったときに合わせるか、build_dataset.py でデータセットを作り直してください。")
        return 2

    years = tuple(config.split.train_years) if args.compare else \
        tuple(args.years or config.split.train_years[:1])
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    table = run_comparison(df, config, out_root=args.ml_data_dir / "runs", run_id=run_id,
                           train_years=years, allow_integrity_fail=args.allow_integrity_fail)
    print(f"\n結果: {args.ml_data_dir / 'runs' / run_id}")
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
