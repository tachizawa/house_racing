"""個別モデル（血統・距離適性 …）を学習し、固定のテスト期間で評価する。

    # 距離適性を10年で
    .venv/Scripts/python.exe scripts/ml/train.py --model distance --years 10
    # 学習期間の比較（設定ファイルの split.train_years。既定 10/20/30/40）
    .venv/Scripts/python.exe scripts/ml/train.py --model pedigree --compare
    # 目的関数を変えて比べる
    .venv/Scripts/python.exe scripts/ml/train.py --model distance --years 10 --loss YetiRank

設定は configs/ml/<model>.toml（--config で差し替え。[experiment] model が --model と
違うファイルは拒否する）。結果は data/ml/runs/<model>/<run_id>/ に出る。
comparison.md が期間別の比較表。
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
from keiba_ml.config import ML_DATA_DIR, MODEL_KINDS, config_path, load_config  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml.experiment import IntegrityError, run_comparison, to_markdown  # noqa: E402


def main(argv: list[str] | None = None, *, model: str | None = None) -> int:
    """model を渡すと --model を固定する（train_pedigree.py などの互換の入口用）。"""
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    if model is None:
        parser.add_argument("--model", required=True, choices=MODEL_KINDS,
                            help="学習する個別モデル")
    parser.add_argument("--config", type=Path, default=None,
                        help="設定ファイル（省略時 configs/ml/<model>.toml）")
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
    model = model or args.model

    try:
        config = load_config(args.config, model=model)
    except FileNotFoundError as error:
        parser.error(str(error))
    except ValueError as error:
        # 値の誤りと TOML の構文誤り（tomllib.TOMLDecodeError は ValueError の派生）。
        parser.error(f"設定ファイル {args.config or config_path(model)} を読めません: {error}")
    model_config = config.model
    if args.loss:
        model_config = replace(model_config, loss_function=args.loss)
    if args.iterations is not None:
        if args.iterations < 1:
            parser.error("--iterations は 1 以上にしてください。")
        model_config = replace(model_config, iterations=args.iterations)
    config = config.with_overrides(model=model_config)

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
    if any(y < 1 for y in years):
        # with_overrides は validate を通らないので、ここで弾かないと壊れた config.json が残る。
        parser.error("--years は 1 以上の年数を指定してください。")
    # 実際に回す学習期間を config.json に残す（--years 10 の実行を設定の全期間の比較と取り違えない）。
    config = config.with_overrides(split=replace(config.split, train_years=years))
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_root = args.ml_data_dir / "runs" / model
    if (out_root / run_id).exists():
        # 上書きすると前の実行の記録と今回の記録が混ざる。
        print(f"実行の出力先が既にあります: {out_root / run_id}。別の --run-id を指定してください。")
        return 2
    try:
        table = run_comparison(df, config, out_root=out_root, run_id=run_id,
                               train_years=years, allow_integrity_fail=args.allow_integrity_fail,
                               dataset_id=manifest["dataset_id"])
    except IntegrityError as error:
        print(error)
        return 2
    print(f"\n結果: {out_root / run_id}")
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
