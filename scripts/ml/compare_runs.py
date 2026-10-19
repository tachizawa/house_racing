"""学習期間の比較（train_pedigree.py --compare）の差を、保存済みの予測から出し直す。

    .venv/Scripts/python.exe scripts/ml/compare_runs.py --run data/ml/runs/<run_id>

学習した順の先頭の期間（comparison.csv の並び。既定の設定なら 10y）に対する各期間の
的中数の差を、同じテストレースでの対応ありブートストラップ（evaluate.metrics.paired_difference。
回数は設定の evaluate.bootstrap）で出し、<run_dir>/window_differences.md|csv に書く。学習はやり直さない。

正解は学習に使ったデータセットから取る（<run_dir>/dataset.json）。記録の無い実行では
--dataset-id が要る。どちらの場合も、保存した予測のレース・馬とデータセットのテスト期間の
レース・馬が完全に一致しなければ止める（違う正解と突き合わせた、もっともらしい表を出さない）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from keiba import console  # noqa: E402
from keiba_ml.config import ML_DATA_DIR, config_from_dict  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml.dataset.split import Window, split_frame  # noqa: E402
from keiba_ml.experiment import to_markdown, write_window_differences  # noqa: E402
from keiba_ml.models.pedigree_catboost import MODEL_NAME  # noqa: E402


def trained_order(run_dir: Path) -> list[str]:
    """学習した順の期間名（comparison.csv の並び）。並べ替えると差の基準が変わる。"""
    table = pd.read_csv(run_dir / "comparison.csv")
    return list(dict.fromkeys(table.loc[table["model"] == MODEL_NAME, "window"]))


def resolve_dataset_id(run_dir: Path, requested: str | None) -> tuple[str | None, str]:
    """(使うデータセット ID, 止める理由)。理由が空でなければ止める。"""
    record = run_dir / "dataset.json"
    recorded = (json.loads(record.read_text(encoding="utf-8"))["dataset_id"]
                if record.exists() else None)
    if recorded and requested and requested != recorded:
        return None, (f"この実行はデータセット {recorded} で学習しています。"
                      f" --dataset-id {requested} とは突き合わせません。")
    if not recorded and not requested:
        return None, (f"{record} がありません（記録を残す前の実行）。"
                      " 学習に使ったデータセットを --dataset-id で指定してください。")
    return recorded or requested, ""


def mismatched_rows(run_dir: Path, labels: list[str], truth: pd.DataFrame) -> list[str]:
    """保存した予測とテスト期間の正解で、レース・馬の組が違う期間と件数。"""
    expected = set(zip(truth["race_id"], truth["umaban"]))
    problems = []
    for label in labels:
        pred = pd.read_parquet(run_dir / label / "predictions_test.parquet",
                               columns=["race_id", "umaban"])
        saved = set(zip(pred["race_id"], pred["umaban"]))
        if saved != expected:
            problems.append(f"{label}: 予測にだけある {len(saved - expected)} 行 /"
                            f" 正解にだけある {len(expected - saved)} 行")
    return problems


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="data/ml/runs/<run_id>")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None,
                        help="学習に使ったデータセット（dataset.json の無い実行だけ必要）")
    args = parser.parse_args(argv)

    run_config = args.run / "config.json"
    missing = [str(p) for p in (run_config, args.run / "comparison.csv") if not p.exists()]
    if missing:
        print(f"学習時の記録がありません: {missing}")
        return 2
    config = config_from_dict(json.loads(run_config.read_text(encoding="utf-8")),
                              source=str(run_config))
    labels = trained_order(args.run)
    if len(labels) < 2:
        print(f"比べる学習期間が2つ以上ありません: {labels}")
        return 2
    dataset_id, reason = resolve_dataset_id(args.run, args.dataset_id)
    if reason:
        print(reason)
        return 2
    saved = json.loads((args.run / labels[0] / "metrics.json").read_text(encoding="utf-8"))["window"]
    window = Window(train_years=saved["train_years"],
                    **{k: date.fromisoformat(str(saved[k])) for k in
                       ("train_start", "actual_train_start", "valid_start", "test_start", "test_end")})
    df, _names, _ = build.load(args.ml_data_dir, dataset_id)
    _, _, test = split_frame(df, window)
    problems = mismatched_rows(args.run, labels, test)
    if problems:
        print(f"データセット {dataset_id} のテスト期間は、保存した予測と同じレース・馬ではありません:")
        print("\n".join(f"  {p}" for p in problems))
        return 2
    table = write_window_differences(args.run, labels, test, config)
    print(to_markdown(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
