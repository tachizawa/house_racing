"""学習済みの個別モデルを、まだ何の判断にも使っていない期間で1回だけ評価する。

    .venv/Scripts/python.exe scripts/ml/evaluate.py --model data/ml/runs/distance/<run_id>/10y \
        --from 2026-01-01 --to 2026-10-18

どのモデル（血統・距離適性 …）かは学習時の記録（config.json の [experiment] model）から決まる。
train.py のテスト期間は、特徴量の組や学習期間を選ぶのに使うと「選んだ後の値」に
なり、独立な評価ではなくなる。そのための未使用期間をここで評価する。

- 期間はモデルのテスト期間の終わりより後でなければ受け付けない（学習・検証・テストと重ねない）
- 結果は <model>/holdout_<from>_<to>.json に書く。同じ期間の結果が既にあれば上書きしない
  （1回目の結果を残す。見てから設定を変えて評価し直すと、その期間も選択に使ったことになる）
- データセットは評価する期間の結果を含む新しいものでよいが、中身を決める設定（血統の出どころ・
  世代数など。build.config_mismatches）が学習時と違えば止める
- データセットの整合性検査に FAIL があれば止める（train.py と同じ）
- 学習時と同じ未来情報の検査（列名・as-of の検算。検算する行は評価する期間から選ぶ）を通す
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba_ml.config import ML_DATA_DIR, config_from_dict  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml import registry  # noqa: E402
from keiba_ml.experiment import IntegrityError, evaluate_period, write_json  # noqa: E402
from keiba_ml.models.catboost_model import CatBoostModel  # noqa: E402


def main(argv: list[str] | None = None, *, expected_model: str | None = None) -> int:
    """expected_model を渡すと、それ以外のモデルの記録を拒否する（互換の入口用）。"""
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True,
                        help="data/ml/runs/<model>/<run_id>/<window>")
    parser.add_argument("--from", dest="period_start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="period_end", type=date.fromisoformat, required=True)
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None)
    args = parser.parse_args(argv)

    run_config, metrics_path = args.model.parent / "config.json", args.model / "metrics.json"
    missing = [str(p) for p in (run_config, metrics_path) if not p.exists()]
    if missing:
        print(f"学習時の記録がありません: {missing}")
        return 2
    if args.period_end < args.period_start:
        parser.error("--to が --from より前です。")
    window = json.loads(metrics_path.read_text(encoding="utf-8"))["window"]
    test_end = date.fromisoformat(str(window["test_end"]))
    if args.period_start <= test_end:
        print(f"--from は、このモデルのテスト期間の終わり（{test_end}）より後にしてください。"
              " それより前は学習・検証・テストに使った期間です。")
        return 2
    out = args.model / f"holdout_{args.period_start:%Y%m%d}_{args.period_end:%Y%m%d}.json"
    if out.exists():
        print(f"この期間の評価は既にあります: {out}。1回目の結果を残すため上書きしません。")
        return 2

    config = config_from_dict(json.loads(run_config.read_text(encoding="utf-8")),
                              source=str(run_config))
    spec = registry.for_config(config)
    if expected_model and spec.name != expected_model:
        print(f"{args.model} は {spec.title}モデル（{spec.name}）です。"
              " scripts/ml/evaluate.py を使ってください。")
        return 2
    model = CatBoostModel.load(args.model)
    df, _names, manifest = build.load(args.ml_data_dir, args.dataset_id)
    if manifest.get("checks_failed"):
        # train.py と同じ関門。重複・壊れた行のあるデータで1回きりの評価を消費しない。
        print(f"データセット {manifest['dataset_id']} は整合性検査で FAIL があります:"
              f" {manifest['checks_failed']}")
        return 2
    # 学習時より新しいデータセット（評価する期間の結果が入ったもの）を使うのは正当なので
    # 版の ID は問わない。中身を決める設定（血統の出どころ・世代数など）は学習時と揃える。
    mismatches = build.config_mismatches(manifest, config)
    if mismatches:
        print(f"データセット {manifest['dataset_id']} は学習時と違う設定で作られています:"
              f" {mismatches}")
        return 2
    try:
        result = evaluate_period(df, config, model,
                                 train_start=date.fromisoformat(str(window["train_start"])),
                                 period_start=args.period_start, period_end=args.period_end)
    except IntegrityError as error:
        print(error)
        return 2
    result["model"] = str(args.model)
    result["dataset_id"] = manifest["dataset_id"]
    result["evaluated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(out, result)

    print(f"評価期間 {result['period']['first_race']} 〜 {result['period']['last_race']}"
          f"（{result['races']:,} レース）→ {out}")
    for name, res in result["results"].items():
        hits = res["ranking"]["hits"]
        print(f"  {name:20s} 的中数 {hits['mean']:.4f} [{hits['ci95'][0]:.4f}, {hits['ci95'][1]:.4f}]"
              f"  偶然 {hits['random']:.4f}")
    for name, diff in result["differences"].items():
        hits = diff["hits"]
        print(f"  {name:34s} 差 {hits['diff']:+.4f} [{hits['ci95'][0]:+.4f}, {hits['ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
