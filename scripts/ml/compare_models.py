"""学習済みの2つのモデルを、同じ期間・同じレースで比べる（対応ありブートストラップ）。

    .venv/Scripts/python.exe scripts/ml/compare_models.py \
        --model data/ml/runs/combined/<new>/10y --versus data/ml/runs/combined/<old>/10y \
        --from 2026-01-01 --to 2026-10-18

特徴量は --model の学習時の設定で作る。--versus のモデルはその表から自分の列だけを使うので、
--versus の列が --model の設定の特徴量に含まれていなければ止める（総合モデルの追加特徴量は
列を足すだけなので、旧版の列はすべて含まれる）。期間は両方のモデルのテスト期間より後に限る。
結果は --model の下に compare_<from>_<to>.json として書く（評価の記録。上書きしない）。

--blank c_weight --blank c_zogen で、前日には分からない馬体重を欠損にした「前日予想の条件」で比べる
（その列を持たないモデル、たとえば combined.use_weight = false で学習したモデルには影響しない。
--model の特徴量に無い --blank の列は欠損の列として足す）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from keiba import console  # noqa: E402
from keiba_ml.config import ML_DATA_DIR, config_from_dict  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml import registry  # noqa: E402
from keiba_ml.dataset.split import history_start  # noqa: E402
from keiba_ml.evaluate import metrics  # noqa: E402
from keiba_ml.experiment import write_json  # noqa: E402
from keiba_ml.models.catboost_model import CatBoostModel  # noqa: E402


def _window(model_dir: Path) -> dict:
    return json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["window"]


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--versus", type=Path, required=True)
    parser.add_argument("--from", dest="period_start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="period_end", type=date.fromisoformat, required=True)
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--blank", action="append", default=[],
                        help="期間の行でこの列を欠損にして両方のモデルで予想する（繰り返し可）。"
                             " 前日には分からない馬体重（c_weight・c_zogen）など")
    args = parser.parse_args(argv)

    suffix = "_blank-" + "-".join(args.blank) if args.blank else ""
    out = args.model / (f"compare_{args.period_start:%Y%m%d}_{args.period_end:%Y%m%d}"
                        f"{suffix}.json")
    if out.exists():
        print(f"この期間の比較は既にあります: {out}。上書きしません。")
        return 2
    windows = {"model": _window(args.model), "versus": _window(args.versus)}
    for name, window in windows.items():
        if args.period_start <= date.fromisoformat(str(window["test_end"])):
            print(f"--from は {name} のテスト期間の終わり（{window['test_end']}）より後にしてください。")
            return 2
    config = config_from_dict(
        json.loads((args.model.parent / "config.json").read_text(encoding="utf-8")),
        source=str(args.model.parent / "config.json"))
    spec = registry.for_config(config)
    new, old = CatBoostModel.load(args.model), CatBoostModel.load(args.versus)
    df, _names, manifest = build.load(args.ml_data_dir, args.dataset_id)
    if manifest.get("checks_failed"):
        print(f"データセット {manifest['dataset_id']} は整合性検査で FAIL があります。")
        return 2
    train_start = date.fromisoformat(str(windows["model"]["train_start"]))
    day = df["race_date"].dt.date
    mask = day <= args.period_end
    start = history_start(train_start, spec.stats_history(config))
    if start is not None:
        mask &= day >= start
    history = df[mask]
    features = spec.build_features(history, config)
    missing = [c for c in old.feature_columns if c not in features.columns]
    if [c for c in missing if c not in args.blank]:
        print(f"--versus の列が特徴量にありません: {missing[:10]}")
        return 2
    rows = history[history["has_result"] & (history["race_date"].dt.date >= args.period_start)]
    if rows.empty:
        print("期間に結果のあるレースがありません。")
        return 2
    if args.blank:
        # 予想時に分からない列（前日の馬体重など）を欠損にして、その条件での予想を比べる
        frame = features.frame.copy()
        for col in args.blank:
            frame.loc[rows.index, col] = np.nan
        features = replace(features, frame=frame)
    ev = config.evaluate
    pred_new = new.predict(features, rows, fold="holdout")
    pred_old = old.predict(features, rows, fold="holdout")
    kwargs = dict(top_n=ev.top_n, bins=ev.calibration_bins, n_boot=ev.bootstrap,
                  seed=ev.bootstrap_seed)
    result = {
        "model": str(args.model), "versus": str(args.versus),
        "period": {"start": str(args.period_start), "end": str(args.period_end),
                   "first_race": str(rows["race_date"].min().date()),
                   "last_race": str(rows["race_date"].max().date())},
        "races": int(rows["race_id"].nunique()), "dataset_id": manifest["dataset_id"],
        "blank": args.blank,
        "results": {"model": metrics.evaluate(pred_new, rows, **kwargs),
                    "versus": metrics.evaluate(pred_old, rows, **kwargs)},
        "difference": metrics.paired_difference(pred_new, pred_old, rows, top_n=ev.top_n,
                                                n_boot=ev.bootstrap, seed=ev.bootstrap_seed),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(out, result)
    hits = {k: v["ranking"]["hits"] for k, v in result["results"].items()}
    print(f"期間 {result['period']['first_race']} 〜 {result['period']['last_race']}"
          f"（{result['races']:,} レース）→ {out}")
    for name, h in hits.items():
        print(f"  {name:8s} 的中数 {h['mean']:.4f} [{h['ci95'][0]:.4f}, {h['ci95'][1]:.4f}]"
              f"  偶然 {h['random']:.4f}")
    d = result["difference"]["hits"]
    print(f"  差 {d['diff']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
