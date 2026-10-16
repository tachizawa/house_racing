"""学習済みの血統モデルで、結果の出ていないレース（出馬表）の上位3頭を出す。

    .venv/Scripts/python.exe scripts/ml/predict_pedigree.py --model data/ml/runs/<run_id>/10y
    .venv/Scripts/python.exe scripts/ml/predict_pedigree.py --model ... --race-id 2026101805041101

対象は「データセットに入っているが結果が無いレース」（出馬表の data_kubun 1/2）。
データセットを作り直してから実行すること（build_dataset.py）。最終的な購入判断は
人が行う前提なので、スコアと一緒に父・母父を並べて根拠を見られるようにする。
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
from keiba_ml.config import ML_DATA_DIR, MLConfig, config_from_dict, load_config  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml.features.pedigree import build_features  # noqa: E402
from keiba_ml.models.pedigree_catboost import PedigreeCatBoost  # noqa: E402


def _config_from_run(model_dir: Path) -> tuple[MLConfig, date | None]:
    run_config = model_dir.parent / "config.json"
    config = load_config(None)
    if run_config.exists():
        config = config_from_dict(json.loads(run_config.read_text(encoding="utf-8")),
                                  source=str(run_config))
    metrics = model_dir / "metrics.json"
    train_start = None
    if metrics.exists():
        window = json.loads(metrics.read_text(encoding="utf-8"))["window"]
        train_start = date.fromisoformat(window["train_start"])
    return config, train_start


def main(argv: list[str] | None = None) -> int:
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="data/ml/runs/<run_id>/<window>")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--race-id", action="append", default=None)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args(argv)

    config, train_start = _config_from_run(args.model)
    model = PedigreeCatBoost.load(args.model)
    df, names, _ = build.load(args.ml_data_dir, args.dataset_id)
    if config.pedigree.stats_history == "window" and train_start is not None:
        df = df[df["race_date"].dt.date >= train_start]
    targets = df[~df["has_result"]]
    if args.race_id:
        targets = targets[targets["race_id"].isin(args.race_id)]
    if targets.empty:
        print("結果の出ていないレースがデータセットにありません。"
              " 出馬表を取得してから build_dataset.py をやり直してください。")
        return 1
    features = build_features(df, pedigree=config.pedigree, features=config.features)
    pred = model.predict(features, targets, fold="live")
    view = pred.merge(targets[["race_id", "umaban", "bamei", "ped_S", "ped_DS"]],
                      on=["race_id", "umaban"])
    view["父"] = view["ped_S"].map(names).fillna("")
    view["母父"] = view["ped_DS"].map(names).fillna("")
    for race_id, race in view.groupby("race_id"):
        race = race.sort_values("pred_rank")
        print(f"\n=== {race_id} ===")
        table = pd.DataFrame({
            "予測順位": race["pred_rank"], "馬番": race["umaban"], "馬名": race["bamei"],
            "スコア": race["score"].round(4), "父": race["父"], "母父": race["母父"],
        })
        print(table.to_string(index=False))
        top = race.head(args.top_n)
        print(f"上位{args.top_n}頭: " + " / ".join(f"{u} {b}" for u, b in
                                                 zip(top["umaban"], top["bamei"])))
    print("\n※ スコアは血統モデル単体の値です。確率とは限りません。購入判断は人が行ってください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
