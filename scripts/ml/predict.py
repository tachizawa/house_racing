"""学習済みの個別モデルで、結果の出ていないレース（出馬表）の上位3頭を出す。

    .venv/Scripts/python.exe scripts/ml/predict.py --model data/ml/runs/distance/<run_id>/10y
    .venv/Scripts/python.exe scripts/ml/predict.py --model ... --race-id 2026101805041101

どのモデル（血統・距離適性 …）かは学習時の記録（<run_id>/config.json の [experiment] model）
から決まる。対象は「データセットに入っているが結果が無いレース」（出馬表の data_kubun 1/2）。
データセットを作り直してから実行すること（build_dataset.py）。最終的な購入判断は
人が行う前提なので、スコアと一緒にモデルごとの根拠の列（registry の describe）を並べる。
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

from keiba import config as keiba_config  # noqa: E402
from keiba import console  # noqa: E402
from keiba_ml import registry  # noqa: E402
from keiba_ml.config import ML_DATA_DIR, MLConfig, config_from_dict  # noqa: E402
from keiba_ml.dataset import build  # noqa: E402
from keiba_ml.experiment import IntegrityError, build_model_features  # noqa: E402
from keiba_ml.models.catboost_model import CatBoostModel  # noqa: E402


def config_from_run(model_dir: Path) -> tuple[MLConfig, date]:
    """学習時の設定と学習開始日。どちらかが無ければ止める。

    既定の設定や全期間の履歴で代用すると、学習時と違う特徴量で予測することになり、
    スコアが黙って狂う。
    """
    run_config = model_dir.parent / "config.json"
    metrics = model_dir / "metrics.json"
    missing = [str(p) for p in (run_config, metrics) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"学習時の記録がありません: {missing}。--model には"
            " train.py が出力した data/ml/runs/<model>/<run_id>/<window> を渡してください。")
    config = config_from_dict(json.loads(run_config.read_text(encoding="utf-8")),
                              source=str(run_config))
    window = json.loads(metrics.read_text(encoding="utf-8"))["window"]
    return config, date.fromisoformat(window["train_start"])


def live_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(特徴量を作る行, 予想する行)。

    出馬表の段階で出走取消・除外（1/2/3）が付いた馬は走らないので、予想せず特徴量からも
    外し、結果の出ていないレースの頭数（field_size）を走る馬だけで数え直す。
    結果のある行は dataset/races.py が同じ規則で既に抜いている。
    """
    scratched = (~df["has_result"]
                 & df["result_ijyo_cd"].fillna("").str.strip().isin(keiba_config.IJYO_EXCLUDED))
    df = df[~scratched].copy()
    live = ~df["has_result"]
    df.loc[live, "field_size"] = (df[live].groupby("race_id")["umaban"]
                                  .transform("size").astype(df["field_size"].dtype))
    return df, df[live]


def main(argv: list[str] | None = None, *, expected_model: str | None = None) -> int:
    """expected_model を渡すと、それ以外のモデルの記録を拒否する（互換の入口用）。"""
    console.enable_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True,
                        help="data/ml/runs/<model>/<run_id>/<window>")
    parser.add_argument("--ml-data-dir", type=Path, default=ML_DATA_DIR)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--race-id", action="append", default=None)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        config, train_start = config_from_run(args.model)
    except FileNotFoundError as error:
        print(error)
        return 2
    spec = registry.for_config(config)
    if expected_model and spec.name != expected_model:
        print(f"{args.model} は {spec.title}モデル（{spec.name}）です。"
              f" scripts/ml/predict.py を使ってください。")
        return 2
    model = CatBoostModel.load(args.model)
    df, names, _ = build.load(args.ml_data_dir, args.dataset_id)
    if spec.stats_history(config) == "window":
        df = df[df["race_date"].dt.date >= train_start]
    df, targets = live_rows(df)
    if args.race_id:
        targets = targets[targets["race_id"].isin(args.race_id)]
    if targets.empty:
        print("結果の出ていないレースがデータセットにありません。"
              " 出馬表を取得してから build_dataset.py をやり直してください。")
        return 1
    try:
        features = build_model_features(df, config)
    except IntegrityError as error:
        print(error)
        return 2
    pred = model.predict(features, targets, fold="live")
    reasons = spec.describe(targets, features, names)
    view = pred.merge(targets[["race_id", "umaban", "bamei"]].join(reasons),
                      on=["race_id", "umaban"])
    for race_id, race in view.groupby("race_id"):
        race = race.sort_values("pred_rank")
        print(f"\n=== {race_id} ===")
        table = pd.DataFrame({"予測順位": race["pred_rank"], "馬番": race["umaban"],
                              "馬名": race["bamei"], "スコア": race["score"].round(4)})
        table = pd.concat([table, race[list(reasons.columns)]], axis=1)
        print(table.to_string(index=False))
        top = race.head(args.top_n)
        print(f"上位{args.top_n}頭: " + " / ".join(f"{u} {b}" for u, b in
                                                 zip(top["umaban"], top["bamei"])))
    print(f"\n※ スコアは{spec.title}モデル単体の値です。確率とは限りません。購入判断は人が行ってください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
