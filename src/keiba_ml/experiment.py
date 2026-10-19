"""学習期間ごとの実験（Phase 2「10年で基準モデル → 20/30/40年へ拡張 → 期間別の比較」）。

どの個別モデル（血統・距離適性 …）も同じ手順で学習・評価する。モデルごとに違う部分
（特徴量・比較用の基準・未来情報の検算）は設定の ``[experiment] model`` から
``registry`` の ModelSpec を引いて使う。

1つの学習期間（Window）について:
  1. 履歴を決める（stats_history="window" なら学習開始日以降だけ）
  2. 特徴量を作る（as-of。テスト期間の行もテスト日より前の情報だけで作られる）
  3. 時系列で分割し、未来情報の検査（列名・分割・as-of の総当たり検算）を通す
  4. CatBoost を学習（検証期間で early stopping）
  5. テスト期間を予測し、モデルと比較用の基準（偶然・学習しない目安）を評価する

出力（data/ml/runs/<run_id>/<window>/）:
  model.cbm / model_meta.json / predictions_valid.parquet / predictions_test.parquet /
  feature_importance.csv / metrics.json / checks.json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_ml import registry
from keiba_ml.config import MLConfig
from keiba_ml.dataset import integrity
from keiba_ml.dataset.split import Window, history_frame, make_window, split_frame
from keiba_ml.evaluate import metrics
from keiba_ml.features.base import FeatureSet
from keiba_ml.models.catboost_model import CatBoostModel


class IntegrityError(RuntimeError):
    """未来情報の混入など、学習を進めてはいけない状態。"""


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (date, pd.Timestamp)):
        return str(value)
    return str(value)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")


def baseline_predictions(features: FeatureSet, rows: pd.DataFrame, *, fold: str,
                         spec: registry.ModelSpec | None = None) -> dict[str, pd.DataFrame]:
    """学習しない比較対象（ModelSpec.baselines）。CatBoost がこれを超えないなら学習の意味がない。

    spec を省くと血統モデルの基準（父 A/E だけ・父 × 芝ダ × 距離区分の A/E だけ）。
    """
    spec = spec or registry.get("pedigree")
    return spec.baselines(features, rows, fold)


def require_columns(df: pd.DataFrame, spec: registry.ModelSpec) -> None:
    missing = spec.missing_columns(df)
    if missing:
        raise IntegrityError(
            f"{spec.title}モデルに要る列がデータセットにありません: {missing}。"
            " scripts/ml/build_dataset.py でデータセットを作り直してください。")


def build_model_features(df: pd.DataFrame, config: MLConfig) -> FeatureSet:
    """設定のモデルの特徴量（予想スクリプトからも使う）。"""
    spec = registry.for_config(config)
    require_columns(df, spec)
    return spec.build_features(df, config)


def run_window(df: pd.DataFrame, config: MLConfig, window: Window, *, out_dir: Path,
               run_id: str, allow_integrity_fail: bool = False,
               log_period: int = 200) -> dict:
    spec = registry.for_config(config)
    require_columns(df, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = history_frame(df, window, spec.stats_history(config))
    features = spec.build_features(history, config)
    train, valid, test = split_frame(history, window)

    checks = [
        integrity.check_feature_columns(features.columns),
        integrity.check_split(train, valid, test),
        *spec.asof_checks(history, features, config, None),
    ]
    write_json(out_dir / "checks.json", checks)
    failed = [c for c in checks if c["status"] == "fail"]
    if failed and not allow_integrity_fail:
        raise IntegrityError("整合性検査に失敗しました:\n" + integrity.summarize(failed))
    if train.empty or valid.empty or test.empty:
        raise IntegrityError(
            f"{window.label}: 学習 {len(train)} 行 / 検証 {len(valid)} 行 / テスト {len(test)} 行。"
            " 空の区間があります。split の日付とデータの期間を確かめてください。")

    model = CatBoostModel(config.model, model_name=spec.model_name,
                          model_version=f"{run_id}/{window.label}")
    model.fit(features, train, valid, log_period=log_period)
    model.save(out_dir)
    pred_valid = model.predict(features, valid, fold="valid")
    pred_test = model.predict(features, test, fold="test")
    pred_valid.to_parquet(out_dir / "predictions_valid.parquet", index=False)
    pred_test.to_parquet(out_dir / "predictions_test.parquet", index=False)
    model.feature_importance().to_csv(out_dir / "feature_importance.csv", index=False,
                                      encoding="utf-8-sig")

    ev = config.evaluate
    kwargs = dict(top_n=ev.top_n, bins=ev.calibration_bins, n_boot=ev.bootstrap,
                  seed=ev.bootstrap_seed)
    results = {spec.model_name: {"valid": metrics.evaluate(pred_valid, valid, **kwargs),
                                 "test": metrics.evaluate(pred_test, test, **kwargs)}}
    for name, pred in baseline_predictions(features, test, fold="test", spec=spec).items():
        results[name] = {"test": metrics.evaluate(pred, test, **kwargs)}

    summary = {
        "window": {**asdict(window), "truncated": window.truncated, "label": window.label},
        "rows": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "races": {"train": int(train["race_id"].nunique()),
                  "valid": int(valid["race_id"].nunique()),
                  "test": int(test["race_id"].nunique())},
        "model": spec.name,
        "best_iteration": model.best_iteration,
        "loss_function": config.model.loss_function,
        "results": results,
        "checks": [{"name": c["name"], "status": c["status"]} for c in checks],
    }
    write_json(out_dir / "metrics.json", summary)
    return summary


def evaluate_period(df: pd.DataFrame, config: MLConfig, model: CatBoostModel, *,
                    train_start: date, period_start: date, period_end: date) -> dict:
    """学習済みのモデルを、学習・検証・テストのどれにも使っていない期間で評価する。

    特徴量は学習時と同じ規則（stats_history が "window" なら train_start 以降の履歴）で、
    period_end までの行から as-of で作る。基準（ModelSpec.baselines）との差は
    同じレースでの対応ありブートストラップ。
    """
    spec = registry.for_config(config)
    require_columns(df, spec)
    day = df["race_date"].dt.date
    mask = day <= period_end
    if spec.stats_history(config) == "window":
        mask &= day >= train_start
    history = df[mask]
    features = spec.build_features(history, config)
    rows = history[history["has_result"]
                   & (history["race_date"].dt.date >= period_start)]
    if rows.empty:
        raise IntegrityError(f"{period_start}〜{period_end} に結果のあるレースがありません。")
    # 学習時（run_window）と同じ検査。as-of の検算は評価する期間の行から選ぶ。
    checks = [
        integrity.check_feature_columns(features.columns),
        *spec.asof_checks(history, features, config, rows.index),
    ]
    failed = [c for c in checks if c["status"] == "fail"]
    if failed:
        raise IntegrityError("整合性検査に失敗しました:\n" + integrity.summarize(failed))
    ev = config.evaluate
    kwargs = dict(top_n=ev.top_n, bins=ev.calibration_bins, n_boot=ev.bootstrap,
                  seed=ev.bootstrap_seed)
    pred = model.predict(features, rows, fold="holdout")
    results = {spec.model_name: metrics.evaluate(pred, rows, **kwargs)}
    differences = {}
    for name, base in baseline_predictions(features, rows, fold="holdout", spec=spec).items():
        results[name] = metrics.evaluate(base, rows, **kwargs)
        differences[f"{spec.model_name}-{name}"] = metrics.paired_difference(
            pred, base, rows, top_n=ev.top_n, n_boot=ev.bootstrap, seed=ev.bootstrap_seed)
    return {
        "period": {"start": str(period_start), "end": str(period_end),
                   "first_race": str(rows["race_date"].min().date()),
                   "last_race": str(rows["race_date"].max().date())},
        "races": int(rows["race_id"].nunique()), "rows": int(len(rows)),
        "results": results, "differences": differences, "checks": checks,
    }


def run_comparison(df: pd.DataFrame, config: MLConfig, *, out_root: Path, run_id: str,
                   train_years: tuple[int, ...] | None = None,
                   allow_integrity_fail: bool = False, log_period: int = 200,
                   dataset_id: str | None = None) -> pd.DataFrame:
    """同じテスト期間に対して学習期間だけを変えて比較する。

    dataset_id を渡すと run_dir/dataset.json に残す。保存した予測を後から正解と突き合わせる
    （compare_runs.py）とき、同じデータセットを使うため。
    """
    train_years = train_years or config.split.train_years
    res = df[df["has_result"]]
    first = res["race_date"].min().date()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(config))
    if dataset_id is not None:
        write_json(run_dir / "dataset.json", {"dataset_id": dataset_id})
    rows, test_races, previous_start = [], None, None
    labels: list[str] = []
    for years in train_years:
        window = make_window(config.split, years, first)
        if window.truncated:
            print(f"注意: {window.label} はデータが {window.actual_train_start} からしか無く、"
                  f"設定の開始日 {window.train_start} まで遡れません（比較表の truncated=True）。")
            if previous_start == window.actual_train_start:
                print(f"      直前の期間と同じデータになるので {window.label} は学習しません。")
                continue
        previous_start = window.actual_train_start
        summary = run_window(df, config, window, out_dir=run_dir / window.label,
                             run_id=run_id, allow_integrity_fail=allow_integrity_fail,
                             log_period=log_period)
        current = summary["races"]["test"]
        if test_races is not None and current != test_races:
            raise IntegrityError("学習期間によってテストのレース数が変わりました。"
                                 f" {test_races} → {current}")
        test_races = current
        rows.extend(comparison_rows(summary))
        labels.append(window.label)
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    (run_dir / "comparison.md").write_text(to_markdown(table), encoding="utf-8")
    if len(labels) > 1:
        _, _, test = split_frame(df, window)
        write_window_differences(run_dir, labels, test, config)
    return table


def window_differences(run_dir: Path, labels: list[str], truth: pd.DataFrame,
                       config: MLConfig) -> pd.DataFrame:
    """先頭の学習期間に対する、各学習期間の差（同じテストレースでの対応ありブートストラップ）。

    保存済みの predictions_test.parquet から計算するので、学習をやり直さずに出し直せる
    （scripts/ml/compare_runs.py）。
    """
    ev = config.evaluate
    base = pd.read_parquet(run_dir / labels[0] / "predictions_test.parquet")
    rows = []
    for label in labels[1:]:
        other = pd.read_parquet(run_dir / label / "predictions_test.parquet")
        diff = metrics.paired_difference(other, base, truth, top_n=ev.top_n,
                                         n_boot=ev.bootstrap, seed=ev.bootstrap_seed)
        hits = diff["hits"]
        rows.append({"window": label, "versus": labels[0], "races": diff["races"],
                     "hits": hits["a"], "hits_versus": hits["b"], "hits_diff": hits["diff"],
                     "diff_ci_low": hits["ci95"][0], "diff_ci_high": hits["ci95"][1],
                     "races_differing": hits["races_differing"],
                     "ndcg_diff": diff["ndcg"]["diff"]})
    return pd.DataFrame(rows)


def write_window_differences(run_dir: Path, labels: list[str], truth: pd.DataFrame,
                             config: MLConfig) -> pd.DataFrame:
    table = window_differences(run_dir, labels, truth, config)
    table.to_csv(run_dir / "window_differences.csv", index=False, encoding="utf-8-sig")
    (run_dir / "window_differences.md").write_text(to_markdown(table), encoding="utf-8")
    return table


def comparison_rows(summary: dict) -> list[dict]:
    window = summary["window"]
    rows = []
    for model_name, by_fold in summary["results"].items():
        test = by_fold["test"]
        ranking = test["ranking"]
        row = {
            "window": window["label"], "train_start": str(window["train_start"]),
            "actual_train_start": str(window["actual_train_start"]),
            "truncated": window["truncated"], "model": model_name,
            "train_races": summary["races"]["train"], "test_races": ranking.get("races"),
            "hits_mean": ranking["hits"]["mean"], "hits_ci_low": ranking["hits"]["ci95"][0],
            "hits_ci_high": ranking["hits"]["ci95"][1],
            "hits_random": ranking["hits"]["random"],
            "all3_rate": ranking["all_hit"]["mean"], "all3_random": ranking["all_hit"]["random"],
            "winner_in_top3": ranking["winner_in_pred"]["mean"],
            "top1_win": ranking["top1_win"]["mean"], "ndcg3": ranking["ndcg"]["mean"],
        }
        cal = test.get("calibration")
        if cal:
            row.update({"brier": cal["brier"], "brier_base": cal["brier_base_rate"],
                        "log_loss": cal["log_loss"], "ece": cal["ece"]})
        rows.append(row)
    return rows


def to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "(結果なし)\n"
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in table.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
