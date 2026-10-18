"""学習期間ごとの実験（Phase 2「10年で基準モデル → 20/30/40年へ拡張 → 期間別の比較」）。

1つの学習期間（Window）について:
  1. 産駒成績の履歴を決める（stats_history="window" なら学習開始日以降だけ）
  2. 特徴量を作る（as-of。テスト期間の行もテスト日より前の情報だけで作られる）
  3. 時系列で分割し、未来情報の検査（列名・分割・as-of の総当たり検算）を通す
  4. CatBoost を学習（検証期間で early stopping）
  5. テスト期間を予測し、モデルと比較用の基準（偶然・父 A/E だけ）を評価する

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

from keiba_ml.config import MLConfig
from keiba_ml.dataset import integrity
from keiba_ml.dataset.split import Window, history_frame, make_window, split_frame
from keiba_ml.evaluate import metrics
from keiba_ml.features.pedigree import build_features
from keiba_ml.models.base import make_predictions
from keiba_ml.models.pedigree_catboost import MODEL_NAME, PedigreeCatBoost


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


def baseline_predictions(features, rows: pd.DataFrame, *, fold: str) -> dict[str, pd.DataFrame]:
    """学習しない比較対象。CatBoost がこれを超えないなら学習の意味がない。

    - sire_ae: 父の産駒の as-of 3着以内 A/E だけで並べる（血統の最も素朴な使い方）
    - sire_cond_ae: 父 × 芝ダ × 距離区分 の as-of A/E だけで並べる
    """
    frame = features.frame.loc[rows.index]
    out = {}
    for name, col in (("sire_ae", "st_S_ae_top3"), ("sire_cond_ae", "st_S_ae_top3_surface_dband")):
        if col in frame:
            out[name] = make_predictions(rows, frame[col].to_numpy(), model_name=name,
                                         model_version="heuristic", fold=fold)
    return out


def run_window(df: pd.DataFrame, config: MLConfig, window: Window, *, out_dir: Path,
               run_id: str, allow_integrity_fail: bool = False,
               log_period: int = 200) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = history_frame(df, window, config.pedigree.stats_history)
    features = build_features(history, pedigree=config.pedigree, features=config.features)
    train, valid, test = split_frame(history, window)

    checks = [
        integrity.check_feature_columns(features.columns),
        integrity.check_split(train, valid, test),
    ]
    checks.append(integrity.check_asof_samples(
        history, features.frame, paths=config.pedigree.stat_paths,
        shrinkage=config.pedigree.shrinkage, exclude_self=config.pedigree.exclude_self,
        sample=100))
    write_json(out_dir / "checks.json", checks)
    failed = [c for c in checks if c["status"] == "fail"]
    if failed and not allow_integrity_fail:
        raise IntegrityError("整合性検査に失敗しました:\n" + integrity.summarize(failed))
    if train.empty or valid.empty or test.empty:
        raise IntegrityError(
            f"{window.label}: 学習 {len(train)} 行 / 検証 {len(valid)} 行 / テスト {len(test)} 行。"
            " 空の区間があります。split の日付とデータの期間を確かめてください。")

    model = PedigreeCatBoost(config.model, model_version=f"{run_id}/{window.label}")
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
    results = {MODEL_NAME: {"valid": metrics.evaluate(pred_valid, valid, **kwargs),
                            "test": metrics.evaluate(pred_test, test, **kwargs)}}
    for name, pred in baseline_predictions(features, test, fold="test").items():
        results[name] = {"test": metrics.evaluate(pred, test, **kwargs)}

    summary = {
        "window": {**asdict(window), "truncated": window.truncated, "label": window.label},
        "rows": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
        "races": {"train": int(train["race_id"].nunique()),
                  "valid": int(valid["race_id"].nunique()),
                  "test": int(test["race_id"].nunique())},
        "best_iteration": model.best_iteration,
        "loss_function": config.model.loss_function,
        "results": results,
        "checks": [{"name": c["name"], "status": c["status"]} for c in checks],
    }
    write_json(out_dir / "metrics.json", summary)
    return summary


def run_comparison(df: pd.DataFrame, config: MLConfig, *, out_root: Path, run_id: str,
                   train_years: tuple[int, ...] | None = None,
                   allow_integrity_fail: bool = False, log_period: int = 200) -> pd.DataFrame:
    """同じテスト期間に対して学習期間だけを変えて比較する。"""
    train_years = train_years or config.split.train_years
    res = df[df["has_result"]]
    first = res["race_date"].min().date()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(config))
    rows, test_races, previous_start = [], None, None
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
    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    (run_dir / "comparison.md").write_text(to_markdown(table), encoding="utf-8")
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
