"""Phase 1 の出口: 出走表 × 血統 を1つの表にし、整合性レポートを付けて保存する。

    data/ml/datasets/<dataset_id>/
        runners.parquet     1行 = 1頭 × 1レース（結果列は target_/result_ 接頭辞）
        pedigree.parquet    血統登録番号 → 祖先表（dataset/pedigree.py）
        ancestor_names.json 繁殖登録番号 → 馬名（レポート表示用）
        integrity.json      整合性検査の結果（dataset/integrity.py）
        manifest.json       入力の版・設定・行数
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from keiba_ml.config import MLConfig
from keiba_ml.dataset import integrity
from keiba_ml.dataset.pedigree import ancestor_names, build_pedigree, load_sources
from keiba_ml.dataset.races import load_runners

DATASET_SCHEMA_VERSION = "1"


def attach_pedigree(runners: pd.DataFrame, ped: pd.DataFrame) -> pd.DataFrame:
    """出走表に祖先表を横付けし、年齢を確定する。

    年齢は「レースの年 - 生年（血統から）」を優先する。表記上の馬齢（数え/満）を
    経由しないので、年代による表記の違いに左右されない。生年が無い馬だけ、
    表記の馬齢を CodeConfig に従って満年齢に直した値を使う。
    """
    df = runners.merge(ped, left_on="ketto_num", right_index=True, how="left")
    ped_cols = [c for c in ped.columns if c.startswith("ped_")]
    df[ped_cols] = df[ped_cols].fillna("")
    df["pedigree_source"] = df["pedigree_source"].fillna("")
    df["inbreeding_f"] = df["inbreeding_f"].astype(float)
    df["n_cross"] = df["n_cross"].astype("Int64")
    by_birth = df["year"] - df["birth_year"].astype("Float64")
    by_birth = by_birth.where(by_birth >= 1)
    df["age"] = by_birth.fillna(df["age_from_barei"].astype("Float64")).astype("Int64")
    df["age_source"] = np.where(by_birth.notna(), "birth_year",
                                np.where(df["age_from_barei"].notna(), "barei", ""))
    return df


def build(config: MLConfig, *, race_data_dir: Path, ml_data_dir: Path
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict]:
    runners = load_runners(race_data_dir, data=config.data, codes_config=config.codes)
    sources = load_sources(ml_data_dir, config.data.pedigree_versions)
    ped, ped_stats = build_pedigree(
        sources["SK"], sources["HN"], sources["UM"],
        generations=config.pedigree.generations,
        inbreeding_generations=config.pedigree.inbreeding_generations,
        horses=runners["ketto_num"].unique())
    names = ancestor_names(sources["HN"], sources["UM"])
    df = attach_pedigree(runners, ped)
    report = {
        "race_versions": runners.attrs.get("race_versions", []),
        "pedigree_versions": sources["versions"],
        "pedigree_stats": ped_stats,
        "checks": integrity.check_dataset(df, ped_stats, config),
    }
    return df, ped, names, report


def manifest_config(config: MLConfig) -> dict:
    """データセットの中身を決めた設定。manifest の "config" に残し、学習時に突き合わせる。"""
    return {"data": asdict(config.data), "codes": asdict(config.codes),
            "pedigree_generations": config.pedigree.generations,
            "inbreeding_generations": config.pedigree.inbreeding_generations}


#: 突き合わせから外す data の項目。読んだ版そのものは manifest の race_versions /
#: pedigree_versions に残っており、build_dataset.py の --race-version で設定ファイルと
#: 違う版を読むのは正当な使い方なので。
_UNCOMPARED_DATA_KEYS = ("race_versions", "pedigree_versions")


def config_mismatches(manifest: dict, config: MLConfig) -> list[str]:
    """manifest の設定と config の食い違い（項目名）。空なら同じ設定で作ったデータセット。"""
    def normalize(raw: dict) -> dict:
        raw = json.loads(json.dumps(raw, default=str))
        data = {k: v for k, v in raw.get("data", {}).items() if k not in _UNCOMPARED_DATA_KEYS}
        return {**raw, "data": data}

    built = normalize(manifest.get("config", {}))
    current = normalize(manifest_config(config))
    problems = []
    for key in current:
        if isinstance(current[key], dict):
            for sub in sorted(set(current[key]) | set(built.get(key, {}))):
                if current[key].get(sub) != built.get(key, {}).get(sub):
                    problems.append(f"{key}.{sub}")
        elif current[key] != built.get(key):
            problems.append(key)
    return problems


def dataset_dir(ml_data_dir: Path, dataset_id: str) -> Path:
    return Path(ml_data_dir) / "datasets" / dataset_id


def save(df: pd.DataFrame, ped: pd.DataFrame, names: dict[str, str], report: dict,
         config: MLConfig, *, ml_data_dir: Path, dataset_id: str | None = None) -> Path:
    dataset_id = dataset_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = dataset_dir(ml_data_dir, dataset_id)
    out.mkdir(parents=True, exist_ok=False)
    df.to_parquet(out / "runners.parquet", index=False)
    ped.to_parquet(out / "pedigree.parquet")
    (out / "ancestor_names.json").write_text(
        json.dumps(names, ensure_ascii=False), encoding="utf-8")
    (out / "integrity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with_result = df[df["has_result"]]
    manifest = {
        "dataset_id": dataset_id,
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "race_versions": report["race_versions"],
        "pedigree_versions": report["pedigree_versions"],
        "rows": int(len(df)), "races": int(df["race_id"].nunique()),
        "rows_with_result": int(len(with_result)),
        "first_race_date": str(df["race_date"].min().date()) if len(df) else None,
        "last_race_date": str(df["race_date"].max().date()) if len(df) else None,
        "config": manifest_config(config),
        "checks_failed": [c["name"] for c in report["checks"] if c["status"] == "fail"],
        "status": "complete",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (Path(ml_data_dir) / "datasets" / "CURRENT").write_text(dataset_id, encoding="utf-8")
    return out


def load(ml_data_dir: Path, dataset_id: str | None = None
         ) -> tuple[pd.DataFrame, dict[str, str], dict]:
    root = Path(ml_data_dir) / "datasets"
    if dataset_id is None:
        marker = root / "CURRENT"
        if not marker.exists():
            raise FileNotFoundError(
                "データセットがありません。scripts/ml/build_dataset.py を先に実行してください。")
        dataset_id = marker.read_text(encoding="utf-8").strip()
    path = root / dataset_id
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError(f"データセット {dataset_id} は完成していません。")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise RuntimeError(
            f"データセット {dataset_id} の schema_version は {manifest.get('schema_version')!r} で、"
            f"このコードが読めるのは {DATASET_SCHEMA_VERSION!r} です。"
            " build_dataset.py で作り直してください。")
    df = pd.read_parquet(path / "runners.parquet")
    names = json.loads((path / "ancestor_names.json").read_text(encoding="utf-8"))
    return df, names, manifest
