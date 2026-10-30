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

from keiba_ml.config import BLOOD_DATA_DIR, MLConfig
from keiba_ml.dataset import integrity
from keiba_ml.dataset.pedigree import (
    ancestor_names, build_pedigree, load_blood_sources, load_sources,
)
from keiba_ml.dataset.races import load_runners

#: 2: 結果の列 result_margin_sec（1着馬とのタイム差）と result_weight_kg（馬体重）を足した。
#: 3: 父・母・母父自身の中央での成績（その馬が生まれる前のレースだけ。parent_* 列）と
#:    生産者コード（breeder_code。SK から）を足した。
#: 4: 総合モデル（combined）用に、枠・斤量・騎手・調教師・所属・ブリンカー・馬体重の増減と、
#:    過去走として使う結果（走破タイム・4角の位置・上がり3F）を足した（dataset.races._RUNNER_OPTIONAL）。
DATASET_SCHEMA_VERSION = "4"
#: 読めるスキーマ。列を足しただけなので 1〜3 も読める。足した列を使うモデル（距離適性・総合）は
#: registry の required_columns で列の有無を確かめ、無ければ作り直しを促す。
READABLE_SCHEMA_VERSIONS = frozenset({"1", "2", "3", "4"})

#: 自身の競走成績を付ける祖先の位置（父・母・母父）。
PARENT_RACING_PATHS: tuple[str, ...] = ("S", "D", "DS")
#: parent_<位置>_<項目> の項目。n 出走数・k3 3着以内数・e3 偶然の期待3着以内数・
#: pf 走りの質の平均・tier3 3着以内に入った最高の格・tiermax 出走した最高の格・
#: linked その祖先の血統登録番号が分かったか（1/0。0 なら外国産などで中央の成績を引けない）。
PARENT_RACING_ITEMS: tuple[str, ...] = ("n", "k3", "e3", "pf", "tier3", "tiermax", "linked")


def parent_racing_columns() -> list[str]:
    return [f"parent_{p}_{k}" for p in PARENT_RACING_PATHS for k in PARENT_RACING_ITEMS]


def attach_parent_racing(df: pd.DataFrame, hn: pd.DataFrame) -> pd.DataFrame:
    """父・母・母父**自身**の中央での成績を、出走馬ごとに付ける（parent_* 列）。

    祖先の繁殖登録番号を HN の血統登録番号に引き、その血統登録番号の出走（データセットの
    結果のある行）のうち、**その馬の生年月日より前**のレースだけを集計する。馬が走るのは
    生まれた後なので、どのレースの行から見ても過去の情報になる（同じ日・後の日は入らない）。
    学習期間（stats_history = "window"）に関係なく全期間から集計する: 母が走ったのは
    子が走るより何年も前で、学習期間に絞ると大半が消えるため。生年月日の無い馬は集計しない。
    """
    from keiba_ml.features.pedigree import beat_fraction, expected_top3

    hn = hn if "ketto_num" in hn.columns else hn.assign(ketto_num="")
    linked = hn[hn["ketto_num"].fillna("") != ""].drop_duplicates("hansyoku_num", keep="last")
    num2ketto = dict(zip(linked["hansyoku_num"], linked["ketto_num"]))
    res = df[df["has_result"]].copy()
    res["_pf"] = beat_fraction(res)
    res["_e3"] = expected_top3(res["field_size"])
    res["_tier"] = pd.to_numeric(res["grade_tier"], errors="coerce").astype(float)
    res["_tier3"] = res["_tier"].where(res["target_top3"] == 1)
    res = res[["ketto_num", "race_date", "target_top3", "_e3", "_pf", "_tier", "_tier3"]] \
        .rename(columns={"ketto_num": "_parent"})
    horses = df.drop_duplicates("ketto_num")[
        ["ketto_num", "birth_date", *[f"ped_{p}" for p in PARENT_RACING_PATHS]]].copy()
    horses["_birth"] = pd.to_datetime(horses["birth_date"], errors="coerce")
    table = pd.DataFrame(index=pd.Index(horses["ketto_num"], name="ketto_num"))
    for path in PARENT_RACING_PATHS:
        parent = horses[f"ped_{path}"].map(num2ketto)
        q = pd.DataFrame({"ketto_num": horses["ketto_num"].to_numpy(),
                          "_parent": parent.to_numpy(), "_birth": horses["_birth"].to_numpy()})
        q = q.dropna(subset=["_parent", "_birth"])
        m = q.merge(res, on="_parent", how="inner")
        m = m[m["race_date"] < m["_birth"]]
        g = m.groupby("ketto_num")
        agg = pd.DataFrame({"n": g.size().astype(float), "k3": g["target_top3"].sum(),
                            "e3": g["_e3"].sum(), "pf": g["_pf"].mean(),
                            "tier3": g["_tier3"].max(), "tiermax": g["_tier"].max()})
        agg = agg.reindex(table.index)
        agg[["n", "k3", "e3"]] = agg[["n", "k3", "e3"]].fillna(0.0)
        agg["linked"] = parent.notna().astype(float).to_numpy()
        for item in PARENT_RACING_ITEMS:
            table[f"parent_{path}_{item}"] = agg[item].astype(float)
    return df.merge(table, left_on="ketto_num", right_index=True, how="left")


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
    if "breeder_code" in df:
        df["breeder_code"] = df["breeder_code"].fillna("")
    df["inbreeding_f"] = df["inbreeding_f"].astype(float)
    df["n_cross"] = df["n_cross"].astype("Int64")
    by_birth = df["year"] - df["birth_year"].astype("Float64")
    by_birth = by_birth.where(by_birth >= 1)
    df["age"] = by_birth.fillna(df["age_from_barei"].astype("Float64")).astype("Int64")
    df["age_source"] = np.where(by_birth.notna(), "birth_year",
                                np.where(df["age_from_barei"].notna(), "barei", ""))
    return df


def build(config: MLConfig, *, race_data_dir: Path, ml_data_dir: Path,
          blood_data_dir: Path = BLOOD_DATA_DIR
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict]:
    runners = load_runners(race_data_dir, data=config.data, codes_config=config.codes)
    if config.data.pedigree_source == "blood":
        sources = load_blood_sources(blood_data_dir, config.data.pedigree_versions)
    else:
        sources = load_sources(ml_data_dir, config.data.pedigree_versions)
    ped, ped_stats = build_pedigree(
        sources["SK"], sources["HN"], sources["UM"],
        generations=config.pedigree.generations,
        inbreeding_generations=config.pedigree.inbreeding_generations,
        horses=runners["ketto_num"].unique())
    names = ancestor_names(sources["HN"], sources["UM"])
    df = attach_pedigree(runners, ped)
    df = attach_parent_racing(df, sources["HN"])
    report = {
        "race_versions": runners.attrs.get("race_versions", []),
        "races_dropped_before_fromdate": runners.attrs.get("dropped_before_fromdate", {}),
        "pedigree_source": config.data.pedigree_source,
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
#: 違う版を読むのは正当な使い方なので。pedigree_source は外さない（"ml" は UM だけの馬の
#: 祖先を持つなど中身が違うので、設定と違う出どころのデータセットで学習させない）。
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
        "pedigree_source": report.get("pedigree_source"),
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
    if manifest.get("schema_version") not in READABLE_SCHEMA_VERSIONS:
        raise RuntimeError(
            f"データセット {dataset_id} の schema_version は {manifest.get('schema_version')!r} で、"
            f"このコードが読めるのは {sorted(READABLE_SCHEMA_VERSIONS)} です。"
            " build_dataset.py で作り直してください。")
    df = pd.read_parquet(path / "runners.parquet")
    names = json.loads((path / "ancestor_names.json").read_text(encoding="utf-8"))
    return df, names, manifest
