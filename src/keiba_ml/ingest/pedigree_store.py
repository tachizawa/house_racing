"""血統レコードの版付き CSV 保存。

既存の ``keiba.ingest.store`` と同じ手順（.partial に書く → manifest → rename →
CURRENT）を、血統専用の置き場所 ``data/ml/pedigree/`` で行う。既存の版
（``data/versions/``）には一切書かない。既存モデルが読む CSV の列構成を
変えないため。
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from keiba_ml.ingest.pedigree_records import (
    EXPECTED_LENGTHS, HN_COLUMNS, SK_COLUMNS, UM_KETTO_COLUMNS,
)

PEDIGREE_SCHEMA_VERSION = "1"

FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "SK": ("sk.csv", SK_COLUMNS),
    "HN": ("hn.csv", HN_COLUMNS),
    "UM": ("um_ketto.csv", UM_KETTO_COLUMNS),
}


class PedigreeVersionError(RuntimeError):
    """血統の版が読めない・壊れている。"""


def pedigree_root(ml_data_dir: Path) -> Path:
    return Path(ml_data_dir) / "pedigree"


def version_dir(ml_data_dir: Path, version_id: str) -> Path:
    return pedigree_root(ml_data_dir) / "versions" / version_id


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_manifest(path: Path) -> dict:
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.exists():
        raise PedigreeVersionError(f"manifest が見つかりません: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PedigreeVersionError(f"manifest が壊れています: {manifest_path}") from error
    if manifest.get("status") != "complete":
        raise PedigreeVersionError(f"血統の版 {path.name} は書き込みが完了していません。")
    return manifest


def complete_versions(ml_data_dir: Path) -> list[str]:
    """完成済みの版を古い順に返す。.partial と壊れた版は含めない。"""
    root = pedigree_root(ml_data_dir) / "versions"
    if not root.exists():
        return []
    result = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.endswith(".partial"):
            continue
        try:
            read_manifest(path)
        except PedigreeVersionError:
            continue
        result.append(path.name)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PedigreeVersionWriter:
    """1つの血統の版を書く。commit() か abort() のどちらかで必ず終える。"""

    def __init__(self, ml_data_dir: Path, version_id: str, *,
                 dataspecs: Sequence[str], fromdate: str, option: int) -> None:
        if isinstance(dataspecs, str):
            raise TypeError("dataspecs は ['BLOD'] のようなシーケンスで渡してください。")
        self.ml_data_dir = Path(ml_data_dir)
        self.version_id = version_id
        self.dataspecs = tuple(dataspecs)
        self.fromdate = fromdate
        self.option = option
        self.fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
        self._partial = version_dir(self.ml_data_dir, version_id + ".partial")
        try:
            self._partial.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise PedigreeVersionError(
                f"書きかけのディレクトリが既にあります: {self._partial}。"
                " 残骸なら削除してください。") from error
        self._counts: Counter[str] = Counter()
        self._length_mismatch: Counter[str] = Counter()
        self._handles: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        try:
            for spec, (name, columns) in FILES.items():
                handle = (self._partial / name).open("w", encoding="utf-8", newline="")
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(columns)
                self._handles[spec] = handle
                self._writers[spec] = writer
        except BaseException:
            self._close()
            shutil.rmtree(self._partial, ignore_errors=True)
            raise

    def add(self, spec: str, row: dict) -> None:
        if spec not in FILES:
            return
        _, columns = FILES[spec]
        self._writers[spec].writerow(["" if row.get(c) is None else row.get(c)
                                      for c in columns])
        self._counts[spec] += 1
        # 長さが仕様と違うレコードは位置がずれている疑いがある。件数を manifest に残す。
        if row.get("record_length") not in (None, EXPECTED_LENGTHS[spec]):
            self._length_mismatch[spec] += 1

    def abort(self) -> None:
        self._close()
        shutil.rmtree(self._partial, ignore_errors=True)

    def commit(self) -> Path:
        self._close()
        files = []
        for spec, (name, _) in FILES.items():
            path = self._partial / name
            files.append({"name": name, "rows": self._counts[spec],
                          "bytes": path.stat().st_size, "sha256": _sha256(path)})
        manifest = {
            "version_id": self.version_id,
            "schema_version": PEDIGREE_SCHEMA_VERSION,
            "fetched_at": self.fetched_at,
            "dataspecs": list(self.dataspecs),
            "fromdate": self.fromdate,
            "option": self.option,
            "record_counts": {spec: self._counts[spec] for spec in FILES},
            "length_mismatch": {spec: self._length_mismatch[spec] for spec in FILES},
            "expected_lengths": dict(EXPECTED_LENGTHS),
            "files": files,
            "status": "complete",
        }
        (self._partial / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        final = version_dir(self.ml_data_dir, self.version_id)
        if final.exists():
            raise PedigreeVersionError(f"血統の版 {self.version_id} は既にあります。")
        self._partial.rename(final)
        (pedigree_root(self.ml_data_dir) / "CURRENT").write_text(
            self.version_id, encoding="utf-8")
        return final

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def length_mismatch(self) -> dict[str, int]:
        return dict(self._length_mismatch)

    def _close(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()
