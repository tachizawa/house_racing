"""JV-Link のレコードを加工せずに保存し、あとで読み戻す。

セットアップ取得（option 3/4）は所有者のダイアログ操作が要るので、パーサが
未完成でも先に取っておけるようにする。読み戻したバイト列は JVRead から得た
ものと同じなので、records.parse_rows() にそのまま渡せる。

形式（2026-09-26 の最初の取得 tmp/fetch_raw/dump_raw.py と同じ）:
  <dir>/<レコード種別>.bin.gz … 「4バイトのビッグエンディアンの長さ + cp932 のバイト列」の繰り返しを gzip
  <dir>/manifest.json         … dataspec / fromdate / option / counts / completed
"""

from __future__ import annotations

import gzip
import json
import struct
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RawDumpError(RuntimeError):
    """生レコードの保存先が読めない・完了していない・壊れている。"""


_LENGTH = struct.Struct(">I")


class RawDumpWriter:
    """with 文で使う。抜けるときに manifest を書く（例外で抜けたら completed=False）。"""

    def __init__(self, directory: Path, *, dataspec: str, fromdate: str, option: int) -> None:
        self.directory = Path(directory)
        # 前の取得の残りと混ざらないよう、空でないディレクトリには書かない
        if self.directory.exists() and any(self.directory.iterdir()):
            raise RawDumpError(f"保存先が空ではありません: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._manifest: dict[str, Any] = {
            "dataspec": dataspec, "fromdate": fromdate, "option": option,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "format": "4-byte big-endian length + cp932 bytes, gzip",
        }
        self._files: dict[str, Any] = {}
        self._counts: dict[str, int] = {}

    def write(self, raw: bytes) -> None:
        spec = raw[:2].decode("ascii", errors="replace")
        handle = self._files.get(spec)
        if handle is None:
            handle = self._files[spec] = gzip.open(self.directory / f"{spec}.bin.gz", "wb",
                                                   compresslevel=6)
        handle.write(_LENGTH.pack(len(raw)))
        handle.write(raw)
        self._counts[spec] = self._counts.get(spec, 0) + 1

    def __enter__(self) -> RawDumpWriter:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        for handle in self._files.values():
            handle.close()
        self._manifest.update(
            completed=exc_type is None,
            counts=dict(sorted(self._counts.items())),
            records=sum(self._counts.values()),
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        if exc is not None:
            self._manifest["error"] = repr(exc)
        (self.directory / "manifest.json").write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def read_manifest(directory: Path) -> dict:
    """manifest を読む。無い・壊れている・完了していないものは拒否する。"""
    path = Path(directory) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RawDumpError(f"manifest が見つかりません: {path}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RawDumpError(f"manifest が壊れています: {path} / {error}") from error
    if not isinstance(manifest, dict) or manifest.get("completed") is not True:
        raise RawDumpError(
            f"生レコードの保存が完了していません: {path}（{manifest.get('error', '理由不明')}）。"
            " 途中までのレコードで版を作ると欠けたまま気づけないので、取り直してください。")
    return manifest


def iter_records(directory: Path, specs: Iterable[str] | None = None) -> Iterator[bytes]:
    """保存したレコードを種別ごとのファイル順に返す（種別の間の順序は保存時と違う）。

    specs を渡せばその種別だけ。長さの途中でファイルが終わっていたら止める。
    """
    directory = Path(directory)
    paths = (sorted(directory.glob("*.bin.gz")) if specs is None
             else [directory / f"{spec}.bin.gz" for spec in specs])
    for path in paths:
        if not path.exists():
            continue
        with gzip.open(path, "rb") as handle:
            while True:
                head = handle.read(_LENGTH.size)
                if not head:
                    break
                if len(head) != _LENGTH.size:
                    raise RawDumpError(f"{path.name} が途中で切れています（長さの欄が不完全）。")
                (length,) = _LENGTH.unpack(head)
                raw = handle.read(length)
                if len(raw) != length:
                    raise RawDumpError(
                        f"{path.name} が途中で切れています（{length} バイトのはずが {len(raw)}）。")
                yield raw
