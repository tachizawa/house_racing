"""JV-Link のレコードを加工せずに保存し、あとで読み戻す。

セットアップ取得（option 3/4）は所有者のダイアログ操作が要るので、パーサが
未完成でも先に取っておけるようにする。読み戻したバイト列は JVRead から得た
ものと同じなので、records.parse_rows() にそのまま渡せる。

形式（2026-09-26 の最初の取得 tmp/fetch_raw/dump_raw.py と同じ）:
  <dir>/<レコード種別>.bin.gz … 「4バイトのビッグエンディアンの長さ + cp932 のバイト列」の繰り返しを gzip
  <dir>/manifest.json         … dataspec / fromdate / option / counts / completed

ResumableRawDump（scripts/dump_raw.py）は同じ形式を、jvd ファイル1つごとに gzip の
member を足しながら書く。gzip は複数 member のファイルを1本として読めるので、
読み戻し（iter_records）は区別しない。途中の状態は <dir>/resume.json に残る。
"""

from __future__ import annotations

import gzip
import json
import os
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


class ResumableRawDump:
    """jvd ファイル1つを読み終えるたびに確定させる保存先。中断しても続きから読める。

    40年ぶんのセットアップ取得は JVRead が遅く（2026-10-17 実測で約35件/秒）30時間
    ほどかかるので、途中で止まっても最初からやり直さずに済むようにする。

    レコードはファイルの終わりまでメモリに溜め、終わったら種別ごとの gz に gzip の
    member を1つ足し、それから resume.json（済んだファイル名・各 gz の大きさ・件数）を
    置き換える。gz を書いた後・resume.json の前に落ちた分は、再開時に記録した大きさ
    まで切り詰めて捨てる。manifest.json は全部読み終えたときにだけ書く。
    """

    STATE_NAME = "resume.json"

    def __init__(self, directory: Path, *, dataspec: str, fromdate: str, option: int) -> None:
        self.directory = Path(directory)
        self._state_path = self.directory / self.STATE_NAME
        params = {"dataspec": dataspec, "fromdate": fromdate, "option": option}
        if self._state_path.exists():
            self._state = json.loads(self._state_path.read_text(encoding="utf-8"))
            recorded = {key: self._state.get(key) for key in params}
            if recorded != params:
                raise RawDumpError(
                    f"{self.directory} は別の条件 {recorded} で取り始めたものです"
                    f"（今回 {params}）。同じ条件で再開するか、別の保存先を指定してください。")
            if not self.finished:
                self._truncate_to_state()
        elif self.directory.exists() and any(self.directory.iterdir()):
            raise RawDumpError(f"保存先が空ではなく、再開の記録（{self.STATE_NAME}）もありません:"
                               f" {self.directory}")
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._state = {**params,
                           "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "done_files": [], "sizes": {}, "counts": {}, "bytes": 0}
            self._save_state()
        self._done = set(self._state["done_files"])
        self._current: str | None = None
        self._pending: list[bytes] = []

    @property
    def finished(self) -> bool:
        path = self.directory / "manifest.json"
        if not path.exists():
            return False
        return json.loads(path.read_text(encoding="utf-8")).get("completed") is True

    @property
    def done_file_count(self) -> int:
        return len(self._done)

    @property
    def record_count(self) -> int:
        return sum(self._state["counts"].values())

    def is_done(self, fname: str) -> bool:
        return fname in self._done

    def add(self, fname: str, raw: bytes) -> None:
        """読み込み中のファイル名つきでレコードを溜める。ファイル名が変われば前を確定する。"""
        if self._current is not None and fname != self._current:
            self.end_file()
        self._current = fname
        self._pending.append(raw)

    def end_file(self) -> None:
        """溜めたファイルの分を gz に足し、済んだファイルとして記録する。"""
        if self._current is None:
            return
        by_spec: dict[str, list[bytes]] = {}
        for raw in self._pending:
            by_spec.setdefault(raw[:2].decode("ascii", errors="replace"), []).append(raw)
        for spec, raws in by_spec.items():
            path = self.directory / f"{spec}.bin.gz"
            with open(path, "ab") as handle:
                with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=6) as member:
                    for raw in raws:
                        member.write(_LENGTH.pack(len(raw)))
                        member.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            self._state["sizes"][spec] = path.stat().st_size
            self._state["counts"][spec] = self._state["counts"].get(spec, 0) + len(raws)
            self._state["bytes"] += sum(len(raw) for raw in raws)
        self._state["done_files"].append(self._current)
        self._done.add(self._current)
        self._save_state()
        self._current = None
        self._pending = []

    def discard_pending(self) -> None:
        """読みかけのファイルの分を捨てる（次の再開でそのファイルを最初から読む）。"""
        self._current = None
        self._pending = []

    def complete(self) -> None:
        """最後のファイルを確定し、manifest.json を書く。これで read_manifest が通る。"""
        self.end_file()
        manifest = {
            "dataspec": self._state["dataspec"], "fromdate": self._state["fromdate"],
            "option": self._state["option"], "started_at": self._state["started_at"],
            "format": "4-byte big-endian length + cp932 bytes, gzip",
            "completed": True,
            "counts": dict(sorted(self._state["counts"].items())),
            "records": self.record_count,
            "bytes": self._state["bytes"],
            "files": len(self._state["done_files"]),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _truncate_to_state(self) -> None:
        sizes = self._state["sizes"]
        for path in self.directory.glob("*.bin.gz"):
            spec = path.name[: -len(".bin.gz")]
            recorded = sizes.get(spec)
            actual = path.stat().st_size
            if recorded is None:
                path.unlink()                   # 1ファイルも確定しないうちに落ちた種別
            elif actual > recorded:
                with open(path, "r+b") as handle:
                    handle.truncate(recorded)
            elif actual < recorded:
                raise RawDumpError(f"{path.name} が記録（{recorded} バイト）より小さい"
                                   f" {actual} バイトです。保存先が壊れています。")
        for spec in sizes:
            if not (self.directory / f"{spec}.bin.gz").exists():
                raise RawDumpError(f"{spec}.bin.gz が見つかりません。保存先が壊れています。")

    def _save_state(self) -> None:
        tmp = self._state_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._state_path)
